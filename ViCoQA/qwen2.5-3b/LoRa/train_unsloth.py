#!/usr/bin/env python3
"""
Train một adapter ViCoQA (Qwen2.5-3B) trong process sạch (thoát Jupyter zombie VRAM).
Usage:
  python train_unsloth.py --method lora
  python train_unsloth.py --all
  python train_unsloth.py --preflight-only
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# Env TRƯỚC import torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ACCELERATE_BYPASS_DEVICE_MAP", "true")
os.environ.setdefault("ACCELERATE_NUM_PROCESSES", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE.parent.parent  # ViCoQA/

NOTEBOOK_VERSION = "V5.1-3B"
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = (
    "Bạn là trợ lý hỏi-đáp tiếng Việt. Dựa trên đoạn văn dưới đây, "
    "trả lời ngắn gọn, tự nhiên theo ngữ cảnh hội thoại.\n\n"
    "Đoạn văn:\n{story}"
)

LOAD_IN_4BIT = True
MIN_FREE_VRAM_GIB = 10.0
MIN_LOAD_VRAM_GIB = 6.0
PROFILING_CONFIG_PATH = HERE / "profiling_config.json"


def load_in_4bit_for_method(method_name: str) -> bool:
    if method_name == "delora":
        return False
    return LOAD_IN_4BIT

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

ADAPTER_VARIANTS = [
    {"name": "lora", "save_path": "qwen2.5-3b-instruct-lora-vicoqa", "output_dir": "outputs_vicoqa_3b_lora"},
    {"name": "tinylora", "save_path": "qwen2.5-3b-instruct-tinylora-vicoqa", "output_dir": "outputs_vicoqa_3b_tinylora"},
    {"name": "dora", "save_path": "qwen2.5-3b-instruct-dora-vicoqa", "output_dir": "outputs_vicoqa_3b_dora"},
    {"name": "delora", "save_path": "qwen2.5-3b-instruct-delora-vicoqa", "output_dir": "outputs_vicoqa_3b_delora"},
]

# 3B + multi-turn seq ~1280: batch=1/accum=8 (effective=8). OOM → giữ batch=1, tăng accum.
TRAIN_COMMON = dict(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=10,
    num_train_epochs=5,
    learning_rate=1e-4,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
)

USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001
EVAL_STEPS = 200
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3


def resolve_resume_checkpoint(output_dir, resume_flag):
    """resume_flag: False | True | path str. True = lấy checkpoint-* mới nhất."""
    if not resume_flag:
        return None
    if isinstance(resume_flag, str) and resume_flag not in ("True", "true", "1"):
        p = Path(resume_flag)
        if not p.is_absolute():
            p = HERE / p
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint không tồn tại: {p}")
        return str(p)
    out = Path(output_dir) if Path(output_dir).is_absolute() else HERE / output_dir
    if not out.exists():
        return None
    ckpts = sorted(
        [p for p in out.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(ckpts[-1]) if ckpts else None


def _patch_transformers_warmup():
    """Transformers warmup pre-alloc ~1.4 GiB — skip khi VRAM thấp."""
    try:
        import transformers.modeling_utils as mu

        def _safe_warmup(model, expanded_device_map, hf_quantizer=None, **kwargs):
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                print(f"[V5-3B] caching_allocator_warmup SKIPPED (free={free / 1024**3:.2f} GiB)", flush=True)
            return None

        mu.caching_allocator_warmup = _safe_warmup
    except Exception as e:
        print(f"[V5-3B] warmup patch failed: {e}", flush=True)


def _force_single_gpu():
    os.environ["ACCELERATE_BYPASS_DEVICE_MAP"] = "true"
    os.environ["ACCELERATE_NUM_PROCESSES"] = "1"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    try:
        from accelerate.state import AcceleratorState
        AcceleratorState._reset_state(reset_partial_state=True)
    except Exception:
        pass
    try:
        import accelerate.accelerator as acc_mod
        acc_mod.Accelerator.verify_device_map = lambda self, model: False
    except Exception:
        pass


def clear_gpu():
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


def report_vram(label="VRAM"):
    if not torch.cuda.is_available():
        print(f"[{label}] CUDA unavailable")
        return 0.0
    free, total = torch.cuda.mem_get_info()
    free_gib = free / 1024**3
    total_gib = total / 1024**3
    pt = torch.cuda.memory_allocated() / 1024**3
    print(
        f"[{label}] used={total_gib - free_gib:.2f} GiB | free={free_gib:.2f}/{total_gib:.2f} GiB | pytorch={pt:.3f} GiB",
        flush=True,
    )
    return free_gib


def preflight(min_free_gib=MIN_FREE_VRAM_GIB):
    clear_gpu()
    free_gib = report_vram("preflight")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
        used_mb, total_mb = [int(x.strip()) for x in out.split(",")]
        print(f"[preflight] nvidia-smi used={used_mb} MiB / {total_mb} MiB", flush=True)
        if used_mb > 3000 and free_gib < min_free_gib:
            print(
                "[preflight] ⚠ nvidia-smi used cao nhưng free thấp → GPU zombie/leak.\n"
                "  Thoát Jupyter, chạy: python gpu_preflight.py\n"
                "  Nếu fail: restart container / sudo nvidia-smi --gpu-reset / reboot",
                flush=True,
            )
    except Exception as e:
        print(f"[preflight] nvidia-smi: {e}", flush=True)

    if free_gib < min_free_gib:
        raise SystemExit(
            f"\n❌ STOP: free={free_gib:.2f} GiB < {min_free_gib} GiB.\n"
            "Không thể load Qwen2.5-3B. Reset GPU trước (không phải lỗi code).\n"
        )
    print(f"✅ preflight OK (free={free_gib:.2f} GiB)", flush=True)


def assert_before_load():
    free_gib = report_vram("before_load")
    if free_gib < MIN_LOAD_VRAM_GIB:
        raise RuntimeError(
            f"BLOCKED from_pretrained: free={free_gib:.2f} GiB < {MIN_LOAD_VRAM_GIB} GiB. Reset GPU."
        )


def build_messages(sample, for_inference=False):
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(story=sample["story"])}]
    for q, a in sample["history"]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": sample["question"]})
    if not for_inference:
        messages.append({"role": "assistant", "content": sample["answer"]})
    return messages


def load_vicoqa_split(path):
    with open(path, encoding="utf-8") as f:
        dialogs = json.load(f)
    samples = []
    for d in dialogs:
        answers_by_turn = {a["turn_id"]: a for a in d["answers"]}
        for turn_idx, q in enumerate(d["questions"]):
            a = answers_by_turn[q["turn_id"]]
            answer = (a.get("input_text") or "").strip()
            if not answer:
                continue
            samples.append({
                "story": d["story"], "question": q["input_text"], "answer": answer,
                "history": [
                    (d["questions"][t]["input_text"], answers_by_turn[d["questions"][t]["turn_id"]]["input_text"].strip())
                    for t in range(turn_idx)
                ],
            })
    return samples


def prepare_datasets(max_seq_length):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    train_samples = load_vicoqa_split(DATASET_ROOT / "train.json")
    dev_samples = load_vicoqa_split(DATASET_ROOT / "dev.json")

    def fmt(samples):
        texts = [tok.apply_chat_template(build_messages(s), tokenize=False, add_generation_prompt=False) for s in samples]
        return Dataset.from_dict({"text": texts})

    train_ds = fmt(train_samples)
    eval_ds = fmt(dev_samples) if USE_EARLY_STOPPING else None
    del tok
    clear_gpu()
    return train_ds, eval_ds


def apply_adapter(model, method_name):
    from unsloth import FastLanguageModel

    def _resolve(m):
        if hasattr(m, "prepare_inputs_for_generation"):
            return m
        if hasattr(m, "get_base_model"):
            b = m.get_base_model()
            if hasattr(b, "prepare_inputs_for_generation"):
                return b
        raise RuntimeError("No ForCausalLM")

    if method_name == "lora":
        return FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=32, target_modules=TARGET_MODULES,
            lora_dropout=0.05, bias="none", use_gradient_checkpointing="unsloth",
            random_state=3407, use_dora=False,
        )
    if method_name == "dora":
        return FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=32, target_modules=TARGET_MODULES,
            lora_dropout=0.05, bias="none", use_gradient_checkpointing="unsloth",
            random_state=3407, use_dora=True,
        )
    if method_name == "tinylora":
        import inspect
        from peft import TinyLoraConfig, get_peft_model as peft_get_model
        sig = inspect.signature(TinyLoraConfig.__init__).parameters
        desired = {"r": 2, "u": 64, "num_projections": 64, "target_modules": TARGET_MODULES,
                   "tinylora_dropout": 0.0, "bias": "none", "task_type": "CAUSAL_LM", "init_weights": True}
        cfg = TinyLoraConfig(**{k: v for k, v in desired.items() if k in sig})
        model = peft_get_model(_resolve(model), cfg)
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        return model
    if method_name == "delora":
        from peft import DeloraConfig, get_peft_model as peft_get_model
        cfg = DeloraConfig(r=16, delora_lambda=15, target_modules=TARGET_MODULES,
                           module_dropout=0.05, bias="none", task_type="CAUSAL_LM", init_weights=True)
        model = peft_get_model(_resolve(model), cfg)
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        return model
    raise ValueError(method_name)


def train_one(method_name: str, max_seq_length: int, train_ds, eval_ds, resume_flag=False):
    import inspect
    import sys
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer
    from transformers import TrainingArguments, EarlyStoppingCallback

    variant = next(v for v in ADAPTER_VARIANTS if v["name"] == method_name)
    use_4bit = load_in_4bit_for_method(method_name)
    print(f"\n>>> TRAIN_FIX_V5.2-3B | {method_name} | 4bit={use_4bit} <<<", flush=True)
    _force_single_gpu()
    preflight()
    assert_before_load()

    model, tokenizer, trainer = None, None, None
    try:
        load_kwargs = dict(
            model_name=BASE_MODEL_NAME,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=use_4bit,
            load_in_8bit=False,
        )
        # Không truyền device_map → tránh transformers warmup 1.4 GiB
        sig = inspect.signature(FastLanguageModel.from_pretrained).parameters
        if "device_map" in sig:
            load_kwargs["device_map"] = None

        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
        if torch.cuda.is_available():
            model = model.to("cuda:0")

        model = apply_adapter(model, method_name)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

        eval_on = USE_EARLY_STOPPING and eval_ds is not None
        train_args = dict(
            **TRAIN_COMMON,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            output_dir=str(HERE / variant["output_dir"]),
            logging_strategy="steps",
            logging_steps=SAVE_STEPS,
            log_level="error",
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            report_to="none",
            dataloader_num_workers=0,
        )
        _ta = inspect.signature(TrainingArguments.__init__).parameters
        ek = "eval_strategy" if "eval_strategy" in _ta else "evaluation_strategy"
        callbacks = []
        if eval_on:
            train_args.update({ek: "steps", "eval_steps": EVAL_STEPS,
                               "load_best_model_at_end": True, "metric_for_best_model": "eval_loss",
                               "greater_is_better": False})
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            ))
        else:
            train_args[ek] = "no"
        train_args = {k: v for k, v in train_args.items() if k in _ta or k == "output_dir"}
        if "local_rank" in _ta:
            train_args["local_rank"] = -1

        sft_kw = dict(model=model, train_dataset=train_ds, eval_dataset=eval_ds if eval_on else None,
                      args=TrainingArguments(**train_args), callbacks=callbacks)
        sp = inspect.signature(SFTTrainer.__init__).parameters
        if "processing_class" in sp:
            sft_kw["processing_class"] = tokenizer
        elif "tokenizer" in sp:
            sft_kw["tokenizer"] = tokenizer
        for k, v in [("dataset_text_field", "text"), ("max_seq_length", max_seq_length),
                     ("packing", False), ("dataset_num_proc", 1)]:
            if k in sp:
                sft_kw[k] = v

        trainer = SFTTrainer(**sft_kw)
        if hasattr(trainer, "accelerator"):
            trainer.accelerator.verify_device_map = lambda m: False

        resume_ckpt = resolve_resume_checkpoint(variant["output_dir"], resume_flag)
        if resume_ckpt:
            print(f"Resume from checkpoint: {resume_ckpt}", flush=True)
            trainer.train(resume_from_checkpoint=resume_ckpt)
        elif resume_flag:
            print("RESUME=True nhưng chưa có checkpoint-* — train từ đầu.", flush=True)
            trainer.train()
        else:
            trainer.train()
        save_path = HERE / variant["save_path"]
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        print(f"Saved → {save_path}", flush=True)
        return str(save_path)
    finally:
        for obj in (trainer, model, tokenizer):
            if obj is not None:
                del obj
        clear_gpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=[v["name"] for v in ADAPTER_VARIANTS])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Resume từ checkpoint mới nhất (hoặc truyền path: --resume outputs_vicoqa_3b_lora/checkpoint-600)",
    )
    args = parser.parse_args()

    print(f"=== train_unsloth.py {NOTEBOOK_VERSION} ===", flush=True)
    _patch_transformers_warmup()

    if args.preflight_only:
        preflight()
        return

    if not args.all and not args.method:
        parser.error("Cần --method hoặc --all hoặc --preflight-only")

    if PROFILING_CONFIG_PATH.exists():
        max_seq_length = json.loads(PROFILING_CONFIG_PATH.read_text(encoding="utf-8"))["max_seq_length"]
    else:
        max_seq_length = 1280
        print(f"[warn] dùng max_seq_length mặc định {max_seq_length}", flush=True)

    methods = [v["name"] for v in ADAPTER_VARIANTS] if args.all else [args.method]
    train_ds, eval_ds = prepare_datasets(max_seq_length)

    for m in methods:
        train_one(m, max_seq_length, train_ds, eval_ds, resume_flag=args.resume)
        preflight()

    print("\n✅ Done", flush=True)


if __name__ == "__main__":
    main()
