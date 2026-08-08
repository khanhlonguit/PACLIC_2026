"""Rebuild clean ViCoQA notebook with serious VRAM + Accelerate fixes."""
import json
import uuid
from pathlib import Path

OUT = Path(__file__).parent / "train_qwen_lora_unsloth.ipynb"


def cell(cell_type, src):
    lines = src.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    c = {"cell_type": cell_type, "metadata": {}, "source": lines, "id": uuid.uuid4().hex[:8]}
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def md(s):
    return cell("markdown", s)


def code(s):
    return cell("code", s)


WARNINGS = r'''import warnings, logging, os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ACCELERATE_BYPASS_DEVICE_MAP"] = "true"
os.environ["ACCELERATE_NUM_PROCESSES"] = "1"
for _k in ("WORLD_SIZE", "LOCAL_RANK", "RANK", "MASTER_ADDR", "MASTER_PORT"):
    os.environ.pop(_k, None)
for _n in ("transformers", "datasets", "torch", "unsloth", "peft", "accelerate", "huggingface_hub"):
    logging.getLogger(_n).setLevel(logging.ERROR)
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass
print("Env OK | PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
'''

CONFIG = r'''import json
import math
import gc
import re
import string
import unicodedata
import subprocess
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT = (
    "Bạn là trợ lý hỏi-đáp tiếng Việt. Dựa trên đoạn văn dưới đây, "
    "trả lời ngắn gọn, tự nhiên theo ngữ cảnh hội thoại.\n\n"
    "Đoạn văn:\n{story}"
)

DATASET_ROOT = Path("../../")
TRAIN_JSON_PATH = DATASET_ROOT / "train.json"
DEV_JSON_PATH = DATASET_ROOT / "dev.json"
TEST_JSON_PATH = DATASET_ROOT / "test.json"
PROFILING_CONFIG_PATH = "profiling_config.json"
COMPARE_EVAL_PATH = "eval_compare_adapters_vicoqa.json"

RUN_TRAINING = True
RUN_METRIC_EVAL = True
RUN_SUBMISSION_EXPORT = True

# === VRAM-safe defaults (4090) ===
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False
MAX_SEQ_CAP = 4096
MIN_SEQ_LENGTH = 512
MAX_NEW_TOKENS = 64
MIN_FREE_VRAM_GIB = 6.0   # dưới ngưỡng này → KHÔNG train (GPU zombie/leak)

SMOKE_TEST = False
SMOKE_TRAIN_SAMPLES = 200
SMOKE_EVAL_DIALOGS = 10
EVAL_SPLIT = "dev"
EVAL_MAX_DIALOGS = None
SUBMISSION_MAX_DIALOGS = None
SUBMISSION_LOG_EVERY = 50
EVAL_LOG_EVERY = 20
SUBPROCESS_EVAL_ALL = True

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

USE_EARLY_STOPPING = True
MAX_TRAIN_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001
EVAL_STEPS = 200
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3

TRAIN_COMMON = dict(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    num_train_epochs=MAX_TRAIN_EPOCHS,
    learning_rate=2e-4,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
)

ADAPTER_VARIANTS = [
    {"name": "lora", "save_path": "qwen2.5-1.5b-instruct-lora-vicoqa", "output_dir": "outputs_vicoqa_lora"},
    {"name": "tinylora", "save_path": "qwen2.5-1.5b-instruct-tinylora-vicoqa", "output_dir": "outputs_vicoqa_tinylora"},
    {"name": "dora", "save_path": "qwen2.5-1.5b-instruct-dora-vicoqa", "output_dir": "outputs_vicoqa_dora"},
    {"name": "delora", "save_path": "qwen2.5-1.5b-instruct-delora-vicoqa", "output_dir": "outputs_vicoqa_delora"},
]
TRAIN_METHODS = ["lora", "tinylora", "dora", "delora"]
EVAL_METHODS = ["lora", "tinylora", "dora", "delora"]
TQDM_BAR = "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def build_messages(sample, for_inference=False):
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(story=sample["story"])}]
    for q, a in sample["history"]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": sample["question"]})
    if not for_inference:
        messages.append({"role": "assistant", "content": sample["answer"]})
    return messages


def sample_to_train_text(sample, tok):
    return tok.apply_chat_template(build_messages(sample), tokenize=False, add_generation_prompt=False)


def load_tokenizer(model_path=BASE_MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.chat_template is None:
        raise RuntimeError(f"Tokenizer {model_path} thiếu chat_template.")
    return tok


def clear_gpu(verbose=False):
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
    if verbose:
        report_vram("after clear_gpu")


def report_vram(label="VRAM"):
    if not torch.cuda.is_available():
        print(f"[{label}] CUDA not available")
        return 0.0
    free, total = torch.cuda.mem_get_info()
    free_gib = free / 1024**3
    total_gib = total / 1024**3
    used_driver = total_gib - free_gib
    pt_alloc = torch.cuda.memory_allocated() / 1024**3
    print(
        f"[{label}] driver_used={used_driver:.2f} GiB | free={free_gib:.2f}/{total_gib:.2f} GiB "
        f"| pytorch_alloc={pt_alloc:.3f} GiB",
        flush=True,
    )
    return free_gib


def release_stale_training_objects():
    """Xóa model/trainer còn sót từ lần train fail trước."""
    stale = (
        "model", "tokenizer", "trainer", "model_eval", "tokenizer_eval",
        "tokenizer_prof", "tokenizer_fmt", "trained_paths",
    )
    g = globals()
    for name in stale:
        if name in g:
            try:
                del g[name]
            except Exception:
                pass
    clear_gpu(verbose=False)


def preflight_vram(min_free_gib=None):
    """Chặn train nếu GPU gần đầy — clear_gpu KHÔNG fix zombie VRAM (nvidia-smi: no processes)."""
    min_free = MIN_FREE_VRAM_GIB if min_free_gib is None else min_free_gib
    release_stale_training_objects()
    clear_gpu(verbose=True)
    free_gib = report_vram("preflight")

    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
        used_mb, total_mb = [int(x.strip()) for x in out.split(",")]
        print(f"[preflight] nvidia-smi used={used_mb} MiB / total={total_mb} MiB", flush=True)
    except Exception as e:
        print(f"[preflight] nvidia-smi skip: {e}", flush=True)

    if free_gib < min_free:
        pt_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        raise RuntimeError(
            f"\n❌ GPU KHÔNG ĐỦ VRAM: free={free_gib:.2f} GiB (cần ≥ {min_free} GiB).\n"
            f"   PyTorch process chỉ alloc ~{pt_alloc:.2f} GiB → VRAM bị process khác/zombie giữ.\n"
            "   clear_gpu() KHÔNG giải phóng được trường hợp này.\n\n"
            "   CÁCH SỬA:\n"
            "   1) Kernel → Restart Kernel\n"
            "   2) Chạy !nvidia-smi — nếu Memory-Usage vẫn ~22GB mà No processes:\n"
            "      → restart container / reboot máy / sudo nvidia-smi --gpu-reset\n"
            "   3) Khi free ≥ 10 GiB mới train lại notebook từ đầu.\n"
        )
    print(f"✅ preflight OK | free={free_gib:.2f} GiB", flush=True)
    return free_gib
'''

DATA = r'''def load_vicoqa_split(path):
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
                "dialog_id": d["id"], "turn_id": q["turn_id"], "story": d["story"],
                "question": q["input_text"], "answer": answer, "turn_idx": turn_idx,
                "history": [
                    (d["questions"][t]["input_text"], answers_by_turn[d["questions"][t]["turn_id"]]["input_text"].strip())
                    for t in range(turn_idx)
                ],
            })
    return samples, dialogs


def split_stats(name, samples, dialogs):
    print(f"{name}: {len(dialogs)} dialogs | {len(samples)} turn-samples")

train_samples, train_dialogs = load_vicoqa_split(TRAIN_JSON_PATH)
dev_samples, dev_dialogs = load_vicoqa_split(DEV_JSON_PATH)
test_samples, test_dialogs = load_vicoqa_split(TEST_JSON_PATH)
if SMOKE_TEST:
    train_samples = train_samples[:SMOKE_TRAIN_SAMPLES]
split_stats("Train", train_samples, train_dialogs)
split_stats("Dev", dev_samples, dev_dialogs)
split_stats("Test", test_samples, test_dialogs)
eval_dialogs = dev_dialogs if EVAL_SPLIT == "dev" else test_dialogs
if SMOKE_TEST and EVAL_MAX_DIALOGS is None:
    EVAL_MAX_DIALOGS = SMOKE_EVAL_DIALOGS
'''

PROFILE = r'''if "train_samples" not in globals():
    raise NameError("Chạy cell tải dataset trước.")

def compute_max_seq_length(samples, tok, cap=MAX_SEQ_CAP, min_len=MIN_SEQ_LENGTH):
    lengths = []
    for s in tqdm(samples, desc="Token profiling", bar_format=TQDM_BAR):
        lengths.append(len(tok.encode(sample_to_train_text(s, tok))))
    lengths.sort()
    n = len(lengths)
    stats = {
        "min": lengths[0], "p50": lengths[n // 2],
        "p95": lengths[int(n * 0.95)], "p99": lengths[int(n * 0.99)], "max": lengths[-1],
    }
    chosen = max(((min(math.ceil(stats["p99"] * 1.05), cap) + 255) // 256) * 256, min_len)
    truncated = sum(1 for L in lengths if L > chosen)
    stats.update({"chosen_max_seq_length": chosen, "truncated_samples": truncated,
                  "truncated_pct": round(100 * truncated / n, 3)})
    return chosen, stats

tokenizer_prof = load_tokenizer()
if Path(PROFILING_CONFIG_PATH).exists() and not RUN_TRAINING:
    prof_cfg = json.load(open(PROFILING_CONFIG_PATH, encoding="utf-8"))
    max_seq_length = prof_cfg["max_seq_length"]
    length_stats = prof_cfg["token_length_stats"]
else:
    max_seq_length, length_stats = compute_max_seq_length(train_samples, tokenizer_prof)
    json.dump({"max_seq_length": max_seq_length, "token_length_stats": length_stats},
              open(PROFILING_CONFIG_PATH, "w", encoding="utf-8"), indent=2)
print(f"max_seq_length = {max_seq_length}")
for k, v in length_stats.items():
    print(f"  {k}: {v}")
del tokenizer_prof
clear_gpu()
'''

FORMAT = r'''tokenizer_fmt = load_tokenizer()

def formatting_prompts_func(examples):
    texts = []
    for story, question, answer, history in zip(
        examples["story"], examples["question"], examples["answer"], examples["history"]
    ):
        sample = {"story": story, "question": question, "answer": answer, "history": history}
        texts.append(tokenizer_fmt.apply_chat_template(build_messages(sample), tokenize=False, add_generation_prompt=False))
    return {"text": texts}

train_hf = Dataset.from_list(train_samples)
dataset = train_hf.map(formatting_prompts_func, batched=True, remove_columns=train_hf.column_names)
print(f"Shared train dataset: {len(dataset)} samples")
eval_dataset = None
if USE_EARLY_STOPPING:
    dev_hf = Dataset.from_list(dev_samples)
    eval_dataset = dev_hf.map(formatting_prompts_func, batched=True, remove_columns=dev_hf.column_names)
    print(f"Eval dataset (early stopping): {len(eval_dataset)} samples")
print(dataset[0]["text"][:500])
'''

TRAIN = r'''def _force_single_gpu_train_env():
    os.environ["ACCELERATE_BYPASS_DEVICE_MAP"] = "true"
    os.environ["ACCELERATE_NUM_PROCESSES"] = "1"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
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


def _strip_hf_device_map(obj, seen=None):
    if seen is None:
        seen = set()
    if obj is None or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, "hf_device_map"):
        try:
            delattr(obj, "hf_device_map")
        except Exception:
            obj.hf_device_map = None
    for attr in ("model", "base_model", "module", "pretrained_model"):
        child = getattr(obj, attr, None)
        if isinstance(child, torch.nn.Module):
            _strip_hf_device_map(child, seen)


def apply_adapter(model, method_name):
    from unsloth import FastLanguageModel

    def _resolve_causallm(m):
        if hasattr(m, "prepare_inputs_for_generation"):
            return m
        if hasattr(m, "get_base_model"):
            base = m.get_base_model()
            if hasattr(base, "prepare_inputs_for_generation"):
                return base
        raise RuntimeError("Không tìm thấy ForCausalLM trên model Unsloth.")

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
        import inspect, peft
        from peft import TinyLoraConfig, get_peft_model as peft_get_model
        sig = inspect.signature(TinyLoraConfig.__init__).parameters
        desired = {"r": 2, "u": 64, "num_projections": 64, "target_modules": TARGET_MODULES,
                   "tinylora_dropout": 0.0, "bias": "none", "task_type": "CAUSAL_LM", "init_weights": True}
        tiny_kwargs = {k: v for k, v in desired.items() if k in sig}
        model = peft_get_model(_resolve_causallm(model), TinyLoraConfig(**tiny_kwargs))
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        return model
    if method_name == "delora":
        import peft
        from peft import DeloraConfig, get_peft_model as peft_get_model
        cfg = DeloraConfig(r=16, delora_lambda=15, target_modules=TARGET_MODULES,
                           module_dropout=0.05, bias="none", task_type="CAUSAL_LM", init_weights=True)
        model = peft_get_model(_resolve_causallm(model), cfg)
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        return model
    raise ValueError(f"Unknown method: {method_name}")


def train_one_variant(variant, max_seq_length, dataset, eval_dataset=None):
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer
    from transformers import TrainingArguments, EarlyStoppingCallback
    import inspect, sys

    print(">>> TRAIN_FIX_V4 <<<", flush=True)
    _force_single_gpu_train_env()
    preflight_vram(min_free_gib=MIN_FREE_VRAM_GIB)

    name = variant["name"]
    print("\n" + "=" * 60)
    print(f" TRAIN VARIANT: {name.upper()} | 4bit={LOAD_IN_4BIT}")
    print("=" * 60)

    eval_on = USE_EARLY_STOPPING and eval_dataset is not None
    model, tokenizer = None, None
    trainer = None

    try:
        load_kwargs = dict(
            model_name=BASE_MODEL_NAME,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
        )
        if "device_map" in inspect.signature(FastLanguageModel.from_pretrained).parameters:
            load_kwargs["device_map"] = {"": 0}

        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
        model = apply_adapter(model, name)
        _strip_hf_device_map(model)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

        train_args = dict(
            **TRAIN_COMMON,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            output_dir=variant["output_dir"],
            disable_tqdm=False,
            logging_strategy="steps",
            logging_steps=SAVE_STEPS,
            logging_first_step=False,
            log_level="error",
            log_level_replica="error",
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            report_to="none",
            dataloader_num_workers=0,
        )
        _ta_params = inspect.signature(TrainingArguments.__init__).parameters
        _eval_key = "eval_strategy" if "eval_strategy" in _ta_params else "evaluation_strategy"
        callbacks = []
        if eval_on:
            train_args.update({_eval_key: "steps", "eval_steps": EVAL_STEPS,
                               "load_best_model_at_end": True, "metric_for_best_model": "eval_loss",
                               "greater_is_better": False})
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            ))
        else:
            train_args[_eval_key] = "no"
        train_args = {k: v for k, v in train_args.items() if k in _ta_params or k == "output_dir"}

        _force_single_gpu_train_env()
        sft_kwargs = dict(model=model, train_dataset=dataset,
                          eval_dataset=eval_dataset if eval_on else None,
                          args=TrainingArguments(**train_args), callbacks=callbacks)
        _sft_params = inspect.signature(SFTTrainer.__init__).parameters
        if "processing_class" in _sft_params:
            sft_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in _sft_params:
            sft_kwargs["tokenizer"] = tokenizer
        for k, v in [("dataset_text_field", "text"), ("max_seq_length", max_seq_length),
                     ("packing", False), ("dataset_num_proc", 1)]:
            if k in _sft_params:
                sft_kwargs[k] = v

        trainer = SFTTrainer(**sft_kwargs)
        if hasattr(trainer, "accelerator"):
            trainer.accelerator.verify_device_map = lambda model: False

        trainer.train()
        Path(variant["save_path"]).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(variant["save_path"])
        tokenizer.save_pretrained(variant["save_path"])
        print(f"Saved adapter → {variant['save_path']}")
        return variant["save_path"]
    finally:
        if trainer is not None:
            del trainer
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        clear_gpu(verbose=True)
'''

LOOP = r'''# === TRAIN LOOP (chạy cell này) ===
preflight_vram(min_free_gib=MIN_FREE_VRAM_GIB)

if RUN_TRAINING:
    variant_map = {v["name"]: v for v in ADAPTER_VARIANTS}
    trained_paths = {}
    for method in TRAIN_METHODS:
        if method not in variant_map:
            raise ValueError(f"Unknown TRAIN_METHODS item: {method}")
        print(f"\n>>> Starting {method} ...", flush=True)
        path = train_one_variant(
            variant_map[method], max_seq_length, dataset,
            eval_dataset=eval_dataset if USE_EARLY_STOPPING else None,
        )
        trained_paths[method] = path
        preflight_vram(min_free_gib=MIN_FREE_VRAM_GIB)
    print("\n✅ Train xong:")
    for k, v in trained_paths.items():
        print(f"  {k}: {v}")
else:
    print("RUN_TRAINING=False")
'''

# Eval cell - read from patch_train_v3 EVAL_CELL or simplified - I'll import from existing file
EVAL = Path(__file__).parent / "build_notebook.py"
eval_src = ""
if EVAL.exists():
    txt = EVAL.read_text(encoding="utf-8")
    i = txt.find('EVAL_CELL = r')
    if i >= 0:
        j = txt.find("'''", i + 12)
        k = txt.find("'''", j + 3)
        eval_src = txt[j + 3:k]

FINAL = Path(__file__).parent / "build_notebook.py"
final_src = ""
if FINAL.exists():
    txt = FINAL.read_text(encoding="utf-8")
    i = txt.find('FINAL_CELL = r')
    if i >= 0:
        j = txt.find("'''", i + 14)
        k = txt.find("'''", j + 3)
        final_src = txt[j + 3:k]

cells = [
    md("# ViCoQA — Qwen2.5-1.5B Unsloth Fine-tuning\n\n**Quan trọng:** Nếu `nvidia-smi` báo ~22GB used + `No running processes` → restart container/reboot trước khi train."),
    code("!pip uninstall torch torchvision torchaudio xformers transformers trl unsloth unsloth_zoo -y"),
    code("!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir"),
    code('''!pip install "peft>=0.19.0" transformers trl accelerate bitsandbytes xformers datasets --no-cache-dir
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-cache-dir
!pip install unsloth_zoo scikit-learn --no-cache-dir'''),
    code("""import importlib
for pkg in ["torch", "transformers", "datasets", "unsloth", "peft"]:
    importlib.import_module(pkg)
    print("OK", pkg)
import peft
print("PEFT", peft.__version__)
"""),
    code(WARNINGS),
    code(CONFIG),
    code(DATA),
    code(PROFILE),
    code(FORMAT),
    md("## Train\n\nCell train loop gọi `preflight_vram()` — **không train** nếu GPU zombie."),
    code(TRAIN),
    code(LOOP),
]
if eval_src:
    cells.append(md("## CoQA Eval"))
    cells.append(code(eval_src))
if final_src:
    cells.append(code(final_src))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12.0"}},
    "cells": cells,
}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote clean notebook: {OUT} ({len(cells)} cells)")
