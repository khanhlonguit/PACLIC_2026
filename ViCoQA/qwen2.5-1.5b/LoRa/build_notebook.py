"""Build train_qwen_lora_unsloth.ipynb for ViCoQA."""
import json
import uuid
from pathlib import Path

OUT = Path(__file__).parent / "train_qwen_lora_unsloth.ipynb"


def cell(cell_type, src):
    if isinstance(src, list):
        lines = src
    else:
        lines = src.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
    c = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
        "id": uuid.uuid4().hex[:8],
    }
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def md(s):
    return cell("markdown", s)


def code(s):
    return cell("code", s)


TRAIN_CELL = r'''def apply_adapter(model, method_name):
    """Gắn đúng PEFT method lên base model đã load bằng Unsloth."""
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
        model = FastLanguageModel.get_peft_model(
            model,
            r=16, lora_alpha=32, target_modules=TARGET_MODULES,
            lora_dropout=0.05, bias="none",
            use_gradient_checkpointing="unsloth", random_state=3407,
            use_dora=False,
        )
        print("Applied: LoRA (r=16, alpha=32)")
        return model

    if method_name == "dora":
        model = FastLanguageModel.get_peft_model(
            model,
            r=16, lora_alpha=32, target_modules=TARGET_MODULES,
            lora_dropout=0.05, bias="none",
            use_gradient_checkpointing="unsloth", random_state=3407,
            use_dora=True,
        )
        print("Applied: DoRA (r=16, alpha=32, use_dora=True)")
        return model

    if method_name == "tinylora":
        import inspect
        import peft
        from peft import TinyLoraConfig, get_peft_model as peft_get_model

        sig = inspect.signature(TinyLoraConfig.__init__).parameters
        desired = {
            "r": 2, "u": 64, "num_projections": 64,
            "target_modules": TARGET_MODULES,
            "tinylora_dropout": 0.0, "lora_dropout": 0.0,
            "bias": "none", "task_type": "CAUSAL_LM",
            "weight_tying": 0.0, "projection_seed": 3407, "init_weights": True,
        }
        tiny_kwargs = {k: v for k, v in desired.items() if k in sig}
        if "target_modules" not in tiny_kwargs:
            tiny_kwargs["target_modules"] = TARGET_MODULES
        tinylora_config = TinyLoraConfig(**tiny_kwargs)
        causallm = _resolve_causallm(model)
        model = peft_get_model(causallm, tinylora_config)
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.print_trainable_parameters()
        print(f"Applied: TinyLoRA (PEFT {peft.__version__})")
        return model

    if method_name == "delora":
        import peft
        from peft import DeloraConfig, get_peft_model as peft_get_model

        delora_config = DeloraConfig(
            r=16, delora_lambda=15, target_modules=TARGET_MODULES,
            module_dropout=0.05, bias="none", task_type="CAUSAL_LM", init_weights=True,
        )
        causallm = _resolve_causallm(model)
        model = peft_get_model(causallm, delora_config)
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.print_trainable_parameters()
        print(f"Applied: DeLoRA (PEFT {peft.__version__})")
        return model

    raise ValueError(f"Unknown method: {method_name}")


def _force_single_gpu_train_env():
    os.environ["ACCELERATE_BYPASS_DEVICE_MAP"] = "true"
    os.environ["ACCELERATE_NUM_PROCESSES"] = "1"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    for k in ("MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK", "ROLE_NAME", "LOCAL_WORLD_SIZE"):
        os.environ.pop(k, None)
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
    model, tokenizer, trainer = None, None, None

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
            train_args.update({
                _eval_key: "steps",
                "eval_steps": EVAL_STEPS,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            })
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            ))
        else:
            train_args[_eval_key] = "no"
        train_args = {k: v for k, v in train_args.items() if k in _ta_params or k == "output_dir"}

        _force_single_gpu_train_env()
        sft_kwargs = dict(
            model=model,
            train_dataset=dataset,
            eval_dataset=eval_dataset if eval_on else None,
            args=TrainingArguments(**train_args),
            callbacks=callbacks,
        )
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

        cfg_cls, tr_cls = type(trainer.args), type(trainer)
        sys.modules["trl.trainer.sft_config"] = sys.modules[cfg_cls.__module__]
        sys.modules["trl.trainer.sft_trainer"] = sys.modules[tr_cls.__module__]
        sys.modules[cfg_cls.__module__].SFTConfig = cfg_cls
        sys.modules[tr_cls.__module__].SFTTrainer = tr_cls

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

EVAL_CELL = r'''PREFIX_RE = re.compile(
    r"^(đáp án|answer|câu trả lời|theo đoạn văn|trong đoạn văn)\s*[:\-]?\s*",
    re.IGNORECASE,
)

ADAPTER_SUBMISSION_DIRS = {
    "lora": "qwen2.5-1.5b-instruct-lora-vicoqa",
    "tinylora": "qwen2.5-1.5b-instruct-tinylora-vicoqa",
    "dora": "qwen2.5-1.5b-instruct-dora-vicoqa",
    "delora": "qwen2.5-1.5b-instruct-delora-vicoqa",
}


def normalize_text(text):
    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.lower().translate(str.maketrans("", "", string.punctuation)).split())


def compute_em(pred, truth):
    return int(normalize_text(pred) == normalize_text(truth))


def compute_f1(pred, truth):
    pt, tt = normalize_text(pred).split(), normalize_text(truth).split()
    if not pt and not tt:
        return 1.0
    if not pt or not tt:
        return 0.0
    common = Counter(pt) & Counter(tt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p, r = n / len(pt), n / len(tt)
    return 2 * p * r / (p + r)


def gold_refs_for_turn(dialog, turn_idx):
    refs = [dialog["answers"][turn_idx]["input_text"].strip()]
    extra = dialog.get("additional_answers") or {}
    alt_list = extra.get("0") or []
    if turn_idx < len(alt_list):
        alt = alt_list[turn_idx].get("input_text", "").strip()
        if alt and alt not in refs:
            refs.append(alt)
    return [r for r in refs if r]


def score_turn(pred, gold_list):
    if not gold_list:
        return 0, 0.0
    em = max(compute_em(pred, g) for g in gold_list)
    f1 = max(compute_f1(pred, g) for g in gold_list)
    return em, f1


def _validate_adapter_files(adapter_path):
    adapter_path = Path(adapter_path)
    st_path = adapter_path / "adapter_model.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(f"Thiếu {st_path}")
    size_mb = st_path.stat().st_size / (1024 * 1024)
    if size_mb < 0.1:
        raise RuntimeError(f"{st_path.name} quá nhỏ ({size_mb:.2f} MB).")
    print(f"Adapter file OK: {st_path.name} ({size_mb:.1f} MB)")


def _infer_dialogs_subprocess(method_name, adapter_path, dialogs, max_seq_length, *, log_every=20, max_dialogs=None):
    import shutil
    import subprocess
    import sys
    import tempfile

    if not Path(adapter_path).exists():
        print(f"⚠ SKIP {method_name}: adapter path không tồn tại.", flush=True)
        return None

    script = Path("eval_infer_subprocess.py")
    if not script.exists():
        raise FileNotFoundError(f"Thiếu {script.resolve()}")

    tmpdir = tempfile.mkdtemp(prefix="vicoqa_eval_")
    dialogs_path = str(Path(tmpdir) / "dialogs.json")
    preds_path = str(Path(tmpdir) / "preds.json")
    with open(dialogs_path, "w", encoding="utf-8") as f:
        json.dump(dialogs, f, ensure_ascii=False)

    cmd = [
        sys.executable, str(script),
        "--adapter-dir", str(adapter_path),
        "--dialogs-json", dialogs_path,
        "--output", preds_path,
        "--base-model", BASE_MODEL_NAME,
        "--max-seq-length", str(max_seq_length),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--log-every", str(log_every),
    ]
    limit = max_dialogs if max_dialogs is not None else EVAL_MAX_DIALOGS
    if limit:
        cmd.extend(["--max-dialogs", str(limit)])

    print(f"[Infer] {method_name}: subprocess multi-turn CoQA eval", flush=True)
    proc = subprocess.Popen(cmd, env=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            print(line.rstrip("\n"), flush=True)
    finally:
        proc.wait()

    if proc.returncode != 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f"{method_name}: subprocess exit {proc.returncode}")

    with open(preds_path, encoding="utf-8") as f:
        preds = json.load(f)
    shutil.rmtree(tmpdir, ignore_errors=True)
    return preds


def eval_one_adapter(method_name, adapter_path, dialogs, max_seq_length):
    print(f"\n--- CoQA eval: {method_name} | {adapter_path} ---", flush=True)
    _validate_adapter_files(adapter_path)

    eval_dialogs = dialogs
    if EVAL_MAX_DIALOGS:
        eval_dialogs = dialogs[:EVAL_MAX_DIALOGS]
        print(f"[Eval] Limited to {len(eval_dialogs)} dialogs", flush=True)

    preds = _infer_dialogs_subprocess(
        method_name, adapter_path, eval_dialogs, max_seq_length, log_every=EVAL_LOG_EVERY,
    )
    if preds is None:
        return None

    ems = [p["em"] for p in preds]
    f1s = [p["f1"] for p in preds]
    metrics = {
        "method": method_name,
        "adapter": adapter_path,
        "coqa_em": round(100 * sum(ems) / max(len(ems), 1), 4),
        "coqa_f1": round(100 * sum(f1s) / max(len(f1s), 1), 4),
        "n_turns": len(preds),
        "n_dialogs": len(eval_dialogs),
    }
    for p in preds[:3]:
        print(f"  [sample] GT='{p['ground_truth'][:50]}' | pred='{p['prediction'][:50]}' | F1={p['f1']:.2f}", flush=True)
    return {"metrics": metrics, "predictions": preds}


def export_submission_one_adapter(method_name, adapter_path, dialogs, max_seq_length):
    print(f"\n--- Submission export: {method_name} | {adapter_path} ---", flush=True)

    submit_dialogs = dialogs
    if SUBMISSION_MAX_DIALOGS:
        submit_dialogs = dialogs[:SUBMISSION_MAX_DIALOGS]

    preds = _infer_dialogs_subprocess(
        method_name, adapter_path, submit_dialogs, max_seq_length,
        log_every=SUBMISSION_LOG_EVERY, max_dialogs=SUBMISSION_MAX_DIALOGS,
    )
    if preds is None:
        return None

    results = [
        {
            "dialog_id": p["dialog_id"],
            "turn_id": p["turn_id"],
            "question": p["question"],
            "prediction": p["prediction"],
        }
        for p in preds
    ]
    out_dir = Path(ADAPTER_SUBMISSION_DIRS.get(method_name, adapter_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Submit] Saved {out_path} | turns={len(results)}")
    return out_path
'''

FINAL_CELL = r'''variant_map = {v["name"]: v for v in ADAPTER_VARIANTS}

if RUN_METRIC_EVAL:
    if "eval_dialogs" not in globals():
        raise NameError("Chạy cell tải dataset trước.")
    all_results = {}
    for method in EVAL_METHODS:
        path = variant_map[method]["save_path"]
        result = eval_one_adapter(method, path, eval_dialogs, max_seq_length)
        if result is not None:
            all_results[method] = result

    line = "=" * 72
    print("\n" + line)
    print(f"  SO SÁNH ADAPTERS — ViCoQA ({EVAL_SPLIT}) | Qwen2.5-1.5B-Instruct")
    print(line)
    print(f"{'Method':<12} {'CoQA EM':>12} {'CoQA F1':>12} {'Turns':>10}")
    print("-" * 72)
    for method, result in all_results.items():
        m = result["metrics"]
        print(f"{method:<12} {m['coqa_em']:>11.2f}% {m['coqa_f1']:>11.2f}% {m['n_turns']:>10}")
    print(line)

    save_payload = {
        "dataset": "ViCoQA",
        "eval_split": EVAL_SPLIT,
        "base_model": BASE_MODEL_NAME,
        "max_seq_length": max_seq_length,
        "train_common": TRAIN_COMMON,
        "summary": {k: v["metrics"] for k, v in all_results.items()},
        "predictions": {k: v["predictions"] for k, v in all_results.items()},
    }
    with open(COMPARE_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved comparison → {COMPARE_EVAL_PATH}")
else:
    print("RUN_METRIC_EVAL=False — bỏ qua metric eval.")

if RUN_SUBMISSION_EXPORT:
    if "test_dialogs" not in globals():
        raise NameError("Chạy cell tải dataset trước.")
    print(f"\n{'='*72}\n  EXPORT results.json — test split ({len(test_dialogs)} dialogs)\n{'='*72}")
    for method in EVAL_METHODS:
        path = variant_map[method]["save_path"]
        export_submission_one_adapter(method, path, test_dialogs, max_seq_length)
else:
    print("RUN_SUBMISSION_EXPORT=False — bỏ qua results.json.")
'''

cells = [
    md("""# ViCoQA — Qwen2.5-1.5B: train 4 adapters (Unsloth)

Train **tuần tự** với **cùng data / prompt / seed / epochs**:

1. **LoRA** → `qwen2.5-1.5b-instruct-lora-vicoqa`
2. **TinyLoRA** → `qwen2.5-1.5b-instruct-tinylora-vicoqa`
3. **DoRA** → `qwen2.5-1.5b-instruct-dora-vicoqa`
4. **DeLoRA** → `qwen2.5-1.5b-instruct-delora-vicoqa`

Sau đó:
- **CoQA metric eval** (EM/F1, max over `additional_answers`) trên dev/test
- **Submission** → `results.json` mỗi adapter

### Cách chạy
1. Pip cells → **Restart Kernel**
2. Warnings → Config → Data → Profiling → Dataset format
3. Chạy **Train loop** (cell cuối section Train) — phải thấy `>>> TRAIN_FIX_V4 <<<`
4. Chạy cell eval cuối (`RUN_METRIC_EVAL=True` / `RUN_SUBMISSION_EXPORT=True`)

**Nếu OOM với free ~0.7 GiB / total 23.5 GiB:** GPU zombie — restart container/reboot, không phải lỗi code.
"""),
    code("!pip uninstall torch torchvision torchaudio xformers transformers trl unsloth unsloth_zoo -y"),
    code("!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir"),
    code("""# TinyLoRA / DeLoRA cần PEFT >= 0.19
!pip install "peft>=0.19.0" transformers trl accelerate bitsandbytes xformers datasets --no-cache-dir
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-cache-dir
!pip install unsloth_zoo scikit-learn --no-cache-dir"""),
    code("""import importlib
import inspect

for pkg in ["torch", "transformers", "datasets", "unsloth", "peft"]:
    importlib.import_module(pkg)
    print(f"OK  {pkg}")

import peft
from peft import TinyLoraConfig

print(f"PEFT version: {peft.__version__}")
sig = inspect.signature(TinyLoraConfig.__init__).parameters
print("TinyLoraConfig params:", ", ".join(k for k in sig if k != "self"))
"""),
    code("""import warnings, logging, os
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
"""),
]

CONFIG_CELL = r'''import json
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

LOAD_IN_4BIT = True
LOAD_IN_8BIT = False
MAX_SEQ_CAP = 4096
MIN_SEQ_LENGTH = 512
MAX_NEW_TOKENS = 64
MIN_FREE_VRAM_GIB = 6.0

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
    """Chặn train nếu GPU gần đầy. clear_gpu KHÔNG fix zombie VRAM."""
    min_free = MIN_FREE_VRAM_GIB if min_free_gib is None else min_free_gib
    release_stale_training_objects()
    clear_gpu(verbose=True)
    free_gib = report_vram("preflight")

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
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

DATA_CELL = r'''def load_vicoqa_split(path):
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
                "dialog_id": d["id"],
                "turn_id": q["turn_id"],
                "story": d["story"],
                "question": q["input_text"],
                "answer": answer,
                "turn_idx": turn_idx,
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
    print(f"SMOKE_TEST: using {len(train_samples)} train turn-samples")

split_stats("Train", train_samples, train_dialogs)
split_stats("Dev", dev_samples, dev_dialogs)
split_stats("Test", test_samples, test_dialogs)

if EVAL_SPLIT == "dev":
    eval_dialogs = dev_dialogs
elif EVAL_SPLIT == "test":
    eval_dialogs = test_dialogs
else:
    raise ValueError(f"EVAL_SPLIT không hợp lệ: {EVAL_SPLIT}")

if SMOKE_TEST and EVAL_MAX_DIALOGS is None:
    EVAL_MAX_DIALOGS = SMOKE_EVAL_DIALOGS
'''

PROFILE_CELL = r'''if "train_samples" not in globals():
    raise NameError("Chạy cell tải dataset trước.")


def compute_max_seq_length(samples, tok, cap=MAX_SEQ_CAP, min_len=MIN_SEQ_LENGTH):
    lengths = []
    total = len(samples)
    pbar = tqdm(samples, total=total, desc="Token profiling", unit="sample", bar_format=TQDM_BAR)
    for i, s in enumerate(pbar, 1):
        lengths.append(len(tok.encode(sample_to_train_text(s, tok))))
        pbar.set_postfix(done=f"{i}/{total}")
    lengths.sort()
    n = len(lengths)
    stats = {
        "min": lengths[0], "p50": lengths[n // 2],
        "p95": lengths[int(n * 0.95)], "p99": lengths[int(n * 0.99)], "max": lengths[-1],
    }
    chosen = max(((min(math.ceil(stats["p99"] * 1.05), cap) + 255) // 256) * 256, min_len)
    truncated = sum(1 for L in lengths if L > chosen)
    stats.update({
        "chosen_max_seq_length": chosen,
        "truncated_samples": truncated,
        "truncated_pct": round(100 * truncated / n, 3),
    })
    return chosen, stats


tokenizer_prof = load_tokenizer()
if Path(PROFILING_CONFIG_PATH).exists() and not RUN_TRAINING:
    prof_cfg = json.load(open(PROFILING_CONFIG_PATH, encoding="utf-8"))
    max_seq_length = prof_cfg["max_seq_length"]
    length_stats = prof_cfg["token_length_stats"]
else:
    max_seq_length, length_stats = compute_max_seq_length(train_samples, tokenizer_prof)
    json.dump(
        {"max_seq_length": max_seq_length, "token_length_stats": length_stats},
        open(PROFILING_CONFIG_PATH, "w", encoding="utf-8"),
        indent=2,
    )
print(f"max_seq_length = {max_seq_length}")
for k, v in length_stats.items():
    print(f"  {k}: {v}")
del tokenizer_prof
clear_gpu()
'''

FORMAT_CELL = r'''tokenizer_fmt = load_tokenizer()

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

cells.extend([
    code(CONFIG_CELL),
    code(DATA_CELL),
    code(PROFILE_CELL),
    code(FORMAT_CELL),
    md("## Train 4 adapters tuần tự\n\nCell train gọi `preflight_vram()` — **không gọi from_pretrained** nếu GPU zombie (~22GB used, free < 6 GiB)."),
    code(TRAIN_CELL),
    code("""# === TRAIN LOOP ===
preflight_vram(min_free_gib=MIN_FREE_VRAM_GIB)

if RUN_TRAINING:
    variant_map = {v["name"]: v for v in ADAPTER_VARIANTS}
    trained_paths = {}
    for method in TRAIN_METHODS:
        if method not in variant_map:
            raise ValueError(f"Unknown TRAIN_METHODS item: {method}")
        print(f"\\n>>> Starting {method} ...", flush=True)
        path = train_one_variant(
            variant_map[method], max_seq_length, dataset,
            eval_dataset=eval_dataset if USE_EARLY_STOPPING else None,
        )
        trained_paths[method] = path
        preflight_vram(min_free_gib=MIN_FREE_VRAM_GIB)
    print("\\n✅ Train xong:")
    for k, v in trained_paths.items():
        print(f"  {k}: {v}")
else:
    print("RUN_TRAINING=False — bỏ qua train.")
"""),
    md("## CoQA Evaluation — LoRA vs TinyLoRA vs DoRA vs DeLoRA\n\nMulti-turn inference, CoQA EM/F1 (max over `additional_answers`)."),
    code(EVAL_CELL),
    code(FINAL_CELL),
])

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells)")
