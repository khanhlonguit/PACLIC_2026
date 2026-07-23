"""W&B integration for ViNewsQA 4-variant LoRA fine-tuning pipeline.

Usage (inside train_one_variant):
    from wandb_integration import (
        ensure_wandb_ready, init_run, finish_run, build_wandb_config,
        WandbAdapterArtifactCallback, WandbGenerationTableCallback, WandbMetricsCallback,
    )
    ensure_wandb_ready(mode=WANDB_MODE)
    run = init_run(variant, config_dict)
    try:
        if WANDB_WATCH_ENABLED:
            wandb.watch(model, log="gradients", log_freq=WANDB_WATCH_LOG_FREQ)
        trainer.train()
        ...
    finally:
        finish_run(run)
"""
from __future__ import annotations

import math
import os
import string
import threading
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

_WANDB_AVAILABLE = False
try:
    import wandb  # noqa: F401

    _WANDB_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import TrainerCallback
except ImportError:
    TrainerCallback = object  # type: ignore[misc, assignment]


# ---------------------------------------------------------------------------
# Module 1 — Setup & Authentication
# ---------------------------------------------------------------------------

def ensure_wandb_ready(mode: str = "online") -> bool:
    """Check W&B auth and set WANDB_MODE.

    Returns True if W&B is usable, False if it falls back to disabled/offline.
    Never raises — training must not fail because of W&B.
    """
    if not _WANDB_AVAILABLE:
        print("[wandb] Package not installed. pip install wandb>=0.18", flush=True)
        os.environ["WANDB_MODE"] = "disabled"
        return False

    api_key = (
        os.environ.get("WANDB_API_KEY", "").strip()
        or getattr(getattr(wandb, "api", None), "api_key", None)
        or ""
    )
    if not api_key and mode == "online":
        print(
            "[wandb] No WANDB_API_KEY found. Falling back to offline mode.\n"
            "        Set os.environ['WANDB_API_KEY'] = '...' before training to enable online sync.",
            flush=True,
        )
        os.environ["WANDB_MODE"] = "offline"
        return True

    os.environ["WANDB_MODE"] = mode
    print(f"[wandb] Ready (mode={os.environ['WANDB_MODE']})", flush=True)
    return True


def build_wandb_config(
    variant: dict,
    base_model: str,
    max_seq_length: int,
    train_common: dict,
    dataset_sizes: dict,
    load_in_4bit: bool,
    bf16: bool,
    target_modules: list,
) -> dict:
    """Build the full config dict for wandb.init(config=...)."""
    name = variant["name"]

    peft_extras: dict[str, Any] = {}
    if name in ("lora", "dora"):
        peft_extras = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "use_dora": name == "dora"}
    elif name == "tinylora":
        peft_extras = {"r": 2, "u": 64, "tinylora_dropout": 0.0}
    elif name == "delora":
        peft_extras = {"r": 16, "delora_lambda": 15, "module_dropout": 0.05}

    return {
        # model spec
        "base_model": base_model,
        "max_seq_length": max_seq_length,
        "load_in_4bit": load_in_4bit,
        "precision": "bf16" if bf16 else "fp16",
        # peft spec
        "peft_type": name.upper(),
        "target_modules": target_modules,
        **peft_extras,
        # train spec
        "learning_rate": train_common.get("learning_rate"),
        "lr_scheduler_type": train_common.get("lr_scheduler_type"),
        "per_device_train_batch_size": train_common.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": train_common.get("gradient_accumulation_steps"),
        "effective_batch_size": (
            train_common.get("per_device_train_batch_size", 1)
            * train_common.get("gradient_accumulation_steps", 1)
        ),
        "optimizer": train_common.get("optim"),
        "num_train_epochs": train_common.get("num_train_epochs"),
        "weight_decay": train_common.get("weight_decay"),
        "warmup_ratio": train_common.get("warmup_ratio"),
        # dataset
        "dataset": "ViNewsQA",
        "train_size": dataset_sizes.get("train", -1),
        "dev_size": dataset_sizes.get("dev", -1),
    }


def init_run(
    variant: dict,
    config: dict,
    project: str,
    entity: str | None,
    group: str | None,
    tags: list[str] | None = None,
    resume_run_id: str | None = None,
) -> Any:
    """Open a new W&B run for one adapter variant. Returns the run object."""
    if not _WANDB_AVAILABLE:
        return None

    name = variant["name"]
    lr = config.get("learning_rate", 2e-4)
    r = config.get("r", "?")
    run_name = f"vinewsqa-{name}-r{r}-lr{lr:.0e}"

    _tags = [name, "vinewsqa", "unsloth"] + (tags or [])

    init_kwargs: dict[str, Any] = dict(
        project=project,
        entity=entity,
        group=group,
        name=run_name,
        config=config,
        tags=_tags,
        reinit="finish_previous",
    )
    if resume_run_id:
        init_kwargs["id"] = resume_run_id
        init_kwargs["resume"] = "allow"

    run = wandb.init(**init_kwargs)
    if resume_run_id:
        print(f"[wandb] Resumed run: {run.name} | {run.url}", flush=True)
    else:
        print(f"[wandb] Run started: {run.name} | {run.url}", flush=True)
    return run


def finish_run(run: Any) -> None:
    """Call wandb.finish() safely — call in finally block."""
    if not _WANDB_AVAILABLE or run is None:
        return
    try:
        wandb.finish(quiet=True)
        print("[wandb] Run finished.", flush=True)
    except Exception as exc:
        print(f"[wandb] finish() error (ignored): {exc}", flush=True)


def log_train_summary(run: Any, metrics: dict) -> None:
    """Log final training metrics as run summary."""
    if not _WANDB_AVAILABLE or run is None:
        return
    summary = {f"summary/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
    if summary:
        wandb.log(summary)


# ---------------------------------------------------------------------------
# Helpers shared by callbacks
# ---------------------------------------------------------------------------

def _wandb_log(metrics: dict, step: int | None = None) -> None:
    """Log to W&B without step-backward warnings.

    HF Trainer (report_to=wandb) may advance the internal step before our
    callbacks fire (e.g. on_evaluate at step 200 while wandb is already at 201).
    Merge supplementary metrics into the current step instead of going backwards.
    """
    if not _WANDB_AVAILABLE:
        return
    if step is not None and wandb.run is not None:
        current = wandb.run.step
        if current is not None and step < current:
            step = current
    wandb.log(metrics, step=step)


def _get_peft_type(model) -> str:
    """Return upper-case PEFT type string, e.g. 'LORA', 'TINYLORA', 'DELORA'."""
    peft_config = getattr(model, "peft_config", None)
    if not peft_config:
        # Unsloth may wrap base_model — try one level deeper
        base = getattr(model, "base_model", None) or getattr(model, "model", None)
        if base is not None:
            peft_config = getattr(base, "peft_config", None)
    if not peft_config:
        return ""
    cfg = next(iter(peft_config.values()), None)
    if cfg is None:
        return ""
    peft_type = getattr(cfg, "peft_type", "") or ""
    if hasattr(peft_type, "value"):
        peft_type = peft_type.value
    return str(peft_type).upper()


def _needs_manual_generate(model, method_name: str = "") -> bool:
    """True when Unsloth fast generate is incompatible (TinyLoRA / DeLoRA)."""
    if method_name.lower() in {"tinylora", "delora"}:
        return True
    return _get_peft_type(model) in {"TINYLORA", "DELORA"}


def _manual_generate(
    model,
    input_ids,
    attention_mask,
    max_new_tokens: int,
    eos_token_id: int,
):
    """Greedy decode via forward() — works for TinyLoRA/DeLoRA where Unsloth fast generate fails."""
    import torch

    generated = input_ids
    attn = attention_mask
    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated, attention_mask=attn, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token = logits[:, -1:, :].argmax(dim=-1)
        generated = torch.cat([generated, next_token], dim=1)
        attn = torch.cat(
            [attn, torch.ones((attn.shape[0], 1), device=attn.device, dtype=attn.dtype)],
            dim=1,
        )
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
    return generated


def _safe_model_generate(
    model,
    inputs,
    max_new_tokens: int,
    pad_token_id,
    eos_token_id,
    force_manual: bool = False,
):
    """Generate tokens, bypassing Unsloth fast kernels for non-standard PEFT adapters."""
    if force_manual or _needs_manual_generate(model):
        return _manual_generate(
            model,
            inputs["input_ids"],
            inputs.get("attention_mask"),
            max_new_tokens,
            eos_token_id,
        )

    return model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.lower().translate(str.maketrans("", "", string.punctuation)).split())


def _compute_em(pred: str, truth: str) -> int:
    return int(_normalize(pred) == _normalize(truth))


def _compute_f1(pred: str, truth: str) -> float:
    pt, tt = _normalize(pred).split(), _normalize(truth).split()
    if not pt and not tt:
        return 1.0
    if not pt or not tt:
        return 0.0
    overlap = sum((Counter(pt) & Counter(tt)).values())
    if not overlap:
        return 0.0
    p, r = overlap / len(pt), overlap / len(tt)
    return 2 * p * r / (p + r)


def _score(pred: str, gold_answers: list[str]) -> tuple[int, float]:
    return (
        max(_compute_em(pred, g) for g in gold_answers),
        max(_compute_f1(pred, g) for g in gold_answers),
    )


_ADAPTER_FILENAMES = {
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "training_args.json",
}

# Real LoRA adapters for Qwen-1.5B are typically several MB; 133B = empty stub.
_MIN_ADAPTER_WEIGHT_BYTES = 50_000


def _adapter_weight_ok(dir_path: Path) -> tuple[bool, str]:
    """Return (ok, message) after checking adapter weight file size."""
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = dir_path / name
        if path.is_file():
            size = path.stat().st_size
            if size < _MIN_ADAPTER_WEIGHT_BYTES:
                return False, f"{name} too small ({size} B) — refusing upload (likely empty stub)"
            return True, f"{name} OK ({size / 1e6:.2f} MB)"
    return False, f"No adapter_model.safetensors/.bin in {dir_path}"


def upload_adapter_artifact(
    run: Any,
    adapter_dir: str | Path,
    method_name: str,
    aliases: list[str] | None = None,
    metadata: dict | None = None,
) -> bool:
    """Upload a finished adapter folder to W&B. Call AFTER model.save_pretrained()."""
    if not _WANDB_AVAILABLE or run is None:
        return False
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.is_dir():
        print(f"[wandb-artifact] Missing dir {adapter_dir}", flush=True)
        return False
    ok, msg = _adapter_weight_ok(adapter_dir)
    print(f"[wandb-artifact] {msg}", flush=True)
    if not ok:
        return False

    artifact = wandb.Artifact(
        name=f"vinewsqa-{method_name}-adapter",
        type="model",
        metadata=metadata or {},
    )
    added = 0
    for path in sorted(adapter_dir.iterdir()):
        if path.is_file() and path.name in _ADAPTER_FILENAMES:
            artifact.add_file(str(path), name=path.name)
            added += 1
    if added == 0:
        print(f"[wandb-artifact] No files to upload in {adapter_dir}", flush=True)
        return False
    run.log_artifact(artifact, aliases=aliases or ["latest", "final"])
    print(f"[wandb-artifact] Uploaded {added} files from {adapter_dir} (aliases={aliases or ['latest', 'final']})", flush=True)
    return True


# ---------------------------------------------------------------------------
# Module 3 — Artifact Auto-Saver Callback
# ---------------------------------------------------------------------------

class WandbAdapterArtifactCallback(TrainerCallback):
    """TrainerCallback that uploads adapter weights to W&B Artifacts.

    Triggers:
      - on_save → upload latest checkpoint-* (validated weight size)
      - on_train_end → upload best/latest checkpoint (NOT save_path —
        save_path is written AFTER trainer.train() returns)
    """

    def __init__(
        self,
        variant: dict,
        project: str,
        run: Any = None,
        upload_checkpoints: bool = True,
        best_eval_loss: float = float("inf"),
    ):
        super().__init__()
        self.variant = variant
        self.project = project
        self.run = run
        self.upload_checkpoints = upload_checkpoints
        self._best_eval_loss = best_eval_loss
        self._best_ckpt: Path | None = None
        self._artifact_aliases: list[str] = []

    def on_save(self, args, state, control, **kwargs):
        if not _WANDB_AVAILABLE or not self.upload_checkpoints:
            return
        output_dir = Path(args.output_dir)
        ckpt_dirs = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if not ckpt_dirs:
            return
        latest_ckpt = ckpt_dirs[-1]
        eval_loss = None
        if state.log_history:
            for entry in reversed(state.log_history):
                if "eval_loss" in entry:
                    eval_loss = entry["eval_loss"]
                    break
        aliases = [f"step-{state.global_step}"]
        if eval_loss is not None and eval_loss < self._best_eval_loss:
            self._best_eval_loss = eval_loss
            self._best_ckpt = latest_ckpt
            aliases.append("best")
        self._upload_dir(
            dir_path=latest_ckpt,
            artifact_name=f"vinewsqa-{self.variant['name']}-ckpt",
            artifact_type="model",
            metadata={"step": state.global_step, "eval_loss": eval_loss},
            aliases=aliases,
        )

    def on_train_end(self, args, state, control, **kwargs):
        """Upload from checkpoint dir only — final save_path is uploaded later via upload_adapter_artifact()."""
        if not _WANDB_AVAILABLE:
            return
        output_dir = Path(args.output_dir)
        ckpt_dirs = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        target = self._best_ckpt or (ckpt_dirs[-1] if ckpt_dirs else None)
        if target is None or not target.exists():
            print(
                "[wandb-artifact] No checkpoint to upload at train_end "
                "(final adapter will be uploaded after save_pretrained).",
                flush=True,
            )
            return
        eval_loss = self._best_eval_loss if self._best_eval_loss < float("inf") else None
        self._upload_dir(
            dir_path=target,
            artifact_name=f"vinewsqa-{self.variant['name']}-ckpt",
            artifact_type="model",
            metadata={"eval_loss": eval_loss, "train_end_checkpoint": True},
            aliases=["train-end"],
        )

    def _upload_dir(
        self,
        dir_path: Path,
        artifact_name: str,
        artifact_type: str,
        metadata: dict,
        aliases: list[str],
    ) -> None:
        ok, msg = _adapter_weight_ok(dir_path)
        print(f"[wandb-artifact] {dir_path.name}: {msg}", flush=True)
        if not ok:
            return
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata=metadata)
        added = 0
        for filename in sorted(dir_path.iterdir()):
            if filename.is_file() and filename.name in _ADAPTER_FILENAMES:
                artifact.add_file(str(filename), name=filename.name)
                added += 1
        if added == 0:
            print(f"[wandb-artifact] No adapter files found in {dir_path}", flush=True)
            return
        self.run.log_artifact(artifact, aliases=aliases)
        print(f"[wandb-artifact] Uploaded {added} files from {dir_path} (aliases={aliases})", flush=True)


# ---------------------------------------------------------------------------
# Module 4 — Visual Generation Table Callback
# ---------------------------------------------------------------------------

class WandbGenerationTableCallback(TrainerCallback):
    """TrainerCallback that logs greedy-decoded predictions to a wandb.Table.

    Runs at every evaluation step and at train_end.
    Uses a fixed set of N_SAMPLES from eval_dataset (reproducible seed).
    """

    def __init__(
        self,
        eval_dataset,
        tokenizer,
        system_prompt: str,
        n_samples: int = 8,
        max_new_tokens: int = 64,
        max_context_chars: int = 300,
        seed: int = 3407,
        method_name: str = "",
    ):
        super().__init__()
        import random

        self.method_name = method_name
        self._use_manual_generate = method_name.lower() in {"tinylora", "delora"}
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars

        rng = random.Random(seed)
        indices = list(range(len(eval_dataset)))
        rng.shuffle(indices)
        self._samples = [eval_dataset[i] for i in indices[:n_samples]]
        self._table_key = "eval/generation_samples"

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None and _needs_manual_generate(model, self.method_name):
            self._use_manual_generate = True
        if self._use_manual_generate:
            print(
                f"[wandb-gentable] {self.method_name or _get_peft_type(model)}: "
                "using manual forward for generation table (Unsloth fast generate incompatible).",
                flush=True,
            )

    def _run_generation(self, model) -> list[dict]:
        import torch

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        PREFIX_RE = __import__("re").compile(
            r"^(đáp án|answer|câu trả lời)\s*[:\-]?\s*", __import__("re").IGNORECASE
        )

        rows = []
        with torch.no_grad():
            for sample in self._samples:
                context = sample.get("context", "") or ""
                question = sample.get("question", "") or ""
                gold_answers = sample.get("gold_answers") or [sample.get("answer", "")]

                messages = [
                    {"role": "system", "content": self.system_prompt.format(context=context)},
                    {"role": "user", "content": question},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=512
                ).to(device)
                out = _safe_model_generate(
                    model,
                    inputs,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    force_manual=self._use_manual_generate,
                )
                raw = self.tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                pred = raw.strip().split("\n")[0].strip().strip("\"'")
                pred = PREFIX_RE.sub("", pred).strip()

                em, f1 = _score(pred, gold_answers)
                rows.append({
                    "context_snippet": context[: self.max_context_chars] + ("…" if len(context) > self.max_context_chars else ""),
                    "question": question,
                    "ground_truth": gold_answers[0],
                    "model_output": pred,
                    "em": em,
                    "f1": round(f1, 4),
                })

        if was_training:
            model.train()
        return rows

    def _log_table(self, model, step: int) -> None:
        if not _WANDB_AVAILABLE:
            return
        try:
            rows = self._run_generation(model)
        except Exception as exc:
            print(
                f"[wandb-gentable] Skipped generation table at step {step} "
                f"({self.method_name or _get_peft_type(model)}): {exc}",
                flush=True,
            )
            return
        if not rows:
            return
        table = wandb.Table(
            columns=["step", "context_snippet", "question", "ground_truth", "model_output", "em", "f1"]
        )
        for row in rows:
            table.add_data(step, row["context_snippet"], row["question"], row["ground_truth"], row["model_output"], row["em"], row["f1"])
        _wandb_log({self._table_key: table, "eval/gen_em": sum(r["em"] for r in rows) / len(rows), "eval/gen_f1": sum(r["f1"] for r in rows) / len(rows)}, step=step)
        print(f"[wandb-gentable] Logged {len(rows)} rows at step {step}", flush=True)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not _WANDB_AVAILABLE or model is None:
            return
        self._log_table(model, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not _WANDB_AVAILABLE or model is None:
            return
        self._log_table(model, state.global_step)


# ---------------------------------------------------------------------------
# Module 2 (extra) — Metrics Callback: perplexity + throughput
# ---------------------------------------------------------------------------

class WandbMetricsCallback(TrainerCallback):
    """TrainerCallback that adds eval/perplexity and train/tokens_per_second."""

    def __init__(self):
        super().__init__()
        self._t0: float | None = None
        self._tokens_seen_t0: int = 0

    def on_train_begin(self, args, state, control, **kwargs):
        import time
        self._t0 = time.time()
        self._tokens_seen_t0 = getattr(state, "num_input_tokens_seen", 0) or 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not _WANDB_AVAILABLE or not metrics:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            try:
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")
            _wandb_log({"eval/perplexity": perplexity}, step=state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not _WANDB_AVAILABLE or not logs:
            return
        import time

        extra: dict[str, Any] = {}

        tokens_now = getattr(state, "num_input_tokens_seen", None)
        if tokens_now and self._t0 is not None:
            elapsed = time.time() - self._t0
            if elapsed > 0:
                tokens_delta = tokens_now - self._tokens_seen_t0
                extra["train/tokens_per_second"] = tokens_delta / elapsed

        # throughput via samples/sec estimate
        if "loss" in logs and state.global_step > 0:
            bs = args.per_device_train_batch_size * args.gradient_accumulation_steps
            if self._t0 is not None:
                elapsed = time.time() - self._t0
                if elapsed > 0:
                    extra["train/samples_per_second"] = (state.global_step * bs) / elapsed

        if extra:
            _wandb_log(extra, step=state.global_step)


# ---------------------------------------------------------------------------
# Module Hardware — WandbHardwareCallback (pynvml power + VRAM + ETA)
# ---------------------------------------------------------------------------

def _pynvml_available() -> bool:
    try:
        import pynvml  # noqa: F401
        return True
    except ImportError:
        return False


class WandbHardwareCallback(TrainerCallback):
    """TrainerCallback that monitors GPU power, VRAM, energy, and ETA.

    Background thread polls pynvml every `poll_interval_s` seconds and
    accumulates energy: energy_wh += power_w * dt / 3600.

    Logs to W&B on every on_log call:
      gpu/power_watts           — instantaneous power draw (W)
      gpu/energy_wh             — cumulative energy consumed (Wh)
      gpu/peak_vram_allocated_gib
      gpu/peak_vram_reserved_gib
      gpu/vram_allocated_gib    — current allocated VRAM
      train/eta_seconds         — estimated time remaining
      train/elapsed_seconds
    """

    def __init__(self, gpu_index: int = 0, poll_interval_s: float = 5.0):
        super().__init__()
        self.gpu_index = gpu_index
        self.poll_interval_s = poll_interval_s

        self._nvml_ok = _pynvml_available()
        self._handle = None
        self._lock = threading.Lock()

        # energy accumulation
        self._energy_wh: float = 0.0
        self._last_poll_t: float | None = None
        self._last_power_w: float = 0.0

        # peak VRAM (also tracked via torch)
        self._peak_vram_alloc_gib: float = 0.0
        self._peak_vram_reserved_gib: float = 0.0

        # timing for ETA
        self._train_start_t: float | None = None
        self._total_steps: int = 0

        # background thread
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None

    def _init_nvml(self) -> None:
        if not self._nvml_ok:
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            print(f"[wandb-hw] pynvml OK — GPU {self.gpu_index}: "
                  f"{pynvml.nvmlDeviceGetName(self._handle)}", flush=True)
        except Exception as exc:
            print(f"[wandb-hw] pynvml init failed ({exc}), hardware metrics disabled.", flush=True)
            self._nvml_ok = False

    def _read_power_w(self) -> float | None:
        """Return instantaneous power in Watts, or None on failure."""
        if not self._nvml_ok or self._handle is None:
            return None
        try:
            import pynvml
            return pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except Exception:
            return None

    def _read_vram_gib(self) -> tuple[float, float] | None:
        """Return (used_gib, total_gib) or None."""
        if not self._nvml_ok or self._handle is None:
            return None
        try:
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return info.used / 1024**3, info.total / 1024**3
        except Exception:
            return None

    def _poll_loop(self) -> None:
        """Background thread: poll power + VRAM every poll_interval_s."""
        while not self._stop_event.is_set():
            now = time.time()
            power_w = self._read_power_w()

            with self._lock:
                if power_w is not None:
                    if self._last_poll_t is not None:
                        dt = now - self._last_poll_t
                        avg_power = (power_w + self._last_power_w) / 2.0
                        self._energy_wh += avg_power * dt / 3600.0
                    self._last_power_w = power_w
                    self._last_poll_t = now

                # VRAM via torch (always available if torch is imported)
                try:
                    import torch
                    if torch.cuda.is_available():
                        alloc = torch.cuda.memory_allocated(self.gpu_index) / 1024**3
                        reserved = torch.cuda.memory_reserved(self.gpu_index) / 1024**3
                        peak_alloc = torch.cuda.max_memory_allocated(self.gpu_index) / 1024**3
                        peak_res = torch.cuda.max_memory_reserved(self.gpu_index) / 1024**3
                        if alloc > self._peak_vram_alloc_gib:
                            self._peak_vram_alloc_gib = peak_alloc
                        if reserved > self._peak_vram_reserved_gib:
                            self._peak_vram_reserved_gib = peak_res
                except Exception:
                    pass

            self._stop_event.wait(self.poll_interval_s)

    # ------------------------------------------------------------------
    # Trainer callback interface
    # ------------------------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):
        self._init_nvml()
        self._train_start_t = time.time()
        self._total_steps = int(state.max_steps or 0)
        self._energy_wh = 0.0
        self._last_poll_t = None
        self._peak_vram_alloc_gib = 0.0
        self._peak_vram_reserved_gib = 0.0

        # Reset torch peak memory counters
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.gpu_index)
        except Exception:
            pass

        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="wandb-hw-poll")
        self._poll_thread.start()
        print(f"[wandb-hw] Hardware monitor started (poll every {self.poll_interval_s}s)", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not _WANDB_AVAILABLE:
            return

        step = state.global_step
        extra: dict[str, Any] = {}

        with self._lock:
            # Energy and power
            extra["gpu/energy_wh"] = round(self._energy_wh, 4)
            extra["gpu/power_watts"] = round(self._last_power_w, 2)
            extra["gpu/peak_vram_allocated_gib"] = round(self._peak_vram_alloc_gib, 3)
            extra["gpu/peak_vram_reserved_gib"] = round(self._peak_vram_reserved_gib, 3)

        # Current VRAM via torch
        try:
            import torch
            if torch.cuda.is_available():
                extra["gpu/vram_allocated_gib"] = round(
                    torch.cuda.memory_allocated(self.gpu_index) / 1024**3, 3
                )
        except Exception:
            pass

        # Elapsed + ETA
        if self._train_start_t is not None:
            elapsed = time.time() - self._train_start_t
            extra["train/elapsed_seconds"] = round(elapsed, 1)
            if step > 0 and self._total_steps > 0:
                steps_remaining = self._total_steps - step
                secs_per_step = elapsed / step
                eta = steps_remaining * secs_per_step
                extra["train/eta_seconds"] = round(eta, 1)
                extra["train/eta_minutes"] = round(eta / 60, 2)

        if extra:
            _wandb_log(extra, step=step)

    def on_train_end(self, args, state, control, **kwargs):
        # Stop background thread
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=10)

        # Final snapshot
        if _WANDB_AVAILABLE:
            with self._lock:
                final_energy = self._energy_wh
                peak_alloc = self._peak_vram_alloc_gib
                peak_res = self._peak_vram_reserved_gib

            # Log final summary metrics
            _wandb_log({
                "gpu/final_energy_wh": round(final_energy, 4),
                "gpu/peak_vram_allocated_gib": round(peak_alloc, 3),
                "gpu/peak_vram_reserved_gib": round(peak_res, 3),
            }, step=state.global_step)
            print(
                f"[wandb-hw] Final energy: {final_energy:.3f} Wh | "
                f"Peak VRAM alloc: {peak_alloc:.2f} GiB | reserved: {peak_res:.2f} GiB",
                flush=True,
            )

        # Shutdown NVML
        if self._nvml_ok:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Dataset artifact helper (one-shot, called outside callbacks)
# ---------------------------------------------------------------------------

def upload_dataset_artifact(
    run: Any,
    profiling_config_path: str | Path,
    artifact_name: str = "vinewsqa-config",
) -> None:
    if not _WANDB_AVAILABLE or run is None:
        return
    path = Path(profiling_config_path)
    if not path.exists():
        return
    artifact = wandb.Artifact(name=artifact_name, type="dataset")
    artifact.add_file(str(path))
    run.log_artifact(artifact)
    print(f"[wandb-artifact] Uploaded dataset artifact: {path}", flush=True)
