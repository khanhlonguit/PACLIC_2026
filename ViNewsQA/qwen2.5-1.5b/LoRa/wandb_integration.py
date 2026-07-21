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
) -> Any:
    """Open a new W&B run for one adapter variant. Returns the run object."""
    if not _WANDB_AVAILABLE:
        return None

    name = variant["name"]
    lr = config.get("learning_rate", 2e-4)
    r = config.get("r", "?")
    run_name = f"vinewsqa-{name}-r{r}-lr{lr:.0e}"

    _tags = [name, "vinewsqa", "unsloth"] + (tags or [])

    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=run_name,
        config=config,
        tags=_tags,
        reinit=True,
    )
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
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "training_args.json",
}


# ---------------------------------------------------------------------------
# Module 3 — Artifact Auto-Saver Callback
# ---------------------------------------------------------------------------

class WandbAdapterArtifactCallback:
    """TrainerCallback that uploads adapter weights to W&B Artifacts.

    Triggers:
      - on_save  → upload latest checkpoint-* directory (lightweight)
      - on_train_end → upload final adapter from variant["save_path"]
    """

    def __init__(
        self,
        variant: dict,
        project: str,
        run: Any = None,
        upload_checkpoints: bool = True,
        best_eval_loss: float = float("inf"),
    ):
        self.variant = variant
        self.project = project
        self.run = run
        self.upload_checkpoints = upload_checkpoints
        self._best_eval_loss = best_eval_loss
        self._artifact_aliases: list[str] = []

    # ------------------------------------------------------------------
    # Trainer callback interface
    # ------------------------------------------------------------------

    def on_save(self, args, state, control, **kwargs):
        if not _WANDB_AVAILABLE or not self.upload_checkpoints:
            return
        output_dir = Path(args.output_dir)
        # Find latest checkpoint dir
        ckpt_dirs = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if not ckpt_dirs:
            return
        latest_ckpt = ckpt_dirs[-1]
        eval_loss = state.log_history[-1].get("eval_loss") if state.log_history else None
        aliases = [f"step-{state.global_step}"]
        if eval_loss is not None and eval_loss < self._best_eval_loss:
            self._best_eval_loss = eval_loss
            aliases.append("best")
        self._upload_dir(
            dir_path=latest_ckpt,
            artifact_name=f"vinewsqa-{self.variant['name']}-ckpt",
            artifact_type="model",
            metadata={"step": state.global_step, "eval_loss": eval_loss},
            aliases=aliases,
        )

    def on_train_end(self, args, state, control, **kwargs):
        if not _WANDB_AVAILABLE:
            return
        save_path = Path(self.variant["save_path"])
        if not save_path.exists():
            print(f"[wandb-artifact] save_path {save_path} not found yet, skipping final upload.", flush=True)
            return
        eval_loss = None
        for entry in reversed(state.log_history):
            if "eval_loss" in entry:
                eval_loss = entry["eval_loss"]
                break
        self._upload_dir(
            dir_path=save_path,
            artifact_name=f"vinewsqa-{self.variant['name']}-adapter",
            artifact_type="model",
            metadata={"eval_loss": eval_loss, "final": True},
            aliases=["latest", "final"],
        )

    def _upload_dir(
        self,
        dir_path: Path,
        artifact_name: str,
        artifact_type: str,
        metadata: dict,
        aliases: list[str],
    ) -> None:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata=metadata)
        added = 0
        for filename in sorted(dir_path.iterdir()):
            if filename.name in _ADAPTER_FILENAMES:
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

class WandbGenerationTableCallback:
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
    ):
        import random

        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars

        rng = random.Random(seed)
        indices = list(range(len(eval_dataset)))
        rng.shuffle(indices)
        self._samples = [eval_dataset[i] for i in indices[:n_samples]]
        self._table_key = "eval/generation_samples"

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
                out = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
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
        rows = self._run_generation(model)
        table = wandb.Table(
            columns=["step", "context_snippet", "question", "ground_truth", "model_output", "em", "f1"]
        )
        for row in rows:
            table.add_data(step, row["context_snippet"], row["question"], row["ground_truth"], row["model_output"], row["em"], row["f1"])
        wandb.log({self._table_key: table, "eval/gen_em": sum(r["em"] for r in rows) / len(rows), "eval/gen_f1": sum(r["f1"] for r in rows) / len(rows)}, step=step)
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

class WandbMetricsCallback:
    """TrainerCallback that adds eval/perplexity and train/tokens_per_second."""

    def __init__(self):
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
            wandb.log({"eval/perplexity": perplexity}, step=state.global_step)

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
            wandb.log(extra, step=state.global_step)


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
