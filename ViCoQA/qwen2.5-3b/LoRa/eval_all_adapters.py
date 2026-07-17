#!/usr/bin/env python3
"""
Eval tất cả adapter ViCoQA (Qwen2.5-3B) trên test/dev.

Outputs:
  - eval_preds_{method}_{split}.json   (x4: lora/tinylora/dora/delora)
  - eval_compare_adapters_vicoqa_3b.json  (bảng EM/F1 + summary)

Usage:
  python eval_all_adapters.py                  # test split, all 4
  python eval_all_adapters.py --split test
  python eval_all_adapters.py --method lora
  python eval_all_adapters.py --load-in-4bit   # nếu VRAM thấp khi eval
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE.parent.parent  # ViCoQA/

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
PROFILING_CONFIG_PATH = HERE / "profiling_config.json"
COMPARE_EVAL_PATH = HERE / "eval_compare_adapters_vicoqa_3b.json"

ADAPTER_VARIANTS = [
    {"name": "lora", "save_path": "qwen2.5-3b-instruct-lora-vicoqa"},
    {"name": "tinylora", "save_path": "qwen2.5-3b-instruct-tinylora-vicoqa"},
    {"name": "dora", "save_path": "qwen2.5-3b-instruct-dora-vicoqa"},
    {"name": "delora", "save_path": "qwen2.5-3b-instruct-delora-vicoqa"},
]

MAX_NEW_TOKENS = 64
EVAL_LOG_EVERY = 20


def validate_adapter(adapter_path: Path) -> None:
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter folder không tồn tại: {adapter_path}")
    cfg = adapter_path / "adapter_config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"Thiếu {cfg}")
    weight_candidates = [
        adapter_path / "adapter_model.safetensors",
        adapter_path / "adapter_model.bin",
        adapter_path / "pytorch_model.bin",
    ]
    # TinyLoRA/DeLoRA có thể shard
    weight_candidates.extend(sorted(adapter_path.glob("adapter_model*.safetensors")))
    weight_candidates.extend(sorted(adapter_path.glob("*.safetensors")))
    found = next((p for p in weight_candidates if p.exists() and p.stat().st_size > 1024), None)
    if found is None:
        raise FileNotFoundError(
            f"Không thấy weight file trong {adapter_path} "
            "(cần adapter_model.safetensors hoặc tương đương)."
        )
    print(f"Adapter OK: {adapter_path.name} | {found.name} ({found.stat().st_size / 1024**2:.1f} MB)", flush=True)


def run_infer(
    method: str,
    adapter_path: Path,
    dialogs_json: Path,
    output_path: Path,
    *,
    max_seq_length: int,
    max_dialogs: int,
    load_in_4bit: bool,
) -> list[dict]:
    script = HERE / "eval_infer_subprocess.py"
    if not script.exists():
        raise FileNotFoundError(script)

    with tempfile.TemporaryDirectory(prefix="vicoqa_eval_") as tmp:
        tmp_dialogs = Path(tmp) / "dialogs.json"
        tmp_preds = Path(tmp) / "preds.json"
        # copy dialogs into tmp so subprocess không phụ thuộc path dài
        tmp_dialogs.write_text(dialogs_json.read_text(encoding="utf-8"), encoding="utf-8")

        cmd = [
            sys.executable,
            str(script),
            "--adapter-dir", str(adapter_path),
            "--dialogs-json", str(tmp_dialogs),
            "--output", str(tmp_preds),
            "--base-model", BASE_MODEL_NAME,
            "--max-seq-length", str(max_seq_length),
            "--max-new-tokens", str(MAX_NEW_TOKENS),
            "--log-every", str(EVAL_LOG_EVERY),
        ]
        if max_dialogs and max_dialogs > 0:
            cmd.extend(["--max-dialogs", str(max_dialogs)])
        if load_in_4bit:
            cmd.append("--load-in-4bit")

        print(f"\n=== EVAL {method} ===", flush=True)
        print(" ".join(cmd), flush=True)
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        proc = subprocess.Popen(
            cmd, cwd=str(HERE), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line.rstrip("\n"), flush=True)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"{method}: eval_infer_subprocess exit {rc}")

        preds = json.loads(tmp_preds.read_text(encoding="utf-8"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)
    print(f"Saved predictions → {output_path} ({len(preds)} turns)", flush=True)
    return preds


def metrics_from_preds(preds: list[dict], method: str, adapter: str, n_dialogs: int) -> dict:
    ems = [p["em"] for p in preds]
    f1s = [p["f1"] for p in preds]
    return {
        "method": method,
        "adapter": adapter,
        "coqa_em": round(100 * sum(ems) / max(len(ems), 1), 4),
        "coqa_f1": round(100 * sum(f1s) / max(len(f1s), 1), 4),
        "n_turns": len(preds),
        "n_dialogs": n_dialogs,
    }


def print_table(summary: dict[str, dict], split: str) -> None:
    line = "=" * 72
    print("\n" + line)
    print(f"  SO SÁNH ADAPTERS — ViCoQA ({split}) | Qwen2.5-3B-Instruct")
    print(line)
    print(f"{'Method':<12} {'CoQA EM':>12} {'CoQA F1':>12} {'Turns':>10}")
    print("-" * 72)
    for method, m in summary.items():
        print(f"{method:<12} {m['coqa_em']:>11.2f}% {m['coqa_f1']:>11.2f}% {m['n_turns']:>10}")
    print(line)


def main():
    parser = argparse.ArgumentParser(description="Eval 4 ViCoQA adapters on test/dev")
    parser.add_argument("--split", choices=["test", "dev"], default="test")
    parser.add_argument("--method", choices=[v["name"] for v in ADAPTER_VARIANTS], default=None)
    parser.add_argument("--all", action="store_true", help="Eval all 4 (default nếu không --method)")
    parser.add_argument("--max-dialogs", type=int, default=0, help="0 = all")
    parser.add_argument("--max-seq-length", type=int, default=0, help="0 = đọc profiling_config.json")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    if args.max_seq_length > 0:
        max_seq = args.max_seq_length
    elif PROFILING_CONFIG_PATH.exists():
        max_seq = json.loads(PROFILING_CONFIG_PATH.read_text(encoding="utf-8"))["max_seq_length"]
    else:
        max_seq = 1280
        print(f"[warn] dùng max_seq_length mặc định {max_seq}", flush=True)

    split_path = DATASET_ROOT / f"{args.split}.json"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    dialogs = json.loads(split_path.read_text(encoding="utf-8"))
    n_dialogs = len(dialogs) if not args.max_dialogs else min(len(dialogs), args.max_dialogs)
    print(
        f"Split={args.split} | dialogs={len(dialogs)} | eval_dialogs={n_dialogs} | "
        f"max_seq={max_seq} | base={BASE_MODEL_NAME}",
        flush=True,
    )

    methods = [args.method] if args.method else [v["name"] for v in ADAPTER_VARIANTS]
    variant_map = {v["name"]: v for v in ADAPTER_VARIANTS}

    all_results: dict[str, dict] = {}
    summary: dict[str, dict] = {}

    for method in methods:
        adapter_rel = variant_map[method]["save_path"]
        adapter_path = HERE / adapter_rel
        validate_adapter(adapter_path)
        pred_path = HERE / f"eval_preds_{method}_{args.split}.json"
        preds = run_infer(
            method,
            adapter_path,
            split_path,
            pred_path,
            max_seq_length=max_seq,
            max_dialogs=args.max_dialogs,
            load_in_4bit=args.load_in_4bit,
        )
        m = metrics_from_preds(preds, method, adapter_rel, n_dialogs)
        summary[method] = m
        all_results[method] = {"metrics": m, "predictions": preds, "pred_file": str(pred_path.name)}
        print(f"→ {method}: EM={m['coqa_em']:.2f}% | F1={m['coqa_f1']:.2f}%", flush=True)

    print_table(summary, args.split)

    payload = {
        "dataset": "ViCoQA",
        "eval_split": args.split,
        "base_model": BASE_MODEL_NAME,
        "max_seq_length": max_seq,
        "pred_files": {k: v["pred_file"] for k, v in all_results.items()},
        "summary": summary,
        "predictions": {k: v["predictions"] for k, v in all_results.items()},
    }
    with open(COMPARE_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved comparison → {COMPARE_EVAL_PATH}", flush=True)
    print("Per-adapter prediction files:")
    for method in methods:
        print(f"  - eval_preds_{method}_{args.split}.json")


if __name__ == "__main__":
    main()
