#!/usr/bin/env python3
"""So sánh CoQA EM/F1 từ các file eval_preds_*_test.json (không cần GPU).

Usage:
  python compare_eval_preds.py
  python compare_eval_preds.py --split test
  python compare_eval_preds.py --dir .
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METHODS = ["lora", "tinylora", "dora", "delora"]


def metrics_from_preds(preds: list[dict]) -> dict:
    n = len(preds)
    if n == 0:
        return {"coqa_em": 0.0, "coqa_f1": 0.0, "n_turns": 0}
    return {
        "coqa_em": round(100 * sum(p["em"] for p in preds) / n, 4),
        "coqa_f1": round(100 * sum(p["f1"] for p in preds) / n, 4),
        "n_turns": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare EM/F1 across eval_preds_* files")
    parser.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--split", default="test", help="Suffix: eval_preds_{method}_{split}.json")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional JSON output path (default: eval_compare_adapters_vicoqa.json)",
    )
    args = parser.parse_args()

    root: Path = args.dir
    summary = {}
    missing = []

    for method in METHODS:
        path = root / f"eval_preds_{method}_{args.split}.json"
        if not path.exists():
            missing.append(path.name)
            continue
        preds = json.loads(path.read_text(encoding="utf-8"))
        m = metrics_from_preds(preds)
        m["method"] = method
        m["pred_file"] = path.name
        summary[method] = m

    if not summary:
        raise SystemExit(f"Không tìm thấy eval_preds_*_{args.split}.json trong {root}")

    line = "=" * 72
    print(line)
    print(f"  SO SÁNH ADAPTERS — ViCoQA ({args.split}) | Qwen2.5-1.5B-Instruct")
    print(line)
    print(f"{'Method':<12} {'CoQA EM':>12} {'CoQA F1':>12} {'Turns':>10}")
    print("-" * 72)
    for method in METHODS:
        if method not in summary:
            continue
        m = summary[method]
        print(f"{method:<12} {m['coqa_em']:>11.2f}% {m['coqa_f1']:>11.2f}% {m['n_turns']:>10}")
    print(line)

    if missing:
        print("Thiếu file:", ", ".join(missing))

    # Best by F1 then EM
    best = max(summary.values(), key=lambda x: (x["coqa_f1"], x["coqa_em"]))
    print(f"Best (F1): {best['method']} | EM={best['coqa_em']:.2f}% | F1={best['coqa_f1']:.2f}%")

    out = args.save or (root / "eval_compare_adapters_vicoqa.json")
    payload = {
        "dataset": "ViCoQA",
        "eval_split": args.split,
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "summary": summary,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
