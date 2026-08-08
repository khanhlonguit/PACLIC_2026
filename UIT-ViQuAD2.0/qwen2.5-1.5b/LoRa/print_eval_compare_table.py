"""In bảng tổng hợp EM / F1 so sánh các adapter từ eval_compare_adapters_viquad2.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_EVAL_PATH = Path(__file__).resolve().parent / "eval_compare_adapters_viquad2.json"

# Thứ tự hiển thị ưu tiên khi so sánh 3 biến thể
METHOD_ORDER = ("lora", "tinylora", "dora")


def _mean_pct(values: list[float]) -> float | None:
    if not values:
        return None
    return 100.0 * sum(values) / len(values)


def metrics_from_predictions(preds: list[dict]) -> dict:
    """Tính lại EM/F1 tổng + HasAns / NoAns từ danh sách prediction."""
    em_all, f1_all = [], []
    has_em, has_f1, no_em = [], [], []

    for row in preds:
        em = row.get("em")
        f1 = row.get("f1")
        if em is None or f1 is None:
            continue
        em_all.append(float(em))
        f1_all.append(float(f1))
        if row.get("is_impossible"):
            no_em.append(float(em))
        else:
            has_em.append(float(em))
            has_f1.append(float(f1))

    return {
        "overall_em": _mean_pct(em_all),
        "overall_f1": _mean_pct(f1_all),
        "hasans_em": _mean_pct(has_em),
        "hasans_f1": _mean_pct(has_f1),
        "noans_accuracy": _mean_pct(no_em),
        "n_hasans": len(has_em),
        "n_noans": len(no_em),
        "total": len(em_all),
    }


def collect_rows(payload: dict) -> list[dict]:
    summary = payload.get("summary") or {}
    predictions = payload.get("predictions") or {}
    methods = list(dict.fromkeys([*METHOD_ORDER, *summary.keys(), *predictions.keys()]))
    methods = [m for m in methods if m in summary or m in predictions]

    rows = []
    for method in methods:
        s = summary.get(method, {})
        recomputed = metrics_from_predictions(predictions.get(method, []))

        hasans_em = s.get("hasans_em", recomputed["hasans_em"])
        hasans_f1 = s.get("hasans_f1", recomputed["hasans_f1"])
        noans = s.get("noans_accuracy", recomputed["noans_accuracy"])
        n_has = s.get("n_hasans", recomputed["n_hasans"])
        n_no = s.get("n_noans", recomputed["n_noans"])
        total = s.get("total", recomputed["total"])

        rows.append(
            {
                "method": method,
                "adapter": s.get("adapter", ""),
                "hasans_em": hasans_em,
                "hasans_f1": hasans_f1,
                "noans_accuracy": noans,
                "overall_em": recomputed["overall_em"],
                "overall_f1": recomputed["overall_f1"],
                "n_hasans": n_has,
                "n_noans": n_no,
                "total": total,
            }
        )
    return rows


def _fmt(value: float | None, width: int = 11) -> str:
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{value:>{width - 1}.2f}%"


def print_table(payload: dict, rows: list[dict]) -> None:
    dataset = payload.get("dataset", "?")
    split = payload.get("eval_split", "?")
    base = payload.get("base_model", "?")
    line = "=" * 88

    print()
    print(line)
    print(f"  SO SÁNH ADAPTERS — {dataset} ({split})")
    print(f"  Base: {base}")
    print(line)
    print(
        f"{'Method':<12} {'HasAns EM':>11} {'HasAns F1':>11} "
        f"{'Overall EM':>11} {'Overall F1':>11} {'NoAns Acc':>11} {'n':>10}"
    )
    print("-" * 88)

    for row in rows:
        n = f"{row['n_hasans']}/{row['n_noans']}"
        print(
            f"{row['method']:<12} "
            f"{_fmt(row['hasans_em'])} {_fmt(row['hasans_f1'])} "
            f"{_fmt(row['overall_em'])} {_fmt(row['overall_f1'])} "
            f"{_fmt(row['noans_accuracy'])} {n:>10}"
        )

    print(line)

    if rows:
        best_em = max(rows, key=lambda r: (r["overall_em"] is not None, r["overall_em"] or -1))
        best_f1 = max(rows, key=lambda r: (r["overall_f1"] is not None, r["overall_f1"] or -1))
        print(
            f"  Best Overall EM : {best_em['method']} ({_fmt(best_em['overall_em']).strip()})"
        )
        print(
            f"  Best Overall F1 : {best_f1['method']} ({_fmt(best_f1['overall_f1']).strip()})"
        )
        print(line)

    expected = set(METHOD_ORDER)
    present = {r["method"] for r in rows}
    missing = expected - present
    if missing:
        print(f"  [Note] Thiếu adapter trong file: {', '.join(sorted(missing))}")
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=DEFAULT_EVAL_PATH,
        help="Đường dẫn eval_compare_adapters_viquad2.json",
    )
    args = parser.parse_args()

    path = args.eval_json
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = collect_rows(payload)
    if not rows:
        raise SystemExit("Không có adapter nào trong summary/predictions.")

    print_table(payload, rows)


if __name__ == "__main__":
    main()
