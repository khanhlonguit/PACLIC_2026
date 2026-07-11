"""Export eval_compare_adapters_viquad2.json → results.json per adapter (submission format).

LƯU Ý: eval_compare chỉ chứa mẫu metric eval (thường 100 câu validation).
Để nộp benchmark cần ~7301 câu → chạy RUN_SUBMISSION_EXPORT trong notebook
(SUBMISSION_MAX_SAMPLES=None) hoặc dùng export_submission_one_adapter() trên full test_samples.
"""

from __future__ import annotations

import json
from pathlib import Path

COMPARE_EVAL_PATH = Path(__file__).resolve().parent / "eval_compare_adapters_viquad2.json"

# method key in eval JSON → adapter folder containing results.json
ADAPTER_OUTPUT_DIRS = {
    "lora": Path(__file__).resolve().parent / "qwen2.5-1.5b-instruct-lora-viquad2",
    "tinylora": Path(__file__).resolve().parent / "qwen2.5-1.5b-instruct-tinyLora-viquad2",
    "dora": Path(__file__).resolve().parent / "qwen2.5-1.5b-instruct-dora-viquad2",
}


NO_ANSWER_SENTINEL_EXPORT = "Không có đáp án trong đoạn văn"


def _normalize_for_noans(text: str) -> str:
    import re
    import string
    import unicodedata

    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.lower().translate(str.maketrans("", "", string.punctuation)).split())


def _is_no_answer_prediction(pred: str) -> bool:
    return _normalize_for_noans(pred) == _normalize_for_noans(NO_ANSWER_SENTINEL_EXPORT)


def predictions_to_results(predictions: list[dict], *, use_dataset_labels: bool = False) -> dict[str, str]:
    """Map id → submission answer.

    Default (use_dataset_labels=False): dùng output model — sentinel → "", ngược lại → span.
    Phù hợp nộp benchmark test (không có nhãn is_impossible lúc inference).

    use_dataset_labels=True: dùng is_impossible trong eval JSON (cần nhãn đúng, ví dụ validation).
    """
    results: dict[str, str] = {}
    for row in predictions:
        qid = row["id"]
        pred = (row.get("prediction") or "").strip()
        if use_dataset_labels:
            if row.get("is_impossible"):
                results[qid] = ""
            else:
                results[qid] = pred
        elif _is_no_answer_prediction(pred):
            results[qid] = ""
        else:
            results[qid] = pred
    return results


def export_all(compare_path: Path = COMPARE_EVAL_PATH, *, use_dataset_labels: bool = False) -> dict[str, Path]:
    data = json.loads(compare_path.read_text(encoding="utf-8"))
    predictions_by_method = data.get("predictions") or {}
    written: dict[str, Path] = {}

    for method, out_dir in ADAPTER_OUTPUT_DIRS.items():
        preds = predictions_by_method.get(method)
        if not preds:
            print(f"[SKIP] {method}: không có predictions trong {compare_path.name}")
            continue

        results = predictions_to_results(preds, use_dataset_labels=use_dataset_labels)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "results.json"
        out_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        n_empty = sum(1 for v in results.values() if v == "")
        n_span = len(results) - n_empty
        print(f"[OK] {method}: {len(results)} câu → {out_path} (hasAns={n_span}, noAns={n_empty})")
        written[method] = out_path

    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export results.json từ eval_compare JSON")
    parser.add_argument(
        "--use-dataset-labels",
        action="store_true",
        help="Dùng is_impossible trong eval JSON thay vì sentinel của model",
    )
    args = parser.parse_args()
    export_all(use_dataset_labels=args.use_dataset_labels)
