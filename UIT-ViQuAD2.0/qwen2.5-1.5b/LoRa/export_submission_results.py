"""Export predictions → results.json per adapter (submission format).

Keys trong results.json dùng ``uit_id`` từ test split (ví dụ ``uit_000001``),
không dùng ``id`` nội bộ (ví dụ ``0001-0001-0001``).

Nguồn dữ liệu (ưu tiên):
1. ``results.json`` hiện có trong thư mục adapter (remap key id → uit_id) — full test ~7301
2. ``eval_compare_adapters_viquad2.json`` predictions — thường chỉ 100 câu validation smoke test

Để sinh full test từ đầu: chạy RUN_SUBMISSION_EXPORT trong notebook
(SUBMISSION_MAX_SAMPLES=None) hoặc export_submission_one_adapter() trên full test_samples.
"""

from __future__ import annotations

import json
from pathlib import Path

COMPARE_EVAL_PATH = Path(__file__).resolve().parent / "eval_compare_adapters_viquad2.json"
DATASET_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_JSON_PATH = DATASET_ROOT / "test_viquad2.json"
DATASET_NAME = "taidng/UIT-ViQuAD2.0"

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


def load_id_to_uit_id_map(*, test_json_path: Path = TEST_JSON_PATH) -> dict[str, str]:
    """Build mapping id → uit_id từ test split."""
    if test_json_path.exists():
        rows = json.loads(test_json_path.read_text(encoding="utf-8"))
        if rows and isinstance(rows[0], dict) and "uit_id" in rows[0]:
            return {row["id"]: row["uit_id"] for row in rows if row.get("id") and row.get("uit_id")}

    from datasets import load_dataset

    split = load_dataset(DATASET_NAME, split="test")
    return {row["id"]: row["uit_id"] for row in split if row.get("id") and row.get("uit_id")}


def submission_key(row: dict, id_to_uit: dict[str, str]) -> str:
    uit_id = row.get("uit_id")
    if uit_id:
        return uit_id
    sample_id = row.get("id")
    if not sample_id:
        raise KeyError(f"Prediction row thiếu cả uit_id lẫn id: {row!r}")
    if sample_id not in id_to_uit:
        raise KeyError(f"Không tìm thấy uit_id cho id={sample_id!r} trong test split")
    return id_to_uit[sample_id]


def remap_results_keys(results: dict[str, str], id_to_uit: dict[str, str]) -> dict[str, str]:
    """Remap results.json keys từ id sang uit_id (bỏ qua nếu đã là uit_id)."""
    if not results:
        return results

    sample_key = next(iter(results))
    if sample_key.startswith("uit_"):
        return results

    remapped: dict[str, str] = {}
    for sample_id, answer in results.items():
        if sample_id not in id_to_uit:
            raise KeyError(f"Không tìm thấy uit_id cho id={sample_id!r} trong test split")
        remapped[id_to_uit[sample_id]] = answer
    return remapped


def predictions_to_results(
    predictions: list[dict],
    id_to_uit: dict[str, str],
    *,
    use_dataset_labels: bool = False,
) -> dict[str, str]:
    """Map uit_id → submission answer."""
    results: dict[str, str] = {}
    for row in predictions:
        qid = submission_key(row, id_to_uit)
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


def export_all(
    compare_path: Path = COMPARE_EVAL_PATH,
    *,
    use_dataset_labels: bool = False,
    test_json_path: Path = TEST_JSON_PATH,
) -> dict[str, Path]:
    id_to_uit = load_id_to_uit_id_map(test_json_path=test_json_path)
    print(f"[INFO] Loaded {len(id_to_uit)} id->uit_id mappings from test split")

    compare_data = {}
    if compare_path.exists():
        compare_data = json.loads(compare_path.read_text(encoding="utf-8"))
    predictions_by_method = compare_data.get("predictions") or {}
    written: dict[str, Path] = {}

    for method, out_dir in ADAPTER_OUTPUT_DIRS.items():
        preds = predictions_by_method.get(method)
        existing_path = out_dir / "results.json"
        existing_results: dict[str, str] | None = None
        if existing_path.exists():
            existing_results = json.loads(existing_path.read_text(encoding="utf-8"))

        if existing_results and (not preds or len(existing_results) > len(preds)):
            results = remap_results_keys(existing_results, id_to_uit)
            source = f"remap {existing_path.name} ({len(existing_results)} samples)"
        elif preds:
            results = predictions_to_results(preds, id_to_uit, use_dataset_labels=use_dataset_labels)
            source = f"predictions in {compare_path.name} ({len(preds)} samples)"
        else:
            print(f"[SKIP] {method}: no results.json or predictions")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "results.json"
        out_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        n_empty = sum(1 for v in results.values() if v == "")
        n_span = len(results) - n_empty
        print(
            f"[OK] {method}: {len(results)} samples -> {out_path} "
            f"(hasAns={n_span}, noAns={n_empty}) [{source}]"
        )
        written[method] = out_path

    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export results.json (uit_id keys) từ predictions hoặc remap file hiện có")
    parser.add_argument(
        "--use-dataset-labels",
        action="store_true",
        help="Dùng is_impossible trong eval JSON thay vì sentinel của model",
    )
    parser.add_argument(
        "--compare-path",
        type=Path,
        default=COMPARE_EVAL_PATH,
        help="Đường dẫn eval_compare JSON (mặc định: eval_compare_adapters_viquad2.json)",
    )
    parser.add_argument(
        "--test-json",
        type=Path,
        default=TEST_JSON_PATH,
        help="test_viquad2.json local (nếu có); không có thì tải từ HuggingFace",
    )
    args = parser.parse_args()
    export_all(
        compare_path=args.compare_path,
        use_dataset_labels=args.use_dataset_labels,
        test_json_path=args.test_json,
    )
