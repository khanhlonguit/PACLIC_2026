"""
evaluate_metrics.py
-------------------
Tính EM (Exact Match), F1-score (token-level) và BERTScore
cho các file eval_preds của ViCoQA / Qwen2.5-3B.

Cài đặt thư viện cần thiết:
    pip install bert-score

Chạy:
    python evaluate_metrics.py
"""

import json
import os
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Helper: normalise Vietnamese text (lowercase + strip punctuation)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase, strip leading/trailing whitespace, remove punctuation."""
    text = text.lower().strip()
    # Remove punctuation (Unicode-aware)
    text = "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("P")
    )
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    return normalize_text(text).split()


# ─────────────────────────────────────────────────────────────────────────────
# EM & F1 (so sanh prediction voi *moi* gold_ref, lay max)
# ─────────────────────────────────────────────────────────────────────────────

def exact_match_score(prediction: str, gold_refs: list) -> int:
    pred_norm = normalize_text(prediction)
    return int(any(pred_norm == normalize_text(ref) for ref in gold_refs))


def f1_score_single(prediction: str, ground_truth: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(ground_truth)

    if not pred_tokens or not gold_tokens:
        return int(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall    = num_common / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1_score_multi(prediction: str, gold_refs: list) -> float:
    return max(f1_score_single(prediction, ref) for ref in gold_refs)


# ─────────────────────────────────────────────────────────────────────────────
# Load & evaluate one JSON file
# ─────────────────────────────────────────────────────────────────────────────

def load_file(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_em_f1(records: list) -> tuple:
    """Returns (EM_mean, F1_mean) recomputed from raw predictions."""
    em_scores, f1_scores = [], []
    for r in records:
        pred      = r["prediction"]
        gold_refs = r.get("gold_refs", [r["ground_truth"]])

        em_scores.append(exact_match_score(pred, gold_refs))
        f1_scores.append(f1_score_multi(pred, gold_refs))

    em  = 100.0 * sum(em_scores)  / len(em_scores)
    f1  = 100.0 * sum(f1_scores)  / len(f1_scores)
    return em, f1


# ─────────────────────────────────────────────────────────────────────────────
# BERTScore
# ─────────────────────────────────────────────────────────────────────────────

def compute_bertscore(
    records: list,
    model_name: str = "bert-base-multilingual-cased",
    batch_size: int = 32,
    device=None,
) -> tuple:
    """
    BERTScore implementation truc tiep bang transformers (khong dung bert_score lib).
    Returns (P_mean, R_mean, F1_mean) in % averaged over all samples.
    Dung gold_refs[0] lam reference chinh.

    Cach tinh:
      - Lay last_hidden_state (token embeddings) tu BERT
      - Normalize -> cosine similarity matrix [T_pred x T_ref]
      - Precision = mean(max similarity theo chieu ref cho moi token pred)
      - Recall    = mean(max similarity theo chieu pred cho moi token ref)
      - F1        = harmonic mean(P, R)
    """
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    predictions = [r["prediction"] or " " for r in records]
    references  = [(r.get("gold_refs", [r["ground_truth"]])[0] or " ") for r in records]

    print(f"  [BERTScore] device={device}, model={model_name}, n={len(predictions)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def get_embeddings(texts):
        """Return list of (embedding_tensor, mask_tensor) per sample."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = model(**enc)
            embs  = out.last_hidden_state.cpu()   # [B, T, H]
            masks = enc["attention_mask"].cpu()   # [B, T]
            for j in range(len(batch)):
                # Only keep non-padding tokens
                mask_j = masks[j].bool()
                results.append(embs[j][mask_j])   # [T_j, H]
        return results

    pred_embs = get_embeddings(predictions)
    ref_embs  = get_embeddings(references)

    P_scores, R_scores, F_scores = [], [], []
    for p_emb, r_emb in zip(pred_embs, ref_embs):
        # Normalize to unit vectors
        p_emb = F.normalize(p_emb, dim=-1)  # [Tp, H]
        r_emb = F.normalize(r_emb, dim=-1)  # [Tr, H]

        sim = torch.mm(p_emb, r_emb.T)      # [Tp, Tr]

        # Precision: each pred token -> best matching ref token
        prec = sim.max(dim=1).values.mean().item()
        # Recall: each ref token -> best matching pred token
        rec  = sim.max(dim=0).values.mean().item()
        # F1
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        P_scores.append(prec)
        R_scores.append(rec)
        F_scores.append(f1)

    P_mean = sum(P_scores) / len(P_scores) * 100
    R_mean = sum(R_scores) / len(R_scores) * 100
    F_mean = sum(F_scores) / len(F_scores) * 100

    return P_mean, R_mean, F_mean



# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

FILES = {
    "LoRA"     : BASE_DIR / "eval_preds_lora_test.json",
    "TinyLoRA" : BASE_DIR / "eval_preds_tinylora_test.json",
    "DoRA"     : BASE_DIR / "eval_preds_dora_test.json",
    "DeLoRA"   : BASE_DIR / "eval_preds_delora_test.json",
}

# BERTScore model (multilingual BERT works well for Vietnamese)
BERTSCORE_MODEL = "bert-base-multilingual-cased"
# Set BERTSCORE_MODEL = "vinai/phobert-base" neu muon dung PhoBERT


def main():
    results = {}

    print("=" * 70)
    print("  ViCoQA Evaluation  |  Qwen2.5-3B")
    print("=" * 70)

    for name, path in FILES.items():
        if not path.exists():
            print(f"\n[WARNING] File not found, skipping: {path}")
            continue

        print(f"\n> {name}  ({path.name})")
        records = load_file(str(path))
        print(f"  Loaded {len(records)} records.")

        # EM & F1
        em, f1 = compute_em_f1(records)
        print(f"  EM  = {em:.2f}%")
        print(f"  F1  = {f1:.2f}%")

        # BERTScore
        bp, br, bf = compute_bertscore(records, model_name=BERTSCORE_MODEL)
        print(f"  BERTScore  P={bp:.2f}%  R={br:.2f}%  F1={bf:.2f}%")

        results[name] = {
            "n_samples"       : len(records),
            "EM"              : round(em, 4),
            "F1"              : round(f1, 4),
            "BERTScore_P"     : round(bp, 4),
            "BERTScore_R"     : round(br, 4),
            "BERTScore_F1"    : round(bf, 4),
        }

    # -- Summary table -------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    header = f"{'Model':<12} {'N':>6} {'EM':>8} {'F1':>8} {'BS-P':>8} {'BS-R':>8} {'BS-F1':>8}"
    print(header)
    print("-" * 70)
    for name, m in results.items():
        print(
            f"{name:<12} {m['n_samples']:>6} "
            f"{m['EM']:>8.2f} {m['F1']:>8.2f} "
            f"{m['BERTScore_P']:>8.2f} {m['BERTScore_R']:>8.2f} {m['BERTScore_F1']:>8.2f}"
        )
    print("=" * 70)

    # -- Save JSON results ---------------------------------------------------
    out_path = BASE_DIR / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Results saved to: {out_path}")


if __name__ == "__main__":
    main()
