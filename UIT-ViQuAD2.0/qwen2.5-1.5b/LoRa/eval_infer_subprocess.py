"""Standalone eval inference — chạy trong PROCESS RIÊNG, KHÔNG import unsloth.

Vì sao cần file này:
    Unsloth monkey-patch TOÀN CỤC lớp ``Qwen2Attention`` của transformers ngay khi được
    import trong kernel (để chạy LoRA/DoRA nhanh). Sau đó, model nạp bằng transformers
    thuần trong cùng kernel vẫn bị dùng nhầm ``LlamaAttention_fast_forward`` của Unsloth và
    crash: ``AttributeError: 'Qwen2Attention' object has no attribute 'apply_qkv'``. Ngoài
    ra loader "ép offline" của Unsloth còn có thể báo thiếu base weights. Chạy tách hẳn một
    process (interpreter mới, chưa từng import unsloth) đảm bảo transformers KHÔNG bị patch.

Input : --samples-json  (list sample {id, context, question, answer, is_impossible, ...})
Output: --output        (list PREDICTIONS {id, question, is_impossible, has_label,
                          ground_truth, prediction_raw, prediction}) để kernel dùng lại
                          cho cả metric eval lẫn submission export.
Base model + peft_type đọc từ adapter_config.json trong --adapter-dir.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

NO_ANSWER_SENTINEL = "Không có đáp án trong đoạn văn"

SYSTEM_PROMPT = (
    "Bạn là hệ thống trích xuất câu trả lời từ văn bản tiếng Việt. "
    "QUY TẮC BẮT BUỘC:\n"
    "1) Chỉ trả về ĐÚNG cụm từ xuất hiện trong đoạn văn, không thêm từ nào khác.\n"
    "2) Không viết câu hoàn chỉnh, không giải thích, không thêm 'Đáp án:'.\n"
    f"3) Nếu không có đáp án trong đoạn văn, chỉ trả về: {NO_ANSWER_SENTINEL}"
)
USER_PROMPT_TEMPLATE = (
    "Trích xuất câu trả lời từ đoạn văn. Chỉ trả về cụm từ trong đoạn văn.\n\n"
    "Đoạn văn:\n{context}\n\n"
    "Câu hỏi: {question}\n\n"
    f"Câu trả lời (span-only hoặc '{NO_ANSWER_SENTINEL}'):"
)

PREFIX_RE = re.compile(
    r"^(đáp án|answer|câu trả lời|theo đoạn văn|trong đoạn văn)\s*[:\-]?\s*",
    re.IGNORECASE,
)


def build_messages(context, question):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(context=context, question=question)},
    ]


def normalize_text(text):
    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.lower().translate(str.maketrans("", "", string.punctuation)).split())


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


def is_no_answer(pred):
    return normalize_text(pred) == normalize_text(NO_ANSWER_SENTINEL)


def clean_prediction(raw):
    pred = raw.strip().split("\n")[0].strip().strip("\"'")
    return PREFIX_RE.sub("", pred).strip()


def align_to_context(pred, context):
    if not pred or is_no_answer(pred):
        return pred
    idx = context.lower().find(pred.lower())
    if idx >= 0:
        return context[idx:idx + len(pred)]
    words, pred_words = context.split(), normalize_text(pred).split()
    if not pred_words:
        return pred
    n = len(pred_words)
    best_span, best_f1 = pred, 0.0
    for i in range(len(words)):
        for j in range(i + 1, min(i + n + 4, len(words)) + 1):
            span = " ".join(words[i:j])
            f1 = compute_f1(span, pred)
            if f1 > best_f1:
                best_f1, best_span = f1, span
    return best_span.strip() if best_f1 >= 0.5 else pred


def parse_args():
    p = argparse.ArgumentParser(description="Eval inference (no-unsloth subprocess) → predictions JSON")
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--samples-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-samples", type=int, default=0, help="0 = full")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--no-span-postprocess", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if "unsloth" in sys.modules:
        raise RuntimeError(
            "unsloth đã bị import trong process này — mất tác dụng cách ly. Script phải chạy "
            "như một process độc lập, không phải trong kernel đã import unsloth."
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    adapter_dir = Path(args.adapter_dir)
    cfg = json.load(open(adapter_dir / "adapter_config.json", encoding="utf-8"))
    base_model = cfg.get("base_model_name_or_path", args.base_model)
    peft_type = (cfg.get("peft_type") or "").upper()
    _required_cfg = {"TINYLORA": "TinyLoraConfig", "DELORA": "DeloraConfig"}.get(peft_type)
    if _required_cfg:
        try:
            getattr(__import__("peft", fromlist=[_required_cfg]), _required_cfg)
        except Exception as e:
            raise RuntimeError(
                f"PEFT {__import__('peft').__version__} không có {_required_cfg} "
                f"({type(e).__name__}: {e}). Cài peft>=0.19 (TinyLoRA/DeLoRA) rồi chạy lại."
            ) from e

    samples = json.load(open(args.samples_json, encoding="utf-8"))
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
    total = len(samples)
    print(f"[Sub] base={base_model} | adapter={adapter_dir} | peft_type={peft_type} | samples={total}", flush=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Sub] Loading base (dtype={dtype}, device={device})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False).to(device)
    if not getattr(model, "peft_config", None):
        raise RuntimeError("Adapter KHÔNG được gắn.")
    print(f"[Sub] Adapter OK: {list(model.peft_config.keys())}", flush=True)

    model.config.use_cache = True
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Chỉ dùng max_new_tokens → tránh spam "Both max_new_tokens and max_length".
    model.generation_config.max_length = None
    model.generation_config.max_new_tokens = args.max_new_tokens
    for _k in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, _k):
            setattr(model.generation_config, _k, None)

    use_span = not args.no_span_postprocess
    preds = []
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        msgs = build_messages(s["context"], s["question"])
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_length,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_raw = clean_prediction(raw)
        pred = align_to_context(pred_raw, s["context"]) if use_span else pred_raw
        preds.append({
            "id": s.get("id", ""),
            "uit_id": s.get("uit_id", ""),
            "question": s.get("question", ""),
            "is_impossible": s.get("is_impossible"),
            "has_label": s.get("has_label", True),
            "ground_truth": s.get("answer"),
            "prediction_raw": pred_raw,
            "prediction": pred,
        })
        if i == 1 or i % args.log_every == 0 or i == total:
            el = time.time() - t0
            rate = i / max(el, 1e-3)
            eta = (total - i) / max(rate, 1e-3)
            print(f"[Sub] {i}/{total} | {el/60:.1f}m | ETA {eta/60:.1f}m | {rate:.2f} sample/s", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)
    print(f"[Sub] Saved {len(preds)} predictions → {out_path}", flush=True)


if __name__ == "__main__":
    main()
