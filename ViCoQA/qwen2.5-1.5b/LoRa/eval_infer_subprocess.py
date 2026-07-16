"""Standalone ViCoQA eval inference — separate process, NO unsloth import.

Processes CoQA-style multi-turn dialogs sequentially: for each dialog, generate
turn 1..5 with prior Q-A history (predicted answers), matching training layout.

Input : --dialogs-json  (ViCoQA dialog list with story/questions/answers)
Output: --output        (list of per-turn predictions with gold refs for scoring)
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

SYSTEM_PROMPT = (
    "Bạn là trợ lý hỏi-đáp tiếng Việt. Dựa trên đoạn văn dưới đây, "
    "trả lời ngắn gọn, tự nhiên theo ngữ cảnh hội thoại.\n\n"
    "Đoạn văn:\n{story}"
)

PREFIX_RE = re.compile(
    r"^(đáp án|answer|câu trả lời|theo đoạn văn|trong đoạn văn)\s*[:\-]?\s*",
    re.IGNORECASE,
)


def build_messages(story: str, history: list[tuple[str, str]], question: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(story=story)}]
    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})
    return messages


def clean_prediction(raw: str) -> str:
    pred = raw.strip().split("\n")[0].strip().strip("\"'")
    return PREFIX_RE.sub("", pred).strip()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.lower().translate(str.maketrans("", "", string.punctuation)).split())


def compute_f1(pred: str, truth: str) -> float:
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


def compute_em(pred: str, truth: str) -> int:
    return int(normalize_text(pred) == normalize_text(truth))


def gold_refs_for_turn(dialog: dict, turn_idx: int) -> list[str]:
    refs = [dialog["answers"][turn_idx]["input_text"].strip()]
    extra = dialog.get("additional_answers") or {}
    alt_list = extra.get("0") or []
    if turn_idx < len(alt_list):
        alt = alt_list[turn_idx].get("input_text", "").strip()
        if alt and alt not in refs:
            refs.append(alt)
    return [r for r in refs if r]


def score_turn(pred: str, gold_list: list[str]) -> tuple[int, float]:
    if not gold_list:
        return 0, 0.0
    em = max(compute_em(pred, g) for g in gold_list)
    f1 = max(compute_f1(pred, g) for g in gold_list)
    return em, f1


def parse_args():
    p = argparse.ArgumentParser(description="ViCoQA multi-turn eval inference (no-unsloth subprocess)")
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--dialogs-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-dialogs", type=int, default=0, help="0 = all dialogs")
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    if "unsloth" in sys.modules:
        raise RuntimeError(
            "unsloth already imported in this process — subprocess isolation broken."
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
                f"PEFT {__import__('peft').__version__} missing {_required_cfg}: {e}. "
                "Install peft>=0.19."
            ) from e

    dialogs = json.load(open(args.dialogs_json, encoding="utf-8"))
    if args.max_dialogs and args.max_dialogs > 0:
        dialogs = dialogs[: args.max_dialogs]

    total_turns = sum(len(d["questions"]) for d in dialogs)
    print(
        f"[Sub] base={base_model} | adapter={adapter_dir} | peft_type={peft_type} | "
        f"dialogs={len(dialogs)} | turns={total_turns}",
        flush=True,
    )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Sub] Loading base (dtype={dtype}, device={device})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False).to(device)
    if not getattr(model, "peft_config", None):
        raise RuntimeError("Adapter not attached.")
    print(f"[Sub] Adapter OK: {list(model.peft_config.keys())}", flush=True)

    model.config.use_cache = True
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.generation_config.max_length = None
    model.generation_config.max_new_tokens = args.max_new_tokens
    for _k in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, _k):
            setattr(model.generation_config, _k, None)

    preds = []
    t0 = time.time()
    turn_count = 0

    for di, dialog in enumerate(dialogs, 1):
        story = dialog["story"]
        history: list[tuple[str, str]] = []
        n_turns = len(dialog["questions"])

        for ti in range(n_turns):
            turn_count += 1
            question = dialog["questions"][ti]["input_text"]
            turn_id = dialog["questions"][ti]["turn_id"]
            msgs = build_messages(story, history, question)
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_length,
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
            pred = clean_prediction(raw)
            gold_refs = gold_refs_for_turn(dialog, ti)
            em, f1 = score_turn(pred, gold_refs)
            primary_gold = dialog["answers"][ti]["input_text"].strip()

            preds.append({
                "dialog_id": dialog.get("id", ""),
                "turn_id": turn_id,
                "turn_idx": ti,
                "question": question,
                "ground_truth": primary_gold,
                "gold_refs": gold_refs,
                "prediction_raw": raw.strip(),
                "prediction": pred,
                "em": em,
                "f1": f1,
            })
            history.append((question, pred))

            if turn_count == 1 or turn_count % args.log_every == 0 or turn_count == total_turns:
                el = time.time() - t0
                rate = turn_count / max(el, 1e-3)
                eta = (total_turns - turn_count) / max(rate, 1e-3)
                print(
                    f"[Sub] turn {turn_count}/{total_turns} | dialog {di}/{len(dialogs)} | "
                    f"{el/60:.1f}m | ETA {eta/60:.1f}m",
                    flush=True,
                )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)

    avg_em = 100 * sum(p["em"] for p in preds) / max(len(preds), 1)
    avg_f1 = 100 * sum(p["f1"] for p in preds) / max(len(preds), 1)
    print(f"[Sub] Saved {len(preds)} predictions → {out_path}", flush=True)
    print(f"[Sub] CoQA EM={avg_em:.2f}% | F1={avg_f1:.2f}%", flush=True)


if __name__ == "__main__":
    main()
