from __future__ import annotations

"""
Standalone eval inference — chạy trong PROCESS RIÊNG.

Mục tiêu:
  - Không import unsloth (tránh monkey patch conflict)
  - Load full fine-tuned model từ --model-dir
  - Batch left-padding generation
  - Output predictions JSON tương thích notebook:
      [
        {
          "id": str,
          "question": str,
          "ground_truth": str,
          "gold_answers": [str, ...],
          "prediction_raw": str,
          "prediction": str
        },
        ...
      ]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


SYSTEM_PROMPT = (
    "Bạn là hệ thống hỏi-đáp trích xuất tiếng Việt. Chỉ trả lời bằng một cụm từ xuất hiện "
    "nguyên văn trong đoạn văn, không giải thích và không thêm tiền tố.\n\n"
    "Đoạn văn:\n{context}"
)

PREFIX_RE = re.compile(r"^(đáp án|answer|câu trả lời)\s*[:\-]?\s*", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--samples-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def clean_prediction(value: str) -> str:
    value = (value or "").strip().split("\n")[0].strip().strip("\"'")
    return PREFIX_RE.sub("", value).strip()


def main():
    # Safety: subprocess phải độc lập kernel.
    if "unsloth" in sys.modules:
        raise RuntimeError("Subprocess isolation failed: unsloth was imported.")

    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[fullft-sub] Loading model: {model_dir} | dtype={dtype} | device={device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=dtype).to(device)
    model.eval()
    model.config.use_cache = True

    samples = json.load(open(args.samples_json, encoding="utf-8"))
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT.format(context=row["context"])},
                {"role": "user", "content": row["question"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in samples
    ]

    order = sorted(range(len(samples)), key=lambda index: len(prompts[index]))
    predictions = [None] * len(samples)

    started = time.time()
    done = 0
    for start in range(0, len(order), args.batch_size):
        indices = order[start : start + args.batch_size]
        encoded = tokenizer(
            [prompts[index] for index in indices],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        for batch_index, sample_index in enumerate(indices):
            row = samples[sample_index]
            raw = tokenizer.decode(new_tokens[batch_index], skip_special_tokens=True)
            predictions[sample_index] = {
                "id": row.get("id", ""),
                "question": row.get("question", ""),
                "ground_truth": row.get("answer"),
                "gold_answers": row.get("gold_answers", [row.get("answer", "")]),
                "prediction_raw": raw,
                "prediction": clean_prediction(raw),
            }

        done += len(indices)
        if done == len(samples) or done % max(args.log_every, args.batch_size) < args.batch_size:
            rate = done / max(time.time() - started, 1e-3)
            print(f"[fullft-sub] [infer] {done}/{len(samples)} | {rate:.2f} samples/s", flush=True)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False)


if __name__ == "__main__":
    main()

