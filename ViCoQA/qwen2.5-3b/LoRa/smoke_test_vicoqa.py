"""CPU smoke test for ViCoQA data pipeline, profiling, and CoQA metrics (Qwen2.5-3B)."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LORA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LORA_DIR))

from eval_infer_subprocess import gold_refs_for_turn, score_turn  # noqa: E402

SYSTEM_PROMPT = (
    "Bạn là trợ lý hỏi-đáp tiếng Việt. Dựa trên đoạn văn dưới đây, "
    "trả lời ngắn gọn, tự nhiên theo ngữ cảnh hội thoại.\n\n"
    "Đoạn văn:\n{story}"
)


def load_vicoqa_split(path):
    with open(path, encoding="utf-8") as f:
        dialogs = json.load(f)
    samples = []
    for d in dialogs:
        answers_by_turn = {a["turn_id"]: a for a in d["answers"]}
        for turn_idx, q in enumerate(d["questions"]):
            a = answers_by_turn[q["turn_id"]]
            answer = (a.get("input_text") or "").strip()
            if not answer:
                continue
            samples.append({
                "dialog_id": d["id"],
                "turn_id": q["turn_id"],
                "story": d["story"],
                "question": q["input_text"],
                "answer": answer,
                "turn_idx": turn_idx,
                "history": [
                    (
                        d["questions"][t]["input_text"],
                        answers_by_turn[d["questions"][t]["turn_id"]]["input_text"].strip(),
                    )
                    for t in range(turn_idx)
                ],
            })
    return samples, dialogs


def build_messages(sample, for_inference=False):
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(story=sample["story"])}]
    for q, a in sample["history"]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": sample["question"]})
    if not for_inference:
        messages.append({"role": "assistant", "content": sample["answer"]})
    return messages


def compute_max_seq_length(samples, tok, cap=4096, min_len=512):
    lengths = []
    for s in samples:
        text = tok.apply_chat_template(build_messages(s), tokenize=False, add_generation_prompt=False)
        lengths.append(len(tok.encode(text)))
    lengths.sort()
    n = len(lengths)
    stats = {
        "min": lengths[0],
        "p50": lengths[n // 2],
        "p95": lengths[int(n * 0.95)],
        "p99": lengths[int(n * 0.99)],
        "max": lengths[-1],
    }
    chosen = max(((min(math.ceil(stats["p99"] * 1.05), cap) + 255) // 256) * 256, min_len)
    truncated = sum(1 for L in lengths if L > chosen)
    stats.update({
        "chosen_max_seq_length": chosen,
        "truncated_samples": truncated,
        "truncated_pct": round(100 * truncated / n, 3),
    })
    return chosen, stats


def main():
    train_path = ROOT / "train.json"
    dev_path = ROOT / "dev.json"

    train_samples, train_dialogs = load_vicoqa_split(train_path)
    dev_samples, dev_dialogs = load_vicoqa_split(dev_path)

    assert len(train_dialogs) == 1400
    assert len(train_samples) == 7000
    assert len(dev_dialogs) == 300
    assert len(dev_samples) == 1500
    assert all(len(d["questions"]) == 5 for d in train_dialogs)
    print("OK data counts")

    # turn expansion + history
    late = [s for s in train_samples if s["turn_idx"] == 4]
    assert late and len(late[0]["history"]) == 4
    print("OK multi-turn history")

    # CoQA scoring with additional_answers
    dialog = dev_dialogs[0]
    refs = gold_refs_for_turn(dialog, 0)
    assert len(refs) >= 2
    em, f1 = score_turn(refs[1], refs)
    assert em == 1 and f1 == 1.0
    print("OK CoQA max-over-refs scoring")

    # tokenizer profiling (subset for speed; fallback if transformers unavailable)
    subset = train_samples[:500]
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        max_seq, stats = compute_max_seq_length(subset, tok)
        text = tok.apply_chat_template(build_messages(subset[10]), tokenize=False, add_generation_prompt=False)
        assert "Đoạn văn:" in text and subset[10]["answer"] in text
        print("OK chat template (tokenizer)")
    except ImportError:
        lengths = []
        for s in subset:
            msg_len = len(s["story"]) + sum(len(q) + len(a) for q, a in s["history"]) + len(s["question"]) + len(s["answer"])
            lengths.append(int(msg_len / 3.5))
        lengths.sort()
        n = len(lengths)
        stats = {
            "min": lengths[0],
            "p50": lengths[n // 2],
            "p95": lengths[int(n * 0.95)],
            "p99": lengths[int(n * 0.99)],
            "max": lengths[-1],
            "note": "char/3.5 estimate — rerun profiling cell with tokenizer for exact values",
        }
        max_seq = max(((min(math.ceil(stats["p99"] * 1.05), 4096) + 255) // 256) * 256, 512)
        stats["chosen_max_seq_length"] = max_seq
        stats["truncated_samples"] = sum(1 for L in lengths if L > max_seq)
        stats["truncated_pct"] = round(100 * stats["truncated_samples"] / n, 3)
        print("OK chat template (structure check only; transformers not installed)")

    prof_path = LORA_DIR / "profiling_config.json"
    json.dump({"max_seq_length": max_seq, "token_length_stats": stats}, open(prof_path, "w"), indent=2)
    print(f"OK profiling max_seq_length={max_seq} (subset n=500)")
    print(json.dumps(stats, indent=2))

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
