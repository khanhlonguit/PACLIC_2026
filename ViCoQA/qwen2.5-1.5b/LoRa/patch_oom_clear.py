"""Patch clear_gpu + pre-load VRAM cleanup for ViCoQA notebook."""
import json
from pathlib import Path

NB = Path(__file__).parent / "train_qwen_lora_unsloth.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))

OLD_CLEAR = '''def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
'''

NEW_CLEAR = '''def clear_gpu(verbose=False):
    """Giải phóng VRAM triệt để trước khi load/train model mới."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    if verbose:
        free, total = torch.cuda.mem_get_info()
        print(
            f"[VRAM] free={free/1024**3:.2f} GiB / total={total/1024**3:.2f} GiB "
            f"| allocated={torch.cuda.memory_allocated()/1024**3:.2f} GiB",
            flush=True,
        )


def assert_vram_for_load(min_free_gib=6.0):
    """Fail sớm nếu GPU gần đầy (thường do kernel/process khác giữ model)."""
    if not torch.cuda.is_available():
        return
    clear_gpu(verbose=True)
    free, total = torch.cuda.mem_get_info()
    free_gib = free / 1024**3
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"GPU chỉ còn {free_gib:.2f} GiB trống (cần ≥ {min_free_gib} GiB).\\n"
            "→ Kernel → Restart Kernel, rồi chạy lại từ đầu.\\n"
            "→ Hoặc tắt process khác đang chiếm GPU: nvidia-smi\\n"
            "→ Tạm thời set LOAD_IN_4BIT = True trong config."
        )
'''

# Patch config cell clear_gpu
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if "def clear_gpu():" in src and "def build_messages" in src:
        src2 = src.replace(OLD_CLEAR, NEW_CLEAR)
        if src2 == src:
            # try without trailing issues
            if "def clear_gpu():\n    gc.collect()" in src:
                src2 = src.replace(
                    "def clear_gpu():\n    gc.collect()\n    if torch.cuda.is_available():\n        torch.cuda.empty_cache()\n        torch.cuda.synchronize()\n",
                    NEW_CLEAR,
                )
        if "assert_vram_for_load" not in src2:
            print("WARN: clear_gpu replace may have failed")
        else:
            print("Patched clear_gpu in config cell")
        c["source"] = [line + "\n" for line in src2.splitlines(True)]
        if not c["source"][-1].endswith("\n"):
            c["source"][-1] += "\n"

    # LOAD_IN_4BIT True as safer default after OOM
    if "LOAD_IN_4BIT = False" in src and "BASE_MODEL_NAME" in src:
        src2 = "".join(c.get("source", [])).replace("LOAD_IN_4BIT = False", "LOAD_IN_4BIT = True  # OOM-safe; set False nếu VRAM trống >12GB")
        c["source"] = [line + "\n" for line in src2.splitlines(True)]
        print("Set LOAD_IN_4BIT = True")

# Patch train_one_variant to call assert_vram_for_load before from_pretrained
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if "def train_one_variant" not in src:
        continue
    if "assert_vram_for_load" in src:
        print("train cell already has assert_vram")
        continue
    needle = "    eval_on = USE_EARLY_STOPPING and eval_dataset is not None\n    clear_gpu()\n"
    repl = (
        "    eval_on = USE_EARLY_STOPPING and eval_dataset is not None\n"
        "    clear_gpu(verbose=True)\n"
        "    assert_vram_for_load(min_free_gib=4.0 if LOAD_IN_4BIT else 8.0)\n"
    )
    # notebook may have double newlines from previous patch
    src2 = src.replace(
        "    eval_on = USE_EARLY_STOPPING and eval_dataset is not None\n\n    clear_gpu()\n\n",
        "    eval_on = USE_EARLY_STOPPING and eval_dataset is not None\n\n    clear_gpu(verbose=True)\n\n    assert_vram_for_load(min_free_gib=4.0 if LOAD_IN_4BIT else 8.0)\n\n",
    )
    if src2 == src:
        src2 = src.replace(needle, repl)
    if src2 == src:
        src2 = src.replace(
            "    clear_gpu()\n\n\n    load_kwargs = dict(",
            "    clear_gpu(verbose=True)\n    assert_vram_for_load(min_free_gib=4.0 if LOAD_IN_4BIT else 8.0)\n\n    load_kwargs = dict(",
        )
    if "assert_vram_for_load" in src2:
        print("Patched train_one_variant pre-load VRAM check")
    else:
        print("FAILED to patch train_one_variant")
    c["source"] = [line + "\n" for line in src2.splitlines(True)]

# Patch end of train_one_variant clear
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if "def train_one_variant" not in src:
        continue
    src2 = src.replace(
        "    del trainer, model, tokenizer\n    clear_gpu()\n    return variant[\"save_path\"]",
        "    del trainer, model, tokenizer\n    clear_gpu(verbose=True)\n    return variant[\"save_path\"]",
    )
    src2 = src2.replace(
        "    del trainer, model, tokenizer\n\n    clear_gpu()\n\n    return variant[\"save_path\"]",
        "    del trainer, model, tokenizer\n\n    clear_gpu(verbose=True)\n\n    return variant[\"save_path\"]",
    )
    c["source"] = [line + "\n" for line in src2.splitlines(True)]
    print("Patched post-train clear_gpu")

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote", NB)
