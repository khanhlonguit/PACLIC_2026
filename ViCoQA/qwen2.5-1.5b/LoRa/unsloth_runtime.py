"""Unsloth runtime setup — gcc check, env flags, Triton fallback."""
from __future__ import annotations

import os
import shutil
import sys


def has_c_compiler() -> bool:
    return bool(shutil.which("gcc") or shutil.which("cc") or shutil.which("g++"))


def setup_unsloth_env() -> bool:
    """Gọi TRƯỚC `import unsloth`. Trả về True nếu có gcc (Triton OK)."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

    cc = shutil.which("gcc") or shutil.which("cc")
    if cc:
        os.environ.setdefault("CC", cc)
        print(f"[unsloth_runtime] gcc OK: {cc}", flush=True)
        return True

    # Container thiếu build-essential → tắt custom Triton kernels
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
    os.environ["UNSLOTH_DISABLE_CUSTOM_KERNELS"] = "1"
    print(
        "[unsloth_runtime] WARNING: no gcc/cc — UNSLOTH_COMPILE_DISABLE=1, "
        "will patch RMSNorm fallback",
        flush=True,
    )
    print(
        "  Fix lâu dài: sudo apt-get update && sudo apt-get install -y build-essential",
        flush=True,
    )
    return False


def patch_rms_layernorm_fallback():
    """PyTorch RMSNorm thay Triton khi không có C compiler."""
    try:
        import torch
        import unsloth.kernels.rms_layernorm as rms_mod

        def _pytorch_rms_layernorm(norm, hidden_states):
            input_dtype = hidden_states.dtype
            hs = hidden_states.to(torch.float32)
            eps = getattr(norm, "variance_epsilon", 1e-6)
            variance = hs.pow(2).mean(-1, keepdim=True)
            hs = hs * torch.rsqrt(variance + eps)
            out = norm.weight * hs
            return out.to(input_dtype)

        rms_mod.fast_rms_layernorm = _pytorch_rms_layernorm
        print("[unsloth_runtime] Patched fast_rms_layernorm → PyTorch (no Triton)", flush=True)
        return True
    except Exception as e:
        print(f"[unsloth_runtime] RMSNorm fallback patch failed: {e}", flush=True)
        return False


def import_unsloth_safe():
    """Import unsloth đúng thứ tự (trước transformers nếu có thể)."""
    has_gcc = setup_unsloth_env()
    if "transformers" in sys.modules:
        print(
            "[unsloth_runtime] WARN: transformers đã import trước unsloth — có thể chậm hơn",
            flush=True,
        )
    import unsloth  # noqa: F401
    if not has_gcc:
        patch_rms_layernorm_fallback()
    return has_gcc


def gradient_checkpointing_mode(has_gcc: bool | None = None) -> bool | str:
    """unsloth GC cần Triton; fallback True (HF standard) khi không có gcc."""
    if has_gcc is None:
        has_gcc = has_c_compiler()
    return "unsloth" if has_gcc else True
