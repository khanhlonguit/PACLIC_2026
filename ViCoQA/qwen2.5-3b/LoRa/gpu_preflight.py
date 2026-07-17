#!/usr/bin/env python3
"""Kiểm tra VRAM trước train Qwen2.5-3B — chạy độc lập: python gpu_preflight.py"""
import gc
import subprocess
import sys

import torch

MIN_FREE_GIB = 10.0


def main():
    print("=== GPU Preflight (Qwen2.5-3B) ===")
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    free, total = torch.cuda.mem_get_info()
    free_gib = free / 1024**3
    total_gib = total / 1024**3
    pt_alloc = torch.cuda.memory_allocated() / 1024**3

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"driver_used={total_gib - free_gib:.2f} GiB | free={free_gib:.2f}/{total_gib:.2f} GiB")
    print(f"pytorch_alloc={pt_alloc:.3f} GiB")

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
        used_mb, total_mb = [int(x.strip()) for x in out.split(",")]
        print(f"nvidia-smi: used={used_mb} MiB / total={total_mb} MiB")
    except Exception as e:
        print(f"nvidia-smi skip: {e}")

    if free_gib < MIN_FREE_GIB:
        print(f"\n❌ FAIL: free={free_gib:.2f} GiB < {MIN_FREE_GIB} GiB")
        if pt_alloc < 1.0 and (total_gib - free_gib) > 10:
            print("→ Zombie VRAM: PyTorch alloc nhỏ nhưng driver used lớn.")
            print("  Restart kernel/container/reboot. clear_gpu() không fix được.")
        sys.exit(2)

    print(f"\n✅ OK: free={free_gib:.2f} GiB — có thể train Qwen2.5-3B 4bit")
    sys.exit(0)


if __name__ == "__main__":
    main()
