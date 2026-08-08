# PASTE & RUN THIS CELL before train (sau Restart Kernel tốt nhất)
# Fix OOM: clear VRAM + bật 4-bit nếu GPU gần đầy (Qwen2.5-3B)

import gc, os, torch

def clear_gpu(verbose=True):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        if verbose:
            free, total = torch.cuda.mem_get_info()
            print(f"[VRAM] free={free/1024**3:.2f}/{total/1024**3:.2f} GiB")

clear_gpu()

# Nếu free < 10 GiB → bắt buộc 4-bit (3B cần nhiều VRAM hơn 1.5B)
if torch.cuda.is_available():
    free, _ = torch.cuda.mem_get_info()
    if free / 1024**3 < 10:
        LOAD_IN_4BIT = True
        print("→ GPU gần đầy: LOAD_IN_4BIT = True")
    else:
        print("→ VRAM đủ; giữ LOAD_IN_4BIT =", globals().get("LOAD_IN_4BIT", False))

print("Nếu free vẫn < 6 GiB: Kernel → Restart Kernel, rồi chạy lại từ đầu.")
print("Check process khác: !nvidia-smi")
