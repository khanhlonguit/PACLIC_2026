#!/usr/bin/env python3
"""
Chạy trên máy REMOTE sau khi train xong.
Nén folder model thành 1 file .zip để tải về local.

Usage (trên remote, trong thư mục chứa model):
    python compress_model_folder.py
    python compress_model_folder.py --source qwen2.5-3b-instruct-fullft-vinewsqa
"""

from __future__ import annotations

import argparse
import os
import time
import zipfile
from pathlib import Path


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def folder_size(path: Path) -> int:
    return sum(
        (Path(root) / name).stat().st_size
        for root, _dirs, files in os.walk(path)
        for name in files
    )


def list_files(source: Path) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for root, _dirs, files in os.walk(source):
        for name in files:
            abs_path = Path(root) / name
            arcname = str(abs_path.relative_to(source.parent))
            items.append((abs_path, arcname))
    return items


def compress_zip(source: Path, output: Path, compresslevel: int = 1) -> None:
    files = list_files(source)
    if not files:
        raise FileNotFoundError(f"Folder rỗng hoặc không có file: {source}")

    total = sum(p.stat().st_size for p, _ in files)
    done = 0
    t0 = time.time()

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compresslevel,
    ) as zf:
        for i, (abs_path, arcname) in enumerate(files, 1):
            zf.write(abs_path, arcname=arcname)
            done += abs_path.stat().st_size
            print(
                f"\r[{i}/{len(files)}] {100 * done / total:5.1f}% | "
                f"{human_size(done)}/{human_size(total)} | {Path(arcname).name[:40]}",
                end="",
                flush=True,
            )

    elapsed = time.time() - t0
    print()
    print(f"Hoàn tất trong {elapsed:.1f}s")
    print(f"Folder gốc : {human_size(total)}  ->  {source}")
    print(f"File zip   : {human_size(output.stat().st_size)}  ->  {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nén folder model FullFT trên remote thành .zip để tải về."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("qwen2.5-3b-instruct-fullft-vinewsqa"),
        help="Folder model trên remote",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="File .zip đầu ra (default: <source>.zip)",
    )
    parser.add_argument(
        "--compresslevel",
        type=int,
        default=1,
        choices=range(0, 10),
        help="Mức nén 0-9. 1=nhanh (khuyến nghị cho safetensors).",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_dir():
        raise SystemExit(f"Không tìm thấy folder: {source}")

    output = (args.output or source.with_suffix(".zip")).resolve()

    print("=" * 60)
    print("Nén model folder -> ZIP (remote)")
    print("=" * 60)
    print(f"Source : {source}")
    print(f"Output : {output}")
    print(f"Size   : {human_size(folder_size(source))} (chưa nén)")
    print(f"Level  : zip {args.compresslevel}")
    print("-" * 60)

    compress_zip(source, output, compresslevel=args.compresslevel)

    print("-" * 60)
    print("Tải về local:")
    print(f"  scp user@remote:{output} .")
    print(f"  rsync -avP user@remote:{output} .")
    print()
    print("Giải nén:")
    print(f"  unzip {output.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
