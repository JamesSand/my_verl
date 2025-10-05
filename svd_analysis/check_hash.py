#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import hashlib

ROOT = Path("checkpoints_lora_merged")
TARGET_NAME = "model.safetensors"  # 如果想要所有 .safetensors，把这一行设为 None

def sha256sum(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()

def main():
    if not ROOT.exists():
        raise SystemExit(f"目录 {ROOT.resolve()} 不存在，先确认路径。")

    safetensors = []
    for file in ROOT.rglob("*.safetensors"):
        if TARGET_NAME and file.name != TARGET_NAME:
            continue
        safetensors.append(file)

    if not safetensors:
        print("没有找到匹配的 safetensors 文件。")
        return

    safetensors.sort()
    for file in safetensors:
        digest = sha256sum(file)
        rel_path = file.relative_to(ROOT.parent)
        print(f"{digest}  {rel_path}")

if __name__ == "__main__":
    main()



