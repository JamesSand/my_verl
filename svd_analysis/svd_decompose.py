



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

SRC_ROOT = Path("checkpoints_lora_merged")     # 读取这里
DST_ROOT = Path("checkpoints_svd_decompose")   # 保存到这里

# 指定要分解的权重名；若为 None 则自动选择第一个二维矩阵参数
WEIGHT_NAME = "model.layers.23.mlp.down_proj.weight"  # 改成你需要的；或设为 None

def sanitize(name: str) -> str:
    # 用于作为子目录/文件名
    return name.replace(".", "_").replace("/", "_")

@torch.no_grad()
def pick_weight(model, name=None):
    """返回 (param_name, tensor[float32, on CPU])；如果 name=None 则取第一个二维矩阵。"""
    if name is not None:
        for n, p in model.named_parameters():
            if n == name and p.ndim == 2:
                return n, p.detach().float().cpu()
        raise ValueError(f"Param {name} not found or not 2D.")
    for n, p in model.named_parameters():
        if p.ndim == 2:
            return n, p.detach().float().cpu()
    raise RuntimeError("No 2D weight found.")

def collect_model_dirs(root: Path):
    """收集所有包含 global_step_100 的目录（即模型权重目录）。"""
    return sorted([p for p in root.rglob("global_step_100") if p.is_dir()])

def main():
    to_process = collect_model_dirs(SRC_ROOT)
    total = len(to_process)
    if total == 0:
        print(f"没有在 {SRC_ROOT} 下找到任何 global_step_100 目录，退出。")
        return

    merged = 0
    skipped = 0
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"共 {total} 个模型目录。输出根目录：{DST_ROOT.resolve()}")

    for model_dir in tqdm(to_process, desc="SVD decompose", unit="model"):
        # 目标路径：镜像结构 + 参数名子目录
        rel_path = model_dir.relative_to(SRC_ROOT)
        param_subdir = sanitize(WEIGHT_NAME) if WEIGHT_NAME else "auto_first_2D"
        target_dir = (DST_ROOT / rel_path / param_subdir)

        # 目标已存在 -> 跳过
        if target_dir.exists():
            skipped += 1
            continue

        target_dir.mkdir(parents=True, exist_ok=True)

        # 加载模型（仅用于取权重），尽量通用
        model = AutoModelForCausalLM.from_pretrained(
            model_dir.as_posix()
        ).eval()

        # 取权重并做 SVD
        name, W = pick_weight(model, WEIGHT_NAME)
        print(f"[INFO] Selected: {name}, shape={tuple(W.shape)}")

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # 保存
        np.save((target_dir / "U.npy").as_posix(), U.numpy())
        np.save((target_dir / "S.npy").as_posix(), S.numpy())
        np.save((target_dir / "Vh.npy").as_posix(), Vh.numpy())

        merged += 1

    print("\n完成：")
    print(f"  总计   : {total}")
    print(f"  已分解 : {merged}")
    print(f"  已跳过 : {skipped}")
    print(f"输出根目录：{DST_ROOT.resolve()}")

if __name__ == "__main__":
    main()













# import torch, numpy as np
# from transformers import AutoModelForCausalLM
# import os

# # MERGED_DIR = "/ssd2/zhizhou/workspace/verl/checkpoints_merged/grpo-gsm8k/qwen2.5-0.5b-lora32-lr3e-5/global_step_100-lora_merged"   # ← 改成你的合并后模型目录

# # MERGED_DIR = "/ssd2/zhizhou/workspace/verl/checkpoints_merged/grpo-gsm8k/qwen2.5-0.5b-pissa32-lr3e-5/global_step_100-lora_merged"   # ← 改成你的合并后模型目录

# MERGED_DIR="/ssd2/zhizhou/workspace/verl/models/Qwen2.5-0.5B-Instruct"

# # extract specific weight from the model
# WEIGHT_NAME = "model.layers.23.mlp.down_proj.weight"             


# def print_param_sizes(model, only_trainable=False):
#     """
#     打印所有权重名称、shape、参数量；only_trainable=True 时只看 requires_grad 的参数
#     """
#     total = 0
#     rows = []
#     for name, p in model.named_parameters():
#         if only_trainable and not p.requires_grad:
#             continue
#         shape = tuple(p.shape)
#         n = p.numel()
#         total += n
#         rows.append((name, shape, n, str(p.dtype)))

#     # 按参数量从大到小排个序更直观
#     rows.sort(key=lambda x: x[2], reverse=True)

#     # 漂亮一点的打印
#     name_w = max(len(r[0]) for r in rows) if rows else 10
#     print(f"{'name'.ljust(name_w)}  {'shape':>20}  {'#params':>12}  {'dtype':>12}")
#     print("-" * (name_w + 50))
#     for name, shape, n, dtype in rows:
#         print(f"{name.ljust(name_w)}  {str(shape):>20}  {n:12,d}  {dtype:>12}")
#     print("-" * (name_w + 50))
#     print(f"Total parameters: {total:,}")



# @torch.no_grad()
# def pick_weight(m, name=None):
#     if name is not None:
#         for n, p in m.named_parameters():
#             if n == name and p.ndim == 2:
#                 return n, p.detach().float().cpu()
#         raise ValueError(f"Param {name} not found or not 2D.")
#     # 自动取第一个二维矩阵
#     for n, p in m.named_parameters():
#         if p.ndim == 2:
#             return n, p.detach().float().cpu()
#     raise RuntimeError("No 2D weight found.")

# def main():
    
#     # save to folder
#     svd_dir = "svd_results"
#     # svd dir 后边加上 Merge dir 的最后两个目录名
#     svd_dir = os.path.join(svd_dir, *MERGED_DIR.strip("/").split("/")[-2:])
    
#     model = AutoModelForCausalLM.from_pretrained(MERGED_DIR, trust_remote_code=True).eval()
    
#     # print_param_sizes(model, only_trainable=False)
#     # breakpoint()
    
#     name, W = pick_weight(model, WEIGHT_NAME)
#     print(f"[INFO] Selected: {name}, shape={tuple(W.shape)}")

#     # SVD: W = U @ diag(S) @ Vh
#     U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
#     os.makedirs(svd_dir, exist_ok=True)
    
#     # save U, S, Vh to npy
#     np.save(os.path.join(svd_dir, "U.npy"), U.numpy())
#     np.save(os.path.join(svd_dir, "S.npy"), S.numpy())
#     np.save(os.path.join(svd_dir, "Vh.npy"), Vh.numpy())
    
#     print(f"[INFO] SVD results saved to {svd_dir}")
#     # print dtype and shape
#     print(f"U: dtype={U.dtype}, shape={U.shape}")
#     print(f"S: dtype={S.dtype}, shape={S.shape}")
#     print(f"Vh: dtype={Vh.dtype}, shape={Vh.shape}")
    
    

    

# if __name__ == "__main__":
#     main()



