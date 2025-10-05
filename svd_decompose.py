# svd_simple.py
import torch, numpy as np
from transformers import AutoModelForCausalLM
import os

# MERGED_DIR = "/ssd2/zhizhou/workspace/verl/checkpoints_merged/grpo-gsm8k/qwen2.5-0.5b-lora32-lr3e-5/global_step_100-lora_merged"   # ← 改成你的合并后模型目录

# MERGED_DIR = "/ssd2/zhizhou/workspace/verl/checkpoints_merged/grpo-gsm8k/qwen2.5-0.5b-pissa32-lr3e-5/global_step_100-lora_merged"   # ← 改成你的合并后模型目录

MERGED_DIR="/ssd2/zhizhou/workspace/verl/models/Qwen2.5-0.5B-Instruct"


WEIGHT_NAME = "model.layers.23.mlp.down_proj.weight"              # ← 想指定就填准确名字；不想指定就设为 None
# TOPK = 64                                   # 只保留前 k 个奇异值/向量；设为 None 做全量（可能很慢/占内存）


def print_param_sizes(model, only_trainable=False):
    """
    打印所有权重名称、shape、参数量；only_trainable=True 时只看 requires_grad 的参数
    """
    total = 0
    rows = []
    for name, p in model.named_parameters():
        if only_trainable and not p.requires_grad:
            continue
        shape = tuple(p.shape)
        n = p.numel()
        total += n
        rows.append((name, shape, n, str(p.dtype)))

    # 按参数量从大到小排个序更直观
    rows.sort(key=lambda x: x[2], reverse=True)

    # 漂亮一点的打印
    name_w = max(len(r[0]) for r in rows) if rows else 10
    print(f"{'name'.ljust(name_w)}  {'shape':>20}  {'#params':>12}  {'dtype':>12}")
    print("-" * (name_w + 50))
    for name, shape, n, dtype in rows:
        print(f"{name.ljust(name_w)}  {str(shape):>20}  {n:12,d}  {dtype:>12}")
    print("-" * (name_w + 50))
    print(f"Total parameters: {total:,}")



@torch.no_grad()
def pick_weight(m, name=None):
    if name is not None:
        for n, p in m.named_parameters():
            if n == name and p.ndim == 2:
                return n, p.detach().float().cpu()
        raise ValueError(f"Param {name} not found or not 2D.")
    # 自动取第一个二维矩阵
    for n, p in m.named_parameters():
        if p.ndim == 2:
            return n, p.detach().float().cpu()
    raise RuntimeError("No 2D weight found.")

def main():
    
    # save to folder
    svd_dir = "svd_results"
    # svd dir 后边加上 Merge dir 的最后两个目录名
    svd_dir = os.path.join(svd_dir, *MERGED_DIR.strip("/").split("/")[-2:])
    
    model = AutoModelForCausalLM.from_pretrained(MERGED_DIR, trust_remote_code=True).eval()
    
    # print_param_sizes(model, only_trainable=False)
    # breakpoint()
    
    name, W = pick_weight(model, WEIGHT_NAME)
    print(f"[INFO] Selected: {name}, shape={tuple(W.shape)}")

    # SVD: W = U @ diag(S) @ Vh
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
    os.makedirs(svd_dir, exist_ok=True)
    
    # save U, S, Vh to npy
    np.save(os.path.join(svd_dir, "U.npy"), U.numpy())
    np.save(os.path.join(svd_dir, "S.npy"), S.numpy())
    np.save(os.path.join(svd_dir, "Vh.npy"), Vh.numpy())
    
    print(f"[INFO] SVD results saved to {svd_dir}")
    # print dtype and shape
    print(f"U: dtype={U.dtype}, shape={U.shape}")
    print(f"S: dtype={S.dtype}, shape={S.shape}")
    print(f"Vh: dtype={Vh.dtype}, shape={Vh.shape}")
    
    

    

if __name__ == "__main__":
    main()



