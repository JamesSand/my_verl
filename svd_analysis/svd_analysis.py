#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, re
from pathlib import Path
import numpy as np

# ===== 配置 =====
SRC_ROOT = Path("checkpoints_svd_decompose")    # 输入根（含 U.npy/S.npy/Vh.npy）
DST_ROOT_NAME = "checkpoints_svd_results"       # 输出根目录名（将替换 checkpoints_svd_decompose）
BASE_DIR = Path("/ssd2/zhizhou/workspace/verl/svd_results/models/Qwen2.5-0.5B-Instruct")  # 原始 Qwen2.5 SVD
TOPK = 64
# ===============

def load_svd(d: Path):
    U  = np.load((d / "U.npy").as_posix())
    S  = np.load((d / "S.npy").as_posix())
    Vh = np.load((d / "Vh.npy").as_posix())
    return U, S, Vh

def principal_angles(U1, U2):
    M = U1.T @ U2
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    s = np.clip(s, 0.0, 1.0)
    theta = np.arccos(s)
    return theta, theta * 180/np.pi

def row_norms(Uk: np.ndarray) -> np.ndarray:
    return np.linalg.norm(Uk, axis=1)

def colwise_cosine_U(U1: np.ndarray, U2: np.ndarray, use_abs: bool=True) -> np.ndarray:
    k = min(U1.shape[1], U2.shape[1])
    A = U1[:, :k]
    B = U2[:, :k]
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    cos = np.sum(A * B, axis=0)
    return np.abs(cos) if use_abs else cos

def summarize_change(x: np.ndarray, y: np.ndarray):
    delta = y - x
    ad = np.abs(delta)
    return {
        "abs_min": float(ad.min()),
        "abs_mean": float(ad.mean()),
        "abs_max": float(ad.max()),
        "signed_mean": float(delta.mean())
    }

def find_all_svd_dirs(root: Path):
    """找到所有包含 U/S/Vh 的叶子目录。"""
    out = []
    for d in root.rglob("*"):
        if d.is_dir() and all((d/f).is_file() for f in ("U.npy","S.npy","Vh.npy")):
            out.append(d)
    return sorted(out)

def is_lora(p: Path) -> bool:
    return re.search(r"lora", p.as_posix(), flags=re.IGNORECASE) is not None

def lora_to_pissa(p: Path) -> Path:
    return Path(re.sub(r"lora", "pissa", p.as_posix(), flags=re.IGNORECASE))

def make_out_dir(lora_dir: Path, pissa_dir: Path) -> Path:
    """
    输出路径规则：
      1) 用 lora/pissa 的共同前缀，将其中的 'checkpoints_svd_decompose' 替换成 'checkpoints_svd_results' 作为输出根；
         若无该段，则在共同前缀旁建 'checkpoints_svd_results'。
      2) 保留共同前缀之后到 'global_step_100/<weight_dir>' 为止的相对路径；
      3) 最后加子目录 'PAIR__<lora_token>__VS__BASE__AND__<pissa_token>__VS__BASE'
    """
    common = Path(os.path.commonpath([lora_dir.as_posix(), pissa_dir.as_posix()]))
    parts = list(common.parts)
    if "checkpoints_svd_decompose" in parts:
        i = parts.index("checkpoints_svd_decompose")
        out_root = Path(*parts[:i]) / DST_ROOT_NAME
    else:
        out_root = common.parent / DST_ROOT_NAME

    # 取到 global_step_100/weight_dir 这一层的相对路径
    def rel_until_weight(d: Path) -> Path:
        rel = d.relative_to(common)
        segs = list(rel.parts)
        if "global_step_100" in segs:
            j = segs.index("global_step_100")
            # 追加 weight 子目录名（最后一段）
            return Path(*segs[:j+1]) / d.name
        return rel / d.name

    base_rel = rel_until_weight(lora_dir)

    # token（如 qwen2.5-0.5b-lora32-lr1e-3）
    def token_from(p: Path) -> str:
        for seg in p.parts:
            if re.search(r"(lora|pissa)", seg, flags=re.IGNORECASE):
                return seg
        return p.parts[-2] if len(p.parts)>=2 else p.name

    ltok = token_from(lora_dir)
    ptok = token_from(pissa_dir)

    pair_leaf = Path(f"PAIR__{ltok}__VS__BASE__AND__{ptok}__VS__BASE")
    return out_root / base_rel / pair_leaf

def analyze_vs_base(model_dir: Path, base_dir: Path, topk: int):
    U1, S1, Vh1 = load_svd(model_dir)
    U0, S0, Vh0 = load_svd(base_dir)

    k1 = min(U1.shape[1], S1.shape[0], Vh1.shape[0])
    k0 = min(U0.shape[1], S0.shape[0], Vh0.shape[0])
    k  = int(min(topk, k1, k0))

    Uk1, Uk0 = U1[:, :k], U0[:, :k]
    e1, e0 = (S1[:k]**2), (S0[:k]**2)

    rel_change = (e1 - e0) / (e0 + 1e-12)
    _, deg = principal_angles(Uk1, Uk0)
    r1 = row_norms(Uk1)
    r0 = row_norms(Uk0)
    rn_stats = summarize_change(r0, r1)
    cos_cols = colwise_cosine_U(Uk1, Uk0, use_abs=True)
    
    num_to_keep = 15

    return {
        "topk_used": k,
        "eigenvalues": {
            "model": e1.tolist()[:num_to_keep],
            "base":  e0.tolist()[:num_to_keep],
            "rel_change_vs_base": rel_change.tolist()[:num_to_keep],
            "rel_change_stats": {
                "min": float(rel_change.min()),
                "mean": float(rel_change.mean()),
                "max": float(rel_change.max()),
            }
        },
        "principal_angles_deg": {
            "min": float(deg.min()),
            "mean": float(deg.mean()),
            "max": float(deg.max()),
        },
        "row_norms_change_model_minus_base": rn_stats,
        "colwise_cosine_U_abs": {
            "min": float(cos_cols.min()),
            "mean": float(cos_cols.mean()),
            "max": float(cos_cols.max()),
        },
        "shapes": {
            "model": {"U": list(U1.shape), "S": list(S1.shape), "Vh": list(Vh1.shape)},
            "base":  {"U": list(U0.shape), "S": list(S0.shape), "Vh": list(Vh0.shape)},
        },
    }

def main():
    # 基线检查
    if not all((BASE_DIR / f).is_file() for f in ("U.npy","S.npy","Vh.npy")):
        raise FileNotFoundError(f"Base SVD not found: {BASE_DIR}")

    all_dirs = find_all_svd_dirs(SRC_ROOT)
    lora_dirs = [d for d in all_dirs if is_lora(d)]

    total_pairs, done_pairs, skipped_pairs = 0, 0, 0

    for ldir in lora_dirs:
        pdir = lora_to_pissa(ldir)
        if not (pdir.is_dir() and all((pdir/f).is_file() for f in ("U.npy","S.npy","Vh.npy"))):
            continue  # 没有对应 pissa 就跳过
        total_pairs += 1

        out_dir = make_out_dir(ldir, pdir)
        out_file = out_dir / "summary.json"
        
        # if out_file.exists():
        #     skipped_pairs += 1
        #     continue

        out_dir.mkdir(parents=True, exist_ok=True)

        # 分别与 base 对比
        lora_vs_base  = analyze_vs_base(ldir, BASE_DIR, TOPK)
        pissa_vs_base = analyze_vs_base(pdir, BASE_DIR, TOPK)

        summary = {
            "paths": {
                "lora": ldir.as_posix(),
                "pissa": pdir.as_posix(),
                "base": BASE_DIR.as_posix(),
            },
            "lora_vs_base": lora_vs_base,
            "pissa_vs_base": pissa_vs_base,
            "topk_config": TOPK,
        }

        with open(out_file.as_posix(), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        done_pairs += 1

    print("完成：")
    print(f"  可配对总数 : {total_pairs}")
    print(f"  新生成结果 : {done_pairs}")
    print(f"  已存在跳过 : {skipped_pairs}")

if __name__ == "__main__":
    main()
