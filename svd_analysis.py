# analyze_svd_three.py
import os, numpy as np
from itertools import combinations

# ===== 修改为你的三个 SVD 目录（每个目录里有 U.npy / S.npy / Vh.npy）=====
DIRS = [
    "/ssd2/zhizhou/workspace/verl/svd_results/models/Qwen2.5-0.5B-Instruct",
    "/ssd2/zhizhou/workspace/verl/svd_results/qwen2.5-0.5b-lora32-lr3e-5/global_step_100-lora_merged",
    "/ssd2/zhizhou/workspace/verl/svd_results/qwen2.5-0.5b-pissa32-lr3e-5/global_step_100-lora_merged",
]
TOPK = 64   # 统一比较的前 K 个奇异分量；会自动截到三个模型的共同最小值
# ======================================================================

def load_svd(d):
    U = np.load(os.path.join(d, "U.npy"))
    S = np.load(os.path.join(d, "S.npy"))
    Vh = np.load(os.path.join(d, "Vh.npy"))
    return U, S, Vh

def principal_angles(U1, U2):
    # 用各自的 U_k（列正交）比较列子空间：SVD(U1^T U2) 的奇异值 = cos(theta)
    M = U1.T @ U2
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    s = np.clip(s, 0.0, 1.0)
    theta = np.arccos(s)            # 弧度
    return theta, theta * 180/np.pi # (radians, degrees)

def row_norms(Uk: np.ndarray) -> np.ndarray:
    # 每个原始维度在 Top-K 子空间中的能量
    return np.linalg.norm(Uk, axis=1)

def summarize_change(x: np.ndarray, y: np.ndarray):
    # 统计 y - x 的变化
    delta = y - x
    ad = np.abs(delta)
    return {
        "abs_min": float(ad.min()),
        "abs_mean": float(ad.mean()),
        "abs_max": float(ad.max()),
        "signed_mean": float(delta.mean())
    }

def colwise_cosine_U(U1: np.ndarray, U2: np.ndarray, use_abs: bool = True) -> np.ndarray:
    """
    对齐列索引：比较 U1[:,t] 与 U2[:,t] 的余弦相似度，t=0..k-1。
    默认对结果取绝对值以消除 SVD 列向量的符号不定性。
    """
    k = min(U1.shape[1], U2.shape[1])
    A = U1[:, :k]
    B = U2[:, :k]
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    cos = np.sum(A * B, axis=0)  # 每一列的点积
    return np.abs(cos) if use_abs else cos

def main():
    svds = [load_svd(d) for d in DIRS]
    # 统一 k：受限于 TOPK 和三个分解的列数
    ks = [min(U.shape[1], S.shape[0], Vh.shape[0]) for (U,S,Vh) in svds]
    k = min(TOPK, *ks)
    print("[INFO] use top-k =", int(k))

    # 1) Top-K eigen values (S^2) —— 打印完整数组
    evals = []
    for i, (U,S,Vh) in enumerate(svds):
        e = (S[:k] ** 2)
        evals.append(e)
        print("\n=== Model", i, "===", DIRS[i])
        print("Top-K eigen values (length =", len(e), "):")
        print(e.tolist())

    print()
    # 2) Eigenvalue change stats（相对 Model 0 的 min / mean / max）
    base = evals[0]
    for j in [1, 2]:
        rel_change = (evals[j] - base) / (base + 1e-12)  # 相对变化
        print("[Eigen change vs. Model 0] Model", j, "/ Model 0 (length =", len(rel_change), "):")
        print(rel_change.tolist())
        
    print()
        
    for j in [1, 2]:
        rel_change = (evals[j] - base) / (base + 1e-12)  # 相对变化
        print("[Eigen change stats vs. Model 0] Model", j, "/ Model 0:",
              "min =", float(rel_change.min()),
              ", mean =", float(rel_change.mean()),
              ", max =", float(rel_change.max()))

    # 3) Principal angles（用 U_k 比较列子空间）—— 报告 min/max/mean
    print("\n=== Principal angles (degrees) on top-K column subspaces ===")
    Uk = [U[:, :k] for (U,_,_) in svds]
    for i, j in combinations(range(3), 2):
        _, deg = principal_angles(Uk[i], Uk[j])
        print("(", i, ",", j, "): min =", float(deg.min()), ", mean =", float(deg.mean()), ", max =", float(deg.max()))

    # 4) U-space row norms（每个原始维度在 Top-K 子空间里的能量）
    print("\n=== U-space row-norms (||U_k[i,:]||_2) and pairwise changes ===")
    R = [row_norms(U[:, :k]) for (U,_,_) in svds]

    # 4.1 各模型自身的分布概览
    for i, r in enumerate(R):
        print("Model", i, "row-norms: min =", float(r.min()), ", mean =", float(r.mean()), ", max =", float(r.max()))

    # 4.2 与其他模型对比（|Δ| 的 min/mean/max + 有符号均值）
    for i, j in combinations(range(3), 2):
        stats = summarize_change(R[i], R[j])
        print("(", i, ",", j, ") |Δrow-norm|: min =", stats["abs_min"], ", mean =", stats["abs_mean"], ", max =", stats["abs_max"], ", signed_mean =", stats["signed_mean"])

    # 5) U-space 逐列余弦相似度（对应列）—— 报告 min/max/mean
    print("\n=== Column-wise cosine similarity on U-space (abs) ===")
    for i, j in combinations(range(3), 2):
        cos_cols = colwise_cosine_U(Uk[i], Uk[j], use_abs=True)
        print("(", i, ",", j, "): min =", float(cos_cols.min()), ", mean =", float(cos_cols.mean()), ", max =", float(cos_cols.max()))
        # 如需完整逐列列表，请取消注释下一行
        # print("cos per column:", cos_cols.tolist())

if __name__ == "__main__":
    main()
