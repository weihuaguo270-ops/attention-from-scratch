"""
PyTorch 版正弦位置编码 — 与 NumPy 版 positional_encoding.py 对应
"""
import torch


def sinusoidal_positional_encoding(seq_len, d_model):
    """
    正弦位置编码 — 公式与 NumPy 版完全一致

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    参数:
        seq_len: 序列长度
        d_model: 向量维度

    返回:
        (seq_len, d_model) 的位置编码矩阵
    """
    pe = torch.zeros(seq_len, d_model)

    # pos shape: (seq_len, 1) — 位置索引列向量
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)

    # i shape: (d_model,) — 维度索引行向量
    i = torch.arange(0, d_model, dtype=torch.float)

    # 频率系数: 1 / 10000^(2i/d_model) = exp(2i * -ln(10000) / d_model)
    # low i → fast change, high i → slow change
    div_term = torch.exp(i * -torch.tensor(10000.0).log() / d_model)

    # 偶数维用 sin, 奇数维用 cos
    pe[:, 0::2] = torch.sin(pos * div_term[0::2])
    pe[:, 1::2] = torch.cos(pos * div_term[1::2])

    return pe


# ============================================================
# RoPE 旋转位置编码（PyTorch 版）
# ============================================================
def precompute_rope(d_k: int, max_seq_len: int = 128, base: float = 10000.0):
    """
    预计算 RoPE 的 cos/sin 表
    返回:
        cos_table: (max_seq_len, d_k // 2)
        sin_table: (max_seq_len, d_k // 2)
    """
    theta = base ** (-2 * torch.arange(0, d_k, 2, dtype=torch.float32) / d_k)
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    angles = pos[:, None] * theta[None, :]
    return torch.cos(angles), torch.sin(angles)


# ============================================================
# 演示
# ============================================================
if __name__ == "__main__":
    seq_len, d_model = 10, 8
    pe = sinusoidal_positional_encoding(seq_len, d_model)

    print(f"位置编码矩阵 ({seq_len} 个位置, 每个 {d_model} 维):")
    print(pe)
    print()

    # 展示不同维度的周期性
    print("不同维度的周期差异:")
    print(f"  dim 0 (低频): {pe[:6, 0].tolist()} — 变化快，区分相邻位置")
    print(f"  dim 6 (高频): {pe[:6, 6].tolist()} — 变化慢，区分远距离位置")
