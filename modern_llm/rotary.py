"""
RoPE (Rotary Position Embedding) — 旋转位置编码

现代 LLM（Llama、Mistral、DeepSeek、Gemma）通用的位置编码方案。
与 Sinusoidal PE 的核心区别：
  Sinusoidal PE: 生成位置向量加到输入上（加法）
  RoPE:         旋转 Q/K 向量（乘法），Attention 分数自带位置信息

独立模块，供 modern_llm/ 各模块使用。
"""
import numpy as np


def precompute_rotary_frequencies(d_k: int, max_seq_len: int = 128, base: float = 10000.0):
    """
    预计算旋转角度（RoPE 的 theta 表）
    公式：theta_i = base^(-2i/d_k)
    返回:
        cos_table: (max_seq_len, d_k // 2)
        sin_table: (max_seq_len, d_k // 2)
    """
    theta = base ** (-2 * np.arange(0, d_k, 2) / d_k)
    pos = np.arange(max_seq_len)
    angles = pos[:, None] * theta[None, :]
    return np.cos(angles), np.sin(angles)


def apply_rotary(x: np.ndarray, cos_table: np.ndarray, sin_table: np.ndarray,
                 positions: np.ndarray = None) -> np.ndarray:
    """
    对 Q 或 K 应用 RoPE 旋转
    参数:
        x: (seq_len, d_k) — 一个 head 的 Q 或 K
        cos_table, sin_table: (max_seq_len, d_k // 2)
        positions: 位置索引数组 (seq_len,)，默认 [0, 1, ..., seq_len-1]
    返回:
        (seq_len, d_k) — 旋转后的向量
    """
    seq_len = x.shape[0]
    if positions is None:
        positions = np.arange(seq_len)
    cos_val = cos_table[positions]  # (seq_len, d_k // 2)
    sin_val = sin_table[positions]

    x_even = x[:, 0::2]  # (seq_len, d_k // 2)
    x_odd = x[:, 1::2]

    x_even_rotated = x_even * cos_val - x_odd * sin_val
    x_odd_rotated = x_even * sin_val + x_odd * cos_val

    result = np.empty_like(x)
    result[:, 0::2] = x_even_rotated
    result[:, 1::2] = x_odd_rotated
    return result
