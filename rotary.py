"""
RoPE (Rotary Position Embedding) - 旋转位置编码

RoPE 与 Sinusoidal PE 的区别：
  Sinusoidal PE: 生成位置向量，加到输入上（加法）
  RoPE:         旋转 Q 和 K 的向量（乘法），让 Attention 分数自带位置信息

RoPE 的核心公式：
  对 Q 和 K 做二维旋转：
    f(q, pos) = q x cos(pos*theta) + rotate_90(q) x sin(pos*theta)

效果：
  Q[pos_m] * K[pos_n] = f(Q_m, m) * f(K_n, n) = g(Q_m, K_n, m-n)
  即 Attention 分数只依赖 Q/K 的内容和**位置差**，不依赖绝对位置。

优势：
  - 相对位置编码：模型关注的是"距离"而非"绝对坐标"
  - 长度外推：推理时序列可以比训练时更长
  - 主流模型全部在用：LLaMA、Mistral、DeepSeek、Gemma
"""

import numpy as np


# ============================================================
# 1. RoPE 核心函数
# ============================================================

def precompute_rotary_frequencies(d_k: int, max_seq_len: int = 128, base: float = 10000.0):
    """预计算旋转角度（RoPE 的 theta 表）

    公式：theta_i = base^(-2i/d_k)
    跟 Sinusoidal PE 的分母完全一致，只是用法不同。

    参数：
        d_k: 每个 head 的维度
        max_seq_len: 最大序列长度
        base: 频率基数（默认 10000，跟 Transformer 原文一致）

    返回：
        cos_table: (max_seq_len, d_k/2) - 余弦值
        sin_table: (max_seq_len, d_k/2) - 正弦值
    """
    theta = base ** (-2 * np.arange(0, d_k, 2) / d_k)
    pos = np.arange(max_seq_len)
    angles = pos[:, None] * theta[None, :]
    cos_table = np.cos(angles)
    sin_table = np.sin(angles)
    return cos_table, sin_table


def apply_rotary(x: np.ndarray, cos_table: np.ndarray, sin_table: np.ndarray,
                 positions: np.ndarray = None) -> np.ndarray:
    """对 Q 或 K 应用 RoPE 旋转

    参数：
        x: (seq_len, d_k) - 一个 head 的 Q 或 K
        cos_table: (max_seq_len, d_k/2)
        sin_table: (max_seq_len, d_k/2)
        positions: 每个 token 的位置索引，默认 [0, 1, ..., seq_len-1]

    返回：
        (seq_len, d_k) - 旋转后的 Q 或 K
    """
    seq_len = x.shape[0]
    if positions is None:
        positions = np.arange(seq_len)

    cos_val = cos_table[positions]
    sin_val = sin_table[positions]

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]

    x_even_rotated = x_even * cos_val - x_odd * sin_val
    x_odd_rotated = x_even * sin_val + x_odd * cos_val

    result = np.empty_like(x)
    result[:, 0::2] = x_even_rotated
    result[:, 1::2] = x_odd_rotated
    return result


# ============================================================
# 2. RoPE 效果演示
# ============================================================

def demo_rope_basic():
    """基础演示：RoPE 如何旋转向量"""
    print("=" * 60)
    print("RoPE 基础演示：旋转 Q 和 K")
    print("=" * 60)

    d_k = 4
    seq_len = 3
    cos_table, sin_table = precompute_rotary_frequencies(d_k, max_seq_len=10)

    Q = np.array([
        [1.0, 0.0, 0.5, 0.0],
        [0.0, 1.0, 0.0, 0.5],
        [1.0, 1.0, 0.0, 0.0],
    ])

    Q_rotated = apply_rotary(Q, cos_table, sin_table)
    print(f"\n原始 Q:\n{np.round(Q, 3)}")
    print(f"\nRoPE 旋转后 Q:\n{np.round(Q_rotated, 3)}")

    K = Q.copy()
    K_rotated = apply_rotary(K, cos_table, sin_table)
    scores_no_rope = Q @ K.T / np.sqrt(d_k)
    scores_rope = Q_rotated @ K_rotated.T / np.sqrt(d_k)

    print(f"\n不加 RoPE 的 Attention 分数:\n{np.round(scores_no_rope, 3)}")
    print(f"  对称矩阵 -> 顺序颠倒后分数相同")

    print(f"\n加 RoPE 的 Attention 分数:\n{np.round(scores_rope, 3)}")
    print(f"  不对称 -> Q[0]*K[2] 和 Q[2]*K[0] 不同（相对位置信息）")

    q0 = Q_rotated[0]
    k1 = K_rotated[1]
    k2 = K_rotated[2]
    attn_01 = (q0 @ k1) / np.sqrt(d_k)
    attn_02 = (q0 @ k2) / np.sqrt(d_k)
    print(f"\n  同内容: Q[0]*K[1] = {attn_01:.3f}")
    print(f"  同内容: Q[0]*K[2] = {attn_02:.3f}")
    print(f"  距离越远 -> 旋转角度差越大 -> 分数越低（通常情况）")


def demo_rope_vs_sinusoidal():
    """对比 RoPE 和 Sinusoidal PE 的差异"""
    print("\n" + "=" * 60)
    print("RoPE vs Sinusoidal PE 对比")
    print("=" * 60)

    d_model = 8
    num_heads = 2
    d_k = d_model // num_heads
    seq_len = 4
    np.random.seed(42)
    X = np.random.randn(seq_len, d_model)

    from positional_encoding import sinusoidal_positional_encoding
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    X_sinusoidal = X + pe

    cos_table, sin_table = precompute_rotary_frequencies(d_k, max_seq_len=10)
    np.random.seed(123)
    Wq = np.random.randn(d_model, d_model) * 0.1
    Wk = np.random.randn(d_model, d_model) * 0.1

    Q_all = X @ Wq
    K_all = X @ Wk
    from utils import split_heads
    Q_heads = split_heads(Q_all, num_heads)
    K_heads = split_heads(K_all, num_heads)

    Q_rope = np.array([apply_rotary(Q_heads[h], cos_table, sin_table) for h in range(num_heads)])
    K_rope = np.array([apply_rotary(K_heads[h], cos_table, sin_table) for h in range(num_heads)])

    scores_rope = (Q_rope @ K_rope.transpose(0, 2, 1)) / np.sqrt(d_k)
    scores_no_pe = (Q_heads @ K_heads.transpose(0, 2, 1)) / np.sqrt(d_k)

    print(f"\n对比 Head 0 的 Attention 分数：")
    print(f"  无位置编码:\n{np.round(scores_no_pe[0], 3)}")
    print(f"  +RoPE:\n{np.round(scores_rope[0], 3)}")

    diag_no_pe = np.diag(scores_no_pe[0])
    diag_rope = np.diag(scores_rope[0])
    print(f"\n  对角线（自注意力）:")
    print(f"    无 PE: {np.round(diag_no_pe, 3)}")
    print(f"    RoPE:  {np.round(diag_rope, 3)}")
    same = np.allclose(diag_no_pe, diag_rope, atol=1e-6)
    print(f"    对角线不变性: {'成立' if same else '有差异'}")


def demo_permutation():
    """验证 RoPE 让顺序不同的句子产生不同 Attention"""
    print("\n" + "=" * 60)
    print("词序颠倒测试：验证 RoPE 区分正向和反向句子")
    print("=" * 60)

    d_k = 4
    cos_table, sin_table = precompute_rotary_frequencies(d_k)

    Q_forward = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    K_forward = Q_forward.copy()
    Q_backward = Q_forward[::-1].copy()
    K_backward = Q_backward.copy()

    Qf = apply_rotary(Q_forward, cos_table, sin_table)
    Kf = apply_rotary(K_forward, cos_table, sin_table)
    Qb = apply_rotary(Q_backward, cos_table, sin_table)
    Kb = apply_rotary(K_backward, cos_table, sin_table)

    sf = Qf @ Kf.T / np.sqrt(d_k)
    sb = Qb @ Kb.T / np.sqrt(d_k)

    print(f"\n正向序列 - RoPE:\n{np.round(sf, 3)}")
    print(f"\n反向序列 - RoPE:\n{np.round(sb, 3)}")

    same = np.allclose(sf, sb, atol=1e-6)
    print(f"\n  正向 == 反向? {'相等（有问题）' if same else '不等（正确）'}")


def demo_extrapolation():
    """RoPE 的长度外推能力"""
    print("\n" + "=" * 60)
    print("RoPE 长度外推：训练时 max_len=5，推理时扩展到 10")
    print("=" * 60)

    d_k = 4
    train_len = 5
    test_len = 10
    cos_table, sin_table = precompute_rotary_frequencies(d_k, max_seq_len=test_len + 1)

    q_train = np.random.randn(train_len, d_k) * 0.5
    q_all = np.random.randn(test_len, d_k) * 0.5

    q_train_rope = apply_rotary(q_train, cos_table, sin_table, np.arange(train_len))
    q_all_rope = apply_rotary(q_all, cos_table, sin_table, np.arange(test_len))

    k_all_rope = q_all_rope.copy()
    scores = q_all_rope @ k_all_rope.T / np.sqrt(d_k)

    print(f"\n  训练序列长度: {train_len}")
    print(f"  推理序列长度: {test_len}")
    print(f"  前 {train_len} 个位置的 cos/sin 表在训练时见过")
    print(f"  位置 5-9 的 cos/sin 表在测试时才用到")
    print(f"  RoPE 的位置 5-9 的旋转角度 = 5*theta, 6*theta... 公式一直存在")
    print(f"  不需要重新训练，RoPE 天然支持长度外推")


if __name__ == "__main__":
    print("RoPE（旋转位置编码）演示\n")
    demo_rope_basic()
    demo_rope_vs_sinusoidal()
    demo_permutation()
    demo_extrapolation()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
RoPE 与 Sinusoidal PE 的对比：

                 Sinusoidal PE        RoPE
  作用方式       加到输入 (加法)      旋转 Q/K (乘法)
  位置信息在     X + PE -> Attention  Q/K 内部自带
  绝对位置       有 (固定向量)        无 (靠旋转角度)
  相对位置       隐式 (sin/cos 组合)  显式 (分数含 pos_diff)
  外推更长序列   困难                 天然支持
  主流模型       Transformer原文      LLaMA/Mistral/DeepSeek
    """)
