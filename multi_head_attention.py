"""
Multi-Head Self-Attention — 纯 NumPy 实现
多个注意力头并行，每个头从不同角度理解句子
"""
import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


# ============================================================
# 1. 输入数据
# ============================================================
X = np.array([
    [1.0, 2.0, 3.0, 4.0],   # 词1
    [2.0, 3.0, 4.0, 5.0],   # 词2
    [3.0, 4.0, 5.0, 6.0],   # 词3
])
seq_len, d_model = X.shape  # (3, 4)

print("输入 X (3个词, 每个4维):")
print(X)
print(f"shape: ({seq_len}, {d_model})")
print()

# ============================================================
# 2. 多头配置
# ============================================================
num_heads = 2                     # 头数
d_k = d_model // num_heads        # 每个头的维度: 4 / 2 = 2

print(f"配置: {num_heads} 个头, 每个头维度 = {d_k}")
print()

# 创建 Q/K/V 权重矩阵（每个头有自己的 Q/K/V 权重）
# 整体投影: d_model(=4) → d_model(=4)
# 之后拆成 num_heads 份，每份 d_k 维
np.random.seed(42)
Wq = np.random.randn(d_model, d_model)  # (4, 4)
Wk = np.random.randn(d_model, d_model)  # (4, 4)
Wv = np.random.randn(d_model, d_model)  # (4, 4)
Wo = np.random.randn(d_model, d_model)  # (4, 4)  输出投影

# 整体计算 Q/K/V
Q = X @ Wq  # (3, 4)
K = X @ Wk  # (3, 4)
V = X @ Wv  # (3, 4)

print("整体投影后的 Q shape:", Q.shape)
print()

# ============================================================
# 3. 把 Q/K/V 拆成多个头
# ============================================================
# 原始 Q: (3, 4) → reshape 成 (3, 2, 2) → transpose 成 (2, 3, 2)
# 解释: (seq_len, d_model) → (seq_len, num_heads, d_k) → (num_heads, seq_len, d_k)
# 这样每个头的 shape = (seq_len, d_k)，方便并行计算

def split_heads(x, num_heads):
    """将 (seq_len, d_model) 拆成 (num_heads, seq_len, d_k)"""
    seq_len, d_model = x.shape
    d_k = d_model // num_heads
    x = x.reshape(seq_len, num_heads, d_k)   # (3, 2, 2)
    x = x.transpose(1, 0, 2)                 # (2, 3, 2)
    return x

Q_heads = split_heads(Q, num_heads)
K_heads = split_heads(K, num_heads)
V_heads = split_heads(V, num_heads)

print("拆分后的 Q_heads shape: (num_heads, seq_len, d_k) =", Q_heads.shape)
print()

# ============================================================
# 4. 每个头独立算 Attention
# ============================================================
# score_heads[h] = Q_heads[h] @ K_heads[h].T / sqrt(d_k)
# 所有头一起算: (2, 3, 2) @ (2, 2, 3) → (2, 3, 3)
# 即 (num_heads, seq_len, d_k) @ (num_heads, d_k, seq_len) → (num_heads, seq_len, seq_len)

scores_heads = (Q_heads @ K_heads.transpose(0, 2, 1)) / np.sqrt(d_k)
attention_heads = softmax(scores_heads)
output_heads = attention_heads @ V_heads

print("多头 Attention 结果 shape: (num_heads, seq_len, d_k) =", output_heads.shape)
print()

# 打印每个头的注意力权重
for h in range(num_heads):
    print(f"--- 头 {h+1} 的 Attention 权重 ---")
    print(np.round(attention_heads[h], 3))
    print()

# ============================================================
# 5. 合并所有头 → 最终输出
# ============================================================
# output_heads: (2, 3, 2) → transpose(1, 0, 2) → (3, 2, 2) → reshape → (3, 4)
# 即: (num_heads, seq_len, d_k) → (seq_len, num_heads, d_k) → (seq_len, d_model)

def combine_heads(x, num_heads):
    """将 (num_heads, seq_len, d_k) 合并回 (seq_len, d_model)"""
    num_heads, seq_len, d_k = x.shape
    x = x.transpose(1, 0, 2)          # (seq_len, num_heads, d_k)
    x = x.reshape(seq_len, -1)        # (seq_len, d_model)
    return x

combined = combine_heads(output_heads, num_heads)

# 最后的输出投影 Wo
final_output = combined @ Wo

print("合并后的 shape:", combined.shape)
print("最终输出 (经过 Wo 投影后):")
print(np.round(final_output, 3))
print()

# ============================================================
# 6. 与单头对比
# ============================================================
# 用同样的 Q/K/V 做单头 Attention
Wq_single = Wq
Wk_single = Wk
Wv_single = Wv

Q_single = X @ Wq_single
K_single = X @ Wk_single
V_single = X @ Wv_single

score_single = Q_single @ K_single.T / np.sqrt(d_model)
attn_single = softmax(score_single)
output_single = (attn_single @ V_single)

print("=" * 50)
print("对比: 单头 vs 多头")
print("=" * 50)
print(f"\n单头 attention (所有维度在一个头里算):")
print(np.round(attn_single, 3))
print(f"\n多头 attention (2个头各看各的):")
for h in range(num_heads):
    print(f"  头{h+1}: {np.round(attention_heads[h], 3)}")
print()
print("多头让模型从不同角度同时观察句子关系。")
