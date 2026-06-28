"""
Mini Self-Attention — 纯 NumPy 实现
理解 Attention 最核心的计算过程
"""
import numpy as np


def softmax(x):
    """对矩阵的每一行做 softmax"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 减最大值防溢出
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


# ============================================================
# 1. 准备输入数据
# ============================================================
# 假设输入有 3 个词，每个词用 4 维向量表示
# 实际场景中，这 4 维是 embedding 层学出来的
X = np.array([
    [1.0, 2.0, 3.0, 4.0],   # 词1: "猫"
    [2.0, 3.0, 4.0, 5.0],   # 词2: "坐"
    [3.0, 4.0, 5.0, 6.0],   # 词3: "垫子"
])
vocab_size, d = X.shape  # (3, 4)

print("=" * 50)
print("输入 X (3个词, 每个4维):")
print(X)
print()

# ============================================================
# 2. 定义 Q/K/V 的权重矩阵
# ============================================================
np.random.seed(42)
Wq = np.random.randn(d, 3)  # (4, 3)
Wk = np.random.randn(d, 3)  # (4, 3)
Wv = np.random.randn(d, 3)  # (4, 3)

Q = X @ Wq  # (3, 3)
K = X @ Wk  # (3, 3)
V = X @ Wv  # (3, 3)

print("Q (Query) — 每个词作为查询者:")
print(np.round(Q, 3))
print("K (Key)   — 每个词作为被匹配者:")
print(np.round(K, 3))
print("V (Value) — 每个词的实际内容:")
print(np.round(V, 3))
print()

# ============================================================
# 3. 普通 Self-Attention（无掩码，双向）
# ============================================================
print("=" * 50)
print("Part A: 普通 Self-Attention（每个词看所有词）")
print("=" * 50)

# Q @ K^T + 缩放
scores = Q @ K.T
d_k = K.shape[1]
scores = scores / np.sqrt(d_k)

print("\n[Step 1] 注意力分数 (Q @ K^T, 已缩放):")
print(np.round(scores, 3))
print("  行 i = 查询词, 列 j = 被关注词")

attention_weights = softmax(scores)
print("\n[Step 2] Attention 权重 (Softmax 后):")
print(np.round(attention_weights, 3))
print("  每行和为 1")

output = attention_weights @ V
print("\n[Step 3] 输出 (加权求和):")
print(np.round(output, 3))
print("  每个词都看了所有词（包括未来的词）")
print()

# ============================================================
# 4. 因果掩码（Causal Mask）— GPT 自回归生成
# ============================================================
print("=" * 50)
print("Part B: 因果掩码 Self-Attention（只看过去和自己）")
print("=" * 50)

# 创建上三角掩码：未来位置 = -inf，当前/过去 = 0
seq_len = scores.shape[0]
causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9

print("\n[Step 1] 因果掩码矩阵:")
print(causal_mask)
print("  0    = 允许看")
print("  -∞  = 禁止看")

masked_scores = scores + causal_mask
print("\n[Step 2] 加掩码后的分数:")
print(np.round(masked_scores, 3))
print("  未来位置变成极大负数 → Softmax 后 = 0")

causal_attention = softmax(masked_scores)
print("\n[Step 3] 加掩码后的 Attention 权重:")
print(np.round(causal_attention, 3))
print("  对角线以上全是 0")
print("  每行依然和为 1")

causal_output = causal_attention @ V
print("\n[Step 4] 带因果掩码的输出:")
print(np.round(causal_output, 3))
print("  词1只看自己")
print("  词2看词1+自己")
print("  词3看词1+词2+自己")
print()

# ============================================================
# 5. 对比：无掩码 vs 有掩码
# ============================================================
print("=" * 50)
print("Part C: 对比总结")
print("=" * 50)
print(f"\n词1（猫）的输出对比:")
print(f"  无掩码: {np.round(output[0], 3)}  ← 看了所有词")
print(f"  有掩码: {np.round(causal_output[0], 3)}  ← 只看了自己")
print(f"\n词3（垫子）的输出对比:")
print(f"  无掩码: {np.round(output[2], 3)}  ← 看了所有词（包括未来的）")
print(f"  有掩码: {np.round(causal_output[2], 3)}  ← 看了词1+词2+自己")
print()
print("结论: 没有掩码时，每个词能看到完整的上下文（适合 BERT 等双向模型）")
print("      有掩码时，只能看过去和当前，不能偷看未来（适合 GPT 等生成模型）")
