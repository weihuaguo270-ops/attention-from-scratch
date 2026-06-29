"""
Self-Attention + 因果掩码 — 纯 NumPy 实现

理解 Attention 最核心的计算过程:

  公式: Attention(Q, K, V) = softmax(QK^T / √d_k) V

  1. Q @ K^T      → 每个词跟所有词的相似度分数
  2. / √d_k       → 缩放，防止分数太大导致 Softmax 极端化
  3. + 因果掩码    → 遮住未来位置（GPT 生成时不能偷看）
  4. Softmax      → 分数转成概率（每行和为 1）
  5. @ V          → 加权求和，得到每个词的"上下文感知表示"

两种模式:
  Part A: 无掩码 Self-Attention → 每个词看所有词（BERT 风格）
  Part B: 因果掩码 Self-Attention → 每个词只看过去+自己（GPT 风格）
"""
import numpy as np
from utils import softmax


# ============================================================
# 1. 准备输入数据
# ============================================================
# 假设输入有 3 个词，每个词用 4 维向量表示
# 实际场景中，这 4 维是 embedding 层学出来的
X = np.array([
    [1.0, 2.0, 3.0, 4.0],   # 词0: "猫"
    [2.0, 3.0, 4.0, 5.0],   # 词1: "坐"
    [3.0, 4.0, 5.0, 6.0],   # 词2: "垫子"
])
vocab_size, d = X.shape  # (3, 4) — 3个词，每个4维

print("=" * 50)
print("输入 X (3个词, 每个4维):")
print(X)
print()

# ============================================================
# 2. 定义 Q/K/V 的权重矩阵
# ============================================================
# 每个词通过三个不同的权重矩阵得到 Q、K、V
# Wq, Wk, Wv 将 4 维输入映射到 3 维的 Q/K/V 空间
# 这里随机初始化，实际训练中这些矩阵是学出来的
np.random.seed(42)
Wq = np.random.randn(d, 3)  # (4, 3) — 从 4 维映射到 3 维的 Query 空间
Wk = np.random.randn(d, 3)  # (4, 3) — 从 4 维映射到 3 维的 Key 空间
Wv = np.random.randn(d, 3)  # (4, 3) — 从 4 维映射到 3 维的 Value 空间

# Q/K/V = 输入 × 权重矩阵
# Q: 每个词作为"查询者"，去匹配其他词
# K: 每个词作为"被匹配者"，提供自己的索引标签
# V: 每个词的实际内容信息
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
# Part A: 普通 Self-Attention（无掩码，双向）
# ============================================================
# 每个词可以看到句子里的所有词（包括未来的词）
# 适用于理解整个句子的任务（BERT 分类、情感分析等）
print("=" * 50)
print("Part A: 普通 Self-Attention（每个词看所有词）")
print("=" * 50)

# Step 1: Q @ K^T — 算相似度分数
# scores[i][j] = 词 i 的 Q 与 词 j 的 K 的点积
# 值越大 → 词 i 越关注词 j
scores = Q @ K.T              # (3, 3)

# Step 2: 缩放 — 除以 √d_k
# 防止点积结果太大导致 Softmax 极端化
d_k = K.shape[1]              # K 的维度，这里是 3
scores = scores / np.sqrt(d_k)  # 让方差保持在 1 附近

print("\n[Step 1] 注意力分数 (Q @ K^T, 已缩放):")
print(np.round(scores, 3))
print("  行 i = 查询词, 列 j = 被关注词")

# Step 3: Softmax — 分数转成概率
# 每行独立做 Softmax，使每行的权重加起来 = 1
attention_weights = softmax(scores)

print("\n[Step 2] Attention 权重 (Softmax 后):")
print(np.round(attention_weights, 3))
print("  每行和为 1")
print("  注意: 词1(猫)的关注: 99.7%看自己, 0.3%看词2")

# Step 4: 加权求和 — 用注意力权重 × V
# 每个词的输出 = Σ(权重 × 对应词的 V)
# 输出向量包含了"所有词的信息"，权重越大贡献越多
output = attention_weights @ V

print("\n[Step 3] 输出 (加权求和):")
print(np.round(output, 3))
print("  每个词的输出 = 融合了其他词信息的'上下文感知表示'")
print()

# ============================================================
# Part B: 因果掩码（Causal Mask）— GPT 自回归生成
# ============================================================
# 生成文字时，词 i 只能看 i 及 i 之前的词，不能看 i 之后的词
# 比如写"猫坐垫子"→ 生成"坐"时不能提前看到"垫子"
# 否则模型直接抄答案，不用真学推理
print("=" * 50)
print("Part B: 因果掩码 Self-Attention（只看过去和自己）")
print("=" * 50)

# 创建上三角掩码矩阵
# 对角线及以下 = 0（允许看），对角线以上 = -1e9（禁止看）
# -1e9 ≈ 负无穷，Softmax 后权重精确为 0
seq_len = scores.shape[0]
causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9

print("\n[Step 1] 因果掩码矩阵:")
print(causal_mask)
print("  0    = 允许看（过去位置 + 当前位置）")
print("  -∞  = 禁止看（未来位置）")

# 把掩码加到分数上：未来位置被加上 -1e9 → 极大负数
masked_scores = scores + causal_mask

print("\n[Step 2] 加掩码后的分数:")
print(np.round(masked_scores, 3))
print("  未来位置 → 极大负数 → Softmax 后权重 = 0")

# 对加掩码后的分数做 Softmax
# 未来位置的权重被精确归零
causal_attention = softmax(masked_scores)

print("\n[Step 3] 加掩码后的 Attention 权重:")
print(np.round(causal_attention, 3))
print("  对角线以上全为 0 ✅ 未来位置被硬性封杀")
print("  每行依然和为 1")

# 加权求和：只看过去和当前，不看未来
causal_output = causal_attention @ V

print("\n[Step 4] 带因果掩码的输出:")
print(np.round(causal_output, 3))
print("  词1（猫）:   只看自己")
print("  词2（坐）:   看词1+自己（不能看词3）")
print("  词3（垫子）: 看词1+词2+自己（没有未来位置）")
print()

# ============================================================
# Part C: 对比总结
# ============================================================
print("=" * 50)
print("Part C: 对比总结")
print("=" * 50)
print(f"\n词1（猫）的输出对比:")
print(f"  无掩码: {np.round(output[0], 3)}  ← 看了所有词")
print(f"  有掩码: {np.round(causal_output[0], 3)}  ← 只看了自己")
print(f"\n词2（坐）的输出对比:")
print(f"  无掩码: {np.round(output[1], 3)}  ← 看了包括词3在内的所有词")
print(f"  有掩码: {np.round(causal_output[1], 3)}  ← 只能看词1+自己")
print(f"\n词3（垫子）的输出对比:")
print(f"  无掩码: {np.round(output[2], 3)}  ← 看了所有词")
print(f"  有掩码: {np.round(causal_output[2], 3)}  ← 看了词1+词2+自己")
print()
print("结论:")
print("  无掩码 = 双向注意力（BERT 风格）— 整个句子理解")
print("  有掩码 = 自回归注意力（GPT 风格）— 逐词生成，不能偷看未来")
