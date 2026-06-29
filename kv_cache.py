"""
KV Cache — 自回归生成中的推理加速

面试高频题: "为什么 LLM 生成时第一个字慢，后面越来越快？"

核心观察:
  自回归生成时，每步只产生 1 个新词。
  旧词的 K 和 V 在后续步骤中不会变，但会被重新计算多次。

  无缓存: 每步对所有词重新算 Q/K/V → O(N²) 计算量
  有缓存: 每步只算新词的 Q/K/V，存 K/V 供后续复用 → O(N) 计算量

为什么不缓存 Q?
  旧 Q 在后续步骤中不再被需要（旧词不会作为"查询者"再次出现）。
  旧 K/V 一直被需要（新词需要匹配所有旧 K，加权所有旧 V）。
"""
import numpy as np
from utils import softmax


# ============================================================
# 1. 设定场景
# ============================================================
# 模拟生成 4 个词的过程
# 每个词用 4 维向量，Q/K/V 投影到 3 维
np.random.seed(42)

# 模拟的输入 embedding: 相邻词之间有些许重叠，模拟语义相似
embeddings = {
    0: np.array([1.0, 0.5, 0.0, 0.0]),    # 词0
    1: np.array([0.5, 1.0, 0.5, 0.0]),    # 词1
    2: np.array([0.0, 0.5, 1.0, 0.5]),    # 词2
    3: np.array([0.0, 0.0, 0.5, 1.0]),    # 词3
}

d_model = 4   # 输入维度
d_k = 3       # Q/K/V 目标维度

# Q/K/V 权重矩阵（随机初始化，演示用）
Wq = np.random.randn(d_model, d_k)  # (4, 3)
Wk = np.random.randn(d_model, d_k)  # (4, 3)
Wv = np.random.randn(d_model, d_k)  # (4, 3)

print("=" * 60)
print("KV Cache 演示：逐步生成 4 个词")
print("=" * 60)

# ============================================================
# 2. 无 KV Cache 版本（每次从头算全部）
# ============================================================
print("\n--- 无 KV Cache：每次重新算所有词的 Attention ---")

def generate_no_cache(num_tokens):
    """
    朴素方式：每步生成时，对已有的全部词重新算 Attention

    问题:
      第 1 步: 算 K₀ V₀
      第 2 步: 算 K₀ V₀ + K₁ V₁  ← K₀ V₀ 重复了
      第 3 步: 算 K₀ V₀ + K₁ V₁ + K₂ V₂  ← K₀V₀ K₁V₁ 重复了
      ...
      重复计算随着生成长度 N 呈 O(N²) 增长
    """
    for step in range(num_tokens):
        # 当前已有的全部输入
        tokens = [embeddings[i] for i in range(step + 1)]
        X = np.array(tokens)  # (step+1, 4)
        #         ↑ 注意: step=0 时 shape (1,4); step=1 时 shape (2,4)

        # 重新算所有词的 Q/K/V（包括旧词）
        Q = X @ Wq  # (step+1, 3)
        K = X @ Wk  # (step+1, 3)
        V = X @ Wv  # (step+1, 3)

        # Attention（带因果掩码）
        seq_len = step + 1
        scores = Q @ K.T / np.sqrt(d_k)
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        masked_scores = scores + mask
        attn_weights = softmax(masked_scores)

        # 只取最后一个词的输出作为预测结果
        # output[0..step-1] 是旧词的结果，被丢弃了
        output = attn_weights @ V
        # predicted = output[-1]  # 新词的预测结果

        print(f"  第{step+1}步: 已有{step+1}个词, 重新算了{step+1}个Q, {step+1}个K, {step+1}个V")

generate_no_cache(4)

# ============================================================
# 3. 有 KV Cache 版本（只算新词的 Q，复用旧的 K/V）
# ============================================================
print("\n--- 有 KV Cache：只算新词的 Q，旧的 K/V 存起来复用 ---")

def generate_with_cache(num_tokens):
    """
    优化方式：每步只算新词的 Q/K/V，K/V 存到缓存中

    关键区别:
      - 第 2 步: 只算词1 的 Q₁K₁V₁, K₀V₀ 从缓存拿
      - 第 3 步: 只算词2 的 Q₂K₂V₂, K₀V₀K₁V₁ 从缓存拿
      - 每步只算 1 个词的 Q/K/V，跟总词数无关

    为什么缓存 K 和 V 而不是 Q?
      - 新词 (Q_new) 需要匹配所有旧 K → 旧 K 必须留着
      - 新词 (注意力权重) 需要乘以所有旧 V → 旧 V 必须留着
      - 旧 Q 只用于旧词自己的查询，生成后不再被需要 → 不缓存
    """
    cache_k = []  # K 缓存列表
    cache_v = []  # V 缓存列表

    for step in range(num_tokens):
        # 取当前词的 embedding，reshape 成 (1, d_model) 用于矩阵乘法
        token = embeddings[step]
        x = token.reshape(1, -1)  # (1, 4)
        #   ↑ reshape 的作用:
        #     token shape (4,) → 一维数组，不能 @ Wq
        #     x shape (1, 4)  → 二维矩阵，可以 @ Wq

        # 只算当前这个词的 Q/K/V（不是全部）
        q_new = x @ Wq  # (1, 3)
        k_new = x @ Wk  # (1, 3)
        v_new = x @ Wv  # (1, 3)

        # 把新的 K/V 追加到缓存
        cache_k.append(k_new)
        cache_v.append(v_new)

        # 从缓存拿出全部 K/V（直接拼接，不用重新算）
        K_all = np.concatenate(cache_k, axis=0)  # (step+1, 3)
        V_all = np.concatenate(cache_v, axis=0)  # (step+1, 3)

        # Attention: 只用新词的 Q 去匹配缓存的全部 K
        # Q shape: (1, 3), K_all shape: (step+1, 3)
        # scores shape: (1, step+1) — 只需 1 行
        scores = q_new @ K_all.T / np.sqrt(d_k)
        attn_weights = softmax(scores)
        output = attn_weights @ V_all  # (1, 3)

        # 汇报: 新算了多少 vs 复用了多少
        q_count = 1
        k_count = step + 1
        reused_from_cache = step
        print(f"  第{step+1}步: 新算 {q_count} 个Q, {q_count} 个K, {q_count} 个V")
        print(f"           从缓存复用 {reused_from_cache} 个K, {reused_from_cache} 个V")

generate_with_cache(4)

# ============================================================
# 4. 计算量对比
# ============================================================
print("\n" + "=" * 60)
print("计算量对比（生成长度 = N）")
print("=" * 60)

def compute_flops_no_cache(N, d):
    """无缓存：每步重新算所有词的 QKV + Attention"""
    total_qkv = 0
    total_attn = 0
    for step in range(N):
        seq_len = step + 1
        total_qkv += 3 * seq_len * d * d   # 3 个矩阵乘 (Q/K/V)
        total_attn += seq_len * seq_len * d  # Attention 分数 + 加权求和
    return total_qkv + total_attn

def compute_flops_with_cache(N, d):
    """有缓存：每步只算 1 个新词的 Q/K/V"""
    total_qkv = 0
    total_attn = 0
    for step in range(N):
        total_qkv += 3 * d * d              # 每次只算 1 个词的 Q/K/V
        total_attn += (step + 1) * d        # 1 个 Q 匹配 step+1 个 K
    return total_qkv + total_attn

print(f"\n{'N':>6} {'无缓存':>10} {'有缓存':>10} {'加速比':>8}")
print("-" * 38)
for N in [4, 10, 100, 1000]:
    no_cache = compute_flops_no_cache(N, d_k)
    with_cache = compute_flops_with_cache(N, d_k)
    speedup = no_cache / with_cache
    print(f"{N:>6} {no_cache:>10,} {with_cache:>10,} {speedup:>7.1f}x")
print()
print("结论: N 越大加速越明显。无缓存 O(N²) → 有缓存 O(N)")
print("这就是 LLM 生成时'第一个字慢，后面越来越快'的根本原因")
print("（第 1 步没有缓存 + Prefill 阶段要处理全部 prompt）")
