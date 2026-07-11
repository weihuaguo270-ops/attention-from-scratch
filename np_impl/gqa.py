"""
Grouped Query Attention (GQA) — 分组查询注意力

GQA 是 MHA (Multi-Head Attention) 和 MQA (Multi-Query Attention) 的折中方案。

问题背景：
  MHA: 每个 Q 头有自己独立的 K/V 头 → KV Cache 大（参数量大，缓存占用高）
  MQA: 所有 Q 头共用一个 K/V 头 → KV Cache 最小（但精度可能下降）
  GQA: Q 头分成 g 组，每组共享一个 K/V 头 → 折中

KV Cache 大小对比（num_heads=32, d_k=128, seq_len=4096）：
  MHA: 32 × 4096 × 128 × 2 × 2 bytes = 64 MB
  MQA: 1 × 4096 × 128 × 2 × 2 bytes = 2 MB
  GQA (g=8): 4 × 4096 × 128 × 2 × 2 bytes = 8 MB

GQA 工作方式：
  1. Q 投影到 num_heads 个头，K/V 投影到 num_kv_heads 个头
  2. K/V 头被「重复」以匹配 Q 头的数量（num_heads / num_kv_heads 次）
  3. 每个 Q 头与自己组内的 K/V 头做 Attention

形状变化：
  输入: (seq_len, d_model)
  Q: (num_heads, seq_len, d_k)        ← 多头
  K: (num_kv_heads, seq_len, d_k)     ← 少头
  V: (num_kv_heads, seq_len, d_k)     ← 少头
  K 重复 → (num_heads, seq_len, d_k)
  V 重复 → (num_heads, seq_len, d_k)

参考: Llama 2 70B 使用 GQA (num_heads=64, num_kv_heads=8)
      Llama 3 70B 使用 GQA (num_heads=64, num_kv_heads=8)
      Mistral 使用 GQA (num_heads=32, num_kv_heads=8)
"""
import sys, os
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "np_impl"
import numpy as np
from .utils import softmax, layer_norm


class GroupedQueryAttention:
    """
    分组查询注意力 (GQA)

    与 MultiHeadAttention 的核心区别：
      - MHA:  num_kv_heads == num_heads（每个 Q 头独自拥有 K/V）
      - GQA:  num_kv_heads < num_heads（多个 Q 头共享一组 K/V）
      - MQA:  num_kv_heads == 1（所有 Q 头共享一套 K/V）

    参数:
        d_model: 模型维度
        num_heads: Q 头数
        num_kv_heads: K/V 头数（必须能整除 num_heads）
        use_rope: 是否对 Q/K 应用 RoPE
        max_seq_len: 最大序列长度（RoPE 需要）
    """
    def __init__(self, d_model, num_heads, num_kv_heads, use_rope=False, max_seq_len=128):
        assert num_heads % num_kv_heads == 0, \
            f"num_heads({num_heads}) 必须能被 num_kv_heads({num_kv_heads}) 整除"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads  # 每组包含的 Q 头数
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        # Q 投影: (d_model, d_model) — 完整的多头投影
        self.Wq = np.random.randn(d_model, d_model) * 0.01
        # K/V 投影: (d_model, d_model_kv) — 更少的头数
        self.d_kv = self.d_k * num_kv_heads  # K/V 总维度
        self.Wk = np.random.randn(d_model, self.d_kv) * 0.01
        self.Wv = np.random.randn(d_model, self.d_kv) * 0.01
        # 输出投影
        self.Wo = np.random.randn(d_model, d_model) * 0.01

        # RoPE 预计算
        if use_rope:
            from .rotary import precompute_rotary_frequencies
            self._cos_table, self._sin_table = precompute_rotary_frequencies(
                self.d_k, max_seq_len=max_seq_len
            )

    def _split_heads(self, x, num_heads):
        """将投影后的 x 拆成多头"""
        seq_len, _ = x.shape
        d_k = self.d_k if num_heads == self.num_heads else self.d_k
        x = x.reshape(seq_len, num_heads, d_k)
        return x.transpose(1, 0, 2)  # (num_heads, seq_len, d_k)

    def _repeat_kv(self, kv, num_repeats):
        """将 K/V 头重复以匹配 Q 头的数量"""
        # kv shape: (num_kv_heads, seq_len, d_k)
        # 返回: (num_heads, seq_len, d_k)
        if num_repeats == 1:
            return kv
        return np.repeat(kv, num_repeats, axis=0)

    def forward(self, x, use_mask=True, positions=None):
        """
        前向传播

        流程:
          x → Q(多头投影) + K/V(少头投影) → 拆头 → [RoPE] → K/V 重复 → Attention → 合并 → Wo

        参数:
            x: 输入 (seq_len, d_model)
            use_mask: 是否使用因果掩码
            positions: RoPE 位置索引
        返回:
            (seq_len, d_model)
        """
        seq_len = x.shape[0]

        # Step 1: 投影
        Q = x @ self.Wq  # (seq_len, d_model)
        K = x @ self.Wk  # (seq_len, d_kv)
        V = x @ self.Wv  # (seq_len, d_kv)

        # Step 2: 拆头
        Q = self._split_heads(Q, self.num_heads)      # (num_heads, seq_len, d_k)
        K = self._split_heads(K, self.num_kv_heads)   # (num_kv_heads, seq_len, d_k)
        V = self._split_heads(V, self.num_kv_heads)   # (num_kv_heads, seq_len, d_k)

        # Step 3: RoPE
        if self.use_rope:
            from .rotary import apply_rotary
            if positions is None:
                positions = np.arange(seq_len)
            for h in range(self.num_kv_heads):
                K[h] = apply_rotary(K[h], self._cos_table, self._sin_table, positions)
            for h in range(self.num_heads):
                Q[h] = apply_rotary(Q[h], self._cos_table, self._sin_table, positions)

        # Step 4: 重复 K/V 以匹配 Q 头
        num_repeats = self.num_heads // self.num_kv_heads
        K = self._repeat_kv(K, num_repeats)  # (num_heads, seq_len, d_k)
        V = self._repeat_kv(V, num_repeats)  # (num_heads, seq_len, d_k)

        # Step 5: Attention
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(self.d_k)  # (num_heads, seq_len, seq_len)
        if use_mask:
            mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
            scores = scores + mask
        attn_weights = softmax(scores)
        head_outputs = attn_weights @ V  # (num_heads, seq_len, d_k)

        # Step 6: 合并 + 输出投影
        combined = head_outputs.transpose(1, 0, 2).reshape(seq_len, -1)  # (seq_len, d_model)
        return combined @ self.Wo


# ============================================================
# 演示：对比 MHA / GQA / MQA 的 KV Cache 大小
# ============================================================
def compare_kv_cache_sizes():
    """比较 MHA / GQA / MQA 的 KV Cache 大小"""
    print("=" * 60)
    print("KV Cache 大小对比（num_heads=32, d_k=128, seq_len=4096）")
    print("=" * 60)

    num_heads = 32
    d_k = 128
    seq_len = 4096
    bytes_per_float = 2  # FP16

    configs = [
        ("MHA (32 KV heads)", 32),
        ("GQA (8 KV heads) ", 8),
        ("GQA (4 KV heads) ", 4),
        ("MQA  (1 KV head) ", 1),
    ]

    print(f"\n{'配置':>20} | {'KV Cache 大小':>15} | {'相对 MHA':>10}")
    print("-" * 50)
    mha_size = None
    for name, num_kv in configs:
        # KV Cache = 2 (K+V) × num_kv_heads × seq_len × d_k × bytes
        size = 2 * num_kv * seq_len * d_k * bytes_per_float / (1024 * 1024)
        if mha_size is None:
            mha_size = size
            ratio = "1x"
        else:
            ratio = f"{size / mha_size:.1%}"
        print(f"{name:>20} | {size:>8.1f} MB | {ratio:>10}")
    print()


# ============================================================
# 演示：GQA 的 K/V 重复机制
# ============================================================
def demo_gqa_mechanism():
    """可视化 GQA 的 K/V 重复机制"""
    print("=" * 60)
    print("GQA K/V 重复机制演示")
    print("=" * 60)

    d_model = 8
    num_heads = 4
    num_kv_heads = 2
    seq_len = 3

    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
    x = np.random.randn(seq_len, d_model)

    Q = x @ gqa.Wq
    K = x @ gqa.Wk
    V = x @ gqa.Wv

    Q_heads = gqa._split_heads(Q, num_heads)
    K_heads = gqa._split_heads(K, num_kv_heads)
    V_heads = gqa._split_heads(V, num_kv_heads)

    print(f"\nQ 头数: {num_heads}  → Q shape: {Q_heads.shape}")
    print(f"K/V 头数: {num_kv_heads} → K shape: {K_heads.shape}")
    print(f"  每组 {num_heads // num_kv_heads} 个 Q 头共享 1 个 K/V 头")

    K_repeated = gqa._repeat_kv(K_heads, num_heads // num_kv_heads)
    V_repeated = gqa._repeat_kv(V_heads, num_heads // num_kv_heads)

    print(f"\nK/V 重复后: {K_repeated.shape} ← 匹配 Q 头数")
    print(f"  K 原头 0 → Q 头 0, 1")
    print(f"  K 原头 1 → Q 头 2, 3")

    # 验证重复正确性
    for q_idx in range(num_heads):
        kv_origin = q_idx // (num_heads // num_kv_heads)
        assert np.allclose(K_heads[kv_origin], K_repeated[q_idx])
    print(f"\n✅ K/V 重复验证通过")

    # 输出 GQA Attention 结果
    out = gqa.forward(x, use_mask=False)
    print(f"\nGQA 输出 shape: {out.shape}")
    print(f"GQA 输出 stable: {np.all(np.isfinite(out))}")


if __name__ == "__main__":
    print("GQA（Grouped Query Attention）演示")
    print()

    demo_gqa_mechanism()
    compare_kv_cache_sizes()

    # 完整 GQA 前向验证
    print("=" * 60)
    print("GQA 前向验证")
    print("=" * 60)
    np.random.seed(42)
    gqa = GroupedQueryAttention(d_model=8, num_heads=4, num_kv_heads=2, use_rope=True)
    x = np.random.randn(4, 8)
    out = gqa.forward(x, use_mask=True)
    print(f"\n  输入: (4, 8)")
    print(f"  输出: {out.shape}")
    print(f"  稳定: {np.all(np.isfinite(out))}")
    print(f"  非零: {np.linalg.norm(out) > 0}")
    print()
