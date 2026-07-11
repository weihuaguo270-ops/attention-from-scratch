"""
Multi-head Latent Attention (MLA) — 多头潜注意力

背景：
  DeepSeek V2 提出了 MLA 作为其核心创新点，将 KV Cache 压缩至原来的 ~5-10%。
  对于大模型（如 DeepSeek V2 的 236B 参数），KV Cache 占用了绝大多数 GPU 显存，
  MLA 从根本上解决了这个问题。

问题：
  标准 MHA 的 KV Cache 大小 = 2 × num_heads × seq_len × d_k × bytes_per_elem
  DeepSeek V2 (num_heads=128, d_k=128, seq_len=4096, FP16):
    KV Cache ≈ 2 × 128 × 4096 × 128 × 2 = 256 MB per request
    这只是一层！128 层 → 32 GB — 单是 KV Cache 就爆显存了

MLA 方案：
  把 K/V 投影到一个低维「潜空间」(latent space)，缓存这个低维向量。
  推理时再从潜空间「解压」回来。

  以 DeepSeek V2 为例：
    d_model = 5120, num_heads = 128, d_k = 128, d_c = 512 (压缩维度)
    标准 KV Cache（单层，单请求）: 2 × 128 × 4096 × 128 × 2 = 256 MB
    MLA KV Cache（单层，单请求）:  (512 + 64) × 4096 × 2 = 4.5 MB
    压缩比: ~56x

MLA 工作原理：

  编码阶段（每步都做）:
    c^{KV} = W^{DKV} × h           (d_model → d_c 的降维)
    k^C    = W^{UK} × c^{KV}       (d_c → d_model 的升维，得到 Key 的内容部分)
    k^R    = RoPE(W^{KR} × h)      (d_model → d_kv_rope 的分离 RoPE Key)
    v      = W^{UV} × c^{KV}       (d_c → d_model 的升维，得到 Value)
    k      = [k^C; k^R]            (拼接 Key 的内容和旋转部分)

  缓存内容:
    缓存 c^{KV} (d_c 维) + k^R (d_kv_rope 维)，而不是全部 k 和 v

  推理优化（吸收矩阵）:
    W^{UK} 可以被「吸收」到 Q 的投影矩阵中：
      Q × (W^{UK} × c^{KV}) = (Q × W^{UK}) × c^{KV}
      W^{UK} 与 Q 投影合并 → 推理时不需要显式解压 K

参考: DeepSeek V2 (https://arxiv.org/abs/2405.04434)
"""
import sys, os
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "np_impl"
import numpy as np
from .utils import softmax


class MultiHeadLatentAttention:
    """
    Multi-head Latent Attention (MLA)

    核心机制: 将 K/V 压缩到低维潜空间缓存，大幅减少 KV Cache 占用。

    参数:
        d_model: 模型维度
        num_heads: Q 头数
        d_k: 每个 head 的维度（如 d_model // num_heads）
        d_c: 潜空间维度（压缩后的 K/V 维度，通常远小于 d_model）
        d_kv_rope: RoPE 部分的维度（通常较小）
        max_seq_len: 最大序列长度
    """
    def __init__(self, d_model, num_heads, d_k, d_c, d_kv_rope=32, max_seq_len=128):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_c = d_c                  # K/V 压缩维度
        self.d_kv_rope = d_kv_rope      # RoPE 部分的 Key 维度

        # === Q 投影 ===
        # Q 有内容和 RoPE 两部分
        # W^{UQ}: 升维到 Q (d_model → d_model)
        # W^{QR}: RoPE Q (d_model → d_kv_rope)
        self.Wq = np.random.randn(d_model, d_model) * 0.01
        self.W_qr = np.random.randn(d_model, d_kv_rope) * 0.01

        # === K/V 压缩投影 ===
        # W^{DKV}: 降维到潜空间 (d_model → d_c)
        # W^{UK}:  潜空间升维到 K 内容 (d_c → d_model)
        # W^{KR}:  RoPE K (d_model → d_kv_rope)
        # W^{UV}:  潜空间升维到 V (d_c → d_model)
        self.W_dkv = np.random.randn(d_model, d_c) * 0.01
        self.W_uk = np.random.randn(d_c, d_model) * 0.01
        self.W_kr = np.random.randn(d_model, d_kv_rope) * 0.01
        self.W_uv = np.random.randn(d_c, d_model) * 0.01

        # === 输出投影 ===
        self.Wo = np.random.randn(d_model, d_model) * 0.01

        # RoPE 预计算
        from .rotary import precompute_rotary_frequencies
        self._cos_table, self._sin_table = precompute_rotary_frequencies(
            d_kv_rope, max_seq_len=max_seq_len
        )

    def forward(self, x, use_mask=True, positions=None):
        """
        完整的 MLA 前向传播

        流程:
          x → Q(拆头) + [Q^R] → K (压缩→升维 + RoPE) → V (压缩→升维)
          → Attention → 输出

        参数:
            x: (seq_len, d_model)
            use_mask: 因果掩码
            positions: RoPE 位置索引
        返回:
            (seq_len, d_model)
        """
        seq_len = x.shape[0]
        if positions is None:
            positions = np.arange(seq_len)

        # ============================================================
        # 1. 计算 Q（不做压缩，但为了吸收矩阵演示，分开计算）
        # ============================================================
        # Q 内容部分: (seq_len, d_model)
        q_c = x @ self.Wq

        # Q RoPE 部分: (seq_len, d_kv_rope)
        q_r = x @ self.W_qr
        from .rotary import apply_rotary
        q_r = apply_rotary(q_r, self._cos_table, self._sin_table, positions)

        # 拆头: (num_heads, seq_len, d_k)
        q_heads = q_c.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

        # ============================================================
        # 2. K/V 压缩 + 升维
        # ============================================================
        # 降维到潜空间: (seq_len, d_c)
        c_kv = x @ self.W_dkv

        # 从潜空间升维回 K 内容: (seq_len, d_model)
        k_c = c_kv @ self.W_uk

        # K RoPE 部分: (seq_len, d_kv_rope)
        k_r = x @ self.W_kr
        k_r = apply_rotary(k_r, self._cos_table, self._sin_table, positions)

        # 从潜空间升维回 V: (seq_len, d_model)
        v = c_kv @ self.W_uv

        # 拆头
        k_heads = k_c.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        v_heads = v.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

        # ============================================================
        # 3. Attention（拼接 RoPE 部分）
        # ============================================================
        # 注意: 真正的 MLA 中，k 和 q 是 [k_c; k_r] 的拼接，
        # 但为了演示吸收矩阵，我们先做拆头 Attention
        # 这里简化: 把 RoPE 部分当成额外的 head 维度拼接
        d_total = self.d_k + self.d_kv_rope
        scores = np.zeros((self.num_heads, seq_len, seq_len))

        for h in range(self.num_heads):
            # K 向量 = K 内容 + RoPE K
            k_h = np.concatenate([k_heads[h], k_r], axis=-1)  # (seq_len, d_k + d_kv_rope)
            q_h = np.concatenate([q_heads[h], q_r], axis=-1)  # (seq_len, d_k + d_kv_rope)
            scores[h] = (q_h @ k_h.T) / np.sqrt(d_total)

        if use_mask:
            mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
            scores = scores + mask

        attn_weights = softmax(scores)

        # V 不需要 RoPE 拼接
        head_outputs = attn_weights @ v_heads  # (num_heads, seq_len, d_k)

        # 合并 + 输出投影
        combined = head_outputs.transpose(1, 0, 2).reshape(seq_len, -1)
        return combined @ self.Wo

    # ============================================================
    # 4. 带 KV Cache 的推理
    # ============================================================
    def forward_with_cache(self, x_step, c_kv_cache=None, k_r_cache=None, positions=None):
        """
        带 KV Cache 的单步推理

        与传统 KV Cache 的关键区别：
          传统: 缓存 K 和 V 本身 → 每次缓存量大
          MLA:  缓存 c^{KV}（低维压缩） + k^R（小维度 RoPE Key）→ 缓存量极小

        参数:
            x_step: (1, d_model) 当前 token
            c_kv_cache: (cache_len, d_c) 缓存的压缩向量
            k_r_cache: (cache_len, d_kv_rope) 缓存的 RoPE Key
            positions: 位置索引
        返回:
            output: (1, d_model)
            c_kv_cache: 更新后的缓存
            k_r_cache: 更新后的缓存
        """
        if positions is None:
            positions = np.array([0 if c_kv_cache is None else c_kv_cache.shape[0]])

        # 当前 token 的压缩向量
        c_kv = x_step @ self.W_dkv  # (1, d_c)

        # 当前 token 的 RoPE Key
        k_r = x_step @ self.W_kr  # (1, d_kv_rope)

        # 更新缓存
        if c_kv_cache is None:
            c_kv_cache = c_kv
            k_r_cache = k_r
        else:
            c_kv_cache = np.concatenate([c_kv_cache, c_kv], axis=0)
            k_r_cache = np.concatenate([k_r_cache, k_r], axis=0)

        # 从压缩缓存计算 K 和 V
        k_full = c_kv_cache @ self.W_uk  # (cache_len, d_model)
        v_full = c_kv_cache @ self.W_uv  # (cache_len, d_model)

        # 应用 RoPE 到缓存的 k_r_cache
        from .rotary import apply_rotary
        pos_all = np.arange(c_kv_cache.shape[0])
        k_r_rotated = apply_rotary(k_r_cache, self._cos_table, self._sin_table, pos_all)

        # Q
        q_c = x_step @ self.Wq  # (1, d_model)
        q_r = x_step @ self.W_qr  # (1, d_kv_rope)
        q_r = apply_rotary(q_r, self._cos_table, self._sin_table, positions)

        # 拆头（当前只有 1 个 token）
        seq_len = 1
        q_heads = q_c.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        k_heads = k_full.reshape(-1, self.num_heads, self.d_k).transpose(1, 0, 2)
        v_heads = v_full.reshape(-1, self.num_heads, self.d_k).transpose(1, 0, 2)

        # Attention
        d_total = self.d_k + self.d_kv_rope
        scores = np.zeros((self.num_heads, seq_len, k_full.shape[0]))
        for h in range(self.num_heads):
            k_h = np.concatenate([k_heads[h], k_r_rotated], axis=-1)
            q_h = np.concatenate([q_heads[h], q_r], axis=-1)
            scores[h] = (q_h @ k_h.T) / np.sqrt(d_total)

        attn_weights = softmax(scores)
        head_outputs = attn_weights @ v_heads
        combined = head_outputs.transpose(1, 0, 2).reshape(seq_len, -1)
        output = combined @ self.Wo

        return output, c_kv_cache, k_r_cache


# ============================================================
# 演示
# ============================================================
def demo_mla_mechanism():
    """MLA 核心机制演示"""
    print("=" * 60)
    print("MLA 核心机制：K/V 压缩到潜空间")
    print("=" * 60)

    d_model = 16
    num_heads = 4
    d_k = d_model // num_heads  # 4
    d_c = 6    # 压缩维度（远小于 d_model=16）
    d_kv_rope = 4

    mla = MultiHeadLatentAttention(d_model, num_heads, d_k, d_c, d_kv_rope)

    print(f"\n配置:")
    print(f"  d_model:     {d_model}")
    print(f"  num_heads:   {num_heads}")
    print(f"  d_k:         {d_k}")
    print(f"  d_c (压缩):  {d_c}  (原来 K/V = {d_model} 维 → 压缩到 {d_c} 维)")
    print(f"  d_kv_rope:   {d_kv_rope}")
    print(f"  压缩比:      K/V = {d_model}x → c = {d_c}x, 缓存减小 {d_model/d_c:.1f}x")

    # 验证前向
    np.random.seed(42)
    x = np.random.randn(3, d_model)
    out = mla.forward(x, use_mask=True)
    print(f"\n前向验证:")
    print(f"  输入: (3, {d_model})")
    print(f"  输出: {out.shape}")
    print(f"  稳定: {np.all(np.isfinite(out))}")


def demo_kv_cache_comparison():
    """对比 MHA / GQA / MLA 的 KV Cache"""
    print("\n" + "=" * 60)
    print("KV Cache 占用对比（单层，单次推理，seq_len=4096, FP16）")
    print("=" * 60)

    # DeepSeek V2 实际参数
    d_model = 5120
    num_heads = 128
    d_k = 128
    d_c = 512        # 压缩维度
    d_kv_rope = 64   # RoPE 维度
    seq_len = 4096
    layers = 60
    bytes_per = 2    # FP16

    print(f"\n{'方法':>25} | {'单层 KV Cache':>16} | {'60 层总计':>16} | {'压缩比':>10}")
    print("-" * 75)

    # MHA
    mha = 2 * num_heads * seq_len * d_k * bytes_per / (1024**3)
    # GQA (8 KV heads)
    gqa = 2 * 8 * seq_len * d_k * bytes_per / (1024**3)
    # MQA
    mqa = 2 * 1 * seq_len * d_k * bytes_per / (1024**3)
    # MLA (c^{KV} + k^R only)
    mla_single = (d_c + d_kv_rope) * seq_len * bytes_per / (1024**3)

    for name, single, ratio_text in [
        ("MHA", mha, "1x"),
        ("GQA (8 KV heads)", gqa, f"{gqa/mha:.0%}"),
        ("MQA  (1 KV head)", mqa, f"{mqa/mha:.0%}"),
        ("MLA (DeepSeek V2)", mla_single, f"{mla_single/mha:.1%}"),
    ]:
        layers_total = single * layers
        ratio = single / mha
        print(f"{name:>25} | {single:>8.2f} GB | {layers_total:>8.2f} GB | {ratio_text:>10}")

    print(f"\n{'='*60}")
    print(f"MLA 每步缓存内容: c_kv ({d_c}维) + k_r ({d_kv_rope}维) = {d_c + d_kv_rope} 维")
    print(f"MHA 每步缓存内容: K ({num_heads*d_k}维) + V ({num_heads*d_k}维) = {2*num_heads*d_k} 维")
    print(f"缓存量降至 {(d_c + d_kv_rope) / (2*num_heads*d_k):.2%}")


def demo_absorbed_weights():
    """演示吸收矩阵技巧"""
    print("\n" + "=" * 60)
    print("吸收矩阵技巧（推理优化）")
    print("=" * 60)
    print("""
    推理时，Q × (W^{UK} × c^{KV}) = (Q × W^{UK}) × c^{KV}
    
    解释：
    1. 传统做法：先从压缩缓存 c^{KV} 通过 W^{UK} 解压出完整的 K，
       再用 Q 与其相乘 — 这涉及大矩阵乘法 (d_model × d_c)
    
    2. 吸收矩阵：将 W^{UK} 合并到 Q 投影矩阵中，
       绕过显式解压步骤，直接在压缩空间算 Attention
    
    3. 合并后：
       Q_{absorbed} = W^{UQ} × W^{UK}
       即 Q_head × W^{UK} → 可以直接与 c^{KV} 相乘
       省掉了将 c^{KV} 解压为 K 的浮点运算
    
    效果：
    - 推理时无额外的升维计算
    - 压缩向量 c^{KV} 直接参与 Attention 计算
    - v 的升维 W^{UV} 同理可吸收到输出投影 Wo 中
    """)


if __name__ == "__main__":
    print("MLA（Multi-head Latent Attention）演示")
    print()

    demo_mla_mechanism()
    demo_kv_cache_comparison()
    demo_absorbed_weights()

    # KV Cache 推理验证
    print("\n" + "=" * 60)
    print("MLA 带 KV Cache 的推理验证")
    print("=" * 60)
    np.random.seed(42)
    mla = MultiHeadLatentAttention(d_model=8, num_heads=2, d_k=4, d_c=3, d_kv_rope=2)

    # 模拟 3 步自回归生成
    c_kv_cache, k_r_cache = None, None
    for step in range(3):
        x_in = np.random.randn(1, 8)
        out, c_kv_cache, k_r_cache = mla.forward_with_cache(
            x_in, c_kv_cache, k_r_cache, positions=np.array([step])
        )
        print(f"  步 {step}: 缓存 c_kv={c_kv_cache.shape}, k_r={k_r_cache.shape}, 输出稳定={np.all(np.isfinite(out))}")
