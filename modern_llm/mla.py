"""
Multi-head Latent Attention (MLA) — 多头潜注意力

DeepSeek V2/V3 的核心创新。将 K/V 压缩到低维潜空间，KV Cache 降至 ~2%。

核心思路：
  MHA: 缓存 K (d_model 维) + V (d_model 维)
  MLA: 缓存 c_kv (d_c 维, 压缩) + k_r (d_kv_rope 维, 小 RoPE Key)

  DeepSeek V2 实际参数 (d_model=5120):
    MHA 每步缓存: 2 × 5120 = 10240 维
    MLA 每步缓存: 512 + 64 = 576 维  → 压缩比 18x

吸收矩阵技巧：
  推理时，解压步骤可以省略：
    Q · (W_uk · c_kv) = (Q · W_uk) · c_kv
  W_uk 被吸收到 Q 投影中，不增加推理计算量。

参考: DeepSeek V2 (https://arxiv.org/abs/2405.04434)
"""
import numpy as np
from .utils import softmax
from .rotary import precompute_rotary_frequencies, apply_rotary


class MultiHeadLatentAttention:
    """
    多头潜注意力

    参数:
        d_model: 模型维度
        num_heads: Q 头数
        d_k: 每个 head 的维度
        d_c: 潜空间维度（压缩后的 K/V 维度）
        d_kv_rope: RoPE 部分的维度
        max_seq_len: 最大序列长度
    """
    def __init__(self, d_model, num_heads, d_k, d_c, d_kv_rope=32, max_seq_len=128):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_c = d_c
        self.d_kv_rope = d_kv_rope

        self.Wq = np.random.randn(d_model, d_model) * 0.01
        self.W_qr = np.random.randn(d_model, d_kv_rope) * 0.01

        self.W_dkv = np.random.randn(d_model, d_c) * 0.01
        self.W_uk = np.random.randn(d_c, d_model) * 0.01
        self.W_kr = np.random.randn(d_model, d_kv_rope) * 0.01
        self.W_uv = np.random.randn(d_c, d_model) * 0.01
        self.Wo = np.random.randn(d_model, d_model) * 0.01

        self._cos_table, self._sin_table = precompute_rotary_frequencies(
            d_kv_rope, max_seq_len=max_seq_len
        )

    def forward(self, x, use_mask=True, positions=None):
        seq_len = x.shape[0]
        if positions is None:
            positions = np.arange(seq_len)

        # 1. Q 投影 + 拆头
        q_c = x @ self.Wq
        q_r = apply_rotary(x @ self.W_qr, self._cos_table, self._sin_table, positions)
        q_heads = q_c.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

        # 2. K/V 压缩 → 升维
        c_kv = x @ self.W_dkv               # (seq_len, d_c)
        k_c = c_kv @ self.W_uk              # (seq_len, d_model)
        k_r = apply_rotary(x @ self.W_kr, self._cos_table, self._sin_table, positions)
        v = c_kv @ self.W_uv                 # (seq_len, d_model)

        k_heads = k_c.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        v_heads = v.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

        # 3. Attention（Q/K 拼接 RoPE 部分）
        d_total = self.d_k + self.d_kv_rope
        scores = np.zeros((self.num_heads, seq_len, seq_len))
        for h in range(self.num_heads):
            q_h = np.concatenate([q_heads[h], q_r], axis=-1)
            k_h = np.concatenate([k_heads[h], k_r], axis=-1)
            scores[h] = (q_h @ k_h.T) / np.sqrt(d_total)

        if use_mask:
            scores += np.triu(np.full((seq_len, seq_len), -1e9), k=1)

        attn_weights = softmax(scores)
        head_outputs = attn_weights @ v_heads

        combined = head_outputs.transpose(1, 0, 2).reshape(seq_len, -1)
        return combined @ self.Wo

    def forward_with_cache(self, x_step, c_kv_cache=None, k_r_cache=None, positions=None):
        """带 KV Cache 的单步推理（缓存 c_kv + k_r 而非完整 K/V）"""
        if positions is None:
            positions = np.array([0 if c_kv_cache is None else c_kv_cache.shape[0]])

        c_kv = x_step @ self.W_dkv
        k_r = x_step @ self.W_kr

        if c_kv_cache is None:
            c_kv_cache, k_r_cache = c_kv, k_r
        else:
            c_kv_cache = np.concatenate([c_kv_cache, c_kv], axis=0)
            k_r_cache = np.concatenate([k_r_cache, k_r], axis=0)

        # 从压缩缓存解压 K 和 V
        k_full = c_kv_cache @ self.W_uk
        v_full = c_kv_cache @ self.W_uv

        pos_all = np.arange(c_kv_cache.shape[0])
        k_r_rotated = apply_rotary(k_r_cache, self._cos_table, self._sin_table, pos_all)

        q_c = x_step @ self.Wq
        q_r = apply_rotary(x_step @ self.W_qr, self._cos_table, self._sin_table, positions)

        seq_len = 1
        cache_len = c_kv_cache.shape[0]
        q_heads = q_c.reshape(1, self.num_heads, self.d_k).transpose(1, 0, 2)
        k_heads = k_full.reshape(cache_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        v_heads = v_full.reshape(cache_len, self.num_heads, self.d_k).transpose(1, 0, 2)

        d_total = self.d_k + self.d_kv_rope
        scores = np.zeros((self.num_heads, 1, cache_len))
        for h in range(self.num_heads):
            q_h = np.concatenate([q_heads[h], q_r], axis=-1)
            k_h = np.concatenate([k_heads[h], k_r_rotated], axis=-1)
            scores[h] = (q_h @ k_h.T) / np.sqrt(d_total)

        attn_weights = softmax(scores)
        head_outputs = attn_weights @ v_heads
        combined = head_outputs.transpose(1, 0, 2).reshape(1, -1)

        return combined @ self.Wo, c_kv_cache, k_r_cache
