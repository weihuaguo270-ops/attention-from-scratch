"""
Multi-head Latent Attention (MLA) — 多头潜注意力

DeepSeek V2/V3 的核心创新。将 K/V 压缩到低维潜空间，KV Cache 降至 ~2%。

核心思路：
  MHA: 缓存 K (d_model 维) + V (d_model 维)
  MLA: 缓存 c_kv (d_c 维, 压缩) + k_r (d_kv_rope 维, 小 RoPE Key)

吸收矩阵技巧（推理）：
  内容路径上不必逐步解压完整 K/V：
    Q_h · (C · W_UK_h) = (Q_h · W_UK_h^T) · C
    attn · (C · W_UV_h) = (attn · C) · W_UV_h
  把 W_UK / W_UV 吸收进 Q / 输出侧后，注意力直接对潜变量 c 计算。

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
        assert d_model == num_heads * d_k, "d_model 必须等于 num_heads * d_k"
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
        self._absorbed_q = None  # list[(d_model, d_c)]
        self._absorbed_v = None  # list[(d_c, d_k)]

    # ── 吸收矩阵预计算 ──────────────────────────

    def absorb_weights(self):
        """预计算各头的吸收矩阵（Q←W_UK，V←W_UV）。

        AbsQ_h = Wq_h @ W_uk_h.T   → (d_model, d_c)
        AbsV_h = W_uv_h            → (d_c, d_k)  （直接作用在 attn@c 上）
        """
        Wq = self.Wq.reshape(self.d_model, self.num_heads, self.d_k)
        Wuk = self.W_uk.reshape(self.d_c, self.num_heads, self.d_k)
        Wuv = self.W_uv.reshape(self.d_c, self.num_heads, self.d_k)

        self._absorbed_q = []
        self._absorbed_v = []
        for h in range(self.num_heads):
            Wq_h = Wq[:, h, :]          # (d_model, d_k)
            Wuk_h = Wuk[:, h, :]        # (d_c, d_k)
            self._absorbed_q.append(Wq_h @ Wuk_h.T)  # (d_model, d_c)
            self._absorbed_v.append(Wuv[:, h, :])    # (d_c, d_k)
        return self._absorbed_q, self._absorbed_v

    def _ensure_absorbed(self):
        if self._absorbed_q is None or self._absorbed_v is None:
            self.absorb_weights()

    # ── 训练式前向：显式解压 ─────────────────────

    def forward(self, x, use_mask=True, positions=None):
        """全序列前向（解压路径，便于与训练对照）。"""
        seq_len = x.shape[0]
        if positions is None:
            positions = np.arange(seq_len)

        q_c = x @ self.Wq
        q_r = apply_rotary(x @ self.W_qr, self._cos_table, self._sin_table, positions)
        q_heads = q_c.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

        c_kv = x @ self.W_dkv
        k_c = c_kv @ self.W_uk
        k_r = apply_rotary(x @ self.W_kr, self._cos_table, self._sin_table, positions)
        v = c_kv @ self.W_uv

        k_heads = k_c.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        v_heads = v.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

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

    # ── 推理式带缓存：解压 / 吸收 ─────────────────

    def forward_with_cache(
        self,
        x_step,
        c_kv_cache=None,
        k_r_cache=None,
        positions=None,
        use_absorb=False,
    ):
        """带 KV Cache 的单步推理。

        use_absorb=False: 逐步解压完整 K/V（对照路径）
        use_absorb=True:  吸收 W_UK/W_UV，注意力直接作用在 c 上
        """
        if use_absorb:
            return self._forward_with_cache_absorb(
                x_step, c_kv_cache, k_r_cache, positions
            )
        return self._forward_with_cache_decompress(
            x_step, c_kv_cache, k_r_cache, positions
        )

    def _forward_with_cache_decompress(
        self, x_step, c_kv_cache=None, k_r_cache=None, positions=None
    ):
        if positions is None:
            positions = np.array([0 if c_kv_cache is None else c_kv_cache.shape[0]])

        c_kv = x_step @ self.W_dkv
        k_r = x_step @ self.W_kr

        if c_kv_cache is None:
            c_kv_cache, k_r_cache = c_kv, k_r
        else:
            c_kv_cache = np.concatenate([c_kv_cache, c_kv], axis=0)
            k_r_cache = np.concatenate([k_r_cache, k_r], axis=0)

        k_full = c_kv_cache @ self.W_uk
        v_full = c_kv_cache @ self.W_uv

        pos_all = np.arange(c_kv_cache.shape[0])
        k_r_rotated = apply_rotary(k_r_cache, self._cos_table, self._sin_table, pos_all)

        q_c = x_step @ self.Wq
        q_r = apply_rotary(x_step @ self.W_qr, self._cos_table, self._sin_table, positions)

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

    def _forward_with_cache_absorb(
        self, x_step, c_kv_cache=None, k_r_cache=None, positions=None
    ):
        """吸收路径：内容注意力在 d_c 维上计算，再投影回各 head。"""
        self._ensure_absorbed()
        if positions is None:
            positions = np.array([0 if c_kv_cache is None else c_kv_cache.shape[0]])

        c_kv = x_step @ self.W_dkv
        k_r = x_step @ self.W_kr

        if c_kv_cache is None:
            c_kv_cache, k_r_cache = c_kv, k_r
        else:
            c_kv_cache = np.concatenate([c_kv_cache, c_kv], axis=0)
            k_r_cache = np.concatenate([k_r_cache, k_r], axis=0)

        pos_all = np.arange(c_kv_cache.shape[0])
        k_r_rotated = apply_rotary(k_r_cache, self._cos_table, self._sin_table, pos_all)
        q_r = apply_rotary(x_step @ self.W_qr, self._cos_table, self._sin_table, positions)

        cache_len = c_kv_cache.shape[0]
        d_total = self.d_k + self.d_kv_rope
        head_outputs = np.zeros((self.num_heads, 1, self.d_k))

        for h in range(self.num_heads):
            # 内容分： (x @ AbsQ_h) @ c.T  ≡  q_h @ (c @ W_uk_h).T
            q_abs = x_step @ self._absorbed_q[h]          # (1, d_c)
            score_c = q_abs @ c_kv_cache.T                # (1, cache_len)
            score_r = q_r @ k_r_rotated.T                 # (1, cache_len)
            # 与解压路径一致：完整 concat 后统一 /sqrt(d_k + d_rope)
            scores = (score_c + score_r) / np.sqrt(d_total)
            attn = softmax(scores)                        # (1, cache_len)

            # V 吸收： (attn @ c) @ W_uv_h
            latent = attn @ c_kv_cache                    # (1, d_c)
            head_outputs[h] = latent @ self._absorbed_v[h]

        combined = head_outputs.transpose(1, 0, 2).reshape(1, -1)
        return combined @ self.Wo, c_kv_cache, k_r_cache
