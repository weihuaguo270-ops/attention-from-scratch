"""
Multi-Head Self-Attention — 模块版

多头注意力的核心思想:
  单头 Attention 只从"一个角度"看句子。
  多头 Attention 从"多个角度"同时看——一个头关注语法、一个头关注语义等。

  实现方式:
  1. 整体投影: 输入 × Wq/Wk/Wv → 映射到 d_model 维
  2. split_heads: 拆成 num_heads 份，每份 d_k 维
  3. 所有头并行算 Attention
  4. combine_heads: 合并所有头的输出
  5. Wo 投影: 将各头信息混合

  形状变化:
    输入 (seq_len, d_model)
      → 投影后 (seq_len, d_model)
      → 拆头 (num_heads, seq_len, d_k)
      → 每头独立 Attention
      → 合并 (seq_len, d_model)
      → Wo 投影 (seq_len, d_model)

位置编码支持两种模式（由 use_rope 参数控制）:
  use_rope=False (默认): Sinusoidal PE 加到输入上
                            调用者自己加，MHA 不感知
  use_rope=True:           RoPE 在 MHA 内部旋转 Q/K
                            调用者不需要做任何额外处理
"""
import numpy as np
from utils import softmax, split_heads, combine_heads
from rotary import precompute_rotary_frequencies, apply_rotary


class MultiHeadAttention:
    """
    多头自注意力机制

    每个头有自己的 Q/K/V 权重矩阵（其实是整体矩阵的不同分片），
    从不同角度观察句子中的词间关系。

    位置编码:
      - use_rope=False: 调用者自己在输入上加 Sinusoidal PE
      - use_rope=True:  MHA 内部在 Q/K 上应用 RoPE
    """
    def __init__(self, d_model, num_heads, use_rope=False, max_seq_len=128):
        """
        参数:
            d_model: 模型维度（输入输出都是这个维度）
            num_heads: 头数（通常 8 或 16，这里演示用 2）
            use_rope: 是否使用 RoPE 代替 Sinusoidal PE
            max_seq_len: 最大序列长度（仅 RoPE 需要，预计算 cos/sin 表）
        """
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        # Q/K/V 投影矩阵
        self.Wq = np.random.randn(d_model, d_model) * 0.01
        self.Wk = np.random.randn(d_model, d_model) * 0.01
        self.Wv = np.random.randn(d_model, d_model) * 0.01

        # 输出投影矩阵
        self.Wo = np.random.randn(d_model, d_model) * 0.01

        # RoPE 预计算表（use_rope=True 时使用）
        if use_rope:
            self._cos_table, self._sin_table = precompute_rotary_frequencies(
                self.d_k, max_seq_len=max_seq_len
            )

    def forward(self, x, use_mask=True, positions=None):
        """
        前向传播

        流程:
        x → Q/K/V 投影 → 拆头 → [RoPE] → 各头并行 Attention → 合并 → Wo 投影 → 输出

        参数:
            x: 输入矩阵 (seq_len, d_model)
            use_mask: 是否使用因果掩码
            positions: 位置索引数组 (seq_len,)，仅 use_rope=True 时需要
                       默认 None = [0, 1, ..., seq_len-1]

        返回:
            (seq_len, d_model) 的输出矩阵
        """
        seq_len = x.shape[0]

        # Step 1: 整体投影 — 算 Q/K/V
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        # Step 2: 拆头
        Q = split_heads(Q, self.num_heads)  # (num_heads, seq_len, d_k)
        K = split_heads(K, self.num_heads)
        V = split_heads(V, self.num_heads)

        # Step 3: 可选 —— 对 Q 和 K 应用 RoPE
        if self.use_rope:
            if positions is None:
                positions = np.arange(seq_len)
            for h in range(self.num_heads):
                Q[h] = apply_rotary(Q[h], self._cos_table, self._sin_table, positions)
                K[h] = apply_rotary(K[h], self._cos_table, self._sin_table, positions)

        # Step 4: 所有头同时算 Attention
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(self.d_k)

        # Step 5: 因果掩码（可选）
        if use_mask:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
            scores = scores + mask

        # Step 6: Softmax → 注意力权重
        attn_weights = softmax(scores)

        # Step 7: 加权求和
        head_outputs = attn_weights @ V

        # Step 8: 合并所有头的输出
        combined = combine_heads(head_outputs, self.num_heads)

        # Step 9: Wo 输出投影
        return combined @ self.Wo


# ============================================================
# 演示（仅直接运行时执行）
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("多头 Attention 演示（对比 Sinusoidal PE vs RoPE）")
    print("=" * 60)

    X = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
    ])
    d_model = 4
    num_heads = 2
    seq_len = 3

    # --- 模式 A: Sinusoidal PE（默认） ---
    from positional_encoding import sinusoidal_positional_encoding
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    X_sinusoidal = X + pe

    mha_no_rope = MultiHeadAttention(d_model, num_heads, use_rope=False)
    output_no_rope = mha_no_rope.forward(X_sinusoidal, use_mask=False)

    print(f"\n模式 A: Sinusoidal PE（外部加到输入）")
    print(f"  输出:\n{np.round(output_no_rope, 3)}")

    # --- 模式 B: RoPE ---
    mha_rope = MultiHeadAttention(d_model, num_heads, use_rope=True, max_seq_len=10)
    output_rope = mha_rope.forward(X, use_mask=False)  # 不加 Sinusoidal PE

    print(f"\n模式 B: RoPE（MHA 内部旋转 Q/K）")
    print(f"  输出:\n{np.round(output_rope, 3)}")

    print(f"\n配置: {num_heads} 个头, d_model={d_model}, 每头维度={d_model//num_heads}")
    print(f"Sinusoidal: 位置信息在输入层加")
    print(f"RoPE:       位置信息在 Attention 内部的 Q/K 上旋转")
