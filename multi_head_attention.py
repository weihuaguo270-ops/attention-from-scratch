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
"""
import numpy as np
from utils import softmax, split_heads, combine_heads


class MultiHeadAttention:
    """
    多头自注意力机制

    每个头有自己的 Q/K/V 权重矩阵（其实是整体矩阵的不同分片），
    从不同角度观察句子中的词间关系。
    """
    def __init__(self, d_model, num_heads):
        """
        参数:
            d_model: 模型维度（输入输出都是这个维度）
            num_heads: 头数（通常 8 或 16，这里演示用 2）

        每个头维度 d_k = d_model // num_heads
        要求 d_model 能被 num_heads 整除
        """
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # Q/K/V 投影矩阵: 从 d_model 映射到 d_model
        # 注意: 这里只有一个整体矩阵，后面拆成多份
        # 等价于每个头有自己的 Wq，但写法上更高效
        self.Wq = np.random.randn(d_model, d_model) * 0.01
        self.Wk = np.random.randn(d_model, d_model) * 0.01
        self.Wv = np.random.randn(d_model, d_model) * 0.01

        # 输出投影矩阵: 混合各头信息
        self.Wo = np.random.randn(d_model, d_model) * 0.01

    def forward(self, x, use_mask=True):
        """
        前向传播

        流程:
        x → Q/K/V 投影 → 拆头 → 各头并行 Attention → 合并 → Wo 投影 → 输出

        参数:
            x: 输入矩阵 (seq_len, d_model)
            use_mask: 是否使用因果掩码

        返回:
            (seq_len, d_model) 的输出矩阵
        """
        seq_len = x.shape[0]

        # Step 1: 整体投影 — 算 Q/K/V
        # 所有头共享一个投影矩阵，后续拆开
        Q = split_heads(x @ self.Wq, self.num_heads)  # (num_heads, seq_len, d_k)
        K = split_heads(x @ self.Wk, self.num_heads)  # (num_heads, seq_len, d_k)
        V = split_heads(x @ self.Wv, self.num_heads)  # (num_heads, seq_len, d_k)

        # Step 2: 所有头同时算 Attention
        # Q @ K^T: (num_heads, seq_len, d_k) @ (num_heads, d_k, seq_len) → (num_heads, seq_len, seq_len)
        # 一次矩阵乘法 = 所有头同时算完，不用循环
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(self.d_k)

        # Step 3: 因果掩码（可选）
        # 如果需要自回归生成，遮住未来位置
        if use_mask:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
            scores = scores + mask  # 广播到所有头

        # Step 4: Softmax → 注意力权重
        attn_weights = softmax(scores)

        # Step 5: 加权求和 — 每头独立算出输出
        head_outputs = attn_weights @ V  # (num_heads, seq_len, d_k)

        # Step 6: 合并所有头的输出
        combined = combine_heads(head_outputs, self.num_heads)  # (seq_len, d_model)

        # Step 7: Wo 输出投影 — 混合各头信息
        # 没有 Wo 的话，各头信息是并排放置、互不交流的
        # Wo 让各头的发现可以互相影响
        return combined @ self.Wo


# ============================================================
# 演示（仅直接运行时执行）
# ============================================================
if __name__ == "__main__":
    # 输入: 3 个词，每个 4 维
    X = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
    ])
    d_model = 4
    num_heads = 2

    mha = MultiHeadAttention(d_model, num_heads)
    output = mha.forward(X, use_mask=False)

    print("多头 Attention 输出:")
    print(np.round(output, 3))
    print(f"\n配置: {num_heads} 个头, d_model={d_model}, 每头维度={d_model//num_heads}")
