"""
Cross-Attention（交叉注意力）— 纯 NumPy 实现

与 Self-Attention 的区别:
  Self-Attention:  Q, K, V 都来自同一个序列
  Cross-Attention: Q 来自一个序列，K, V 来自另一个序列

使用场景（Encoder-Decoder）:
  Encoder 读完整个句子 → 输出 encoder_output
  Decoder 生成 token 时 → Q 来自 decoder, K,V 来自 encoder_output
  → Decoder 知道"当前生成的内容和原句子的哪些部分最相关"

import sys, os
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "np_impl"
形状变化:
  query:       (seq_len_q, d_model)  ← Decoder 的当前层输出
  key_value:   (seq_len_kv, d_model) ← Encoder 的最终输出
  → Q 投影:   (seq_len_q, d_k)
  → K/V 投影: (seq_len_kv, d_k)
  → 注意力分数: (seq_len_q, seq_len_kv)
"""
import sys, os
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "np_impl"
import numpy as np
from .utils import softmax, split_heads, combine_heads


class MultiHeadCrossAttention:
    """
    多头交叉注意力
    Encoder-Decoder 之间的桥梁。
    Decoder 每生成一个词，通过 Cross-Attention 去"看"原句子的每个词。
    与 MultiHeadAttention 的区别:
      1. forward 接受两个参数 (query, key_value) 而非一个 (x)
      2. 没有因果掩码（不需要遮住未来—Encoder 的序列已经完整了）
      3. use_mask 参数仍然保留，但传 True 也没有意义
    """
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Q 投影 — 来自 query 序列
        self.Wq = np.random.randn(d_model, d_model) * 0.01
        # K/V 投影 — 来自 key_value 序列
        self.Wk = np.random.randn(d_model, d_model) * 0.01
        self.Wv = np.random.randn(d_model, d_model) * 0.01
        # 输出投影
        self.Wo = np.random.randn(d_model, d_model) * 0.01
    def forward(self, query, key_value):
        """
        前向传播
        参数:
            query:    Decoder 的当前输出 (seq_len_q, d_model)
            key_value: Encoder 的最终输出 (seq_len_kv, d_model)
        返回:
            (seq_len_q, d_model) — 融合了原句子信息的 Decoder 输出
        """
        seq_len_q = query.shape[0]
        # Q 来自 query（Decoder），K/V 来自 key_value（Encoder）
        Q = split_heads(query @ self.Wq, self.num_heads)
        K = split_heads(key_value @ self.Wk, self.num_heads)
        V = split_heads(key_value @ self.Wv, self.num_heads)
        # Attention 计算 — 注意 K^T 的 seq_len 维度来自 key_value
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        # Cross-Attention 不需要因果掩码
        attn_weights = softmax(scores)
        head_outputs = attn_weights @ V
        combined = combine_heads(head_outputs, self.num_heads)
        return combined @ self.Wo
# ============================================================
# 演示
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    d_model, num_heads = 4, 2
    d_k = d_model // num_heads
    # Encoder 的输出: 原句子有 3 个词
    # 假设原句子 "I love you" 的编码结果
    encoder_output = np.array([
        [1.0, 0.5, 0.0, 0.0],   # "I" 的编码
        [0.5, 1.0, 0.5, 0.0],   # "love" 的编码
        [0.0, 0.5, 1.0, 1.0],   # "you" 的编码
    ])
    print(f"Encoder 输出形状: {encoder_output.shape}")
    print(f"  (3 个词, 每个 {d_model} 维)")
    # Decoder 当前只生成了 2 个词（比如 "我" 和 "爱"）
    decoder_input = np.array([
        [0.1, 0.2, 0.3, 0.4],   # 第1步生成的词
        [0.2, 0.3, 0.4, 0.5],   # 第2步生成的词
    ])
    print(f"\nDecoder 当前输入形状: {decoder_input.shape}")
    print(f"  (已生成 2 个词, 每个 {d_model} 维)")
    # Cross-Attention: Decoder 用自己当前的 Q 去"看"原句子
    cross_attn = MultiHeadCrossAttention(d_model, num_heads)
    output = cross_attn.forward(decoder_input, encoder_output)
    print(f"\nCross-Attention 输出形状: {output.shape}")
    print(f"  (2 个词, 每个 {d_model} 维)")
    print("\n每个 Decoder 词现在包含了原句子的上下文信息。")