"""
PyTorch 版 Cross-Attention（交叉注意力）— 与 NumPy 版对应

Q 来自 query 序列，K/V 来自 key_value 序列。
"""
import torch
import torch.nn as nn
from pytorch.utils import softmax, split_heads, combine_heads


class MultiHeadCrossAttention(nn.Module):
    """
    多头交叉注意力 — PyTorch 版

    Encoder-Decoder 之间的桥梁：
      Decoder 每生成一个词，通过 Cross-Attention 去"看"原句子。
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Wq = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.Wk = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.Wv = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.Wo = nn.Parameter(torch.randn(d_model, d_model) * 0.01)

    def forward(self, query, key_value):
        """
        参数:
            query:    Decoder 当前输出 (seq_len_q, d_model)
            key_value: Encoder 最终输出 (seq_len_kv, d_model)

        Q 从 query 投影（来自 Decoder）
        K/V 从 key_value 投影（来自 Encoder）
        """
        Q = split_heads(query @ self.Wq, self.num_heads)
        K = split_heads(key_value @ self.Wk, self.num_heads)
        V = split_heads(key_value @ self.Wv, self.num_heads)

        scores = (Q @ K.transpose(1, 2)) / (self.d_k ** 0.5)
        attn_weights = softmax(scores)

        head_outputs = attn_weights @ V
        combined = combine_heads(head_outputs, self.num_heads)
        return combined @ self.Wo


if __name__ == "__main__":
    torch.manual_seed(42)
    d_model, num_heads = 4, 2

    encoder_output = torch.tensor([
        [1.0, 0.5, 0.0, 0.0],
        [0.5, 1.0, 0.5, 0.0],
        [0.0, 0.5, 1.0, 1.0],
    ])

    decoder_input = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
    ])

    cross_attn = MultiHeadCrossAttention(d_model, num_heads)
    output = cross_attn(decoder_input, encoder_output)

    print(f"Cross-Attention 输出形状: {tuple(output.shape)}")
    print("Decoder 每个词已融合原句子信息")
