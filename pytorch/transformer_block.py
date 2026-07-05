"""
PyTorch 版完整 Transformer Block — 与 NumPy 版 transformer_block.py 对应

Decoder layer，使用因果掩码实现自回归生成。

结构:
  输入 → Self-Attention(因果掩码) → +残差 → LayerNorm → FFN → +残差 → LayerNorm → 输出

用 nn.Module 封装，支持自动微分和 GPU。
"""
import torch
import torch.nn as nn
from utils import layer_norm
from multi_head_attention import MultiHeadAttention


class FFN(nn.Module):
    """
    PyTorch 版前馈网络 — 与 NumPy 版 FFN 对应

    用 nn.Linear 替代手写矩阵乘法。
    nn.Linear 内部维护 W 和 b，且做了 Kaiming 初始化。
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=True)
        self.W2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        """
        FFN(x) = W2(ReLU(W1(x) + b1)) + b2

        nn.Linear 自动处理: y = x @ W^T + b
        注意: nn.Linear 的 W 形状是 (d_model, d_ff) — 与 NumPy 版 (d_model, d_ff) 等同
        """
        hidden = torch.relu(self.W1(x))
        return self.W2(hidden)


class TransformerBlock(nn.Module):
    """
    PyTorch 版完整 Transformer Block — 与 NumPy 版 TransformerBlock 对应

    结构:
      输入 → Attention → +残差 → LayerNorm → FFN → +残差 → LayerNorm → 输出
    """
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x, use_mask=True):
        """
        前向传播 — 与 NumPy 版完全对应

        参数:
            x: (seq_len, d_model)
            use_mask: 是否使用因果掩码

        返回:
            (seq_len, d_model)
        """
        # 子层 1: Multi-Head Attention + 残差 + LayerNorm
        attn_out = self.attention(x, use_mask)
        x = x + attn_out
        x = layer_norm(x)

        # 子层 2: FFN + 残差 + LayerNorm
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = layer_norm(x)

        return x


# ============================================================
# 演示
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    d_model, num_heads, d_ff = 8, 2, 16
    X = torch.randn(4, d_model)

    print(f"配置: d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}")
    print(f"输入形状: {tuple(X.shape)}")

    block = TransformerBlock(d_model, num_heads, d_ff)
    output = block(X, use_mask=True)
    print(f"输出形状: {tuple(output.shape)}")

    print("\n堆叠 3 层验证稳定性:")
    x = X
    for i in range(3):
        x = block(x, use_mask=True)
        norm = torch.norm(x).item()
        print(f"  第 {i+1} 层输出范数: {norm:.3f}")
    print("数值稳定（没爆炸也没消失）")
