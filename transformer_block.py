"""
完整 Transformer Block — 纯 NumPy 实现

┌─────────────────────────────────────────────────────────┐
│ 这是一个完整的 Transformer 层（Decoder layer）           │
│                                                         │
│ 输入 → Multi-Head Self-Attention → +残差 → LayerNorm    │
│                                      → FFN → +残差 → LayerNorm → 输出
│                                                         │
│ 每个子层的结构:                                          │
│   sublayer(x) = LayerNorm(x + Sublayer(x))              │
│                 ↑ 残差          ↑ Attention 或 FFN      │
└─────────────────────────────────────────────────────────┘
"""
import numpy as np
from utils import layer_norm
from multi_head_attention import MultiHeadAttention


# ============================================================
# 1. Feed-Forward Network（前馈神经网络）
# ============================================================
class FFN:
    """
    每个词独立地做"深度思考"——两次线性变换 + 一次非线性激活

    形状变化:
      (seq_len, d_model) → @W1 → (seq_len, d_ff) → ReLU → (seq_len, d_ff) → @W2 → (seq_len, d_model)

    设计原理:
      - 先升维（d_model → d_ff，通常是 4 倍）：给模型更多参数空间去表达复杂模式
      - ReLU 激活：引入非线性，否则两层线性=一层线性
      - 再降维（d_ff → d_model）：恢复原维度，方便堆叠后续层

    Attention 做的是"词与词之间的交互"（横向交流），
    FFN 做的是"每个词独立地深加工"（纵向消化）。
    """
    def __init__(self, d_model, d_ff):
        """
        参数:
            d_model: 输入/输出维度
            d_ff: 中间隐藏层维度（通常是 d_model × 4）
        """
        # W1: 从 d_model 升维到 d_ff
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)

        # W2: 从 d_ff 降维回 d_model
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

        # 注意: * 0.01 是让初始权重足够小，防止一开始数值爆炸
        #       偏置初始化为 0，让模型自己决定要不要学出非零偏置

    def forward(self, x):
        """
        前向传播:

        FFN(x) = W2 × ReLU(W1 × x + b1) + b2
        """
        # 第 1 层线性变换：升维 (seq_len, d_model) → (seq_len, d_ff)
        hidden = x @ self.W1 + self.b1

        # ReLU 激活函数：所有负数变 0，正数不变
        # ReLU(x) = max(0, x)
        hidden = np.maximum(0, hidden)

        # 第 2 层线性变换：降维 (seq_len, d_ff) → (seq_len, d_model)
        output = hidden @ self.W2 + self.b2
        return output


# ============================================================
# 2. 完整 Transformer Block
# ============================================================
class TransformerBlock:
    """
    一个完整的 Transformer 层，包含两个子层：

      子层 1: Multi-Head Self-Attention → + 残差连接 → LayerNorm
      子层 2: Feed-Forward Network (FFN) → + 残差连接 → LayerNorm

    每个子层的结构都遵循:
      output = LayerNorm(x + sublayer(x))
                ↑ 残差     ↑ 子层处理

    残差连接（Residual Connection）:
      - 公式: output = input + sublayer(input)
      - 作用: 让梯度可以直接穿过深层网络，防止信息逐层衰减
      - 如果 sublayer 输出全 0，output 至少保留输入值

    层归一化（Layer Normalization）:
      - 公式: LN(x) = (x - mean) / (std + eps)
      - 对每个词的向量独立做标准化，让均值≈0，标准差≈1
      - 让训练更稳定，防止数值过大或过小
    """
    def __init__(self, d_model, num_heads, d_ff):
        """
        参数:
            d_model: 向量维度（所有子层的输入输出都是这个维度）
            num_heads: 多头注意力中的头数
            d_ff: FFN 中间层维度
        """
        # 子层 1: 多头自注意力 — 让每个词看到其他词，融合上下文
        self.attention = MultiHeadAttention(d_model, num_heads)

        # 子层 2: 前馈网络 — 每个词独立深加工
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x, use_mask=True):
        """
        前向传播

        流程:
        输入 x (seq_len, d_model)
          ↓
        ┌──────────────────────────────────────────────┐
        │ 子层 1: Multi-Head Attention                  │
        │   attn_out = Attention(x)                      │
        │   x = x + attn_out         ← 残差连接         │
        │   x = LayerNorm(x)         ← 层归一化         │
        └──────────────────────────────────────────────┘
          ↓
        ┌──────────────────────────────────────────────┐
        │ 子层 2: Feed-Forward Network                   │
        │   ffn_out = FFN(x)                             │
        │   x = x + ffn_out          ← 残差连接         │
        │   x = LayerNorm(x)         ← 层归一化         │
        └──────────────────────────────────────────────┘
          ↓
        输出 x (seq_len, d_model) — 给下一层或预测头

        参数:
            x: 输入矩阵，shape (seq_len, d_model)
            use_mask: 是否使用因果掩码（True=GPT风格, False=BERT风格）

        返回:
            shape (seq_len, d_model) 的输出
        """
        # ═══════════════════════════════════════════════════
        # 子层 1: Multi-Head Self-Attention + 残差 + LayerNorm
        # ═══════════════════════════════════════════════════

        # Attention 输出: 每个词融合了其他词的信息
        attn_out = self.attention.forward(x, use_mask)

        # 残差连接: 保留原始输入，叠加 Attention 学到的新信息
        # 即使 attn_out 全为 0，x 也至少保留原始值
        x = x + attn_out

        # 层归一化: 标准化数值范围，稳定训练
        x = layer_norm(x)

        # ═══════════════════════════════════════════════════
        # 子层 2: FFN + 残差 + LayerNorm
        # ═══════════════════════════════════════════════════

        # FFN 输出: 每个词独立地"消化吸收"Attention 学到的东西
        ffn_out = self.ffn.forward(x)

        # 残差连接
        x = x + ffn_out

        # 层归一化
        x = layer_norm(x)

        return x
        # 输出 shape 跟输入一样 (seq_len, d_model)，所以可以堆叠多层


# ============================================================
# 3. 运行验证
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("完整 Transformer Block 演示")
    print("=" * 60)

    d_model = 8       # 向量维度
    num_heads = 2     # 头数
    d_ff = 16         # FFN 中间维度
    seq_len = 4       # 序列长度（词数）

    np.random.seed(42)
    X = np.random.randn(seq_len, d_model)

    print(f"\n配置: d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}")
    print(f"输入 shape: {X.shape}")

    # 创建一层 Transformer Block
    block = TransformerBlock(d_model, num_heads, d_ff)
    output = block.forward(X, use_mask=True)

    print(f"输出 shape: {output.shape}")
    print("输入输出 shape 一致，所以可以堆叠多层")

    # 堆叠 2 层: 每层的输出作为下一层的输入
    print("\n堆叠 2 层:")
    x = X
    for i in range(2):
        x = block.forward(x, use_mask=True)
        print(f"  第 {i+1} 层输出范数: {np.linalg.norm(x):.3f}")
    print("数值稳定（没爆炸也没消失）= 残差连接 + LayerNorm 在起作用")
