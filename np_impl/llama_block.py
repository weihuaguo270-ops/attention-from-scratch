"""
完整 Llama 风格 Decoder Block — 纯 NumPy 实现

Llama 架构 vs 原始 Transformer 的关键差异：

                原始 Transformer (Vaswani 2017)    Llama 系列 (2023-2024)
  ─────────────────────────────────────────────────────────────────────
  LayerNorm       Post-Norm (残差后归一化)        Pre-Norm (归一化后进子层)
  归一化方式      LayerNorm (均值和标准差)          RMSNorm (只有标准差)
  FFN 激活函数    ReLU                              SwiGLU (门控激活)
  Attention       MHA                               GQA (Grouped Query)
  位置编码        Sinusoidal PE                      RoPE (旋转位置编码)
  Dropout         内置 Dropout                       无 / 极少 Dropout

结构:
  输入
    ↓
  RMSNorm                           ← Pre-Norm：先归一化
    ↓
  GQA (RoPE + 因果掩码)             ← 分组查询注意力
    ↓
  + 残差连接
    ↓
  RMSNorm                           ← Pre-Norm
    ↓
  SwiGLU FFN                        ← 门控激活函数
    ↓
  + 残差连接
    ↓
  输出

"Pre-Norm" 为什么更好？
  Post-Norm: LayerNorm(子层输出 + 残差) — 梯度信号弱，深层不稳定
  Pre-Norm:  子层(LayerNorm(输入)) + 残差 — 梯度直通残差路径，深层训练稳定
  结果: Llama 可以堆叠 70B 参数（65+ 层）而不炸
"""
import sys, os
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "np_impl"
import numpy as np


# ============================================================
# 1. RMSNorm — 均方根归一化
# ============================================================
class RMSNorm:
    """
    RMSNorm (Root Mean Square Normalization)

    公式: RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    与 LayerNorm 的区别：
      LayerNorm:  (x - mean) / std × weight + bias  ← 减均值、除标准差、加偏置
      RMSNorm:     x / rms(x) × weight               ← 只除标准差，不减均值，无偏置

    为什么 Llama 用 RMSNorm？
      1. 计算更快（少算了均值）
      2. 实验效果与 LayerNorm 相当或更好
      3. bias 在现代 LLM 中已证实用处不大

    参数:
        d_model: 模型维度
        eps: 防止除零的小常数
    """
    def __init__(self, d_model, eps=1e-6):
        self.weight = np.ones(d_model) * 0.1  # 可学习的缩放参数
        self.eps = eps

    def forward(self, x):
        """
        x: (seq_len, d_model) 或 (batch, seq_len, d_model)
        返回: 与 x 同 shape
        """
        # RMS = sqrt(mean(x^2))
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


# ============================================================
# 2. SwiGLU — 门控激活的 FFN
# ============================================================
class SwiGLU:
    """
    SwiGLU 激活函数

    传统 FFN: FFN(x) = W_down * ReLU(W_up * x)
    SwiGLU:  FFN(x) = W_down * (Swish(W_gate * x) * (W_up * x))

    多了一个 W_gate 矩阵用于「门控」：
      - W_gate 的输出经过 Swish 激活 → [0~1] 的开关信号
      - W_up 的输出作为「内容」
      - 两者逐元素相乘 → 选择性的信息传递

    Swish 函数: Swish(x) = x * sigmoid(x)
    Swish 与 GELU 形状相似，但 Llama 选择 Swish（更简单）

    为什么用 SwiGLU 而不是 ReLU？
      1. 门控机制让网络可以「选择」激活哪部分信息
      2. Swish 在负半轴不平为零 → 保留微弱负信号
      3. 相同参数下，SwiGLU 比 ReLU 效果好，且训练更稳定
      4. 代价：3个矩阵 (gate/up/down) vs 原始FFN的2个，参数多了50%

    参数:
        d_model: 输入输出维度
        d_ff: 中间隐藏维度（通常 8/3 * d_model，因为 gate+up 都会投影到 d_ff）
    """
    def __init__(self, d_model, d_ff):
        # SwiGLU 有三个权重矩阵
        # W_gate: 门控信号 (d_model, d_ff)
        # W_up:   内容投影 (d_model, d_ff)
        # W_down: 输出投影 (d_ff, d_model)
        self.W_gate = np.random.randn(d_model, d_ff) * 0.01
        self.W_up = np.random.randn(d_model, d_ff) * 0.01
        self.W_down = np.random.randn(d_ff, d_model) * 0.01

    def swish(self, x):
        """Swish 激活函数: x * sigmoid(x)"""
        return x * (1.0 / (1.0 + np.exp(-x)))

    def forward(self, x):
        """
        x: (seq_len, d_model)
        返回: (seq_len, d_model)
        """
        # 1. 门控路径
        gate = self.W_gate.T @ x.T  # (d_ff, seq_len)
        gate = gate.T                # (seq_len, d_ff)
        gate = self.swish(gate)      # (seq_len, d_ff)  ← 门控信号

        # 2. 内容路径
        up = self.W_up.T @ x.T       # (d_ff, seq_len)
        up = up.T                     # (seq_len, d_ff)

        # 3. 门控 × 内容
        gated = gate * up            # (seq_len, d_ff)  ← 选择性激活

        # 4. 输出投影
        out = self.W_down.T @ gated.T  # (d_model, seq_len)
        return out.T                   # (seq_len, d_model)


# ============================================================
# 3. Llama Decoder Block
# ============================================================
class LlamaDecoderBlock:
    """
    完整的 Llama 风格 Decoder Block

    结构:
      x → RMSNorm → GQA (RoPE + 因果掩码) → +残差 → RMSNorm → SwiGLU → +残差 → 输出

    与原始 TransformerBlock 的对比：
      ┌──────────────────┬────────────────────┐
      │ TransformerBlock │ LlamaDecoderBlock  │
      ├──────────────────┼────────────────────┤
      │ Post-Norm        │ Pre-Norm           │
      │ LayerNorm        │ RMSNorm            │
      │ ReLU FFN         │ SwiGLU FFN         │
      │ MHA              │ GQA                │
      │ Sinusoidal PE    │ RoPE               │
      │ 可选 Dropout     │ 无 Dropout         │
      └──────────────────┴────────────────────┘

    参数:
        d_model: 模型维度
        num_heads: Q 头数
        num_kv_heads: K/V 头数（GQA）
        d_ff: FFN 中间维度
        use_rope: 是否使用 RoPE
        max_seq_len: 最大序列长度
    """
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff,
                 use_rope=True, max_seq_len=128):
        from .gqa import GroupedQueryAttention

        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)
        self.self_attn = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
        )
        self.swiglu = SwiGLU(d_model, d_ff)

    def forward(self, x, use_mask=True, positions=None):
        """
        前向传播

        Pre-Norm 结构:
          x = x + 子层(RMSNorm(x))
          梯度可以绕过子层，直接沿残差路径传播 → 深层训练稳定

        参数:
            x: (seq_len, d_model)
            use_mask: 是否使用因果掩码
            positions: RoPE 位置索引
        返回:
            (seq_len, d_model)
        """
        # Pre-Norm Attention
        residual = x
        x = self.rmsnorm1.forward(x)
        x = self.self_attn.forward(x, use_mask=use_mask, positions=positions)
        x = x + residual

        # Pre-Norm SwiGLU FFN
        residual = x
        x = self.rmsnorm2.forward(x)
        x = self.swiglu.forward(x)
        x = x + residual

        return x


# ============================================================
# 4. 多层堆叠
# ============================================================
class LlamaModel:
    """
    多层 Llama Decoder 堆叠

    参数:
        num_layers: 层数
        d_model: 模型维度
        num_heads: Q 头数
        num_kv_heads: K/V 头数
        d_ff: FFN 维度
        max_seq_len: 最大序列长度
    """
    def __init__(self, num_layers=4, d_model=8, num_heads=4, num_kv_heads=2,
                 d_ff=32, max_seq_len=128):
        self.layers = [
            LlamaDecoderBlock(d_model, num_heads, num_kv_heads, d_ff,
                              use_rope=True, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ]

    def forward(self, x, use_mask=True, positions=None):
        for layer in self.layers:
            x = layer.forward(x, use_mask=use_mask, positions=positions)
        return x


# ============================================================
# 演示：逐模块验证
# ============================================================
def demo_rmsnorm():
    """演示 RMSNorm 与 LayerNorm 的差异"""
    print("=" * 60)
    print("RMSNorm 演示")
    print("=" * 60)

    rms = RMSNorm(8)
    x = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ])
    out = rms.forward(x)

    # RMS 应该接近 1
    rms_val = np.sqrt(np.mean(out ** 2, axis=-1, keepdims=True))
    print(f"\n输入: {x.shape}")
    print(f"输出: {np.round(out, 4)}")
    print(f"每行 RMS: {np.round(rms_val.flatten(), 6)} (应 ≈ {np.round(rms.weight[0], 4)})")
    print(f"✅ 没有减均值（LayerNorm 会做）：mean ≈ {np.round(out.mean(axis=-1), 6)}")


def demo_swiglu():
    """演示 SwiGLU 与 ReLU FFN 的差异"""
    print("\n" + "=" * 60)
    print("SwiGLU 激活演示")
    print("=" * 60)

    d_model, d_ff = 4, 8
    swiglu = SwiGLU(d_model, d_ff)

    x = np.array([
        [1.0, 0.5, -1.0, -2.0],  # 混合正负输入
        [-1.0, -2.0, 1.0, 0.5],
    ])
    out = swiglu.forward(x)

    print(f"\n输入: {x.shape}")
    print(f"  {np.round(x, 3)}")
    print(f"输出: {out.shape}")
    print(f"  {np.round(out, 3)}")

    # 验证：负值是否被封杀
    gate = swiglu.swish(x @ swiglu.W_gate)
    print(f"\n门控信号 (gate) — 前 4 列:")
    print(f"  {np.round(gate[:, :4], 3)}")
    # 检查 gate 中是否有接近于 0 的值（对应 x 的负值）
    neg_mask = x[:, :1] < 0  # x 第一列为负的那些行
    if neg_mask.any():
        print(f"  负输入的对应 gate: {np.round(gate[neg_mask.flatten(), :4], 3)}")


def demo_llama_block():
    """演示完整的 Llama Block"""
    print("\n" + "=" * 60)
    print("LlamaDecoderBlock 完整演示")
    print("=" * 60)

    d_model, num_heads, num_kv_heads, d_ff = 8, 4, 2, 16
    block = LlamaDecoderBlock(d_model, num_heads, num_kv_heads, d_ff, use_rope=True)

    np.random.seed(42)
    x = np.random.randn(4, d_model)

    print(f"\n输入: (4, 8)")
    out = block.forward(x, use_mask=True)
    print(f"输出: {out.shape}")
    print(f"  稳定: {np.all(np.isfinite(out))}")
    print(f"  非零: {np.linalg.norm(out) > 0}")

    # 多层堆叠
    model = LlamaModel(num_layers=3, d_model=8, num_heads=4, num_kv_heads=2, d_ff=16)
    out_multi = model.forward(x, use_mask=True)
    print(f"\n3 层堆叠:")
    print(f"  输出: {out_multi.shape}")
    print(f"  稳定: {np.all(np.isfinite(out_multi))}")


# ============================================================
# 5. Llama vs 原始 Transformer 架构对比
# ============================================================
def architecture_comparison():
    """罗列所有架构差异，形成参考表"""
    print("\n" + "=" * 60)
    print("Llama vs 原始 Transformer 架构对比")
    print("=" * 60)
    print(f"""
{'维度':>20} | {'Transformer (2017)':>28} | {'Llama (2023-24)':>28}
{'-'*80}
{'归一化位置':>20} | {'Post-Norm':>28} | {'Pre-Norm':>28}
{'归一化类型':>20} | {'LayerNorm':>28} | {'RMSNorm (~30%更快)':>28}
{'FFN 激活':>20} | {'ReLU':>28} | {'SwiGLU (门控)':>28}
{'Attention':>20} | {'MHA':>28} | {'GQA (节省80% KV Cache)':>28}
{'位置编码':>20} | {'Sinusoidal PE':>28} | {'RoPE (外推能力)':>28}
{'Dropout':>20} | {'内置':>28} | {'极少/无':>28}
{'归一化公式':>20} | {'(x-μ)/σ×γ+β':>28} | {'x/rms(x)×γ (无μ,无β)':>28}
{'FFN 参数':>20} | {'2个矩阵: W_up, W_down':>28} | {'3个矩阵: W_gate, W_up, W_down':>28}
{'FFN 计算':>20} | {'W_down·ReLU(W_up·x)':>28} | {'W_down·(Swish(W_gate·x)⊙W_up·x)':>28}
    """)


if __name__ == "__main__":
    print("Llama 架构 — RMSNorm + SwiGLU + GQA + RoPE + Pre-Norm")
    print()

    demo_rmsnorm()
    demo_swiglu()
    demo_llama_block()
    architecture_comparison()
