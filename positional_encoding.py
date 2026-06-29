"""
正弦位置编码（Sinusoidal Positional Encoding）

为什么需要位置编码？
  Self-Attention 本身不区分词的先后顺序。
  "猫坐垫子"和"垫子坐猫"在 Attention 眼里分数矩阵的数值完全一样。

位置编码的作用：
  在输入中加入每个词的位置信息，让模型知道词在句子里的次序。

实现方式：
  偶数维（2i）用 sin，奇数维（2i+1）用 cos:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

  不同维度有不同的周期（频率）:
    低维 → 周期短，变化快 → 区分相邻位置
    高维 → 周期长，变化慢 → 区分远距离位置

最终输入 = 词向量 + 位置编码（逐元素相加）
  模型既知道词义（词向量），又知道位置（位置编码）。
"""
import numpy as np


def sinusoidal_positional_encoding(seq_len, d_model):
    """
    生成正弦位置编码矩阵

    参数:
        seq_len: 句子长度（几个词）
        d_model: 向量维度

    返回:
        (seq_len, d_model) 的位置编码矩阵

    关键设计:
      1. 不需要训练，由数学公式直接生成
      2. sin/cos 的结合让模型可以通过线性变换计算相对位置
      3. 不同频率的维度组合为每个位置产生"独特"的编码
    """
    # 初始化全零矩阵，准备填入位置编码值
    pe = np.zeros((seq_len, d_model))

    # pos: 每个词在句子中的位置 [0, 1, 2, ..., seq_len-1]
    # reshape(-1, 1) 升成列向量 (seq_len, 1)，方便广播
    pos = np.arange(seq_len).reshape(-1, 1)

    # i: 维度索引 [0, 1, 2, ..., d_model-1]
    # 偶数维 (0,2,4...) 用 sin，奇数维 (1,3,5...) 用 cos
    i = np.arange(d_model)

    # 计算公式中的分母: 10000^(2i/d_model)
    # 用 exp/log 技巧改写防止溢出:
    #   exp(i * -ln(10000) / d) = 1 / 10000^(i/d)
    # 相当于对每个维度算一个"频率系数"
    #   低维 i=0: 系数=1.0 → 周期最短，变化最快
    #   高维 i=7: 系数≈0.002 → 周期最长，变化最慢
    div_term = np.exp(i * -np.log(10000.0) / d_model)

    # pos × div_term: 广播乘法
    #   pos shape: (seq_len, 1)
    #   div_term shape: (d_model,)
    #   结果 shape: (seq_len, d_model)
    #   每个位置 pos × 每个维度的频率系数 → 得到该位置在该维度的"角度"
    angles = pos * div_term  # (seq_len, d_model)

    # 偶数维 (0,2,4,6...) 取 sin
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    # 奇数维 (1,3,5,7...) 取 cos
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return pe


# ============================================================
# 1. 生成位置编码
# ============================================================
seq_len = 6     # 句子长度: 6 个词
d_model = 8     # 向量维度: 8 维

pe = sinusoidal_positional_encoding(seq_len, d_model)

print("位置编码矩阵 (6个词, 每个8维):")
print(np.round(pe, 3))
print()

# ============================================================
# 2. 观察不同维度的周期差异
# ============================================================
print("=" * 50)
print("观察: 不同维度的周期差异")
print("=" * 50)
print(f"\n第0维 (i=0, 周期最短):")
print(f"  pos 0~5: {np.round(pe[:, 0], 3)}")
print(f"  低维变化快 → 区分相邻位置 (第3和第4个词)")

print(f"\n第2维 (i=1):")
print(f"  pos 0~5: {np.round(pe[:, 2], 3)}")
print(f"  中速变化 → 区分中等距离位置")

print(f"\n第6维 (i=3):")
print(f"  pos 0~5: {np.round(pe[:, 6], 3)}")
print(f"  高维变化慢 → 区分远距离位置")

print(f"\n第7维 (i=3, cos):")
print(f"  pos 0~5: {np.round(pe[:, 7], 3)}")
print(f"  几乎不变 → 对所有位置提供接近的基准值")
print()

# ============================================================
# 3. 完整数值表
# ============================================================
print("=" * 50)
print("每个维度的值随位置变化")
print("=" * 50)  
print(f"\n{'pos':>4}", end="")
for d in range(d_model):
    print(f"  dim{d:>2}", end="")
print()
for pos_i in range(seq_len):
    print(f"{pos_i:>4}", end="")
    for d in range(d_model):
        print(f"  {pe[pos_i, d]:>5.2f}", end="")
    print()
print()

# ============================================================
# 4. 位置编码 + 词向量
# ============================================================
print("=" * 50)
print("最终输入 = 词向量 + 位置编码")
print("=" * 50)

# 模拟 6 个词的词向量
X = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0],
    [0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
])

# 逐元素相加：词义 + 位置信息
X_final = X + pe

print(f"\n词向量 X shape: {X.shape}")
print(f"位置编码 PE shape: {pe.shape}")
print(f"最终输入 X+PE shape: {X_final.shape}")
print()
print("位置编码的特点:")
print("  - 不需要训练，由公式直接生成")
print("  - 不同维度有不同的周期（低维快、高维慢）")
print("  - sin/cos 组合使得模型可以学习相对位置关系")
print("  - 最终输入 = 词向量 + 位置编码，模型同时知道词义和位置")
