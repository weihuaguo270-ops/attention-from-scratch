# Attention From Scratch

> 从零实现 Transformer Attention 机制，理解底层原理，掌握框架使用。
> 项目包含**两个独立实现**：
> - **NumPy 版** — 逐行手写，理解每一步的计算过程
> - **PyTorch 版** — 用框架 API 重写，感受封装 vs 手写的差异，掌握 PyTorch 工程实践
> 
> 覆盖 Self-Attention、多头注意力、因果掩码、KV Cache、位置编码（Sinusoidal + RoPE）、完整 Transformer Block、Cross-Attention、Encoder-Decoder 完整架构、训练流程、位置编码对比实验。

## 项目动机

Transformer 架构的核心是 Attention 机制，但主流框架（PyTorch、TensorFlow）把它封装成了 `nn.MultiheadAttention` 这样的一行调用。这种封装虽然方便，但隐藏了关键细节：

- QKV 是怎么通过线性变换得到的？
- 因果掩码是如何遮住未来位置的？
- 多头注意力中的"头"是怎么拆分和合并的？
- KV Cache 为什么能加速推理？加速了多少？
- Sinusoidal PE 和 RoPE 有什么区别？
- Encoder-Decoder 之间 Cross-Attention 如何连接？

为此项目提供**两个视角**：

**NumPy 版：** 用纯 NumPy 逐行实现这些过程，每一步都可以打印出中间张量的形状，直观理解 Attention 的计算本质。适合**理解原理**。

**PyTorch 版：** 用 PyTorch 的 `nn.Linear`、`nn.Module`、自动微分、优化器重写同一套逻辑，感受框架封装带来的代码量减少和训练能力。适合**框架实践**。

## 项目结构

```
self_attention/
├── np_impl/              # NumPy 版（逐行手写，理解原理）
│   ├── attention.py      # 单头 Self-Attention + 因果掩码
│   ├── multi_head_attention.py  # 多头 Self-Attention（支持 RoPE）
│   ├── kv_cache.py        # KV Cache 推理加速
│   ├── positional_encoding.py  # 正弦位置编码
│   ├── rotary.py          # RoPE 旋转位置编码
│   ├── transformer_block.py    # 完整 Decoder Block
│   ├── cross_attention.py      # Cross-Attention
│   ├── encoder_block.py        # Encoder Block
│   ├── encoder_decoder.py      # Encoder-Decoder 完整架构
│   └── utils.py           # 公共工具函数
│
├── pytorch/              # PyTorch 版（框架封装，可训练）
│   ├── attention.py      # 单头 Self-Attention
│   ├── multi_head_attention.py  # 多头注意力
│   ├── kv_cache.py        # KV Cache
│   ├── positional_encoding.py  # 位置编码
│   ├── encoder_block.py        # Encoder Block
│   ├── encoder_decoder.py      # Encoder-Decoder
│   ├── transformer_block.py    # Decoder Block
│   ├── cross_attention.py      # Cross-Attention
│   ├── compare_pos_encoding.py  # Sinusoidal vs RoPE 对比实验
│   ├── train_transformer.py    # 完整训练流程（200 epoch → Acc 100%）
│   ├── test_all.py        # PyTorch 版测试
│   └── utils.py           # PyTorch 版工具函数
│
├── test_all.py            # NumPy 版测试（36 项）
├── pyproject.toml         # 项目配置
└── README.md
```

## 运行

```bash
# 安装依赖
pip install numpy

# （可选）如需运行 PyTorch 版
pip install torch

# NumPy 版
python np_impl/attention.py

# 多头注意力
python np_impl/multi_head_attention.py

# KV Cache 推理加速（含速度对比）
python np_impl/kv_cache.py

# 位置编码
python np_impl/positional_encoding.py

# RoPE 旋转位置编码
python np_impl/rotary.py

# 完整 Transformer Block（Sinusoidal PE / RoPE 可切换）
python np_impl/transformer_block.py

# Cross-Attention（交叉注意力）
python np_impl/cross_attention.py

# Encoder Block（双向 Attention）
python np_impl/encoder_block.py

# Encoder-Decoder 完整架构
python np_impl/encoder_decoder.py
```

### 运行测试

```bash
python test_all.py                    # NumPy 版 36 项测试
cd pytorch && python test_all.py      # PyTorch 版 25+ 项测试
```

输出示例：

```
【utils 工具函数】
  ✅ softmax 形状
  ✅ softmax 行和为1
  ✅ attention 单头 Self-Attention
  ✅ 因果掩码形状
  ✅ 词0只看自己
  ...
🎉 全部测试通过!
```

当前共 **36 项测试**，覆盖 utils / attention / multi_head_attention / kv_cache / positional_encoding / transformer_block 六个模块，**无需外部框架**。

### PyTorch 版

项目同时提供了 [PyTorch 版](./pytorch/) 实现，与 NumPy 版一一对应：

| 模块 | NumPy | PyTorch |
|------|-------|---------|
| 工具函数 | [`np_impl/utils.py`](./np_impl/utils.py) | [`pytorch/utils.py`](./pytorch/utils.py) |
| 单头 Attention | [`np_impl/attention.py`](./np_impl/attention.py) | [`pytorch/attention.py`](./pytorch/attention.py) |
| 多头注意力 | [`np_impl/multi_head_attention.py`](./np_impl/multi_head_attention.py) | [`pytorch/multi_head_attention.py`](./pytorch/multi_head_attention.py) |
| KV Cache | [`np_impl/kv_cache.py`](./np_impl/kv_cache.py) | [`pytorch/kv_cache.py`](./pytorch/kv_cache.py) |
| 位置编码 | [`np_impl/positional_encoding.py`](./np_impl/positional_encoding.py) | [`pytorch/positional_encoding.py`](./pytorch/positional_encoding.py) |
| Transformer Block | [`np_impl/transformer_block.py`](./np_impl/transformer_block.py) | [`pytorch/transformer_block.py`](./pytorch/transformer_block.py) |
| Cross-Attention | [`np_impl/cross_attention.py`](./np_impl/cross_attention.py) | [`pytorch/cross_attention.py`](./pytorch/cross_attention.py) |
| Encoder Block | [`np_impl/encoder_block.py`](./np_impl/encoder_block.py) | [`pytorch/encoder_block.py`](./pytorch/encoder_block.py) |
| Encoder-Decoder | [`np_impl/encoder_decoder.py`](./np_impl/encoder_decoder.py) | [`pytorch/encoder_decoder.py`](./pytorch/encoder_decoder.py) |
| 测试 | [`test_all.py`](./test_all.py) | [`pytorch/test_all.py`](./pytorch/test_all.py) |
| 训练 | — | [`pytorch/train_transformer.py`](./pytorch/train_transformer.py) |
| 位置编码对比 | — | [`pytorch/compare_pos_encoding.py`](./pytorch/compare_pos_encoding.py) |

```bash
# PyTorch 版
cd pytorch
python test_all.py                     # PyTorch 版 25+ 项测试
python multi_head_attention.py         # 可独立运行
python compare_pos_encoding.py         # Sinusoidal vs RoPE 对比实验
```

两版对比可以直观感受框架封装 vs 手写的差异：PyTorch 版代码量更少（`nn.Linear` / `F.softmax` 替代手写）、支持自动微分、可跑在 GPU 上；NumPy 版每一步运算都暴露在外，适合理解原理。

此外，项目还提供了包含 **完整训练流程** 的 [`pytorch/train_transformer.py`](./pytorch/train_transformer.py)：

- 完整的 Encoder-Decoder 模型（含 Token Embedding、LM Head）
- Teacher Forcing 训练（CrossEntropyLoss + Adam 优化器）
- 训练 200 epoch 后 Acc 100%，Teacher Forcing 测试全部通过

### 运行示例

```bash
$ python attention.py
=== Part A: 无掩码 Self-Attention ===
输入形状: (3, 4)  ← 3个词，每个4维
Q形状: (3, 3)      ← 3个query，每个3维
注意力权重形状: (3, 3)  ← 词与词之间的注意力分数
输出形状: (3, 3)

=== Part B: 因果掩码 Self-Attention ===
词"坐"的注意力分布: [0.38, 0.62, 0.00]
只看了"猫"和自己，没看"垫子"
=== 验证通过 ===
```

## 学习路线

### 1. Self-Attention 原理 → `attention.py`

实现 Attention 计算公式：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

每个步骤对应一段代码：
1. **Q @ K^T** → 计算词与词之间的相似度
2. **/ √d_k** → 缩放，防止 softmax 梯度消失
3. **+ 因果掩码**（Part B）→ 上三角矩阵填 -inf，遮住未来位置
4. **Softmax** → 每行归一化为概率分布
5. **@ V** → 加权求和，得到上下文感知表示

### 2. 因果掩码 — GPT 自回归生成

掩码矩阵：

```
[[0, -inf, -inf],     词0 只看自己
 [0,   0,  -inf],     词1 看词0和词1
 [0,   0,    0]]      词2 看所有词
```

-inf 经过 softmax 后会变成 0，保证未来信息不会被"偷看"。

### 3. 多头注意力 (Multi-Head Attention)

- 将 Q、K、V 拆成 `h` 个头（split_heads）
- 每个头独立计算 Attention
- 合并结果（combine_heads）后过输出投影

```python
def forward(self, Q, K, V):
    # 1. 线性投影
    # 2. 拆头: (batch, seq, d_model) → (batch, h, seq, d_k)
    # 3. 每个头独立算 Attention
    # 4. 合并: (batch, h, seq, d_k) → (batch, seq, d_model)
    # 5. 输出投影
```

`multi_head_attention.py` 支持 `use_rope` 参数——设为 `True` 时在拆头后对每个 head 的 Q/K 应用 RoPE 旋转。

### 4. KV Cache — 推理加速

自回归生成时，每步只生成一个新 token。
如果不做 KV Cache，每步都要重新算所有历史 token 的 K 和 V —— 重复计算。
KV Cache 把之前算过的 K、V 存起来，每步只算新 token 的 K、V：

```python
# 无 KV Cache: O(n²) 每步重新算
# 有 KV Cache: O(n)  每步只算新的
```

`kv_cache.py` 包含自回归循环模拟，直观对比有/无缓存的计算量差异：

```
>>> 使用 KV Cache:
   步数 |    缓存K数量 |   本次算K数量 |      总计算量
-------------------------------------------------------
    1 |        1 |        1 |               1
    2 |        2 |        1 |               1
  ...（每步只算 1 次新 K）
  用缓存: total K 计算次数 = 5（每步1次）
  不用缓存: total K 计算次数 = 20（第t步算t+1次）
```

PyTorch 版（`pytorch/kv_cache.py`）逻辑完全一致。

### 5. 位置编码 — 两种方案对比

#### 5a. Sinusoidal PE（`positional_encoding.py`）

绝对位置编码：每个位置有唯一的 sin/cos 向量，加到输入上。

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**核心特征：** 位置信息**独立于 Attention 计算**，通过加法注入。

#### 5b. RoPE（`rotary.py`）

相对位置编码：不生成位置向量，而是旋转 Q 和 K，让 Attention 分数本身包含位置差信息。

```
f(q, pos) = q × cos(pos×θ) + rotate_90(q) × sin(pos×θ)
```

**核心特征：** 位置信息**编码进 Attention 分数里**，通过旋转 Q/K 实现。

**RoPE 演示（`python rotary.py`）：**
1. 基础演示 — 旋转 Q 和 K，看到不对称 Attention 分数
2. 对比 Sinusoidal — 对角线不变性验证
3. 词序颠倒测试 — 正向和反向句子的 Attention 不同
4. 长度外推 — 训练 5 个位置，推理扩展到 10

#### 5c. TransformerBlock 可切换（`transformer_block.py`）

```python
# Sinusoidal PE（默认）
block = TransformerBlock(d_model=8, num_heads=2, d_ff=16, pos_encoding="sinusoidal")
block.forward(x)  # → 入口加 PE → Attention → FFN

# RoPE
block = TransformerBlock(d_model=8, num_heads=2, d_ff=16, pos_encoding="rope")
block.forward(x)  # → 不加 PE，Attention 内部旋转 Q/K → FFN
```

#### 5d. 对比实验（`pytorch/compare_pos_encoding.py`）

在同一训练任务上，分别用 Sinusoidal PE 和 RoPE 训练 Transformer，打印 loss/acc 曲线对比。

```bash
python pytorch/compare_pos_encoding.py
```

输出示例：

```
指标                   Sinusoidal PE      RoPE
--------------------------------------------------------
最终 Loss              0.0014             0.0013
最终 Acc               100.0%             100.0%
测试通过率              100%               100%
到达 90% Acc           epoch 20           epoch 20
```

两种方案在简单任务上都能收敛，核心差异在**作用位置不同**：Sinusoidal 在输入层做加法，RoPE 在 Attention 内部做旋转。

### 6. 完整 Transformer Block

将以上所有组件组装为一个完整的 Decoder Block：

```
输入 → [Positional Encoding] → Multi-Head Attention → 残差 + LayerNorm → FFN → 残差 + LayerNorm → 输出
```

`transformer_block.py` 包含了完整的 Block，可堆叠 N 层。这是 Decoder-only 架构（GPT 风格）。

### 7. Cross-Attention（`cross_attention.py`）

与 Self-Attention 的区别：Q 来自 Decoder 当前输出，K/V 来自 Encoder 的输出。Encoder 和 Decoder 之间的桥梁，是翻译/摘要任务的核心机制。

### 8. Encoder Block（`encoder_block.py`）

双向 Self-Attention（无因果掩码），每个词看到整个句子。适用场景：理解整句话（BERT 风格）。

### 9. Encoder-Decoder 完整架构（`encoder_decoder.py`）

将 Encoder × N 层 + Decoder × N 层（含 Cross-Attention）串联：Encoder 编码原句子 → Decoder 通过 Cross-Attention 逐词生成译文。结构：

```
Encoder:  双向 Self-Attention → +残差 → LayerNorm → FFN → +残差 → LayerNorm
Decoder:  Self-Attention(因果掩码) → +残差 → LayerNorm → Cross-Attention → +残差 → LayerNorm → FFN → +残差 → LayerNorm
```

### 10. 训练流程（`pytorch/train_transformer.py`）

完整的 PyTorch 训练 Pipeline：
- Token Embedding + Positional Encoding → Encoder → Decoder → **LM Head** → vocab 概率
- Teacher Forcing 训练（CrossEntropyLoss + Adam）
- 200 epoch 后 Acc 100%

## 架构总览

### Decoder-only（`transformer_block.py`）

适用于自回归生成（GPT 风格）。

![Decoder-only 架构](./docs/decoder_only.svg)

### Encoder-Decoder（`encoder_decoder.py`）

适用于翻译、摘要等需要"理解输入再生成"的任务。Decoder 自回归逐词生成。

![Encoder-Decoder 架构](./docs/encoder_decoder.svg)

**自回归生成循环（推理时）：**

```python
generated = [<BOS>]
for step in range(max_len):
    logits = decoder(encoder_output, generated)
    next_token = argmax(logits[-1])   # 只取最后一步的预测
    generated.append(next_token)      # 拼回输入，继续下一轮
```

## 模块依赖关系

```
utils.py ← 所有文件从这里 import
  ├── attention.py
  ├── multi_head_attention.py
  ├── kv_cache.py
  ├── positional_encoding.py
  ├── rotary.py ← 独立文件，不依赖其他模块
  └── transformer_block.py ← 还 import 了 multi_head_attention.py

cross_attention.py ← 独立，不依赖其他模块（只依赖 utils.py）
encoder_block.py ← import multi_head_attention.py + utils.py
encoder_decoder.py ← import encoder_block.py + cross_attention.py

pytorch/ 目录 ← 与根目录结构完全对应，每个文件有 PyTorch 版
  ├── utils.py
  ├── attention.py
  ├── multi_head_attention.py
  ├── kv_cache.py
  ├── positional_encoding.py
  ├── transformer_block.py
  ├── cross_attention.py
  ├── encoder_block.py
  ├── encoder_decoder.py
  ├── train_transformer.py  ← 额外：完整训练流程
  ├── compare_pos_encoding.py  ← 位置编码对比实验
  └── test_all.py
```

每个文件独立可运行，按顺序阅读效果最佳。

## 后续计划

- [x] PyTorch 版实现（pytorch/ 目录，与 NumPy 版一一对应）
- [x] PyTorch 训练流程（LM Head + Teacher Forcing 训练）
- [x] 交叉注意力（Cross-Attention）— Encoder-Decoder 架构
- [x] RoPE（旋转位置编码）实现 — `rotary.py` + `transformer_block` 可切换
- [x] 位置编码对比实验 — `pytorch/compare_pos_encoding.py`
- [ ] PyTorch 实战：加载真实数据集训练语言模型
- [ ] KV Cache 的进一步优化：PagedAttention、MQA、GQA
- [ ] Flash Attention 原理与数值对比

## License

MIT
