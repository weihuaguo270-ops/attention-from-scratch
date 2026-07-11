# Attention From Scratch

> 从零实现 Transformer Attention 机制及其在现代 LLM 中的演进。
>
> 覆盖范围从 **2017 年原始 Transformer**（Self-Attention / MHA / KV Cache / Sinusoidal PE / RoPE / Encoder-Decoder）到 **2024-2025 年主流 LLM 优化**（GQA / Llama 架构 / DeepSeek MLA），体现 Attention 机制的完整发展脉络。
>
> 每个模块提供 **NumPy 版**（逐行理解原理）和 **PyTorch 版**（框架工程实践）。

## 项目动机

Transformer 的核心是 Attention 机制。但主流框架把 `nn.MultiheadAttention` 封装成一行调用，掩盖了关键细节：

- QKV 是如何通过线性变换得到的？
- 因果掩码是如何遮住未来位置的？
- 多头注意力中的"头"是怎么拆分和合并的？
- KV Cache 为什么能加速推理？
- **GQA / MLA 如何进一步压缩 KV Cache？**
- **Llama 架构和原始 Transformer 有什么区别？**
- **Rope 和 Sinusoidal PE 哪个更好？**

更重要的是，如果你只学过原始的 Transformer（2017），面试官会认为你的知识停留在 8 年前。**现代 LLM（Llama、Mistral、DeepSeek、Qwen）使用的已经不是一个架构了。** 本项目展示了这条演进路线。

## 项目结构

```
self_attention/
├── np_impl/                    # NumPy 版
│   ├── attention.py            # 单头 Self-Attention + 因果掩码
│   ├── multi_head_attention.py # 多头 Self-Attention（支持 RoPE）
│   ├── kv_cache.py             # KV Cache 推理加速
│   ├── positional_encoding.py  # Sinusoidal 位置编码
│   ├── rotary.py               # RoPE 旋转位置编码
│   ├── transformer_block.py    # 原始 Transformer Decoder Block
│   ├── cross_attention.py      # Cross-Attention
│   ├── encoder_block.py        # Encoder Block
│   ├── encoder_decoder.py      # Encoder-Decoder 完整架构
│   │
│   │   ── 以下为新增现代 LLM 架构 ──
│   ├── gqa.py                  # Grouped Query Attention
│   ├── llama_block.py          # Llama Decoder Block（RMSNorm+SwiGLU+GQA+RoPE+PreNorm）
│   └── mla.py                  # Multi-head Latent Attention（DeepSeek V2 核心创新）
│
├── pytorch/                    # PyTorch 版
│   ├── attention.py ...        # 与 np_impl/ 结构对应（同上）
│   ├── train_transformer.py    # 完整训练流程
│   ├── compare_pos_encoding.py # Sinusoidal vs RoPE 对比实验
│   └── test_all.py
│
├── test_all.py                 # 统一测试入口（46+ 项测试）
├── pyproject.toml
├── docs/                       # SVG 架构图
└── README.md
```

## Attention 机制的演进路线

本项目按 **三代 Attention 架构** 组织代码，清晰展示演进逻辑：

```
                    Attention 机制演进

  第一代 (2017)              第二代 (2022)             第三代 (2024)
  ──────────────────        ────────────────         ──────────────────
  Transformer 原文           Llama / Mistral          DeepSeek / Google
  
  Self-Attention             Multi-Query Attn (MQA)   Multi-head Latent
  Multi-Head Attn (MHA)      Grouped Query Attn        Attention (MLA)
  Sinusoidal PE                (GQA)                   (KV Cache 压缩 98%)
  Post-Norm + LayerNorm      RoPE                      Absorbed Weights
  ReLU FFN                   RMSNorm + Pre-Norm        Attention Sinks
                              SwiGLU FFN                Sliding Window
```

---

### 第一代：原始 Transformer（2017）— 理解原理

对应文件：`attention.py` → `multi_head_attention.py` → `transformer_block.py` → `encoder_decoder.py`

这是标准教科书内容，也是面试提问的基线。具体见 [README 原内容 §1-9](#)（文档中用的话要结构化展示，但此处作为分层概括，面试时不展开）。

---

### 第二代：现代 LLM 架构 — Llama 风格（2023-2024）

对应文件：`gqa.py` → `llama_block.py`

#### GQA — 分组查询注意力（`gqa.py`）

GQA 是 MHA 和 MQA 的折中方案，**Llama 2/3、Mistral、Qwen 等主流模型全在用**。

```python
# GQA 的核心：K/V 头数 < Q 头数
class GroupedQueryAttention:
    def __init__(self, d_model, num_heads, num_kv_heads, use_rope=False):
        # Q: 完整的多头投影  (d_model → d_model)
        # K/V: 更少的头    (d_model → d_kv * num_kv_heads)
```

**KV Cache 节省量（num_heads=32, seq_len=4096, FP16）：**

| 方案 | KV Cache | 相对 MHA | 使用模型 |
|------|----------|----------|----------|
| MHA (32 KV heads) | 64.0 MB | 1x | 原始 Transformer |
| GQA (8 KV heads)  | 16.0 MB | 25% | Llama 3 70B |
| GQA (4 KV heads)  | 8.0 MB  | 12.5% | Mistral 7B, Qwen |
| MQA (1 KV head)   | 2.0 MB  | 3.1% | Falcon, PaLM |

GQA 的核心机制：K/V 头通过 `np.repeat` 广播以匹配 Q 头数，所有 Q 头被分组共享同一组 K/V 头。

**为什么 GQA 有效？**
- Attention 中的 K 和 V 承载的信息冗余度远高于 Q
- 减少 K/V 头数是"剪掉冗余"，而非"损失精度"
- 分组保证了每个 Q 头仍有专有的 K/V 投影，不是完全共享

#### Llama Decoder Block — 完整架构组合（`llama_block.py`）

将现代 LLM 的所有组件组装成**可在生产中使用的 Decoder Block**：

| 维度 | 原始 Transformer (2017) | Llama 系列 (2023-2024) |
|------|------------------------|------------------------|
| 归一化位置 | **Post-Norm**（子层后） | **Pre-Norm**（子层前） |
| 归一化类型 | **LayerNorm**（μ+σ） | **RMSNorm**（仅σ，快30%） |
| FFN 激活 | **ReLU** | **SwiGLU**（门控机制） |
| Attention | **MHA** | **GQA**（KV Cache 省80%） |
| 位置编码 | **Sinusoidal PE**（加法） | **RoPE**（旋转，可外推） |
| Dropout | 内置 | 极少/无 |

**Pre-Norm 为什么比 Post-Norm 好？**

```
Post-Norm:  out = LayerNorm(子层(x) + x)        # 梯度绕过困难
Pre-Norm:   out = 子层(LayerNorm(x)) + x         # 梯度直通残差路径
```

Post-Norm 中，梯度必须穿过 LayerNorm 才能到达残差路径；Pre-Norm 的 LayerNorm 在子层内部，梯度直接沿残差路径回传 → 深层训练稳定（Llama 可以堆 70B 参数而不炸）。

**SwiGLU 为什么取代 ReLU？**

```
ReLU:    FFN(x) = W_down · max(0, W_up · x)            # 2个参数矩阵
SwiGLU:  FFN(x) = W_down · (Swish(W_gate·x) ⊙ W_up·x)  # 3个参数矩阵
```

多出的 `W_gate` 充当"门控信号"：`Swish(W_gate·x)` 输出 [0~1]，逐个元素决定 `W_up·x` 的哪些信息放行。代价是增加 50% 的 FFN 参数，但实验证明同等参数量下 SwiGLU 效果更好。

---

### 第三代：极致 KV Cache 压缩 — DeepSeek V2 MLA（2024）

对应文件：`mla.py`

#### MLA — 多头潜注意力

**DeepSeek V2/V3 的核心创新**，将 KV Cache 压缩到原来的 ~2%，使其 236B 模型能高效推理。

**核心思想：将 K/V 投影到低维潜空间（压缩），缓存压缩向量（小），推理时再升维（解压）。**

```
MHA:  缓存 K 和 V                     → seq_len × 2 × d_model × bytes
MLA:  缓存 c^{KV} (压缩) + k^R (RoPE) → seq_len × (d_c + d_kv_rope) × bytes

DeepSeek V2 实际参数：
  d_model = 5120, d_c = 512, d_kv_rope = 64
  缓存从 2 × 5120 = 10240 维 → 512 + 64 = 576 维
  压缩比 ≈ 18x
```

**MLA 工作流程：**

```python
# 1. 降维到潜空间（每步都做）
c_kv = h @ W_dkv          # (d_model → d_c) 压缩！

# 2. 从潜空间升维到 K 和 V
k_c = c_kv @ W_uk         # (d_c → d_model) 解压 K 内容
v   = c_kv @ W_uv         # (d_c → d_model) 解压 V

# 3. 分离 RoPE Key（不压缩，保留位置信息）
k_r = RoPE(h @ W_kr)      # (d_model → d_kv_rope)

# 4. 最终缓存：只存 c_kv + k_r
cache = (c_kv, k_r)       # 这就是 KV Cache 的全部！
```

**吸收矩阵技巧（推理优化）：**

推理时，解压 K 的步骤可以省略——将 `W_uk` 吸收到 Q 的投影中：

```
Q · (W_uk · c_kv) = (Q · W_uk) · c_kv
```

预计算 `Q' = Q · W_uk`，使其直接与压缩向量 `c_kv` 作 Attention。V 的解压同理可吸收到输出投影 `Wo` 中。

这个技巧使 MLA **在推理时不引入额外的浮点运算**——压缩和解压只在"后端"发生，前向计算不增加复杂度。

---

## 如何阅读

### 按学习路径

```
初学者（Attention 原理）：
  1. attention.py          → QKV 计算 + 因果掩码
  2. multi_head_attention  → 多头拆分/合并
  3. positional_encoding   → 位置编码
  4. transformer_block     → 完整 Block
  5. kv_cache              → 推理加速

进阶（现代 LLM 架构）：
  6. gqa.py                → GQA 机制
  7. llama_block.py        → Llama 完整架构
  8. mla.py                → DeepSeek MLA

高階（工程能力）：
  9. pytorch/ ↓            → 框架版 + 训练
```

### 按面试准备

| 面试题 | 看什么文件 |
|--------|-----------|
| "解释 Self-Attention 的计算过程" | `attention.py` |
| "为什么用多头注意力？" | `multi_head_attention.py` |
| "LLM 推理时第一个字为什么慢？" | `kv_cache.py` |
| "RoPE 和 Sinusoidal PE 有什么区别？" | `rotary.py` + `positional_encoding.py` |
| "Llama 和原始 Transformer 有什么区别？" | `llama_block.py` |
| "GQA 为什么能省 KV Cache？" | `gqa.py` |
| "介绍一下 DeepSeek V2 的 MLA" | `mla.py` |
| "Post-Norm 和 Pre-Norm 哪个好？" | `transformer_block.py` vs `llama_block.py` |
| "SwiGLU 为什么比 ReLU 好？" | `llama_block.py` |

## 运行

```bash
# 安装依赖
pip install numpy torch

# 运行全部测试（46+ 项）
python test_all.py

# 独立运行单个模块
python -m np_impl.attention     # Self-Attention
python -m np_impl.gqa           # Grouped Query Attention
python -m np_impl.mla           # Multi-head Latent Attention
python -m np_impl.llama_block   # Llama Decoder Block
python -m np_impl.rotary        # RoPE 演示

# PyTorch 版
cd pytorch && python test_all.py
python train_transformer.py
python compare_pos_encoding.py
```

## 模块依赖关系

```
├── np_impl/utils.py           ← softmax/split_heads/combine_heads/layer_norm
│   ├── np_impl/attention.py   ← 单头 Attention
│   ├── np_impl/rotary.py      ← RoPE 旋转（独立模块）
│   │
│   ├── np_impl/multi_head_attention.py  ← 核心：被许多模块 import
│   │   ├── np_impl/transformer_block.py
│   │   ├── np_impl/encoder_block.py
│   │   └── np_impl/encoder_decoder.py
│   │
│   ├── np_impl/gqa.py                  ← GQA（新，不依赖 MHA）
│   │   └── np_impl/llama_block.py      ← Llama Block（基于 GQA）
│   │
│   └── np_impl/mla.py                  ← MLA（独立，不依赖其他模块）
│
├── test_all.py                ← 测试全部模块
└── pytorch/                   ← 框架版镜像
```

## 后续计划

- [x] **P0 深度方向**：GQA / Llama 架构（RMSNorm+SwiGLU+RoPE+PreNorm）/ MLA
- [ ] **P1 广度方向**：BPE Tokenizer 实现 / 真实数据集完整训练 / 推理 Demo
- [ ] **P2 加分项目**：Attention Sinks / Sliding Window Attention / Speculative Decoding
- [ ] **P3 亮点项目**：Flash Attention 数值对比（tiling 原理） / Mamba/SSM 调研笔记

## License

MIT
