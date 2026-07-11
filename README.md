# Attention From Scratch

> 从零实现 Attention 机制，完整覆盖三**代**架构演进。
>
> 项目划分为**两个独立目录**，每代架构独立运行、互不依赖：
>
> - **`np_impl/`** — 第一代：原始 Transformer（2017）
> - **`modern_llm/`** — 第二/三代：现代 LLM 架构（2023-2024）

## 架构演进

```
                    Attention 机制演进

   第一代 (2017)          第二代 (2023)          第三代 (2024)
   ─────────────         ─────────────          ─────────────
   Transformer 原文       Llama / Mistral        DeepSeek V2/V3

   Self-Attention        Grouped Query Attn     Multi-head Latent
   Multi-Head Attn        (GQA)                   Attention (MLA)
   Sinusoidal PE         RoPE                   潜空间 KV 压缩
   Post-Norm             RMSNorm + Pre-Norm     吸收矩阵技巧
   ReLU FFN              SwiGLU FFN
   Encoder-Decoder       Decoder-only
```

---

## 项目结构

```
self_attention/
│
├── np_impl/                    ← 第一代：原始 Transformer（2017）
│   ├── attention.py           单头 Self-Attention + 因果掩码
│   ├── multi_head_attention.py 多头自注意力（MHA）
│   ├── kv_cache.py            KV Cache 推理加速
│   ├── positional_encoding.py Sinusoidal 位置编码
│   ├── rotary.py              RoPE 旋转位置编码
│   ├── transformer_block.py   原始 Decoder Block（Post-Norm + ReLU）
│   ├── cross_attention.py     编码器-解码器交叉注意力
│   ├── encoder_block.py       Encoder Block
│   ├── encoder_decoder.py     Encoder-Decoder 完整架构
│   ├── utils.py               公共工具函数
│   └── test.py                独立测试（36+ 项）
│
├── modern_llm/                 ← 第二/三代：现代 LLM 架构（2023-2024）
│   ├── gqa.py                 Grouped Query Attention
│   ├── llama_block.py         Llama Decoder Block（RMSNorm + SwiGLU + GQA + RoPE + Pre-Norm）
│   ├── mla.py                 Multi-head Latent Attention（DeepSeek V2 核心创新）
│   ├── utils.py               独立工具函数（不依赖 np_impl/）
│   ├── rotary.py              独立 RoPE（不依赖 np_impl/）
│   └── test.py                独立测试（15+ 项）
│
├── pytorch/                    PyTorch 版（与 np_impl/ 镜像）
├── test_all.py                 统一测试入口
├── pyproject.toml
├── docs/
└── README.md
```

---

## 第一代：原始 Transformer（`np_impl/`）

标准 Transformer 教科书实现。从最基本的 Self-Attention 开始，逐步组装出完整的 Encoder-Decoder 架构。

| 文件 | 覆盖内容 | 面试对应题 |
|------|---------|-----------|
| `attention.py` | QKV 投影、缩放点积 Attention、因果掩码 | "Attention 的计算公式是什么？" |
| `multi_head_attention.py` | 多头拆分/合并、RoPE 可切换 | "为什么用多头注意力？" |
| `kv_cache.py` | 有/无缓存的计算量对比 | "为什么 LLM 首字生成慢？" |
| `positional_encoding.py` | Sinusoidal PE 公式、维度周期 | "位置编码为什么用 sin/cos？" |
| `rotary.py` | RoPE 旋转、对角线不变性、长度外推 | "RoPE 和 Sinusoidal 有什么区别？" |
| `transformer_block.py` | Post-Norm + ReLU FFN | "原始 Transformer Block 的结构？" |
| `encoder_decoder.py` | Encoder-Decoder 完整串联 | "Encoder-Decoder 如何衔接？" |

运行：`python -m np_impl.test`

---

## 第二/三代：现代 LLM 架构（`modern_llm/`）

从原始 Transformer 演进到当前主流 LLM 使用的架构。**独立包，不与 `np_impl/` 共享代码。**

### GQA — 分组查询注意力（`gqa.py`）

MHA 到 MQA 的折中方案，**Llama 2/3、Mistral、Qwen 全在用**。

```python
class GroupedQueryAttention:
    def __init__(self, d_model, num_heads, num_kv_heads):
        # Q: 完整的多头投影 (d_model → d_model)
        # K/V: 更少的头    (d_model → d_k * num_kv_heads)
```

KV Cache 节省量（num_heads=32, seq_len=4096, FP16）：

| 方案 | KV Cache | 相对 MHA | 典型模型 |
|------|----------|----------|---------|
| MHA (32 KV heads) | 64.0 MB | 1x | 原始 Transformer |
| GQA (8 KV heads)  | 16.0 MB | 25% | Llama 3 70B |
| GQA (4 KV heads)  |  8.0 MB | 12.5% | Mistral 7B |
| MQA (1 KV head)   |  2.0 MB | 3.1% | Falcon |

### Llama Decoder Block（`llama_block.py`）

现代 LLM 架构的核心组件——与原始 Transformer 的 5 项关键差异：

| 维度 | 原始 Transformer | Llama 系列 | 为什么改？ |
|------|-----------------|------------|-----------|
| 归一化位置 | Post-Norm（子层后） | **Pre-Norm**（子层前） | 梯度直通残差路径，深层稳定 |
| 归一化类型 | LayerNorm（μ+σ+β） | **RMSNorm**（仅σ） | 计算快 30%，效果持平 |
| FFN 激活 | ReLU | **SwiGLU**（门控） | 可学习的选择性激活 |
| Attention | MHA | **GQA** | KV Cache 省 80% |
| 位置编码 | Sinusoidal PE（加法） | **RoPE**（旋转） | 支持长度外推 |

结构对比：

```
# 原始 Transformer Block（Post-Norm）
x → MHA → +残差 → LayerNorm → FFN → +残差 → LayerNorm → 输出
                                ↑ ReLU

# Llama Decoder Block（Pre-Norm）
x → RMSNorm → GQA → +残差 → RMSNorm → SwiGLU → +残差 → 输出
                 ↑ RoPE            ↑ 3 个权重矩阵
```

### MLA — 多头潜注意力（`mla.py`）

**DeepSeek V2/V3 的核心创新**，将 KV Cache 压缩至 ~2%。

核心公式：

```
MHA:  K = h · W_K,  缓存 K (d_model 维)
MLA:  c = h · W_DKV, 缓存 c (d_c 维, d_c << d_model)
      K = c · W_UK   (从压缩缓存解压)
```

吸收矩阵技巧——推理时解压步骤可省略：

```
Q · (W_UK · c) = (Q · W_UK) · c    # W_UK 被吸收到 Q 投影中
```

运行：`python -m modern_llm.test`

---

## 统一测试

```bash
# 全部运行
python test_all.py

# 分别运行（推荐——两者完全独立）
python -m np_impl.test         # 原始 Transformer 36+ 项
python -m modern_llm.test      # 现代 LLM 15+ 项
```

输出示例：

```
$ python test_all.py

############################################################
# Part 1: 原始 Transformer（np_impl/）
############################################################
【utils 工具函数】
  ✅ softmax 形状
  ✅ softmax 行和为1
  ...
🎉 原始 Transformer 全部测试通过!

############################################################
# Part 2: 现代 LLM 架构（modern_llm/）
############################################################
【GQA 分组查询注意力】
  ✅ GQA 输出形状
  ...
🎉 Modern LLM 全部测试通过!
```

---

## 阅读指南

```
想学什么                    → 看哪个目录
──────────────────────────────────────────────
第一次学 Attention           → np_impl/attention.py
Transformer 完整架构         → np_impl/encoder_decoder.py
Llama 面试题                → modern_llm/llama_block.py
Scaling KV Cache            → modern_llm/gqa.py → mla.py
DeepSeek MLA 实现细节       → modern_llm/mla.py
```

---

## 后续计划

- [x] **P0 - 深度方向**：GQA / Llama Block / MLA（`modern_llm/`）
- [ ] **P1 - 广度方向**：BPE Tokenizer / 真实数据训练 / 推理 Demo
- [ ] **P2 - 加分项目**：Attention Sinks / Sliding Window / Speculative Decoding
- [ ] **P3 - 亮点项目**：Flash Attention / Mamba / 简单量化

---

## License

MIT
