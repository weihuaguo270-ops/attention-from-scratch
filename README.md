# Attention From Scratch

> 从零实现 Attention 机制，覆盖从原始 Transformer 到现代 LLM 架构的两条演进路线。
>
> 项目划分为**两个独立目录**，互不依赖：
>
> - **`np_impl/`** — 原始 Transformer（2017），共同起点
> - **`modern_llm/`** — 现代方案合集（GQA / Llama Block / DeepSeek MLA）
> - **`pytorch/`** — 框架工程版，含 GQA / Llama Block 的 nn.Module 实现，
>   完整 GPT 训练 pipeline（TinyStories 数据集 + 命令行调参 + 自动实验记录）

## 两条演进路线

原始 Transformer（2017）是共同的起点。此后主流 LLM 分两条路线演进：

```
              原始 Transformer（2017）
              /                    \
        Llama 路线               DeepSeek 路线
  (GQA + Pre-Norm + SwiGLU)    (MLA 潜空间压缩)
       2023-2024                   2024
```

两条路线解决不同的问题：

| 路线 | 核心痛点 | 方案 |
|------|---------|------|
| **Llama 路线** | 模型堆不深、KV Cache 太大 | GQA + Pre-Norm + RMSNorm + SwiGLU + RoPE |
| **DeepSeek 路线** | KV Cache 仍是瓶颈（236B 模型无法部署） | MLA：K/V 压缩到潜空间，缓存降至 ~2% |

两条路线不互斥——理论上一家公司可以同时用 GQA + MLA。实际上 GQA 和 MLA 解决的是同一问题的不同层次，可以叠加使用。

项目结构对应：

| 目录 | 内容 | 对应路线 |
|------|------|---------|
| `np_impl/` | 原始 Transformer（NumPy 手写） | 共同起点 |
| `modern_llm/` | GQA + Llama Block + MLA + Spec Decoding + StreamingLLM（NumPy） | 两条路线的现代方案合集 |
| `pytorch/` | GQA + Llama Block 的 PyTorch 实现 + GPT 训练（TinyStories） + 实验记录 | 训练工程实践 |

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
│   ├── test.py                独立测试（36+ 项）
│   └── README.md              ← 本目录详细说明
│
├── modern_llm/                 ← 第二/三代：现代 LLM 架构（2023-2024）
│   ├── gqa.py                 Grouped Query Attention
│   ├── llama_block.py         Llama Decoder Block
│   ├── mla.py                 Multi-head Latent Attention
│   ├── utils.py               独立工具函数
│   ├── rotary.py              独立 RoPE
│   ├── test.py                独立测试（15+ 项）
│   └── README.md              ← 本目录详细说明
│
├── experiments/                ← 对比实验（横向比较各模块）
│   ├── compare_attention.py    MHA vs GQA vs MLA 缓存/参数量
│   ├── compare_cache.py        完整缓存 vs StreamingLLM 质量/节省
│   ├── compare_decoding.py     标准 vs Spec Decoding 加速比
│   ├── compare_training.py     超参数训练效果对比
│   ├── runs/                   实验记录系统（自动存档 + 交互式对比）
│   │   ├── compare.py          一键查看/筛选/对比所有实验
│   │   ├── README.md
│   │   ├── legacy_001_baseline/
│   │   ├── legacy_005_early_stop/
│   │   └── 20260712_..._auto/  （每次训练自动生成）
│   └── README.md
├── pytorch/                    ← PyTorch 版 — 从原理到训练的全流程实现
│   ├── gqa.py                  GQA + RoPE（PyTorch 版）
│   ├── llama_block.py          RMSNorm + SwiGLU + 完整 GPT 模型
│   ├── train_gpt.py            训练脚本（交互式/命令行，自动记录实验）
│   ├── data.py                 TinyStories 数据加载 + 词表构建
│   ├── attention.py ...        np_impl 各模块的 PyTorch 镜像
│   └── README.md              ← 本目录详细说明
├── test_all.py                 统一测试入口
├── pyproject.toml
├── docs/
└── README.md
```

> 💡 每个子目录有自己的 README，详细说明该目录的模块和用法：
> - 原始 Transformer → [`np_impl/README.md`](./np_impl/README.md)
> - 现代 LLM 架构 → [`modern_llm/README.md`](./modern_llm/README.md)
> - PyTorch 框架版 → [`pytorch/README.md`](./pytorch/README.md)

---

## 第一代：原始 Transformer（`np_impl/`）

标准 Transformer 教科书实现。从最基本的 Self-Attention 开始，逐步组装出完整的 Encoder-Decoder 架构。

| 文件 | 覆盖内容 |
|------|---------|
| `attention.py` | QKV 投影、缩放点积 Attention、因果掩码 |
| `multi_head_attention.py` | 多头拆分/合并、RoPE 可切换 |
| `kv_cache.py` | 有/无缓存的计算量对比 |
| `positional_encoding.py` | Sinusoidal PE 公式、维度周期 |
| `rotary.py` | RoPE 旋转、对角线不变性、长度外推 |
| `transformer_block.py` | Post-Norm + ReLU FFN |
| `encoder_decoder.py` | Encoder-Decoder 完整串联 |

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

输出示例（实际运行结果）：

```text
$ python test_all.py

============================================================
Attention From Scratch — 全部测试
============================================================

############################################################
# Part 1: 原始 Transformer（np_impl/）
############################################################

【utils 工具函数】
  ✅ softmax 形状
  ✅ softmax 行和为1
  ✅ softmax 单调性
  ✅ split_heads 形状
  ✅ combine_heads 还原
  ✅ layer_norm 均值≈0
  ✅ layer_norm 方差≈1

【attention 单头 Self-Attention】
  ✅ 无掩码输出形状
  ✅ 权重行和为1
  ✅ 词0只看自己
  ✅ 词1只看前2
  ✅ 词2看全部

【multi_head_attention 多头注意力】
  ✅ 多头输出形状
  ✅ 多头输出非零

【kv_cache KV Cache 验证】
  ✅ KV Cache 第2步一致

【positional_encoding 位置编码】
  ✅ 位置编码形状
  ✅ 相邻位置编码不同
  ✅ 值域在[-1,1]

【transformer_block 完整 Block】
  ✅ Block输出形状
  ✅ Block输出稳定
  ✅ 3层堆叠稳定

【encoder_decoder 完整架构】
  ✅ Encoder完整输出形状
  ✅ Decoder完整输出形状
  ✅ 完整流程稳定

🎉 原始 Transformer 全部测试通过!

############################################################
# Part 2: 现代 LLM 架构（modern_llm/）
############################################################

【GQA 分组查询注意力】
  ✅ GQA 输出形状
  ✅ GQA 输出稳定
  ✅ GQA+掩码输出形状
  ✅ GQA+RoPE 输出稳定
  ✅ GQA K/V 重复正确

【Llama Block】
  ✅ RMSNorm 输出形状
  ✅ RMSNorm 输出稳定
  ✅ SwiGLU 输出稳定
  ✅ LlamaBlock 输出形状
  ✅ LlamaBlock 输出稳定
  ✅ Llama 3层堆叠形状
  ✅ Llama 3层堆叠稳定

【MLA 多头潜注意力】
  ✅ MLA 输出形状
  ✅ MLA 输出稳定
  ✅ MLA KV Cache 3步生成
  ✅ MLA KV Cache 输出稳定

🎉 Modern LLM 全部测试通过!

============================================================
🎉 全部测试通过!（共 51+ 项）
============================================================
```

---

## 阅读指南

```
想学什么                    → 看哪个目录
──────────────────────────────────────────────
第一次学 Attention           → np_impl/attention.py
Transformer 完整架构         → np_impl/encoder_decoder.py
Llama 架构                  → modern_llm/llama_block.py
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
