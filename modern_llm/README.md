# 现代 LLM 架构（2023-2024）

> `modern_llm/` — 纯 NumPy 实现，覆盖当前主流大模型使用的 Attention 变体。
>
> 包含 Llama 路线（GQA + RMSNorm + SwiGLU）和 DeepSeek 路线（MLA）两大类方案。
> 两条路线从原始 Transformer 分叉而来，解决不同层面的优化问题。

独立包，不依赖 `np_impl/` 目录。

## 文件说明

| 文件 | 内容 |
|------|------|
| `gqa.py` | Grouped Query Attention（分组查询注意力） |
| `llama_block.py` | 完整 Llama Decoder Block（RMSNorm+SwiGLU+GQA+RoPE+Pre-Norm） |
| `mla.py` | Multi-head Latent Attention（DeepSeek V2 核心） |
| `speculative_decoding.py` | Speculative Decoding（投机解码：小模型草拟→大模型验证→Rejection Sampling） |
| `utils.py` | softmax |
| `rotary.py` | RoPE 旋转位置编码 |

## 模块详解

### GQA — Grouped Query Attention（`gqa.py`）

MHA 到 MQA 的折中方案。多个 Q 头共享一组 K/V 头，KV Cache 减少 75-90%。

```python
# K/V 头数 < Q 头数
Q = x @ W_q  → split → (num_heads, seq, d_k)
K = x @ W_k  → split → (num_kv_heads, seq, d_k)
# 关键：K/V 头重复以匹配 Q 头
K = np.repeat(K, num_heads // num_kv_heads, axis=0)
```

### Llama Decoder Block（`llama_block.py`）

5 项关键改进 vs 原始 Transformer：

| 维度 | 原始 Transformer | Llama |
|------|-----------------|-------|
| 归一化位置 | **Post-Norm**（子层后） | **Pre-Norm**（子层前）→ 梯度直通残差 |
| 归一化类型 | **LayerNorm**（μ+σ+β） | **RMSNorm**（仅σ）→ 快 30% |
| FFN 激活 | **ReLU**（2 个矩阵） | **SwiGLU**（3 个矩阵，门控） |
| Attention | **MHA** | **GQA**（省 80% KV Cache） |
| 位置编码 | **Sinusoidal PE**（加法） | **RoPE**（旋转，可外推） |

```
# Pre-Norm 结构
x → RMSNorm → GQA → +残差 → RMSNorm → SwiGLU → +残差 → 输出
```

### MLA — Multi-head Latent Attention（`mla.py`）

DeepSeek V2/V3 的核心创新。将 K/V 压缩到低维潜空间，推理时缓存压缩向量而非完整 K/V。

```python
# 1. 降维到潜空间
c_kv = x @ W_dkv          # (d_model → d_c)  缓存的只有这个！
# 2. 从缓存解压
k_c = c_kv @ W_uk          # (d_c → d_model)
v   = c_kv @ W_uv          # (d_c → d_model)
```

**吸收矩阵技巧**：推理时 `W_uk` 可吸收到 Q 投影中，不增加计算量。

DeepSeek V2 实际 KV Cache 对比（d_model=5120, seq_len=4096, FP16）：

| 方案 | 单层缓存 | 60 层总计 |
|------|---------|----------|
| MHA | 0.25 GB | 15.00 GB |
| MLA | 0.004 GB | 0.26 GB |
| **压缩比** | **~18x** | **~58x** |

## 运行

```bash
# 测试全部模块
python -m modern_llm.test

# 独立运行单个模块
python -c "from modern_llm.gqa import GroupedQueryAttention; ..."
python -c "from modern_llm.llama_block import LlamaDecoderBlock; ..."
python -c "from modern_llm.mla import MultiHeadLatentAttention; ..."
```

## 模块依赖

```
utils.py        ← softmax
rotary.py       ← RoPE（独立）
  ├── gqa.py    ← GQA（引用 utils + rotary）
  │   └── llama_block.py  ← Llama Block（引用 gqa）
  ├── mla.py    ← MLA（引用 utils + rotary）
  └── speculative_decoding.py  ← Spec Decoding（独立，引用 utils）

---

> 🔄 框架工程实践版 → [`pytorch/README.md`](../pytorch/README.md)
