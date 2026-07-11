# 原始 Transformer 架构（2017）

> `np_impl/` — 纯 NumPy 实现，逐行手写，理解每一步计算。

覆盖从单头 Self-Attention 到完整 Encoder-Decoder 的全过程。是理解 Attention 机制的起点。

## 文件说明

| 文件 | 内容 |
|------|------|
| `attention.py` | 单头 Self-Attention + 因果掩码 |
| `multi_head_attention.py` | 多头自注意力（支持 RoPE 切换） |
| `kv_cache.py` | KV Cache 推理加速（有/无缓存对比） |
| `positional_encoding.py` | Sinusoidal 位置编码 |
| `rotary.py` | RoPE 旋转位置编码 |
| `transformer_block.py` | 完整 Decoder Block（Post-Norm + ReLU） |
| `cross_attention.py` | 编码器-解码器交叉注意力 |
| `encoder_block.py` | Encoder Block（双向 Attention） |
| `encoder_decoder.py` | Encoder-Decoder 完整串联 |
| `utils.py` | softmax / split_heads / combine_heads / layer_norm |

## 学习路径

按顺序阅读效果最佳：

```
1. utils.py               → 基础函数
2. attention.py           → QKV 计算 + 因果掩码
3. multi_head_attention.py → 多头拆分合并
4. positional_encoding.py → 位置编码
5. rotary.py              → RoPE 旋转位置编码
6. kv_cache.py            → 推理加速
7. transformer_block.py   → 组装为完整 Block
8. cross_attention.py     → 交叉注意力
9. encoder_block.py       → Encoder
10. encoder_decoder.py    → Encoder-Decoder 组合
```

## 运行

```bash
# 独立运行单个模块
python -m np_impl.attention
python -m np_impl.multi_head_attention
python -m np_impl.kv_cache
python -m np_impl.positional_encoding
python -m np_impl.rotary
python -m np_impl.transformer_block
python -m np_impl.cross_attention
python -m np_impl.encoder_block
python -m np_impl.encoder_decoder

# 运行全部测试（36+ 项）
python -m np_impl.test
```

## 模块依赖

```
utils.py
  ├── attention.py
  ├── multi_head_attention.py  ← 被 transformer/encoder 引用
  ├── kv_cache.py
  ├── positional_encoding.py
  └── rotary.py                ← 被 MHA 引用

transformer_block.py  ← 依赖 multi_head_attention.py
cross_attention.py    ← 只依赖 utils.py
encoder_block.py      ← 依赖 multi_head_attention.py
encoder_decoder.py    ← 依赖 encoder_block + cross_attention
```

---

> 🔄 框架工程实践版 → [`pytorch/README.md`](../pytorch/README.md)
