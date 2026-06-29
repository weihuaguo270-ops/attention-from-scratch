# Transformer Attention 机制 — 手写学习笔记

纯 NumPy 实现，从零到完整 Transformer Block，理解 Attention 最核心的计算过程。

## 文件说明

| 文件 | 内容 |
|------|------|
| `utils.py` | 公共函数库（softmax、split_heads、combine_heads、layer_norm） |
| `attention.py` | 单头 Self-Attention + 因果掩码（Causal Mask） |
| `multi_head_attention.py` | 多头 Self-Attention（类，可被 import） |
| `kv_cache.py` | KV Cache — 自回归生成推理加速原理 |
| `positional_encoding.py` | 正弦位置编码（Sinusoidal PE） |
| `transformer_block.py` | 完整 Transformer Block（Attention + 残差 + LayerNorm + FFN） |

## 模块依赖关系

```
utils.py ← 所有文件从这里 import
  ├── attention.py
  ├── multi_head_attention.py
  ├── kv_cache.py
  ├── positional_encoding.py
  └── transformer_block.py ← 还 import 了 multi_head_attention.py
```

每个文件独立可运行，按顺序阅读效果最佳。

## 运行

```bash
pip install numpy
python attention.py
python multi_head_attention.py
python kv_cache.py
python positional_encoding.py
python transformer_block.py
```

## 学习路线

1. Self-Attention 原理 → `attention.py`
2. 因果掩码（GPT 自回归）→ `attention.py Part B`
3. 多头注意力（Multi-Head）→ `multi_head_attention.py`
4. KV Cache 推理加速 → `kv_cache.py`
5. 位置编码 → `positional_encoding.py`
6. 完整 Transformer Block → `transformer_block.py`

## 架构总览

```
输入 (seq_len, d_model)
        ↓
  [Positional Encoding]         ← positional_encoding.py
        ↓
  ┌────────────────────────┐
  │ Transformer Block × N   │     ← transformer_block.py
  │  ┌──────────────────┐   │
  │  │ Multi-Head        │   │     ← multi_head_attention.py
  │  │ Self-Attention    │   │
  │  │  (含因果掩码)      │   │     ← attention.py
  │  └────────┬─────────┘   │
  │           ↓              │
  │  + 残差连接 + LayerNorm  │     ← utils.py (layer_norm)
  │           ↓              │
  │  ┌──────────────────┐   │
  │  │ FFN               │   │     ← transformer_block.py 内
  │  └────────┬─────────┘   │
  │           ↓              │
  │  + 残差连接 + LayerNorm  │
  └────────────────────────┘
        ↓
  KV Cache（推理时复用）       ← kv_cache.py
        ↓
  输出 → 预测下一个词
```
