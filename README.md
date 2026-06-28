# Transformer Attention 机制 — 手写学习笔记

纯 NumPy 实现，理解 Attention 最核心的计算过程。

## 文件说明

| 文件 | 内容 |
|------|------|
| `attention.py` | 单头 Self-Attention + 因果掩码（Causal Mask） |
| `multi_head_attention.py` | 多头 Self-Attention |

## 运行

```bash
pip install numpy
python attention.py
python multi_head_attention.py
```

---

> 📝 **待补充**：后续继续添加 Positional Encoding、KV Cache、完整 Transformer Block 等内容。
