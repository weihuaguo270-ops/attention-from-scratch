# PyTorch 版 — 框架工程实践

`pytorch/` 用 PyTorch 框架 API（`nn.Linear`、`F.softmax`、自动微分等）重写各模块，支持完整 GPT 训练。

## 与 NumPy 版的对应关系

| 模块 | NumPy 版 | PyTorch 版 | 差异 |
|------|---------|-----------|------|
| Self-Attention | `np_impl/attention.py` | `attention.py` | nn.Linear + F.softmax |
| MHA | `np_impl/multi_head_attention.py` | `attention.py` | 同上 |
| RoPE | `np_impl/rotary.py` | `attention.py` | 集成到 attention 中 |
| Cross Attention | `np_impl/cross_attention.py` | `cross_attention.py` | nn.Linear + F.softmax |
| Encoder Block | `np_impl/encoder_block.py` | `encoder_block.py` | nn.Module 封装 |
| Encoder-Decoder | `np_impl/encoder_decoder.py` | `encoder_decoder.py` | nn.Module 封装 |
| GQA | `modern_llm/gqa.py` | `gqa.py` | nn.Linear + F.scaled_dot_product_attention |
| Llama Block | `modern_llm/llama_block.py` | `llama_block.py` | RMSNorm + SwiGLU + 完整 GPT 模型 |

## 训练

```bash
python -m pytorch.train_gpt --epochs 3 --d_model 64 --num_heads 4
python -m pytorch.train_gpt --epochs 5 --num_kv_heads 2  # GQA
```

- 数据集：TinyStories
- 支持命令行调参和交互式输入
- 自动记录实验配置和结果

## 实验记录

参见 [`experiments/runs/`](../experiments/runs/README.md)。