# Attention From Scratch

**NumPy/PyTorch implementation of Transformer attention mechanisms** -- covering the full evolution from the original 2017 Transformer to modern LLM architectures (GQA, Llama Block, DeepSeek MLA,Speculative Decoding, Attention Sinks).

### Attention Mechanisms

### MHA - Multi-Head Attention (2017)

The foundation. Q, K, V each projected to d_model, split into n_heads, scored with scaled dot-product attention.

**KV Cache** (np_impl/kv_cache.py): During autoregressive decoding, cached K/V save redundant recomputation.

### GQA - Grouped Query Attention (2023)

Used by Llama 2/3, Mistral, Qwen. Reduces KV heads while keeping Q heads.

| Variant | KV Heads | KV Cache (32h, 4096seq, FP16) | Models |
|---------|---------|--------------------------------|-------|
| MHA | 32 | 64.0 MB | Original Transformer |
| GQA | 8 | 16.0 MB | Llama 3 70B |
| GQA | 4 |  8.0 MB | Mistral 7B |
| MQA | 1 |  2.0 MB | Falcon |

### Llama Decoder Block

Five key differences from the original Transformer:

| Dimension | Original | Llama | Rationale |
|-------------|-----------|-----------|--------------|
| Normalization position | Post-Norm | Pre-Norm | Gradient flows through residual path |
| Norm type | LayerNorm | RMSNorm | 30% faster, comparable quality |
| FFF activation | ReLU | SwiGLU | Learnable gated activation |
| Attention | MHA | GQA | 80% KV cache reduction |
| Position encoding | Sinusoidal PE | RoPE | Length extrapolation |

### MLA - Multi-head Latent Attention (2024)

**DeepSeek V2/V3's core innovation.** Compresses K/V into a low-dimensional latent space.

```
 MHA:  K = h · W_K,      cache K (d_model dim)
 MLA:  c = h · W_DKV,    cache c (d_c dim, d_c << d_model)
       K = c · W_UK      (decompress from cached latent)
```

**Absorption matrix trick** -- at inference, decompression is elided:
```
 Q · (W_UK ° c) = (Q · W_UK) · c    # W_UK absorbed into Q projection
```
DeepSeek V2 (d_model=5120): MHA cache 10240 dims vs MLA cache 576 dims → **18x compression**

### Speculative Decoding
Accelerates autoregressive generation: draft model proposes multiple tokens, target model verifies in parallel.

### Attention Sinks / StreamingLLM
Maintains recent tokens + initial tokens (attention sink) for long-context inference beyond training length.
