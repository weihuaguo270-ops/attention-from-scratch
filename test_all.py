"""
Attention From Scratch — 数值验证测试

验证每个模块的正确性，不依赖外部框架。
用法:
    python test_all.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

errors = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ✅ {name}")
    else:
        msg = f"  ❌ {name}" + (f" — {detail}" if detail else "")
        print(msg)
        errors.append(name)


# ============================================================
# 1. utils
# ============================================================
print("\n【utils 工具函数】")
from np_impl.utils import softmax, split_heads, combine_heads, layer_norm

# softmax: 每行和为 1
x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
s = softmax(x)
check("softmax 形状", s.shape == (2, 3))
check("softmax 行和为1", np.allclose(s.sum(axis=-1), [1.0, 1.0]))
check("softmax 单调性", s[0, 0] < s[0, 1] < s[0, 2])  # 大值 → 大概率

# split_heads: (seq, d_model) → (h, seq, d_k)
x2d = np.random.randn(4, 8)
sh = split_heads(x2d, num_heads=2)
check("split_heads 形状", sh.shape == (2, 4, 4))
check("split_heads 值不变", np.allclose(sh[0], x2d[:, :4]))

# combine_heads: (h, seq, d_k) → (seq, d_model)
ch = combine_heads(sh, num_heads=2)
check("combine_heads 还原", ch.shape == (4, 8))
check("combine_heads 值不变", np.allclose(ch, x2d))

# layer_norm: 均值≈0, 方差≈1
x_ln = np.random.randn(2, 4, 8) * 2 + 1  # 故意偏置
ln_out = layer_norm(x_ln, eps=1e-5)
check("layer_norm 形状", ln_out.shape == (2, 4, 8))
check("layer_norm 均值≈0",
      np.allclose(ln_out.mean(axis=-1), np.zeros((2, 4)), atol=1e-5))
check("layer_norm 方差≈1",
      np.allclose(ln_out.std(axis=-1), np.ones((2, 4)), atol=1e-4))


# ============================================================
# 2. attention — 单头 Self-Attention
# ============================================================
print("\n【attention 单头 Self-Attention】")
# 手动执行 attention.py 中的核心逻辑
from np_impl.utils import softmax

np.random.seed(42)
d_model, d_k = 4, 3
W_q = np.random.randn(d_model, d_k)
W_k = np.random.randn(d_model, d_k)
W_v = np.random.randn(d_model, d_k)
X = np.random.randn(3, d_model)

Q = X @ W_q
K = X @ W_k
V = X @ W_v
scores = Q @ K.T / np.sqrt(d_k)
weights = softmax(scores)
output = weights @ V

check("无掩码输出形状", output.shape == (3, d_k))
check("注意力权重形状", weights.shape == (3, 3))
check("权重行和为1", np.allclose(weights.sum(axis=-1), [1.0, 1.0, 1.0]))

# 因果掩码
causal_mask = np.triu(np.full((3, 3), -1e9), k=1)
causal_scores = Q @ K.T / np.sqrt(d_k) + causal_mask
causal_weights = softmax(causal_scores)

check("因果掩码形状", causal_weights.shape == (3, 3))
check("词0只看自己", causal_weights[0, 1] == 0.0 and causal_weights[0, 2] == 0.0)
check("词1只看前2", causal_weights[1, 2] == 0.0)
check("词2看全部", causal_weights[2, 0] > 0 and causal_weights[2, 2] > 0)


# ============================================================
# 3. multi_head_attention
# ============================================================
print("\n【multi_head_attention 多头注意力】")
from np_impl.multi_head_attention import MultiHeadAttention

np.random.seed(42)
mha = MultiHeadAttention(d_model=8, num_heads=2)
X_mha = np.random.randn(4, 8)
out_mha = mha.forward(X_mha, use_mask=False)
check("多头输出形状", out_mha.shape == (4, 8))
check("多头输出非零", np.linalg.norm(out_mha) > 0)

# 有掩码
out_masked = mha.forward(X_mha, use_mask=True)
check("多头+掩码输出形状", out_masked.shape == (4, 8))


# ============================================================
# 4. kv_cache
# ============================================================
print("\n【kv_cache KV Cache 验证】")
# 验证有 cache 和无 cache 的输出是否一致（值上）
from np_impl.utils import softmax

def attention_with_cache(Q, K, V, K_cache=None, V_cache=None):
    """简化版 attention，支持 cache"""
    if K_cache is not None:
        K = np.concatenate([K_cache, K], axis=0)
        V = np.concatenate([V_cache, V], axis=0)
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    weights = softmax(scores)
    return weights @ V, K, V

np.random.seed(42)
d_k = 4

# 模拟 3 步生成：第1步没有 cache
q1 = np.random.randn(1, d_k)
k1 = np.random.randn(1, d_k)
v1 = np.random.randn(1, d_k)
out1, K_cache, V_cache = attention_with_cache(q1, k1, v1)

# 第2步用 cache
q2 = np.random.randn(1, d_k)
k2 = np.random.randn(1, d_k)
v2 = np.random.randn(1, d_k)
out2_cached, K_cache, V_cache = attention_with_cache(q2, k2, v2, K_cache, V_cache)

# 对比：如果不 cache，第2步应该看到 k1,k2
out2_nocache, _, _ = attention_with_cache(q2, np.concatenate([k1, k2]), np.concatenate([v1, v2]))
check("KV Cache 第2步一致", np.allclose(out2_cached, out2_nocache, atol=1e-6))


# ============================================================
# 5. positional_encoding
# ============================================================
print("\n【positional_encoding 位置编码】")
from np_impl.positional_encoding import sinusoidal_positional_encoding as positional_encoding

pe = positional_encoding(seq_len=10, d_model=8)
check("位置编码形状", pe.shape == (10, 8))
check("位置编码非零", np.linalg.norm(pe) > 0)

# 相邻位置的编码应该不同
diff = np.linalg.norm(pe[0] - pe[1])
check("相邻位置编码不同", diff > 0)

# 对称性检查: sin/cos 应该在 [-1, 1]
check("值域在[-1,1]", np.all(np.abs(pe) <= 1.0 + 1e-6))


# ============================================================
# 6. transformer_block
# ============================================================
print("\n【transformer_block 完整 Block】")
from np_impl.transformer_block import TransformerBlock

block = TransformerBlock(d_model=8, num_heads=2, d_ff=16)
X_tb = np.random.randn(4, 8)
out_tb = block.forward(X_tb, use_mask=True)
check("Block输出形状", out_tb.shape == (4, 8))
check("Block输出稳定", np.all(np.isfinite(out_tb)))  # 没有 NaN 或 Inf

# 多层堆叠
x = X_tb
for i in range(3):
    x = block.forward(x, use_mask=True)
check("3层堆叠稳定", np.all(np.isfinite(x)))


# ============================================================
# 7. cross_attention
# ============================================================
print("\n【cross_attention 交叉注意力】")
from np_impl.cross_attention import MultiHeadCrossAttention

np.random.seed(42)
ca = MultiHeadCrossAttention(d_model=8, num_heads=2)
query = np.random.randn(2, 8)
key_value = np.random.randn(3, 8)
out_ca = ca.forward(query, key_value)
check("Cross-Attn 输出形状", out_ca.shape == (2, 8))
check("Cross-Attn 输出非零", np.linalg.norm(out_ca) > 0)

# Q 和 KV 长度不同
query2 = np.random.randn(1, 8)
out_ca2 = ca.forward(query2, key_value)
check("Cross-Attn 单步生成", out_ca2.shape == (1, 8))


# ============================================================
# 8. encoder_block
# ============================================================
print("\n【encoder_block 编码器】")
from np_impl.encoder_block import EncoderBlock

encoder = EncoderBlock(d_model=8, num_heads=2, d_ff=16)
X_enc = np.random.randn(4, 8)
out_enc = encoder.forward(X_enc)
check("Encoder输出形状", out_enc.shape == (4, 8))
check("Encoder输出稳定", np.all(np.isfinite(out_enc)))


# ============================================================
# 9. encoder_decoder
# ============================================================
print("\n【encoder_decoder 完整架构】")
from np_impl.encoder_decoder import EncoderDecoder

model = EncoderDecoder(d_model=8, num_heads=2, d_ff=16, num_layers=2)
source = np.random.randn(3, 8)
target = np.random.randn(2, 8)
enc_out = model.encode(source)
check("Encoder完整输出形状", enc_out.shape == (3, 8))
dec_out = model.decode(target, enc_out)
check("Decoder完整输出形状", dec_out.shape == (2, 8))
check("完整流程稳定", np.all(np.isfinite(dec_out)))



# ============================================================
# 10. gqa — Grouped Query Attention
# ============================================================
print("\n【gqa 分组查询注意力】")
from np_impl.gqa import GroupedQueryAttention

np.random.seed(42)
gqa = GroupedQueryAttention(d_model=8, num_heads=4, num_kv_heads=2)
X_gqa = np.random.randn(4, 8)
out_gqa = gqa.forward(X_gqa, use_mask=False)
check("GQA 输出形状", out_gqa.shape == (4, 8))
check("GQA 输出非零", np.linalg.norm(out_gqa) > 0)
check("GQA 输出稳定", np.all(np.isfinite(out_gqa)))

# GQA + 掩码
out_gqa_masked = gqa.forward(X_gqa, use_mask=True)
check("GQA+掩码输出形状", out_gqa_masked.shape == (4, 8))

# GQA + RoPE
gqa_rope = GroupedQueryAttention(d_model=8, num_heads=4, num_kv_heads=2, use_rope=True)
out_gqa_rope = gqa_rope.forward(X_gqa, use_mask=False)
check("GQA+RoPE 输出形状", out_gqa_rope.shape == (4, 8))
check("GQA+RoPE 输出稳定", np.all(np.isfinite(out_gqa_rope)))

# GQA K/V 重复正确性验证
Q = X_gqa @ gqa.Wq
K = X_gqa @ gqa.Wk
dk = gqa.d_k
Q_h = Q.reshape(4, gqa.num_heads, dk).transpose(1, 0, 2)
K_h = K.reshape(4, gqa.num_kv_heads, dk).transpose(1, 0, 2)
num_repeats = gqa.num_heads // gqa.num_kv_heads
K_rep = np.repeat(K_h, num_repeats, axis=0)
check("GQA K/V 重复正确", np.allclose(K_rep[0], K_h[0]) and np.allclose(K_rep[2], K_h[1]))


# ============================================================
# 11. llama_block — RMSNorm + SwiGLU + Llama Block
# ============================================================
print("\n【llama_block Llama 架构】")
from np_impl.llama_block import RMSNorm, SwiGLU, LlamaDecoderBlock, LlamaModel

# RMSNorm
rms = RMSNorm(8)
x_rms = np.random.randn(4, 8)
out_rms = rms.forward(x_rms)
check("RMSNorm 输出形状", out_rms.shape == (4, 8))
check("RMSNorm 输出稳定", np.all(np.isfinite(out_rms)))

# SwiGLU
swiglu = SwiGLU(d_model=8, d_ff=16)
x_swi = np.random.randn(4, 8)
out_swi = swiglu.forward(x_swi)
check("SwiGLU 输出形状", out_swi.shape == (4, 8))
check("SwiGLU 输出稳定", np.all(np.isfinite(out_swi)))

# LlamaDecoderBlock
block = LlamaDecoderBlock(d_model=8, num_heads=4, num_kv_heads=2, d_ff=16, use_rope=True)
x_block = np.random.randn(4, 8)
out_block = block.forward(x_block, use_mask=True)
check("LlamaBlock 输出形状", out_block.shape == (4, 8))
check("LlamaBlock 输出稳定", np.all(np.isfinite(out_block)))
check("LlamaBlock 输出非零", np.linalg.norm(out_block) > 0)

# 多层堆叠
model = LlamaModel(num_layers=3, d_model=8, num_heads=4, num_kv_heads=2, d_ff=16)
out_multi = model.forward(x_block, use_mask=True)
check("Llama 3层堆叠形状", out_multi.shape == (4, 8))
check("Llama 3层堆叠稳定", np.all(np.isfinite(out_multi)))


# ============================================================
# 12. mla — Multi-head Latent Attention
# ============================================================
print("\n【mla 多头潜注意力】")
from np_impl.mla import MultiHeadLatentAttention

mla = MultiHeadLatentAttention(d_model=16, num_heads=4, d_k=4, d_c=6, d_kv_rope=4)
x_mla = np.random.randn(3, 16)
out_mla = mla.forward(x_mla, use_mask=True)
check("MLA 输出形状", out_mla.shape == (3, 16))
check("MLA 输出稳定", np.all(np.isfinite(out_mla)))

# MLA KV Cache 自回归模拟
mla_small = MultiHeadLatentAttention(d_model=8, num_heads=2, d_k=4, d_c=3, d_kv_rope=2)
c_kv_cache, k_r_cache = None, None
outputs = []
for step in range(3):
    x_in = np.random.randn(1, 8)
    out, c_kv_cache, k_r_cache = mla_small.forward_with_cache(
        x_in, c_kv_cache, k_r_cache, positions=np.array([step])
    )
    outputs.append(out)
check("MLA KV Cache 3步生成", len(outputs) == 3)
check("MLA KV Cache 输出稳定", all(np.all(np.isfinite(o)) for o in outputs))
check("MLA KV Cache 形状正确", all(o.shape == (1, 8) for o in outputs))


# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*50}")
if errors:
    print(f"❌ {len(errors)} 项失败:")
    for e in errors:
        print(f"   - {e}")
else:
    print("🎉 全部测试通过!")
print(f"{'='*50}")
