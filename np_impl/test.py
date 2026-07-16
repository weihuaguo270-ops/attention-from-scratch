"""
原始 Transformer（np_impl）测试

独立运行：python -m np_impl.test
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from console_io import FAIL, PASS, configure_stdio, safe_print

configure_stdio()

errors = []


def check(name, cond, detail=""):
    if cond:
        safe_print(f"  {PASS} {name}")
    else:
        msg = f"  {FAIL} {name}" + (f" — {detail}" if detail else "")
        safe_print(msg)
        errors.append(name)


# ============================================================
# 1. utils
# ============================================================
print("\n【utils 工具函数】")
from np_impl.utils import softmax, split_heads, combine_heads, layer_norm

x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
s = softmax(x)
check("softmax 形状", s.shape == (2, 3))
check("softmax 行和为1", np.allclose(s.sum(axis=-1), [1.0, 1.0]))
check("softmax 单调性", s[0, 0] < s[0, 1] < s[0, 2])

x2d = np.random.randn(4, 8)
sh = split_heads(x2d, num_heads=2)
check("split_heads 形状", sh.shape == (2, 4, 4))
check("split_heads 值不变", np.allclose(sh[0], x2d[:, :4]))

ch = combine_heads(sh, num_heads=2)
check("combine_heads 还原", ch.shape == (4, 8))
check("combine_heads 值不变", np.allclose(ch, x2d))

x_ln = np.random.randn(2, 4, 8) * 2 + 1
ln_out = layer_norm(x_ln, eps=1e-5)
check("layer_norm 形状", ln_out.shape == (2, 4, 8))
check("layer_norm 均值≈0",
      np.allclose(ln_out.mean(axis=-1), np.zeros((2, 4)), atol=1e-5))
check("layer_norm 方差≈1",
      np.allclose(ln_out.std(axis=-1), np.ones((2, 4)), atol=1e-4))


# ============================================================
# 2. attention
# ============================================================
print("\n【attention 单头 Self-Attention】")
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

out_masked = mha.forward(X_mha, use_mask=True)
check("多头+掩码输出形状", out_masked.shape == (4, 8))


# ============================================================
# 4. kv_cache
# ============================================================
print("\n【kv_cache KV Cache 验证】")
from np_impl.utils import softmax


def attention_with_cache(Q, K, V, K_cache=None, V_cache=None):
    if K_cache is not None:
        K = np.concatenate([K_cache, K], axis=0)
        V = np.concatenate([V_cache, V], axis=0)
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    weights = softmax(scores)
    return weights @ V, K, V


np.random.seed(42)
d_k = 4

q1 = np.random.randn(1, d_k)
k1 = np.random.randn(1, d_k)
v1 = np.random.randn(1, d_k)
out1, K_cache, V_cache = attention_with_cache(q1, k1, v1)

q2 = np.random.randn(1, d_k)
k2 = np.random.randn(1, d_k)
v2 = np.random.randn(1, d_k)
out2_cached, K_cache, V_cache = attention_with_cache(q2, k2, v2, K_cache, V_cache)

out2_nocache, _, _ = attention_with_cache(
    q2, np.concatenate([k1, k2]), np.concatenate([v1, v2]))
check("KV Cache 第2步一致", np.allclose(out2_cached, out2_nocache, atol=1e-6))


# ============================================================
# 5. positional_encoding
# ============================================================
print("\n【positional_encoding 位置编码】")
from np_impl.positional_encoding import sinusoidal_positional_encoding as positional_encoding

pe = positional_encoding(seq_len=10, d_model=8)
check("位置编码形状", pe.shape == (10, 8))
check("位置编码非零", np.linalg.norm(pe) > 0)

diff = np.linalg.norm(pe[0] - pe[1])
check("相邻位置编码不同", diff > 0)

check("值域在[-1,1]", np.all(np.abs(pe) <= 1.0 + 1e-6))


# ============================================================
# 5b. rotary (RoPE)
# ============================================================
print("\n【rotary RoPE】")
from np_impl.rotary import precompute_rotary_frequencies, apply_rotary

d_k = 8
cos_t, sin_t = precompute_rotary_frequencies(d_k, max_seq_len=16)
check("RoPE 频率表形状", cos_t.shape == (16, d_k // 2) and sin_t.shape == (16, d_k // 2))
q = np.random.randn(4, d_k)
q_rot = apply_rotary(q, cos_t, sin_t)
check("RoPE 输出形状", q_rot.shape == (4, d_k))
check("RoPE 输出稳定", np.all(np.isfinite(q_rot)))
check("RoPE 改变向量", np.linalg.norm(q_rot - q) > 0)


# ============================================================
# 6. transformer_block
# ============================================================
print("\n【transformer_block 完整 Block】")
from np_impl.transformer_block import TransformerBlock

block = TransformerBlock(d_model=8, num_heads=2, d_ff=16)
X_tb = np.random.randn(4, 8)
out_tb = block.forward(X_tb, use_mask=True)
check("Block输出形状", out_tb.shape == (4, 8))
check("Block输出稳定", np.all(np.isfinite(out_tb)))

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
# 汇总
# ============================================================
print(f"\n{'='*50}")
if errors:
    safe_print(f"{FAIL} {len(errors)} 项失败:")
    for e in errors:
        safe_print(f"   - {e}")
else:
    safe_print(f"{PASS} 原始 Transformer 全部测试通过!")
print(f"{'='*50}")
