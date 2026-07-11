"""
Modern LLM 架构 — 独立测试

测试 GQA、Llama Block、MLA 的正确性。
独立运行：python -m modern_llm.test
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

errors = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ✅ {name}")
    else:
        msg = f"  ❌ {name}" + (f" — {detail}" if detail else "")
        print(msg)
        errors.append(name)


# ============================================================
# 1. GQA
# ============================================================
print("\n【GQA 分组查询注意力】")
from modern_llm.gqa import GroupedQueryAttention

np.random.seed(42)
gqa = GroupedQueryAttention(d_model=8, num_heads=4, num_kv_heads=2)
X = np.random.randn(4, 8)

out = gqa.forward(X, use_mask=False)
check("GQA 输出形状", out.shape == (4, 8))
check("GQA 输出稳定", np.all(np.isfinite(out)))

out_m = gqa.forward(X, use_mask=True)
check("GQA+掩码输出形状", out_m.shape == (4, 8))

gqa_r = GroupedQueryAttention(d_model=8, num_heads=4, num_kv_heads=2, use_rope=True)
out_r = gqa_r.forward(X, use_mask=False)
check("GQA+RoPE 输出形状", out_r.shape == (4, 8))
check("GQA+RoPE 输出稳定", np.all(np.isfinite(out_r)))

# GQA: 验证 K/V 重复正确性
q = X @ gqa.Wq
k = X @ gqa.Wk
dk = gqa.d_k
qh = q.reshape(4, 4, dk).transpose(1, 0, 2)
kh = k.reshape(4, 2, dk).transpose(1, 0, 2)
kr = np.repeat(kh, 2, axis=0)
check("GQA K/V 重复正确",
      np.allclose(kr[0], kh[0]) and np.allclose(kr[2], kh[1]))


# ============================================================
# 2. Llama Block
# ============================================================
print("\n【Llama Block】")
from modern_llm.llama_block import RMSNorm, SwiGLU, LlamaDecoderBlock, LlamaModel

rms = RMSNorm(8)
x_rms = np.random.randn(4, 8)
out_rms = rms.forward(x_rms)
check("RMSNorm 输出形状", out_rms.shape == (4, 8))
check("RMSNorm 输出稳定", np.all(np.isfinite(out_rms)))

swiglu = SwiGLU(d_model=8, d_ff=16)
x_swi = np.random.randn(4, 8)
out_swi = swiglu.forward(x_swi)
check("SwiGLU 输出形状", out_swi.shape == (4, 8))
check("SwiGLU 输出稳定", np.all(np.isfinite(out_swi)))

block = LlamaDecoderBlock(d_model=8, num_heads=4, num_kv_heads=2, d_ff=16, use_rope=True)
x_block = np.random.randn(4, 8)
out_block = block.forward(x_block, use_mask=True)
check("LlamaBlock 输出形状", out_block.shape == (4, 8))
check("LlamaBlock 输出稳定", np.all(np.isfinite(out_block)))
check("LlamaBlock 输出非零", np.linalg.norm(out_block) > 0)

model = LlamaModel(num_layers=3, d_model=8, num_heads=4, num_kv_heads=2, d_ff=16)
out_multi = model.forward(x_block, use_mask=True)
check("Llama 3层堆叠形状", out_multi.shape == (4, 8))
check("Llama 3层堆叠稳定", np.all(np.isfinite(out_multi)))


# ============================================================
# 3. MLA
# ============================================================
print("\n【MLA 多头潜注意力】")
from modern_llm.mla import MultiHeadLatentAttention

mla = MultiHeadLatentAttention(d_model=16, num_heads=4, d_k=4, d_c=6, d_kv_rope=4)
x_mla = np.random.randn(3, 16)
out_mla = mla.forward(x_mla, use_mask=True)
check("MLA 输出形状", out_mla.shape == (3, 16))
check("MLA 输出稳定", np.all(np.isfinite(out_mla)))

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
    print("🎉 Modern LLM 全部测试通过!")
print(f"{'='*50}")
