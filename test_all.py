"""
Transformer Attention — 统一测试入口

运行两个独立测试套件：
  np_impl/     — 原始 Transformer 架构（2017）
  modern_llm/  — 现代 LLM 架构（GQA / Llama Block / DeepSeek MLA）

用法：
  python test_all.py           # 运行全部
  python -m np_impl.test       # 仅运行原始 Transformer
  python -m modern_llm.test    # 仅运行现代 LLM
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from console_io import FAIL, PASS, configure_stdio, safe_print

configure_stdio()

safe_print("=" * 60)
safe_print("Transformer Attention — 全部测试")
safe_print("=" * 60)

# ── Part 1: 原始 Transformer ──
safe_print("\n" + "#" * 60)
safe_print("# Part 1: 原始 Transformer（np_impl/）")
safe_print("#" * 60)
import np_impl.test as test_np
np_result = len(test_np.errors)

# ── Part 2: 现代 LLM ──
safe_print("\n" + "#" * 60)
safe_print("# Part 2: 现代 LLM 架构（modern_llm/）")
safe_print("#" * 60)
import modern_llm.test as test_modern
modern_result = len(test_modern.errors)

# ── 汇总 ──
safe_print("\n" + "=" * 60)
total = np_result + modern_result
if total == 0:
    safe_print(f"{PASS} 全部测试通过!")
else:
    safe_print(f"{FAIL} 共 {total} 项失败: np_impl={np_result}, modern_llm={modern_result}")
safe_print(f"{'='*60}")
