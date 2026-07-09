"""
KV Cache — 模拟自回归生成循环（与 pytorch/ 版对应）

面试高频题: "为什么 LLM 生成时第一个字慢，后面越来越快？"

对比：
  有缓存：每步只算新词的 K、V，拼到缓存后面 → 每步计算量恒定
  无缓存：每步重新算所有词的 K、V → 计算量线性增长

核心:
  无缓存: O(N²)
  有缓存: O(N)
"""
import sys, os
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "np_impl"
import numpy as np


def generate(num_steps, use_cache=True, d_k=4):
    """模拟自回归生成，追踪每步计算量
    参数:
        num_steps: 生成的 token 数
        use_cache: True=用KV Cache，False=每次都重新算全部
        d_k: 向量维度
    """
    np.random.seed(42)
    # 首个 token（Prompt 阶段）
    _ = np.random.randn(d_k)   # 首个 token 的 embedding
    K_cache, V_cache = None, None
    print(f"{'步数':>5} | {'缓存K数量':>8} | {'本次算K数量':>8} | {'总计算量(算K次数)':>15}")
    print("-" * 55)
    for step in range(num_steps):
        if use_cache:
            # 有缓存：只算当前 1 个 token 的 K、V
            k_cur = np.random.randn(1, d_k)
            v_cur = np.random.randn(1, d_k)
            num_k_computed = 1
            if K_cache is None:
                K_cache, V_cache = k_cur, v_cur
            else:
                K_cache = np.concatenate([K_cache, k_cur], axis=0)
                V_cache = np.concatenate([V_cache, v_cur], axis=0)
        else:
            # 无缓存：全部重算 step+2 个（首个 token + 当前步）
            num_k_computed = step + 2
            K_cache = np.random.randn(step + 2, d_k)
            V_cache = np.random.randn(step + 2, d_k)
        cache_size = 0 if K_cache is None else K_cache.shape[0]
        total = (step + 2) if not use_cache else num_k_computed
        print(f"{step+1:>5} | {cache_size:>8} | {num_k_computed:>8} | {total:>15}")
    print("-" * 55)
    total_no_cache = sum(range(2, num_steps + 2))
    print(f"\n生成 {num_steps} 个 token 的总结：")
    print(f"  用缓存: total K 计算次数 = {num_steps}（每步1次）")
    print(f"  不用缓存: total K 计算次数 = {total_no_cache}（第t步算t+1次）")
    print(f"  节省: {total_no_cache - num_steps} 次 K、V 计算")
if __name__ == "__main__":
    num_steps = 5
    print("=" * 55)
    print("KV Cache 自回归生成模拟")
    print("=" * 55)
    print("\n>>> 使用 KV Cache:")
    generate(num_steps, use_cache=True)
    print("\n>>> 不使用 KV Cache:")
    generate(num_steps, use_cache=False)