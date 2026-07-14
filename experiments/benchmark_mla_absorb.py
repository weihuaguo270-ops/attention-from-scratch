"""
MLA 解压路径 vs 吸收路径微基准

对同一权重、同一序列做逐步自回归推理计时，导出 CSV，便于 README 对照。

用法:
    python -m experiments.benchmark_mla_absorb
    python -m experiments.benchmark_mla_absorb --seq_len 256 --steps 64 --csv experiments/runs/mla_absorb_bench.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modern_llm.mla import MultiHeadLatentAttention


def time_path(mla, x, *, use_absorb: bool, warmup: int, reps: int) -> float:
    """对整段序列逐步 decode，返回平均单次（整段）耗时 ms。"""
    steps = x.shape[0]

    def _run_once():
        c_cache = k_cache = None
        for t in range(steps):
            _, c_cache, k_cache = mla.forward_with_cache(
                x[t : t + 1],
                c_cache,
                k_cache,
                positions=np.array([t]),
                use_absorb=use_absorb,
            )

    for _ in range(warmup):
        _run_once()

    t0 = time.perf_counter()
    for _ in range(reps):
        _run_once()
    elapsed = time.perf_counter() - t0
    return elapsed / reps * 1000.0


def cache_bytes_per_token(d_c: int, d_kv_rope: int, dtype_bytes: int = 4) -> int:
    """MLA 每 token 缓存：c_kv + k_r（FP32 教学默认）。"""
    return (d_c + d_kv_rope) * dtype_bytes


def main():
    parser = argparse.ArgumentParser(description="MLA absorb vs decompress microbench")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_c", type=int, default=64)
    parser.add_argument("--d_kv_rope", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument(
        "--csv",
        default="",
        help="输出 CSV 路径（默认 experiments/runs/mla_absorb_YYYYMMDD.csv）",
    )
    args = parser.parse_args()

    d_k = args.d_model // args.num_heads
    assert args.d_model == args.num_heads * d_k

    np.random.seed(42)
    mla = MultiHeadLatentAttention(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_k=d_k,
        d_c=args.d_c,
        d_kv_rope=args.d_kv_rope,
        max_seq_len=max(args.seq_len, 256),
    )
    mla.absorb_weights()
    x = np.random.randn(args.seq_len, args.d_model).astype(np.float64)

    # 数值对齐检查（短前缀）
    c_d = k_d = c_a = k_a = None
    max_diff = 0.0
    for t in range(min(8, args.seq_len)):
        od, c_d, k_d = mla.forward_with_cache(
            x[t : t + 1], c_d, k_d, positions=np.array([t]), use_absorb=False
        )
        oa, c_a, k_a = mla.forward_with_cache(
            x[t : t + 1], c_a, k_a, positions=np.array([t]), use_absorb=True
        )
        max_diff = max(max_diff, float(np.max(np.abs(od - oa))))

    ms_dec = time_path(mla, x, use_absorb=False, warmup=args.warmup, reps=args.reps)
    ms_abs = time_path(mla, x, use_absorb=True, warmup=args.warmup, reps=args.reps)
    speedup = ms_dec / ms_abs if ms_abs > 0 else float("inf")

    mha_cache = 2 * args.d_model * 4
    mla_cache = cache_bytes_per_token(args.d_c, args.d_kv_rope)
    compress = mha_cache / mla_cache

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    csv_path = args.csv or os.path.join(
        os.path.dirname(__file__), "runs", f"mla_absorb_{stamp}.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "d_c": args.d_c,
        "d_kv_rope": args.d_kv_rope,
        "seq_len": args.seq_len,
        "reps": args.reps,
        "decompress_ms": round(ms_dec, 4),
        "absorb_ms": round(ms_abs, 4),
        "speedup_decompress_over_absorb": round(speedup, 4),
        "max_abs_diff_prefix8": f"{max_diff:.2e}",
        "mha_cache_bytes_per_tok": mha_cache,
        "mla_cache_bytes_per_tok": mla_cache,
        "cache_compress_vs_mha": round(compress, 2),
        "device": "cpu-numpy",
    }

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print("=" * 60)
    print("MLA absorb vs decompress microbench (NumPy CPU)")
    print(
        f"  d_model={args.d_model} heads={args.num_heads} d_c={args.d_c} "
        f"rope={args.d_kv_rope} seq={args.seq_len}"
    )
    print("=" * 60)
    print(f"  数值对齐 max|Δ| (前 8 步): {max_diff:.2e}")
    print(f"  解压路径: {ms_dec:.2f} ms / 全序列 decode")
    print(f"  吸收路径: {ms_abs:.2f} ms / 全序列 decode")
    print(f"  加速比 (解压/吸收): {speedup:.2f}x")
    print(f"  KV Cache/token: MLA {mla_cache}B vs MHA {mha_cache}B ({compress:.1f}x)")
    print(f"  CSV: {csv_path}")


if __name__ == "__main__":
    main()
