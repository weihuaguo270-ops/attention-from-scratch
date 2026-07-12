"""
实验对比工具 — 查看、筛选、对比实验记录

用法:
  python -m experiments.runs.compare                       # 全部
  python -m experiments.runs.compare --tags baseline        # 按标签筛选
  python -m experiments.runs.compare --tags lr-test         # 只看学习率实验
  python -m experiments.runs.compare --ids 001 005          # 只看指定实验
  python -m experiments.runs.compare --last 3               # 最近 3 次
  python -m experiments.runs.compare --table brief           # 简表
  python -m experiments.runs.compare --table full           # 详表（默认）
"""
import json, os, sys, argparse
from datetime import datetime

RUNS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_all():
    """加载所有实验，按编号排序"""
    experiments = []
    for d in sorted(os.listdir(RUNS_DIR)):
        if not d[0].isdigit():
            continue
        config_path = os.path.join(RUNS_DIR, d, "config.json")
        results_path = os.path.join(RUNS_DIR, d, "results.json")
        if not os.path.exists(config_path) or not os.path.exists(results_path):
            continue
        with open(config_path) as f:
            config = json.load(f)
        with open(results_path) as f:
            results = json.load(f)
        experiments.append((d, config, results))
    return experiments


def filter_experiments(experiments, tags=None, ids=None, last=None):
    """按条件筛选"""
    if ids:
        ids = [f"{int(i):03d}" for i in ids]
        experiments = [e for e in experiments if any(i in e[0] for i in ids)]
    if tags:
        tags = [t.lower() for t in tags]
        experiments = [
            e for e in experiments
            if any(t in [x.lower() for x in e[1].get("tags", [])] for t in tags)
        ]
    if last:
        experiments = experiments[-last:]
    return experiments


def fmt(v, default="-"):
    """格式化数值"""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        if abs(v) < 10:
            return f"{v:.2f}"
        elif abs(v) < 10000:
            return f"{v:.0f}"
        else:
            return f"{v:.1e}"
    return str(v)


def print_table(experiments, mode="full"):
    """打印对比表"""
    if not experiments:
        print("没有找到匹配的实验。")
        return

    if mode == "brief":
        print(f"  {'ID':>10}  {'描述':>28}  {'Val Loss':>8}  {'PPL':>6}  {'轮次':>5}")
        print("  " + "-" * 65)
        for exp_id, config, results in experiments:
            desc = config.get("description", config.get("desc", ""))[:28]
            print(f"  {exp_id:>10}  {desc:>28}  "
                  f"{fmt(results.get('best_val_loss')):>8}  "
                  f"{fmt(results.get('perplexity')):>6}  "
                  f"{fmt(results.get('epochs_actual')):>5}")
        return

    # full mode
    # 表头
    print(f"\n{'=' * 100}")
    print(f"{'ID':>10}  {'描述':>28}  {'模型配置':>30}  {'Val Loss':>8}  "
          f"{'PPL':>8}  {'轮次':>5}  {'最佳轮':>5}  {'差距':>6}")
    print(f"{'=' * 100}")

    for exp_id, config, results in experiments:
        desc = config.get("description", config.get("desc", ""))[:28]
        d_model = config.get("d_model", "?")
        lr = config.get("lr", "?")
        stories = config.get("data_stories", "?")
        tags = ",".join(config.get("tags", []))[:14]
        model_str = f"d={d_model} lr={lr} |{stories}s {tags}"

        best_val = results.get("best_val_loss")
        ppl = results.get("perplexity")
        epochs_actual = results.get("epochs_actual")
        best_epoch = results.get("best_epoch") if results.get("best_epoch") else "-"
        train_loss = results.get("final_train_loss")
        val_loss = results.get("final_val_loss")
        gap = round(val_loss - train_loss, 2) if isinstance(train_loss, (int, float)) and isinstance(val_loss, (int, float)) else "-"

        print(f"{exp_id:>10}  {desc:>28}  {model_str:>30}  "
              f"{fmt(best_val):>8}  {fmt(ppl):>8}  "
              f"{fmt(epochs_actual):>5}  {str(best_epoch):>5}  {fmt(gap):>6}")


def print_deltas(experiments, baseline_id="001"):
    """打印相对 baseline 的变化"""
    baseline = None
    others = []
    for exp_id, config, results in experiments:
        if baseline_id in exp_id:
            baseline = results
        else:
            others.append((exp_id, config, results))

    if not baseline:
        print(f"\n⚠️  未找到 baseline ({baseline_id})")
        return

    b_val = baseline.get("best_val_loss", 0)
    b_ppl = baseline.get("perplexity", 0)
    b_ep = baseline.get("epochs_actual", 0)

    print(f"\n{'=' * 100}")
    print(f"相对 baseline ({baseline_id}) 的变化（负数=变好 ✅，正数=变差 ❌）")
    print(f"{'=' * 100}")
    print(f"{'ID':>10}  {'描述':>28}  {'Val变化':>10}  {'PPL变化':>10}  {'轮次变化':>8}")
    print("-" * 70)

    for exp_id, config, results in others:
        desc = config.get("description", config.get("desc", ""))[:28]
        val = results.get("best_val_loss")
        ppl = results.get("perplexity")
        ep = results.get("epochs_actual")

        val_d = f"{val - b_val:+.2f}" if isinstance(val, (int, float)) else "?"
        ppl_d = f"{ppl / b_ppl:.1%}" if isinstance(ppl, (int, float)) and b_ppl else "?"
        ep_d = f"{ep - b_ep:+d}" if isinstance(ep, int) else "?"

        print(f"{exp_id:>10}  {desc:>28}  {val_d:>10}  {ppl_d:>10}  {ep_d:>8}")


def print_generated(experiments, max_len=120):
    """打印生成文本"""
    print(f"\n{'=' * 100}")
    print("生成文本对比")
    print(f"{'=' * 100}")
    for exp_id, config, results in experiments:
        gen = results.get("generated", "")
        if gen:
            desc = config.get("description", config.get("desc", ""))[:28]
            print(f"\n  {exp_id} {desc}:")
            print(f"    {gen[:max_len]}")


def print_details(experiments):
    """打印某次实验的详细信息"""
    for exp_id, config, results in experiments:
        print(f"\n{'=' * 60}")
        print(f"实验: {exp_id}")
        print(f"{'=' * 60}")
        print(f"\n配置:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print(f"\n结果:")
        for k, v in results.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实验对比工具")
    parser.add_argument("--tags", nargs="+", help="按标签筛选 (e.g. --tags lr-test baseline)")
    parser.add_argument("--ids", nargs="+", help="按 ID 筛选 (e.g. --ids 001 005)")
    parser.add_argument("--last", type=int, help="只看最近 N 次")
    parser.add_argument("--table", choices=["brief", "full"], default="full",
                        help="显示模式 (brief/full)")
    parser.add_argument("--detail", action="store_true", help="显示实验完整配置和结果")
    parser.add_argument("--no-delta", action="store_true", help="不显示变化量")
    parser.add_argument("--no-gen", action="store_true", help="不显示生成文本")
    parser.add_argument("--baseline", default="001", help="指定 baseline ID")
    args = parser.parse_args()

    experiments = load_all()
    experiments = filter_experiments(experiments, args.tags, args.ids, args.last)

    if args.detail:
        print_details(experiments)
    else:
        print_table(experiments, args.table)
        if not args.no_delta:
            print_deltas(experiments, args.baseline)
        if not args.no_gen:
            print_generated(experiments)
