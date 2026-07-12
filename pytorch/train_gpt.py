"""
GPT 训练脚本 — 在 TinyStories 上训练迷你语言模型

用法：
  python -m pytorch.train_gpt                           # 默认参数
  python -m pytorch.train_gpt --d_model 128 --lr 1e-4   # 自定义参数
  python -m pytorch.train_gpt --d_model 32 --epochs 20  # 小模型快速验证

训练完成后自动保存实验记录到 experiments/runs/
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def train(d_model=64, num_layers=4, d_ff=None, num_heads=4, num_kv_heads=2,
          lr=3e-3, weight_decay=0.01, num_epochs=60, batch_size=8,
          max_len=64, patience=5, data_limit=2000,
          _tag="", _desc=""):
    if d_ff is None:
        d_ff = d_model * 2
    params = {k: v for k, v in locals().items()}

    from .data import create_dataloaders, decode, load_stories_from_file
    data_file = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tinystories.txt")
    all_stories = load_stories_from_file(data_file)
    stories = all_stories[:data_limit]
    print(f"使用 {len(stories)} 个故事（共 {len(all_stories)} 个）")

    train_loader, val_loader, word2idx, idx2word = create_dataloaders(
        stories=stories, batch_size=batch_size, max_len=max_len,
    )
    vocab_size = len(word2idx)

    from .llama_block import GPT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = GPT(
        vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,
        num_heads=num_heads, num_kv_heads=num_kv_heads, d_ff=d_ff,
        max_seq_len=max_len + 32, use_rope=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"配置: d_model={d_model}, layers={num_layers}, lr={lr}, "
          f"batch={batch_size}, max_len={max_len}, data={data_limit}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    os.makedirs("checkpoints", exist_ok=True)

    best_val = float('inf')
    best_epoch = 0
    wait = 0

    print(f"\n开始训练 {num_epochs} epoch...")
    print(f"早停耐心值: {patience}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>8} | {'LR':>8} | {'早停':>6}")
    print("-" * 50)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                val_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        stopping = ""
        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), "checkpoints/gpt_best.pt")
            stopping = "↓"
        else:
            wait += 1
            stopping = f"{wait}/{patience}"
            if wait >= patience:
                print(f"{epoch:>6} | {avg_train:>10.4f} | {avg_val:>8.4f} | "
                      f"{scheduler.get_last_lr()[0]:>8.2e} | {stopping:>6}")
                print(f"\n早停触发！最佳 Val Loss: {best_val:.4f} (epoch {best_epoch})")
                break

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} | {avg_train:>10.4f} | {avg_val:>8.4f} | "
                  f"{scheduler.get_last_lr()[0]:>8.2e} | {stopping:>6}")

    # 加载最佳模型
    print(f"\n加载最佳模型（Val Loss: {best_val:.4f}）...")
    model.load_state_dict(torch.load("checkpoints/gpt_best.pt"))

    # 生成示例
    prompts = ["once upon a time"]
    model.eval()
    generated = ""
    for prompt in prompts:
        from .data import encode_prompt
        input_ids = encode_prompt(prompt, word2idx, max_len).to(device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=20, temperature=0.8)
        result = decode(output_ids[0].tolist(), idx2word)
        print(f"\n  输入: {prompt}")
        print(f"  输出: {result}")
        generated = result

    # 困惑度
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            non_pad = (y != 0).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
    perplexity = float(torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item())
    print(f"\n困惑度 (Perplexity): {perplexity:.2f}")

    # 自动保存实验记录
    _save_experiment_log({**params, "_tag": _tag, "_desc": _desc}, {
        "best_val_loss": round(best_val, 4),
        "best_epoch": best_epoch,
        "final_train_loss": round(avg_train, 4),
        "final_val_loss": round(avg_val, 4),
        "perplexity": round(perplexity, 2),
        "epochs_actual": epoch,
        "generated": generated,
    })


def _save_experiment_log(params, results):
    """自动保存实验记录，目录名反映参数变化"""
    import json
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 从参数生成描述性目录名（突出与默认值不同的参数）
    parts = [ts]
    tag = params.pop("_tag", "")
    desc = params.pop("_desc", "")

    # 如果有 tag，优先作为目录名标识
    if tag:
        parts.append(tag)

    defaults = dict(d_model=64, num_layers=4, d_ff=None, num_heads=4,
                    num_kv_heads=2, lr=0.003, weight_decay=0.01,
                    num_epochs=60, batch_size=8, max_len=64,
                    patience=5, data_limit=2000)
    for k, v in params.items():
        if k in ("_tag", "_desc", "params"):
            continue
        if k == "d_ff" and v is None:
            continue
        dv = defaults.get(k)
        if k == "d_ff":
            dv = params.get("d_model", 64) * 2
        if v != dv:
            short = {"d_model": "d", "num_layers": "L", "num_heads": "H",
                     "num_kv_heads": "KV", "lr": "lr", "weight_decay": "wd",
                     "num_epochs": "ep", "batch_size": "bs", "max_len": "ml",
                     "patience": "pat", "data_limit": "N"}.get(k, k)
            parts.append(f"{short}{v}")
    parts.append("auto" if len(parts) > 1 else "default")

    runs_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments", "runs")
    os.makedirs(runs_dir, exist_ok=True)

    exp_name = "_".join(parts)
    exp_dir = os.path.join(runs_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 自动打标签
    tags = ["auto"]
    if params.get("d_model", 64) != 64:
        tags.append(f"d{params['d_model']}")
    if abs(params.get("lr", 0.003) - 0.003) > 1e-6:
        tags.append(f"lr{params['lr']}")
    if params.get("data_limit", 2000) != 2000:
        tags.append(f"N{params['data_limit']}")
    if tag:
        tags.append(tag)

    config = {
        "description": desc or f"train_gpt (d={params.get('d_model',64)}, "
                       f"lr={params.get('lr',0.003)})",
        "script": "train_gpt.py",
        "source": "auto",
        "date": ts,
        "tags": tags,
        **params,
    }
    # 移除内部变量
    config.pop("params", None)

    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(os.path.join(exp_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"📝 实验记录: experiments/runs/{exp_name}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPT 训练")
    parser.add_argument("--d_model", type=int, default=64, help="模型维度")
    parser.add_argument("--num_layers", type=int, default=4, help="层数")
    parser.add_argument("--lr", type=float, default=3e-3, help="学习率")
    parser.add_argument("--epochs", type=int, default=60, help="最大训练轮次")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--max_len", type=int, default=64, help="序列长度")
    parser.add_argument("--patience", type=int, default=5, help="早停耐心值")
    parser.add_argument("--data_limit", type=int, default=2000, help="故事数")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_kv_heads", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--tag", type=str, default="", help="实验标签（用于分类，如 lr-test）")
    parser.add_argument("--desc", type=str, default="", help="实验描述")
    args = parser.parse_args()

    tag = args.tag or ""
    desc = args.desc or ""

    # 交互式参数调整
    if not any(v for v in vars(args).values() if v not in (False, None, "", 64, 4, 2, 0.003, 0.01, 60, 8, 64, 5, 2000, None)):
        # 全默认 → 进入交互配置模式
        cfg = {
            "d_model": args.d_model, "num_layers": args.num_layers,
            "lr": args.lr, "epochs": args.epochs,
            "batch_size": args.batch_size, "max_len": args.max_len,
            "data_limit": args.data_limit, "patience": args.patience,
            "tag": tag, "desc": desc,
        }
        labels = {
            "d_model": "模型维度", "num_layers": "层数", "lr": "学习率",
            "epochs": "训练轮次", "batch_size": "批次大小", "max_len": "序列长度",
            "data_limit": "故事数", "patience": "早停耐心",
            "tag": "实验标签", "desc": "实验描述",
        }

        while True:
            print(f"\n{'=' * 55}")
            print("当前参数配置：")
            print(f"{'=' * 55}")
            print(f"{'编号':>4} | {'参数名':>15} | {'当前值':>12} | {'说明':>12}")
            print("-" * 55)
            keys = list(cfg.keys())
            for i, k in enumerate(keys, 1):
                v = cfg[k]
                print(f"{i:>4} | {k:>15} | {str(v):>12} | {labels.get(k, ''):>12}")
            print("-" * 55)
            print("  0) 开始训练")
            print("  q) 退出")
            cmd = input("\n输入编号修改参数，或 0 开始: ").strip().lower()
            if cmd == "q":
                print("已退出。")
                sys.exit(0)
            elif cmd == "0":
                break
            elif cmd.isdigit():
                idx = int(cmd)
                if 1 <= idx <= len(keys):
                    k = keys[idx - 1]
                    current = cfg[k]
                    new_val = input(f"  输入 {labels.get(k, k)} 新值 (当前={current}): ").strip()
                    if not new_val:
                        continue
                    # 类型转换
                    if k == "tag" or k == "desc":
                        cfg[k] = new_val
                    elif k == "lr":
                        cfg[k] = float(new_val)
                    else:
                        cfg[k] = int(new_val)
                    print(f"  ✅ {k} = {cfg[k]}")
                else:
                    print("  无效编号。")
            else:
                print("  无效输入。")

        # 应用配置
        d_model = cfg["d_model"]
        num_layers = cfg["num_layers"]
        lr = cfg["lr"]
        num_epochs = cfg["epochs"]
        batch_size = cfg["batch_size"]
        max_len = cfg["max_len"]
        data_limit = cfg["data_limit"]
        patience = cfg["patience"]
        tag = cfg["tag"]
        desc = cfg["desc"]
        print(f"\n开始训练: d_model={d_model}, lr={lr}, epochs={num_epochs}, tag={tag or '(无)'}")
    else:
        print(f"使用自定义参数: d_model={args.d_model}, lr={args.lr}, "
              f"epochs={args.epochs}, tag={tag or '(无)'}")

    train(
        d_model=args.d_model, num_layers=args.num_layers,
        d_ff=args.d_ff, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, lr=args.lr,
        weight_decay=args.weight_decay, num_epochs=args.epochs,
        batch_size=args.batch_size, max_len=args.max_len,
        patience=args.patience, data_limit=args.data_limit,
        _tag=tag, _desc=desc,
    )
