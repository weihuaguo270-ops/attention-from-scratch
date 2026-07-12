"""
GPT 训练脚本 — 在 TinyStories 上训练迷你语言模型

用法：
  cd attention-from-scratch && python -m pytorch.train_gpt

训练完成后生成示例文本。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def train():
    import os
    # ============================================================
    # 1. 数据准备
    # ============================================================
    from .data import create_dataloaders, decode

    data_file = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tinystories.txt")

    # 只取前 2000 个故事
    from .data import load_stories_from_file
    all_stories = load_stories_from_file(data_file)
    stories = all_stories[:2000]
    print(f"使用 {len(stories)} 个故事（共 {len(all_stories)} 个）")

    batch_size = 8
    max_len = 64

    train_loader, val_loader, word2idx, idx2word = create_dataloaders(
        stories=stories, batch_size=batch_size, max_len=max_len,
    )
    vocab_size = len(word2idx)

    print(f"词表大小: {vocab_size}")
    print(f"训练样本: {len(train_loader.dataset)}")
    print(f"验证样本: {len(val_loader.dataset)}")

    # ============================================================
    # 2. 模型创建
    # ============================================================
    from .llama_block import GPT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = GPT(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        d_ff=128,
        max_seq_len=max_len + 32,  # 预留生成空间
        use_rope=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # ============================================================
    # 3. 训练配置
    # ============================================================
    optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad>

    os.makedirs("checkpoints", exist_ok=True)

    num_epochs = 60
    print_every = 5
    patience = 5      # 连续几次 val loss 不降就停
    best_val = float('inf')
    best_epoch = 0
    wait = 0

    print(f"\\n开始训练 {num_epochs} epoch...")
    print(f"早停耐心值: {patience}（val loss 连续 {patience} 次不降即停止）")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>8} | {'LR':>8} | {'早停等待':>8}")
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

        # 验证
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

        # 早停 + 保存最佳模型
        stopping_msg = ""
        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), "checkpoints/gpt_best.pt")
            stopping_msg = "↓ 保存"
        else:
            wait += 1
            stopping_msg = f"{wait}/{patience}"
            if wait >= patience:
                print(f"{epoch:>6} | {avg_train:>10.4f} | {avg_val:>8.4f} | "
                      f"{scheduler.get_last_lr()[0]:>8.2e} | {stopping_msg:>8}")
                print(f"\\n早停触发！最佳 Val Loss: {best_val:.4f} (epoch {epoch - patience})")
                break

        if epoch % print_every == 0 or epoch == 1:
            print(f"{epoch:>6} | {avg_train:>10.4f} | {avg_val:>8.4f} | "
                  f"{scheduler.get_last_lr()[0]:>8.2e} | {stopping_msg:>8}")

    # 加载最佳模型用于生成
    print(f"\\n加载最佳模型（Val Loss: {best_val:.4f}）...")
    model.load_state_dict(torch.load("checkpoints/gpt_best.pt"))

    # ============================================================
    # 5. 生成示例
    # ============================================================
    print(f"\n{'='*50}")
    print("训练完成！生成示例：")
    print(f"{'='*50}")

    prompts = ["once upon a time", "there was a", "the little"]
    model.eval()
    generated = ""  # 用于实验记录

    for prompt in prompts:
        from .data import encode_prompt
        input_ids = encode_prompt(prompt, word2idx, max_len).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, max_new_tokens=20, temperature=0.8
            )

        result = decode(output_ids[0].tolist(), idx2word)
        print(f"\\n  输入: {prompt}")
        print(f"  输出: {result}")
        if not generated:
            generated = result

    # 计算困惑度
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            # 计算非 pad token 数量
            non_pad = (y != 0).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    print(f"\\n验证集困惑度 (Perplexity): {perplexity:.2f}")
    print(f"预期: <50 说明模型已学到基本语言模式")

    # 自动记录实验
    _save_experiment_log(
        config={
            "d_model": model.token_embedding.embedding_dim,
            "num_layers": len(model.layers),
            "d_ff": model.layers[0].swiglu.W_gate.out_features,
            "num_heads": model.layers[0].self_attn.num_heads,
            "num_kv_heads": model.layers[0].self_attn.num_kv_heads,
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0]["weight_decay"],
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "max_len": max_len,
            "early_stop": patience > 0,
            "patience": patience if patience > 0 else None,
            "data_stories": len(stories),
        },
        results={
            "best_val_loss": round(best_val, 4),
            "best_epoch": best_epoch,
            "final_train_loss": round(avg_train, 4),
            "final_val_loss": round(avg_val, 4),
            "perplexity": round(perplexity, 2),
            "epochs_actual": epoch,
            "generated": generated,
        }
    )


def _save_experiment_log(config, results):
    """自动保存实验配置和结果到 experiments/runs/"""
    import json
    from datetime import datetime

    runs_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments", "runs")
    os.makedirs(runs_dir, exist_ok=True)

    # 自动编号
    existing = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    next_id = f"{len(existing) + 1:03d}"
    exp_dir = os.path.join(runs_dir, f"{next_id}_auto")
    os.makedirs(exp_dir, exist_ok=True)

    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "config": config,
        "results": results,
    }

    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**config, "timestamp": log["timestamp"]}, f, indent=2, ensure_ascii=False)

    with open(os.path.join(exp_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\\n📝 实验已自动记录到 experiments/runs/{next_id}_auto/")


if __name__ == "__main__":
    train()
