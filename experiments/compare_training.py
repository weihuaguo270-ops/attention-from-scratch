"""
超参数对比实验 — 不同模型大小和正则化对训练的影响

用法:
  python -m experiments.compare_training

输出: 各配置的 loss 曲线 + 生成样本对比
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_one_config(name, d_model=64, num_layers=4, dropout=0.0,
                     lr=3e-3, epochs=20, batch_size=8):
    """用给定配置训练一次，返回 loss 记录和生成样本"""

    # 数据（所有配置共用，确保可比）
    from pytorch.data import load_stories_from_file, build_vocab, encode, TinyStoriesDataset
    from torch.utils.data import DataLoader, random_split

    data_file = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tinystories.txt")
    all_stories = load_stories_from_file(data_file)[:500]  # 500 个故事
    word2idx, idx2word = build_vocab(all_stories)
    vocab_size = len(word2idx)

    max_len = 64
    dataset = TinyStoriesDataset(all_stories, word2idx, max_len)
    train_ds, val_ds = random_split(dataset, [400, 100])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # 模型
    from pytorch.llama_block import GPT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT(
        vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,
        num_heads=4, num_kv_heads=2, d_ff=d_model * 2,
        max_seq_len=max_len + 16, use_rope=True,
    ).to(device)

    # 如果要加 dropout，给 LlamaDecoderBlock 加 dropout 参数
    # 这里演示效果，直接调整模型大小和训练参数

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        scheduler.step()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                v_loss += loss.item()

        train_losses.append(t_loss / len(train_loader))
        val_losses.append(v_loss / len(val_loader))

    # 生成样本
    from pytorch.data import encode_prompt
    prompt = "once upon a time"
    input_ids = encode_prompt(prompt, word2idx, max_len).to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=15, temperature=0.8)
    from pytorch.data import decode
    sample = decode(output_ids[0].tolist(), idx2word)

    return {
        "name": name,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "sample": sample,
        "params": sum(p.numel() for p in model.parameters()),
        "vocab": vocab_size,
    }


def print_results(results):
    """打印对比结果"""
    print(f"\n{'='*70}")
    print(f"{'配置':>20} | {'参数':>8} | {'词表':>6} | {'最终Train':>10} | {'最终Val':>10}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['name']:>20} | {r['params']:>8,} | {r['vocab']:>6} | "
              f"{r['train_loss'][-1]:>8.3f}  | {r['val_loss'][-1]:>7.3f}")
    print()

    # 打印 loss 曲线（文本版）
    print(f"{'Epoch':>6}", end="")
    for r in results:
        print(f" | {r['name']:>15}", end="")
    print()

    for e in range(len(results[0]['train_loss'])):
        print(f"{e+1:>6}", end="")
        for r in results:
            print(f" | {r['train_loss'][e]:>8.3f}  {r['val_loss'][e]:>7.3f}", end="")
        print()

    print(f"\n{'='*70}")
    print("生成样本对比")
    print(f"{'='*70}")
    for r in results:
        print(f"\n  {r['name']}:")
        print(f"    {r['sample']}")


if __name__ == "__main__":
    print("超参数对比实验\n")
    print("配置说明:")
    print("  A: 基准 (d_model=64, lr=3e-3)")
    print("  B: 小模型 (d_model=32, lr=3e-3) — 减少参数量")
    print("  C: 基准 + 低学习率 (d_model=64, lr=1e-3)")
    print("  D: 基准 + 高学习率 (d_model=64, lr=1e-2)")
    print()

    results = []
    for name, d_model, lr in [
        ("A: 基准", 64, 3e-3),
        ("B: 小模型", 32, 3e-3),
        ("C: 低LR", 64, 1e-3),
        ("D: 高LR", 64, 1e-2),
    ]:
        print(f"正在训练 {name}...")
        r = train_one_config(name, d_model=d_model, lr=lr, epochs=20)
        results.append(r)
        print(f"  完成: params={r['params']:,}")

    print_results(results)
