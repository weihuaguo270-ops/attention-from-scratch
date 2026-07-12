# 实验记录系统

每次训练的结果记录在 `runs/` 下。

## 目录命名规则

| 来源 | 命名格式 | 示例 |
|------|---------|------|
| 旧实验（手动归档） | `{序号}_{描述}/` | `001_baseline/` |
| 新实验（自动记录） | `{时间戳}_auto/` | `20260711_143021_auto/` |

时间戳永不重复，不会出现命名冲突。

## 对比查看

```bash
# 查看全部实验
python -m experiments.runs.compare

# 只看学习率实验（按标签筛选）
python -m experiments.runs.compare --tags lr-test

# 只看特定实验（按 ID 筛选）
python -m experiments.runs.compare --ids 001 005

# 只看最近 2 次
python -m experiments.runs.compare --last 2

# 简表模式（只看关键数字）
python -m experiments.runs.compare --table brief
```

## 实验记录

每个实验目录包含两个文件：

**config.json** — 实验配置

```json
{
  "description": "实验目的描述",
  "script": "train_gpt.py",
  "source": "自动记录",
  "date": "2026-07-11",
  "tags": ["auto", "d64", "s2000"],
  "d_model": 64,
  "lr": 0.003,
  ...
}
```

**results.json** — 实验结果

```json
{
  "best_val_loss": 3.73,
  "best_epoch": 4,
  "perplexity": 41.52,
  "epochs_actual": 9,
  "generated": "once upon a time..."
}
```

## 实验列表

```bash
python -m experiments.runs.compare --table brief
```
