# 实验记录

每次训练实验的配置和结果归档在 `runs/` 目录下。

## 查看方式

```bash
# 列出所有实验
ls experiments/runs/

# 查看某次实验的配置
cat experiments/runs/001_baseline/config.json

# 查看某次实验的结果
cat experiments/runs/001_baseline/results.json
```

## 实验列表

| ID | 名称 | 描述 | 最佳 Val Loss | 困惑度 |
|----|------|------|--------------|--------|
| 001 | baseline | 基准配置，60 epoch | 4.19 | 13335 |
| 002 | small_model | d_model=32，减少参数量 | 5.28 | — |
| 003 | low_lr | lr=1e-3，学习更慢 | 5.04 | — |
| 004 | high_lr | lr=1e-2，学习太快 | 6.85 | — |
| 005 | early_stop | 基准+早停+最佳模型保存 | **3.73** | **41** |

## 关键结论

1. **早停效果最明显**：005 比 001 的 Val Loss 从 4.19 降到 3.73，PPL 从 13335 降到 41
2. **学习率要适中**：003（lr=1e-3）比 001（lr=3e-3）略好，但 004（lr=1e-2）最差
3. **小模型抗过拟合**：002（d_model=32）的 Val-Train 差距（3.05）小于 001（5.21）
4. **早停后只用 9/60 轮**：训练时间节省 85%
