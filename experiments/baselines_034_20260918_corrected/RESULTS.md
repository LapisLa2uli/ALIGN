# 034.zip 修正后测试结果

036–040 已用 036-040.zip 整包替换。模型输入的 160 个文件与旧版逐一核对，均完全一致；复用已冻结的 80 个预测 MIDI，仅按新 gold 重新计分。实际标注变化为 036 新增 5 条错音、039 新增 3 条错音。

完整集仍为 40 条；剔除 005、007、010、012、013、020、026、030、034、036 后仍为 30 条。五类错误 gold 数分别为 46 和 21。数值为 0–1，非百分比。

| 集合 | 模型 | 匹配/预测/gold（50 ms） | Precision | Recall | F1@50 ms | F1@IoU≥0.3 |
|---|---|---|---:|---:|---:|---:|
| full40 | polytune | 7/7759/46 | 0.000902 | 0.152174 | 0.001794 | 0.002306 |
| full40 | laddersym | 4/5004/46 | 0.000799 | 0.086957 | 0.001584 | 0.001980 |
| filtered30 | polytune | 1/5147/21 | 0.000194 | 0.047619 | 0.000387 | 0.001548 |
| filtered30 | laddersym | 1/4024/21 | 0.000249 | 0.047619 | 0.000494 | 0.000989 |

新增的错音没有获得上述两种指标的匹配分，预测数和匹配数不变；gold 增加，因此 recall 和 F1 略降。

## F1@50 ms 的 95% clip-bootstrap 区间

| 集合 | 模型 | 下界 | 上界 |
|---|---|---:|---:|
| full40 | polytune | 0.000509 | 0.003233 |
| full40 | laddersym | 0.000335 | 0.003194 |
| filtered30 | polytune | 0.000000 | 0.001224 |
| filtered30 | laddersym | 0.000000 | 0.001462 |

五类为 wrong_note、missed_note、extra_note、rhythm_error、repetition。全部十类标注的补充视图分别有 63/32 条 gold，包含在 JSON 和 CSV 中。空标注继续按人工确认的无错误样本处理。

这些是时间匹配错误检测指标，仍不能与 paper 的 canonical combined-pipeline F1 直接比较。原来的位置标注审计问题仍存在；两个指定集合的 canonical 指标均为 unavailable，没有额外删除任何样本。

[完整 JSON](results.json) · [CSV](summary.csv) · [验证记录](verification.json) · [预测复用核验](prediction_reuse_audit.json)
