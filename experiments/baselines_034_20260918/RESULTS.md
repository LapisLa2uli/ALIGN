> 已由 [036–040 替换后的修正版](../baselines_034_20260918_corrected/RESULTS.md) 取代。以下保留原始结果用于追溯。

# 034.zip 测试结果

两个模型均完成全部 40 条推理；剔除组复用同一批预测，只改变汇总名单。数值均为 0–1，非百分比。

主表采用五类错误（错音、漏音、多音、节奏、重复）的 typed micro 指标；正常音符不参与。人工确认的空标签按无错误样本统计误报。

| 集合 | 模型 | 匹配/预测/gold（50 ms） | Precision | Recall | F1@50 ms | F1@IoU≥0.3 |
|---|---|---|---:|---:|---:|---:|
| full40 | polytune | 7/7759/38 | 0.000902 | 0.184211 | 0.001796 | 0.002309 |
| full40 | laddersym | 4/5004/38 | 0.000799 | 0.105263 | 0.001587 | 0.001983 |
| filtered30 | polytune | 1/5147/18 | 0.000194 | 0.055556 | 0.000387 | 0.001549 |
| filtered30 | laddersym | 1/4024/18 | 0.000249 | 0.055556 | 0.000495 | 0.000990 |

剔除编号：005、007、010、012、013、020、026、030、034、036。

低分主要来自误报：完整集 21 条完全无错误标注的录音上，Polytune 输出 3,932 条错误事件，LadderSym 输出 2,678 条。原生输出大多被预测为 Extra：Polytune 8,130 个 Extra / 250 个 Correct，LadderSym 4,884 个 Extra / 1,070 个 Correct。这里只陈述观察结果，没有据此调整阈值或重新选择模型。

## F1@50 ms 的 95% clip-bootstrap 区间

| 集合 | 模型 | 下界 | 上界 |
|---|---|---:|---:|
| full40 | polytune | 0.000510 | 0.003241 |
| full40 | laddersym | 0.000336 | 0.003197 |
| filtered30 | polytune | 0.000000 | 0.001225 |
| filtered30 | laddersym | 0.000000 | 0.001464 |

## 包含全部十类标注的补充视图

这个视图另计 click、bad_start、squeak、sliding、bad_timbre，完整/剔除组分别有 55/29 条 gold；两个模型不会输出这些额外类别。

| 集合 | 模型 | F1@50 ms | F1@IoU≥0.3 |
|---|---|---:|---:|
| full40 | polytune | 0.001792 | 0.002304 |
| full40 | laddersym | 0.001581 | 0.001977 |
| filtered30 | polytune | 0.000386 | 0.001546 |
| filtered30 | laddersym | 0.000493 | 0.000987 |

## 指标与验证限制

这些是时间匹配错误检测指标，不能与 paper 的 canonical combined-pipeline F1=0.6104 直接比较。完整演奏逐音符真值缺失；五类 gold 的位置审计在完整集中失败的编号为 002、005、008、013、014、016、030、035，在剔除组中仍有 002、008、014、016、035，因此两个指定集合的 canonical 分数均记为 unavailable，没有擅自删去这些样本。

原生解码器按既有规则跳过非法 token 事件，日志中分别出现 4,281 / 16,736 条 invalid-event 记录。所有 40 个 MIDI 均已写出且可读取；这不是样本漏跑。预测哈希、checkpoint 哈希、源文件（200 项）、两组名单和独立评估器匹配计数均已核对。

详见 [results.json](results.json)、[summary.csv](summary.csv)、[protocol.json](protocol.json)、[verification.json](verification.json)。复现说明见 [README.md](README.md)。
