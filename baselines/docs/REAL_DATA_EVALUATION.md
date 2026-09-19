# 真实录音 baseline F1 与使用流程

当前结果来自 2026-09-18 修复后从头训练的 Polytune 和 prompted LadderSym。
入口是 `experiments/baselines_retrain_20260918/`。两个 best checkpoint 均选自第
30 epoch，选择依据是固定 32 条**合成验证录音**的 class-aware note F1，真实集不参与选模。

## 数据和指标

- `full40`：`034.zip` 的 001–040，036–040 用 `036-040.zip` 整包替换。
- `filtered30`：排除 005、007、010、012、013、020、026、030、034、036。
- 人工 `labels.json` 是 GT；空标签表示无错误，仍参与评估。full40 有 19 条完全无错误录音。
- 主表只统计 wrong_note、missed_note、extra_note、rhythm_error、repetition 五类错误，
  两个集合分别有 46、21 条 GT。正常音符不进入这个分母，GT 原样保留，包括重复标注。
- 两个模型原生输出 Extra/Missing/Correct，不能原生预测 rhythm_error；该类 GT 仍计入主表。
  JSON 另报去掉 rhythm_error 的 `shared_four_type` 和包含全部人工类型的 `all_annotated_types`。

真实集没有完整的逐音符演奏真值，所以 `baselines/common/evaluate_notes.py` 对
`real_test=True` 返回 `micro=null`。推理 loader 使用的空标签 MIDI 是占位文件，
不能把针对这些文件打印的分数当成真实集 F1。真实集的谱面位置标注仍有审计失败项，
因此 canonical note-wise / combined-pipeline 分数为 unavailable。
这里可用的是**按时间匹配的错误事件 micro-F1**。

## 从 MIDI 预测到错误事件

`experiments/baselines_034_20260918_corrected/evaluate.py::freeze` 按 MIDI track 名
读取类别，然后调用 `baselines/common/eval_bridge.py::notes_to_spans`：

1. Extra 与 Missing 的 onset 互为最近邻、间距不超过 100 ms 时，合成一个 wrong_note，
   使用 Extra 的起止时间，并消耗这两个预测。当前真实集转换**没有额外要求音高不同**。
2. 未配对的 Missing 各自成为 missed_note。
3. 剩余 Extra 中，至少 3 个连续音符、相邻 onset 间距小于 350 ms，且与前文音高序列的
   LCS 匹配比例至少为 0.8 的片段可成为 repetition。Correct 会打断 Extra 序列；
   搜索前文窗口为候选片段时长的 2 倍，首尾最多各裁剪 3 个音符；
   相隔小于 2 秒的 repetition 区间会合并。完整常量见评分脚本的 `PARAMETERS`。
4. 其他 Extra 各自成为 extra_note。Correct 只辅助重复检测，不直接成为评测事件。
5. 无类别名的解码音符保留为 unclassified，只增加预测分母，不获得匹配分。
   带非空未知类别名的 track 会报错。

转换不读取人工标签，单独进程冻结预测 JSON、MIDI/代码/checkpoint 哈希后，评分进程才读取 GT。
这里的 100 ms **转换窗口**与下面的 50 ms **评分容差**是不同参数。

## F1 如何计算

每条录音内部，预测与 GT 必须**类型相同**且满足时间条件，然后用最大数量的一对一匹配
（`scipy.optimize.linear_sum_assignment`）求命中数。一个预测或 GT 最多使用一次，
命中得 1 分，没有半分；最终匹配不检查音高、offset（onset 指标）或谱面位置。

| 指标 | 合格条件 |
|---|---|
| `onset_50ms`，原始主诊断指标 | onset 绝对差 ≤ 0.05 秒 |
| `onset_100ms` / `onset_200ms` | onset 绝对差 ≤ 0.1 / 0.2 秒 |
| `iou_0.3` / `iou_0.5` | 两个时间区间的交集长度 / 并集长度 ≥ 0.3 / 0.5 |

将所选集合所有录音的计数相加：命中数为 M、预测数为 N、GT 数为 G。

```text
TP = M       FP = N - M       FN = G - M
Precision = M / N
Recall    = M / G
micro-F1  = 2M / (N + G)
```

分母为 0 时相应值取 0。先汇总计数再计算 F1，不是平均每条录音的 F1。
无错误录音上的所有错误预测均计入 FP；不统计 true negatives。
另报的 macro-F1 是有 GT 支持的类型的 F1 均值。
95% 区间采用录音级 bootstrap，2,000 次，seed=365。

例如当前 full40 Polytune 的 onset_50ms：M=7、N=6924、G=46，
所以 F1=14/6970=0.002009，即 **0.2009%**。主要问题是预测错误事件远多于人工 GT。

| 集合 | 模型 | 命中/预测/GT，50 ms | F1@50 ms | F1@IoU≥0.3 |
|---|---|---|---:|---:|
| full40 | Polytune | 7 / 6924 / 46 | 0.002009 | 0.002009 |
| full40 | LadderSym | 4 / 3747 / 46 | 0.002109 | 0.003164 |
| filtered30 | Polytune | 3 / 4779 / 21 | 0.001250 | 0.001667 |
| filtered30 | LadderSym | 1 / 2600 / 21 | 0.000763 | 0.002289 |

表内数值为 0–1。结果摘要见
[重训结果](../../experiments/baselines_retrain_20260918/test_results/RESULTS.md)。

## 使用流程

以下命令都在 ALIGN 仓库根目录运行。模型环境按 [GPU runbook](GPU_RUNBOOK.md) 安装。
本机评分环境为 `align-model/runs/env/bin/python`，需要 numpy、scipy、mido、pretty_midi
和 eval_bridge 的依赖。Git 保存源码、配置与说明；音频、标签数据、权重、冻结预测和
机器运行记录需另外传输。仅 clone 源码不能直接重算已有实验。

### 1. 直接复算当前真实集结果（已有预测，无需 GPU）

先确认以下本地产物存在：

- `experiments/baselines_retrain_20260918/protocol.json` 中 `test_gold` 指向修正后的实验目录。
- `experiments/baselines_retrain_20260918/test_results/` 下有 `dataset_manifest.json`、
  `label_audit.json`、`bundles/<id>/labels.json`，以及两个 `<model>_frozen/` 目录。
- `bundles` 若为符号链接，目标存在；搬机器后需要恢复链接和协议中的绝对路径。
- 冻结 manifest 保存的评分代码和预测哈希通过检查。不要为了绕过检查修改历史哈希。

```sh
align-model/runs/env/bin/python \
  experiments/baselines_retrain_20260918/finalize.py --stage score
```

输出写入 `experiments/baselines_retrain_20260918/test_results/`：

- `RESULTS.md`：full40 / filtered30 主表。
- `results.json`：逐录音、逐类、所有时间条件和置信区间。
- `summary.csv`：扁平汇总。

读取主指标的 JSON 路径：

```text
models.polytune.subsets.full40.results.five_type.onset_50ms.micro
models.laddersym.subsets.full40.results.five_type.iou_0.3.micro
```

### 2. 从当前重训模型的新 MIDI 预测开始

原实验 `run_model.py` 在训练结束后严格加载 `training/<model>/best.pt`，自动对完整
40 条录音推理，写入 `training/<model>/test_predictions/<id>/mix.mid` 和 `evaluated_ids.json`。
使用该实验的 `source/Polytune`、`source/LadderSym` 快照与 bf16 autocast、batch=1、
event/prompt budget=1024、确定性 prompt；旧通用 wrapper 的 float32 推理不等价于本次重训协议。
`run_model.py --model ...` 会启动训练，不是单独推理命令，复算现有结果无需运行它。

输入准备使用 `baselines/common/prepare_dataset.py --real-test`，例如迁移后从修正版 bundles
创建一个新的本地输入目录：

```sh
align-model/runs/env/bin/python baselines/common/prepare_dataset.py \
  --set zip034=experiments/baselines_034_20260918_corrected/bundles \
  --out baselines/data/real_034_corrected --real-test --workers 4
```

这一步转换 16k mono 模型输入和参考 MIDI，不生成真实音符 GT。重训配置的 `test_data`
必须指向实际输入目录。checkpoint、推理精度、数据版本和全体 40 个 ID 应先固定。

两个模型推理完成，且目标 `test_results/<model>_frozen` 尚不存在时，分进程执行：

```sh
align-model/runs/env/bin/python experiments/baselines_retrain_20260918/finalize.py --stage prepare
align-model/runs/env/bin/python experiments/baselines_retrain_20260918/finalize.py --stage freeze --model polytune
align-model/runs/env/bin/python experiments/baselines_retrain_20260918/finalize.py --stage freeze --model laddersym
align-model/runs/env/bin/python experiments/baselines_retrain_20260918/finalize.py --stage score
```

`prepare` 读取两份 `training/<model>/best.json`，建立输入、GT 与 checkpoint 记录；
`freeze` 检查完整 40 条预测及 checkpoint 哈希，拒绝覆盖已有冻结目录。
已有冻结结果只执行步骤 1；新实验应使用独立实验目录保存自己的协议和产物。
当前脚本固定为这两个 40/30 子集，换真实数据集时需先显式定义新 ID、GT 和审计记录。

### 3. 区间合并敏感性分析

```sh
align-model/runs/env/bin/python \
  experiments/baselines_retrain_20260918/interval_eval.py
```

该脚本读取上述 `test_results` 的冻结预测，结果写入 `interval_results/`。
分别报告不合并，以及 gap=0、50、100、200、500 ms：同类预测的相交、相接或小间隔区间
递归取并集；不同类型不合并，unclassified 仍各自计为 FP。GT 不合并，匹配和 micro 公式不变。
**0 ms 也会合并相交/相接区间，和不合并不同。**
这是额外的事后敏感性分析，以 IoU 为主要区间指标；所有 gap 都报告，不根据真实集挑最佳阈值。
基础 `notes_to_spans` 自带的 repetition 合并在“不合并”设置下也仍然存在。
脚本会校验原始数据/预测哈希，并要求“不合并”精确复现步骤 1 的三个共同指标。
见 [区间结果](../../experiments/baselines_retrain_20260918/interval_results/RESULTS.md)。

扩展分析使用 `interval_eval.py --extended`，另外报告 500、1000、2000、5000、10000 ms
以及每条录音内同类全部合并，输出到 `interval_results_extended/`，保留默认分析结果。
“同类全部合并”不跨录音、不跨类型，unclassified 仍单独计数。

## 实现位置

- MIDI → 错误事件：`baselines/common/eval_bridge.py::notes_to_spans`。
- 冻结、最大数量匹配、micro 和 bootstrap：
  `experiments/baselines_034_20260918_corrected/evaluate.py`。
- 当前重训实验入口：`experiments/baselines_retrain_20260918/finalize.py`。
- 区间合并：`experiments/baselines_retrain_20260918/interval_eval.py`。
- 合成数据原生 note 指标：`baselines/common/evaluate_notes.py`，onset 50 ms、pitch 50 cents、
  按 Extra/Missing/Correct 分别匹配；`class_aware` 保留类别，`all` 忽略类别。
