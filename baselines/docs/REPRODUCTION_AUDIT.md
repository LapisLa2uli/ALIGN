# Baseline 复现检查与修复记录（2026-09-13）

2026-09-15 补充：完整验证集评估发现 prompted LadderSym 的生成前缀
缺少训练时在 padded prompt 后追加的、attention mask 为 1 的起始 token。
已修复生成前缀，并加入实际 checkpoint 的训练/推理输入一致性检查。
可选 KV cache 保留位置偏移，且仅在首次解码时设置 prompt 内部的双向 attention；
小模型逐步等价测试和两个完整样本的 MIDI 字节一致性检查通过。
最终评估额外验证未分类音符：不按 track 顺序猜测类别，并将其计入区分类别的
总体 F1 的假阳性。LadderSym 当前 17 项回归测试全部通过；Polytune 通过 14 项，
跳过 3 项 LadderSym 专属测试。此次修改不改变 checkpoint 权重。
完整协议、checkpoint 轮次和结果见
[2026-09-15 F1 记录](../../experiments/baselines_f1_20260915/README.md)。

## 结论与范围

Polytune 与 LadderSym 的 ALIGN 接入已做代码审计和实际运行检查。本次恢复了主仓库到最新 origin；修改限制在 `baselines/`，不需要对主模型或 synth generator 再打旧补丁。当前结果证明数据/训练/保存/续训/推理/评估链路的工程正确性，不证明长时间 GPU 训练收敛，也不等于作者论文表格的数值复现。

主仓库从 `4a14317` fast-forward 到 `b451d6a`，同步了 origin 的 3 个提交。原有 4 个 tracked 文件改动保存在 stash `pre-baseline-repair local synth changes 2026-09-13`；源码 patch、baseline 修复前快照和 4 个旧的未跟踪 synth 源文件/测试保存在项目旁的 `ALIGN-backup-20260913-072330/`。原始数据和旧训练记录均保留。

同日随后再次同步 origin，从 `b451d6a` fast-forward 到 `2424130`（`Changed model and GUI`），没有合并冲突。主仓库 tracked 文件与 origin 一致；同步前后核对的 51 项 baseline 文件、作者源码 diff 和数据划分校验值全部一致。两套环境各重新运行 14 个回归测试：LadderSym 全通过，Polytune 跳过 1 个 LadderSym 专属测试，其余通过；日志为 `runs/audit_20260913/origin_2424130_*_regressions.log`。上游改动的 37 个 Python 文件语法解析通过，bridge 所依据的 `is_repeated_pass` 函数没有变化。以下完整训练/续训 smoke 记录仍是此前在 `b451d6a` 时的验证记录；此次同步未重跑完整 smoke 或 CUDA 训练。迁移文档和传输包已更新为 `2424130`。

作者代码固定为 Polytune `d2055bb21759d457c8f21c1cf2e47c79af6248f5`、LadderSym `381179754cf6bcb435f9decf0d5e24eada6c68ec`。来源和官方协议：[Polytune 仓库](https://github.com/ben2002chou/Polytune)、[LadderSym 仓库](https://github.com/ben2002chou/LadderSym)。`patches/*.patch` 是相对这些提交的完整 diff；ALIGN config 由 `configs/` 单独跟踪，不再依赖遗漏的 untracked 配置。

## 本次确认并修复的问题

| 问题 | 影响 | 修复 |
|---|---|---|
| 8 个运行入口写死 Mac 绝对路径 | 新服务器找不到解释器、数据或源码 | 按脚本位置解析 baseline 根目录；支持显式环境变量覆盖 |
| 两套 `_to_event` 用 `np.argmax` 找 EOS | 没生成 EOS 时返回 0，清空整段长度截断的预测 | 只有存在 EOS 时才切片；回归测试覆盖有/无 EOS |
| 两套 inference 吞掉异常 | 缺少预测仍然返回成功，可能沿用旧 MIDI | 异常传播；禁止重复 tag；检查评估 split 与预测全集 |
| LadderSym 最后一段 prompt 的 event index 用 0 补齐 | 尾段 end index 变成 0，丢掉有效 prompt | event index 按边界延伸，音频仍补零；尾段回归测试 |
| `.pt` 评估 `strict=False` | 不匹配权重可能全部被忽略而继续评估随机模型 | 统一严格加载 raw / DDP / Lightning state dict |
| LadderSym warm start 把 inner state dict 加到 Lightning wrapper | `model.` key 不匹配；路径为 `.pt` 时还可能原地覆盖输入 | 加载到 inner model，检查全部 key/shape，不写源 checkpoint |
| LadderSym wrapper 缺少 resume 入口 | 热启动与恢复 optimizer/epoch 混淆 | 新增 `--resume`，与 warm start 互斥；增加 batch-size、dry-run |
| 学习率下限被误当作绝对数值 | LambdaLR 实际乘以 `1e-5`，峰值 2e-5 时会降到约 2e-10 | ALIGN 的 `optim.min_lr=0.5`，即最低为峰值一半；保留作者 scheduler 函数语义 |
| scheduler 用曲目数/batch 估算更新步数 | 梯度累积、GPU 数、smoke batch 限制改变 schedule | 两套 tasks 使用 Lightning `estimated_stepping_batches` |
| Polytune 没有 error token 时训练日志除零 | 出现 `train_loss_inst=nan`，误导训练诊断 | 空 error-token 集合记录为 0；不修改作者 weighted CE 公式 |
| 验证集仍做随机片段/随机 token 操作 | checkpoint 选择的 val_loss 有额外随机波动 | ALIGN validation 采用 deterministic chunk、关闭 token randomization |
| 作者评估按 track 位置对齐类别 | 空的 Extra/Missing track 被 MIDI reader 删除后发生错位 | 新 `note_metrics.json` 按 track 名匹配，保留零类分母 |
| first-n 使用未固定随机偏移 | 多个模型可能评估不同曲目 | 按 track_id 稳定排序，保存实际评估 id 和 split 文件 |
| bridge 未合并 missed-note ties | converter 和 note 真值定义可能不一致 | bridge 与 converter 共用类别/连音合并逻辑 |
| 旧标签 MIDI 自检通过但 reference/performance 语义不一致 | 部分缺失音符没有 Missing 标签，连音错误解释不唯一 | 扫描全部 12k 源 JSON，新增审计 split，排除 86 条；保留原文件/划分，转换器也执行同样检查 |
| converter 接受未知 class 或默默跳过新 schema | 静默丢监督音符/数据 | schema/类别检查；明确拒绝将 note_map 当 note_labels |
| source split 不满足条件时回退 per-clip | 用户以为隔离作品，实际发生泄漏 | 不足三个已知 source 时直接报错 |
| 检查脚本根据 manifest 内旧输出绝对路径读取 | 迁移后报文件不存在，甚至误读原位置 | 输出路径从当前 DATA_ROOT 推导，支持 `--set NAME=NEW_ROOT` 定位源 bundles |

还保留并重新打包了原接入的必要修复：CPU/MPS device fallback、Polytune 不等长音频的 frame-time grid、输出 MIDI 类别名、prompt budget 配置、LadderSym deterministic prompt、编译与 debug 开关。没有把模型架构或两套不同的 weighted CE 公式统一成新的算法。

## 实际验证证据

本机为 macOS arm64，Python 3.11，torch 2.3.0；模型 smoke 运行 profile=cpu。

| 检查 | 结果 / 可复查产物 |
|---|---|
| 主仓库同步 | `git diff origin/master --` 为空；没有重新应用旧 synth 改动 |
| 全量转换数据结构 | 原始 12,000 tracks 的 6 个必要文件及 16 kHz mono 等长音频检查通过；这不代表监督语义全部正确 |
| 全量监督语义审计 | 检查所有源 note_labels JSON：排除 86 条，保留 11914 条（10725 train / 594 validation / 595 test）；保留原 split，只新增 split.audited.json；具体理由在 splits/align_v1/supervision_audit.json |
| 审计版 split SHA-256 | `598faa8f2402e34b7b0395363d0596b3e1f24d7448fa76aa8e4bb41fc7281c88` |
| 原始 split SHA-256 | `7c5586b202ff6802544e1b460ffe3af4a1056b6e9def9ed4ff8c22d3e3bc849e` |
| 源标签核对 | 28/28 smoke bundles；审计版正式数据按 seed=365 从 multi/raw 各随机抽 30 条，共 60/60 MIDI onset/pitch 与 reference 计数检查通过；原始抽样中的失败日志也保留 |
| 官方 Dataset | 两套各检查 14 条 train items；target token 解码后的类别/pitch/onset 与标签 MIDI 匹配，`RESULT: OK` |
| 新转换器 | 在独立临时目录转换 6 条冻结 bundles，0 errors；不导入被恢复/移走的旧生成器源码 |
| 回归测试 | 每个环境运行 14 个测试；LadderSym 全通过，Polytune 跳过 1 个仅 LadderSym 的 prompt 测试，其余全通过 |
| Polytune CPU train | `runs/polytune/repair_polytune_20260913/`；2 train batches + 1 val batch，生成 `.ckpt` 和 `.pt` |
| LadderSym prompted CPU train | `runs/laddersym/repair_prompted_20260913/`；同上，prompt_length=256 |
| LadderSym unprompted CPU train | `runs/laddersym/repair_unprompted_20260913/`；同上 |
| 断点恢复 | 两套 `repair_resume_20260913/`；从 epoch=0 / global_step=2 恢复，完成 epoch=1 / global_step=4 |
| LadderSym `.pt` 热启动 | `runs/laddersym/repair_warm_start_20260913/`；严格加载导出的 inner model 权重，再从 epoch 0 开始训练 |
| 评估链路 | 两套 `.pt` 或 `.ckpt` 重载；smoke test split 的 1 条音频，生成 MIDI、`evaluated_ids.json`、`note_metrics.json` 和可选 bridge JSON |
| Prompt 对齐 | 3 个 smoke 片段的首段训练 prompt 与 deterministic inference prompt token 完全相同；另有尾段索引回归测试 |
| Note oracle | 28 条样本，Extra 1025 / Missing 40 / Correct 1230，三类及 all 的 micro P/R/F1 均为 1.0；见 `runs/audit_20260913/oracle_metrics.json` |
| 依赖 | CPython 3.11 / Linux x86_64 两套服务器 requirements 与全部传递约束均成功解析；CUDA runtime 由选定 torch wheel 提供 |
| 干净源码重建 | 新目录 clone 固定提交，最终源码 diff 与 patch 逐字节一致、所有 config 一致；bootstrap 第二次运行识别补丁已应用 |
| 全新环境安装 | 两套 server requirements + server.lock 在新建 Mac venv 安装，pip check 与 MPS 矩阵前向/反向通过（有 uv 时优先使用，pip 可回退）；Linux x86_64 另做依赖解析 |
| 干净目录完整 smoke | 在新 clone、新 venv、独立转换数据目录执行 smoke_test.sh cpu：三个 variant 均训练、严格重载 .ckpt、推理和 note_metrics 通过，最终 PASS；完整日志已复制到 runs/audit_20260913/align-clean-full-smoke.log |

测试命令和日志保存在 `runs/audit_20260913/`，测试后的源补丁与 `configs/` 同步。smoke 的目标 token 长度为 128，推理上限为 16；F1/空预测不用于任何效果结论。作者底层 token parser 仍会计数并跳过非法生成事件，未收敛模型的解析诊断可能包含 traceback；顶层运行异常则传播并使流程失败。新增 `scripts/smoke_test.sh` 可以在目标 GPU 重新执行同类流程。

## 旧训练到底跑到了哪里

旧 README 的“正在训练 / 未跑正式训练”描述已过时。检查实际日志后：

- `laddersym_prompted_align_v1_mac_20260909-005531` 完成了 3 epochs、16200 updates，末次 val_loss 约 1.89323，保存了 `last.pt`。
- `polytune_align_v1_mac_20260909-005531` 在 epoch 0 的约 1537/5400 batches 后终止；日志末尾只有资源清理提示，不能据此断言具体终止原因。
- 未发现对应三 epoch 的 unprompted 完整训练结果；此前有两种 LadderSym 的 smoke 日志。
- 旧 LadderSym 的 3 条样本评估中 correct micro-F1 约 4.65%，all 约 1.96%。这是旧代码/短训练/小子集结果，不能作为修复后效果；也不能靠训练 loss 下降认定 baseline 已复现。

旧 checkpoint 和历史结果保留，可以辅助诊断；正式比较应使用修复后的流程重新训练/评估，并记录它们不同的 prompt budget、LR schedule 和初始化方式。

## 不能掩盖的实验边界

1. **数据版本**：现有 12k 数据来自旧 `note_labels.json`，在本次 pull 之前已经生成。最新 origin 的 `note_map.json` 是另一种 lineage schema，不足以直接给出所有 missed-note 的 performance-time onset。为了保持 origin 干净，本次使用冻结监督数据，新增 schema 的标签导出没有凭启发式冒充精确真值。
2. **作品泄漏**：raw 部分来自少量乐谱，现有划分是 per-clip。同谱片段可能跨 split，不能声称跨作品泛化。若做跨作品实验，需要重新按 source 分组，并核对真实测试曲目与训练来源的交集。
3. **模型输入对齐**：参考音频和参考 MIDI没有经过 performance-time warping。补零不等于对齐；长 repetition/tempo changes 会让相同时间窗口指向不同谱面位置。改变这个设置是新的 adaptation 实验。
4. **任务不可完全等价**：三类 note 输出没有直接的 rhythm/intonation 类别，重复通过 Extra 启发式重建。即使输入 oracle note MIDI，28 条样本的 span-F1 也远低于满分。因此主要比较类别正确的 note 指标；span 比较须注明可表达类别与映射方法。
5. **原论文协议**：这里默认从 scratch 在 ALIGN 上训练 40 epochs，数据、切分、batch、验证确定性、LR 下限等均属于 ALIGN 实验选择。要复现作者已发表数字，还需作者数据/初始化/原实验配置及足够训练。作者公布的 robust-piano 变体也不等同于原论文 checkpoint。
6. **硬件**：本机没有 CUDA GPU，本次未完成目标服务器上的 CUDA/bf16、显存或吞吐验证，未完成全量 40 epochs。文档给出了目标机器的 doctor、smoke 和正式训练命令，硬件适配以目标机器实际测试为准。
