# 两个 baseline 修复后重训（2026-09-18）

用户授权：重新训练 Polytune 与 LadderSym，测试集沿用当前版本。完整集为 40 条；036–040 使用 `036-040.zip` 的替换版本。仍另报剔除 005、007、010、012、013、020、026、030、034、036 后的 30 条。空人工标签表示无错误。

## 固定实验设置

- 两套模型均从头初始化，seed=365，统一 30 epochs，不继承旧 best 权重。
- 沿用已经过标签/MIDI/render timing 审计的合成数据：10,049 train，1,116 validation。真实测试数据不参与训练、验证选模或调参。
- 训练关闭会破坏相邻同音高音符顺序的 token permutation；每条录音从完整时间范围均匀抽取一个 2.048s 窗口，绕过 16s 预切块，保留尾部与跨块上下文。
- 每条验证录音固定检查 5 个等间隔位置；val loss 用于训练诊断。
- 每 5 epochs 对固定的 32 条合成验证录音进行整曲自由生成。按原生 class-aware micro note F1（onset 50ms / pitch 50 cents）选择 best，包含 Correct、Extra、Missing；未分类预测计入 FP。同分保留较早 checkpoint。
- 32 条按 procedural/raw 各 16 条、每个来源错误密度四分位各抽 4 条，seed=365；ID 在 protocol.json 冻结。其余验证录音仍参与 5 位置的 teacher-forced 验证。
- 修复 LadderSym 首窗参考特征错误清零；另在 GPU 冒烟测试中发现并修复 Polytune 短音频首窗参考切片的广播错误。长音频首窗结果不变。
- 训练、完整生成验证和最终推理统一使用 bf16（原流程为 bf16 训练、float32 推理）。交叉精度记忆对照发现 LadderSym 单窗口在 bf16→bf16 和 float32→float32 均能完整还原，而交叉精度可能出错；这不是已经测出的真实集收益。
- 模型架构、lr=2e-5、warmup=4000 updates、error_loss_weight=8、effective batch=16、bf16-mixed、event/prompt budget=1024 沿用旧配置。每 GPU batch=1，梯度累积 16，避免占用其他任务的显存。

**输入适配边界**：本轮没有引入未经验证的动态 reference warping，也没有根据这 40 条录音的标签选择响度增益。16k mono、原输入幅度和独立参考时间轴沿用原协议。输入分布/进度差异仍可能影响真实集表现，不能宣称所有旧问题都已解决。本轮主要验证训练目标、采样、验证和推理边界修复。

## 代码与验证

`source/Polytune` 和 `source/LadderSym` 是本次专用代码快照，原始 baseline 源码和历史训练产物保留。`source_hashes.json` 固定正式启动所用代码。

- `{model}_fix_checks.json`：完整录音尾部可采样、5 位置验证、禁用增强后的相邻音符完整 round-trip、首窗训练/推理特征一致。
- `{model}_regressions.log`：现有回归测试，Polytune 14 pass / 3 skip，LadderSym 17 pass。
- `smoke/{model}/status.json`：实际 GPU 两个训练 batch、验证、自由生成、F1 计算及 best 权重保存流程。随机初始模型的 smoke F1 不是效果结论。
- `{model}_overfit.json`：独立初始化模型在一个训练窗口上的 150-step 记忆检查，使用诊断学习率 3e-4；这些权重不用于正式训练。
- `smoke_attempt1` 保留首次集成失败产物：LadderSym runner 未设置 num_steps_per_epoch，Polytune smoke 的两秒输入触发短窗广播错误；均在第二次 smoke 前修正。

## 长任务和自动评测

`supervisor.py` 在 GPU 0、1 可用显存超过 11GB 时分别启动两个训练进程；磁盘不足 12GiB 时暂停自己启动的进程，回升至 20GiB 后恢复，不操作其他任务。

- 总状态：`status.json`；监督日志：`supervisor.log`。
- 分模型状态：`training/{model}/status.json`；训练日志：`{model}_training.log`。
- 可恢复训练：`training/{model}/checkpoints/last.ckpt`。
- 按生成 F1 选出的权重：`training/{model}/best.pt`、`best.json`。
- 每五轮完整验证结果：`training/{model}/dev_predictions/epoch_XX/note_metrics.json`。
- 两个模型各自训练结束后，严格加载 best，对当前 40 条测试输入重新生成原生 MIDI。
- 两者都完成后，`finalize.py` 在独立进程中先冻结预测，再读取人工 GT，使用旧修正版的同一套转换和评分代码生成 `test_results/results.json`、`summary.csv`、`RESULTS.md`，报告 full40 与 filtered30。

最终真实集结果仍是时间匹配的错误事件诊断指标；不是论文 canonical combined-pipeline F1。
