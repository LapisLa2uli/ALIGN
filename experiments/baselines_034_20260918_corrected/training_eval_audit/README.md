# 训练与评测链路审计（2026-09-18）

结论：确实存在可复现的训练目标构造 bug、LadderSym 推理特征 bug，以及明显偏向开头的验证/采样设置。当前低分不能直接归结为 baseline 方法本身差，也不能归结为 F1 公式算错。尚未通过修复后重新训练测定各因素的最终影响。

本次仅写入此审计目录，使用原 best checkpoint 做诊断；没有修改生产模型源码、训练数据、人工标签、正式预测或论文。审计依据当前源码、保存的 Hydra 配置和实际数据；没有历史每一步训练 token 的录像，故随机增强的历史损伤比例不能直接由本次抽样推断。下列推理对照在独立进程内运行。

## 1. 已复现：随机 token 重排破坏同音高音符的 on/off 顺序

两套训练配置均开启 `is_randomize_tokens=true`。`randomize_tokens()` 对同一时间点的事件组直接随机排列，没有保留同一类别、同一音高的 note-off → note-on 依赖。

最小例子：两个 Correct / MIDI 65 音符，时间分别为 [0.20, 0.50] 和 [0.50, 0.80]。正常目标在 0.50 秒先关闭旧音再开启新音；重排可能先开新音再关旧音。原生 decoder 因而立即关闭新音，最短时长保护将其输出为 [0.50, 0.51]，0.80 秒的 note-off 又成为非法事件。

- 两套真实 tokenizer/decoder 的 100 个随机种子试验，均有 **45 次**改变还原音符；关闭重排正常还原两个音符。
- 12 条固定抽样样本 × 3 个种子：两套各出现 2 次还原不一致，均为 `synth_gen_5623`，音符 [1.18, 1.78] 被变成 [1.18, 1.19]。
- 另按 seed=365 随机抽取 100 条真实训练样本、每条按实际采样方法取一个窗口，再各用 3 个增强种子：Polytune 的 **300 次中 26 次**改变音符，**100 个窗口中 14 个**至少一次受影响。这是一个固定样本审计，不是全训练过程受损率。
- 全量源标签中 8,738/10,049 条训练录音含同类同音高相接的潜在边界；不表示每条录音实际抽到的窗口都会受损。

这解释了为何之前的“类别/音高/onset 计数一致”检查会漏检：token 里的 note-on 仍存在，错误发生在成对事件解码后的 duration / 活跃音符状态上。

源码：`baselines/Polytune/dataset/dataset_2_random.py:818`、`baselines/LadderSym/dataset/dataset_2_random.py:932`。

建议修复：先禁用此增强验证，或将随机排列限制为不破坏同一 (class, pitch) 的 note-off → note-on 顺序；新增完整音符 round-trip 检查，而不只查 onset 计数。已学到的错误监督不会因单独更换推理解码器而自动消失，需重新训练对照。

证据：[Polytune 最小复现](polytune_target_reproducer.json)、[LadderSym 最小复现](laddersym_target_reproducer.json)、[100 条训练窗口抽样](augmentation_sample.json)。

## 2. 已复现：LadderSym 首窗的有效参考 Mel 被清零

首个参考窗口有 128 个左侧 padding 帧和 384 个有效音频帧，合计 512 帧。推理记录 `score_paddings.append(score_slice_len)`，只记录 384，后续用 `int(p/2)+2` 清尾，漏算了左侧 padding 对有效末端位置的影响。

因此压缩后的参考 Mel 第 **194–255 列（62/256 列）**被清零，约对应参考音频 2.08–3.07 秒的有效未来上下文。问题主要作用于首窗；不能据此解释整条录音所有错误。

- 12 条样本：Polytune 首窗两路特征与 Dataset **逐元素相同**；LadderSym 演奏特征相同，但 **12/12 条参考特征不同**。
- LadderSym 同 12 条的 prompt token 及 attention mask 均与 Dataset 相同，当前 prompt 前缀修复也通过原有回归测试。
- 独立进程恢复误清零的首窗后，抽查参考特征与 Dataset 的最大绝对差为 **0**。
- 扩展到 4 条合成录音与 2 条真实录音，每条检查首窗、第二窗、中间窗、最后一个完整窗（每模型 24 窗）：Polytune 两路特征 24/24 相同；LadderSym 演奏 24/24 相同、参考 18/24 相同，6 个差异全部局限于首窗的 194–255 列。这不包含最后不足 256 帧的残余窗，也不检查训练 2000 帧预切块边缘引入的上下文裁切。

原 checkpoint、原音量、原对齐，仅更改首窗参考特征的局部生成对照：

| 录音 | 原首窗 Extra / Missing / Correct | 恢复后首窗 Extra / Missing / Correct |
|---|---|---|
| 004 | 0 / 0 / 3 | 0 / 0 / 3 |
| 009 | 5 / 0 / 0 | 4 / 0 / 0 |
| 033 | 16 / 0 / 0 | 14 / 0 / 0 |

这是首个 2.048 秒窗口的原生预测数，不是整曲 F1；没有宣称该修复解决了整体失败。

源码：`baselines/LadderSym/inference_error.py:373` 与 `:588`。修复方向：padding 记录应是有效内容在补齐窗口中的结束位置，即 `start_padding + score_slice_len`，不能仅用有效内容长度。

证据：[局部修正对照](padding_control.json)、[12 条训练/推理检查](laddersym_probe.json)。

## 3. 已确认：选 checkpoint 的验证只看录音开头

保存的配置是 `num_rows_per_batch=1`、validation `is_deterministic=true`。Dataset 先取第一个 2000 帧块，再取该块开头 256 帧，因此每条录音每次验证只覆盖约 **0–2.048 秒**，不是整曲。

1,116 条验证数据的源监督音符数（按 tokenizer 的起始窗口边界统计）：

| 类别 | 整条录音 | 开头验证窗口 | 覆盖率 |
|---|---:|---:|---:|
| Correct | 37,681 | 4,389 | 11.65% |
| Extra | 27,161 | 202 | **0.74%** |
| Missing | 2,029 | 219 | 10.79% |

开头窗口中 91.25% 的音符是 Correct，整体验证标签中则为 56.35%。这些是音符分布，不是 CE token 的权重分布；不能把二者混同。由于后段错误几乎没有进入验证，最低 teacher-forced `val_loss` 无法有效选择整曲错误检测最好的 checkpoint。

本次在两套 checkpoint 各 12 条固定样本、每条两个窗口的诊断中，正确历史前缀条件下的 token accuracy 约为 72.54% / 72.20%。这不是 F1，也不是完整验证性能。8 个窗口的独立自由生成分别出现 12 / 76 个非法事件；LadderSym 有 1 个窗口达到 256 token 诊断上限，正式评测的上限是 1024。不存在“teacher loss 看起来下降，因此完整生成已可靠”的证据。

建议：每条验证录音覆盖固定的多个位置或全部窗口；定期运行完整自由生成和 class-aware note F1，用实际任务指标选 checkpoint。

## 4. 已确认：训练尾部被永久排除

两套 `_split_frame()` 都执行 `if split + length >= input_length: continue`。已有完整 2000 帧块时，不足 16 秒的尾块被丢弃；只有整条不足一个块时才回退保留。后续随机采样只在保留下来的块里进行。

按当前音频长度和该代码路径计算，训练集 **599,825 个原生监督音符中，84,467 个（14.08%）的起始点位于永远采不到的尾部**；其中 Extra 45,472 个。验证源数据也有约 14.01% 位于这种尾部，但实际验证覆盖更少，如上一节。统计不包含那些起始点在保留区、仅尾音越界的音符，也不代表音频尾部全部有声音。

建议保留最后不足一块的样本，按实际长度 pad，并对音频/目标/prompt 同步处理。

## 5. 输入分布与任务适配存在额外问题

- **未对齐的参考时间轴**：全量合成数据中，训练集 48,304/334,980 个 Correct 音符（14.42%），在当前固定推理窗口规则下，其对应参考音符完全不在输入参考上下文内。验证集为 5,245/37,681（13.92%）。这使用源数据的明确 lineage，不依赖真实录音自动对齐结果。具体例子：`synth_001_0081` 的正常音符发生在演奏 10.35 秒，对应参考 4.80 秒，而当前参考窗口为 9.216–13.312 秒。训练随机窗口的覆盖关系不完全相同；这些数字明确针对固定推理窗口。
- **真实参考音量偏小**：此前抽样训练的演奏/参考 RMS 比中位数 1.26，真实集 30.89。此前 +20dB 对照让 Polytune 004 的 Extra 198→54、Correct 8→156，但 LadderSym 004 反而增加错误。参考响度有影响，统一放大尚不是已验证的通用修复。
- **错误先验不同**：训练集仅 137/10,049 条无 Extra/Missing 监督（约 1.36%，仍可能包含三类表达不了的错误）；真实集 19/40 条人工确认完全无错误（47.5%）。这两个“无错误”的定义不完全相同，但显示训练的错误密度与测试明显不同。
- **输出到人工标签的转换**：原生 Extra 包含错音、插入音和重复音，固定启发式才能拆开；人工区域级标注与逐音符预测粒度不完全一致。原生没有节奏错误类。50ms 严格时间匹配是诊断指标，不能直接当作原论文同口径结果。

这些因素与代码 bug 可能叠加，但未完成独立消融，不能给它们分配精确的因果贡献比例。

## 6. 已排查且未发现的问题

- checkpoint 使用 strict=True，实际权重完整加载；未发现随机模型替代或类号反转。
- Extra / Missing / Correct 的 vocab ID 实测为 1135 / 1136 / 1137，训练与命名输出对应一致。
- 两套各 12 条样本的确定性训练目标 onset/class/pitch 与 label MIDI 一致；这不覆盖上面的 duration/事件次序 bug。
- Polytune 首窗 Mel parity 12/12 通过；LadderSym 演奏 Mel 和 prompt parity 12/12 通过。
- 033 的原生 Extra 为 Polytune 197、LadderSym 96；转换后 extra_note 为 192、88。评分器没有制造大量 Extra。
- 现有回归测试：Polytune 14 项通过、3 项 LadderSym 专属测试跳过；LadderSym 17 项通过。上述新发现说明旧测试覆盖不足，不能用旧测试通过证明整个算法正确。

## 修复与重评顺序

1. 修复/禁用会破坏 note-on/off 次序的训练增强；完整 note round-trip 必须通过。
2. 修复 LadderSym 首窗 padding，覆盖首窗/中间窗/尾窗的训练与推理 parity。
3. 保留训练尾块，扩大验证位置覆盖，按自由生成任务指标选 checkpoint。
4. 明确参考音频和 MIDI 在训练/推理两端的时间对齐及响度协议；在开发集上确定，不用这 40 条真实测试标签调参。
5. 用极小训练集过拟合、相同音频双输入、无错误录音误报率等基础对照，确认模型能完成基本任务，再完整重训和冻结重评。

现在直接延长旧训练、修改 F1 容差，或只修首窗后宣布复现成功，都不足以解决已发现的问题。

## 可复查产物

- `probe.py` / `{model}_probe.json`：12 条固定样本，每条两个监督诊断窗口；训练/推理特征、prompt、token 和原生生成检查。
- `reproduce_target_bug.py` / `{model}_target_reproducer.json`：无权重的最小 on/off 次序复现。
- `augmentation_sample.py` / `augmentation_sample.json`：100 条真实训练样本、300 次增强 round-trip。
- `padding_control.py` / `padding_control.json`：LadderSym 首窗隔离修正。
- `population.py` / `population.json`：11,165 条数据的标签覆盖、尾部和参考上下文分析。
- `window_parity.py` / `{model}_window_parity.json`：首窗、第二窗、中间窗、最后完整窗的特征对照。
- `{model}_regressions.log`：现有回归检查。

运行各模型 probe / reproducer 需使用对应 `baselines/envs/{model}/bin/python`。`population.py` 使用 `align-model/runs/env/bin/python`。所有脚本均可从 ALIGN 根目录运行，输出在此目录；重新运行会覆盖本次诊断文件，不覆盖正式评测。
