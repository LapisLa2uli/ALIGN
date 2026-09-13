# ALIGN：Polytune 与 LadderSym baselines

这套接入使用作者代码，在 ALIGN 的冻结合成数据上训练、推理和评估三个 note class：Extra、Missing、Correct。LadderSym 包含 prompted（额外输入 reference MIDI）和 unprompted 两种设置。

**先读 [GPU 服务器运行文档](docs/GPU_RUNBOOK.md)**，包含代码/数据迁移、安装、smoke test、正式训练、断点续训、评估、显存不足及常见错误处理。

[复现检查记录](docs/REPRODUCTION_AUDIT.md) 说明本次修复、测试证据、与作者实验的区别和目前尚未验证的部分。这里完成的是 ALIGN 数据上的 baseline 接入；不能把 smoke test 或在 ALIGN 上重新训练称为已经复现论文表格分数。

## 当前基线

| 项目 | 固定代码版本 | 执行方式 |
|---|---|---|
| ALIGN | 上游基础版本 `2424130` | baseline 接入随主仓库提交；运行时记录实际 checkout 的 commit |
| [Polytune](https://github.com/ben2002chou/Polytune) | `d2055bb21759d457c8f21c1cf2e47c79af6248f5` + 补丁 | `scripts/polytune_train.sh` / `polytune_eval.sh` |
| [LadderSym](https://github.com/ben2002chou/LadderSym) | `381179754cf6bcb435f9decf0d5e24eada6c68ec` + 补丁 | `scripts/laddersym_train.sh` / `laddersym_eval.sh` |

作者代码保持固定版本；**不要在嵌套目录直接 `git pull`**，否则完整补丁可能不再适用。`scripts/bootstrap.sh` 会在干净 clone 上应用 `patches/*.patch` 并复制 `configs/`，重复运行不会重复打补丁，遇到冲突会退出。

## 文件组织

- `scripts/`：可从任意工作目录运行的安装、训练、评估和迁移入口。
- `common/`：数据转换、数据检查、严格权重加载和类别正确的评估。
- `configs/`：被版本管理的 ALIGN 配置；bootstrap 复制到作者仓库。
- `patches/`：作者仓库的完整修改及说明，包含推理修复。
- `envs/requirements.*.server.txt`、`*.server.lock.txt`：Python 3.11 的服务器依赖及约束。
- `tests/`：解码、checkpoint、标签与评估回归测试。
- `splits/align_v1/`：保留的原始 split、审计版 split 和 86 条排除理由。
- `Polytune/`、`LadderSym/`、`envs/{polytune,laddersym}/`：可重建的本地目录，Git 忽略。
- `data/`、`runs/`：本地数据和结果，Git 忽略；迁移时需要单独传输。

## 数据版本必须固定

本机已有 `data/align_v1`：原始 12,000 tracks，标签来自旧的 `note_labels.json`。全量监督语义审计排除了 86 条存在未映射参考音符/音符计数不一致等问题的样本，保留 **11,914 条：train/validation/test = 10,725/594/595**。原数据及 `split.json` 没有改写；训练和评估默认优先使用新增的 `split.audited.json`。这份数据不依赖当前生成器源码即可训练。建议先迁移并使用它，不要为运行 baseline 把旧生成器改动重新套回主仓库。

最新 origin 的生成器输出 `note_map.json`，与 `note_labels.json` 语义不同，尤其没有直接提供完整的 performance timeline 上的 missed-note 时间。转换器遇到这种输入会明确报错，**不会静默丢掉数据或生成假的监督标签**。需要重新生成训练数据时，应先单独实现并验证新 schema 的标签导出；现有冻结数据的运行不需要这一步。

`data/align_v1` 的 raw 部分来自少量相同乐谱、采用 per-clip split，不适合宣称跨作品泛化。要做跨作品实验，使用 `prepare_dataset.py --split-by-source raw` 建立新数据版本，并让所有对比模型使用同一划分。该选项在不足三个已知 source 时会报错。

历史 Mac 日志说明见 [旧记录](docs/LEGACY_20260909.md)，不作为本次验收依据。
