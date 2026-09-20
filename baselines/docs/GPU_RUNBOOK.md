# GPU 服务器运行文档

本文在项目根目录执行命令，路径不包含任何本机用户名。`B` 必须是绝对路径。实际 CUDA 训练尚未在本次 Mac 环境验证；Linux 依赖已解析，训练/保存/重载/推理由本机 CPU 测试验证。先在目标机器跑 doctor 和 smoke test，再启动长时间训练。

## 1. 把代码和数据带过去

baseline 接入基于已同步的上游 `2424130`（包含后续的 model 和 GUI 更新），运行时应使用包含 `baselines/` 的后续提交。旧的主仓库改动保留在 Git stash，以及本机项目旁的 `ALIGN-backup-20260913-072330/`。不要直接 `stash pop`，它会重新改变已恢复的生成器。

baseline 源码、补丁、配置、依赖、审计划分及文档随主仓库版本管理，可直接 clone。**数据、训练结果和环境不在 Git 中**；下面的传输包用于连同 28 条冻结 smoke 数据一起迁移。正式数据仍需单独 rsync。

在本机 ALIGN 项目目录：

```bash
python3 baselines/scripts/export_bundle.py \
  --out output/align-baselines-transfer.tar.gz --include-smoke
# 把 USER、GPU_HOST、/srv/ALIGN 换成自己的账户、地址和目标目录。
scp output/align-baselines-transfer.tar.gz USER@GPU_HOST:/tmp/
scp output/align-baselines-transfer.tar.gz.sha256 USER@GPU_HOST:/tmp/
```

在 GPU 服务器：

```bash
git clone --branch master https://github.com/LapisLa2uli/ALIGN /srv/ALIGN
cd /srv/ALIGN
git rev-parse HEAD  # 将实际 commit 保存到实验记录；不要回退到尚无 baseline 的 2424130。
(cd /tmp && sha256sum -c align-baselines-transfer.tar.gz.sha256)
tar -xzf /tmp/align-baselines-transfer.tar.gz -C /srv/ALIGN
export B=/srv/ALIGN/baselines
```

`--include-smoke` 包含 28 条冻结标签源及转换好的 smoke 数据。作者仓库、环境、正式数据和 checkpoints 不在代码包内。

接着在本机传正式转换数据（rsync 中断后可以执行同一命令继续）：

```bash
rsync -avh --progress baselines/data/align_v1/ \
  USER@GPU_HOST:/srv/ALIGN/baselines/data/align_v1/
# 可选：真实录音，只用于推理；没有三类音符真值。
rsync -avh --progress baselines/data/real_test/ \
  USER@GPU_HOST:/srv/ALIGN/baselines/data/real_test/
```

**不要复制 Mac 的 venv。** `manifest.json` 里保留旧的 source 路径不会影响训练及 `note_metrics.json`，两者通过新 `DATA_ROOT` 寻找文件。要计算原 ALIGN span 指标或核对源 MIDI，还需要原始 bundles；见第 7 节。

Git 跟踪 baseline 的 `common/configs/docs/envs/patches/scripts/tests/splits/README.md/.gitignore`。嵌套作者仓库、环境和大型数据由 baseline 的 `.gitignore` 排除，作者源码由 bootstrap 按固定版本和补丁重建。

## 2. 安装环境和设备检查

需要 Linux x86_64、NVIDIA 驱动、Git、Python **3.11**（含 `venv`）。固定 PyTorch 2.3.0；默认使用官方 CUDA 12.1 wheel，不需要为它额外安装整个系统 CUDA toolkit。可选 `--device cu118`；CPU 用 `cpu`；Mac 用 `mac`。安装依据：[PyTorch 历史版本官方命令](https://pytorch.org/get-started/previous-versions/)。

```bash
nvidia-smi
python3.11 --version
bash "$B/scripts/bootstrap.sh" --device cu121
# 如果系统解释器名字不同：
# BASELINE_SYSTEM_PYTHON=/path/to/python3.11 bash "$B/scripts/bootstrap.sh" --device cu121
```

如果系统已有 `uv`，bootstrap 会优先用它安装依赖；没有则使用 pip。可设置 `BASELINE_INSTALLER=pip` 强制使用 pip。无需复制或激活 Mac 的 venv。

bootstrap 会做以下工作：固定作者 commit，应用完整补丁，复制 ALIGN 配置；为两个模型分别创建 venv；安装对应 torch/torchvision/torchaudio 和锁定依赖；执行 `pip check`、关键库导入以及设备矩阵乘法/反向传播；最后保存 `envs/<model>/installed.freeze.txt`。

```bash
"$B/envs/polytune/bin/python" "$B/common/doctor.py" --device cuda
"$B/envs/laddersym/bin/python" "$B/common/doctor.py" --device cuda
```

输出应有 `device: cuda` 和 `forward_backward: OK`，记录 GPU、显存、CUDA wheel 版本和 `bf16`。当前默认配置用 `bf16-mixed`。如果 `bf16: false`，所有训练命令后面追加 `-- trainer.precision=16-mixed`；已有 `--` 时把该 override 加到它后面。

若非常新的 GPU 报 `no kernel image` 等错误，这套 torch 2.3 wheel 不适配该设备。不要仅靠 `torch.cuda.is_available()` 判断，也不要单独把 Transformers 升级到最新版：作者代码复制过 T5 实现，新版 API 不兼容。使用能通过 doctor 的设备，或者单独验证新的整套依赖组合后重跑回归测试。

服务器 pin 与原作者依赖的主要差异：LadderSym 的 Transformers 4.40.1、Hydra 1.3.0、pretty-midi 0.2.10、protobuf 3.20.3；这解决 Python 3.11、NumPy 和旧 protobuf 兼容问题。服务器另固定 numba 0.59.1 / llvmlite 0.42.0，避免依赖漂移。旧 Mac freeze 仅供溯源，安装使用 `*.server.*`。

## 3. 先做数据检查和小规模验收

```bash
export DATA="$B/data/align_v1"
"$B/envs/polytune/bin/python" "$B/common/audit_dataset.py" --data "$DATA"
# 预期使用 split.audited.json：11914 tracks、10725 train / 594 validation / 595 test。
# split_sha256: 598faa8f2402e34b7b0395363d0596b3e1f24d7448fa76aa8e4bb41fc7281c88

bash "$B/scripts/smoke_test.sh" cuda
```

smoke 会检查小数据集、运行回归测试和官方 loader、训练 Polytune / LadderSym prompted / unprompted 各 2 个 batch，验证 1 个 batch，保存并重载 `.ckpt`，推理 1 条 test 音频，生成 `note_metrics.json`。最后必须打印 `PASS`。日志和 checkpoints 在 `runs/<model>/smoke_*`，每个 variant 会产生数 GB checkpoint。

smoke 使用 `event_length=128`、LadderSym `prompt_length=256`、推理 `max_length=16`，仅用于检查连通性。短训练可能得到空 MIDI 或 F1=0；这不是模型效果验收。正式实验保留默认 1024 token 预算，不带 `--smoke`、`--first-n` 或缩短的 `--max-length`。

审计发现原始 12,000 条中 86 条监督语义存在问题，已保守排除；没有猜测或重写音符标签。`split.json` 和原数据完整保留，新增 `split.audited.json` 沿用其余样本原来的 train/validation/test 归属。训练、评估及 loader 检查默认优先使用审计版；显式 `--split-json` / `ALIGN_BASELINE_SPLIT_JSON` 会覆盖它。所有对比模型都应使用这一相同审计版划分。

完整 rsync 会带上审计文件；如果只迁移过旧数据，可以从本次代码包恢复相同划分：

```bash
cp "$B/splits/align_v1/split.audited.json" "$DATA/split.audited.json"
cp "$B/splits/align_v1/supervision_audit.json" "$DATA/supervision_audit.json"
```

要在已有源 bundles 的服务器重新执行标签审计：

```bash
"$B/envs/polytune/bin/python" "$B/common/audit_supervision.py" --data "$DATA" \
  --set multi=/srv/ALIGN/synth-pipeline/output_10k_multi \
  --set raw=/srv/ALIGN/synth-pipeline/output_2k_rawdata
```

正式数据的官方 loader 检查：

```bash
(cd "$B/Polytune" && MPLBACKEND=Agg "$B/envs/polytune/bin/python" \
  "$B/common/verify_loaders.py" --flavor polytune --root "$DATA" --split train --items 8)
(cd "$B/LadderSym" && MPLBACKEND=Agg "$B/envs/laddersym/bin/python" \
  "$B/common/verify_loaders.py" --flavor laddersym --root "$DATA" --split train --items 8)
```

预期 `RESULT: OK`。它会把官方 loader 的 token 解码回来，与对应片段的标签 MIDI onset 和 pitch 比较；不是只看张量形状。

## 4. 正式训练

以下从 scratch 开始，固定 seed 365、40 epochs；这是起始实验预算，不保证收敛。三个 variant 使用相同数据划分。建议分别在 tmux 会话中运行，一张 GPU 上依次执行。使用 `CUDA_VISIBLE_DEVICES` 选择卡。

先采用每个 batch 1 条曲目、每条曲目 1 个片段、梯度累积 16 步，降低未知服务器上的初始显存需求。可以根据显存增大 batch/rows 并相应减少 `grad_accum`，务必记录最终配置。片段数受曲目长度影响，不应只用 `batch_size * rows` 推断所有 batch 的实际大小。

```bash
export CUDA_VISIBLE_DEVICES=0
export DATA="$B/data/align_v1"

bash "$B/scripts/polytune_train.sh" \
  --data "$DATA" --profile cuda --epochs 40 --batch-size 1 \
  --run-name polytune_align_v1_s365 \
  -- num_rows_per_batch=1 grad_accum=16

bash "$B/scripts/laddersym_train.sh" \
  --data "$DATA" --profile cuda --prompted --epochs 40 --batch-size 1 \
  --run-name laddersym_prompted_align_v1_s365 \
  -- num_rows_per_batch=1 grad_accum=16

bash "$B/scripts/laddersym_train.sh" \
  --data "$DATA" --profile cuda --unprompted --epochs 40 --batch-size 1 \
  --run-name laddersym_unprompted_align_v1_s365 \
  -- num_rows_per_batch=1 grad_accum=16
```

不指定 batch/rows 时，配置默认 Polytune batch=4，LadderSym prompted=2 / unprompted=4，rows=8，可能需要更多显存。`optim.warmup_steps=4000` 按 optimizer update 计数；梯度累积增大时，每 epoch 的 update 会减少。调小训练集或大幅增加累积时同步考虑 warmup，例如 `optim.warmup_steps=500`，不要让全部训练落在 warmup 内。scheduler 现在读取 Lightning 的真实 optimizer-step 数，包含累积和 batch 限制。

训练入口把 W&B 设为 offline，不要求登录。输出包括 `.hydra/config.yaml`、stdout 日志、W&B offline 文件、每 epoch 的 top-3 / `last.ckpt`，训练结束额外导出 `last.pt`。原始配置里的 `min_lr` 是 LambdaLR 倍率，当前 `0.5` 表示最低学习率为峰值的一半。

## 5. 断点续训与热启动

`--resume` 必须使用 `.ckpt`，恢复模型、optimizer、scheduler、epoch；`--epochs` 是希望达到的**总 epoch 数**，不是再跑多少轮。续训通常保留原来的总预算、配置和 split。旧 Mac checkpoint 用的是未审计 split，不应把换用审计版后的续训标为同一完整实验；正式结果建议从头训练。中途改变总 epochs 会改变余下的 scheduler，不能当成完全相同的实验。

```bash
find "$B/runs" -name last.ckpt
# 用实际路径替换 CKPT；最好给续训一个新的 run-name。
export CKPT=/absolute/path/to/last.ckpt
bash "$B/scripts/polytune_train.sh" --data "$DATA" --profile cuda \
  --resume "$CKPT" --epochs 40 --batch-size 1 --run-name polytune_align_v1_resume \
  -- num_rows_per_batch=1 grad_accum=16

# LadderSym 的 resumed variant 必须和 checkpoint 一致。
bash "$B/scripts/laddersym_train.sh" --data "$DATA" --profile cuda --prompted \
  --resume "$CKPT" --epochs 40 --batch-size 1 --run-name laddersym_prompted_resume \
  -- num_rows_per_batch=1 grad_accum=16
```

LadderSym 的 `--warm-start <.ckpt|.pt>` 只读取权重，重新开始 optimizer/epoch；与 resume 互斥。现在严格检查 key 和 shape，不再可能“加载成功但模型仍是随机权重”，也不会把输入 `.pt` 覆盖成轻量 checkpoint。

```bash
bash "$B/scripts/laddersym_train.sh" --data "$DATA" --profile cuda --prompted \
  --warm-start /absolute/path/to/weights.pt --run-name laddersym_prompted_finetune \
  --batch-size 1 -- num_rows_per_batch=1 grad_accum=16
```

官方预训练权重与从 scratch 的结果应分开报告。需要作者数据上的论文复现时，按 [Polytune](https://github.com/ben2002chou/Polytune) / [LadderSym](https://github.com/ben2002chou/LadderSym) 的官方数据、配置和 checkpoint 协议另建实验；不要把本项目的 ALIGN 默认配置当成论文配置。

## 6. 验证集选 checkpoint，再评估 test

`last.pt` 是最后状态，不一定是最好的。先按 `val_loss` 选择候选 `.ckpt`（文件名含 val_loss），必要时在 validation split 上比较 per-class F1，然后锁定 checkpoint 和参数，最后评估 test。不能用 test 反复挑参数。

```bash
export CKPT=/absolute/path/to/selected.ckpt
bash "$B/scripts/polytune_eval.sh" --data "$DATA" --ckpt "$CKPT" \
  --split validation --profile cuda --tag polytune_s365_val
bash "$B/scripts/polytune_eval.sh" --data "$DATA" --ckpt "$CKPT" \
  --split test --profile cuda --tag polytune_s365_test

bash "$B/scripts/laddersym_eval.sh" --data "$DATA" --ckpt "$CKPT" \
  --prompted --split test --profile cuda --tag laddersym_prompted_s365_test
# unprompted 使用它自己的 checkpoint，并替换为 --unprompted 和不同 --tag。
```

每次用唯一 `--tag`，已有预测目录会报错，防止复用旧结果。音频读取、模型执行和写文件等顶层推理异常会立即失败，split 中缺文件、预测缺失或类别无法识别也会失败。作者 token 解析器对非法生成事件的计数/跳过行为仍保留；未收敛 smoke 模型可能在这里打印 traceback，验收应同时检查进程退出码、预测覆盖和指标 JSON。`--first-n K` 现在稳定选按 track_id 排序的前 K 条，实际名单写入 `evaluated_ids.json` 和 `evaluated_split.json`。

主要结果在：

```text
runs/<model>/eval_<tag>/note_metrics.json
runs/<model>/eval_<tag>/<tag>/<track_id>/mix.mid
runs/<model>/eval_<tag>/<tag>/evaluated_ids.json
```

**报告 `note_metrics.json` 的 `official_note_wise` 作为 ALIGN 对比 F1。** 该指标把 Extra/Missing/Correct 映射到 canonical score-event identity，同类同位置 1.0、错类同位置 0.5、错位置 0。作者原始协议保留在 `legacy_mir_eval_onset_50ms`：onset tolerance 50 ms、pitch tolerance 50 cents，音符 pitch 转 Hz 后交给 mir_eval，按 MIDI track 名识别类别。没有某类音符时也保留正确分母。`all` 混合三类，衡量 class-agnostic transcription，不是 error-class F1，也不是 ALIGN 官方分数。

作者 `evaluate_errors.py` 的 stdout 继续保留用于对照，但其按 track 位置配对的 per-class 指标会在某类别为空时错位，不能作为这里的类别最终结果。

训练时若改过 token budget，评估和单对推理保持同样的 `event_length` / `prompt_length`。尤其 LadderSym 的 prompt padding 长度会影响 decoder 位置；例如旧 Mac 3-epoch checkpoint 需 `-- event_length=256 prompt_length=384`，不能直接用新默认 1024 假装同一配置。

## 7. ALIGN spans 与真实录音（可选）

只传转换后的 DATA_ROOT 就能得到正确的 synthetic note 指标。如果需要原项目 `labels.json` 的 span 指标，把原始 bundles 也传过去：

```bash
# 本机执行；它们可能很大，只在需要源审计/span 评估时传。
rsync -avh --progress synth-pipeline/output_10k_multi/ \
  USER@GPU_HOST:/srv/ALIGN/synth-pipeline/output_10k_multi/
rsync -avh --progress synth-pipeline/output_2k_rawdata/ \
  USER@GPU_HOST:/srv/ALIGN/synth-pipeline/output_2k_rawdata/
rsync -avh --progress data/test/ USER@GPU_HOST:/srv/ALIGN/data/test/
```

评估时明确提供新路径，不依赖旧 Mac manifest 的绝对路径：

```bash
bash "$B/scripts/polytune_eval.sh" --data "$DATA" --ckpt "$CKPT" --profile cuda \
  --tag polytune_s365_test_spans \
  --bundles /srv/ALIGN/synth-pipeline/output_10k_multi \
  --bundles /srv/ALIGN/synth-pipeline/output_2k_rawdata
# LadderSym eval 同样支持重复 --bundles。

"$B/envs/polytune/bin/python" "$B/common/check_labels.py" --root "$DATA" --n 60 \
  --set multi=/srv/ALIGN/synth-pipeline/output_10k_multi \
  --set raw=/srv/ALIGN/synth-pipeline/output_2k_rawdata
```

bridge 的输出为 `eval_bridge.json`。类映射：wrong_note = Extra + Missing 配对；missed_note = Missing；extra_note = Extra；repetition 由连续 Extra 的启发式规则推断。rhythm_error、intonation_error 不在这两个模型的输出空间，不能把六类整体分数当作同等任务的直接比较。参数只在 validation 上调整。

真实录音的 label MIDI 都是空占位文件：`note_metrics.json` 明确写 `real_test: true`，note 指标为 null。若有 `--bundles /srv/ALIGN/data/test`，可算人工 span 标签指标。不要把空占位 MIDI 算出的零分当成 note 真值结果，也不要用 real_test 训练。

reference WAV/MIDI 仍处于谱面时间线；转换器的补零只让长度相等，不做时间对齐。大幅重复/速度变化可能超出模型按相同绝对时间取 reference 窗口的范围。这是当前 ALIGN adaptation 的任务限制，要在结果中说明；若加入 DTW 或 score warping，需要作为另一个明确标注的实验。

## 8. 常见失败

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: align_runtime` | 使用 `scripts/*` 入口；手工进作者目录运行时先 `export PYTHONPATH="$B/common${PYTHONPATH:+:$PYTHONPATH}"` |
| bootstrap 报 patch/config 冲突 | 保留本地修改，将嵌套仓库移到备份路径，再运行 bootstrap；不要直接 reset |
| CUDA OOM | 先 batch=1、rows=1；按需提高梯度累积。不要首先减小 prompt/event budget 后直接与完整配置比较 |
| DataLoader 进程退出或 `/dev/shm` 不足 | 追加 `dataloader.train.num_workers=0 dataloader.val.num_workers=0`；必要时增加容器 shm |
| `note_labels.json is required` | 使用已传输的 `align_v1` 或冻结 labelled bundles。新 origin 的 `note_map.json` 不能直接作为输入 |
| checkpoint keys/shape 不匹配 | 核对模型种类和权重来源；保持严格加载，不要用 `strict=False` 掩盖错误 |
| 训练 loss 下降但 F1 很低 | 检查完整解码、prompt budget、数据分布和三类各自分数；smoke 模型本来就未收敛 |
| 评估目录已存在 | 换一个 tag；已完成的预测不要混入另一个 checkpoint 的结果 |
| bridge 找不到 bundle | 传新服务器的 `--bundles`；只需要 note 指标时不必传 source bundles |

保留训练 `.hydra/config.yaml`、依赖 freeze、作者 commit/patch、`split_sha256`、所选 checkpoint 和评估 JSON，才可以在下一台机器重做同一实验。
