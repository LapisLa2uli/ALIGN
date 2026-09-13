# 完整源码补丁

固定上游：

- Polytune：`d2055bb21759d457c8f21c1cf2e47c79af6248f5`
- LadderSym：`381179754cf6bcb435f9decf0d5e24eada6c68ec`

`polytune.patch` / `laddersym.patch` 为相对上述 commit 的完整 tracked-file diff（包括早期兼容修复和本次审计修复）。新增 ALIGN YAML 放在 `../configs/{Polytune,LadderSym}/`，辅助 Python 放在 `../common/`。直接运行作者 entry point 时需把 common 加入 PYTHONPATH；`../scripts/` 已自动设置。

推荐运行 `bash baselines/scripts/bootstrap.sh --repos-only`，它会克隆固定提交、检查 patch 是否已应用并复制配置，不覆盖冲突修改。

补丁修改推理 EOS 处理、异常传播、frame-time / prompt 边界、类别 MIDI 命名、设备和 checkpoint 加载；训练修正空 error-token 日志和真实 optimizer-step 计数；LadderSym 修复 warm-start 并限制编译/debug 行为。ALIGN 配置使用 LR multiplier floor=0.5 和 deterministic validation。详细原因、测试与保留限制见 [审计记录](../docs/REPRODUCTION_AUDIT.md)。

作者模型架构与 weighted CE 公式保留。补丁会改变原来有 bug 的解码结果，不能把新旧预测混入同一个评估目录。使用唯一 `--tag`，通过 `note_metrics.json` 做类别正确的评估。

维护修改时：先改作者工作树，测试后用 `git -C baselines/Polytune diff --binary` 更新对应 patch；配置修改同步到 `configs/`；最后在干净 clone 上重建检查。不要把嵌套 `.git` 或数据加进主仓库。
