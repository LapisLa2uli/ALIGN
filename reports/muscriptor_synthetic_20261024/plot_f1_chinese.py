"""Plot paired 100-clip micro-F1 results and clip-bootstrap confidence intervals."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
report = json.loads((HERE / 'results_audited.json').read_text(encoding='utf-8'))
ctc_path = ROOT / 'align-model/runs/realistic92-transcriber-v1/TEST_CTC_R2A.json'
assert hashlib.sha256(ctc_path.read_bytes()).hexdigest() == report['saved_ctc_same_100_clips']['source_sha256']
ctc = {r['sample']: r for r in json.loads(ctc_path.read_text(encoding='utf-8'))['per_clip']}
selected = [r for r in report['per_clip'] if r['corpus'] == 'realistic92']
assert len(selected) == 100 and len({r['sample'] for r in selected}) == 100
fields = ('matched', 'predicted', 'gold')
mus_counts = np.array([[r['sequence'][k] for k in fields] for r in selected])
ctc_counts = np.array([[ctc[r['sample']][k] for k in fields] for r in selected])
assert np.array_equal(mus_counts[:, 2], ctc_counts[:, 2])
counts = np.stack([mus_counts, ctc_counts])
totals = counts.sum(axis=1)
scores = 2 * totals[:, 0] / totals[:, 1:].sum(axis=1)
assert np.allclose(scores, [report['results']['realistic92']['sequence']['f1'],
                            report['saved_ctc_same_100_clips']['f1']])
rng = np.random.default_rng(20261024)
boot = []
for _ in range(2000):
    indices = rng.integers(100, size=100)
    summed = counts[:, indices, :].sum(axis=1)
    boot.append(2 * summed[:, 0] / summed[:, 1:].sum(axis=1))
ci = np.quantile(boot, [0.025, 0.975], axis=0).T
assert np.allclose(ci[0], report['results']['realistic92']['sequence_clip_bootstrap_95ci'])
summary = dict(metric='Micro pitch-sequence LCS F1', clips=100, reference_notes=int(totals[0, 2]),
               bootstrap_unit='clip', bootstrap_replicates=2000, seed=20261024,
               method='Paired clip resampling; percentile 2.5% and 97.5%; micro-F1 recomputed each replicate',
               caveat='Timing ignored; clip interval does not estimate generalization across source scores',
               models=[dict(model=name, f1=float(score), ci95=list(map(float, bounds)))
                       for name, score, bounds in zip(['MuScriptor medium', 'ALIGN CTC R2A'], scores, ci)])
(HERE / 'f1_chinese_chart_data.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
font = font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
plt.rcParams.update({'font.family': font, 'axes.unicode_minus': False,
                     'svg.fonttype': 'path', 'pdf.fonttype': 42, 'font.size': 12})
fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
fig.subplots_adjust(left=0.12, right=0.96, bottom=0.23, top=0.79)
x = np.arange(2)
values = scores * 100
bounds = ci * 100
ax.bar(x, values, width=0.48, color=['#367BB5', '#25998A'], zorder=3)
ax.errorbar(x, values, yerr=np.stack([values-bounds[:,0], bounds[:,1]-values]),
            fmt='none', ecolor='#182B3A', elinewidth=2, capsize=9, capthick=2, zorder=4)
for i, (value, interval) in enumerate(zip(values, bounds)):
    ax.text(i, 73, f'{value:.2f}%', ha='center', va='center', fontsize=22,
            color='white', fontweight='bold')
    ax.text(i, interval[1]+2, f'{interval[0]:.2f}%—{interval[1]:.2f}%',
            ha='center', va='bottom', fontsize=11, color='#25394B', linespacing=1.5)
ax.set_ylim(0, 110)
ax.set_yticks(np.arange(0, 101, 20))
ax.set_ylabel('总体 F1（%）', labelpad=12)
ax.set_xticks(x, ['MuScriptor（中型）', 'AudioEval CTC 转录器'])
ax.set_xlim(-0.65, 1.65)
ax.grid(axis='y', color='#E5EAF0', linewidth=0.8, zorder=0)
ax.spines[['top','right','left']].set_visible(False)
ax.spines['bottom'].set_color('#CCD4DD')
ax.tick_params(axis='both', length=0, pad=10)
fig.text(0.12, 0.925, '两种转录模型的总体 F1 对比', fontsize=23, fontweight='bold', color='#182B3A')
fig.text(0.12, 0.865, '相同的 100 段合成测试音频 · 11,710 个参考音符', fontsize=13, color='#526578')
fig.text(0.12, 0.13, '指标：音高序列最长公共子序列 F1（微平均，不计时序误差）', fontsize=11, color='#43566A')
fig.text(0.12, 0.086, '误差线：以音频片段为单位，配对自助重采样 2,000 次所得的 95% 置信区间', fontsize=10, color='#627286')
fig.text(0.12, 0.046, 'CTC 使用相同片段的已保存预测；置信区间不代表跨曲目泛化范围。', fontsize=10, color='#627286')
for extension in ('png', 'svg', 'pdf'):
    fig.savefig(HERE / f'f1_comparison_zh.{extension}', dpi=220, facecolor='white')
plt.close(fig)
print(json.dumps(summary, indent=2))
