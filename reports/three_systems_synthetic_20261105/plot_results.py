"""Chinese Matplotlib plot of the three paired synthetic error-detection scores."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'results.json').read_text(encoding='utf-8'))
order=['audioeval','polytune','laddersym']
scores=np.array([data['results'][m]['f1'] for m in order])*100
ci=np.array([data['results'][m]['ci95'] for m in order])*100
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
                    'axes.unicode_minus':False,'svg.fonttype':'path','pdf.fonttype':42,'font.size':12})
fig,ax=plt.subplots(figsize=(11,7),facecolor='white')
fig.subplots_adjust(left=.10,right=.97,bottom=.28,top=.78)
x=np.arange(3)
ax.bar(x,scores,width=.53,color=['#24998A','#377CB5','#B58246'],zorder=3)
ax.errorbar(x,scores,yerr=np.stack([scores-ci[:,0],ci[:,1]-scores]),fmt='none',
            color='#1B3040',elinewidth=1.7,capsize=7,capthick=1.7,zorder=4)
for i,(v,bounds) in enumerate(zip(scores,ci)):
    ax.text(i,bounds[1]+3,f'{v:.2f}%',ha='center',fontsize=19,fontweight='bold',color='#1B3040')
    ax.text(i,bounds[1]+11,f'{bounds[0]:.2f}%—{bounds[1]:.2f}%',ha='center',fontsize=11,color='#526578')
ax.set_ylim(0,100);ax.set_yticks(np.arange(0,101,20));ax.set_ylabel('内容错误检测 F1（%）',labelpad=12)
ax.set_xticks(x,['AudioEval + MuScriptor\n（中型模型）','PolyTune\n（作者预训练）','LadderSym\n（作者预训练，带提示）'])
ax.grid(axis='y',color='#E3E9EF',zorder=0);ax.spines[['top','right','left']].set_visible(False)
ax.spines['bottom'].set_color('#CCD4DD');ax.tick_params(length=0,pad=10)
fig.text(.10,.925,'三种系统的合成数据测试结果',fontsize=23,fontweight='bold',color='#1B3040')
fig.text(.10,.865,'相同的 30 段测试音频 · 135 个参考内容错误 · 固定模型与参数',fontsize=13,color='#526578')
fig.text(.10,.16,'指标：谱面音符位置与错误类型的微平均 F1；位置相同、类型不同计半分。',fontsize=11,color='#43566A')
fig.text(.10,.115,'包含错音、漏音、多音及重复；不含节奏与音准偏差。此指标不同于转录音高序列 F1。',fontsize=10,color='#627286')
fig.text(.10,.07,'误差线：2,000 次音频级自助重采样的 95% 置信区间；未定位预测计为未命中。',fontsize=10,color='#627286')
fig.text(.10,.03,'AudioEval 的 1 次推理失败与 1 次弃权均保留在测试集中；结果仅代表本次小样本。',fontsize=10,color='#627286')
for ext in ('png','svg','pdf'):fig.savefig(HERE/f'three_systems_f1_zh.{ext}',dpi=220,facecolor='white')
plt.close(fig)
