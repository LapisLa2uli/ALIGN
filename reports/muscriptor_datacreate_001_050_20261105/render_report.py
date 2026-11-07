from pathlib import Path
import json
from collections import Counter
import numpy as np

out=Path(__file__).resolve().parent
d=json.loads((out/'evaluation.json').read_bytes())
ss=d['summary']
rows=d['samples']
scored=[v for v in rows.values() if v['scored']]
pct=lambda x:f'{100*x:.2f}%'
lines=['# DataCreate 001–050: MuScriptor vs. CTC','',f"Published MuScriptor medium + ALIGN v9 agent labels for all {len(rows)} takes. Scored {len(scored)} against the pre-seed user archive ({sum(r['manual_count'] for r in scored)} human labels). 004 was explicitly reviewed clean. The remaining takes lack verified manual review and were not scored.",'', 'The same ALIGN v9 candidate and feedback thresholds were used for both. The CTC baseline uses the original, unedited publisher outputs. Only the labels published to the GUI are scored; an abstention contributes no predictions.', '', '| Label metric | Pipeline | Precision | Recall | F1 | Match credit | Predictions | Human labels |','|---|---|---:|---:|---:|---:|---:|---:|']
for k,t in [('all/official','All error types, official'),('all/strict','All error types, strict'),('content/strict','Four content types, strict')]:
 for name in ['CTC','MuScriptor']:
  m=ss[name][k];lines.append(f"| {t} | {name} + ALIGN | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | {m['credit']:g} | {m['predicted']} | {m['gold']} |")
lines+=['','Official micro-F1 matches errors one-to-one on the exact affected score-note location, excluding display padding. Matching the type earns one point; the wrong type at that location earns half. Strict F1 requires the same type. Content types: wrong note, missed note, extra note, repetition. This is not a metric of transcription or timestamp accuracy.','', '## Strict results by type','','| Error type | Human labels | CTC calls | CTC matches | MuScriptor calls | MuScriptor matches |','|---|---:|---:|---:|---:|---:|']
for k in sorted(set(ss['CTC']).union(ss['MuScriptor'])):
 if not k.startswith('type:') or not k.endswith('/strict'): continue
 a,b=[ss[n].get(k,{'gold':0,'credit':0,'predicted':0}) for n in ['CTC','MuScriptor']]
 lines.append(f"| {k[5:-7]} | {max(a['gold'],b['gold'])} | {a['predicted']} | {a['credit']:g} | {b['predicted']} | {b['credit']:g} |")
def f1(c):return 2*c[:,0]/np.maximum(1,c[:,1]+c[:,2])
ix=np.random.default_rng(20261105).integers(0,len(scored),(10000,len(scored)))
def boot(n):
 c=np.array([[v['metrics'][n]['all/official'][k] for k in ['credit','gold','predicted']] for v in scored])
 return f1(c[ix].sum(axis=1))
lo,hi=np.quantile(boot('MuScriptor')-boot('CTC'),[.025,.975])*100
delta=100*(ss['MuScriptor']['all/official']['f1']-ss['CTC']['all/official']['f1'])
uncertain=[n for n,r in rows.items() if r['alignment_status']!='ok']
lines+=['',f'Official micro-F1 difference (MuScriptor minus CTC): {delta:+.2f} percentage points; paired 95% bootstrap interval {lo:+.2f} to {hi:+.2f} points (10,000 resamples of reviewed takes, seed 20261105). The interval describes this small reviewed sample.','',f"Agent labels installed: {sum(r['published'] for r in rows.values())} across 50 takes. Alignment status other than ok: {', '.join(uncertain) or 'none'}. All before/after user-file hashes matched. The live user-label files were agent-seeded and were not used as ground truth.",'', '## Per-take results','','A dash means there was no verified human reference. A reviewed take with zero labels still counts against precision if it receives a prediction.','','| Take | Status | Installed labels | Human labels | CTC match credit | MuScriptor match credit |','|---|---|---:|---:|---:|---:|']
for n,r in rows.items():
 v=[r['manual_count'],*(r['metrics'][m]['all/official']['credit'] for m in ('CTC','MuScriptor'))] if r['scored'] else ['—']*3
 lines.append(f"| {n} | {r['alignment_status']} | {r['published']} | {v[0]} | {v[1]} | {v[2]} |")
lines+=['','[Full per-take metrics and provenance](evaluation.json) · [Installation manifest](publish/manifest.json) · [Previous GUI artifacts](publish/backup/). The pre-seed user archive was read only.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('delta, 95% CI (points):',delta,lo,hi)
