"""Render the unified canonical comparison only after complete verified results."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT.parent/'align-paper'
RUN=ROOT/'baselines/runs/evaluate_20260915/outputraw358_canonical'
reports={name:json.loads((RUN/(name+'_results.json')).read_text()) for name in ('polytune','laddersym')}
for report in reports.values():
    assert report['validation_count']==358 and report['micro']['gold']==40712
assert {r['sample']:r['gold_sha256'] for r in reports['polytune']['per_clip']} == {r['sample']:r['gold_sha256'] for r in reports['laddersym']['per_clip']}
def table_row(name, report):
    m=report['micro'];b=report['bootstrap']
    return f"{name} & {m['precision']:.4f} & {m['recall']:.4f} & {m['f1']:.4f} & [{b['lower_95']:.4f}, {b['upper_95']:.4f}] " + r"\\"
rows='\n'.join([table_row('Polytune + shared location adapter',reports['polytune']),
                 table_row('LadderSym (prompted) + shared location adapter',reports['laddersym']),
                 r'Ours & 0.5552 & 0.6779 & 0.6104 & [0.5937, 0.6272] \\'])
section=(Path(__file__).parent/'paper_protocol.tex').read_text().replace('%RESULT_ROWS%', rows)
path=PAPER/'sections/current_results.tex'
text=path.read_text()
start=text.index(chr(92)+'subsection{Shared Canonical Note-Wise Comparison}')
end=text.index(chr(92)+'subsection{Joint Transcription and Alignment}',start)
path.write_text(text[:start]+section.lstrip()+text[end:])
summary=dict(metric='canonical combined-pipeline micro precision/recall/F1',validation_count=358,
             gold_count=40712,models={name:{k:r[k] for k in ('metric','task','validation_count','split_sha256','target_sha256','adapter_version','micro','bootstrap','freeze_manifest_sha256')} for name,r in reports.items()},
             ours=dict(precision=.5552,recall=.6779,f1=.6104,lower_95=.5937,upper_95=.6272,
                       provenance='Existing Overleaf canonical combined-pipeline result; not rerun in this baseline evaluation'),
             baseline_completed_epochs=dict(polytune=31,laddersym=26),
             exact_audio_overlap=dict(baseline_train=21,baseline_selection_validation=3))
(ROOT/'experiments/baselines_outputraw_20260915/results.json').write_text(json.dumps(summary,indent=2)+'\n')
(PAPER/'baseline_validation_results.json').write_text(json.dumps(summary,indent=2)+'\n')
print(rows)
