"""Score frozen policy-3 exports against content-only AudioEval+MuScriptor labels."""
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np
from run import ROOT, RUN, IDS, KINDS, read, write, sha
from datacreate.melody import canonical_note_location, match_note_wise_labels_detail
from datacreate.validation import validate_labels_file


def prf(credit, predicted, gold):
    return dict(credit=credit,predicted=predicted,gold=gold,
        precision=credit/max(1,predicted),recall=credit/max(1,gold),f1=2*credit/max(1,predicted+gold))


def aggregate(rows):
    return prf(*(sum(r[k] for r in rows) for k in ['credit','predicted','gold']))


def units(labels):
    result=Counter()
    for ordinal,label in enumerate(labels):
        kind=label['type'];indices=tuple(sorted(set(label.get('score_event_indices') or [])))
        if not indices:
            result[('unlocated',kind,ordinal)]+=1
        elif kind=='extra_note':
            result[(kind,indices)]+=1
        else:
            copies=max(1,int(label.get('extra_copies') or 1)) if kind=='repetition' else 1
            for index in indices:result[(kind,index)]+=copies
    return result


def note_metric(gold,pred):
    g,p=units(gold),units(pred)
    assert not any(k[0]=='unlocated' for k in g)
    return prf(sum((g&p).values()),sum(p.values()),sum(g.values()))


def strict(gold,pred,count):
    result=match_note_wise_labels_detail(gold,pred,score_event_count=count,type_mismatch_credit=0.)
    assert result['status']=='available',result
    return result


def assess():
    protocol=read(RUN/'protocol.json')
    for name,h in protocol['source_sha256'].items():assert sha(name)==h,name
    records=read(RUN/'data/manifest.json')['tracks']
    for sid in IDS:
        for name,h in protocol['samples'][sid]['hashes'].items():
            assert sha(RUN/'bundles'/sid/name)==h,(sid,name)
        for key,h in records[sid]['hashes'].items():
            assert sha(records[sid]['paths'][key])==h,(sid,key)
        teacher=read(RUN/'teacher_gold'/f'{sid}.json')
        frozen=read(RUN/'bundles'/sid/'labels_agent.json')
        expected=[dict(l) for l in frozen['labels'] if l['type'] in KINDS]
        for label in expected:
            if label['type']=='repetition' and label.get('extra_copies') is None:
                label['extra_copies']=1
        assert teacher['labels']==expected,sid
        assert teacher['score_event_count']==protocol['samples'][sid]['score_event_count']
    results={}
    for model in protocol['models']:
        rows=[]
        for sid in IDS:
            path=RUN/'labels'/sid/f'labels_{model}.json'
            assert not validate_labels_file(path)
            document=read(path);metadata=document['baseline_labeling']
            assert metadata['alignment_policy']==3
            assert metadata['alignment_lookup']['alignment_sha256']==protocol['samples'][sid]['hashes']['note_alignment_v2.json']
            pred=[l for l in document['labels'] if l['type'] in KINDS]
            assert len(pred)==len(document['labels'])
            pred+=[dict(id=f'unclassified_{i}',type='unclassified',score_event_indices=[])
                   for i in range(metadata['unclassified_count'])]
            gold_doc=read(RUN/'teacher_gold'/f'{sid}.json')
            gold=gold_doc['labels'];n=gold_doc['score_event_count']
            assert all(l['type'] in KINDS for l in gold)
            native=RUN/'predictions/test'/model/'n00/step000'/sid
            done=read(native/'mix.done.json')
            assert sha(native/'mix.mid')==done['sha256']
            assert done['checkpoint_sha256']==protocol['initial_checkpoints'][model]['sha256']
            assert done['labels_read'] is False
            feature_path=RUN/'cache'/model/'infer'/f'{sid}.pt'
            feature_metadata=read(feature_path.with_suffix('.json'))
            assert sha(feature_path)==feature_metadata['sha256']
            assert feature_metadata['input_hashes']==records[sid]['hashes']
            assert sha(RUN/'alignment'/f'{sid}.json')==feature_metadata['alignment_sha256']
            if done.get('reused_from'):
                original=Path(done['reused_from'])
                assert sha(original/'mix.mid')==done['sha256']
                assert sha(original/'mix.done.json')==done['reused_source_done_sha256']
            times=[l for l in pred if l.get('start_time') is not None and l.get('end_time') is not None]
            row=dict(sample=sid,status=gold_doc['status'],scored=sid in protocol['score_samples'],
                gold=len(gold),predicted=len(pred),unlocated=sum(canonical_note_location(l,score_event_count=n) is None for l in pred),
                unclassified=metadata['unclassified_count'],prediction_types=dict(Counter(l['type'] for l in pred)),
                gold_types=dict(Counter(l['type'] for l in gold)),rhythm_labels_ignored=gold_doc['ignored_types'].get('rhythm_error',0),
                capped_windows=done['capped_windows'],windows=done['windows'],
                reused_prediction=bool(done.get('reused_from')),inference_seconds=done['seconds'],
                short_fragments_under_50ms=sum(l['end_time']-l['start_time']<.05 for l in times),
                labels_sha256=sha(path))
            if row['scored']:
                row['strict']=strict(gold,pred,n)
                row['note_units']=note_metric(gold,pred)
                row['by_type']={k:strict([l for l in gold if l['type']==k],[l for l in pred if l['type']==k],n) for k in KINDS}
                row['note_units_by_type']={k:note_metric([l for l in gold if l['type']==k],[l for l in pred if l['type']==k]) for k in KINDS}
                exact=[p for p in row['strict']['pairs'] if p['credit']==1]
                row['matches']=[dict(prediction=pred[p['prediction_index']],gold=gold[p['gold_index']]) for p in exact]
            rows.append(row)
        scored=[r for r in rows if r['scored']]
        counts=np.array([[r['strict'][k] for k in ['credit','predicted','gold']] for r in scored])
        draws=np.random.default_rng(20261107).integers(0,len(scored),(2000,len(scored)))
        sums=counts[draws].sum(1)
        boot=2*sums[:,0]/np.maximum(1,sums[:,1]+sums[:,2])
        clean=[r for r in scored if r['gold']==0]
        results[model]=dict(strict=aggregate([r['strict'] for r in scored]),note_units=aggregate([r['note_units'] for r in scored]),
            by_type={k:aggregate([r['by_type'][k] for r in scored]) for k in KINDS},
            note_units_by_type={k:aggregate([r['note_units_by_type'][k] for r in scored]) for k in KINDS},
            strict_f1_95pct_bootstrap=np.quantile(boot,[.025,.975]).tolist(),samples=rows,
            scored_count=len(scored),inferred_count=len(rows),reused_predictions=sum(r['reused_prediction'] for r in rows),
            all_predicted=sum(r['predicted'] for r in rows),unlocated_scored=sum(r['unlocated'] for r in scored),
            capped_windows_scored=sum(r['capped_windows'] for r in scored),windows_scored=sum(r['windows'] for r in scored),
            content_clean_samples=len(clean),content_clean_samples_with_predictions=sum(r['predicted']>0 for r in clean),
            content_clean_predictions=sum(r['predicted'] for r in clean),
            short_fragments_under_50ms_scored=sum(r['short_fragments_under_50ms'] for r in scored))
    for ck in protocol['initial_checkpoints'].values():assert sha(ck['path'])==ck['sha256']
    unchanged=[];changed=[]
    for name,h in protocol['protected_sha256'].items():
        (unchanged if sha(name)==h else changed).append(name)
    # Report concurrent user changes rather than silently using changed inputs;
    # every model/projection/score above uses the immutable snapshot.
    report=dict(protocol_sha256=sha(RUN/'protocol.json'),results=results,excluded=protocol['abstentions'],
        ignored_rhythm_total=sum(s['ignored_rhythm'] for s in protocol['samples'].values()),
        live_files_unchanged=len(unchanged),live_files_changed_during_run=changed,
        assessment_source_sha256=sha(Path(__file__)))
    write(RUN/'results.json',report)
    render(report,protocol)
    write(RUN/'assessment_complete.json',dict(status='complete',time=time.time(),
        documents_validated=sum(r['inferred_count'] for r in results.values()),
        policy=3,rhythm_excluded=True,source_inputs_and_checkpoint_hashes_verified=True,
        results_sha256=sha(RUN/'results.json'),report_sha256=sha(RUN/'REPORT.md')))
    print(json.dumps({m:{k:v for k,v in r.items() if k!='samples'} for m,r in results.items()},indent=2))


def percent(x):return f'{100*x:.2f}%'


def render(report,protocol):
    results=report['results'];r0=next(iter(results.values()))
    lines=['# DataCreate 001–050: PolyTune and LadderSym with policy 3','',
        'Evaluated the unchanged author checkpoints using the corrected reference-input frontend and policy 3 label mapping. No training or threshold tuning was performed.', '',
        f"Processed all 50 recordings. Scored {r0['scored_count']} against the frozen current AudioEval+MuScriptor labels; excluded {', '.join(report['excluded'])} because the teacher marked their alignment uncertain. Excluded all {report['ignored_rhythm_total']} rhythm labels. Empty content-label sets on otherwise valid recordings are retained and penalize false positives.",'',
        'The primary metric matches the exact affected reference-score note set and error type, with repetition copy counts. It does not match timestamps or padded GUI context. Unlocated and unclassified predictions remain false positives.','',
        '| Model | Precision | Recall | Exact note/type F1 | Matches | Predictions | Reference labels |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for model,r in results.items():
        s=r['strict'];lines.append(f"| {model} | {percent(s['precision'])} | {percent(s['recall'])} | {percent(s['f1'])} | {s['credit']:g} | {s['predicted']} | {s['gold']} |")
    lines+=['','## Individual note accounting','',
        'This supplementary metric splits affected ranges into individual reference notes; extras remain gap identities and repetitions retain multiplicity. It credits partially recovered ranges.','',
        '| Model | Note precision | Note recall | Note F1 | Matches / predictions / reference units |',
        '|---|---:|---:|---:|---:|']
    for model,r in results.items():
        s=r['note_units'];lines.append(f"| {model} | {percent(s['precision'])} | {percent(s['recall'])} | {percent(s['f1'])} | {s['credit']:g} / {s['predicted']} / {s['gold']} |")
    lines+=['','## By error type','',
        '| Model | Type | Precision | Recall | F1 | Matches / predictions / references |',
        '|---|---|---:|---:|---:|---:|']
    for model,r in results.items():
        for kind,s in r['by_type'].items():
            lines.append(f"| {model} | {kind} | {percent(s['precision'])} | {percent(s['recall'])} | {percent(s['f1'])} | {s['credit']:g} / {s['predicted']} / {s['gold']} |")
    lines+=['','## Output problems','']
    for model,r in results.items():
        s=r['strict'];ci=r['strict_f1_95pct_bootstrap']
        worst=sorted([x for x in r['samples'] if x['scored']],key=lambda x:x['predicted']-x['strict']['credit'],reverse=True)[:5]
        capped=sorted([x for x in r['samples'] if x['scored'] and x['capped_windows']],key=lambda x:x['capped_windows'],reverse=True)[:5]
        lines += [f"- **{model}:** {s['predicted']-s['credit']:g} unmatched predictions and {s['gold']-s['credit']:g} missed reference labels. {r['unlocated_scored']} predictions have no usable reference-note location.",
            f"- {model} predicts errors on {r['content_clean_samples_with_predictions']}/{r['content_clean_samples']} recordings that have no teacher content errors, producing {r['content_clean_predictions']} predictions there.",
            f"- {model}: {r['capped_windows_scored']}/{r['windows_scored']} scored audio windows reached the generation cap without EOS; {r['short_fragments_under_50ms_scored']} predicted intervals are shorter than 50 ms. Neither was removed from scoring.",
            f"- {model} largest false-positive counts: "+', '.join(f"{w['sample']} ({w['predicted']-w['strict']['credit']:g})" for w in worst)+'.',
            f"- {model} recordings with the most capped windows: "+(', '.join(f"{w['sample']} ({w['capped_windows']}/{w['windows']})" for w in capped) or 'none')+'.',
            f"- {model} recording-bootstrap 95% interval for exact F1: {percent(ci[0])}–{percent(ci[1])} (2,000 resamples; descriptive uncertainty on this dataset)."]
    lines+=['','## Interpretation and provenance','',
        '- The reference labels are model-generated, not independently verified human ground truth. Policy 3 shares the MuScriptor/ALIGN timing and score mapping used to produce these references, although projection never consumes teacher error labels or classes.',
        '- Scores measure agreement with the published teacher labels. On accepted recordings, a prediction without a matching teacher label counts as a false positive; teacher omissions or unassessed notes can therefore overstate the baseline\'s errors.',
        '- Current teacher outputs were frozen for this run. They include the full-recording MuScriptor context update and may differ from the older November 5 teacher snapshot. Rhythm labels are excluded, as requested.',
        '- These are unchanged-checkpoint results. Some recordings were used to train or validate earlier fine-tuned experiments; no fine-tuned checkpoint is evaluated here. This is a descriptive benchmark, not a new independent holdout.',
        '- Score-note identity does not distinguish repeated performance occurrences. Short fragments can map to the correct note despite inaccurate duration. These are unchanged limitations of policy 3.',
        '- All model inference uses only performance audio and reference audio/MIDI. Label projection consumes sanitized stored transcript timing and score spans. Scoring reads teacher labels only after inference/export.',
        f"- {r0['reused_predictions']} verified earlier control predictions per model were reused after matching input/cache/checkpoint hashes; the remaining {50-r0['reused_predictions']} recordings per model were decoded in this run.",
        '- Native predictions, exported labels, teacher snapshot, hashes, and per-sample metrics are saved under this run. Live human/agent/baseline labels were not overwritten.',
        f"- Integrity check: {report['live_files_unchanged']} protected files unchanged; {len(report['live_files_changed_during_run'])} concurrent source changes reported in results.json. Model checkpoint hashes match the frozen protocol.",
        '','## Per-recording exact matches','',
        '| Sample | Teacher content labels | PolyTune matches / predictions | LadderSym matches / predictions |',
        '|---|---:|---:|---:|']
    by_model={m:{r['sample']:r for r in v['samples']} for m,v in results.items()}
    for sid in IDS:
        a,b=by_model['polytune'][sid],by_model['laddersym'][sid]
        if a['scored']:
            lines.append(f"| {sid} | {a['gold']} | {a['strict']['credit']:g} / {a['predicted']} | {b['strict']['credit']:g} / {b['predicted']} |")
        else:lines.append(f"| {sid} | excluded: {a['status']} | — / {a['predicted']} | — / {b['predicted']} |")
    lines+=['','[Full metrics and matching evidence](results.json) · [Frozen protocol](protocol.json) · [Policy-3 exported labels](labels/)', '']
    (RUN/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':assess()
