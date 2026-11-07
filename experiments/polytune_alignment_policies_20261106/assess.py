"""Prepare, freeze, project, and evaluate three policies on identical predictions."""
import argparse
from collections import Counter
import os
import subprocess
import sys
import time
from policy_lib import (ROOT, RUN, PREVIOUS, LATEST, PRED, KINDS, SETTINGS, read, write, sha,
                        core_labels, interval, TimeScoreLookup, timestamp_score, note_score, aggregate)

sys.path[:0] = [str(ROOT/'DataCreate/src'), str(ROOT/'baselines/scripts')]

def signature(labels):
    return [(l.get('id'),l['type'],l.get('start_time'),l.get('end_time'),l.get('extra_copies')) for l in labels]

def prepare():
    from datacreate.melody import parse_sounding_notes
    assert not (RUN/'protocol.json').exists(), 'Already frozen'
    previous_protocol = read(LATEST/'protocol.json')
    tracks = read(PREVIOUS/'data/manifest.json')['tracks']
    ids = previous_protocol['split']['test']
    protected = {}
    def protect(path):
        protected[str(path.relative_to(ROOT))] = sha(path)
    for file in [LATEST/'protocol.json',LATEST/'trained_results.json',LATEST/'note_unit_results.json']:
        protect(file)
    for sid in ids:
        for name in ['labels.json','mix.mid','mix.done.json']:
            protect(PRED/sid/name)
        bundle = PREVIOUS/'bundles'/sid
        for path in [bundle/'note_alignment_v2.json',bundle/'verified_score.musicxml',PREVIOUS/f'teacher_gold/{sid}.json']:
            protect(path)
        score = parse_sounding_notes(bundle/'verified_score.musicxml')
        alignment = read(bundle/'note_alignment_v2.json')
        notes = []
        for ordinal,n in enumerate(alignment['transcribed_notes']):
            if n.get('ignored'):
                continue
            a,b = float(n['start']),float(n['end'])
            span = n.get('score_span')
            indices = list(range(*span)) if span else []
            assert all(0<=i<len(score) for i in indices)
            if b>a:
                # Deliberate allowlist: no relationship, label, error, or repetition fields.
                notes.append(dict(ordinal=ordinal,start=a,end=b,pitch=n['pitch'],score_event_indices=indices))
        lookup = dict(sample=sid,notes=notes,score=[dict(pitch=n.pitch,ql_start=n.ql_start,ql_end=n.ql_end) for n in score],
                      source_sha256=sha(bundle/'note_alignment_v2.json'),duration=tracks[sid]['duration'])
        write(RUN/f'lookup_inputs/{sid}.json',lookup)
        doc = read(PRED/sid/'labels.json')
        labels = core_labels(doc['labels'])
        labels += [dict(id=f'unclassified_{i}',type='unclassified') for i in range(doc['baseline_labeling']['unclassified_count'])]
        write(RUN/f'prediction_inputs/{sid}.json',dict(labels=labels,source_sha256=sha(PRED/sid/'labels.json')))
        done = read(PRED/sid/'mix.done.json')
        assert done['checkpoint_sha256']==previous_protocol['initial_checkpoints']['polytune']['sha256']
        assert sha(PRED/sid/'mix.mid')==done['sha256']
    files = list((RUN/'lookup_inputs').glob('*.json'))+list((RUN/'prediction_inputs').glob('*.json'))
    protocol = dict(
        model='PolyTune unchanged author checkpoint, revised inference frontend',
        checkpoint=previous_protocol['initial_checkpoints']['polytune'],
        prediction_directory=str(PRED.relative_to(ROOT)),test_ids=ids,
        abstentions=previous_protocol['test_abstentions'],settings=SETTINGS,
        policies=['1: existing v2 adapter + exact reference-note/type scoring',
                  '2: existing error intervals + timestamp IoU/type scoring',
                  '3: stored MuScriptor/ALIGN timing lookup + exact reference-note/type scoring'],
        gold='Frozen AudioEval v9 + MuScriptor medium pseudo-gold; not human ground truth',
        gold_timestamp_policy='Use recorded performance-audio intervals already attached to every teacher label; missed-note intervals are teacher estimates, not audible notes.',
        p3_shared_teacher_caveat='Stored alignment produced by the same system as pseudo-gold. Only whitelisted time/pitch/score-span fields are consumed, but structural dependence remains.',
        no_threshold_selection_on_test=True,retrospective_previously_examined_test_set=True,
        seed=20261106,bootstrap_replicates=2000,
        protected_sha256=protected,
        input_sha256={str(p.relative_to(RUN)):sha(p) for p in files},
        source_sha256={name:sha(RUN/name) for name in ['policy_lib.py','assess.py','test_policies.py']})
    write(RUN/'protocol.json',protocol)
    print('FROZEN',sha(RUN/'protocol.json'),flush=True)

def verify(protocol):
    for name,digest in protocol['source_sha256'].items():
        assert sha(RUN/name)==digest,name
    for name,digest in protocol['input_sha256'].items():
        assert sha(RUN/name)==digest,name
    for name,digest in protocol['protected_sha256'].items():
        assert sha(ROOT/name)==digest,name

def project():
    p=read(RUN/'protocol.json')
    # A separate process denies access to teacher decisions and original alignment
    # documents; only sanitized lookup inputs and PolyTune predictions are needed.
    def guard(event,args):
        if event=='open' and args:
            name=str(args[0]).replace('\\','/')
            if any(s in name for s in ['/teacher_gold/','/bundles/','/teacher_transcriptions/','/supervision/']):
                raise PermissionError('Projection cannot read teacher decisions: '+name)
    sys.addaudithook(guard)
    for sid in p['test_ids']:
        start=time.perf_counter()
        lookup=TimeScoreLookup(read(RUN/f'lookup_inputs/{sid}.json'))
        before=read(RUN/f'prediction_inputs/{sid}.json')['labels']
        after,traces=lookup.project(before)
        assert signature(before)==signature(after)
        write(RUN/f'projected/{sid}.json',dict(labels=after,traces=traces,seconds=time.perf_counter()-start,
            virtual_missing_intervals=len(lookup.virtual),backward_boundaries=len(lookup.jumps),
            gold_labels_read=False,protocol_sha256=sha(RUN/'protocol.json')))
    print('PROJECTED_ALL',len(p['test_ids']),flush=True)

def evaluate():
    import numpy as np
    from datacreate.melody import match_note_wise_labels_detail
    from label_datacreate_baselines_v2 import document_from_midi
    p=read(RUN/'protocol.json');verify(p)
    subprocess.run([sys.executable,str(RUN/'assess.py'),'project'],check=True,env={**os.environ,'PYTHONNOUSERSITE':'1'})
    policies={key:[] for key in ['status_quo','timestamps','transcription_lookup']}
    sensitivities={f'iou_{t:g}':[] for t in SETTINGS['timestamp_iou_sensitivity']}
    sensitivities.update({f'onset_{t:g}s':[] for t in SETTINGS['timestamp_onset_sensitivity_seconds']})
    timings=[];time_audit=[];mapping_changes=[];unscored=[]
    for sid in p['test_ids']:
        before=read(RUN/f'prediction_inputs/{sid}.json')['labels']
        projected=read(RUN/f'projected/{sid}.json');after=projected['labels']
        # Re-run the current conversion from the same native MIDI to verify the
        # status quo and measure adapter cost, never re-run model inference.
        started=time.perf_counter()
        native=document_from_midi(PRED/sid/'mix.mid',PREVIOUS/f'bundles/{sid}/verified_score.musicxml',model='polytune',sample_id=sid)
        seconds=time.perf_counter()-started
        fresh=core_labels(native['labels'])
        assert signature(fresh)==signature(before[:len(fresh)])
        assert [l.get('score_event_indices',[]) for l in fresh]==[l.get('score_event_indices',[]) for l in before[:len(fresh)]]
        timings.append(dict(sample=sid,current_adapter_seconds=seconds,lookup_seconds=projected['seconds']))
        gold_document=read(PREVIOUS/f'teacher_gold/{sid}.json')
        if gold_document['status']!='ok':
            unscored.append(dict(sample=sid,status=gold_document['status'],predictions=len(before)))
            continue
        gold=core_labels(gold_document['labels']);count=gold_document['score_event_count']
        assert all(interval(l) is not None for l in gold)
        write(RUN/f'timestamp_gold/{sid}.json',dict(labels=[{k:l[k] for k in ['id','type','start_time','end_time','extra_copies'] if k in l} for l in gold],
            source='existing teacher performance-audio interval',new_estimates=0))
        for l in gold:
            time_audit.append(dict(sample=sid,id=l['id'],type=l['type'],start=l['start_time'],end=l['end_time'],
                                   estimated_missing_time=l['type']=='missed_note'))
        def reference_score(pred):
            return match_note_wise_labels_detail(gold,pred,score_event_count=count,type_mismatch_credit=0.)
        t0=time.perf_counter();ts=timestamp_score(gold,before);timings[-1]['timestamp_scoring_seconds']=time.perf_counter()-t0
        outputs={'status_quo':(before,reference_score(before)),
                 'timestamps':(before,ts),'transcription_lookup':(after,reference_score(after))}
        for key,(pred,metric) in outputs.items():
            row=dict(sample=sid,strict=metric,unlocated=sum(not l.get('score_event_indices') for l in pred),
                     predicted_by_type=dict(Counter(l['type'] for l in pred)),gold_by_type=dict(Counter(l['type'] for l in gold)))
            row['by_type']={kind:(timestamp_score([l for l in gold if l['type']==kind],[l for l in pred if l['type']==kind])
                if key=='timestamps' else match_note_wise_labels_detail([l for l in gold if l['type']==kind],[l for l in pred if l['type']==kind],score_event_count=count,type_mismatch_credit=0.)) for kind in KINDS}
            if key!='timestamps':
                row['note_units']=note_score(gold,pred)
                row['note_units_by_type']={kind:note_score([l for l in gold if l['type']==kind],[l for l in pred if l['type']==kind]) for kind in KINDS}
            policies[key].append(row)
        for threshold in SETTINGS['timestamp_iou_sensitivity']:
            sensitivities[f'iou_{threshold:g}'].append(timestamp_score(gold,before,'iou',threshold))
        for threshold in SETTINGS['timestamp_onset_sensitivity_seconds']:
            sensitivities[f'onset_{threshold:g}s'].append(timestamp_score(gold,before,'onset',threshold))
        oldpairs={x['prediction_index']:x['gold_index'] for x in outputs['status_quo'][1]['pairs'] if x['credit']>0}
        newpairs={x['prediction_index']:x['gold_index'] for x in outputs['transcription_lookup'][1]['pairs'] if x['credit']>0}
        timepairs={x['prediction_index']:x['gold_index'] for x in ts['pairs']}
        for i,(old,new) in enumerate(zip(before,after)):
            if old.get('score_event_indices',[])!=new.get('score_event_indices',[]) or i in newpairs or i in timepairs:
                mapping_changes.append(dict(sample=sid,id=old.get('id'),type=old['type'],start=old.get('start_time'),end=old.get('end_time'),
                    old_indices=old.get('score_event_indices',[]),new_indices=new.get('score_event_indices',[]),
                    old_match=oldpairs.get(i),new_match=newpairs.get(i),time_match=timepairs.get(i),evidence=projected['traces'][i],
                    matched_gold=gold[newpairs[i]] if i in newpairs else None))
        print('SCORED',sid,'matches',*[f'{k}={r[1]["credit"]:g}' for k,r in outputs.items()],flush=True)
    results={}
    draws=np.random.default_rng(p['seed']).integers(0,17,(p['bootstrap_replicates'],17))
    boots={}
    for key,rows in policies.items():
        counts=np.array([[r['strict'][k] for k in ['credit','predicted','gold']] for r in rows])
        totals=counts[draws].sum(axis=1);boots[key]=2*totals[:,0]/np.maximum(1,totals[:,1]+totals[:,2])
        result=dict(strict=aggregate([r['strict'] for r in rows]),samples=rows,
                    by_type={k:aggregate([r['by_type'][k] for r in rows]) for k in KINDS},
                    unlocated=sum(r['unlocated'] for r in rows),f1_95pct_ci=np.quantile(boots[key],[.025,.975]).tolist())
        if key!='timestamps':
            result['note_units']=aggregate([r['note_units'] for r in rows])
            result['note_units_by_type']={k:aggregate([r['note_units_by_type'][k] for r in rows]) for k in KINDS}
        results[key]=result
    results['transcription_lookup']['paired_delta_vs_status_quo_95pct_ci']=np.quantile(boots['transcription_lookup']-boots['status_quo'],[.025,.975]).tolist()
    assert len(policies['status_quo'])==17 and len(unscored)==4
    assert all(r['strict']['gold']==76 and r['strict']['predicted']==365 for r in results.values())
    baseline=read(LATEST/'trained_results.json')['results']['polytune']['0']['strict']
    for k in ['credit','predicted','gold','f1']:
        assert abs(results['status_quo']['strict'][k]-baseline[k])<1e-12
    verify(p)
    write(RUN/'results.json',dict(protocol_sha256=sha(RUN/'protocol.json'),results=results,
        timestamp_sensitivity={k:aggregate(v) for k,v in sensitivities.items()},timings=timings,excluded=unscored,
        timestamp_audit=dict(labels=len(time_audit),already_timestamped=len(time_audit),new_estimates=0,
                             teacher_estimated_missing=sum(x['estimated_missing_time'] for x in time_audit),rows=time_audit)))
    write(RUN/'mapping_changes.json',mapping_changes)
    write(RUN/'audit.json',dict(status='passed',same_model_checkpoint=True,same_native_predictions=True,
        same_label_types_times_and_counts=True,reference_status_quo_reproduced=True,
        test_recordings=21,scored_recordings=17,reference_labels=76,
        original_files_unchanged=True,policy3_projection_denies_gold_access=True,
        policy3_teacher_shared_source=True,results_sha256=sha(RUN/'results.json'),
        source_checks=len(p['source_sha256']),protected_file_checks=len(p['protected_sha256']),
        sanitized_input_checks=len(p['input_sha256'])))
    print('ASSESSMENT_COMPLETE',json_summary(results),flush=True)

def json_summary(results):
    return {k:v['strict'] for k,v in results.items()}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','project','run'])
    action=parser.parse_args().action
    {'prepare':prepare,'project':project,'run':evaluate}[action]()
