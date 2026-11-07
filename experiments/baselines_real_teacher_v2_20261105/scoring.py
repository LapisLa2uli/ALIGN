"""Shared validation/test scorer, strictly based on reference-note identity."""
import sys
from common import ROOT,RUN,PREVIOUS,read,write
sys.path[:0]=[str(ROOT/'DataCreate/src'),str(ROOT/'align-model/src'),str(ROOT/'baselines/scripts'),str(ROOT/'baselines/common')]
sys.path.append(str(PREVIOUS))
from supervision import canonical,KINDS

def score_directory(directory,ids,flavor,save=True):
    from datacreate.melody import match_note_wise_labels_detail
    from alignmodel.melody import micro_note_wise
    from label_datacreate_baselines_v2 import document_from_midi
    rows=[];caps=windows=0
    for sid in ids:
        midi=directory/sid/'mix.mid'
        doc=document_from_midi(midi,PREVIOUS/'bundles'/sid/'verified_score.musicxml',model=flavor,sample_id=sid)
        if save:write(directory/sid/'labels.json',doc)
        gold=read(PREVIOUS/'teacher_gold'/f'{sid}.json')
        pred=canonical(doc['labels'])+[{'type':'unclassified'} for _ in range(doc['baseline_labeling']['unclassified_count'])]
        row={'sample':sid,'scored':gold['status']=='ok','predicted_labels':len(pred),'teacher_status':gold['status']}
        if row['scored']:
            row['strict']=match_note_wise_labels_detail(gold['labels'],pred,score_event_count=gold['score_event_count'],type_mismatch_credit=0.)
            row['by_type']={k:match_note_wise_labels_detail([l for l in gold['labels'] if l['type']==k],[l for l in pred if l['type']==k],score_event_count=gold['score_event_count'],type_mismatch_credit=0.) for k in KINDS}
            assert row['strict']['status']=='available'
        done=read(midi.with_suffix('.done.json'))
        caps+=done.get('capped_windows',0);windows+=done.get('windows',0)
        rows.append(row)
    scored=[r for r in rows if r['scored']]
    result={'strict':micro_note_wise([r['strict'] for r in scored]),'by_type':{k:micro_note_wise([r['by_type'][k] for r in scored]) for k in KINDS},
            'scored_recordings':len(scored),'inferred_recordings':len(rows),'capped_windows':caps,'windows':windows,
            'capped_rate':caps/max(1,windows),'samples':rows}
    if save:write(directory/'metrics.json',result)
    return result
