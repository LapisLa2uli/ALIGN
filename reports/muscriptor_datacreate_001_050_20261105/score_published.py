"""Score published DataCreate agent labels against the verified pre-seed user archive."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path[:0] = [str(ROOT/'align-model/src'),str(ROOT/'DataCreate/src')]
from alignmodel.melody import load_bundle_notes, micro_note_wise
from datacreate.melody import match_note_wise_labels_detail
ARCHIVE=ROOT/'DataCreate/archives/user-labels-before-agent-seed-20261006T074109Z-763e1b7d'
CONTENT={'extra_note','wrong_note','missed_note','repetition'}
def digest(raw): return hashlib.sha256(raw).hexdigest()
def load(p): return json.loads(p.read_bytes())
def dump(p,d): p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def core(labels):
    # The v9 evaluation uses the marked core (inclusive), excluding display padding.
    result=[]
    for lab in labels:
        l=dict(lab); p=l.get('score_part') or {}
        if not l.get('score_event_indices') and p.get('core_start_note_index') is not None:
            l['score_event_indices']=list(range(p['core_start_note_index'],p['core_end_note_index']+1))
        elif not l.get('score_event_indices') and p.get('start_note_index') is not None:
            a=p['start_note_index']+p.get('pad_notes',0); b=p['end_note_index']-p.get('pad_notes',0)
            l['score_event_indices']=list(range(a,max(a,b)+1))
        if l['type']=='repetition': l['extra_copies']=l.get('extra_copies') or 1
        result.append(l)
    return result

def metrics(gold,pred,n):
    sets={'all':(gold,pred),'content':([l for l in gold if l['type'] in CONTENT],[l for l in pred if l['type'] in CONTENT])}
    for kind in sorted({l['type'] for l in gold+pred}):
        sets['type:'+kind]=([l for l in gold if l['type']==kind],[l for l in pred if l['type']==kind])
    report={}
    for scope,(g,p) in sets.items():
        for method,credit in [('official',0.5),('strict',0.0)]:
            m=match_note_wise_labels_detail(g,p,score_event_count=n,type_mismatch_credit=credit)
            if m['status']!='available': raise ValueError((scope,m))
            report[scope+'/'+method]={k:v for k,v in m.items() if k!='pairs'}
    return report

def run(preflight):
    before=load(OUT/'before.json'); assert sorted(before)==[f'{i:03}' for i in range(1,51)]
    records={r['user_file']:r for r in load(ARCHIVE/'manifest.json')['records']}
    results={}; manifest=None if preflight else load(OUT/'publish/manifest.json')
    if manifest: assert manifest['status']=='complete' and set(manifest['protected'])==set(before)
    with ZipFile(ARCHIVE/'user_labels_before.zip') as users, ZipFile(ARCHIVE/'agent_labels_snapshot.zip') as agents:
        for name,h in before.items():
            s=ROOT/'DataCreate/samples'/name
            for f in ('labels.json','verified_score.musicxml','performance_audio.wav'): assert digest((s/f).read_bytes())==h[f],(name,f)
            old=s/'labels_agent.json' if preflight else OUT/'publish/backup'/name/'labels_agent.json'
            assert digest(old.read_bytes())==h['labels_agent.json']
            # Use the original frozen CTC predictions (sample 005 was edited in the GUI).
            ctc_run=Path(load(old)['agent_labeling']['run'])
            ctc_path=ctc_run/'staged'/name/'labels_agent.json'
            reference=next(r for r in load(ctc_run/'manifest.json')['results'] if r['sample']==name)
            assert digest(ctc_path.read_bytes())==reference['file_sha256']['labels_agent.json']
            ctc=load(ctc_path); provenance=ctc['agent_labeling']
            assert provenance['method']=='align_stack_v9' and not provenance.get('edited_in_gui')
            key=f'samples/{name}/labels.json'; raw=users.read(key)
            assert digest(raw)==records[key]['user_before_sha256']
            snap=agents.read(f'samples/{name}/labels_agent.json'); assert digest(snap)==records[key]['agent_sha256']
            for doc in (json.loads(snap)['agent_labeling'],provenance):
                for k,f in [('audio_sha256','performance_audio.wav'),('score_sha256','verified_score.musicxml')]: assert doc[k]==h[f],name
            gdoc=json.loads(raw);g=core([l for l in gdoc['labels'] if l.get('source')=='manual'])
            eligible=bool(g) or (not gdoc['labels'] and gdoc.get('annotator_id')=='annotator01')
            row=results[name]={'scored':eligible,'manual_count':len(g),'archive_member':key,'archive_sha256':digest(raw),'ctc_path':str(ctc_path),'ctc_sha256':digest(ctc_path.read_bytes())}
            documents={'CTC':ctc}
            if not preflight:
                ap=s/'labels_agent.json'; a=load(ap); prov=a['agent_labeling']
                assert prov['method']=='muscriptor_align_v9' and prov['transcriber']=='muscriptor' and not prov['human_labels_used']
                assert prov['candidate_sha256']==provenance['candidate_sha256']
                assert prov['audio_sha256']==h['performance_audio.wav'] and prov['score_sha256']==h['verified_score.musicxml']
                pub=next(r for r in manifest['results'] if r['sample']==name)
                assert digest(ap.read_bytes())==pub['file_sha256']['labels_agent.json']
                row.update(published=len(a['labels']),alignment_status=prov['status'],agent_sha256=digest(ap.read_bytes()))
                documents['MuScriptor']=a
            if eligible:
                n=len(load_bundle_notes(s))
                row['gold_types']=dict(Counter(l['type'] for l in g))
                row['metrics']={k:metrics(g,core(d['labels']),n) for k,d in documents.items()}
    summary={}
    for model in documents:
        rows=[r['metrics'][model] for r in results.values() if r['scored']]
        summary[model]={k:micro_note_wise([r[k] for r in rows if k in r]) for k in sorted(set().union(*rows))}
    rep={'gold_archive':str(ARCHIVE),'method':'Micro over reviewed takes. One-to-one exact canonical core location; full credit for matching type, 0.5 for wrong type (zero in strict metric). Repetitions include copy count. Only GUI-published labels scored.','content_types':sorted(CONTENT),'summary':summary,'samples':results}
    dump(OUT/('preflight.json' if preflight else 'evaluation.json'), rep)
    print('scored',sum(r['scored'] for r in results.values()),'manual labels',sum(r['manual_count'] for r in results.values()))
    for m,ss in summary.items():
        for k in ['all/official','all/strict','content/strict']: print(m,k,ss[k])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');run(ap.parse_args().preflight)
