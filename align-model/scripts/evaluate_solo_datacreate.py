"""Paired real-take regression check; only explicit manual labels 001–094 are gold."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
for p in ['align-model/src','align-model/scripts','DataCreate/src']:sys.path.insert(0,str(ROOT/p))
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'align-model/runs/solo-v1/numba-cache'))
import numpy as np
import torch
from alignmodel.transcription.mel_ctc_v3 import load_dual_checkpoint,extract_dual_mel,infer_dual_outputs
from alignmodel.transcription.mel_v1 import load_audio_mono
from alignmodel.joint import stack_v10,stack_solo_v1
from alignmodel.joint.presence_verifier_v1 import load_verifier,score_presence
from alignmodel.melody import load_bundle_notes,micro_note_wise
from datacreate.melody import match_note_wise_labels_detail
from precision_harness_v4 import rms_db


def core(label):
    if label.get('score_event_indices'):return label['score_event_indices']
    p=label.get('score_part') or {};a=p.get('core_start_note_index');b=p.get('core_end_note_index')
    if a is None and p.get('start_note_index') is not None:
        a=p['start_note_index']+p.get('pad_notes',0);b=p['end_note_index']-p.get('pad_notes',0)
    return list(range(a,max(a,b)+1)) if a is not None and b is not None else None


def canonical(labels):
    return [{**l,'score_event_indices':core(l),**({'extra_copies':l.get('extra_copies') or 1} if l['type']=='repetition' else {})}
            for l in labels]


def human_gold(archive):
    """Use explicit human annotations, never labels copied from model feedback."""
    gold={};provenance={}
    for i in range(1,95):
        name=f'{i:03}';path=ROOT/'DataCreate/samples'/name/'labels.json'
        if not path.exists():continue
        raw=path.read_bytes();doc=json.loads(raw)
        manual=[l for l in doc.get('labels',[]) if l.get('source')=='manual']
        if manual or (not doc.get('labels') and doc.get('annotator_id')=='annotator01'):
            gold[name]=manual;provenance[name]={'source':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
    if archive and archive.exists():
        manifest=json.loads((archive/'manifest.json').read_text())
        records={r['user_file']:r for r in manifest['records']}
        with zipfile.ZipFile(archive/'user_labels_before.zip') as users, zipfile.ZipFile(archive/'agent_labels_snapshot.zip') as agents:
            for i in range(1,95):
                name=f'{i:03}';member=f'samples/{name}/labels.json'
                if name in gold or member not in users.namelist():continue
                raw=users.read(member);doc=json.loads(raw)
                manual=[l for l in doc.get('labels',[]) if l.get('source')=='manual']
                if not manual and not (not doc.get('labels') and doc.get('annotator_id')=='annotator01'):continue
                if hashlib.sha256(raw).hexdigest()!=records[member]['user_before_sha256']:
                    raise ValueError(f'Archived human label checksum mismatch: {name}')
                snapshot_raw=agents.read(f'samples/{name}/labels_agent.json')
                if hashlib.sha256(snapshot_raw).hexdigest()!=records[member]['agent_sha256']:
                    raise ValueError(f'Archived input-provenance checksum mismatch: {name}')
                snapshot=json.loads(snapshot_raw)['agent_labeling']
                sample=ROOT/'DataCreate/samples'/name
                for key,file in [('audio_sha256','performance_audio.wav'),('score_sha256','verified_score.musicxml')]:
                    if snapshot.get(key)!=hashlib.sha256((sample/file).read_bytes()).hexdigest():
                        raise ValueError(f'Archived human labels refer to changed {file}: {name}')
                gold[name]=manual
                provenance[name]={'source':str(archive/'user_labels_before.zip'),'member':member,
                                  'sha256':hashlib.sha256(raw).hexdigest(),'score_audio_hashes_verified':True}
    if not gold:
        raise ValueError('No trustworthy human labels found; do not evaluate against agent predictions')
    return gold,provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--versions',nargs='+',choices=['v10','solo'],default=['v10','solo'])
    p.add_argument('--gold-archive',type=Path,default=ROOT/'DataCreate/archives/user-labels-before-agent-seed-20261006T074109Z-763e1b7d')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(3)
    c=json.loads((ROOT/'align-model/runs/stack-v10/CANDIDATE_STACK_V10.json').read_text())
    base=json.loads(Path(c['base_candidate']).read_text());boundary=json.loads(Path(c['boundary_model']).read_text())
    models={v:load_dual_checkpoint(c['checkpoint'] if v=='v10' else args.checkpoint,args.device)[0]
            for v in args.versions}
    verifier=load_verifier(Path(base['verifier']),args.device) if 'v10' in args.versions else None
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    checkpoints={v:sha(c['checkpoint'] if v=='v10' else args.checkpoint) for v in args.versions}
    configuration=hashlib.sha256(json.dumps({'v10':c,'base':base,'boundary':boundary},sort_keys=True).encode()).hexdigest()
    source_files=sorted((ROOT/'align-model/src/alignmodel/joint').glob('*.py'))+sorted((ROOT/'align-model/src/alignmodel/transcription').glob('*.py'))
    sources=hashlib.sha256(''.join(sha(path) for path in source_files).encode()).hexdigest()
    cfg={k:c[k] for k in ['decoder','gate','minimum_match_fraction','same_pitch','boundary']}
    gold,gold_provenance=human_gold(args.gold_archive)
    reports={version:collections.defaultdict(list) for version in args.versions};rows=[]
    for name in sorted(set(gold)|{'007','095'}):
        sample=ROOT/'DataCreate/samples'/name
        row={'sample':name,'gold_used':name in gold,'models':{},
             'audio_sha256':hashlib.sha256((sample/'performance_audio.wav').read_bytes()).hexdigest(),
             'score_sha256':hashlib.sha256((sample/'verified_score.musicxml').read_bytes()).hexdigest()}
        mel=None
        for version in args.versions:
            provenance={k:row[k] for k in ['audio_sha256','score_sha256']}
            provenance.update(checkpoint_sha256=checkpoints[version],configuration_sha256=configuration,
                              source_sha256=sources,device=args.device,written_to_sounding=-2)
            path=args.output/f'{name}-{version}.json'
            payload=json.loads(path.read_text()) if path.exists() else {}
            if payload.get('provenance')!=provenance:
                if mel is None:
                    audio=load_audio_mono(sample/'performance_audio.wav',22050)
                    mel,_=extract_dual_mel(audio,args.device)
                model=models[version]
                outputs=infer_dual_outputs(model,mel,args.device)
                if version=='v10':
                    outputs={k:v.astype(np.float16).astype(np.float32) for k,v in outputs.items()}
                outputs['rms_db']=rms_db(audio,len(outputs['ctc']))
                if version=='v10':
                    presence=lambda q:score_presence(verifier,mel,q,args.device)
                    result=stack_v10.align_outputs(outputs,sample/'verified_score.musicxml',base,mel=mel,audio=audio,
                                                   presence=presence,config=cfg,boundary_model=boundary)
                    module=stack_v10
                else:
                    result=stack_solo_v1.align_outputs(outputs,sample/'verified_score.musicxml',base,
                                                       midi_min=model.config.midi_min,written_to_sounding=-2)
                    module=stack_solo_v1
                document=module.feedback(*result)
                summary={'status':document['status'],'labels':len(document['labels']),
                         'notes':len(result[1].notes),'match_fraction':result[1].match_fraction}
                payload={'feedback':document,'summary':summary,'provenance':provenance,
                         'events':[{'pitch':e.pitch,'score_span':e.score_span,'relationship':e.relationship,
                            'copy_pass':e.copy_pass,'start':e.start,'end':e.end} for e in result[1].events]}
                path.write_text(json.dumps(payload,indent=2,allow_nan=False))
            document=payload['feedback']
            row['models'][version]=payload['summary']
            if name in gold:
                count=len(load_bundle_notes(sample));g=canonical(gold[name]);pred=canonical(document['labels'])
                for kind in ['wrong_note','missed_note','extra_note','repetition']:
                    reports[version][kind].append(match_note_wise_labels_detail([l for l in g if l['type']==kind],
                        [l for l in pred if l['type']==kind],score_event_count=count))
        rows.append(row);print(json.dumps(row),flush=True)
    result={'gold_population':len(gold),'gold_provenance':gold_provenance,'regression_samples':rows,
            'metric':'canonical affected score-note identity; no timestamp matching',
            'labels_095_onwards_used':False,
            'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'summary':{v:{k:micro_note_wise(values) for k,values in r.items()} for v,r in reports.items()}}
    filename='report.json' if len(args.versions)==2 else f'report-{args.versions[0]}.json'
    (args.output/filename).write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
