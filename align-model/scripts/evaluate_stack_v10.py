"""Fresh paired DataCreate inference; v10 changes only repeat-boundary handling.

No >=095 labels are read as truth. Input files and UI outputs stay untouched.
"""
import os,sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'align-model/runs/stack-v10'
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'align-model/runs/stack-v9/numba-cache'))
for p in ('align-model/src','align-model/scripts','DataCreate/src','synth-pipeline/src'):
    sys.path.insert(0,str(ROOT/p))
import numpy as np
import torch
from alignmodel.transcription.mel_v1 import load_audio_mono
from alignmodel.transcription.mel_ctc_v3 import load_dual_checkpoint,extract_dual_mel,infer_dual_outputs
from alignmodel.joint.presence_verifier_v1 import load_verifier,score_presence
from alignmodel.joint import stack_v9_passage as v9,stack_v10 as v10
from precision_harness_v4 import rms_db


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    c=read(OUT/'CANDIDATE_STACK_V10.json');base=read(c['base_candidate']);model=read(c['boundary_model'])
    transcriber,_=load_dual_checkpoint(Path(c['checkpoint']),'cuda');transcriber.eval()
    verifier=load_verifier(Path(base['verifier']),'cuda')
    config={k:c[k] for k in ('decoder','gate','minimum_match_fraction','same_pitch','boundary')}
    report=[]
    for i in range(1,96):
        name=f'{i:03}';sample=ROOT/'DataCreate/samples'/name
        dest=OUT/'paired'/f'{name}.json'
        if dest.exists():report.append(read(dest));continue
        audio=load_audio_mono(sample/'performance_audio.wav',22050)
        with torch.inference_mode():
            mel,_=extract_dual_mel(audio,'cuda');outputs=infer_dual_outputs(transcriber,mel,'cuda')
        outputs={k:v.astype(np.float16).astype(np.float32) for k,v in outputs.items()}
        outputs['rms_db']=rms_db(audio,len(outputs['ctc']))
        presence=lambda q:score_presence(verifier,mel,q,'cuda')
        t=time.perf_counter()
        before=v9.align_outputs(outputs,sample/'verified_score.musicxml',base,mel=mel,audio=audio,presence=presence,config=config)
        baseline_seconds=time.perf_counter()-t;t=time.perf_counter()
        after=v10.align_outputs(outputs,sample/'verified_score.musicxml',base,mel=mel,audio=audio,presence=presence,config=config,boundary_model=model,baseline=before)
        seconds=time.perf_counter()-t
        row={'sample':name,'baseline_seconds':baseline_seconds,'refinement_seconds':seconds,'labels_used':False,
             'audio_sha256':sha(sample/'performance_audio.wav'),'score_sha256':sha(sample/'verified_score.musicxml')}
        for version,result,module in [('v9',before,v9),('v10',after,v10)]:
            index,alignment,events,deletions,info=result
            document=module.feedback(*result)
            payload={'sample':name,'diagnostics':info,'feedback':document,
                'events':[{'pitch':e.pitch,'start':e.start,'end':e.end,'score_span':e.score_span,
                    'copy_pass':e.copy_pass,'relationship':e.relationship,'note_index':e.rendered_index} for e in events],
                'missed_score_event_indices':sorted(deletions)}
            write(OUT/'predictions'/version/'datacreate'/f'{name}.json',payload)
            row[version]={'status':info['status'],'notes':len(alignment.notes),'labels':len(document['labels']),
                          'merged':info['same_pitch_repair']['merged_boundaries']}
        row['refinement']=after[-1]['boundary_refinement']
        write(dest,row);report.append(row)
        print(name,'restored',row['refinement']['accepted_restorations'],'labels',row['v9']['labels'],row['v10']['labels'],flush=True)
    import evaluate_stack_v7 as ev
    ev.OUT=OUT
    ev.real_metrics('v9');ev.real_metrics('v10')
    write(OUT/'evaluation.json',{'population':95,'accuracy_population':'reviewed 001-094 only; 095 output regression only',
        'synthetic_end_to_end':'blocked: E: and acoustic caches unavailable','candidate_sha256':sha(OUT/'CANDIDATE_STACK_V10.json'),
        'restored_boundaries':sum(len(r['refinement']['accepted_restorations']) for r in report),
        'changed_clips':[r['sample'] for r in report if r['refinement']['accepted_restorations']],
        'samples':report})


if __name__=='__main__':main()
