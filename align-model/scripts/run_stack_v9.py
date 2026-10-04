"""Run the separately versioned v9 same-pitch repair candidate on a score and performance recording."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'align-model/runs/stack-v9/numba-cache'))
for p in ('align-model/src','align-model/scripts','DataCreate/src','synth-pipeline/src'):sys.path.insert(0,str(ROOT/p))
import numpy as np
import torch
from alignmodel.transcription.mel_ctc_v3 import load_dual_checkpoint,extract_dual_mel,infer_dual_outputs
from alignmodel.transcription.mel_v1 import load_audio_mono
from alignmodel.joint.presence_verifier_v1 import load_verifier,score_presence
from alignmodel.joint.stack_v9 import align_outputs,feedback
from precision_harness_v4 import rms_db

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score',type=Path,required=True)
    parser.add_argument('--audio',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,default=ROOT/'align-model/runs/stack-v9/CANDIDATE_STACK_V9.json')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    candidate=json.loads(args.candidate.read_text())
    base=json.loads(Path(candidate['base_candidate']).read_text())
    for relative,digest in candidate.get('code_sha256',{}).items():
        if hashlib.sha256((ROOT/'align-model'/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Candidate source changed: {relative}')
    checkpoint=Path(candidate['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=candidate['checkpoint_sha256']:
        raise ValueError('Candidate checkpoint hash mismatch')
    if hashlib.sha256(Path(base['verifier']).read_bytes()).hexdigest()!=base['verifier_sha256']:
        raise ValueError('Presence verifier hash mismatch')
    # v9 retains the checkpoint's scope: written Bb-clarinet score.
    torch.set_num_threads(4)
    model,_=load_dual_checkpoint(checkpoint,args.device);model.eval()
    verifier=load_verifier(Path(base['verifier']),args.device)
    audio=load_audio_mono(args.audio,22050)
    with torch.inference_mode():
        mel,_=extract_dual_mel(audio,args.device)
        outputs=infer_dual_outputs(model,np.asarray(mel,np.float32),args.device)
    # Match the validated serialized probability precision.
    outputs={k:v.astype(np.float16).astype(np.float32) for k,v in outputs.items()}
    outputs['rms_db']=rms_db(audio,len(outputs['ctc']))
    presence=lambda queries:score_presence(verifier,np.asarray(mel,np.float32),queries,args.device)
    index,alignment,events,deletions,info=align_outputs(outputs,args.score,base,mel=mel,audio=audio,presence=presence,
        config={'minimum_match_fraction':candidate.get('minimum_match_fraction',.45),'decoder':candidate['decoder'],'gate':candidate.get('gate',{}),'same_pitch':candidate.get('same_pitch',{})})
    document=feedback(index,alignment,events,deletions,info)
    document['provenance']={'candidate':str(args.candidate.resolve()),'checkpoint_sha256':candidate['checkpoint_sha256'],
                            'candidate_sha256':hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                            'score':str(args.score.resolve()),'audio':str(args.audio.resolve()),
                            'audio_sha256':hashlib.sha256(args.audio.read_bytes()).hexdigest(),
                            'score_sha256':hashlib.sha256(args.score.read_bytes()).hexdigest(),
                            'pitch_convention':'written Bb-clarinet, sounding = written - 2'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(document,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':document['status'],'labels':len(document['labels']),'output':str(args.output)}))
if __name__=='__main__':main()



