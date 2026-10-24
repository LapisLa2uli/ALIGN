"""Transcribe solo audio in sounding pitch; optionally align to a MusicXML score."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'align-model/src'))
import torch
from alignmodel.transcription.mel_ctc_v3 import load_dual_checkpoint,extract_dual_mel,infer_dual_outputs
from alignmodel.transcription.mel_v1 import load_audio_mono
from alignmodel.transcription.solo_v1 import predict_notes
from alignmodel.joint.stack_solo_v1 import score_transposition,align_outputs,feedback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,help='Override the frozen selected candidate checkpoint')
    p.add_argument('--candidate',type=Path,default=ROOT/'align-model/runs/solo-v1/CANDIDATE_SOLO_V1.json')
    p.add_argument('--score',type=Path)
    p.add_argument('--written-to-sounding',type=int)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(3)
    candidate=None
    if args.checkpoint is None:
        if not args.candidate.is_file():
            raise FileNotFoundError('No selected solo candidate yet; train/select it or pass --checkpoint explicitly')
        candidate=json.loads(args.candidate.read_text())
        args.checkpoint=Path(candidate['checkpoint'])
        if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=candidate['checkpoint_sha256']:
            raise ValueError('Selected checkpoint hash mismatch')
    model,payload=load_dual_checkpoint(args.checkpoint,args.device)
    if payload.get('pitch_space')!='sounding':raise ValueError('Expected a solo sounding-pitch checkpoint')
    audio=load_audio_mono(args.audio,22050);mel,_=extract_dual_mel(audio,args.device)
    decoder=(candidate or payload).get('decoder',{'blank_scale':.5,'candidate_threshold':.08})
    notes,outputs=predict_notes(model,mel,args.device,blank_scale=decoder.get('blank_scale',.5))
    shift=score_transposition(args.score,args.written_to_sounding) if args.score else (args.written_to_sounding or 0)
    for i,n in enumerate(notes):
        n.update(sounding_pitch=n['pitch'],written_pitch=n['pitch']-shift,
                 end=notes[i+1]['start'] if i+1<len(notes) else min(len(audio)/22050,n['start']+.1))
    document={'schema_version':'align-solo-transcription-v1','pitch_space':'sounding','written_to_sounding':shift,
              'notes':notes,'note_end_policy':'estimated from next onset; not acoustic offsets',
              'checkpoint':str(args.checkpoint.resolve()),
              'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}
    if args.score:
        from precision_harness_v4 import rms_db
        outputs['rms_db']=rms_db(audio,len(outputs['ctc']))
        base_path=Path(candidate['base_candidate']) if candidate else ROOT/'align-model/runs/precision-v4/CANDIDATE_STACK_V6.json'
        if candidate and hashlib.sha256(base_path.read_bytes()).hexdigest()!=candidate['base_candidate_sha256']:
            raise ValueError('Selected aligner configuration hash mismatch')
        base=json.loads(base_path.read_text())
        base['decoder']={'blank_scale':decoder.get('blank_scale',.5),'candidate_threshold':decoder.get('candidate_threshold',.08)}
        result=align_outputs(outputs,args.score,base,midi_min=model.config.midi_min,written_to_sounding=shift)
        document['feedback']=feedback(*result)
        document['score_mapping']=[{'pitch':e.pitch,'score_span':e.score_span,'relationship':e.relationship,
                                   'start':e.start,'end':e.end} for e in result[1].events]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'notes':len(notes),'output':str(args.output)}))


if __name__=='__main__':main()
