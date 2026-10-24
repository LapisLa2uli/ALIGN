"""Publish one solo-model result with backups; never edits human labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
for part in ['align-model/src','align-model/scripts','DataCreate/src']:
    sys.path.insert(0,str(ROOT/part))
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'align-model/runs/solo-v1/numba-cache'))
import numpy as np
import torch
from alignmodel.transcription.mel_ctc_v3 import load_dual_checkpoint,extract_dual_mel,infer_dual_outputs
from alignmodel.transcription.mel_v1 import load_audio_mono
from alignmodel.joint.stack_solo_v1 import align_outputs,feedback,REVISION
from alignmodel.joint.datacreate_v9 import gui_documents
from datacreate.align_bridge import _compatibility_alignment
from datacreate.config import PipelineConfig
from datacreate.validation import validate_labels_file
from precision_harness_v4 import rms_db


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def report_progress(sample, step):
    path = sample/'alignment_progress.json'
    temporary = path.with_suffix('.tmp')
    write(temporary, {'step':step})
    os.replace(temporary, path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--sample-rate',type=int,default=22050)
    p.add_argument('--hop-length',type=int,default=512)
    p.add_argument('--written-to-sounding',type=int)
    args=p.parse_args();sample=args.sample.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False);stage=out/'staged';stage.mkdir();backup=out/'backup';backup.mkdir()
    candidate=json.loads(args.candidate.read_text())
    if sha(candidate['checkpoint'])!=candidate['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
    for file,digest in candidate.get('code_sha256',{}).items():
        if sha(ROOT/file)!=digest:raise ValueError(f'Solo source changed: {file}')
    base=json.loads(Path(candidate['base_candidate']).read_text())
    if sha(candidate['base_candidate'])!=candidate['base_candidate_sha256']:raise ValueError('Aligner configuration changed')
    base['decoder']=candidate['decoder']
    files=['note_alignment_v2.json','transcription_notes.json','labels_agent.json','alignment.npz','candidates.json',
           'feedback_solo.json','note_alignment_solo.json']
    before={name:sha(sample/name) if (sample/name).exists() else None for name in files}
    protected={name:sha(sample/name) for name in ['performance_audio.wav','verified_score.musicxml','labels.json'] if (sample/name).exists()}
    for name in files:
        if before[name] is not None:shutil.copy2(sample/name,backup/name)
    report_progress(sample, 'transcriber')
    torch.set_num_threads(3);model,payload=load_dual_checkpoint(candidate['checkpoint'],args.device)
    if payload.get('pitch_space')!='sounding':raise ValueError('Expected sounding-pitch checkpoint')
    audio=load_audio_mono(sample/'performance_audio.wav',22050);mel,_=extract_dual_mel(audio,args.device)
    outputs=infer_dual_outputs(model,np.asarray(mel,np.float32),args.device)
    outputs['rms_db']=rms_db(audio,len(outputs['ctc']))
    result=align_outputs(outputs,sample/'verified_score.musicxml',base,midi_min=model.config.midi_min,
                         written_to_sounding=args.written_to_sounding,
                         progress=lambda step:report_progress(sample,step))
    report_progress(sample, 'labels')
    index,alignment,events,deletions,info=result
    doc=feedback(*result);doc['schema_version']='align-score-feedback-solo-v1'
    provenance={'pipeline_revision':REVISION,'candidate_sha256':sha(args.candidate),'checkpoint_sha256':candidate['checkpoint_sha256'],
                'audio_sha256':protected['performance_audio.wav'],'score_sha256':protected['verified_score.musicxml'],
                'written_to_sounding':info['written_to_sounding'],'pitch_convention':'acoustics=sounding; alignment=written score',
                'human_labels_used':False}
    doc['provenance']=provenance
    gui,labels=gui_documents(sample,index,alignment,doc,duration=len(audio)/22050,provenance=provenance)
    gui['label_generation'].update(method='align_stack_solo',annotator_id='align_solo_review')
    gui['summary'].update(backend='ALIGN solo multi-instrument v1 (experimental)',candidate_generation='wide-range dual-mel CTC',
        review_notice='Experimental feedback: review suggested errors carefully, especially for instruments other than clarinet.')
    labels['annotator_id']='align_solo_review';labels['agent_labeling']['method']='align_stack_solo'
    for label in gui['labels']:label['comment']='Suggested by the solo model; review the highlighted score notes.'
    for name in ['note_alignment_v2.json','note_alignment_solo.json']:write(stage/name,gui)
    write(stage/'labels_agent.json',labels);write(stage/'feedback_solo.json',doc)
    write(stage/'transcription_notes.json',{'engine':'align-solo','transcribed_notes':gui['transcribed_notes'],'summary':gui['summary'],'provenance':provenance})
    write(stage/'candidates.json',{'schema_version':'1.2','labels':[{**l,'source':'auto'} for l in gui['labels']]})
    config=PipelineConfig.load();config.audio['sample_rate']=args.sample_rate;config.mel['hop_length']=args.hop_length
    _compatibility_alignment(gui,stage,config)
    errors=validate_labels_file(stage/'labels_agent.json',config)
    if errors:raise ValueError(errors)
    for name,digest in protected.items():
        if sha(sample/name)!=digest:raise RuntimeError(f'Input changed: {name}')
    for name,digest in before.items():
        if (sha(sample/name) if (sample/name).exists() else None)!=digest:raise RuntimeError(f'Output changed: {name}')
    for name in files:
        temporary=sample/(name+'.solo.tmp');shutil.copyfile(stage/name,temporary);os.replace(temporary,sample/name)
    write(out/'manifest.json',{'status':'complete','sample':str(sample),'backup':str(backup),'protected':protected,'before':before,'provenance':provenance})
    print(json.dumps({'status':doc['status'],'labels':len(gui['labels']),'backup':str(backup)}))


if __name__=='__main__':main()
