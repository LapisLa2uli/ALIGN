"""Fresh v9 inference for all DataCreate bundles, staged/validated before UI install.

Human labels are hash-protected. Existing UI files are backed up byte-for-byte.
Includes >=095 for visual review ONLY; never reads their human label content.
"""
from __future__ import annotations
import os,sys,json,hashlib,shutil,argparse
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'align-model/runs/stack-v9/numba-cache'))
for p in ('align-model/src','align-model/scripts','DataCreate/src','synth-pipeline/src'):sys.path.insert(0,str(ROOT/p))
import numpy as np
import torch
from alignmodel.transcription.mel_ctc_v3 import load_dual_checkpoint,extract_dual_mel,infer_dual_outputs
from alignmodel.transcription.mel_v1 import load_audio_mono
from alignmodel.joint.presence_verifier_v1 import load_verifier,score_presence
from alignmodel.joint.stack_v9_passage import align_outputs,feedback
from alignmodel.joint.passage_v1 import REVISION
from alignmodel.joint.datacreate_v9 import gui_documents
from precision_harness_v4 import rms_db
from datacreate.align_bridge import _compatibility_alignment
from datacreate.config import PipelineConfig
from datacreate.note_alignment import build_note_alignment
from datacreate.validation import validate_labels_file

FILES=('note_alignment_v2.json','transcription_notes.json','labels_agent.json','alignment.npz','candidates.json',
       'note_alignment_v9.json','transcription_v9.json','feedback_v9.json')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def report_progress(sample, step):
    path = sample / 'alignment_progress.json'
    temporary = path.with_suffix('.tmp')
    write(temporary, {'step': step})
    os.replace(temporary, path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=Path,default=ROOT/'DataCreate/samples')
    parser.add_argument('--sample',type=Path,help='Regenerate just this sample (used by the annotation UI)')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sample-rate',type=int,default=22050,help='UI compatibility timing rate')
    parser.add_argument('--hop-length',type=int,default=512,help='UI compatibility timing hop')
    parser.add_argument('--candidate',type=Path,default=ROOT/'align-model/runs/stack-v9/CANDIDATE_STACK_V9.json')
    parser.add_argument('--transcription',type=Path,help='MuScriptor written-pitch notes for a single sample')
    parser.add_argument('--transcriptions-dir',type=Path,help='Publish only the numbered samples with matching NNN.json note files')
    args=parser.parse_args();out=args.output.resolve();samples_root=args.samples.resolve()
    batch_notes={}
    external_contexts={}
    if args.transcriptions_dir:
        if args.sample or args.transcription:
            parser.error('--transcriptions-dir cannot be combined with single-sample options')
        batch_notes={p.stem:p for p in sorted(args.transcriptions_dir.glob('[0-9][0-9][0-9].json'))}
        if not batch_notes:
            parser.error('No numbered transcriptions found')
    if args.transcription and not args.sample:
        parser.error('--transcription requires --sample')
    if out.exists():raise FileExistsError(out)
    c=json.loads(args.candidate.read_text());base=json.loads(Path(c['base_candidate']).read_text())
    for relative,digest in c['code_sha256'].items():assert sha(ROOT/'align-model'/relative)==digest,relative
    assert sha(c['checkpoint'])==c['checkpoint_sha256']
    assert sha(base['verifier'])==base['verifier_sha256']
    if args.sample:
        samples=[args.sample.resolve()];samples_root=samples[0].parent
        for name in ('performance_audio.wav','verified_score.musicxml'):
            if not (samples[0]/name).is_file():raise FileNotFoundError(samples[0]/name)
    elif batch_notes:
        samples=[samples_root/name for name in batch_notes]
    else:
        samples=sorted(p for p in samples_root.iterdir() if p.is_dir() and (p/'performance_audio.wav').is_file() and (p/'verified_score.musicxml').is_file())
    out.mkdir(parents=True)
    protected={p.name:{f:sha(p/f) for f in ('labels.json','performance_audio.wav','verified_score.musicxml') if (p/f).is_file()} for p in samples}
    backup={}
    for sample in samples:
        backup[sample.name]={}
        for name in FILES:
            path=sample/name
            if path.exists():
                dest=out/'backup'/sample.name/name;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,dest);assert sha(dest)==sha(path)
                backup[sample.name][name]=sha(path)
            else:backup[sample.name][name]=None
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'samples_root':str(samples_root),
        'candidate':str(args.candidate.resolve()),'candidate_sha256':sha(args.candidate),
        'backup':backup,'protected':protected,'files':FILES,'status':'staging','results':[]}
    write(out/'manifest.json',manifest)
    torch.set_num_threads(4);device=args.device
    model,_=load_dual_checkpoint(Path(c['checkpoint']),device);model.eval()
    verifier=load_verifier(Path(base['verifier']),device);ui_config=PipelineConfig.load()
    ui_config.audio['sample_rate']=args.sample_rate;ui_config.mel['hop_length']=args.hop_length
    cfg={k:c[k] for k in ('decoder','gate','minimum_match_fraction','same_pitch')}
    for pos,sample in enumerate(samples,1):
        stage=out/'staged'/sample.name;stage.mkdir(parents=True)
        report_progress(sample, 'transcriber')
        audio=load_audio_mono(sample/'performance_audio.wav',22050)
        with torch.inference_mode():
            mel,_=extract_dual_mel(audio,device)
            outputs=infer_dual_outputs(model,np.asarray(mel,np.float32),device)
        outputs={k:v.astype(np.float16).astype(np.float32) for k,v in outputs.items()}
        outputs['rms_db']=rms_db(audio,len(outputs['ctc']))
        presence=lambda queries:score_presence(verifier,np.asarray(mel,np.float32),queries,device)
        # Decoding still belongs to transcription; align_outputs reports passage
        # retrieval and alignment when those operations actually begin.
        external = None
        transcription_path=batch_notes.get(sample.name) or args.transcription
        if transcription_path:
            from alignmodel.joint.stack_v9_passage import align_notes
            from alignmodel.joint.index import ScoreEventIndex
            from datacreate.muscriptor_bridge import REVISION as MUSCRIPTOR_REVISION
            external = json.loads(transcription_path.read_text(encoding='utf-8'))
            if external.get('transcribed_notes_pitch_space') != 'written':
                raise ValueError('External notes must use written pitch')
            for key, filename in [('audio_sha256','performance_audio.wav'),('score_sha256','verified_score.musicxml')]:
                if external['provenance'][key] != protected[sample.name][filename]:
                    raise ValueError('MuScriptor input changed before alignment: ' + filename)
            from datacreate.muscriptor_audio import context_signature
            external_context = external['provenance'].get('audio_context', {})
            expected_context = context_signature(sample)
            if any(external_context.get(k) != v for k, v in expected_context.items()):
                raise ValueError('MuScriptor transcription has stale audio context; transcribe again')
            external_contexts[sample.name] = expected_context
            rows = [(n['pitch'], n['start'], n['end'], n.get('confidence', 1.0))
                    for n in external['transcribed_notes']]
            report_progress(sample, 'locator')
            index = ScoreEventIndex.from_musicxml(sample/'verified_score.musicxml')
            alignment,events,deletions,info = align_notes(rows,index,sample/'verified_score.musicxml',base,
                evidence={**outputs,'hop':256/22050,'midi_min':52,'presence':presence}, config=cfg,
                progress=lambda step:report_progress(sample,step))
            info.update(pipeline_revision=MUSCRIPTOR_REVISION,
                same_pitch_repair={'source_groups':[[i] for i in range(len(rows))], 'merged_boundaries':0, 'pairs_checked':0},
                transcription_backend='muscriptor-medium')
        else:
            index,alignment,events,deletions,info=align_outputs(outputs,sample/'verified_score.musicxml',base,
                mel=mel,audio=audio,presence=presence,config=cfg,progress=lambda step:report_progress(sample,step))
        report_progress(sample, 'labels')
        doc=feedback(index,alignment,events,deletions,info, measured_offsets=bool(external))
        provenance={'candidate':str(args.candidate.resolve()),'candidate_sha256':sha(args.candidate),
            'pipeline_revision':REVISION,
            'checkpoint_sha256':c['checkpoint_sha256'],'audio_sha256':protected[sample.name]['performance_audio.wav'],
            'score_sha256':protected[sample.name]['verified_score.musicxml'],
            'pitch_convention':'written Bb-clarinet; sounding = written - 2',
            'note_end_policy':'next decoded onset, last onset + 0.1s, merged as needed; not measured offsets',
            'run':str(out),'inference':'fresh waveform inference','human_labels_used':False}
        from alignmodel.joint.rhythm_v1 import REVISION as RHYTHM_REVISION
        provenance['rhythm_revision']=RHYTHM_REVISION
        doc['provenance']=provenance
        if external:
            provenance.update(pipeline_revision=MUSCRIPTOR_REVISION, transcriber='muscriptor',
                transcription=external['provenance'], transcription_sha256=sha(transcription_path),
                note_end_policy='MuScriptor note offsets; score mapping may merge notes',
                feedback_calibration='Existing ALIGN clarinet gates; MuScriptor combination is experimental')
        gui,labels=gui_documents(sample,index,alignment,doc,duration=len(audio)/22050,provenance=provenance)
        if external:
            gui['summary'].update(backend='MuScriptor + ALIGN', candidate_generation='MuScriptor medium',
                note_end_policy='MuScriptor note offsets; score mapping may merge notes.',
                review_notice='MuScriptor transcription with ALIGN score alignment. Review suggested errors.')
            gui['label_generation']['method']='muscriptor_align_v9'
            labels['agent_labeling']['method']='muscriptor_align_v9'
            for label in gui['labels']:
                if label['type'] != 'rhythm_error':
                    label['comment']='MuScriptor transcription with ALIGN score alignment; review suggested errors.'
            for label in labels['labels']:
                if label['type'] != 'rhythm_error':
                    label['comment']='MuScriptor transcription with ALIGN score alignment; review suggested errors.'
        transcription={'engine':'align-v9','sample_id':sample.name,'transcribed_notes':gui['transcribed_notes'],
                       'summary':gui['summary'],'provenance':provenance,'same_pitch_repair':info['same_pitch_repair']}
        if external:
            transcription['engine']='muscriptor-align-v9'
        for name in ('note_alignment_v2.json','note_alignment_v9.json'):write(stage/name,gui)
        for name in ('transcription_notes.json','transcription_v9.json'):write(stage/name,transcription)
        write(stage/'labels_agent.json',labels);write(stage/'feedback_v9.json',doc)
        write(stage/'candidates.json',{'schema_version':'1.2','labels':[{**l,'source':'auto'} for l in labels['labels']]})
        _compatibility_alignment(gui,stage,ui_config)
        errors=validate_labels_file(stage/'labels_agent.json',ui_config)
        if errors:raise ValueError(errors)
        loaded=build_note_alignment(stage)
        assert len(loaded['transcribed_notes'])==len(gui['transcribed_notes'])
        assert len(gui['note_mapping'])==len(gui['transcribed_notes'])
        manifest['results'].append({'sample':sample.name,'status':doc['status'],'notes':len(gui['transcribed_notes']),
            'labels':len(labels['labels']),'merged':info['same_pitch_repair']['merged_boundaries'],
            'file_sha256':{f:sha(stage/f) for f in FILES}})
        if pos%5==0 or pos==len(samples):
            write(out/'manifest.json',manifest);print('staged',pos,len(samples),sample.name,flush=True)
    for sample in samples:
        if sample.name in external_contexts:
            if context_signature(sample) != external_contexts[sample.name]:
                raise RuntimeError('MuScriptor source context changed during alignment: ' + sample.name)
    # Check for concurrent human edits or input changes before any replacement.
    for sample in samples:
        for f,digest in protected[sample.name].items():assert sha(sample/f)==digest,f'Input changed: {sample/f}'
        for f,digest in backup[sample.name].items():assert (sha(sample/f) if (sample/f).exists() else None)==digest,f'UI file changed: {sample/f}'
    manifest['status']='installing';write(out/'manifest.json',manifest)
    for sample in samples:
        for name in FILES:
            target=sample/name;temporary=sample/(name+'.v9-install.tmp')
            shutil.copyfile(out/'staged'/sample.name/name,temporary);os.replace(temporary,target)
    for row in manifest['results']:
        sample=samples_root/row['sample']
        for f,digest in row['file_sha256'].items():assert sha(sample/f)==digest
        for f,digest in protected[sample.name].items():assert sha(sample/f)==digest
    manifest['status']='complete';manifest['installed_utc']=datetime.now(timezone.utc).isoformat()
    write(out/'manifest.json',manifest)
    print('INSTALLED',len(samples),'samples; backup',out/'backup',flush=True)


if __name__=='__main__':main()
