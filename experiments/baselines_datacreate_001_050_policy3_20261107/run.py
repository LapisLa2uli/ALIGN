"""Unchanged PolyTune/LadderSym on DataCreate 001-050, policy 3, no rhythm scoring."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PREVIOUS = ROOT / 'experiments/baselines_real_teacher_20261105'
LATEST = ROOT / 'experiments/baselines_real_teacher_v2_20261105'
IDS = [f'{i:03}' for i in range(1, 51)]
KINDS = ['wrong_note', 'missed_note', 'extra_note', 'repetition']
sys.path[:0] = [str(LATEST), str(ROOT/'baselines/scripts'), str(ROOT/'baselines/common'), str(ROOT/'DataCreate/src')]
from common import read, write, sha


def relocate(path):
    path = Path(path)
    if path.exists():
        return path
    name = str(path).replace('\\', '/')
    return ROOT / 'experiments' / name.split('/experiments/', 1)[1]


def prepare():
    from label_datacreate_baselines_v3 import load_lookup
    from datacreate.melody import canonical_note_location
    from prepare_dataset import load_audio_16k, write_wav16
    import numpy as np
    import engine
    from reference_alignment import align
    import soundfile as sf
    if (RUN/'protocol.json').exists():
        print('Frozen protocol already exists', flush=True)
        return
    previous_records = read(PREVIOUS/'data/manifest.json')['tracks']
    checkpoints = read(LATEST/'protocol.json')['initial_checkpoints']
    for ck in checkpoints.values():
        ck['path'] = str(relocate(ck['path']))
        assert sha(ck['path']) == ck['sha256']
    protected, tracks, samples = {}, {}, {}
    filenames = ['performance_audio.wav', 'verified_score.musicxml', 'reference_audio.wav',
                 'reference_audio.mid', 'labels_agent.json', 'note_alignment_v2.json']
    for sid in IDS:
        source = ROOT/'DataCreate/samples'/sid
        bundle = RUN/'bundles'/sid
        bundle.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name in filenames:
            path = source/name
            digest = sha(path)
            shutil.copy2(path, bundle/name)
            assert sha(bundle/name) == digest
            hashes[name] = digest
            protected[str(path)] = digest
        protected[str(source/'labels.json')] = sha(source/'labels.json')
        lookup, score, metadata = load_lookup(bundle/'verified_score.musicxml', bundle/'performance_audio.wav',
                                             bundle/'note_alignment_v2.json', sample_id=sid)
        alignment = read(bundle/'note_alignment_v2.json')
        agent = read(bundle/'labels_agent.json')
        prov = agent['agent_labeling']
        assert not prov.get('edited_in_gui') and prov['transcriber'] == 'muscriptor'
        assert prov['score_sha256'] == hashes['verified_score.musicxml']
        assert prov['audio_sha256'] == hashes['performance_audio.wav']
        assert agent['labels'] == alignment['labels']
        assert alignment['provenance']['transcription']['written_shift'] == 2
        labels = [dict(l) for l in agent['labels'] if l['type'] in KINDS]
        for label in labels:
            if label['type'] == 'repetition' and label.get('extra_copies') is None:
                label['extra_copies'] = 1
            assert canonical_note_location(label, score_event_count=len(score)) is not None, (sid, label)
        status = alignment['diagnostics']['status']
        write(RUN/'teacher_gold'/f'{sid}.json', dict(sample=sid, status=status,
            labels=labels, score_event_count=len(score), ignored_types=dict(Counter(l['type'] for l in agent['labels'] if l['type'] not in KINDS)),
            source_sha256=hashes['labels_agent.json'], alignment_sha256=hashes['note_alignment_v2.json']))
        samples[sid] = dict(status=status, hashes=hashes, score_event_count=len(score),
                            content_labels=len(labels), teacher_revision=prov['pipeline_revision'],
                            ignored_rhythm=sum(l['type']=='rhythm_error' for l in agent['labels']))
        reusable = sid in previous_records and all(sha(PREVIOUS/'bundles'/sid/name) == hashes[name]
            for name in filenames[:4])
        paths = {key: RUN/'data'/sid/name for key, name in
                 [('mistake_wav','performance_16k.wav'), ('score_wav','reference_16k.wav'), ('score_mid','reference.mid')]}
        if reusable:
            record = previous_records[sid]
            for key, path in paths.items():
                original = relocate(record['paths'][key])
                assert sha(original) == record['hashes'][key]
                protected[str(original)] = sha(original)
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, path)
            duration = record['duration']
        else:
            with engine.no_gold():
                perf = load_audio_16k(bundle/'performance_audio.wav')
                ref = load_audio_16k(bundle/'reference_audio.wav')
                duration = len(perf)/16000
                length = max(len(perf),len(ref))
                write_wav16(paths['mistake_wav'], np.pad(perf,(0,length-len(perf))))
                write_wav16(paths['score_wav'], np.pad(ref,(0,length-len(ref))))
                shutil.copy2(bundle/'reference_audio.mid', paths['score_mid'])
        input_hashes = {key: sha(path) for key,path in paths.items()}
        tracks[sid] = dict(duration=duration, paths={k:str(v) for k,v in paths.items()},
                           hashes=input_hashes, reusable_previous_inputs=reusable)
        old_map = LATEST/'alignment'/f'{sid}.json'
        old = read(old_map) if reusable and old_map.exists() else None
        mapping_hashes = {k:input_hashes[k] for k in ['mistake_wav','score_wav']}
        if old and old['input_sha256'] == mapping_hashes:
            mapping = old
            protected[str(old_map)] = sha(old_map)
        else:
            with engine.no_gold():
                perf,sr = sf.read(paths['mistake_wav'], dtype='float32')
                ref,rr = sf.read(paths['score_wav'], dtype='float32')
                assert sr == rr == 16000
                mapping = align(perf[:round(duration*sr)],ref,sr)
                mapping.update(sample=sid,input_sha256=mapping_hashes,labels_read=False)
        write(RUN/'alignment'/f'{sid}.json',mapping)
        print('PREPARED',sid,status,'content',len(labels),'rhythm excluded',samples[sid]['ignored_rhythm'],flush=True)
    sources = [Path(__file__), LATEST/'engine.py', LATEST/'features.py', LATEST/'reference_alignment.py',
               LATEST/'cached_decode.py', PREVIOUS/'run_model.py', ROOT/'baselines/scripts/label_datacreate_baselines_v3.py',
               ROOT/'baselines/common/transcription_label_alignment.py']
    write(RUN/'data/manifest.json',dict(tracks=tracks))
    write(RUN/'protocol.json',dict(schema='baseline-001-050-policy3-v1', models=list(checkpoints),
        initial_checkpoints=checkpoints, split={'test':IDS}, training_performed=False, alignment_policy=3,
        reference='Current stored AudioEval+MuScriptor pseudo-gold, frozen at preparation',
        shared_teacher_alignment=True, ignored_types=['rhythm_error'], content_types=KINDS,
        metric='Exact reference-score note set and error type, repetition copy count; unlocated predictions are false positives',
        score_samples=[s for s in IDS if samples[s]['status']=='ok'],
        abstentions=[s for s in IDS if samples[s]['status']!='ok'], samples=samples,
        protected_sha256=protected, source_sha256={str(p):sha(p) for p in sources},
        native_inference='Corrected v2 audio-only reference retrieval, deterministic decoding, 1024-token budget; verified PolyTune cached decoding',
        all_records_are_evaluation_only_for_unchanged_models=True))


def cache(flavor):
    import torch
    import soundfile as sf
    import note_seq
    import engine
    from features import Features
    torch.set_num_threads(3)
    features = Features(flavor)
    records = read(RUN/'data/manifest.json')['tracks']
    with engine.no_gold():
        for sid in IDS:
            path = RUN/'cache'/flavor/'infer'/f'{sid}.pt'
            if path.exists() and path.with_suffix('.json').exists():
                assert sha(path) == read(path.with_suffix('.json'))['sha256']
                continue
            rec = records[sid]
            old = LATEST/'cache'/flavor/'infer'/f'{sid}.pt'
            mapping_path = RUN/'alignment'/f'{sid}.json'
            can_reuse = rec['reusable_previous_inputs'] and old.exists()
            if can_reuse:
                meta = read(old.with_suffix('.json'))
                can_reuse = sha(mapping_path) == meta['alignment_sha256']
            path.parent.mkdir(parents=True,exist_ok=True)
            if can_reuse:
                assert sha(old) == meta['sha256']
                shutil.copy2(old,path)
                meta = {**meta,'reused_from':str(old)}
            else:
                perf,sr = sf.read(rec['paths']['mistake_wav'],dtype='float32')
                ref,rr = sf.read(rec['paths']['score_wav'],dtype='float32')
                assert sr == rr == 16000
                perf = perf[:round(rec['duration']*sr)]
                ns = note_seq.midi_file_to_note_sequence(rec['paths']['score_mid'])
                mapping = read(mapping_path)
                bounds = [[i*2.048,(i+1)*2.048] for i in range(math.ceil(rec['duration']/2.048))]
                entries,starts = [],[]
                for a,b in bounds:
                    entry,start = features.inputs(perf,ref,ns,mapping,a,b)
                    entries.append(entry);starts.append(start)
                torch.save(entries,path)
                meta = dict(sample=sid,mode='infer',windows=len(entries),bounds=bounds,reference_starts=starts,
                    sha256=sha(path),alignment_sha256=sha(mapping_path),labels_read=False)
            meta['input_hashes'] = rec['hashes']
            write(path.with_suffix('.json'),meta)
            old_pred = LATEST/'predictions/validation'/flavor/'n00/step000'/sid
            if can_reuse and (old_pred/'mix.done.json').exists():
                done = read(old_pred/'mix.done.json')
                ck = read(RUN/'protocol.json')['initial_checkpoints'][flavor]
                assert done['checkpoint_sha256'] == ck['sha256'] and sha(old_pred/'mix.mid') == done['sha256']
                dest = RUN/'predictions/test'/flavor/'n00/step000'/sid
                dest.mkdir(parents=True,exist_ok=True)
                shutil.copy2(old_pred/'mix.mid',dest/'mix.mid')
                write(dest/'mix.done.json',{**done,'reused_from':str(old_pred),'reused_source_done_sha256':sha(old_pred/'mix.done.json')})
            print('FEATURES',flavor,sid,meta['windows'],'reused',can_reuse,flush=True)


def infer(flavor):
    import torch
    import engine
    from cached_decode import install
    torch.set_num_threads(3)
    spec = importlib.util.spec_from_file_location('original_model_builder',PREVIOUS/'run_model.py')
    builder = importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    def build(model,path):
        result = builder.build(model,path)
        return install(result) if model=='polytune' else result
    engine.RUN=RUN
    engine.PREVIOUS=RUN
    engine.build=build
    engine.infer(flavor,0,0,'test',read(RUN/'protocol.json'))


def export():
    for model in ['polytune','laddersym']:
        out=RUN/'exports'/model
        if (out/f'{model}_label_manifest.json').exists():
            continue
        ck=read(RUN/'protocol.json')['initial_checkpoints'][model]
        subprocess.run([sys.executable,str(ROOT/'baselines/scripts/label_datacreate_baselines.py'),
            '--model',model,'--pred-dir',str(RUN/'predictions/test'/model/'n00/step000'),
            '--samples',str(RUN/'bundles'),'--labels-root',str(RUN/'labels'),'--output',str(out),
            '--checkpoint-sha256',ck['sha256']],check=True)


def main():
    os.environ.update(PYTHONNOUSERSITE='1',LADDERSYM_DETERMINISTIC_PROMPT='1',LADDERSYM_PROMPT_LENGTH='1024',LADDERSYM_USE_CACHE='1')
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','cache','infer','export','all'])
    parser.add_argument('--model',choices=['polytune','laddersym']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='cache':cache(args.model)
    elif args.action=='infer':infer(args.model)
    elif args.action=='export':export()
    else:
        jobs=[['prepare']]+[[a,'--model',m] for m in ['polytune','laddersym'] for a in ['cache','infer']]+[['export']]
        for job in jobs:
            write(RUN/'progress.json',dict(stage=job,status='running',time=time.time()))
            print('STAGE',job,flush=True)
            subprocess.run([sys.executable,str(Path(__file__)),*job],cwd=ROOT,check=True)
        write(RUN/'progress.json',dict(stage='inference_and_export_complete',status='complete',time=time.time()))


if __name__=='__main__':main()
