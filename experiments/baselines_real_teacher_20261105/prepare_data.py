"""Build audited teacher targets and a fixed nested sample-count experiment."""
import random
import shutil
import sys
from common import ROOT,RUN,TRAIN_POOL,TEST,SIZES,SEED,read,write,sha,verify_inputs
sys.path[:0]=[str(ROOT/'DataCreate/src'),str(ROOT/'baselines/common')]
from supervision import targets,canonical

def main():
    import numpy as np
    import soundfile as sf
    from datacreate.melody import parse_sounding_notes
    from prepare_dataset import load_audio_16k,write_wav16,write_label_midi,output_paths
    manifest=verify_inputs();records={};rejected={}
    for sid in TRAIN_POOL+TEST:
        sample=RUN/'bundles'/sid;gui=read(sample/'note_alignment_v2.json')
        p=gui['provenance'];h=manifest['samples'][sid]['hashes']
        assert p['transcriber']=='muscriptor' and p['audio_sha256']==h['performance_audio.wav'] and p['score_sha256']==h['verified_score.musicxml']
        pitches=[n.pitch for n in parse_sounding_notes(sample/'verified_score.musicxml')]
        shift=p['transcription']['written_shift'];assert shift==2,'Native evaluation adapter assumes Bb clarinet'
        labels=canonical(gui['labels'])
        write(RUN/'teacher_gold'/f'{sid}.json',{'sample':sid,'reference_kind':'AudioEval+MuScriptor pseudo-gold',
              'status':gui['diagnostics']['status'],'score_event_count':len(pitches),'labels':labels,
              'provenance':p,'published_labels_sha256':sha(sample/'labels_agent.json')})
        if sid in TRAIN_POOL:
            target=targets(gui,pitches,shift)
            write(RUN/'supervision'/f'{sid}.json',target)
            if not target['usable']:
                rejected[sid]=target['reason'];continue
        else:target=None
        paths=output_paths(RUN/'data',sid)
        perf=load_audio_16k(sample/'performance_audio.wav');ref=load_audio_16k(sample/'reference_audio.wav')
        length=max(len(perf),len(ref));duration=len(perf)/16000
        write_wav16(paths['mistake_wav'],np.pad(perf,(0,length-len(perf))))
        write_wav16(paths['score_wav'],np.pad(ref,(0,length-len(ref))))
        shutil.copy2(sample/'reference_audio.mid',paths['score_mid'])
        if target:
            for kind,notes in target['classes'].items():write_label_midi(paths[kind],notes)
        records[sid]={'duration':duration,'padded_duration':length/16000,'paths':{k:str(v) for k,v in paths.items()},
                      'hashes':{k:sha(v) for k,v in paths.items() if v.exists()},
                      'teacher_sha256':sha(sample/'note_alignment_v2.json')}
        print('PREPARED',sid,target['class_counts'] if target else 'test inputs only',flush=True)
    eligible=sorted(set(records)-set(TEST))
    if len(eligible)<50:raise ValueError(f'Only {len(eligible)} usable teacher examples: {rejected}')
    random.Random(SEED).shuffle(eligible)
    write(RUN/'data/manifest.json',{'tracks':records,'rejected':rejected})
    # Final ordering can only drop examples with no fully assessed windows during caching.
    write(RUN/'eligible.json',{'ordered_ids':eligible,'rejected':rejected})
    for size in SIZES[1:]:
        ids=eligible[:size]
        write(RUN/'data'/f'split-{size}.json',{'midi_filename':{str(i):sid+'.midi' for i,sid in enumerate(ids)},
              'split':{str(i):'train' for i in range(len(ids))}})
    print('PREPARED_DATA',len(eligible),'eligible training cases;',len(TEST),'test cases',flush=True)

if __name__=='__main__':main()
