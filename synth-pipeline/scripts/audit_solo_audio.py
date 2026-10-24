"""Check rendered solo audio against MIDI, and create local audition montages."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks


def silent_note_indices(audio, sr, notes):
    result = []
    for i, n in enumerate(notes):
        clip = audio[int((n['start']+.025)*sr):int(min(n['end'], n['start']+.22)*sr)]
        if len(clip) >= 128 and float(np.sqrt(np.mean(clip**2))) < 1e-5:
            result.append(i)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--per-instrument',type=int,default=3)
    p.add_argument('--all-silence',action='store_true',help='Validate audibility/clipping on every clip, independent of model output')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.data/'manifest.json').read_text())
    rng=random.Random(20261017);checks=[];montage=[];timeline=[];offset=0
    for inst in sorted({r['instrument'] for r in manifest['rows']}):
        choices=[r for r in manifest['rows'] if r['instrument']==inst]
        chosen=rng.sample(choices,min(args.per_instrument,len(choices)))
        for row in chosen:
            path=Path(row['path']);audio,sr=sf.read(path/'performance_audio.wav',dtype='float32')
            notes=json.loads((path/'notes.json').read_text())
            silent=0;pitch_checks=[]
            for n in notes:
                a=int((n['start']+.025)*sr);b=int(min(n['end'],n['start']+.22)*sr)
                clip=audio[a:b]
                if len(clip)<128:continue
                if np.sqrt(np.mean(clip**2)) < 1e-5:silent+=1
                if len(clip)<1000:continue
                # Check harmonic energy near known F0, without assuming the fundamental is strongest.
                fft=np.abs(np.fft.rfft(clip*np.hanning(len(clip)),n=32768))
                hz=np.fft.rfftfreq(32768,1/sr);f0=440*2**((n['pitch']-69)/12)
                peaks,_=find_peaks(fft)
                strongest=sorted(peaks,key=lambda i:fft[i],reverse=True)[:12]
                cents=min((abs(1200*np.log2(hz[i]/(f0*h))) for i in strongest if hz[i]>0 for h in range(1,6)),default=999)
                pitch_checks.append(cents)
            checks.append({'id':row['id'],'instrument':inst,'bank':row['bank'],'silent_note_windows':silent,
                           'notes_checked':len(notes),'peak':float(np.max(np.abs(audio))),
                           'finite':bool(np.isfinite(audio).all()),'clipped_fraction':float(np.mean(np.abs(audio)>.999)),
                           'harmonic_check_median_cents':float(np.median(pitch_checks)) if pitch_checks else None})
        preview=Path(chosen[0]['path'])/'performance_audio.wav'
        audio,sr=sf.read(preview,dtype='float32');audio=audio[:sr*7]
        audio=audio*.8/max(.001,float(np.max(np.abs(audio))))
        sf.write(args.output/f'{inst}.wav',audio,sr)
        timeline.append({'instrument':inst,'sample':chosen[0]['id'],'start':offset,'end':offset+len(audio)/sr})
        silence=np.zeros(int(sr*.6),np.float32)
        montage.extend([audio,silence]);offset+=(len(audio)+len(silence))/sr
    sf.write(args.output/'audition.wav',np.concatenate(montage),sr)
    report={'checks':checks,'audition_timeline':timeline,'listening_performed':False,
            'quality_method':'finite/clipping/silence checks and independent FFT harmonic diagnostic; not a perceptual listening claim'}
    if args.all_silence:
        failures=[]
        for i,row in enumerate(manifest['rows']):
            path=Path(row['path']);x,sr=sf.read(path/'performance_audio.wav',dtype='float32')
            notes=json.loads((path/'notes.json').read_text())
            silent=silent_note_indices(x,sr,notes)
            if silent or not np.isfinite(x).all() or np.max(np.abs(x))>.999:
                failures.append({'id':row['id'],'silent_note_indices':silent,
                                 'finite':bool(np.isfinite(x).all()),'peak':float(np.max(np.abs(x)))})
            if (i+1)%2000==0:print(f'full audio audit {i+1}/{len(manifest["rows"])}',flush=True)
        report['full_corpus_check']={'clips':len(manifest['rows']),'failures':failures}
    (args.output/'quality.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'clips':len(checks),'silent_note_windows':sum(r['silent_note_windows'] for r in checks),
                      'max_clipping':max(r['clipped_fraction'] for r in checks),
                      'max_median_harmonic_cents':max(r['harmonic_check_median_cents'] or 0 for r in checks)}))
    if args.all_silence and failures:
        raise ValueError(f'{len(failures)} audio clips failed QA; repair them before training')


if __name__=='__main__':main()
