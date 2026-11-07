"""Bound external notes to actual audio, excluding predictions in padded tails."""
from common import RUN,TRAIN_POOL,TEST,read,write

def main():
    import soundfile as sf
    inputs=read(RUN/'inputs.json');changes={}
    for sid in TRAIN_POOL+TEST:
        doc=read(RUN/'teacher_transcriptions'/f'{sid}.json')
        duration=sf.info(RUN/'bundles'/sid/'performance_audio.wav').duration
        notes=[];dropped=0;clipped=0
        for note in doc['transcribed_notes']:
            if note['start']>=duration or note['end']<=0 or note['end']<=note['start']:
                dropped+=1;continue
            value={**note,'start':max(0.,note['start']),'end':min(duration,note['end'])}
            clipped+=int(value['start']!=note['start'] or value['end']!=note['end']);notes.append(value)
        if dropped or clipped:
            changes[sid]={'dropped':dropped,'clipped':clipped,'duration':duration}
        if dropped or clipped or sid in inputs['fresh_teacher_ids']:
            doc['transcribed_notes']=notes
            doc['provenance']['audio_bounds_policy']={'rule':'Discard notes starting outside recording; clip offsets to audio end','dropped':dropped,'clipped':clipped,'duration':duration}
            write(RUN/'alignment_transcriptions'/f'{sid}.json',doc)
    write(RUN/'teacher_bounds_audit.json',changes)
    print('BOUNDS_AUDIT',changes,flush=True)

if __name__=='__main__':main()
