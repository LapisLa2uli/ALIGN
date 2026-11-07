"""Check exact audio duplicates and composition overlap without target labels."""
from common import RUN,PREVIOUS,read,write,split

def main():
    p=split();inputs=read(PREVIOUS/'inputs.json')['samples'];pairs={}
    for left,right in [('train','validation'),('train','test'),('validation','test')]:
        audio=[];scores=[];sources=[]
        for a in p[left]:
            for b in p[right]:
                x,y=inputs[a],inputs[b]
                if x['hashes']['performance_audio.wav']==y['hashes']['performance_audio.wav']:audio.append([a,b])
                if x['hashes']['verified_score.musicxml']==y['hashes']['verified_score.musicxml']:scores.append([a,b])
                if x['source_filename'] and x['source_filename']==y['source_filename']:sources.append([a,b])
        assert not audio and not sources,(left,right,audio,sources)
        pairs[left+'/'+right]={'exact_audio_duplicates':audio,'same_source_filename':sources,'identical_score_xml':scores}
    write(RUN/'split_audit.json',{'pairs':pairs,'limitation':'Source filenames are absent for many older recordings; absence of overlap cannot rule out excerpts of one original take.'})
    print(pairs,flush=True)

if __name__=='__main__':main()
