"""Audit retrieval against training/validation identities, never test labels."""
import sys
from common import ROOT, RUN, PREVIOUS, read, write, split
from reference_alignment import context_start

def main():
    import pretty_midi
    sys.path.insert(0, str(ROOT/'DataCreate/src'))
    from datacreate.melody import parse_sounding_notes
    records=read(PREVIOUS/'data/manifest.json')['tracks']; parts=split(); rows={}; rejected=[]
    for sid in parts['train']+parts['validation']:
        score=parse_sounding_notes(PREVIOUS/'bundles'/sid/'verified_score.musicxml')
        midi=pretty_midi.PrettyMIDI(records[sid]['paths']['score_mid'])
        ref=sorted([n for i in midi.instruments for n in i.notes],key=lambda n:(n.start,n.pitch))
        if [n.pitch for n in ref]!=[n.pitch-2 for n in score]:
            rejected.append(sid);continue
        teacher=read(PREVIOUS/'bundles'/sid/'note_alignment_v2.json')['transcribed_notes']
        targets=read(PREVIOUS/'supervision'/f'{sid}.json')['classes']['correct']
        mapping=read(RUN/'alignment'/f'{sid}.json'); assessed=old=new=0
        for a,b in read(PREVIOUS/'cache/polytune'/f'{sid}.json')['bounds']:
            c=context_start(mapping,a,b)
            for n in targets:
                if n['start']>=b or n['end']<=a:continue
                span=teacher[n['source_note']].get('score_span')
                if not span or span[1]!=span[0]+1:continue
                r=ref[span[0]];assessed+=1
                old+=int(r.end<=a-1.024-.1 or r.start>=b+1.024+.1)
                new+=int(r.end<=c-.1 or r.start>=c+4.096+.1)
        rows[sid]={'assessed':assessed,'old_absent':old,'new_absent':new}
    summary={}
    for name,ids in {**parts['subsets'],'validation':parts['validation']}.items():
        totals={k:sum(rows[s][k] for s in ids if s in rows) for k in ['assessed','old_absent','new_absent']}
        summary[name]={**totals,'old_absent_fraction':totals['old_absent']/max(1,totals['assessed']),
                      'new_absent_fraction':totals['new_absent']/max(1,totals['assessed'])}
    write(RUN/'context_audit.json',{'summary':summary,'recordings':rows,'ambiguous_excluded':rejected,'test_labels_read':False})
    print(summary,flush=True)

if __name__=='__main__':main()
