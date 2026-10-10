"""Replay archived acoustic controls with explicit score-support conditions.

These isolate the repair rule, not fresh transcription or end-to-end accuracy.
Also expose the hard case: a false split aligned across two repeated score notes.
"""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'align-model/src'))
from alignmodel.transcription.same_pitch_v3 import boundary_features,repeat_score


def main():
    out=ROOT/'align-model/runs/stack-v10'
    rows=json.loads((ROOT/'align-model/runs/stack-v9/control_features.json').read_text())
    model=json.loads((out/'boundary_model.json').read_text())
    config=json.loads((out/'CANDIDATE_STACK_V10.json').read_text())
    v9=config['same_pitch'];threshold=config['boundary']['repeat_threshold']
    summary={}
    for kind in ('genuine_repeat','injected_split'):
        cases=[r for r in rows if r['kind']==kind]
        before=after=adverse=0
        for row in cases:
            e=row.get('evidence');f=boundary_features(e)
            merged=bool(e and e['gap']['depth_db']<v9['gap_depth_db'] and e['energy_range_db']<=v9['energy_range_db']
                and e['spectral_step']<=v9['spectral_step_max'] and e['spectral_drift']<=v9['spectral_drift_max']
                and e['attack_max']<=v9['merge_attack_max'] and e['voiced_min']>=v9['voiced_min'] and e['energy_min_db']>=v9['silence_db'])
            vote=f is not None and repeat_score(f,model)>=threshold
            before+=merged
            # Normal control: actual repeat has two score notes; split has one.
            after+=merged and not(kind=='genuine_repeat' and vote)
            adverse+=merged and vote
        summary[kind]={'count':len(cases),'v9_merged':before,'v10_merged_with_expected_score_support':after,
            'restoration_votes_if_score_maps_to_two_notes':adverse}
    doc={'scope':__doc__,'summary':summary,'raw_audio_available':False,
         'independent_test':False,'warning':'Score-support cases are controlled assumptions. Full realignment and wrong score-span cases require end-to-end evaluation.'}
    (out/'control_replay.json').write_text(json.dumps(doc,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
