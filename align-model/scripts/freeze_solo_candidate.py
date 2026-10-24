"""Freeze a reviewed validation candidate before either held-out test evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'align-model/runs/solo-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['transfer','augmented','chromatic','calibrated'],required=True)
    parser.add_argument('--reason',required=True,help='Record the validation review and any material regression')
    args=parser.parse_args()
    if any((RUN/name).exists() for name in ['baseline_test.json','candidate_test.json','CANDIDATE_SOLO_V1.json']):
        raise RuntimeError('The test split has been evaluated; freeze a new experiment version for further selection')
    comparison={}
    for name in ['baseline','transfer','augmented','chromatic','calibrated']:
        path=RUN/f'validation_{name}_full.json'
        if not path.exists() and name=='calibrated':continue
        report=json.loads(path.read_text())
        checkpoint=Path(report['checkpoint'])
        if sha(checkpoint)!=report['checkpoint_sha256']:
            raise ValueError(f'Checkpoint changed since validation: {name}')
        comparison[name]={'checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':sha(checkpoint),
                          'validation_sha256':sha(path),'macro_instrument_f1':report['macro_instrument_f1'],
                          'legacy_clarinet':report['instruments']['legacy_clarinet'],'styles':report['styles'],
                          'decoder':{'blank_scale':report['blank_scale'],'candidate_threshold':.08},
                          'musical_macro_f1':sum(v['f1'] for k,v in report['instruments'].items() if k!='legacy_clarinet')/24}
    selected=comparison[args.variant]
    range_results={}
    for name in ['baseline','augmented','chromatic','calibrated']:
        path=RUN/f'validation_range_{name}.json'
        if path.exists():
            report=json.loads(path.read_text())
            if report['checkpoint_sha256']!=comparison[name]['checkpoint_sha256']:
                raise ValueError(f'Range validation checkpoint mismatch: {name}')
            range_results[name]={'macro_f1':report['macro_instrument_f1'],'overall':report['overall'],
                                 'report_sha256':sha(path)}
    base=ROOT/'align-model/runs/precision-v4/CANDIDATE_STACK_V6.json'
    files=list((ROOT/'align-model/src/alignmodel/transcription').glob('*.py'))
    files+=list((ROOT/'align-model/src/alignmodel/joint').glob('*.py'))
    files += [ROOT/name for name in [
        'align-model/scripts/run_solo_transcriber.py','align-model/scripts/publish_datacreate_solo.py',
        'DataCreate/src/datacreate/score_transposition.py','DataCreate/src/datacreate/align_bridge.py',
        'DataCreate/src/datacreate/feedback_visibility.py','DataCreate/src/datacreate/melody.py',
        'DataCreate/src/datacreate/models.py','DataCreate/src/datacreate/validation.py']]
    frozen_checkpoint=RUN/'solo-transcriber.pt'
    if frozen_checkpoint.exists():raise FileExistsError(frozen_checkpoint)
    shutil.copy2(selected['checkpoint'],frozen_checkpoint)
    assert sha(frozen_checkpoint)==selected['checkpoint_sha256']
    manifest={'schema_version':'align-solo-candidate-v1','pipeline_revision':'solo-multi-v1',
        'frozen_utc':datetime.now(timezone.utc).isoformat(),'variant':args.variant,
        'checkpoint':str(frozen_checkpoint.resolve()),'checkpoint_sha256':selected['checkpoint_sha256'],
        'base_candidate':str(base.resolve()),'base_candidate_sha256':sha(base),
        'decoder':selected['decoder'],'pitch_space':'sounding','midi_range':[28,108],
        'instrument_conditioning':False,'promoted':False,
        'scope':'Opt-in shared solo acoustic model; inherited error gates remain experimental outside clarinet',
        'selection':{'reason':args.reason,'comparison':comparison,'range_validation':range_results,
                     'test_unopened_at_selection':True},
        'code_sha256':{path.relative_to(ROOT).as_posix():sha(path) for path in sorted(set(files))}}
    path=RUN/'CANDIDATE_SOLO_V1.json'
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copyfile(RUN/f'validation_{args.variant}_full.json',RUN/'validation_candidate_full.json')
    print(json.dumps({'candidate':str(path),'sha256':sha(path),'selected':args.variant},indent=2))


if __name__=='__main__':main()
