"""Run the reproducible multi-instrument experiment stages, retaining every log."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'align-model/runs/solo-v1'
DATA=Path('E:/ALIGN_Data/solo_multinstrument_v1')
CACHE=Path('E:/ALIGN_Data/solo_multi_cache_v1')


def run(name,arguments):
    status={'stage':name,'status':'running','started_unix':time.time()}
    (RUN/'campaign_status.json').write_text(json.dumps(status,indent=2))
    print('START',name,flush=True)
    with (RUN/f'{name}.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-u',*map(str,arguments)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('COMPLETE',name,flush=True)


def main():
    stages=['quality','prepare','baseline','transfer_trial','augmented_training']
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-at',choices=stages,default='quality')
    args=parser.parse_args()
    def execute(name,arguments):
        if stages.index(name)>=stages.index(args.start_at):
            run(name,arguments)
    RUN.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+4*3600
    while not (DATA/'manifest.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Corpus generation did not finish')
        time.sleep(5)
    script=ROOT/'align-model/scripts/solo_experiment.py'
    execute('quality',[ROOT/'synth-pipeline/scripts/audit_solo_audio.py','--data',DATA,'--output',RUN/'audio-review','--per-instrument',8,'--all-silence'])
    execute('prepare',[script,'prepare','--data',DATA,'--output',CACHE,'--rehearsal'])
    execute('baseline',[script,'eval','--cache',CACHE,'--output',RUN/'baseline_validation.json','--limit',20])
    execute('transfer_trial',[script,'train','--cache',CACHE,'--output',RUN/'transfer-trial','--epochs',2,'--train-limit',96,'--limit',20])
    execute('augmented_training',[script,'train','--cache',CACHE,'--output',RUN/'augmented',
        '--checkpoint',RUN/'transfer-trial/best.pt','--epochs',6,'--lr',.0001,'--limit',20,'--augment','--distill'])
    # Review validation results and freeze the selection before opening test.
    (RUN/'campaign_status.json').write_text(json.dumps({'stage':'validation_review','status':'ready','finished_unix':time.time()},indent=2))
    print('VALIDATION_READY_TEST_UNOPENED',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        (RUN/'campaign_status.json').write_text(json.dumps({'status':'failed','error':str(error)},indent=2))
        raise
