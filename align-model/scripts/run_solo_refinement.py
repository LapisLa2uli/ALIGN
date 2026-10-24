"""Finish chromatic coverage refinement and compare candidates on all validation clips."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'align-model/runs/solo-v1'
CACHE=Path('E:/ALIGN_Data/solo_multi_cache_v1')
SUPPLEMENT=Path('E:/ALIGN_Data/solo_chromatic_cache_v1')


def status(stage, state='running', **extra):
    (RUN/'refinement_status.json').write_text(json.dumps({'stage':stage,'status':state,**extra},indent=2))


def execute(stage, arguments):
    status(stage);print('START',stage,flush=True)
    with (RUN/f'{stage}.log').open('w') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/'align-model/scripts/solo_experiment.py'),*map(str,arguments)],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('COMPLETE',stage,flush=True)


def main():
    status('waiting_for_training_and_coverage')
    deadline=time.monotonic()+4*3600
    while True:
        current=json.loads((RUN/'campaign_status.json').read_text())
        if current['status']=='failed':raise RuntimeError(current)
        if current['status']=='ready' and (SUPPLEMENT/'train.json').exists():break
        if time.monotonic()>deadline:raise TimeoutError('Prerequisite jobs did not finish')
        time.sleep(5)
    quality=json.loads((RUN/'coverage-audio/quality.json').read_text())
    assert not quality['full_corpus_check']['failures']
    execute('chromatic_refinement',['train','--cache',CACHE,'--supplement-cache',SUPPLEMENT,
        '--output',RUN/'chromatic','--checkpoint',RUN/'augmented/best.pt',
        '--epochs',2,'--lr',.00007,'--limit',20,'--augment','--distill','--rehearsal-weight',4])
    for name,checkpoint in [('baseline',ROOT/'align-model/runs/precision-v4/v5-dclike/best.pt'),
                            ('transfer',RUN/'transfer-trial/best.pt'),('augmented',RUN/'augmented/best.pt'),
                            ('chromatic',RUN/'chromatic/best.pt')]:
        execute('validation_'+name,['eval','--cache',CACHE,'--checkpoint',checkpoint,
            '--output',RUN/f'validation_{name}_full.json','--limit',72])
    status('validation_review','ready');print('VALIDATION_READY_TEST_UNOPENED',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed','failed',error=str(error));raise
