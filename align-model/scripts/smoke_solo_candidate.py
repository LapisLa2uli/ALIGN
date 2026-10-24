"""Exercise the frozen solo candidate through CLI, UI bridge, and read routes."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'DataCreate/src'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample',type=Path,required=True,help='Source inputs; always copied before inference')
    parser.add_argument('--output',type=Path,required=True,help='New scratch directory')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sample=out/'sample';sample.mkdir()
    protected={}
    for name in ['performance_audio.wav','verified_score.musicxml','labels.json','metadata.json']:
        source=args.sample/name
        if source.exists():
            shutil.copy2(source,sample/name);protected[name]=sha(source)
    candidate=ROOT/'align-model/runs/solo-v1/CANDIDATE_SOLO_V1.json'
    manifest=json.loads(candidate.read_text())
    subprocess.run([sys.executable,str(ROOT/'align-model/scripts/run_solo_transcriber.py'),
                    '--audio',str(sample/'performance_audio.wav'),'--score',str(sample/'verified_score.musicxml'),
                    '--candidate',str(candidate),'--output',str(out/'standalone.json'),'--device',args.device],check=True,cwd=ROOT)
    from datacreate.config import PipelineConfig
    from datacreate.align_bridge import dump_transcription,run_preferred_alignment,current_v9_feedback
    config=PipelineConfig.load(ROOT/'DataCreate/config/solo.yaml')
    config.paths.update(samples_root=str(out),work_dir=str(out/'work'),note_alignment_candidate=str(candidate))
    config.alignment['note_alignment_device']=args.device
    logger=logging.getLogger('solo-smoke')
    dump_transcription(sample,config,logger,output=out/'locator_transcription.json')
    run_preferred_alignment(sample/'performance_audio.wav',sample/'reference_audio.wav',sample,config,logger)
    assert current_v9_feedback(sample,config)
    standalone=json.loads((out/'standalone.json').read_text())
    locator=json.loads((out/'locator_transcription.json').read_text())
    gui=json.loads((sample/'note_alignment_v2.json').read_text())
    assert standalone['checkpoint_sha256']==locator['checkpoint_sha256']==manifest['checkpoint_sha256']
    assert gui['provenance']['checkpoint_sha256']==manifest['checkpoint_sha256']
    assert standalone['notes']==locator['notes']
    assert standalone['feedback']['labels']==json.loads((sample/'feedback_solo.json').read_text())['labels']
    from fastapi.testclient import TestClient
    from datacreate.web.app import create_app
    with TestClient(create_app(config)) as client:
        statuses={url:client.get(url).status_code for url in ['/', '/api/samples/sample/note-alignment']}
    assert set(statuses.values())=={200}
    assert 'torch' not in sys.modules, 'UI process must not load the ML runtime'
    for name,digest in protected.items():
        assert sha(sample/name)==sha(args.sample/name)==digest
    result={'status':'passed','http':statuses,'ui_imports_torch':False,
            'checkpoint_sha256':manifest['checkpoint_sha256'],'notes':len(standalone['notes']),
            'protected_inputs_unchanged':protected,'routes':['CLI with score','transcription locator','alignment publisher','UI read']}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
