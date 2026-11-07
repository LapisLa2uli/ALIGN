"""Shared paths and immutable input bookkeeping for the real-data learning curve."""
import hashlib
import json
from pathlib import Path

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OLD=ROOT/'reports/muscriptor_datacreate_001_050_20261105'
TRAIN_POOL=[f'{i:03}' for i in range(1,66)]
TEST=[f'{i:03}' for i in range(100,121)]
SIZES=[0,10,20,30,40,50]
SEED=20261105

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def checkpoints():
    base=ROOT/'experiments/baselines_datacreate_095_124_20260921/checkpoints'
    return {'polytune':base/'polytune/CocoChorales-E/last.ckpt',
            'laddersym':base/'laddersym/checkpoints/cocochorales/prompted/model.ckpt'}

def verify_inputs():
    manifest=read(RUN/'inputs.json')
    for sid,row in manifest['samples'].items():
        for file,digest in row['hashes'].items():
            assert sha(RUN/'bundles'/sid/file)==digest,(sid,file)
    return manifest
