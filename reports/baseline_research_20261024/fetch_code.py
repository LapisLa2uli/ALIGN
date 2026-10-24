import requests,pathlib,concurrent.futures,json,subprocess
root=pathlib.Path('reports/baseline_research_20261024/sources')
jobs=[]
for repo,paths in {
'LadderSym':['models/ladder.py','models/ladder_backbone.py','models/laddersym_t5.py','models/t5_prompt.py','tasks/laddersym_prompt_mt3_net.py','config/config_maestro_prompted.yaml','config/config_coco_prompt.yaml','config/model/laddersym_prompt_MT3Net.yaml','dataset/dataset_2_random.py','dataset/dataset_2_random_coco.py'],
'Polytune':['models/polytune.py','models/ast.py','tasks/polytune_net.py','config/config_maestro.yaml','config/config_coco.yaml','config/model/polytune.yaml','contrib/spectrograms.py','baseline/MT3_baseline/README.md','baseline/MT3_baseline/evaluate_errors_old.py'],
'CocoChorales-E_MAESTRO-E':['create_maestro.py','create_cocochorales.py']}.items():
 for path in paths:jobs.append((repo+'/'+path,'https://raw.githubusercontent.com/ben2002chou/'+repo+'/main/'+path))
for name,url in {
'parangonar_README.md':'https://raw.githubusercontent.com/sildater/parangonar/main/README.md',
'matchmaker_README.md':'https://raw.githubusercontent.com/pymatchmaker/matchmaker/develop/README.md',
'basic_pitch.pdf':'https://arxiv.org/pdf/2203.09893',
'mt3.pdf':'https://arxiv.org/pdf/2111.03017',
'maestro_original.html':'https://magenta.tensorflow.org/datasets/maestro',
}.items():jobs.append((name,url))
def run(j):
 n,u=j
 try:
  q=requests.get(u,timeout=40);q.raise_for_status();p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(q.content)
  if n.endswith('.pdf'):subprocess.run(['pdftotext','-layout',str(p),str(p.with_suffix('.txt'))],check=True)
  return {'file':n,'url':u,'status':q.status_code}
 except Exception as e:return {'file':n,'url':u,'error':str(e)}
a=list(concurrent.futures.ThreadPoolExecutor(8).map(run,jobs));(root/'retrieval_code.json').write_text(json.dumps(a,indent=2));print(json.dumps(a,indent=2))
