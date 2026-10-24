import requests,pathlib,concurrent.futures,subprocess,json
r=pathlib.Path('reports/baseline_research_20261024/sources')
urls={
'laddersym.pdf':'https://arxiv.org/pdf/2510.08580v2',
'laddersym_license.txt':'https://raw.githubusercontent.com/ben2002chou/LadderSym/main/LICENSE',
'polytune_license.txt':'https://raw.githubusercontent.com/ben2002chou/Polytune/main/LICENSE',
'laddersym_tree.json':'https://api.github.com/repos/ben2002chou/LadderSym/git/trees/main?recursive=1',
'polytune_tree.json':'https://api.github.com/repos/ben2002chou/Polytune/git/trees/main?recursive=1',
'generator_README.md':'https://raw.githubusercontent.com/ben2002chou/CocoChorales-E_MAESTRO-E/main/README.md',
'mt3_README.md':'https://raw.githubusercontent.com/magenta/mt3/main/README.md',
'basic_pitch_README.md':'https://raw.githubusercontent.com/spotify/basic-pitch/main/README.md',
'parangonar_github.json':'https://api.github.com/repos/sildater/parangonar',
'matchmaker_github.json':'https://api.github.com/repos/pymatchmaker/matchmaker',
'maestro_split.json':'https://huggingface.co/datasets/ben2002chou/MAESTRO-E/resolve/main/maestro-v3.0.0.json',
'coco_split.json':'https://huggingface.co/datasets/ben2002chou/CocoChorales-E/resolve/main/split.json',
'laddersym_checkpoints.json':'https://huggingface.co/api/models/ben2002chou/laddersym-checkpoints',
'polytune_checkpoints.json':'https://huggingface.co/api/models/ben2002chou/Polytune',
}
def fetch(item):
 name,url=item
 try:
  q=requests.get(url,timeout=40);q.raise_for_status();(r/name).write_bytes(q.content)
  if name.endswith('.pdf'):subprocess.run(['pdftotext','-layout',str(r/name),str(r/name.replace('.pdf','.txt'))],check=True)
  return {'file':name,'url':url,'bytes':len(q.content),'status':q.status_code}
 except Exception as e:return {'file':name,'url':url,'error':str(e)}
a=list(concurrent.futures.ThreadPoolExecutor(6).map(fetch,urls.items()));(r/'retrieval_extra.json').write_text(json.dumps(a,indent=2));print(json.dumps(a,indent=2))
