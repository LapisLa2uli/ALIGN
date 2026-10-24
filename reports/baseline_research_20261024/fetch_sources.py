import requests, pathlib, concurrent.futures, json
root=pathlib.Path('reports/baseline_research_20261024/sources')
urls={
'polytune.pdf':'https://arxiv.org/pdf/2501.02030',
'laddersym.pdf':'https://openreview.net/pdf?id=cizuvfyQXs',
'laddersym_github.json':'https://api.github.com/repos/ben2002chou/LadderSym',
'polytune_github.json':'https://api.github.com/repos/ben2002chou/Polytune',
'laddersym_README.md':'https://raw.githubusercontent.com/ben2002chou/LadderSym/main/README.md',
'polytune_README.md':'https://raw.githubusercontent.com/ben2002chou/Polytune/main/README.md',
'maestro_e_card.md':'https://huggingface.co/datasets/ben2002chou/MAESTRO-E/raw/main/README.md',
'coco_e_card.md':'https://huggingface.co/datasets/ben2002chou/CocoChorales-E/raw/main/README.md',
'generator_github.json':'https://api.github.com/repos/ben2002chou/CocoChorales-E_MAESTRO-E',
}
def get(kv):
 n,u=kv
 try:
  r=requests.get(u, timeout=45); r.raise_for_status(); (root/n).write_bytes(r.content)
  if n.endswith('.pdf'):
   import subprocess
   subprocess.run(['pdftotext','-layout',str(root/n),str(root/n.replace('.pdf','.txt'))],check=True)
  return {'file':n,'url':u,'status':r.status_code,'bytes':len(r.content)}
 except Exception as e:return {'file':n,'url':u,'error':str(e)}
results=list(concurrent.futures.ThreadPoolExecutor(6).map(get,urls.items()))
(root/'retrieval.json').write_text(json.dumps(results,indent=2),encoding='utf8')
print(json.dumps(results,indent=2))

