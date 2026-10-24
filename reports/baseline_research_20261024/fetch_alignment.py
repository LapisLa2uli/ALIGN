import requests,pathlib,subprocess,concurrent.futures,json
r=pathlib.Path('reports/baseline_research_20261024/sources')
urls={'thegluenote.pdf':'https://arxiv.org/pdf/2408.04309','thegluenote_README.md':'https://raw.githubusercontent.com/sildater/thegluenote/main/README.md','thegluenote_github.json':'https://api.github.com/repos/sildater/thegluenote','thegluenote_tree.json':'https://api.github.com/repos/sildater/thegluenote/git/trees/main?recursive=1','generator_tree.json':'https://api.github.com/repos/ben2002chou/CocoChorales-E_MAESTRO-E/git/trees/main?recursive=1'}
def fetch(x):
 n,u=x
 try:
  q=requests.get(u,timeout=40);q.raise_for_status();(r/n).write_bytes(q.content)
  if n.endswith('.pdf'):subprocess.run(['pdftotext','-layout',str(r/n),str(r/n.replace('.pdf','.txt'))],check=True)
  return {'file':n,'url':u,'status':q.status_code}
 except Exception as e:return {'file':n,'url':u,'error':str(e)}
a=list(concurrent.futures.ThreadPoolExecutor(5).map(fetch,urls.items()));(r/'retrieval_alignment.json').write_text(json.dumps(a,indent=2));print(json.dumps(a))
