import pathlib,json,hashlib,collections,re
root=pathlib.Path('reports/baseline_research_20261024'); src=root/'sources'
records=[]
for f in sorted(src.glob('retrieval*.json')):records.extend(json.loads(f.read_text(encoding='utf-8-sig')))
urls={r['file']:r['url'] for r in records if r.get('status')==200}
files=[]
for f in sorted(src.rglob('*')):
 if f.is_file():
  name=f.relative_to(src).as_posix();files.append({'path':'sources/'+name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'source_url':urls.get(name)})
manifest={'research_date':'2026-10-24','research_timezone':'Asia/Shanghai','files':files,'upstream_heads':{x:json.loads((src/(x+'_tree.json')).read_text())['sha'] for x in ['laddersym','polytune','thegluenote','generator']},'scope':'Literature, source and metadata inspection; no model execution or full dataset audit.'}
(root/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
report=(root/'REPORT.md').read_text(encoding='utf-8-sig')
print('Report:',len(report.split()),'words;',len(report.splitlines()),'lines')
print('Source files:',len(files));print('PDFs:',[x['path'] for x in files if x['path'].endswith('.pdf')])
print('Section count:',len(re.findall(r'^## ',report,re.M)))
print('Markdown fences balanced:',report.count('```')%2==0)
print('Report bytes:',(root/'REPORT.md').stat().st_size)

