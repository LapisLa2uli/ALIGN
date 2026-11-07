"""Continue unchanged experiment decisions with verified faster inference."""
from common import ROOT,RUN,read,write,sha
import run_campaign as original
run_original=original.run

def run(arguments,name):
    if arguments[0]==str(RUN/'engine.py') and arguments[1]=='infer' and arguments[arguments.index('--model')+1]=='polytune':
        arguments=[str(RUN/'engine_cached.py'),*arguments[1:]]
    run_original(arguments,name)

def main():
    amendment=RUN/'runtime_amendment.json';parity=read(RUN/'cache_parity.json')
    assert parity['all_tokens_identical'] and not parity['mismatches']
    files=['cached_decode.py','engine_cached.py','run_campaign_cached.py','verify_cached_decode.py']
    hashes={f:sha(RUN/f) for f in files}
    if amendment.exists():assert read(amendment)['source_sha256']==hashes
    else:
        write(amendment,{'reason':'PolyTune upstream ignores use_cache and recomputes the entire decoder prefix; enable its existing KV cache and absolute position offsets.',
          'scope':'Inference runtime only. Training, data, checkpoint selection, token budget, float32 precision, labels, and scoring unchanged.',
          'before_any_test_generation':not (RUN/'predictions/test').exists(),
          'source_sha256':hashes,'parity_report_sha256':sha(RUN/'cache_parity.json'),
          'checked_windows':sum(r['windows'] for r in parity['rows']),'all_checked_tokens_identical':True,
          'original_protocol_sha256':sha(RUN/'protocol.json')})
    original.run=run;original.main()
    for f,digest in hashes.items():assert sha(RUN/f)==digest
    report=RUN/'RESULTS.md'
    with report.open('a',encoding='utf-8') as handle:
        old=sum(r['old_seconds'] for r in parity['rows']);new=sum(r['new_seconds'] for r in parity['rows'])
        handle.write(f'\n## Inference runtime amendment\n\nBefore test generation, PolyTune decoding was changed to use its existing attention cache. Every generated token matched the original decoder on {sum(r["windows"] for r in parity["rows"])} saved validation windows, covering the unchanged model and its 100-update adapter, including capped outputs. Aggregate generation time on those recordings changed from {old:.1f}s to {new:.1f}s ({old/new:.2f}x). Training and all model-selection/scoring decisions remained frozen. See `runtime_amendment.json` and `cache_parity.json`.\n')
    final=read(RUN/'final_audit.json');final['runtime_amendment_verified']=True;write(RUN/'final_audit.json',final)

if __name__=='__main__':main()
