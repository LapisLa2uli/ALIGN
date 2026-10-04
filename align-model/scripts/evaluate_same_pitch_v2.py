"""Paired frozen v7/v9 comparison: same audio/model/gates, score identities only.

Reuses v7 acoustic caches. Bounds fitted on injected controls; no DataCreate >=095.
"""
from __future__ import annotations
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'align-model/runs/stack-v9'
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT/'numba-cache'))
import collections
from concurrent.futures import ProcessPoolExecutor
import evaluate_stack_v7 as ev
import numpy as np
from alignmodel.joint.stack_v7 import align_outputs as before, feedback as old_feedback
from alignmodel.joint.stack_v9 import align_outputs as after, feedback
from alignmodel.transcription.mel_v1 import load_audio_mono
from alignmodel.joint.presence_verifier_v1 import score_presence
from realistic92_aligner_common import load_clip, metric_sample
from precision_harness_v4 import remap


def worker(job):
    dataset, name, path, lineage = job
    c = ev.STATE['candidate']; config = c['_config']
    cache_path = ROOT/'align-model/runs/stack-v7/eval-cache/baseline'/dataset/f'{name}.npz'
    with np.load(cache_path) as cache:
        outputs = {k: cache[k].astype(np.float32) for k in cache.files}
    mel = outputs.pop('mel')
    audio = load_audio_mono(Path(path)/'performance_audio.wav', 22050)
    presence = lambda queries: score_presence(ev.STATE['presence'], mel, queries, 'cpu')
    result = {}
    for variant, align, make_feedback in [('before', before, old_feedback), ('after', after, feedback)]:
        extra = {'mel': mel, 'audio': audio} if variant == 'after' else {}
        index, alignment, events, deletions, info = align(outputs, Path(path)/'verified_score.musicxml',
            c, presence=presence, config=config, **extra)
        payload = {'sample': name, 'variant': variant, 'diagnostics': info,
                   'feedback': make_feedback(index, alignment, events, deletions, info),
                   'missed_score_event_indices': sorted(deletions),
                   'events': [{'note_index':e.rendered_index, 'pitch':e.pitch, 'start':e.start, 'end':e.end,
                               'score_span':e.score_span, 'relationship':e.relationship, 'copy_pass':e.copy_pass}
                              for e in events]}
        ev.write(OUT/'predictions'/variant/dataset/f'{name}.json', payload)
        if lineage:
            # Gold lineage is loaded after prediction, used for evaluation only.
            clip = load_clip(Path(path).parent, name, Path(lineage))
            result[variant] = metric_sample(clip, remap(events, clip), deletions)
    return dataset, name, result, info['same_pitch_repair']


def main():
    candidate_path = OUT/'CANDIDATE_STACK_V9.json'
    candidate = ev.read(candidate_path)
    for relative, digest in candidate['code_sha256'].items():
        assert ev.sha(ROOT/'align-model'/relative) == digest, relative
    c = ev.read(candidate['base_candidate'])
    c['_config'] = {k:candidate[k] for k in ('decoder', 'gate', 'minimum_match_fraction', 'same_pitch')}
    jobs = ev.read(ROOT/'reports/current_pipeline_20261003/manifest.json')['jobs']
    assert all(d!='datacreate' or 1<=int(n)<=94 for d,n,_,_ in jobs)
    ev.write(OUT/'evaluation_protocol.json', {'candidate_sha256':ev.sha(candidate_path), 'jobs':jobs,
        'threshold_selection':'Bounds fitted to 316 injected splits in the same development clips; not held out. DataCreate not used.',
        'metrics':'canonical reference-note identity; no timestamps',
        'synthetic_status':'90 previously inspected validation clips, not a sealed test',
        'datacreate_status':'001-094 processed; only reviewed takes scored'})
    samples = collections.defaultdict(list); audits = []
    with ProcessPoolExecutor(max_workers=3, initializer=ev.init, initargs=(c,)) as pool:
        for i,(dataset,name,result,audit) in enumerate(pool.map(worker,jobs),1):
            for variant,sample in result.items(): samples[(variant,dataset)].append(sample)
            audits.append({'dataset':dataset, 'sample':name, **audit})
            if i%10==0 or i==len(jobs): print('processed', i, len(jobs), flush=True)
    metrics = {v:{d:ev.aggregate(samples[(v,d)]) for d in c['datasets']} for v in ('before','after')}
    ev.write(OUT/'synthetic.json', metrics)
    ev.write(OUT/'same_pitch_audit.json', audits)
    ev.OUT = OUT
    ev.real_metrics('before'); ev.real_metrics('after')
    summary = {d:{'pairs_checked':sum(x['pairs_checked'] for x in audits if x['dataset']==d),
        'merged_boundaries':sum(x['merged_boundaries'] for x in audits if x['dataset']==d),
        'clips_changed':sum(x['merged_boundaries']>0 for x in audits if x['dataset']==d)}
        for d in [*c['datasets'],'datacreate']}
    ev.write(OUT/'COMPLETED.json', {'clips':len(jobs), 'summary':summary,
        'macro_content_error_f1':{v:float(np.mean([x['error_f1'] for x in metrics[v].values()]))
                                  for v in metrics}})
    print('COMPLETE', summary, flush=True)


if __name__ == '__main__': main()

