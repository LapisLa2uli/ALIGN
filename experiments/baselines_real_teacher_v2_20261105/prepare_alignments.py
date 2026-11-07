"""Compute input-only retrieval maps and a disjoint recording validation split."""
import sys
import time
from common import ROOT, RUN, PREVIOUS, read, write, sha, split
from reference_alignment import align

def main():
    import soundfile as sf
    partitions = split()
    assert len(partitions['validation']) == 12
    assert not set(partitions['train']) & set(partitions['validation'] + partitions['test'])
    write(RUN / 'split.json', partitions)
    records = read(PREVIOUS / 'data/manifest.json')['tracks']
    ids = partitions['train'] + partitions['validation'] + partitions['test']
    def guard(event, values):
        if event == 'open' and values:
            name = str(values[0]).replace('\\', '/')
            if any(x in name for x in ['/teacher_gold/', '/supervision/', '/teacher_transcriptions/']) or name.endswith(('/labels.json', '/labels_agent.json', '/note_alignment_v2.json', '/feedback_v9.json')):
                raise PermissionError('Acoustic alignment cannot read labels: ' + name)
    sys.addaudithook(guard)
    for i, sid in enumerate(ids, 1):
        out = RUN / 'alignment' / f'{sid}.json'
        paths = records[sid]['paths']
        digests = {k: sha(paths[k]) for k in ['mistake_wav', 'score_wav']}
        if out.exists():
            assert read(out)['input_sha256'] == digests
            continue
        began = time.monotonic()
        perf, sr = sf.read(paths['mistake_wav'], dtype='float32')
        ref, rr = sf.read(paths['score_wav'], dtype='float32')
        assert sr == rr == 16000
        perf = perf[:round(records[sid]['duration'] * sr)]
        result = align(perf, ref, sr)
        result.update(sample=sid, input_sha256=digests, labels_read=False, seconds=time.monotonic() - began)
        write(out, result)
        print('ALIGNED', i, len(ids), sid, round(result['seconds'], 2), 'jumps', result['backward_jumps'], result['forward_jumps'], flush=True)

if __name__ == '__main__':
    main()
