"""Label-free audio-to-reference retrieval, including costly backward jumps.

No teacher transcription, error labels, or human annotations are inputs.
The same acoustic path selects reference contexts during training and inference.
"""
import numpy as np

PARAMETERS = {'sample_rate': 16000, 'hop': 512, 'cqt_bins': 84,
    'jump_penalty': 8.0, 'stay_penalty': .035, 'skip_penalty': .025,
    'energy_mismatch_weight': .35, 'context_seconds': 4.096}

def acoustic_features(audio, sr=16000):
    import librosa
    from scipy.ndimage import median_filter
    cqt = np.abs(librosa.cqt(audio, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C1'), n_bins=84))
    cqt = median_filter(cqt, size=(1, 3))
    cqt = np.log1p(10 * cqt / np.maximum(cqt.max(axis=0, keepdims=True), 1e-8))
    chroma = cqt.reshape(7, 12, -1).sum(axis=0)
    def norm(x):
        return x / np.maximum(np.linalg.norm(x, axis=0, keepdims=True), 1e-8)
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    energy = np.interp(np.arange(cqt.shape[1]), np.arange(len(rms)), rms)
    energy = np.clip(energy / max(float(np.quantile(energy, .8)), 1e-8), 0., 1.)
    return norm(cqt).astype('float32'), norm(chroma).astype('float32'), energy.astype('float32')

def track_cost(cost):
    """Viterbi path over reference frames with local tempo changes and jumps."""
    n_ref, n_perf = cost.shape
    parents = np.empty((n_perf, n_ref), dtype=np.int32)
    positions = np.arange(n_ref)
    previous = cost[:, 0] + .002 * positions
    parents[0] = positions
    for t in range(1, n_perf):
        best = previous + PARAMETERS['stay_penalty']
        source = positions.copy()
        for advance in [1, 2, 3, 4]:
            candidate = previous[:-advance] + PARAMETERS['skip_penalty'] * abs(advance - 1)
            improve = candidate < best[advance:]
            best[advance:][improve] = candidate[improve]
            source[advance:][improve] = positions[:-advance][improve]
        origin = int(np.argmin(previous))
        jump = previous[origin] + PARAMETERS['jump_penalty']
        improve = jump < best
        best[improve] = jump
        source[improve] = origin
        previous = best + cost[:, t]
        previous -= previous.min()
        parents[t] = source
    path = np.empty(n_perf, dtype=np.int32)
    path[-1] = int(np.argmin(previous))
    for t in range(n_perf - 1, 0, -1):
        path[t - 1] = parents[t, path[t]]
    return path

def align(performance, reference, sr=16000):
    assert sr == PARAMETERS['sample_rate']
    pf, pc, pe = acoustic_features(performance, sr)
    rf, rc, re = acoustic_features(reference, sr)
    cost = .65 * (1 - rf.T @ pf) + .35 * (1 - rc.T @ pc)
    cost += PARAMETERS['energy_mismatch_weight'] * np.abs(re[:, None] - pe[None, :])
    cost = np.maximum(cost, 0.).astype('float32')
    path = track_cost(cost)
    hop_sec = PARAMETERS['hop'] / sr
    return {'performance_times': (np.arange(len(path)) * hop_sec).tolist(),
        'reference_times': (path * hop_sec).tolist(),
        'mean_acoustic_cost': float(cost[path, np.arange(len(path))].mean()),
        'backward_jumps': int(np.sum(np.diff(path) < -4)),
        'forward_jumps': int(np.sum(np.diff(path) > 4)), 'parameters': PARAMETERS}

def context_start(mapping, start, end):
    """Center a reference crop on the locally matched performance window.

Use the longest continuous path section when a window straddles a jump; do
not average two distant score passages into an unrelated middle passage.
"""
    times = np.asarray(mapping['performance_times'])
    refs = np.asarray(mapping['reference_times'])
    indices = np.flatnonzero((times >= start) & (times <= end))
    if not len(indices):
        indices = np.array([int(np.argmin(abs(times - (start + end) / 2)))])
    cuts = np.flatnonzero((np.diff(refs[indices]) < -.13) | (np.diff(refs[indices]) > .16)) + 1
    parts = np.split(indices, cuts)
    chosen = max(parts, key=lambda part: (len(part), -abs(float(np.median(times[part])) - (start + end) / 2)))
    center = float(np.median(refs[chosen]))
    return round((center - PARAMETERS['context_seconds'] / 2) * 125) / 125

def crop(audio, start, seconds, sr=16000):
    begin = round(start * sr)
    size = round(seconds * sr)
    result = np.zeros(size, dtype=np.float32)
    lo, hi = max(0, begin), min(len(audio), begin + size)
    if hi > lo:
        result[lo - begin:hi - begin] = audio[lo:hi]
    return result
