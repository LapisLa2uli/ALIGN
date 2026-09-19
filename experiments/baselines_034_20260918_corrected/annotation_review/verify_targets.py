"""Second F0 estimator on targeted windows; no annotation mutation."""
from pathlib import Path
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent


def check(job):
    import librosa
    from scipy.signal import resample_poly
    from math import gcd
    path = HERE.parent / "bundles" / job["sample"] / "performance_audio.wav"
    info = sf.info(path)
    lo, hi = max(0., job["start"] - .30), min(info.duration, job["end"] + .30)
    y, sr = sf.read(path, start=int(lo * info.samplerate), stop=int(hi * info.samplerate), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    g = gcd(sr, 16000)
    y = resample_poly(y, 16000 // g, sr // g)
    f0, voiced, prob = librosa.pyin(y, fmin=110, fmax=2000, sr=16000, frame_length=1024, hop_length=80,
                                  fill_na=np.nan)
    times = np.arange(len(f0)) * .005 + lo
    midi = librosa.hz_to_midi(f0)
    core = (times >= job["start"] + .015) & (times <= job["end"] - .015)
    selected = core & voiced & (prob >= .50) & np.isfinite(midi)
    pitches = Counter(np.rint(midi[selected]).astype(int).tolist())
    frames_path = HERE / "pyin" / (job["key"] + ".npz")
    np.savez_compressed(frames_path, time=times, midi=midi, voiced=voiced, probability=prob)
    # FFT peaks are an additional diagnostic, not a independent proof of F0.
    a, b = int((job["start"] - lo) * 16000), int((job["end"] - lo) * 16000)
    segment = y[a:b]
    rms = float(np.sqrt(np.mean(segment * segment))) if len(segment) else 0.
    result = dict(**job, reliable_frames=int(selected.sum()), total_core_frames=int(core.sum()),
                  dominant_sounding_pitches=pitches.most_common(6),
                  voiced_probability_median=float(np.median(prob[core])) if core.any() else None,
                  midi_median=float(np.median(midi[selected])) if selected.any() else None, rms=rms)
    return result


if __name__ == "__main__":
    (HERE / "pyin").mkdir(exist_ok=True)
    jobs = json.loads((HERE / "pyin_jobs.json").read_text())
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(check, job) for job in jobs]):
            result = future.result();results.append(result)
            print(result["key"], result["dominant_sounding_pitches"], result["reliable_frames"], flush=True)
    (HERE / "pyin_checks.json").write_text(json.dumps(sorted(results, key=lambda x: x["key"]), indent=2) + "\n")
