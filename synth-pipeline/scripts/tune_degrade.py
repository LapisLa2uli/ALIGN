"""Tune the ``degrade`` block so degraded synth log-mels match DataCreate recordings.

Each round refits the EQ curve and background-noise colour from the measured
spectral residuals, then coordinate-searches the scalar knobs. Clips are
degraded with fixed per-clip seeds so every evaluation sees the same draws.

Example:
    python scripts/tune_degrade.py --synth-root E:/realism_cache/tune_set \
        --config config/realistic_10k_random.yaml --out E:/realism_cache/tuned_degrade.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from synthpipeline.config import SynthConfig
from synthpipeline.degrade import degrade_audio
from synthpipeline.realism import clip_stats, compare, load_mono

from compare_mel_stats import load_real

SR = 22050


def objective(res: dict) -> float:
    gaps = [min(abs(v["gap_sd"]), 3.0) for v in res["scalars"].values()]
    spread = [
        abs(np.log((v["synth_iqr"] + 1e-3) / (v["real_iqr"] + 1e-3)))
        for k, v in res["scalars"].items()
        if v["real_iqr"] > 1e-3
    ]
    d = np.abs(np.subtract(res["delta_bands"]["real"], res["delta_bands"]["synth"])).mean()
    c = np.abs(np.subtract(res["contrast_bands"]["real"], res["contrast_bands"]["synth"])).mean()
    return float(
        np.mean(gaps)
        + 0.1 * np.mean(spread)
        + 0.15 * res["ltas_active_mae_db"]
        + 0.15 * res["ltas_quiet_mae_db"]
        + d
        + 0.1 * c
        + 2.0 * max(0.0, res["classifier_auc"] - 0.5)
    )


_CLIPS: list[np.ndarray] = []


def _init_worker(paths: list[str], max_seconds: float) -> None:
    _CLIPS.extend(load_mono(Path(p), SR, max_seconds) for p in paths)


def _degrade_stats(job: tuple[int, dict]):
    i, params = job
    y, _ = degrade_audio(_CLIPS[i], SR, params, np.random.default_rng(1000 + i))
    return clip_stats(y, SR)


class Evaluator:
    """Degrade the tuning clips in a persistent process pool (each worker holds all clips)."""

    def __init__(self, paths: list[Path], real, workers: int, max_seconds: float):
        self.n = len(paths)
        self.real = real
        self.calls = 0
        self.pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=([str(p) for p in paths], max_seconds),
        )

    def __call__(self, params: dict) -> tuple[float, dict]:
        self.calls += 1
        synth = list(self.pool.map(_degrade_stats, [(i, params) for i in range(self.n)]))
        res = compare(self.real, synth)
        return objective(res), res


def _mel_bin_hz() -> np.ndarray:
    import librosa

    return librosa.mel_frequencies(n_mels=128 + 2, fmin=30.0, fmax=SR / 2)[1:-1]


def refit_curves(params: dict, res: dict, real, rate: float = 0.35) -> None:
    """Move EQ gains and noise colour toward the real median spectra."""
    hz = _mel_bin_hz()
    ra = np.median([s.ltas_active for s in real], axis=0)
    sa = np.array(res["_synth_ltas_active"])
    rq = np.median([s.ltas_quiet for s in real], axis=0)
    sq = np.array(res["_synth_ltas_quiet"])

    eq = params["eq"]
    anchors = np.asarray(eq["anchors_hz"], dtype=float)
    resid = np.convolve(ra - sa, np.ones(7) / 7, mode="same")
    at = np.interp(np.log(anchors), np.log(hz), resid)
    eq["gains_db"] = [round(float(g + rate * r), 2) for g, r in zip(eq["gains_db"], at)]

    noise = params["noise"]
    q = np.convolve(rq - sq, np.ones(7) / 7, mode="same")
    level = float(np.median(q[8:96]))
    shape = np.interp(np.log(np.asarray(noise["anchors_hz"], float)), np.log(hz), q) - level
    noise["color_db"] = [round(float(g + rate * s), 2) for g, s in zip(noise["color_db"], shape)]
    lo, hi = noise["snr_db"]
    noise["snr_db"] = [round(lo - rate * level, 2), round(hi - rate * level, 2)]


def _shift(params, path, delta):
    node = params
    for key in path[:-1]:
        node = node[key]
    v = node[path[-1]]
    node[path[-1]] = [round(v[0] + delta, 3), round(v[1] + delta, 3)] if isinstance(v, list) else round(v + delta, 3)


def _scale(params, path, factor):
    node = params
    for key in path[:-1]:
        node = node[key]
    v = node[path[-1]]
    node[path[-1]] = [round(v[0] * factor, 4), round(v[1] * factor, 4)] if isinstance(v, list) else round(v * factor, 4)


KNOBS = [
    ("snr", lambda p, s: _shift(p, ["noise", "snr_db"], 3.0 * s)),
    ("snr_spread", lambda p, s: (_shift_lo(p, ["noise", "snr_db"], -2.0 * s))),
    ("reverb_wet", lambda p, s: _shift(p, ["reverb", "wet_db"], 3.0 * s)),
    ("rt60", lambda p, s: _scale(p, ["reverb", "rt60"], 1.25 ** s)),
    ("breath", lambda p, s: _shift(p, ["breath", "level_db"], 3.0 * s)),
    ("tongue_depth", lambda p, s: _shift(p, ["tonguing", "depth_db"], 3.0 * s)),
    ("tongue_width", lambda p, s: _scale(p, ["tonguing", "width_ms"], 1.3 ** s)),
    ("tongue_share", lambda p, s: _clip_share(p, 0.15 * s)),
    ("comp_ratio", lambda p, s: _scale(p, ["compressor", "ratio"], 1.25 ** s)),
    ("comp_threshold", lambda p, s: _shift(p, ["compressor", "threshold_db"], 3.0 * s)),
    ("pitch_cents", lambda p, s: _scale(p, ["micro", "pitch_cents"], 1.4 ** s)),
    ("shimmer", lambda p, s: _scale(p, ["micro", "shimmer_db"], 1.4 ** s)),
    ("dereverb", lambda p, s: _scale(p, ["dereverb", "strength"], 1.25 ** s)),
    ("dereverb_t60", lambda p, s: _scale(p, ["dereverb", "t60"], 1.3 ** s)),
    ("bitrate", lambda p, s: _shift(p, ["codec", "bitrate_k"], 6.0 * s)),
    ("sharpen", lambda p, s: _sharpen(p, 1.3 ** s)),
]


def _sharpen(params, factor):
    lo, hi = params["sharpen"]["gamma"]
    params["sharpen"]["gamma"] = [round(1.0 + (lo - 1.0) * factor, 4), round(1.0 + (hi - 1.0) * factor, 4)]

# Physically plausible limits: player pitch wander stays far below the 40–80
# cent intonation-error range, AGC never expands, bitrates stay voice-memo-like.
BOUNDS = {
    ("micro", "pitch_cents"): (1.0, 10.0),
    ("micro", "shimmer_db"): (0.1, 2.4),
    ("compressor", "ratio"): (1.0, 4.0),
    ("tonguing", "width_ms"): (8.0, 80.0),
    ("tonguing", "depth_db"): (2.0, 24.0),
    ("dereverb", "strength"): (0.1, 0.95),
    ("dereverb", "t60"): (0.4, 3.0),
    ("codec", "bitrate_k"): (32.0, 96.0),
    ("reverb", "rt60"): (0.1, 1.5),
    ("sharpen", "gamma"): (1.0, 1.6),
}


def clamp(params: dict) -> dict:
    for (stage, key), (lo, hi) in BOUNDS.items():
        node = params.get(stage)
        if not node or key not in node:
            continue
        v = node[key]
        if isinstance(v, list):
            node[key] = [round(float(np.clip(v[0], lo, hi)), 4), round(float(np.clip(v[1], lo, hi)), 4)]
        else:
            node[key] = round(float(np.clip(v, lo, hi)), 4)
    return params


def _shift_lo(params, path, delta):
    node = params
    for key in path[:-1]:
        node = node[key]
    lo, hi = node[path[-1]]
    node[path[-1]] = [round(min(lo + delta, hi - 1.0), 3), hi]


def _clip_share(params, delta):
    lo, hi = params["tonguing"]["share"]
    params["tonguing"]["share"] = [round(float(np.clip(lo + delta, 0.0, 0.95)), 3), round(float(np.clip(hi + delta, 0.05, 1.0)), 3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-root", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max-seconds", type=float, default=25.0)
    ap.add_argument("--n-clips", type=int, default=80)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", type=Path, default=Path("E:/realism_cache/real_stats.pkl"))
    args = ap.parse_args()

    real = load_real(args.cache, args.workers)
    wavs = sorted(args.synth_root.glob("synth_*/performance_audio.wav"))
    if args.n_clips and len(wavs) > args.n_clips:
        keep = np.linspace(0, len(wavs) - 1, args.n_clips).round().astype(int)
        wavs = [wavs[i] for i in keep]
    print(f"real={len(real)} tune clips={len(wavs)}", flush=True)
    ev = Evaluator(wavs, real, args.workers, args.max_seconds)

    def run(p):
        t = time.time()
        score, res = ev(p)
        return score, res, time.time() - t

    params = clamp(copy.deepcopy(SynthConfig.load(args.config).degrade))
    best, res, dt = run(params)
    print(f"start J={best:.3f} auc={res['classifier_auc']:.3f} ({dt:.0f}s)", flush=True)
    log = [{"round": 0, "J": best, "auc": res["classifier_auc"], "params": copy.deepcopy(params)}]

    for rnd in range(1, args.rounds + 1):
        for _ in range(2):
            trial = copy.deepcopy(params)
            refit_curves(trial, res, real)
            score, tres, _ = run(trial)
            print(f"r{rnd} refit J={score:.3f} (best {best:.3f}) auc={tres['classifier_auc']:.3f}", flush=True)
            if score < best:
                best, params, res = score, trial, tres
        for name, fn in KNOBS:
            for step in (1, -1):
                trial = copy.deepcopy(params)
                try:
                    fn(trial, step)
                except (KeyError, TypeError):
                    break
                clamp(trial)
                if trial == params:
                    continue
                score, tres, _ = run(trial)
                if score < best - 1e-3:
                    print(f"r{rnd} {name}{'+' if step > 0 else '-'} J={score:.3f} auc={tres['classifier_auc']:.3f}", flush=True)
                    best, params, res = score, trial, tres
                    break
        log.append({"round": rnd, "J": best, "auc": res["classifier_auc"], "params": copy.deepcopy(params)})
        args.out.write_text(yaml.safe_dump({"degrade": params}, sort_keys=False), encoding="utf-8")
        print(f"== round {rnd} J={best:.3f} auc={res['classifier_auc']:.3f} evals={ev.calls}", flush=True)

    from synthpipeline.realism import format_report

    print(format_report(res), flush=True)
    args.out.with_suffix(".json").write_text(
        json.dumps({"log": log, "final": {k: v for k, v in res.items() if not k.startswith("_")}}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
