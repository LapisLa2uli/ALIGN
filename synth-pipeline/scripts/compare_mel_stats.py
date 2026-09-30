"""Compare log-mel statistics of synth performance audio against DataCreate recordings.

Example:
    python scripts/compare_mel_stats.py --synth-root E:/outputRaw_orn_10k --n 160 \
        --degrade-config config/realistic_10k.yaml
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from synthpipeline.realism import clip_stats, compare, file_stats, format_report, load_mono

ROOT = Path(__file__).resolve().parents[2]
REAL_ROOT = ROOT / "DataCreate" / "samples"


def real_paths(root: Path = REAL_ROOT) -> list[Path]:
    out = []
    for d in sorted(root.iterdir()):
        wav = d / "performance_audio.wav"
        if d.is_dir() and re.fullmatch(r"\d{3}", d.name) and wav.is_file():
            out.append(wav)
    return out


def synth_paths(roots: list[Path], n: int, seed: int, clean: bool) -> list[Path]:
    rng = random.Random(seed)
    picks: list[Path] = []
    per_root = max(1, n // max(1, len(roots)))
    for root in roots:
        dirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("synth_")]
        rng.shuffle(dirs)
        got = 0
        for d in dirs:
            wav = d / ("performance_audio_clean.wav" if clean else "performance_audio.wav")
            if not wav.is_file():
                wav = d / "performance_audio.wav"
            if wav.is_file():
                picks.append(wav)
                got += 1
            if got >= per_root:
                break
    return picks


def _real_one(path: str):
    return file_stats(Path(path))


def _synth_one(payload: tuple):
    path, cfg_path, seed = payload
    audio = load_mono(Path(path))
    if cfg_path:
        from synthpipeline.config import SynthConfig
        from synthpipeline.degrade import degrade_audio

        params = SynthConfig.load(Path(cfg_path)).degrade
        audio, _info = degrade_audio(audio, 22050, params, np.random.default_rng(seed))
    return clip_stats(audio, 22050, name=path)


def load_real(cache: Path, workers: int):
    paths = real_paths()
    if cache.is_file():
        data = pickle.loads(cache.read_bytes())
        if data.get("paths") == [str(p) for p in paths]:
            return data["stats"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        stats = list(pool.map(_real_one, [str(p) for p in paths]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps({"paths": [str(p) for p in paths], "stats": stats}))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-root", type=Path, action="append", required=True)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--degrade-config", type=str, default=None)
    ap.add_argument("--clean", action="store_true", help="Prefer performance_audio_clean.wav")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cache", type=Path, default=Path("E:/realism_cache/real_stats.pkl"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    real = load_real(args.cache, args.workers)
    paths = synth_paths(args.synth_root, args.n, args.seed, args.clean)
    jobs = [(str(p), args.degrade_config, args.seed * 100003 + i) for i, p in enumerate(paths)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        synth = list(pool.map(_synth_one, jobs))
    result = compare(real, synth)
    print(format_report(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
