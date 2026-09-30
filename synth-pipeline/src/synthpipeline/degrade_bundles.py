"""Apply the realistic-recording degradation to rendered bundles.

Only ``performance_audio.wav`` is degraded; the reference stays a clean
render, as in DataCreate bundles. The clean performance render is kept as
``performance_audio_clean.wav`` so a later degrade version can start from it.
Labels, scores, and note maps are untouched because every degradation stage
preserves timing.
"""

from __future__ import annotations

import json
import logging
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

CLEAN_NAME = "performance_audio_clean.wav"


def bundle_seed(sample_dir: Path, base_seed: int) -> int:
    return (zlib.crc32(Path(sample_dir).name.encode("utf-8")) ^ (int(base_seed) * 0x9E3779B1)) & 0xFFFFFFFF


def degrade_bundle(
    sample_dir: Path,
    params: dict,
    *,
    force: bool = False,
    sample_rate: int = 22050,
    require_render: str | None = "musesounds_v1",
    dataset_version: str | None = None,
) -> str:
    from datacreate.audio_utils import load_audio, save_wav
    from datacreate.stages.stage5_alignment import run_alignment, write_candidates
    from datacreate.stages.stage7_features import extract_mels
    from synthpipeline.degrade import DEGRADE_MARK, degrade_audio
    from synthpipeline.regenerate_audio import _dc_config

    sample_dir = Path(sample_dir)
    meta_path = sample_dir / "metadata.json"
    perf = sample_dir / "performance_audio.wav"
    ref = sample_dir / "reference_audio.wav"
    if not meta_path.is_file() or not perf.is_file() or not ref.is_file():
        return "skip_missing"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    version = str(params.get("version", DEGRADE_MARK))
    if not force and meta.get("audio_degrade") == version:
        return "skip_done"
    if require_render and meta.get("audio_render") != require_render:
        return "skip_render"

    clean_path = sample_dir / CLEAN_NAME
    if clean_path.is_file():
        clean, _ = load_audio(clean_path, sample_rate, mono=True)
    else:
        clean, _ = load_audio(perf, sample_rate, mono=True)
        save_wav(clean_path, clean, sample_rate)

    rng = np.random.default_rng(bundle_seed(sample_dir, int(params.get("seed", 0))))
    audio, info = degrade_audio(clean, sample_rate, params, rng)
    save_wav(perf, audio, sample_rate)

    logger = logging.getLogger("synthpipeline.degrade")
    logger.setLevel(logging.WARNING)
    cfg = _dc_config(meta, Path(str(meta.get("soundfont_path") or "unused.sf2")), sample_rate)
    extract_mels(perf, ref, sample_dir, cfg, logger)
    alignment = run_alignment(perf, ref, sample_dir, cfg, logger)
    schema = "1.2"
    labels_path = sample_dir / "labels.json"
    if labels_path.is_file():
        schema = str(json.loads(labels_path.read_text(encoding="utf-8")).get("schema_version") or schema)
    write_candidates(alignment.candidates, sample_dir, schema)

    meta["audio_degrade"] = version
    meta["degrade"] = info
    meta["performance_audio_clean"] = CLEAN_NAME
    if dataset_version:
        meta["dataset_version"] = dataset_version
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return "converted"


def _one(payload: tuple) -> tuple[str, str]:
    sample, params, force, sr, version = payload
    try:
        return sample, degrade_bundle(
            Path(sample), params, force=force, sample_rate=sr, dataset_version=version
        )
    except Exception as exc:
        print(f"failed {sample}: {type(exc).__name__}: {exc}", flush=True)
        return sample, "failed"


def degrade_root(
    root: Path,
    params: dict,
    *,
    force: bool = False,
    workers: int = 8,
    sample_rate: int = 22050,
    dataset_version: str | None = None,
) -> dict[str, int]:
    from synthpipeline.transpose_audio import discover_bundles

    if not params:
        raise ValueError("config has no degrade block")
    dirs = discover_bundles(root)
    counts = {"converted": 0, "skip_done": 0, "skip_missing": 0, "skip_render": 0, "failed": 0, "n_bundles": len(dirs)}
    jobs = [(str(p), params, bool(force), int(sample_rate), dataset_version) for p in dirs]
    workers = max(1, int(workers))
    if workers == 1:
        statuses = [_one(job)[1] for job in jobs]
    else:
        statuses = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_one, job) for job in jobs]
            for i, fut in enumerate(as_completed(futs), start=1):
                statuses.append(fut.result()[1])
                if i % 100 == 0 or i == len(futs):
                    print(f"  {root}: {i}/{len(futs)}", flush=True)
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return counts
