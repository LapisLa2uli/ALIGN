from __future__ import annotations

import copy
import logging
import os
import random
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from music21 import stream

from synthpipeline.config import SynthConfig
from synthpipeline.errors import inject_error
from synthpipeline.render import render_score_as_clarinet
from synthpipeline.scoregen import generate_score, load_score, resolve_score_inputs, write_musicxml
from synthpipeline.timing import refine_labels


@dataclass
class SampleResult:
    sample_dir: Path
    elapsed_sec: float
    error_type: str
    repeated: bool


def generate_samples(
    config: SynthConfig,
    count: int,
    seed: int,
    output_root: Path | None = None,
    score_arg: Path | None = None,
    logger: logging.Logger | None = None,
    midi_backend: str | None = None,
    soundfont: str | None = None,
) -> list[SampleResult]:
    log = logger or logging.getLogger("synthpipeline")
    if soundfont:
        config.render["soundfont"] = soundfont
    from synthpipeline.soundfonts import resolve_soundfont

    preset = resolve_soundfont(config)
    log.info("SoundFont %s (%s) program %d", preset.id, preset.path.name, preset.program)
    dc_config = config.to_datacreate_config()
    root = output_root or config.output_root()
    root.mkdir(parents=True, exist_ok=True)
    backend = (midi_backend or config.midi_backend()).lower()

    score_paths = resolve_score_inputs(score_arg)
    results: list[SampleResult] = []
    for i in range(count):
        rng = random.Random(seed + i)
        if score_paths is None:
            source = "gen"
            clean = generate_score(rng, config)
        else:
            path = score_paths[i % len(score_paths)]
            source = path.stem
            clean = load_score(path, config)
        sample_id = f"synth_{source}_{seed + i:04d}"
        sample_dir = root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_log = _sample_logger(sample_dir)
        started = time.perf_counter()
        try:
            built = _build_sample(
                clean,
                sample_dir,
                rng,
                config,
                dc_config,
                sample_log,
                extra_meta={
                    "source": source,
                    "seed": seed + i,
                    "index": i,
                    "midi_backend": backend,
                    "soundfont": preset.id,
                    "soundfont_path": str(preset.path),
                    "clarinet_program": preset.program,
                },
                midi_backend=backend,
            )
            elapsed = time.perf_counter() - started
            result = SampleResult(
                sample_dir=sample_dir,
                elapsed_sec=elapsed,
                error_type=built["error_type"],
                repeated=built["repeated"],
            )
            results.append(result)
            log.info("Created %s in %.2fs", sample_dir, elapsed)
        except Exception:
            sample_log.exception("Failed to build sample %s", sample_id)
            log.exception("Failed to build sample %s", sample_id)
            raise
    return results


def generate_samples_parallel(
    config: SynthConfig,
    count: int,
    seed: int,
    workers: int,
    output_root: Path | None = None,
    score_arg: Path | None = None,
    logger: logging.Logger | None = None,
    midi_backend: str | None = None,
    soundfont: str | None = None,
) -> list[SampleResult]:
    log = logger or logging.getLogger("synthpipeline")
    workers = max(1, int(workers))
    if workers == 1 or count <= 1:
        return generate_samples(
            config=config,
            count=count,
            seed=seed,
            output_root=output_root,
            score_arg=score_arg,
            logger=log,
            midi_backend=midi_backend,
            soundfont=soundfont,
        )

    config_path = config._config_path
    if config_path is None:
        from synthpipeline.config import DEFAULT_CONFIG

        config_path = DEFAULT_CONFIG
    jobs = _chunk_jobs(count, workers, seed)
    log.info("Parallel generate: %d samples across %d processes", count, len(jobs))
    payloads = [
        {
            "config_path": str(config_path),
            "count": n,
            "seed": job_seed,
            "output": str(output_root) if output_root else None,
            "score": str(score_arg) if score_arg else None,
            "midi_backend": midi_backend,
            "soundfont": soundfont,
            "worker_id": worker_id,
        }
        for job_seed, n, worker_id in jobs
    ]
    results: list[SampleResult] = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(worker_generate, payload): payload for payload in payloads}
        for future in as_completed(futures):
            payload = futures[future]
            try:
                batch = future.result()
            except Exception:
                log.exception("Worker %s failed", payload.get("worker_id"))
                raise
            for item in batch:
                results.append(
                    SampleResult(
                        sample_dir=Path(item["sample_dir"]),
                        elapsed_sec=float(item["elapsed_sec"]),
                        error_type=str(item["error_type"]),
                        repeated=bool(item["repeated"]),
                    )
                )
                log.info(
                    "Worker %s created %s in %.2fs",
                    payload.get("worker_id"),
                    item["sample_dir"],
                    item["elapsed_sec"],
                )
    results.sort(key=lambda r: r.sample_dir.name)
    return results


def worker_generate(payload: dict) -> list[dict]:
    """Picklable worker entry point for ProcessPoolExecutor."""
    worker_id = int(payload.get("worker_id", 0))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix=f"mpl-w{worker_id}-")
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [w{worker_id}] [%(levelname)s] %(message)s",
        force=True,
    )
    from synthpipeline.config import SynthConfig

    config = SynthConfig.load(payload["config_path"])
    output = Path(payload["output"]) if payload.get("output") else None
    score = Path(payload["score"]) if payload.get("score") else None
    results = generate_samples(
        config=config,
        count=int(payload["count"]),
        seed=int(payload["seed"]),
        output_root=output,
        score_arg=score,
        midi_backend=payload.get("midi_backend"),
        soundfont=payload.get("soundfont"),
    )
    return [
        {
            "sample_dir": str(item.sample_dir),
            "elapsed_sec": item.elapsed_sec,
            "error_type": item.error_type,
            "repeated": item.repeated,
        }
        for item in results
    ]


def _chunk_jobs(count: int, workers: int, seed: int) -> list[tuple[int, int, int]]:
    workers = max(1, min(int(workers), int(count)))
    base, extra = divmod(int(count), workers)
    jobs: list[tuple[int, int, int]] = []
    offset = 0
    for worker_id in range(workers):
        n = base + (1 if worker_id < extra else 0)
        if n <= 0:
            continue
        jobs.append((int(seed) + offset, n, worker_id))
        offset += n
    return jobs


def _build_sample(
    clean: stream.Score,
    sample_dir: Path,
    rng: random.Random,
    config: SynthConfig,
    dc_config,
    logger: logging.Logger,
    extra_meta: dict,
    midi_backend: str = "music21",
) -> dict:
    from datacreate.models import LabelsDocument
    from datacreate.stages.stage4_performance import ingest_performance
    from datacreate.stages.stage5_alignment import run_alignment, write_candidates
    from datacreate.stages.stage7_features import extract_mels
    from datacreate.stages.stage8_bundle import write_metadata
    from datacreate.utils import write_json

    verified_path = sample_dir / "verified_score.musicxml"
    write_musicxml(clean, verified_path)

    result = inject_error(copy.deepcopy(clean), rng, config)
    performance_path = sample_dir / "performance_score.musicxml"
    write_musicxml(result.score, performance_path)

    if midi_backend == "musescore":
        from datacreate.tools.musescore import check_musescore_version

        check_musescore_version(dc_config, logger)
    clarinet_program = config.clarinet_program()
    ref_wav = sample_dir / "reference_audio.wav"
    perf_wav = sample_dir / "performance_audio.wav"
    render_score_as_clarinet(
        dc_config,
        verified_path,
        ref_wav,
        logger,
        clarinet_program,
        midi_backend=midi_backend,
        score=clean,
    )
    render_score_as_clarinet(
        dc_config,
        performance_path,
        perf_wav,
        logger,
        clarinet_program,
        midi_backend=midi_backend,
        score=result.score,
    )
    ingest_performance(perf_wav, sample_dir, dc_config, logger)

    midi_path = perf_wav.with_suffix(".mid")
    label_dicts = refine_labels(result.labels, result.bpm, midi_path)
    labels_doc = LabelsDocument(
        schema_version=config.schema_version,
        audio_reference="performance_audio.wav",
        labels=label_dicts,
        self_reported=[],
    )
    write_json(sample_dir / "labels.json", labels_doc.model_dump(exclude_none=True))

    alignment = run_alignment(perf_wav, ref_wav, sample_dir, dc_config, logger)
    write_candidates(alignment.candidates, sample_dir, config.schema_version)
    extract_mels(perf_wav, ref_wav, sample_dir, dc_config, logger)
    write_metadata(
        sample_dir,
        dc_config,
        {
            "mode": "synth-pipeline",
            "error_type": result.error_type,
            "repeated": result.repeated,
            **extra_meta,
        },
        logger,
    )
    logger.info(
        "Sample complete: error=%s repeated=%s labels=%d",
        result.error_type,
        result.repeated,
        len(label_dicts),
    )
    return {"error_type": result.error_type, "repeated": result.repeated}


def _sample_logger(sample_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"synthpipeline.{sample_dir.name}")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.FileHandler(sample_dir / "pipeline.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger
