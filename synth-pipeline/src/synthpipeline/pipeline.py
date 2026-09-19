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
from synthpipeline.errors import InjectionError, inject_error
from synthpipeline.note_map import (
    attach_rendered_events,
    build_note_map,
    tag_clean_notes,
    write_note_map,
)
from synthpipeline.render import render_score_as_clarinet
from synthpipeline.scoregen import (
    generate_score,
    load_score,
    resolve_score_inputs,
    snippet_score,
    sounding_note_count,
    write_musicxml,
)
from synthpipeline.timing import refine_labels


@dataclass
class SampleResult:
    sample_dir: Path
    elapsed_sec: float
    error_type: str
    repeated: bool


_COMPLETE_FILES = (
    "labels.json",
    "metadata.json",
    "performance_audio.wav",
    "reference_audio.wav",
    "verified_score.musicxml",
    "performance_score.musicxml",
    "note_map.json",
    "candidates.json",
    "alignment.npz",
    "performance_mel.npy",
    "reference_mel.npy",
    "performance_audio.mid",
    "reference_audio.mid",
)


def _sample_id_number(name: str) -> int | None:
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def sample_is_complete(sample_dir: Path) -> bool:
    return sample_dir.is_dir() and all(
        (sample_dir / name).is_file() and (sample_dir / name).stat().st_size > 0
        for name in _COMPLETE_FILES
    )


def complete_sample_ids(root: Path) -> set[int]:
    done: set[int] = set()
    if not root.exists():
        return done
    for path in root.iterdir():
        if not path.is_dir():
            continue
        number = _sample_id_number(path.name)
        if number is not None and sample_is_complete(path):
            done.add(number)
    return done


def generate_samples(
    config: SynthConfig,
    count: int,
    seed: int,
    output_root: Path | None = None,
    score_arg: Path | None = None,
    logger: logging.Logger | None = None,
    midi_backend: str | None = None,
    soundfont: str | None = None,
    skip_existing: bool = False,
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

    score_paths = resolve_score_inputs(score_arg, config)
    done_ids = complete_sample_ids(root) if skip_existing else set()
    results: list[SampleResult] = []
    for i in range(count):
        sid_num = int(seed) + i
        if skip_existing and sid_num in done_ids:
            log.info("Skipping existing sample id %s", sid_num)
            continue
        rng = random.Random(seed + i)
        snippet_meta: dict = {}
        source = "gen"
        clean = None
        last_prep_error: Exception | None = None
        for extra in range(24):
            try:
                if score_paths is None:
                    source = "gen"
                    clean = generate_score(rng, config)
                else:
                    path = score_paths[(i + extra) % len(score_paths)]
                    source = path.stem
                    clean = load_score(path, config)
                    if bool(config.generation.get("use_snippets", False)):
                        clean, snippet_meta = snippet_score(clean, rng, config)
                        snippet_meta["source_score"] = str(path)
                if sounding_note_count(clean) > 0:
                    break
                last_prep_error = InjectionError("Score has no notes")
                clean = None
            except Exception as exc:
                last_prep_error = exc
                clean = None
            rng = random.Random(seed + i + 1009 * (extra + 1))
        if clean is None:
            log.warning(
                "Skipping sample %s: could not find a note-bearing snippet (%s)",
                f"synth_{source}_{seed + i:04d}",
                last_prep_error,
            )
            continue
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
                    **snippet_meta,
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
            done_ids.add(sid_num)
            log.info("Created %s in %.2fs", sample_dir, elapsed)
        except InjectionError as exc:
            sample_log.exception("Failed to build sample %s", sample_id)
            log.warning("Skipping sample %s: %s", sample_id, exc)
            continue
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
    skip_existing: bool = False,
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
            skip_existing=skip_existing,
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
            "skip_existing": bool(skip_existing),
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
        skip_existing=bool(payload.get("skip_existing")),
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
    from synthpipeline.scoregen import assert_playable_clarinet_range

    keep_ornaments = bool(config.generation.get("keep_ornaments", False))
    if bool(config.generation.get("require_playable_range", False)):
        assert_playable_clarinet_range(clean, config, "clean score")

    verified_path = sample_dir / "verified_score.musicxml"
    write_musicxml(clean, verified_path, strip_ornaments=not keep_ornaments)

    tag_clean_notes(clean)
    result = inject_error(copy.deepcopy(clean), rng, config)
    if bool(config.generation.get("require_playable_range", False)):
        assert_playable_clarinet_range(result.score, config, "performance score")
    performance_path = sample_dir / "performance_score.musicxml"
    write_musicxml(result.score, performance_path, strip_ornaments=not keep_ornaments)
    # Capture lineage before MIDI export strips ornaments or otherwise
    # normalizes the in-memory score.
    note_map = build_note_map(clean, result.score)

    if midi_backend == "musescore":
        from datacreate.tools.musescore import check_musescore_version

        check_musescore_version(dc_config, logger)
    clarinet_program = config.clarinet_program()
    sounding = int(config.render.get("sounding_transpose", -2))
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
        sounding_transpose=sounding,
        keep_ornaments=keep_ornaments,
    )
    render_score_as_clarinet(
        dc_config,
        performance_path,
        perf_wav,
        logger,
        clarinet_program,
        midi_backend=midi_backend,
        score=result.score,
        pitch_bends=result.extra.get("pitch_bends"),
        bpm=result.bpm,
        sounding_transpose=sounding,
        keep_ornaments=keep_ornaments,
    )
    attach_rendered_events(
        note_map,
        perf_wav.with_suffix(".mid"),
        sounding_transpose=sounding,
        performed_score_path=performance_path,
    )
    if not note_map["rendered_notes"]:
        raise RuntimeError("Rendered MIDI has no note events; refusing an empty note map")
    write_note_map(sample_dir / "note_map.json", note_map)
    ingest_performance(perf_wav, sample_dir, dc_config, logger)

    midi_path = perf_wav.with_suffix(".mid")
    from datacreate.melody import parse_sounding_notes

    pad_choices = [int(x) for x in (config.errors.get("melody_pad_notes") or [1, 2])]
    if not pad_choices:
        pad_choices = [2]
    pad_notes = rng.choice(pad_choices)
    label_dicts = refine_labels(
        result.labels,
        result.bpm,
        midi_path,
        clean_notes=parse_sounding_notes(verified_path),
        pad_notes=pad_notes,
    )
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
    from synthpipeline.pitch_convention import annotate_pitch_metadata

    write_metadata(
        sample_dir,
        dc_config,
        annotate_pitch_metadata(
            {
                "mode": "synth-pipeline",
                "error_type": result.error_type,
                "error_types": list((result.extra or {}).get("error_types") or [result.error_type]),
                "repeated": result.repeated,
                "extra_copies": int((result.extra or {}).get("extra_copies") or 0),
                "melody_pad_notes": pad_notes,
                "sounding_transpose": sounding,
                "recording_kind": "synthetic",
                **extra_meta,
            },
            midi_space="sounding",
            audio_space="sounding",
            effective_audio_shift=-sounding,
            audio_render="soundfont_v1",
        ),
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
