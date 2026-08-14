from __future__ import annotations

import copy
import logging
import random
from pathlib import Path

from music21 import stream

from synthpipeline.config import SynthConfig
from synthpipeline.errors import inject_error
from synthpipeline.render import render_score_as_clarinet
from synthpipeline.scoregen import generate_score, load_score, resolve_score_inputs, write_musicxml
from synthpipeline.timing import refine_labels


def generate_samples(
    config: SynthConfig,
    count: int,
    seed: int,
    output_root: Path | None = None,
    score_arg: Path | None = None,
    logger: logging.Logger | None = None,
) -> list[Path]:
    log = logger or logging.getLogger("synthpipeline")
    dc_config = config.to_datacreate_config()
    root = output_root or config.output_root()
    root.mkdir(parents=True, exist_ok=True)

    score_paths = resolve_score_inputs(score_arg)
    sample_dirs: list[Path] = []
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
        try:
            _build_sample(
                clean,
                sample_dir,
                rng,
                config,
                dc_config,
                sample_log,
                extra_meta={"source": source, "seed": seed + i, "index": i},
            )
            sample_dirs.append(sample_dir)
            log.info("Created %s", sample_dir)
        except Exception:
            sample_log.exception("Failed to build sample %s", sample_id)
            log.exception("Failed to build sample %s", sample_id)
            raise
    return sample_dirs


def _build_sample(
    clean: stream.Score,
    sample_dir: Path,
    rng: random.Random,
    config: SynthConfig,
    dc_config,
    logger: logging.Logger,
    extra_meta: dict,
) -> None:
    from datacreate.models import LabelsDocument
    from datacreate.stages.stage4_performance import ingest_performance
    from datacreate.stages.stage5_alignment import run_alignment, write_candidates
    from datacreate.stages.stage7_features import extract_mels
    from datacreate.stages.stage8_bundle import write_metadata
    from datacreate.tools.musescore import check_musescore_version
    from datacreate.utils import write_json

    verified_path = sample_dir / "verified_score.musicxml"
    write_musicxml(clean, verified_path)

    result = inject_error(copy.deepcopy(clean), rng, config)
    performance_path = sample_dir / "performance_score.musicxml"
    write_musicxml(result.score, performance_path)

    check_musescore_version(dc_config, logger)
    clarinet_program = config.clarinet_program()
    ref_wav = sample_dir / "reference_audio.wav"
    perf_wav = sample_dir / "performance_audio.wav"
    render_score_as_clarinet(dc_config, verified_path, ref_wav, logger, clarinet_program)
    render_score_as_clarinet(dc_config, performance_path, perf_wav, logger, clarinet_program)
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


def _sample_logger(sample_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"synthpipeline.{sample_dir.name}")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.FileHandler(sample_dir / "pipeline.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger
