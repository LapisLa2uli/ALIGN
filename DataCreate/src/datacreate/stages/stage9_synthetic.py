from __future__ import annotations

import copy
import logging
import random
import tempfile
from pathlib import Path

from music21 import converter, note, stream

from datacreate.config import PipelineConfig
from datacreate.models import Label
from datacreate.stages.stage1_ingest import copy_verified_score
from datacreate.stages.stage3_reference import synthesize_reference
from datacreate.stages.stage4_performance import ingest_performance
from datacreate.stages.stage5_alignment import run_alignment, write_candidates
from datacreate.stages.stage7_features import extract_mels
from datacreate.stages.stage8_bundle import write_labels_template, write_metadata
from datacreate.utils import write_json


CORRUPTIONS = [
    "shift_pitch",
    "shift_timing",
    "delete_note",
    "insert_note",
    "alter_duration",
]


def generate_synthetic_samples(
    score_path: Path,
    output_root: Path,
    config: PipelineConfig,
    logger: logging.Logger,
    count: int | None = None,
) -> list[Path]:
    count = count or int(config.synthetic.get("corruptions_per_score", 3))
    sample_dirs: list[Path] = []
    score = converter.parse(str(score_path))

    for i in range(count):
        sample_id = f"synthetic_{score_path.stem}_{i:03d}"
        sample_dir = output_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        corrupted = copy.deepcopy(score)
        corruption = CORRUPTIONS[i % len(CORRUPTIONS)]
        label = _apply_corruption(corrupted, corruption, i, logger)

        verified_path = copy_verified_score(score_path, sample_dir / "verified_score.musicxml")
        ref_wav = synthesize_reference(verified_path, sample_dir, config, logger)

        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        corrupted.write("musicxml", fp=str(tmp_path))
        render_corrupted = sample_dir / "corrupted_score.musicxml"
        render_corrupted.write_text(tmp_path.read_text(encoding="utf-8"), encoding="utf-8")
        perf_wav = sample_dir / "performance_audio.wav"
        from datacreate.tools.musescore import render_score_to_wav

        render_score_to_wav(config, render_corrupted, perf_wav, logger)
        ingest_performance(perf_wav, sample_dir, config, logger)

        alignment = run_alignment(perf_wav, ref_wav, sample_dir, config, logger)
        write_candidates(alignment.candidates, sample_dir, config.schema_version)
        extract_mels(perf_wav, ref_wav, sample_dir, config, logger)

        labels_doc = {
            "schema_version": config.schema_version,
            "audio_reference": "performance_audio.wav",
            "labels": [label.model_dump(exclude_none=True)],
            "self_reported": [],
        }
        write_json(sample_dir / "labels.json", labels_doc)
        write_metadata(
            sample_dir,
            config,
            {"mode": "synthetic", "corruption": corruption},
            logger,
        )
        sample_dirs.append(sample_dir)
        tmp_path.unlink(missing_ok=True)
        logger.info("Synthetic sample %s created", sample_dir)

    return sample_dirs


def _apply_corruption(
    score: stream.Score, corruption: str, seed: int, logger: logging.Logger
) -> Label:
    rng = random.Random(seed)
    notes = list(score.recurse().getElementsByClass(note.Note))
    if not notes:
        raise RuntimeError("Score has no notes to corrupt")

    target = rng.choice(notes)
    offset = float(target.offset)
    duration = float(target.duration.quarterLength)
    start_time = offset * 0.5
    end_time = start_time + duration * 0.5

    if corruption == "shift_pitch":
        semitones = rng.choice([-2, -1, 1, 2])
        target.pitch.transpose(semitones, inPlace=True)
        label_type = "wrong_pitch"
        comment = f"shifted {semitones} semitones"
    elif corruption == "shift_timing":
        shift = rng.choice([0.25, 0.5, -0.25])
        target.offset = offset + shift
        label_type = "rhythm_error"
        comment = f"onset shifted by {shift} quarters"
    elif corruption == "delete_note":
        parent = target.activeSite
        if parent:
            parent.remove(target)
        label_type = "missed_note"
        comment = "deleted note"
    elif corruption == "insert_note":
        dup = copy.deepcopy(target)
        dup.offset = offset + 0.25
        part = score.parts[0]
        part.insert(dup.offset, dup)
        label_type = "extra_note"
        comment = "inserted duplicate note"
    else:
        target.duration.quarterLength = max(0.25, duration * 0.5)
        label_type = "rhythm_error"
        comment = "shortened note duration"

    logger.info("Applied corruption %s: %s", corruption, comment)
    return Label(
        id="syn_001",
        source="synthetic",
        start_time=start_time,
        end_time=end_time,
        type=label_type,
        severity=5,
        comment=comment,
    )
