from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from datacreate.config import PipelineConfig
from datacreate.pipeline import DataCreatePipeline
from datacreate.utils import ensure_dir, write_json

AUDIO_EXTENSIONS = (".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".wma")


@dataclass
class BatchItemResult:
    sample_id: str
    status: str
    sample_dir: str | None = None
    error: str | None = None


@dataclass
class BatchRangeResult:
    score_path: str
    audio_dir: str
    id_from: int
    id_to: int
    results: list[BatchItemResult] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.status == "ok")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "error")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "skipped")


def format_sample_id(number: int, width: int = 3) -> str:
    return str(number).zfill(width)


def parse_sample_id(value: str | int, width: int = 3) -> tuple[int, str]:
    if isinstance(value, int):
        return value, format_sample_id(value, width)
    text = str(value).strip()
    if text.isdigit():
        n = int(text)
        return n, text.zfill(max(width, len(text)))
    match = re.search(r"(\d+)", text)
    if not match:
        raise ValueError(f"Cannot parse sample id from {value!r}")
    n = int(match.group(1))
    return n, text


def expand_id_range(id_from: int, id_to: int, width: int = 3) -> list[str]:
    if id_from > id_to:
        raise ValueError(f"id_from ({id_from}) must be <= id_to ({id_to})")
    return [format_sample_id(i, width) for i in range(id_from, id_to + 1)]


def find_audio_file(audio_dir: Path, sample_id: str) -> Path | None:
    for ext in AUDIO_EXTENSIONS:
        candidate = audio_dir / f"{sample_id}{ext}"
        if candidate.exists():
            return candidate
    matches = sorted(audio_dir.glob(f"{sample_id}.*"))
    return matches[0] if matches else None


def list_available_audio_ids(audio_dir: Path, width: int = 3) -> list[str]:
    if not audio_dir.exists():
        return []
    ids: set[str] = set()
    for path in audio_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        ids.add(path.stem)
    return sorted(ids, key=lambda s: int(re.sub(r"\D", "", s) or "0"))


def run_batch_range(
    score_path: Path,
    audio_dir: Path,
    id_from: int | str,
    id_to: int | str,
    config: PipelineConfig | None = None,
    id_width: int = 3,
    skip_existing: bool = True,
    logger: logging.Logger | None = None,
) -> BatchRangeResult:
    config = config or PipelineConfig.load()
    logger = logger or logging.getLogger("datacreate.batch")

    score_path = Path(score_path)
    audio_dir = Path(audio_dir)
    if not score_path.exists():
        raise FileNotFoundError(f"Score not found: {score_path}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    start_n, _ = parse_sample_id(id_from, id_width)
    end_n, _ = parse_sample_id(id_to, id_width)
    sample_ids = expand_id_range(start_n, end_n, id_width)

    pipeline = DataCreatePipeline(config)
    result = BatchRangeResult(
        score_path=str(score_path),
        audio_dir=str(audio_dir),
        id_from=start_n,
        id_to=end_n,
    )

    for sample_id in sample_ids:
        audio_path = find_audio_file(audio_dir, sample_id)
        if audio_path is None:
            msg = f"No audio file for id {sample_id} in {audio_dir}"
            logger.warning(msg)
            result.results.append(BatchItemResult(sample_id, "error", error=msg))
            continue

        sample_dir = pipeline.samples_root / sample_id
        if skip_existing and (sample_dir / "performance_audio.wav").exists():
            logger.info("Skipping existing sample %s", sample_id)
            result.results.append(
                BatchItemResult(sample_id, "skipped", sample_dir=str(sample_dir))
            )
            continue

        try:
            logger.info("Processing sample %s (%s)", sample_id, audio_path.name)
            job = pipeline.run_batch(
                score=score_path,
                performance=audio_path,
                sample_id=sample_id,
            )
            result.results.append(
                BatchItemResult(sample_id, "ok", sample_dir=str(job.sample_dir))
            )
        except Exception as exc:
            logger.exception("Failed sample %s", sample_id)
            result.results.append(BatchItemResult(sample_id, "error", error=str(exc)))

    manifest_dir = ensure_dir(pipeline.samples_root)
    write_json(
        manifest_dir / "batch_manifest.json",
        {
            "score_path": str(score_path),
            "audio_dir": str(audio_dir),
            "id_from": start_n,
            "id_to": end_n,
            "succeeded": result.succeeded,
            "failed": result.failed,
            "skipped": result.skipped,
            "results": [r.__dict__ for r in result.results],
        },
    )
    return result
