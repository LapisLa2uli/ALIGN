from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from datacreate.config import PipelineConfig
from datacreate.stages.stage1_ingest import ingest_verified_score, validate_score
from datacreate.stages.stage2_omr import process_pdf
from datacreate.stages.stage3_reference import synthesize_reference
from datacreate.stages.stage4_performance import ingest_performance
from datacreate.stages.stage5_alignment import run_alignment, write_candidates
from datacreate.stages.stage7_features import extract_mels
from datacreate.stages.stage8_bundle import write_labels_template, write_metadata
from datacreate.utils import ensure_dir, setup_sample_logger


@dataclass
class SampleJob:
    sample_id: str
    sample_dir: Path
    score_path: Path | None = None
    performance_path: Path | None = None
    verified_score_path: Path | None = None
    state: dict[str, bool] = field(default_factory=dict)


class DataCreatePipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig.load()
        self.samples_root = ensure_dir(
            self.config.resolved_path("samples_root") or Path("samples")
        )
        self._reference_cache: dict[str, Path] = {}

    def create_sample(
        self,
        sample_id: str | None = None,
        score: Path | None = None,
        performance: Path | None = None,
        verified_score: Path | None = None,
    ) -> SampleJob:
        sample_id = sample_id or f"sample_{uuid.uuid4().hex[:8]}"
        sample_dir = ensure_dir(self.samples_root / sample_id)
        logger = setup_sample_logger(sample_dir)
        job = SampleJob(
            sample_id=sample_id,
            sample_dir=sample_dir,
            score_path=score,
            performance_path=performance,
            verified_score_path=verified_score,
        )
        logger.info("Created sample job %s", sample_id)
        return job

    def run_stage1_2(self, job: SampleJob) -> Path:
        logger = setup_sample_logger(job.sample_dir)
        if job.score_path is None:
            raise ValueError("score_path is required for ingestion")
        suffix = job.score_path.suffix.lower()
        if suffix == ".pdf":
            verified = process_pdf(
                job.score_path,
                job.sample_dir,
                self.config,
                logger,
                job.verified_score_path,
            )
        elif suffix in {".musicxml", ".mxl", ".xml"}:
            verified = validate_score(job.score_path, logger)
            if job.verified_score_path and job.verified_score_path.exists():
                verified = job.verified_score_path
            ingest_verified_score(verified, job.sample_dir)
        else:
            raise ValueError(f"Unsupported score input: {job.score_path}")
        job.state["ingest"] = True
        return job.sample_dir / "verified_score.musicxml"

    def run_stage3(self, job: SampleJob, score_path: Path | None = None) -> Path:
        logger = setup_sample_logger(job.sample_dir)
        score = score_path or job.sample_dir / "verified_score.musicxml"
        if not score.exists():
            raise FileNotFoundError(f"Verified score missing: {score}")
        output_wav = job.sample_dir / "reference_audio.wav"
        cache_key = f"{score.resolve()}:{score.stat().st_mtime_ns}"
        cached = self._reference_cache.get(cache_key)
        if cached is not None and cached.exists():
            shutil.copy2(cached, output_wav)
            logger.info("Reused cached reference audio from %s", cached)
        else:
            wav = synthesize_reference(score, job.sample_dir, self.config, logger)
            self._reference_cache[cache_key] = wav
            output_wav = wav
        job.state["reference"] = True
        return output_wav

    def run_stage4(self, job: SampleJob) -> Path:
        logger = setup_sample_logger(job.sample_dir)
        if job.performance_path is None:
            raise ValueError("performance_path is required")
        wav = ingest_performance(
            job.performance_path, job.sample_dir, self.config, logger
        )
        job.state["performance"] = True
        return wav

    def run_stage5(self, job: SampleJob) -> Path:
        logger = setup_sample_logger(job.sample_dir)
        perf = job.sample_dir / "performance_audio.wav"
        ref = job.sample_dir / "reference_audio.wav"
        if not perf.exists() or not ref.exists():
            raise FileNotFoundError("Both performance and reference audio required")
        result = run_alignment(perf, ref, job.sample_dir, self.config, logger)
        write_candidates(result.candidates, job.sample_dir, self.config.schema_version)
        job.state["alignment"] = True
        return job.sample_dir / "candidates.json"

    def run_stage7(self, job: SampleJob) -> None:
        logger = setup_sample_logger(job.sample_dir)
        perf = job.sample_dir / "performance_audio.wav"
        ref = job.sample_dir / "reference_audio.wav"
        extract_mels(perf, ref, job.sample_dir, self.config, logger)
        write_labels_template(job.sample_dir, self.config)
        write_metadata(job.sample_dir, self.config, {"sample_id": job.sample_id}, logger)
        job.state["features"] = True

    def run_batch(
        self,
        score: Path,
        performance: Path,
        sample_id: str | None = None,
        verified_score: Path | None = None,
    ) -> SampleJob:
        """Run stages 1-5 and 7 without annotation UI."""
        job = self.create_sample(
            sample_id=sample_id,
            score=score,
            performance=performance,
            verified_score=verified_score,
        )
        self.run_stage1_2(job)
        self.run_stage3(job)
        self.run_stage4(job)
        self.run_stage5(job)
        self.run_stage7(job)
        return job

    def resume(self, sample_id: str) -> SampleJob:
        sample_dir = self.samples_root / sample_id
        if not sample_dir.exists():
            raise FileNotFoundError(sample_dir)
        job = SampleJob(sample_id=sample_id, sample_dir=sample_dir)
        if (sample_dir / "verified_score.musicxml").exists():
            job.state["ingest"] = True
        if (sample_dir / "reference_audio.wav").exists():
            job.state["reference"] = True
        if (sample_dir / "performance_audio.wav").exists():
            job.state["performance"] = True
        if (sample_dir / "candidates.json").exists():
            job.state["alignment"] = True
        if (sample_dir / "performance_mel.npy").exists():
            job.state["features"] = True
        return job
