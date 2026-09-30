"""Transcribe and align every DataCreate take with the frozen 9.2 CTC + DP stack.

Inference only: performance_audio.wav -> frozen mel CTC transcriber (blank
scale from its candidate file) -> frozen structured DP aligner against
verified_score.musicxml. Outputs go to a run directory and to new per-sample
files (transcription_ctc_r2a.json, note_alignment_dp_v1.json). Existing files,
including labels.json and labels_agent.json, are never modified.

The CTC head emits one onset frame per note; note ends are set to the next
onset (last note: onset + 0.1 s) and are not measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.perfect_dp_aligner_v1 import PerfectDPCosts, align_perfect
from alignmodel.transcription.mel_ctc_v1 import (
    BLANK,
    greedy_decode,
    infer_ctc_probabilities,
    load_ctc_checkpoint,
)
from alignmodel.transcription.mel_v1 import extract_log_mel, load_audio_mono


SCHEMA_VERSION = "align-datacreate-ctc-r2a-dp-v1"
TRANSCRIPTION_FILE = "transcription_ctc_r2a.json"
ALIGNMENT_FILE = "note_alignment_dp_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    align = root / "align-model"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=root / "DataCreate" / "samples")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--transcriber-candidate", type=Path,
        default=align / "runs/realistic92-transcriber-v1/CANDIDATE_CTC_R2A.json",
    )
    parser.add_argument(
        "--aligner-candidate", type=Path,
        default=align / "runs/realistic92-aligner-v1/CANDIDATE_DP_V1.json",
    )
    parser.add_argument("--no-sample-files", action="store_true")
    args = parser.parse_args()

    manifest_path = args.output / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    transcriber = json.loads(args.transcriber_candidate.read_text(encoding="utf-8"))
    aligner = json.loads(args.aligner_candidate.read_text(encoding="utf-8"))
    checkpoint = Path(transcriber["checkpoint"])
    if _sha256(checkpoint) != transcriber["checkpoint_sha256"]:
        raise ValueError("Transcriber checkpoint changed")
    aligner_code = align / aligner["aligner_path"]
    if _sha256(aligner_code) != aligner["aligner_sha256"]:
        raise ValueError("Aligner code changed")
    costs = PerfectDPCosts(**aligner["costs"])
    blank_scale = float(transcriber["blank_scale"])

    samples = sorted(
        path for path in args.samples.iterdir()
        if path.is_dir()
        and (path / "performance_audio.wav").is_file()
        and (path / "verified_score.musicxml").is_file()
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, frontend, _payload = load_ctc_checkpoint(checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows, failures = [], []
    type_totals: Counter[str] = Counter()
    protected = {
        sample.name: {
            name: _sha256(sample / name)
            for name in ("labels.json", "labels_agent.json")
            if (sample / name).is_file()
        }
        for sample in samples
    }
    for position, sample in enumerate(samples, 1):
        started = time.perf_counter()
        try:
            audio_path = sample / "performance_audio.wav"
            audio = load_audio_mono(audio_path, frontend.sample_rate)
            mel, normalization = extract_log_mel(audio, frontend, device=device)
            probabilities = infer_ctc_probabilities(model, np.asarray(mel, np.float32), device)
            probabilities[:, BLANK] *= blank_scale
            decoded = greedy_decode(probabilities, model.config.midi_min)
            starts = [frame * frontend.hop_sec for _pitch, frame in decoded]
            notes = [
                (int(pitch), round(start, 6),
                 round(max(start + 0.01, starts[k + 1] if k + 1 < len(starts) else start + 0.1), 6))
                for k, ((pitch, _frame), start) in enumerate(zip(decoded, starts))
            ]
            score_path = sample / "verified_score.musicxml"
            index = ScoreEventIndex.from_musicxml(score_path)
            slack_fallback = False
            try:
                result = align_perfect(notes, index.events, score_path, costs)
            except ValueError as error:
                if "length slack" not in str(error):
                    raise
                # The slack only prunes hypotheses; widen it when none survive.
                slack_fallback = True
                result = align_perfect(
                    notes, index.events, score_path, replace(costs, length_slack=10**6)
                )
            counts = Counter(event.relationship for event in result.events)
            counts["missed_note"] = len(result.deletions)
            transcription = {
                "schema_version": f"{SCHEMA_VERSION}-transcription",
                "sample": sample.name,
                "audio_sha256": _sha256(audio_path),
                "transcriber_checkpoint_sha256": transcriber["checkpoint_sha256"],
                "blank_scale": blank_scale,
                "pitch_space": "written",
                "note_end_policy": "next onset; last note onset + 0.1 s (not measured)",
                "normalization": normalization,
                "notes": [
                    {"index": k, "pitch": pitch, "start": start, "end": end}
                    for k, (pitch, start, end) in enumerate(notes)
                ],
                "training_performed": False,
            }
            alignment = {
                "schema_version": f"{SCHEMA_VERSION}-alignment",
                "sample": sample.name,
                "score_sha256": _sha256(score_path),
                "aligner_sha256": aligner["aligner_sha256"],
                "costs": asdict(costs),
                "length_slack_fallback": slack_fallback,
                "score_event_count": len(index.events),
                "transcribed_note_count": len(notes),
                "repeat_hypothesis": {
                    "source_span": list(result.source_span) if result.source_span else None,
                    "extra_copies": result.copies,
                },
                "path_cost": result.cost,
                "missed_score_event_indices": sorted(result.deletions),
                "counts": dict(counts),
                "events": [
                    {
                        "note_index": event.rendered_index,
                        "pitch": event.pitch,
                        "start": event.start,
                        "end": event.end,
                        "relationship": event.relationship,
                        "score_span": list(event.score_span) if event.score_span else None,
                        "copy_pass": event.copy_pass,
                    }
                    for event in result.events
                ],
            }
            _atomic_json(args.output / "transcriptions" / f"{sample.name}.json", transcription)
            _atomic_json(args.output / "alignments" / f"{sample.name}.json", alignment)
            if not args.no_sample_files:
                _atomic_json(sample / TRANSCRIPTION_FILE, transcription)
                _atomic_json(sample / ALIGNMENT_FILE, alignment)
            type_totals.update(counts)
            rows.append({
                "sample": sample.name,
                "notes": len(notes),
                "score_events": len(index.events),
                "extra_copies": result.copies,
                "counts": dict(counts),
                "length_slack_fallback": slack_fallback,
                "seconds": round(time.perf_counter() - started, 2),
            })
            print(f"{position:03d}/{len(samples)} {sample.name}: {len(notes)} notes, "
                  f"{len(index.events)} score events, copies={result.copies}, {dict(counts)}",
                  flush=True)
        except Exception as error:  # noqa: BLE001
            failures.append({"sample": sample.name, "error": f"{type(error).__name__}: {error}"})
            print(f"{position:03d}/{len(samples)} {sample.name}: ERROR {error}", flush=True)

    for sample in samples:
        for name, digest in protected[sample.name].items():
            if _sha256(sample / name) != digest:
                raise RuntimeError(f"{sample.name}/{name} changed during the run")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "samples_root": str(args.samples.resolve()),
        "sample_count": len(samples),
        "succeeded": len(rows),
        "failed": len(failures),
        "failures": failures,
        "transcriber_candidate": str(args.transcriber_candidate.resolve()),
        "transcriber_candidate_sha256": _sha256(args.transcriber_candidate),
        "aligner_candidate": str(args.aligner_candidate.resolve()),
        "aligner_candidate_sha256": _sha256(args.aligner_candidate),
        "per_sample_files": None if args.no_sample_files else [TRANSCRIPTION_FILE, ALIGNMENT_FILE],
        "protected_files_unchanged": ["labels.json", "labels_agent.json"],
        "training_performed": False,
        "totals_by_relationship": dict(type_totals),
        "samples": rows,
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps({key: manifest[key] for key in
                      ("sample_count", "succeeded", "failed", "totals_by_relationship")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
