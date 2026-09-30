"""Transcribe and align every DataCreate take with the frozen stack v3 candidate.

Inference only: performance_audio.wav -> dual-mel CTC transcriber v3 (rich
decoding from the candidate) -> robust DP aligner v2 (candidate costs) against
verified_score.musicxml. Outputs go to a run directory and to new per-sample
files (transcription_ctc_v3.json, note_alignment_dp_v2.json). labels.json and
labels_agent.json are never modified.

The CTC head gives one onset frame per note; note ends are the next kept
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.robust_dp_aligner_v2 import RobustDPCosts, align_robust
from alignmodel.transcription.ctc_decode_v2 import rich_decode
from alignmodel.transcription.mel_ctc_v3 import extract_dual_mel, infer_dual_outputs, load_dual_checkpoint
from alignmodel.transcription.mel_v1 import load_audio_mono


SCHEMA_VERSION = "align-datacreate-stack-v3"
TRANSCRIPTION_FILE = "transcription_ctc_v3.json"
ALIGNMENT_FILE = "note_alignment_dp_v2.json"


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
    parser.add_argument("--candidate", type=Path,
                        default=align / "runs/realistic92-stack-v2/CANDIDATE_STACK_V3.json")
    parser.add_argument("--no-sample-files", action="store_true")
    args = parser.parse_args()

    manifest_path = args.output / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    for relative, digest in candidate["code_sha256"].items():
        if _sha256(align / relative) != digest:
            raise ValueError(f"Code changed after candidate freeze: {relative}")
    checkpoint = Path(candidate["checkpoint"])
    if _sha256(checkpoint) != candidate["checkpoint_sha256"]:
        raise ValueError("Transcriber checkpoint changed")
    costs = RobustDPCosts(**candidate["aligner_costs"])
    decoder = dict(candidate["decoder"])

    samples = sorted(
        path for path in args.samples.iterdir()
        if path.is_dir()
        and (path / "performance_audio.wav").is_file()
        and (path / "verified_score.musicxml").is_file()
    )
    protected = {
        sample.name: {name: _sha256(sample / name)
                      for name in ("labels.json", "labels_agent.json") if (sample / name).is_file()}
        for sample in samples
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _payload = load_dual_checkpoint(checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hop = 256 / 22050

    rows, failures = [], []
    totals: Counter[str] = Counter()
    for position, sample in enumerate(samples, 1):
        started = time.perf_counter()
        try:
            audio_path = sample / "performance_audio.wav"
            audio = load_audio_mono(audio_path, 22050)
            mel, normalization = extract_dual_mel(audio, device)
            outputs = infer_dual_outputs(model, np.asarray(mel, np.float32), device)
            decoded = rich_decode(outputs["ctc"], model.config.midi_min, **decoder)
            starts = [note["frame"] * hop for note in decoded]
            raw = [
                (note["pitch"], round(start, 6),
                 round(max(start + 0.01, starts[k + 1] if k + 1 < len(starts) else start + 0.1), 6),
                 round(note["confidence"], 4), int(note["optional"]),
                 note["alternative_pitch"], round(note["alternative_confidence"], 4))
                for k, (note, start) in enumerate(zip(decoded, starts))
            ]
            score_path = sample / "verified_score.musicxml"
            index = ScoreEventIndex.from_musicxml(score_path)
            result = align_robust(raw, index.events, score_path, costs)
            kept = [raw[i] for i in result.kept_note_indices]
            counts = Counter(event.relationship for event in result.events)
            counts["missed_note"] = len(result.deletions)
            primary = [note for note in raw if not note[4]]
            transcription = {
                "schema_version": f"{SCHEMA_VERSION}-transcription",
                "sample": sample.name,
                "audio_sha256": _sha256(audio_path),
                "candidate_sha256": _sha256(args.candidate),
                "decoder": decoder,
                "pitch_space": "written",
                "note_end_policy": "next onset; last note onset + 0.1 s (not measured)",
                "normalization": normalization,
                "notes": [
                    {"index": k, "pitch": p, "start": s, "end": e, "confidence": c,
                     "alternative_pitch": ap, "alternative_confidence": ac}
                    for k, (p, s, e, c, _o, ap, ac) in enumerate(primary)
                ],
                "optional_candidates": [
                    {"pitch": p, "start": s, "confidence": c}
                    for p, s, _e, c, o, _ap, _ac in raw if o
                ],
                "training_performed": False,
            }
            alignment = {
                "schema_version": f"{SCHEMA_VERSION}-alignment",
                "sample": sample.name,
                "score_sha256": _sha256(score_path),
                "aligner": "robust_dp_aligner_v2",
                "costs": candidate["aligner_costs"],
                "score_event_count": len(index.events),
                "transcribed_note_count": len(primary),
                "optional_candidate_count": len(raw) - len(primary),
                "kept_note_count": len(kept),
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
                        "from_optional_candidate": bool(kept[event.rendered_index][4]),
                    }
                    for event in result.events
                ],
            }
            _atomic_json(args.output / "transcriptions" / f"{sample.name}.json", transcription)
            _atomic_json(args.output / "alignments" / f"{sample.name}.json", alignment)
            if not args.no_sample_files:
                _atomic_json(sample / TRANSCRIPTION_FILE, transcription)
                _atomic_json(sample / ALIGNMENT_FILE, alignment)
            totals.update(counts)
            match_like = counts.get("match", 0) + counts.get("copy", 0)
            rows.append({
                "sample": sample.name,
                "transcribed_notes": len(primary),
                "kept_notes": len(kept),
                "score_events": len(index.events),
                "extra_copies": result.copies,
                "counts": dict(counts),
                "match_or_copy_fraction": match_like / max(len(kept), 1),
                "seconds": round(time.perf_counter() - started, 2),
            })
            print(f"{position:03d}/{len(samples)} {sample.name}: {len(primary)} notes "
                  f"(+{len(raw) - len(primary)} optional), kept {len(kept)}, "
                  f"{len(index.events)} score events, copies={result.copies}, {dict(counts)}", flush=True)
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
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": _sha256(args.candidate),
        "sample_count": len(samples),
        "succeeded": len(rows),
        "failed": len(failures),
        "failures": failures,
        "per_sample_files": None if args.no_sample_files else [TRANSCRIPTION_FILE, ALIGNMENT_FILE],
        "protected_files_unchanged": ["labels.json", "labels_agent.json"],
        "training_performed": False,
        "totals_by_relationship": dict(totals),
        "low_agreement_samples": [
            row["sample"] for row in rows if row["match_or_copy_fraction"] < 0.6
        ],
        "samples": rows,
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps({key: manifest[key] for key in
                      ("sample_count", "succeeded", "failed", "totals_by_relationship", "low_agreement_samples")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
