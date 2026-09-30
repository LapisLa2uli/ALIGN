"""Relabel DataCreate samples from stored alignment artifacts without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import eval_datacreate_current as base
from datacreate.transcription_labeling import (
    build_agent_label_document_from_note_alignment,
)
from datacreate.validation import validate_labels_file


TRACKED_INPUTS = (
    "note_alignment_v2.json",
    "note_alignment_mel_v1.json",
    "transcription_mel_v1.json",
    "performance_audio.wav",
    "verified_score.musicxml",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--id-from", type=int, default=95)
    parser.add_argument("--id-to", type=int, default=124)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.output / "REPORT.json").exists():
        raise FileExistsError(args.output / "REPORT.json")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    totals: Counter[str] = Counter()
    for value in range(args.id_from, args.id_to + 1):
        sample_id = f"{value:03d}"
        sample = args.samples / sample_id
        primary = sample / "labels.json"
        agent = sample / "labels_agent.json"
        before = {
            name: _sha(sample / name)
            for name in TRACKED_INPUTS
            if (sample / name).is_file()
        }
        backup_dir = args.output / "backup" / sample_id
        backup_dir.mkdir(parents=True, exist_ok=True)
        for path in (primary, agent):
            if path.is_file():
                shutil.copy2(path, backup_dir / path.name)

        alignment = sample / "note_alignment_v2.json"
        if alignment.is_file():
            document = build_agent_label_document_from_note_alignment(
                sample, maximum_per_type=10000
            )
            source = "note_alignment_v2.json"
        else:
            fallback = agent if agent.is_file() else primary
            document = json.loads(fallback.read_text(encoding="utf-8"))
            source = (
                "note_alignment_mel_v1.json"
                if (sample / "note_alignment_mel_v1.json").is_file()
                else fallback.name
            )
        base._atomic_json(agent, document)
        errors = validate_labels_file(agent)
        if errors:
            raise ValueError(f"{sample_id}: {'; '.join(errors)}")
        after = {
            name: _sha(sample / name)
            for name in TRACKED_INPUTS
            if (sample / name).is_file()
        }
        if before != after:
            raise RuntimeError(
                f"{sample_id}: alignment/transcription input changed"
            )
        labels = document.get("labels") or []
        totals.update(str(label.get("type")) for label in labels)
        rows.append(
            {
                "sample": sample_id,
                "alignment_source": source,
                "label_count": len(labels),
                "labels_agent_sha256": _sha(agent),
                "tracked_input_hashes": after,
                "transcription_rerun": False,
                "alignment_rerun": False,
            }
        )
        print(
            f"{len(rows):02d}/{args.id_to-args.id_from+1} "
            f"{sample_id}: source={source} labels={len(labels)}",
            flush=True,
        )
    report = {
        "schema_version": "align-datacreate-current-alignment-relabel-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(rows),
        "transcription_rerun": False,
        "alignment_rerun": False,
        "counts_by_type": dict(sorted(totals.items())),
        "rows": rows,
    }
    base._atomic_json(args.output / "REPORT.json", report)
    print(json.dumps(report | {"rows": f"{len(rows)} rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
