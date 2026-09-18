"""Compare frozen ORN targets with global monotonic raw-MIDI lineage."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import audit_training_data as audit
from rebuild_monotonic_render_supervision_v3 import _monotonic_rows

from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.packed_data import sha256_file
from alignmodel.training_resources import resource_lease


EXPECTED_RELEASE_SHA256 = (
    "d4baeb90389136fbdd4549cdfb40fb2aceb2a97d899ae903bebffa2df6aefbc0"
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _identity(event: Any) -> tuple[Any, ...]:
    return (
        event.pitch,
        event.score_span,
        "copy" if event.is_copy else event.relationship,
        event.copy_pass,
        event.rendered_index,
    )


def _backward(events: Sequence[Any]) -> int:
    linked = [
        event
        for event in events
        if event.score_span is not None and event.copy_pass == 0
    ]
    return sum(
        right.score_span[0] < left.score_span[0]
        for left, right in zip(linked, linked[1:])
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource-status", type=Path, required=True)
    args = parser.parse_args(argv)
    if sha256_file(args.release_manifest) != EXPECTED_RELEASE_SHA256:
        raise ValueError("Frozen ORN release mismatch")
    release = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    if (
        args.release_manifest.parent / "lockbox" / "LOCKBOX_OPENED.json"
    ).exists():
        raise ValueError("Replacement lockbox is not sealed")
    target_artifact = release["artifacts"]["development_targets"]
    target_path = Path(target_artifact["path"])
    if sha256_file(target_path) != target_artifact["sha256"]:
        raise ValueError("Development target archive mismatch")
    with gzip.open(target_path, "rt", encoding="utf-8") as stream:
        targets = {
            row["sample"]: row
            for row in (json.loads(line) for line in stream if line.strip())
        }
    rows = [
        row
        for split in ("train", "calibration", "open_validation")
        for row in release["splits"]["development"][split]
    ]
    if set(targets) != {row["sample"] for row in rows}:
        raise ValueError("Structural audit population mismatch")
    per_row = []
    totals: Counter[str] = Counter()
    with resource_lease(
        args.resource_status,
        "cpu_validation",
        track="orn-structural-monotonicity-v1",
        command=[str(Path(__file__).resolve()), *map(str, vars(args).values())],
        metadata={"rows": len(rows), "locked_test": False},
    ):
        for position, row in enumerate(rows, 1):
            sample = row["sample"]
            sample_dir = Path(row["sample_dir"])
            note_map_path = sample_dir / "note_map.json"
            if (
                sha256_file(note_map_path)
                != row["source_hashes"]["note_map.json"]
            ):
                raise ValueError(f"Source note-map changed: {sample}")
            original = json.loads(note_map_path.read_text(encoding="utf-8"))
            metadata = json.loads(
                (sample_dir / "metadata.json").read_text(encoding="utf-8")
            )
            midi = audit._midi_events(sample_dir / "performance_audio.mid")
            if midi is None:
                raise ValueError(f"Unreadable MIDI: {sample}")
            shift = audit._inferred_midi_shift(original, midi, metadata)
            corrected_rows, repair = _monotonic_rows(original, midi, shift)
            corrected = dict(original)
            corrected["rendered_notes"] = corrected_rows
            corrected["rendered_note_count"] = len(corrected_rows)
            score_path = sample_dir / "verified_score.musicxml"
            frozen_index = ScoreEventIndex.from_musicxml(
                score_path, targets[sample]["lineage"]
            )
            corrected_index = ScoreEventIndex.from_musicxml(
                score_path, corrected
            )
            frozen_identities = [
                _identity(event) for event in frozen_index.rendered_events
            ]
            corrected_identities = [
                _identity(event) for event in corrected_index.rendered_events
            ]
            disagreements = sum(
                left != right
                for left, right in zip(
                    frozen_identities, corrected_identities
                )
            ) + abs(len(frozen_identities) - len(corrected_identities))
            frozen_backward = _backward(frozen_index.rendered_events)
            corrected_backward = _backward(corrected_index.rendered_events)
            exact_performed_coverage = (
                int(repair["unmapped_performed_notes"]) == 0
                and int(repair["backward_transfers"]) == 0
                and bool(repair["strictly_monotonic_performed_order"])
            )
            summary = {
                "sample": sample,
                "split": targets[sample]["split"],
                "rendered_events": len(frozen_identities),
                "frozen_backward_identity_steps": frozen_backward,
                "corrected_backward_identity_steps": corrected_backward,
                "identity_disagreements": disagreements,
                "frozen_equals_global_monotonic": disagreements == 0,
                "global_monotonic_performed_coverage": exact_performed_coverage,
                "global_repair": repair,
            }
            per_row.append(summary)
            totals["rows"] += 1
            totals["frozen_backward_identity_steps"] += frozen_backward
            totals["corrected_backward_identity_steps"] += corrected_backward
            totals["identity_disagreements"] += disagreements
            totals["rows_changed"] += int(disagreements > 0)
            totals["rows_frozen_backward"] += int(frozen_backward > 0)
            totals["rows_corrected_backward"] += int(corrected_backward > 0)
            totals["global_monotonic_performed_coverage"] += int(
                exact_performed_coverage
            )
            if position == 1 or position % 25 == 0 or position == len(rows):
                print(f"audit={position}/{len(rows)}", flush=True)
    report = {
        "schema_version": "align-orn-structural-monotonicity-audit-v1",
        "release_manifest_sha256": sha256_file(args.release_manifest),
        "population": {
            "splits": ["train", "calibration", "open_validation"],
            "rows": len(rows),
            "lockbox_rows": 0,
        },
        "totals": dict(totals),
        "fractions": {
            "rows_with_frozen_backward_identity": totals[
                "rows_frozen_backward"
            ]
            / max(totals["rows"], 1),
            "rows_changed_by_global_monotonic_rebuild": totals["rows_changed"]
            / max(totals["rows"], 1),
            "rows_with_complete_global_monotonic_performed_coverage": totals[
                "global_monotonic_performed_coverage"
            ]
            / max(totals["rows"], 1),
        },
        "finding": (
            "The frozen target archive used local forward recovery. A globally "
            "monotonic reconstruction changes canonical identities wherever "
            "that recovery attached an audible event to a future performed note "
            "and later resumed backward."
        ),
        "per_row": per_row,
        "lockbox_targets_read": False,
    }
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "totals": dict(totals),
                "fractions": report["fractions"],
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
