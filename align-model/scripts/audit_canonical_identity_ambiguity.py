from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.outputraw_full import verify_data_ready
from alignmodel.joint.packed_data import PackedJointDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    ready = verify_data_ready(args.ready_marker)
    manifest = json.loads(
        Path(str(ready["paths"]["manifest"])).read_text(encoding="utf-8")
    )
    source_root = Path(str(manifest["source_root"]))
    total = ambiguous = ambiguous_single = 0
    rows = []
    with PackedJointDataset(
        Path(str(ready["paths"]["packed_root"])),
        manifest_sha256=str(ready["hashes"]["manifest_sha256"]),
        verify_records=False,
        load_feature_arrays=False,
    ) as dataset:
        ordinals = sorted(
            dataset.ordinals("train"),
            key=lambda value: hashlib.sha256(
                f"{args.seed}:{value}".encode()
            ).digest(),
        )[: args.limit]
        for ordinal in ordinals:
            packed = dataset[ordinal]
            target = packed.training_example().target_events
            score = ScoreEventIndex.from_musicxml(
                source_root / packed.sample / "verified_score.musicxml"
            ).events
            groups = defaultdict(list)
            for event in score:
                groups[
                    (
                        event.pitch,
                        round(event.ql_start, 8),
                        round(event.ql_end, 8),
                        event.part,
                        event.voice,
                    )
                ].append(event.index)
            by_index = {
                index: indices
                for indices in groups.values()
                if len(indices) > 1
                for index in indices
            }
            row_total = row_ambiguous = 0
            for event in target:
                if event.score_span is None:
                    continue
                row_total += 1
                total += 1
                members = {
                    candidate
                    for index in range(*event.score_span)
                    for candidate in by_index.get(index, ())
                }
                if members:
                    ambiguous += 1
                    row_ambiguous += 1
                    if event.score_span[1] - event.score_span[0] == 1:
                        ambiguous_single += 1
            rows.append(
                {
                    "sample": packed.sample,
                    "linked_events": row_total,
                    "ambiguous_events": row_ambiguous,
                }
            )
    report = {
        "schema_version": "align-canonical-identity-ambiguity-audit-v1",
        "rows": len(rows),
        "linked_events": total,
        "ambiguous_events": ambiguous,
        "ambiguous_single_event_targets": ambiguous_single,
        "ambiguous_rate": ambiguous / max(total, 1),
        "definition": (
            "distinct canonical events with identical pitch, score onset/end, "
            "part, and voice"
        ),
        "samples": rows,
        "locked_test_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
