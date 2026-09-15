"""Write independent agent annotations from saved Basic Pitch transcriptions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datacreate.transcription_labeling import (
    AGENT_LABEL_FILENAME,
    MAX_LABELS_PER_TYPE,
    write_agent_labels,
)
from datacreate.validation import validate_labels_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument(
        "--maximum-per-type",
        type=int,
        default=MAX_LABELS_PER_TYPE,
        help="Discard every label of a type when its per-sample count exceeds this.",
    )
    args = parser.parse_args()
    if args.maximum_per_type < 0:
        raise ValueError("--maximum-per-type must be non-negative")

    sample_dirs = sorted(
        path
        for path in args.samples.iterdir()
        if path.is_dir()
        and (path / "performance_audio.wav").is_file()
        and (path / "verified_score.musicxml").is_file()
    )
    totals: Counter[str] = Counter()
    dismissed: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    for index, sample_dir in enumerate(sample_dirs, 1):
        try:
            if not (sample_dir / "transcription_notes.json").is_file():
                raise FileNotFoundError(
                    sample_dir / "transcription_notes.json"
                )
            output = write_agent_labels(
                sample_dir, maximum_per_type=args.maximum_per_type
            )
            errors = validate_labels_file(output)
            if errors:
                raise ValueError("; ".join(errors))
            document = json.loads(output.read_text(encoding="utf-8"))
            totals.update(label["type"] for label in document["labels"])
            dismissed.update(document["agent_labeling"]["dismissed_types"])
            print(
                f"{index:02d}/{len(sample_dirs)} {sample_dir.name}: "
                f"{len(document['labels'])} kept, "
                f"dismissed={document['agent_labeling']['dismissed_types']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - report all sample failures
            failures.append({"sample": sample_dir.name, "error": str(exc)})
            print(
                f"{index:02d}/{len(sample_dirs)} {sample_dir.name}: ERROR {exc}",
                flush=True,
            )

    report = {
        "sample_count": len(sample_dirs),
        "succeeded": len(sample_dirs) - len(failures),
        "failed": len(failures),
        "output_filename": AGENT_LABEL_FILENAME,
        "maximum_per_type": args.maximum_per_type,
        "kept_labels_by_type": dict(sorted(totals.items())),
        "samples_dismissing_type": dict(sorted(dismissed.items())),
        "failures": failures,
    }
    report_path = args.samples / "agent_labeling_report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
