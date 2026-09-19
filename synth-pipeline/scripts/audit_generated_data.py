"""Audit generated bundles without changing scores, audio, labels, or maps."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import soundfile as sf

from datacreate.alignment_storage import alignment_storage_kind
from synthpipeline.pipeline import sample_is_complete
from synthpipeline.timing import midi_note_times


def audit(root: Path, *, verify_crc: bool = False) -> dict:
    rows = []
    partial = []
    failures = []
    for sample in sorted(root.glob("synth_*")):
        if not sample.is_dir():
            continue
        if not sample_is_complete(sample):
            partial.append(sample.name)
            continue
        try:
            metadata = json.loads((sample / "metadata.json").read_text())
            labels = json.loads((sample / "labels.json").read_text())
            mapping = json.loads((sample / "note_map.json").read_text())
            if metadata["schema_version"] != "1.2" or labels["schema_version"] != "1.2":
                raise ValueError("Annotation schema is not 1.2")
            clean_count = mapping["clean_note_count"]
            if clean_count != len(mapping["clean_notes"]):
                raise ValueError("Inconsistent clean-note count")
            rendered = mapping["rendered_notes"]
            if not rendered or len(rendered) != mapping["rendered_note_count"]:
                raise ValueError("Inconsistent rendered-note count")
            midi = midi_note_times(sample / "performance_audio.mid")
            if len(midi) != len(rendered):
                raise ValueError("Rendered-note count differs from the actual MIDI")
            for row, (pitch, start, end) in zip(rendered, midi):
                if row["pitch_midi_sounding"] != pitch or abs(row["start_sec"] - start) > 1e-8 or abs(row["end_sec"] - end) > 1e-8:
                    raise ValueError("Rendered-note pitch or timing differs from the actual MIDI")
            for note in rendered:
                if not 0 <= note["start_sec"] < note["end_sec"]:
                    raise ValueError("Invalid rendered-note interval")
                if any(not 0 <= i < clean_count for i in note["clean_indices"]):
                    raise ValueError("Rendered note references unknown clean note")
                if note["pitch_midi_written"] - note["pitch_midi_sounding"] != 2:
                    raise ValueError("Unexpected clarinet pitch convention")
            # Standalone repetition may be appended to this metadata list,
            # independently of the 1--8 planted content errors.
            content_errors = [t for t in metadata["error_types"] if t != "repetition"]
            if not 1 <= len(content_errors) <= 8:
                raise ValueError("Planted error count is outside 1--8")
            info = {}
            for name in ("performance", "reference"):
                info[name] = sf.info(sample / f"{name}_audio.wav")
                if info[name].samplerate != 22050 or info[name].channels != 1 or info[name].frames <= 0:
                    raise ValueError(f"Invalid {name} WAV format")
                mel = np.load(sample / f"{name}_mel.npy", mmap_mode="r")
                if mel.ndim != 2 or mel.shape[0] != 128 or mel.shape[1] == 0 or not np.isfinite(mel).all():
                    raise ValueError(f"Invalid {name} mel features")
            with np.load(sample / "alignment.npz") as alignment:
                storage_kind = alignment_storage_kind(alignment)
                path = alignment["warping_path"]
                if path.ndim != 2 or path.shape[1] != 2 or len(path) == 0:
                    raise ValueError("Invalid warping path")
            if verify_crc:
                with zipfile.ZipFile(sample / "alignment.npz") as archive:
                    bad = archive.testzip()
                    if bad:
                        raise ValueError(f"Alignment CRC failure: {bad}")
            rows.append({
                "sample": sample.name, "seed": metadata["seed"],
                "source": metadata["source"],
                "performance_seconds": info["performance"].duration,
                "bytes": sum(p.stat().st_size for p in sample.rglob("*") if p.is_file()),
                "error_types": metadata["error_types"], "repeated": metadata["repeated"],
                "alignment_storage": storage_kind,
                "unmapped_performed_notes": mapping.get("render_validation", {}).get("unmapped_performed_notes", 0),
            })
        except Exception as exc:
            failures.append({"sample": sample.name, "error": str(exc)})
    size = sum(r["bytes"] for r in rows)
    return {
        "root": str(root.resolve()), "audited_at": time.time(),
        "n_complete_audited": len(rows), "n_partial": len(partial),
        "partial": partial, "failures": failures,
        "GB": size / 1e9, "GiB": size / 1024**3,
        "performance_hours": sum(r["performance_seconds"] for r in rows) / 3600,
        "sources": dict(Counter(r["source"] for r in rows)),
        "alignment_storage": dict(Counter(r["alignment_storage"] for r in rows)),
        "planted_error_counts": dict(Counter(t for r in rows for t in r["error_types"])),
        "repeated_samples": sum(r["repeated"] for r in rows),
        "samples_with_mapping_warnings": [r["sample"] for r in rows if r["unmapped_performed_notes"]],
        "note": "Mapping warnings require a supervision audit before selecting training data. This checks artifact integrity, not all semantic labels or acoustic error magnitudes.",
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-crc", action="store_true")
    args = parser.parse_args()
    result = {str(root): audit(root, verify_crc=args.verify_crc) for root in args.root}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for name, row in result.items():
        print(name, json.dumps({k: row[k] for k in (
            "n_complete_audited", "n_partial", "GB", "performance_hours"
        )}), "failures:", len(row["failures"]),
              "mapping warnings:", len(row["samples_with_mapping_warnings"]))
    if any(row["failures"] for row in result.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
