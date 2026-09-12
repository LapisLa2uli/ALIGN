"""Build a strict procedural-only manifest for Basic Pitch refiner training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _corpus(row: dict[str, Any]) -> str:
    return str(row.get("corpus") or row.get("root") or "")


def _note_map_path(cache_root: Path, row: dict[str, Any]) -> Path:
    sample = Path(str(row["sample_dir"])).name
    corpus = _corpus(row)
    candidates = (
        cache_root / corpus / sample / "note_map.json",
        cache_root / sample / "note_map.json",
        Path(str(row["sample_dir"])) / "note_map.json",
    )
    path = next((value for value in candidates if value.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No exact note_map.json for {corpus}/{sample}")
    document = json.loads(path.read_text(encoding="utf-8"))
    rendered = document.get("rendered_notes")
    if not isinstance(rendered, list) or not rendered:
        raise ValueError(f"{path} has no rendered_notes")
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--note-map-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus", default="procedural12k")
    parser.add_argument(
        "--effective-audio-transpose",
        type=int,
        required=True,
        help="Semitones added to detected WAV pitch to recover written pitch",
    )
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    output: dict[str, Any] = {
        "version": 3,
        "seed": source.get("seed", 365),
        "source_manifest": str(args.source_manifest.resolve()),
        "roots": {
            args.corpus: str(
                Path(str((source.get("roots") or {})[args.corpus])).resolve()
            )
        },
        "policy": {
            "procedural_only": True,
            "raw_derived_data_used": False,
            "exact_rendered_note_maps_required": True,
            "audio_pitch_space": "sounding",
            "effective_audio_transpose": args.effective_audio_transpose,
        },
    }
    failures: list[dict[str, str]] = []
    for split in ("train", "val", "test_id"):
        rows = []
        for raw in source.get(split) or []:
            row = dict(raw) if isinstance(raw, dict) else {"sample": raw}
            if _corpus(row) != args.corpus:
                continue
            try:
                note_map = _note_map_path(args.note_map_root, row)
            except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
                failures.append(
                    {
                        "split": split,
                        "sample": Path(str(row.get("sample_dir") or row.get("sample"))).name,
                        "error": str(exc),
                    }
                )
                continue
            row.update(
                {
                    "split": split,
                    "corpus": args.corpus,
                    "root": args.corpus,
                    "note_map": str(note_map),
                    "audio_pitch_space": "sounding",
                    "effective_audio_transpose": args.effective_audio_transpose,
                }
            )
            rows.append(row)
        if not rows:
            raise ValueError(f"Procedural-only {split} split is empty")
        output[split] = rows
    output["test_ood"] = []
    output["excluded"] = {
        "raw_derived_rows": sum(
            _corpus(dict(row) if isinstance(row, dict) else {"sample": row})
            != args.corpus
            for split in ("train", "val", "test_id", "test_ood")
            for row in source.get(split) or []
        ),
        "missing_or_invalid_note_maps": len(failures),
        "failures": failures,
    }
    output["distribution"] = {
        split: {
            "n": len(output[split]),
            "sources": dict(
                Counter(str(row.get("source") or "unknown") for row in output[split])
            ),
            "intonation": sum(
                "intonation_error" in (row.get("error_types") or [])
                for row in output[split]
            ),
            "repeated": sum(bool(row.get("repeated")) for row in output[split]),
        }
        for split in ("train", "val", "test_id")
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps(output["distribution"], indent=2))
    print(json.dumps(output["excluded"], indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
