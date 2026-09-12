"""Create a frozen, uniformly stratified split for full note-alignment training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED = (
    "performance_mel.npy",
    "performance_audio.mid",
    "verified_score.musicxml",
    "performance_score.musicxml",
    "labels.json",
    "metadata.json",
)


def _stable_fraction(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _load_bundle(path: Path, corpus: str) -> dict[str, Any] | None:
    if not path.is_dir() or any(not (path / name).is_file() for name in REQUIRED):
        return None
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    labels = json.loads((path / "labels.json").read_text(encoding="utf-8"))
    types = sorted(
        {
            str(label.get("type"))
            for label in labels.get("labels") or []
            if label.get("type")
        }
    )
    source = Path(
        str(metadata.get("source_score") or metadata.get("source") or corpus)
    ).stem
    try:
        import numpy as np

        mel = np.load(path / "performance_mel.npy", mmap_mode="r")
        n_frames = int(max(mel.shape))
    except Exception:
        return None
    duration_bucket = min(4, n_frames // 500)
    has_intonation = "intonation_error" in types
    return {
        "sample": path.name,
        "sample_dir": str(path),
        "root": corpus,
        "corpus": corpus,
        "source": source,
        "repeated": bool(metadata.get("repeated", "repetition" in types)),
        "error_types": types,
        "duration_bucket": duration_bucket,
        "has_intonation": has_intonation,
        "audio_render": metadata.get("audio_render") or "default",
        "recording_kind": metadata.get("recording_kind") or "synthetic",
    }


def _stratum(row: dict[str, Any], include_source: bool = True) -> tuple:
    return (
        row["corpus"],
        row["source"] if include_source else "*",
        row["repeated"],
        row["duration_bucket"],
        bool(row.get("has_intonation")),
        row.get("recording_kind", "synthetic"),
        row.get("audio_render", "default"),
    )


def _take_uniform(
    rows: list[dict[str, Any]],
    fraction: float,
    seed: int,
    *,
    include_source: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_stratum: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[_stratum(row, include_source)].append(row)
    selected: list[dict[str, Any]] = []
    remainder: list[dict[str, Any]] = []
    for key, group in sorted(by_stratum.items(), key=lambda item: repr(item[0])):
        ordered = sorted(
            group,
            key=lambda row: (
                _stable_fraction(row["sample_dir"], seed),
                row["sample_dir"],
            ),
        )
        n = int(round(len(ordered) * fraction))
        if fraction > 0 and len(ordered) >= 5:
            n = max(1, n)
        n = min(n, max(0, len(ordered) - 1))
        selected.extend(ordered[:n])
        remainder.extend(ordered[n:])
    return selected, remainder


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts: Counter[str] = Counter()
    for row in rows:
        type_counts.update(row["error_types"])
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "corpus": dict(Counter(row["corpus"] for row in rows)),
        "source": dict(Counter(row["source"] for row in rows)),
        "repeated_fraction": round(
            sum(bool(row["repeated"]) for row in rows) / n, 5
        ),
        "error_type_fraction": {
            key: round(value / n, 5) for key, value in sorted(type_counts.items())
        },
        "duration_bucket": dict(Counter(row["duration_bucket"] for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--procedural-root", type=Path, default=Path("E:/output"))
    parser.add_argument("--raw-root", type=Path, default=Path("E:/output_2k_rawdata"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=365)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--ood-source", default="WeberITAV")
    parser.add_argument(
        "--procedural-only",
        action="store_true",
        help="Use only procedural data; do not inspect or emit raw2k data",
    )
    parser.add_argument(
        "--allow-missing-procedural",
        action="store_true",
        help="Create a raw-only development split when the E: procedural root is offline",
    )
    args = parser.parse_args()

    roots = {"procedural12k": args.procedural_root.resolve()}
    if not args.procedural_only:
        roots["raw2k"] = args.raw_root.resolve()
    missing = [
        str(path)
        for key, path in roots.items()
        if not path.exists()
        and not (key == "procedural12k" and args.allow_missing_procedural)
    ]
    if missing:
        raise FileNotFoundError(f"Dataset roots are not mounted: {missing}")

    procedural = (
        [
            row
            for path in sorted(roots["procedural12k"].iterdir())
            if (row := _load_bundle(path, "procedural12k")) is not None
        ]
        if roots["procedural12k"].exists()
        else []
    )
    raw = (
        [
            row
            for path in sorted(roots["raw2k"].iterdir())
            if (row := _load_bundle(path, "raw2k")) is not None
        ]
        if not args.procedural_only
        else []
    )
    raw_ood = [row for row in raw if row["source"] == args.ood_source]
    raw_trainable = [row for row in raw if row["source"] != args.ood_source]
    if not args.procedural_only and not raw_ood:
        raise ValueError(
            f"No raw bundles found for OOD source {args.ood_source!r}; "
            f"available={sorted({row['source'] for row in raw})}"
        )

    val_proc, proc_after_val = _take_uniform(
        procedural, args.val_fraction, args.seed, include_source=False
    )
    test_id, train_proc = _take_uniform(
        proc_after_val,
        args.test_fraction / max(1.0 - args.val_fraction, 1e-6),
        args.seed + 1,
        include_source=False,
    )
    val_raw, train_raw = _take_uniform(
        raw_trainable, args.val_fraction, args.seed, include_source=True
    )
    train = sorted(train_proc + train_raw, key=lambda row: row["sample_dir"])
    val = sorted(val_proc + val_raw, key=lambda row: row["sample_dir"])
    test_id = sorted(test_id, key=lambda row: row["sample_dir"])
    test_ood = sorted(raw_ood, key=lambda row: row["sample_dir"])

    for split, rows in (
        ("train", train),
        ("val", val),
        ("test_id", test_id),
        ("test_ood", test_ood),
    ):
        for row in rows:
            row["split"] = split

    manifest = {
        "version": 1,
        "seed": args.seed,
        "roots": {key: str(value) for key, value in roots.items()},
        "policy": {
            "uniform_by": [
                "corpus",
                "source",
                "repeated",
                "duration_bucket",
                "has_intonation",
                "recording_kind",
                "audio_render",
            ],
            "checked_distribution": ["error_types"],
            "val_fraction": args.val_fraction,
            "procedural_test_fraction": args.test_fraction,
            "raw_ood_source": None if args.procedural_only else args.ood_source,
            "procedural_only": bool(args.procedural_only),
        },
        "train": train,
        "val": val,
        "test_id": test_id,
        "test_ood": test_ood,
        "distribution": {
            split: _distribution(rows)
            for split, rows in (
                ("train", train),
                ("val", val),
                ("test_id", test_id),
                ("test_ood", test_ood),
            )
        },
        "discovered": {
            "procedural_complete": len(procedural),
            **({"raw_complete": len(raw)} if not args.procedural_only else {}),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["distribution"], indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
