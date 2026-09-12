"""Populate local written-space Basic Pitch 0.4.0 activation caches."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from alignmodel.transcription.basic_pitch import (
    basic_pitch_cache_path,
    extract_sample_basic_pitch_features,
)


def _read_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        splits: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            split = str(row.get("split") or "train")
            splits.setdefault(split, []).append(row)
        return splits
    document = json.loads(text)
    if not isinstance(document, dict):
        raise ValueError("Manifest must be a JSON object or JSONL records")
    return document


def _is_raw2k(row: dict[str, Any]) -> bool:
    for field in ("corpus", "root"):
        raw = row.get(field)
        if raw is None:
            continue
        value = str(raw).strip().replace("\\", "/").rstrip("/")
        if value.casefold() == "raw2k" or value.rsplit("/", 1)[-1].casefold() == "raw2k":
            return True
    return False


def _corpus(row: dict[str, Any]) -> str:
    return str(row.get("corpus") or row.get("root") or "unknown")


def _resolve_sample_dir(
    row: dict[str, Any], roots: dict[str, Any] | list[Any] | None
) -> Path:
    raw = row.get("sample_dir", row.get("path", row.get("sample", row.get("id"))))
    if raw is None:
        raise ValueError(f"Manifest row has no sample path or id: {row}")
    sample = Path(str(raw))
    if sample.is_absolute():
        return sample

    root_hint = row.get("root", row.get("corpus"))
    root: Path | None = None
    if isinstance(roots, dict) and root_hint is not None:
        candidate = roots.get(str(root_hint))
        if candidate is not None:
            root = Path(str(candidate))
    elif isinstance(roots, list) and root_hint is not None:
        hint = str(root_hint)
        if hint.isdigit() and int(hint) < len(roots):
            root = Path(str(roots[int(hint)]))
    if root is None and root_hint is not None:
        hinted = Path(str(root_hint))
        if hinted.is_absolute():
            root = hinted
    if root is None:
        raise ValueError(
            f"Cannot resolve relative sample {raw!r}; add sample_dir or manifest roots"
        )
    return root / sample


def _worker(task: tuple[int, str, str, str]) -> dict[str, Any]:
    index, sample_text, corpus, cache_text = task
    sample = Path(sample_text)
    destination = basic_pitch_cache_path(Path(cache_text), sample, corpus)
    features = extract_sample_basic_pitch_features(sample, cache_path=destination)
    return {
        "index": index,
        "sample": sample.name,
        "corpus": corpus,
        "cache": str(destination),
        "frames": int(features.note.shape[0]),
    }


def _flatten_splits(values: list[list[str]] | None) -> list[str] | None:
    if not values:
        return None
    output: list[str] = []
    for group in values:
        for value in group:
            output.extend(item for item in value.split(",") if item)
    return output


def _default_splits(document: dict[str, Any]) -> list[str]:
    preferred = ("train", "val", "test_id", "test_ood", "test")
    selected = [name for name in preferred if isinstance(document.get(name), list)]
    known = set(selected) | {"roots", "version", "metadata", "policy", "distribution"}
    selected.extend(
        key
        for key, value in document.items()
        if key not in known and isinstance(value, list)
    )
    return selected


def _run_tasks(
    tasks: list[tuple[int, str, str, str]], workers: int
) -> Iterable[tuple[dict[str, Any] | None, str | None]]:
    if workers <= 1:
        for task in tasks:
            try:
                yield _worker(task), None
            except Exception as exc:  # noqa: BLE001
                yield None, str(exc)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, task) for task in tasks]
        for future in futures:
            try:
                yield future.result(), None
            except Exception as exc:  # noqa: BLE001
                yield None, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--split",
        action="append",
        nargs="+",
        default=None,
        help="Split name(s); repeat or use comma-separated values",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum selected rows per split",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--procedural-only",
        action="store_true",
        help="Skip raw2k rows before resolving or accessing their paths",
    )
    args = parser.parse_args()

    document = _read_manifest(args.manifest)
    roots = document.get("roots")
    split_names = _flatten_splits(args.split) or _default_splits(document)
    if not split_names:
        raise ValueError("No list-valued splits found in manifest")

    args.cache_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "frontend": "basic-pitch-0.4.0",
        "manifest": str(args.manifest),
        "cache_root": str(args.cache_root),
        "splits": {},
    }
    failures: list[dict[str, str]] = []
    task_index = 0
    for split in split_names:
        rows = document.get(split)
        if not isinstance(rows, list):
            raise ValueError(f"Manifest split {split!r} is not a list")
        selected: list[dict[str, Any]] = []
        raw2k_skipped = 0
        for value in rows:
            row = dict(value) if isinstance(value, dict) else {"sample": value}
            if args.procedural_only and _is_raw2k(row):
                raw2k_skipped += 1
                continue
            selected.append(row)
        if args.max_samples is not None:
            selected = selected[: max(0, args.max_samples)]

        tasks: list[tuple[int, str, str, str]] = []
        for row in selected:
            sample = _resolve_sample_dir(row, roots)
            tasks.append(
                (task_index, str(sample), _corpus(row), str(args.cache_root))
            )
            task_index += 1

        completed = 0
        split_failures = 0
        for position, (result, error) in enumerate(
            _run_tasks(tasks, max(1, args.workers))
        ):
            if error is None:
                assert result is not None
                completed += 1
                if completed == 1 or completed % 25 == 0 or completed == len(tasks):
                    print(
                        f"{split} {completed + split_failures}/{len(tasks)} "
                        f"cached={completed} failed={split_failures}",
                        flush=True,
                    )
            else:
                split_failures += 1
                sample = Path(tasks[position][1]).name
                failures.append(
                    {"split": split, "sample": sample, "error": error}
                )
                print(f"{split} failed {sample}: {error}", flush=True)
        summary["splits"][split] = {
            "selected": len(tasks),
            "cached": completed,
            "failed": split_failures,
            "raw2k_skipped": raw2k_skipped,
        }

    summary["failures"] = failures
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise RuntimeError(f"Basic Pitch caching failed for {len(failures)} row(s)")


if __name__ == "__main__":
    main()
