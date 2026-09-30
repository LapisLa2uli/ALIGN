"""Score-agnostic transcription accuracy on realistic_10k (dataset 9.2).

Compares the two current candidate frontends on written-pitch rendered gold:
  - frozen Basic Pitch 0.4.0 (FROZEN_DECODE_CONFIG)
  - Mel Transcriber v1 candidate-epoch-018

Headline metric: pitch-sequence LCS F1. Onset-tolerance 50 ms is diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from alignmodel.joint.index import JointEvent
from alignmodel.joint.metrics import pair_exact_pitch_onset
from alignmodel.joint.packed_data import sha256_file
from alignmodel.transcription.basic_pitch import (
    BASIC_PITCH_VERSION,
    FROZEN_DECODE_CONFIG,
    basic_pitch_cache_path,
    decode_frozen_basic_pitch,
    extract_sample_basic_pitch_features,
)
from alignmodel.transcription.mel_v1 import (
    SCHEMA_VERSION as MEL_SCHEMA,
    decode_mel_notes,
    extract_log_mel,
    infer_mel_probabilities,
    load_audio_mono,
    load_mel_checkpoint,
)


DEFAULT_ROOT = Path(r"E:\outputRaw_realistic_10k")
DEFAULT_CHECKPOINT = Path(
    "runs/joint-outputraw-full-v1/mel-transcriber-v1/"
    "full-training-all4544-v2/candidate-epoch-018.pt"
)
DEFAULT_OUTPUT = Path(
    "runs/joint-outputraw-full-v1/realistic-10k-transcriber-eval-v1"
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


def _lcs(left: Sequence[int], right: Sequence[int]) -> int:
    row = [0] * (len(right) + 1)
    for item in left:
        previous = 0
        for column, other in enumerate(right, 1):
            saved = row[column]
            row[column] = (
                previous + 1
                if item == other
                else max(row[column], row[column - 1])
            )
            previous = saved
    return row[-1]


def _prf(correct: int, predicted: int, gold: int) -> dict[str, Any]:
    precision = correct / max(predicted, 1)
    recall = correct / max(gold, 1)
    return {
        "correct": correct,
        "predicted": predicted,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "count_ratio": predicted / max(gold, 1),
    }


def _event(note: Any, index: int) -> JointEvent:
    return JointEvent(
        pitch=int(note.pitch if hasattr(note, "pitch") else note["pitch"]),
        start=float(
            note.start
            if hasattr(note, "start")
            else note.get("start", note.get("start_sec"))
        ),
        end=float(
            note.end
            if hasattr(note, "end")
            else note.get("end", note.get("end_sec"))
        ),
        score_span=None,
        relationship="extra",
        rendered_index=index,
        confidence=float(
            note.confidence
            if hasattr(note, "confidence")
            else note.get("confidence", 1.0)
        ),
    )


def _load_gold(sample: Path) -> list[dict[str, Any]]:
    document = json.loads((sample / "note_map.json").read_text(encoding="utf-8"))
    rendered = document.get("rendered_notes")
    if not isinstance(rendered, list) or not rendered:
        raise ValueError(f"Missing rendered_notes: {sample}")
    return [
        {
            "pitch": int(row["pitch_midi_written"]),
            "start": float(row["start_sec"]),
            "end": float(row["end_sec"]),
            "relationship": str(row.get("relationship") or "match"),
        }
        for row in rendered
    ]


def _stratum(metadata: Mapping[str, Any]) -> str:
    error = str(metadata.get("error_type") or "unknown")
    degrade = metadata.get("audio_degrade")
    if degrade is None and isinstance(metadata.get("degrade"), dict):
        degrade = metadata["degrade"].get("version")
    return f"{error}|{degrade or 'none'}"


def _list_samples(root: Path) -> list[Path]:
    print(f"listing={root}", flush=True)
    samples = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("synth_")
    )
    print(f"listed={len(samples)}", flush=True)
    return samples


def _stratified_sample(
    samples: Sequence[Path],
    *,
    limit: int,
    seed: int,
) -> list[Path]:
    """Seeded round-robin strata without reading every metadata file twice.

    Reads metadata only for candidates until each stratum quota is filled, using
    a shuffled walk over the corpus. Falls back to plain seeded shuffle if the
    limit is large relative to the corpus.
    """

    if limit <= 0 or limit >= len(samples):
        return list(samples)
    rng = random.Random(seed)
    order = list(samples)
    rng.shuffle(order)
    # Cap metadata probes so selection stays fast on 10k corpora.
    probe_budget = min(len(order), max(limit * 8, limit * 2))
    buckets: dict[str, list[Path]] = defaultdict(list)
    for sample in order[:probe_budget]:
        metadata_path = sample / "metadata.json"
        if not metadata_path.is_file():
            continue
        if not (sample / "performance_audio.wav").is_file():
            continue
        if not (sample / "note_map.json").is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        buckets[_stratum(metadata)].append(sample)
    selected: list[Path] = []
    keys = sorted(buckets)
    while len(selected) < limit and any(buckets[key] for key in keys):
        for key in keys:
            if len(selected) >= limit:
                break
            if buckets[key]:
                selected.append(buckets[key].pop())
    if len(selected) < limit:
        chosen = {path.name for path in selected}
        for sample in order:
            if len(selected) >= limit:
                break
            if sample.name in chosen:
                continue
            if not (sample / "performance_audio.wav").is_file():
                continue
            if not (sample / "note_map.json").is_file():
                continue
            selected.append(sample)
    print(f"selected={len(selected)} probed={probe_budget}", flush=True)
    return sorted(selected, key=lambda path: path.name)


def _score_predictions(
    predictions: Mapping[str, list[Any]],
    gold_by_sample: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    sequence = [0, 0, 0]
    onset = [0, 0, 0]
    duration = {
        "lt80ms": [0, 0],
        "80to120ms": [0, 0],
        "120to180ms": [0, 0],
        "ge180ms": [0, 0],
    }
    per_sample = []
    for sample, notes in sorted(predictions.items()):
        gold = gold_by_sample[sample]
        pred_pitch = [int(note.pitch) for note in notes]
        gold_pitch = [int(row["pitch"]) for row in gold]
        matched = _lcs(pred_pitch, gold_pitch)
        sequence[0] += matched
        sequence[1] += len(pred_pitch)
        sequence[2] += len(gold_pitch)
        pred_events = tuple(
            _event(note, index) for index, note in enumerate(notes)
        )
        gold_events = tuple(
            _event(row, index) for index, row in enumerate(gold)
        )
        pairs = pair_exact_pitch_onset(
            pred_events, gold_events, tolerance_sec=0.050
        )
        onset[0] += len(pairs)
        onset[1] += len(pred_events)
        onset[2] += len(gold_events)
        paired_gold = {right for _, right in pairs}
        for label, predicate in (
            ("lt80ms", lambda value: value < 0.080),
            ("80to120ms", lambda value: 0.080 <= value < 0.120),
            ("120to180ms", lambda value: 0.120 <= value < 0.180),
            ("ge180ms", lambda value: value >= 0.180),
        ):
            indices = {
                index
                for index, row in enumerate(gold)
                if predicate(float(row["end"]) - float(row["start"]))
            }
            duration[label][0] += len(indices & paired_gold)
            duration[label][1] += len(indices)
        sample_lcs = _prf(matched, len(pred_pitch), len(gold_pitch))
        per_sample.append(
            {
                "sample": sample,
                "pitch_sequence_lcs": sample_lcs,
                "onset_50ms": _prf(
                    len(pairs), len(pred_events), len(gold_events)
                ),
            }
        )
    return {
        "pitch_sequence_lcs": _prf(*sequence),
        "onset_50ms_diagnostic_only": _prf(*onset),
        "short_note_recall_onset50ms": {
            key: {
                "matched": value[0],
                "support": value[1],
                "recall": value[0] / max(value[1], 1),
            }
            for key, value in duration.items()
        },
        "per_sample": per_sample,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--skip-basic-pitch", action="store_true")
    parser.add_argument("--skip-mel", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = (
            Path(__file__).resolve().parents[1] / checkpoint
        ).resolve()
    output = args.output
    if not output.is_absolute():
        output = (Path(__file__).resolve().parents[1] / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_root or (output / "basic-pitch-cache")

    all_samples = _list_samples(root)
    selected = _stratified_sample(
        all_samples, limit=args.limit, seed=args.seed
    )
    selection = {
        "dataset_version": "9.2_realistic_10k_degraded",
        "root": str(root),
        "population_available": len(all_samples),
        "selected": len(selected),
        "seed": args.seed,
        "samples": [path.name for path in selected],
    }
    _atomic_json(output / "selection.json", selection)

    gold_by_sample = {
        sample.name: _load_gold(sample) for sample in selected
    }
    strata = defaultdict(int)
    for sample in selected:
        metadata = json.loads(
            (sample / "metadata.json").read_text(encoding="utf-8")
        )
        strata[_stratum(metadata)] += 1

    report: dict[str, Any] = {
        "schema_version": "align-realistic-transcriber-eval-v1",
        "dataset": selection,
        "strata": dict(sorted(strata.items())),
        "gold": {
            "source": "note_map.json rendered_notes.pitch_midi_written",
            "pitch_space": "written",
        },
        "metric": {
            "headline": "score_agnostic_pitch_sequence_lcs",
            "onset_tolerance_sec": 0.050,
            "onset_role": "diagnostic_only",
        },
        "candidates": {},
    }

    if not args.skip_mel:
        device = torch.device(args.device)
        model, frontend, decode, _payload = load_mel_checkpoint(
            checkpoint, device
        )
        predictions: dict[str, list[Any]] = {}
        started = time.perf_counter()
        for position, sample in enumerate(selected, 1):
            audio = load_audio_mono(
                sample / "performance_audio.wav", frontend.sample_rate
            )
            mel, _ = extract_log_mel(audio, frontend, device=device)
            probabilities = infer_mel_probabilities(model, mel, device)
            predictions[sample.name] = decode_mel_notes(
                probabilities,
                midi_min=model.config.midi_min,
                hop_sec=frontend.hop_sec,
                config=decode,
            )
            if (
                position == 1
                or position % 25 == 0
                or position == len(selected)
            ):
                elapsed = time.perf_counter() - started
                rate = position / max(elapsed, 1e-9)
                print(
                    f"mel={position}/{len(selected)} "
                    f"rows_per_sec={rate:.3f} "
                    f"eta_sec={(len(selected) - position) / max(rate, 1e-9):.1f}",
                    flush=True,
                )
        scored = _score_predictions(predictions, gold_by_sample)
        report["candidates"]["mel_transcriber_v1_epoch018"] = {
            "name": "mel_transcriber_v1_candidate_epoch_018",
            "schema": MEL_SCHEMA,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "decode_config": decode.to_dict(),
            "elapsed_sec": time.perf_counter() - started,
            **{
                key: value
                for key, value in scored.items()
                if key != "per_sample"
            },
        }
        _atomic_json(
            output / "mel_per_sample.json",
            {"rows": scored["per_sample"]},
        )
        # Persist partial report so mel results survive a later BP interruption.
        _atomic_json(output / "REPORT.json", report)
        print(
            json.dumps(
                {
                    "candidate": "mel_transcriber_v1_epoch018",
                    "pitch_sequence_lcs": scored["pitch_sequence_lcs"],
                    "onset_50ms": scored["onset_50ms_diagnostic_only"],
                },
                indent=2,
            ),
            flush=True,
        )

    if not args.skip_basic_pitch:
        predictions = {}
        started = time.perf_counter()
        for position, sample in enumerate(selected, 1):
            cache_path = basic_pitch_cache_path(
                cache_root, sample, "realistic-10k"
            )
            features = extract_sample_basic_pitch_features(
                sample, cache_path=cache_path
            )
            predictions[sample.name] = decode_frozen_basic_pitch(features)
            if (
                position == 1
                or position % 10 == 0
                or position == len(selected)
            ):
                elapsed = time.perf_counter() - started
                rate = position / max(elapsed, 1e-9)
                print(
                    f"basic_pitch={position}/{len(selected)} "
                    f"rows_per_sec={rate:.3f} "
                    f"eta_sec={(len(selected) - position) / max(rate, 1e-9):.1f}",
                    flush=True,
                )
        scored = _score_predictions(predictions, gold_by_sample)
        report["candidates"]["frozen_basic_pitch_0_4_0"] = {
            "name": "frozen_basic_pitch_0.4.0",
            "basic_pitch_version": BASIC_PITCH_VERSION,
            "decode_config": {
                "onset_threshold": FROZEN_DECODE_CONFIG.onset_threshold,
                "frame_threshold": FROZEN_DECODE_CONFIG.frame_threshold,
                "minimum_note_length_ms": (
                    FROZEN_DECODE_CONFIG.minimum_note_length_ms
                ),
            },
            "elapsed_sec": time.perf_counter() - started,
            **{
                key: value
                for key, value in scored.items()
                if key != "per_sample"
            },
        }
        _atomic_json(
            output / "basic_pitch_per_sample.json",
            {"rows": scored["per_sample"]},
        )
        print(
            json.dumps(
                {
                    "candidate": "frozen_basic_pitch_0.4.0",
                    "pitch_sequence_lcs": scored["pitch_sequence_lcs"],
                    "onset_50ms": scored["onset_50ms_diagnostic_only"],
                },
                indent=2,
            ),
            flush=True,
        )

    summary = {
        name: {
            "pitch_sequence_lcs_f1": candidate["pitch_sequence_lcs"]["f1"],
            "pitch_sequence_precision": candidate["pitch_sequence_lcs"][
                "precision"
            ],
            "pitch_sequence_recall": candidate["pitch_sequence_lcs"]["recall"],
            "count_ratio": candidate["pitch_sequence_lcs"]["count_ratio"],
            "onset_50ms_f1_diagnostic": candidate[
                "onset_50ms_diagnostic_only"
            ]["f1"],
            "short_lt80ms_recall": candidate["short_note_recall_onset50ms"][
                "lt80ms"
            ]["recall"],
        }
        for name, candidate in report["candidates"].items()
    }
    report["summary"] = summary
    _atomic_json(output / "REPORT.json", report)
    print(json.dumps({"output": str(output), "summary": summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
