"""Evaluate audio transcription and note-to-score alignment on frozen test splits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from music21 import converter, note

from alignmodel.note_align_train import load_exact_note_map
from alignmodel.stages.note_align import NoteAligner, ObservedNote, alignment_metrics
from alignmodel.transcription import (
    decode_frozen_basic_pitch,
    evaluate_note_lists,
    extract_sample_basic_pitch_features,
    infer_full_clip,
    infer_note_decoder,
    infer_sample_notes,
    infer_sample_notes_hybrid,
    load_note_decoder,
    load_note_transcriber,
    match_notes,
)
from alignmodel.transcription.data import load_written_notes


def _performed_score_notes(path: Path) -> list[ObservedNote]:
    score = converter.parse(str(path))
    flat = score.flatten()
    rows: list[ObservedNote] = []
    for item in flat.secondsMap:
        element = item.get("element")
        if not isinstance(element, note.Note) or element.duration.isGrace:
            continue
        start = float(item.get("offsetSeconds", 0.0))
        end = max(float(item.get("endTimeSeconds", start)), start + 0.01)
        rows.append(
            ObservedNote(
                pitch=int(element.pitch.midi),
                start=start,
                end=end,
                confidence=1.0,
                source_index=len(rows),
            )
        )
    return sorted(rows, key=lambda value: (value.start, value.pitch))


def _audio_midi_notes(sample: Path) -> list[ObservedNote]:
    return [
        ObservedNote(pitch=pitch, start=start, end=end, source_index=index)
        for index, (pitch, start, end) in enumerate(load_written_notes(sample))
    ]


def _sequence_map(source: list[int], target: list[int]) -> list[int | None]:
    """Monotonic edit map from rendered MIDI events to performance-score notes."""
    n, m = len(source), len(target)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    bt[1:, 0] = 1
    bt[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = (
                (dp[i - 1, j - 1] + (source[i - 1] != target[j - 1]), 0),
                (dp[i - 1, j] + 1, 1),
                (dp[i, j - 1] + 1, 2),
            )
            dp[i, j], bt[i, j] = min(options)
    mapping: list[int | None] = [None] * n
    i, j = n, m
    while i or j:
        code = int(bt[i, j])
        if i and j and code == 0:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif i and (not j or code == 1):
            i -= 1
        else:
            j -= 1
    return mapping


def _cache_path(cache_root: Path, row: dict[str, Any]) -> Path:
    corpus = str(row.get("corpus") or row.get("root") or "")
    sample = Path(row["sample_dir"]).name
    candidates = [
        cache_root / corpus / sample / "note_map.json",
        cache_root / sample / "note_map.json",
        Path(row["sample_dir"]) / "note_map.json",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No note map for {sample}")
    return path


def _mapping_scores(
    pred_map: list[int | None],
    transcribed,
    gold_notes: list[ObservedNote],
    target_map: list[int | None],
    target_copy: list[bool],
    *,
    onset_tolerance: float,
) -> dict[str, float | int]:
    matchable_gold = [
        {"pitch": note.pitch, "start": note.start, "end": note.end}
        for note in gold_notes
    ]
    pairs = match_notes(
        transcribed,
        matchable_gold,
        onset_tolerance_sec=onset_tolerance,
    )
    pred_to_gold = {pred_i: gold_i for pred_i, gold_i in pairs}
    correct = assigned = 0
    correct_all = 0
    copy_correct = 0
    for pred_i, predicted in enumerate(pred_map):
        gold_i = pred_to_gold.get(pred_i)
        if gold_i is None:
            assigned += predicted is not None
            continue
        truth = target_map[gold_i]
        assigned += predicted is not None
        correct += predicted is not None and truth is not None and predicted == truth
        correct_all += predicted == truth
        if target_copy[gold_i] and predicted == truth:
            copy_correct += 1
    positives = sum(value is not None for value in target_map)
    copy_positives = sum(target_copy)
    precision = correct / max(assigned, 1)
    recall = correct / max(positives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_accuracy": correct_all / max(len(target_map), 1),
        "copy_recall": copy_correct / max(copy_positives, 1),
        "n_correct": correct,
        "n_assigned": assigned,
        "n_positive": positives,
        "n_copy_correct": copy_correct,
        "n_copy": copy_positives,
        "n_transcription_matches": len(pairs),
    }


def _dtw_map(sample: Path, gold: list[ObservedNote], n_score: int):
    try:
        from datacreate.note_alignment import build_note_alignment

        events = [
            event
            for event in build_note_alignment(sample).get("events") or []
            if not event.get("is_rest")
        ]
    except Exception:
        return None
    if len(events) != n_score:
        return None
    starts = np.asarray([float(event["perf_start"]) for event in events])
    ends = np.asarray([float(event["perf_end"]) for event in events])
    result: list[int | None] = []
    for value in gold:
        inside = np.flatnonzero(
            (starts <= value.start + 1e-6) & (ends >= value.start - 1e-6)
        )
        result.append(
            int(inside[0])
            if inside.size
            else int(np.argmin(np.abs(starts - value.start)))
        )
    return result


def _aggregate(rows: list[dict[str, Any]], prefix: str) -> dict[str, float | int]:
    selected = [row[prefix] for row in rows if row.get(prefix)]
    if not selected:
        return {"n_clips": 0}
    sums: defaultdict[str, float] = defaultdict(float)
    for row in selected:
        for key, value in row.items():
            if isinstance(value, (int, float)) and value is not None:
                sums[key] += float(value)
    count = len(selected)
    output: dict[str, float | int] = {"n_clips": count}
    count_keys = {
        "n_correct",
        "n_assigned",
        "n_positive",
        "n_copy_correct",
        "n_copy",
        "n_transcription_matches",
        "n_pred",
        "n_target",
        "n_matched",
        "n_onset_aligned",
        "n_semitone_errors",
        "n_octave_errors",
        "n_plus_minus_2",
    }
    for key, value in sums.items():
        output[key] = int(value) if key in count_keys else round(value / count, 5)
    if {"n_correct", "n_assigned", "n_positive"} <= output.keys():
        correct = float(output["n_correct"])
        precision = correct / max(float(output["n_assigned"]), 1)
        recall = correct / max(float(output["n_positive"]), 1)
        output["micro_precision"] = round(precision, 5)
        output["micro_recall"] = round(recall, 5)
        output["micro_f1"] = round(
            2 * precision * recall / max(precision + recall, 1e-12), 5
        )
    if {"n_copy_correct", "n_copy"} <= output.keys():
        output["micro_copy_recall"] = round(
            float(output["n_copy_correct"]) / max(float(output["n_copy"]), 1), 5
        )
    if {"n_matched", "n_pred", "n_target"} <= output.keys():
        matched = float(output["n_matched"])
        precision = matched / max(float(output["n_pred"]), 1)
        recall = matched / max(float(output["n_target"]), 1)
        output["micro_precision"] = round(precision, 5)
        output["micro_recall"] = round(recall, 5)
        output["micro_f1"] = round(
            2 * precision * recall / max(precision + recall, 1e-12), 5
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--transcriber-ckpt", type=Path, default=None)
    parser.add_argument("--aligner-ckpt", type=Path, default=None)
    parser.add_argument(
        "--split", action="append", default=None, choices=("val", "test_id", "test_ood")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--onset-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--pitch-change-frames", type=int, default=None)
    parser.add_argument(
        "--decode",
        default="neural",
        choices=("refiner", "basic_pitch", "neural", "fused", "hybrid", "midi"),
        help="Transcription frontend used for the reported transcription/e2e rows",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    splits = args.split or ["test_id", "test_ood"]
    transcriber_path = args.transcriber_ckpt
    if transcriber_path is None:
        candidates = (
            args.weights / "note_decoder.pt",
            args.weights / "note_decoder.json",
            args.weights / "note_transcriber.pt",
            args.weights / "best.pt",
        )
        transcriber_path = next(
            (path for path in candidates if path.exists()), candidates[-1]
        )
    aligner_path = args.aligner_ckpt or (args.weights / "note_aligner.pt")
    decoder = None
    transcriber = None
    decode_cfg = None
    if args.decode in {"refiner", "basic_pitch"} and transcriber_path.exists():
        decoder = load_note_decoder(transcriber_path, args.device)
    elif args.decode not in {"basic_pitch", "midi"}:
        transcriber, decode_cfg = load_note_transcriber(
            transcriber_path, args.device
        )
        if args.pitch_change_frames is not None:
            decode_cfg.pitch_change_frames = int(args.pitch_change_frames)
    learned = NoteAligner.from_checkpoint(
        aligner_path, device=args.device
    )
    symbolic = NoteAligner()
    tolerance = args.onset_tolerance_ms / 1000.0
    report: dict[str, Any] = {
        "manifest": str(args.manifest),
        "weights": str(args.weights),
        "onset_tolerance_ms": args.onset_tolerance_ms,
        "decode": args.decode,
        "splits": {},
    }

    for split in splits:
        source_rows = list(manifest.get(split) or [])
        if args.max_samples:
            source_rows = source_rows[: args.max_samples]
        rows: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for index, source in enumerate(source_rows, start=1):
            sample = Path(source["sample_dir"])
            try:
                exact = load_exact_note_map(_cache_path(args.cache_root, source))
                gold_notes = exact.notes
                audio_target_map = exact.target_score_indices
                audio_target_copy = exact.target_is_copy
                if args.decode == "midi":
                    transcribed = [
                        {"pitch": pitch, "start": start, "end": end, "confidence": 1.0}
                        for pitch, start, end in load_written_notes(sample)
                    ]
                elif args.decode == "basic_pitch":
                    if decoder is not None and decoder.kind == "basic-pitch":
                        transcribed = infer_note_decoder(decoder, sample)
                    else:
                        transcribed = decode_frozen_basic_pitch(
                            extract_sample_basic_pitch_features(sample)
                        )
                elif args.decode == "refiner":
                    if decoder is None:
                        raise RuntimeError("Refiner decoder was not loaded")
                    transcribed = infer_note_decoder(decoder, sample)
                elif args.decode == "hybrid":
                    assert transcriber is not None and decode_cfg is not None
                    transcribed = infer_sample_notes_hybrid(
                        transcriber,
                        sample,
                        args.device,
                        decode_config=decode_cfg,
                    )
                elif args.decode == "fused":
                    assert transcriber is not None and decode_cfg is not None
                    transcribed = infer_sample_notes(
                        transcriber,
                        sample,
                        args.device,
                        decode_config=decode_cfg,
                        fusion="conservative",
                    )
                else:
                    assert transcriber is not None and decode_cfg is not None
                    transcribed = infer_full_clip(
                        transcriber,
                        sample / "performance_mel.npy",
                        args.device,
                        decode_config=decode_cfg,
                    )
                e2e_result = learned.align(transcribed, exact.score)
                symbolic_e2e_result = symbolic.align(transcribed, exact.score)
                oracle_result = learned.align(exact.notes, exact.score)
                symbolic_result = symbolic.align(exact.notes, exact.score)
                dtw = _dtw_map(sample, gold_notes, len(exact.score.notes))
                note_metrics = evaluate_note_lists(
                    transcribed,
                    [
                        {"pitch": note.pitch, "start": note.start, "end": note.end}
                        for note in gold_notes
                    ],
                    onset_tolerance_sec=tolerance,
                )
                row = {
                    "sample": sample.name,
                    "corpus": source.get("corpus"),
                    "source": source.get("source"),
                    "repeated": source.get("repeated"),
                    "transcription": note_metrics,
                    "e2e": _mapping_scores(
                        e2e_result.mapping,
                        transcribed,
                        gold_notes,
                        audio_target_map,
                        audio_target_copy,
                        onset_tolerance=tolerance,
                    ),
                    "symbolic_e2e": _mapping_scores(
                        symbolic_e2e_result.mapping,
                        transcribed,
                        gold_notes,
                        audio_target_map,
                        audio_target_copy,
                        onset_tolerance=tolerance,
                    ),
                    "learned_oracle_notes": alignment_metrics(
                        oracle_result,
                        exact.target_score_indices,
                        target_is_copy=exact.target_is_copy,
                    ),
                    "symbolic_oracle_notes": alignment_metrics(
                        symbolic_result,
                        exact.target_score_indices,
                        target_is_copy=exact.target_is_copy,
                    ),
                    "dtw": (
                        alignment_metrics(
                            dtw,
                            audio_target_map,
                            target_is_copy=audio_target_copy,
                        )
                        if dtw is not None
                        else None
                    ),
                }
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                skipped.append({"sample": sample.name, "error": str(exc)})
            if index == 1 or index % 50 == 0 or index == len(source_rows):
                print(
                    f"{split} {index}/{len(source_rows)} "
                    f"ok={len(rows)} skipped={len(skipped)}",
                    flush=True,
                )
        report["splits"][split] = {
            "n_requested": len(source_rows),
            "n_evaluated": len(rows),
            "n_skipped": len(skipped),
            "transcription": _aggregate(rows, "transcription"),
            "e2e": _aggregate(rows, "e2e"),
            "symbolic_e2e": _aggregate(rows, "symbolic_e2e"),
            "learned_oracle_notes": _aggregate(rows, "learned_oracle_notes"),
            "symbolic_oracle_notes": _aggregate(rows, "symbolic_oracle_notes"),
            "dtw": _aggregate(rows, "dtw"),
            "samples": rows,
            "skipped": skipped,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "splits"}, indent=2))
    for split, result in report["splits"].items():
        print(split, json.dumps({key: value for key, value in result.items() if key not in {"samples", "skipped"}}, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
