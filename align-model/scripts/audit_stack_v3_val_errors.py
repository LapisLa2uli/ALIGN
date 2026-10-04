"""Error audit of the frozen v3 stack on synthetic validation clips.

Runs the frozen transcriber and robust aligner. The sealed test populations
are not read. Gold for transcription is the repaired rendered pitch sequence
when it matches note_map written pitch, which is the training target.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alignmodel.joint.metrics import (
    JointMetricSample,
    _official_note_wise_report,
    evaluate_joint_dataset,
)
from alignmodel.joint.robust_dp_aligner_v2 import RobustDPCosts, align_robust
from alignmodel.transcription.ctc_decode_v2 import lcs_pairs, rich_decode
from alignmodel.transcription.mel_ctc_v3 import extract_dual_mel, infer_dual_outputs, load_dual_checkpoint
from alignmodel.transcription.mel_v1 import load_audio_mono
from realistic92_aligner_common import load_clip, metric_sample


HOP = 256 / 22050
DURATION_BINS = (
    (0.0, 0.05, "dur_lt50"),
    (0.05, 0.08, "dur_50to80"),
    (0.08, 0.12, "dur_80to120"),
    (0.12, 0.20, "dur_120to200"),
    (0.20, 1e9, "dur_ge200"),
)
REGISTER_BINS = (
    (0, 67, "reg_low_lt_G4"),
    (67, 71, "reg_throat_G4_Bb4"),
    (71, 85, "reg_clarion_B4_C6"),
    (85, 200, "reg_high_gt_C6"),
)


def _duration_bin(duration: float) -> str:
    for lo, hi, label in DURATION_BINS:
        if lo <= duration < hi:
            return label
    return "dur_ge200"


def _register_bin(pitch: int) -> str:
    for lo, hi, label in REGISTER_BINS:
        if lo <= pitch < hi:
            return label
    return "reg_high_gt_C6"


def sequence_ops(predicted: list[int], gold: list[int]) -> list[tuple[str, int | None, int | None]]:
    """Unit-cost alignment. Ties keep a substitution rather than a delete plus insert."""

    n, m = len(predicted), len(gold)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    action = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
        action[i][0] = 1
    for j in range(1, m + 1):
        cost[0][j] = j
        action[0][j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = predicted[i - 1] == gold[j - 1]
            best, kind = cost[i - 1][j - 1] + (0 if same else 1), (0 if same else 3)
            if cost[i - 1][j] + 1 < best:
                best, kind = cost[i - 1][j] + 1, 1
            if cost[i][j - 1] + 1 < best:
                best, kind = cost[i][j - 1] + 1, 2
            cost[i][j] = best
            action[i][j] = kind
    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    names = {0: "eq", 3: "sub", 1: "ins", 2: "del"}
    while i or j:
        kind = action[i][j]
        if kind in (0, 3):
            ops.append((names[kind], i - 1, j - 1))
            i -= 1
            j -= 1
        elif kind == 1:
            ops.append(("ins", i - 1, None))
            i -= 1
        else:
            ops.append(("del", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def _rate_table(hits: dict[str, int], support: dict[str, int]) -> dict[str, dict[str, float]]:
    table = {}
    for key in sorted(support):
        total = support[key]
        count = hits.get(key, 0)
        table[key] = {"support": total, "count": count, "rate": round(count / total, 4) if total else 0.0}
    return table


class Tally:
    def __init__(self) -> None:
        self.deletion_support: collections.Counter[str] = collections.Counter()
        self.deletion_hits: collections.Counter[str] = collections.Counter()
        self.substitution_support: collections.Counter[str] = collections.Counter()
        self.substitution_hits: collections.Counter[str] = collections.Counter()
        self.insertions: collections.Counter[str] = collections.Counter()
        self.insertion_confidence: dict[str, list[float]] = collections.defaultdict(list)
        self.matched_confidence: list[float] = []
        self.loss: collections.Counter[str] = collections.Counter()
        self.loss_notes: collections.Counter[str] = collections.Counter()
        self.type_confusion: collections.Counter[str] = collections.Counter()
        self.sub_kinds: collections.Counter[str] = collections.Counter()
        self.deletion_mechanism: collections.Counter[str] = collections.Counter()
        self.score_deletions = {"gold": 0, "gold_full": 0, "predicted": 0, "predicted_full": 0}
        self.examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        self.tx_predicted = 0
        self.tx_gold = 0
        self.tx_matched = 0
        self.exceptions: collections.Counter[str] = collections.Counter()
        self.pitch_sequence_mismatches = 0
        self.samples: list[Any] = []

    def example(self, key: str, payload: dict[str, Any]) -> None:
        if len(self.examples[key]) < 4:
            self.examples[key].append(payload)


def _conditions(gold: dict[str, Any], index: int, pitches: list[int]) -> list[str]:
    duration = gold["duration"]
    pitch = pitches[index]
    tags = ["all", _duration_bin(duration), _register_bin(pitch), f"rel_{gold['relationship']}"]
    same = (index > 0 and pitches[index - 1] == pitch) or (
        index + 1 < len(pitches) and pitches[index + 1] == pitch
    )
    tags.append("same_pitch_neighbor" if same else "pitch_change")
    if gold["ornament"]:
        tags.append("ornament_renderer_only")
    if index > 0 and gold["ioi_prev"] is not None and gold["ioi_prev"] < 0.08:
        tags.append("ioi_prev_lt80")
    if index + 1 < len(pitches) and gold["ioi_next"] is not None and gold["ioi_next"] < 0.08:
        tags.append("ioi_next_lt80")
    return tags


def _load_gold_rows(root: Path, name: str) -> list[dict[str, Any]]:
    rendered = json.loads((root / name / "note_map.json").read_text(encoding="utf-8"))["rendered_notes"]
    rows = []
    starts = [float(row["start_sec"]) for row in rendered]
    for index, row in enumerate(rendered):
        start = starts[index]
        end = float(row["end_sec"])
        rows.append({
            "pitch": int(row["pitch_midi_written"]),
            "sounding": int(row["pitch_midi_sounding"]),
            "duration": end - start,
            "relationship": str(row.get("relationship") or "extra"),
            "ornament": str(row.get("relationship") or "") == "extra" and not row.get("performed_indices"),
            "ioi_prev": None if index == 0 else start - starts[index - 1],
            "ioi_next": None if index + 1 == len(rendered) else starts[index + 1] - start,
        })
    return rows


def _decode_rows(decoded: list[dict[str, Any]]) -> list[list[float]]:
    starts = [note["frame"] * HOP for note in decoded]
    rows = []
    for index, (note, start) in enumerate(zip(decoded, starts)):
        end = starts[index + 1] if index + 1 < len(starts) else start + 0.1
        rows.append([
            note["pitch"], round(start, 6), round(max(start + 0.01, end), 6),
            round(note["confidence"], 4), int(note["optional"]),
            note["alternative_pitch"], round(note["alternative_confidence"], 4),
        ])
    return rows


def _window(values: list[int], index: int) -> list[int]:
    return values[max(0, index - 3):index + 4]


def audit_clip(tally: Tally, name: str, decoded: list[dict[str, Any]], gold_rows: list[dict[str, Any]],
               clip, costs: dict[str, Any]) -> None:
    committed = [note for note in decoded if not note["optional"]]
    committed_input = [index for index, note in enumerate(decoded) if not note["optional"]]
    gold_pitch = [row["pitch"] for row in gold_rows]
    rendered_pitch = [int(event.pitch) for event in clip.rendered]
    if rendered_pitch == gold_pitch:
        for row, event in zip(gold_rows, clip.rendered):
            row["relationship"] = event.relationship
    else:
        tally.pitch_sequence_mismatches += 1
        gold_rows = [
            {
                "pitch": int(event.pitch),
                "sounding": int(event.pitch),
                "duration": float(event.end) - float(event.start),
                "relationship": event.relationship,
                "ornament": event.relationship == "extra",
                "ioi_prev": None,
                "ioi_next": None,
            }
            for event in clip.rendered
        ]
        gold_pitch = [row["pitch"] for row in gold_rows]
    pred_pitch = [int(note["pitch"]) for note in committed]
    ops = sequence_ops(pred_pitch, gold_pitch)
    gold_op: dict[int, tuple[str, int | None]] = {}
    pred_op: dict[int, tuple[str, int | None]] = {}
    for kind, pred_index, gold_index in ops:
        if gold_index is not None:
            gold_op[gold_index] = (kind, pred_index)
        if pred_index is not None:
            pred_op[pred_index] = (kind, gold_index)
    tally.tx_predicted += len(pred_pitch)
    tally.tx_gold += len(gold_pitch)
    tally.tx_matched += sum(kind == "eq" for kind, _gold in gold_op.values())

    for index, row in enumerate(gold_rows):
        kind, _pred = gold_op[index]
        for tag in _conditions(row, index, gold_pitch):
            tally.deletion_support[tag] += 1
            tally.substitution_support[tag] += 1
            if kind == "del":
                tally.deletion_hits[tag] += 1
            elif kind == "sub":
                tally.substitution_hits[tag] += 1

    for index, note in enumerate(committed):
        kind, gold_index = pred_op[index]
        confidence = float(note["confidence"])
        if kind == "eq":
            tally.matched_confidence.append(confidence)
            continue
        if kind != "ins":
            continue
        previous = pred_pitch[index - 1] if index else None
        following = pred_pitch[index + 1] if index + 1 < len(pred_pitch) else None
        pitch = pred_pitch[index]
        if previous == pitch or following == pitch:
            label = "split_same_pitch"
        elif previous is not None and previous == following and previous != pitch:
            label = "flicker_ABA"
        elif gold_index is None and (
            (previous is not None and abs(pitch - previous) in {12, 24})
            or (following is not None and abs(pitch - following) in {12, 24})
        ):
            label = "octave_of_neighbor"
        else:
            label = "other_insertion"
        tally.insertions[label] += 1
        tally.insertion_confidence[label].append(confidence)

    for index, row in enumerate(gold_rows):
        kind, pred_index = gold_op[index]
        if kind != "del":
            continue
        left = gold_op.get(index - 1)
        right = gold_op.get(index + 1)
        if left and right and left[0] == "eq" and right[0] == "eq" and right[1] == left[1] + 1:
            left_frame = committed[left[1]]["frame"]
            right_frame = committed[right[1]]["frame"]
            peaks = [
                note for note in decoded
                if note["optional"] and left_frame < note["frame"] < right_frame
                and note["pitch"] in {row["pitch"], row["sounding"]}
            ]
            mechanism = "weak_peak_left_blank" if peaks else "absorbed_into_neighbor"
        else:
            mechanism = "not_a_single_gap"
        same = (index > 0 and gold_pitch[index - 1] == row["pitch"] and gold_op.get(index - 1, ("", None))[0] == "eq") or (
            index + 1 < len(gold_pitch) and gold_pitch[index + 1] == row["pitch"] and gold_op.get(index + 1, ("", None))[0] == "eq"
        )
        if same:
            mechanism = "same_pitch_merge:" + mechanism
        tally.deletion_mechanism[mechanism] += 1
        if row["duration"] < 0.08:
            tally.deletion_mechanism["short_lt80:" + mechanism] += 1

    notes = _decode_rows(decoded)
    result = align_robust(notes, clip.index.events, clip.score_path, RobustDPCosts(**costs))
    kept = set(result.kept_note_indices)
    events = list(result.events)
    pred_event_pitch = np.asarray(
        [event.pitch for event in sorted(events, key=lambda value: value.rendered_index or 0)], np.int64
    )
    gold_array = np.asarray(gold_pitch, np.int64)
    pairs = (
        {int(i): int(j) for i, j in lcs_pairs(pred_event_pitch, gold_array)}
        if len(pred_event_pitch) and len(gold_array) else {}
    )
    lcs_gold = set(pairs.values())
    remapped = [
        replace(event, rendered_index=pairs.get(event.rendered_index, 1_000_000 + (event.rendered_index or 0)))
        for event in events
    ]
    detail = _official_note_wise_report(
        remapped, clip.rendered, result.deletions, clip.index.deleted_event_indices,
        score_event_count=len(clip.index.events),
    )
    credit_of = {pair["gold_index"]: pair["credit"] for pair in detail["pairs"]}
    pred_type_of = {pair["gold_index"]: pair for pair in detail["pairs"]}
    pred_labels_types = []
    for event in remapped:
        pred_labels_types.append(event.relationship)
    pred_labels_types.extend(["missed_note"] * len(result.deletions))

    for index, row in enumerate(gold_rows):
        credit = float(credit_of.get(index, 0.0))
        loss = 1.0 - credit
        if loss <= 0:
            continue
        kind, pred_index = gold_op[index]
        bucket = "aligner_other"
        if kind == "del":
            same = (index > 0 and gold_pitch[index - 1] == row["pitch"]) or (
                index + 1 < len(gold_pitch) and gold_pitch[index + 1] == row["pitch"]
            )
            if same:
                bucket = "tx_merge_same_pitch"
            elif row["ornament"]:
                bucket = "tx_miss_ornament"
            elif row["duration"] < 0.05:
                bucket = "tx_miss_lt50"
            elif row["duration"] < 0.08:
                bucket = "tx_miss_50to80"
            else:
                bucket = "tx_miss_other"
        elif kind == "sub" and pred_index is not None:
            guessed = pred_pitch[pred_index]
            neighbor = (
                (index > 0 and guessed == gold_pitch[index - 1])
                or (index + 1 < len(gold_pitch) and guessed == gold_pitch[index + 1])
            )
            delta = guessed - row["pitch"]
            if neighbor:
                sub_kind = "neighbor_bleed"
            elif guessed == row["sounding"] and guessed != row["pitch"]:
                sub_kind = "sounding_pitch"
            elif abs(delta) in {12, 24}:
                sub_kind = "octave"
            elif abs(delta) == 1:
                sub_kind = "semitone"
            elif abs(delta) == 2:
                sub_kind = "whole_tone"
            else:
                sub_kind = "other_interval"
            tally.sub_kinds[sub_kind] += 1
            bucket = "tx_sub_" + sub_kind
        elif kind == "eq" and pred_index is not None:
            dropped = committed_input[pred_index] not in kept
            if dropped:
                bucket = "aligner_dropped_correct_note"
            elif credit == 0.5:
                bucket = "aligner_type_mismatch"
                pair = pred_type_of.get(index)
                if pair is not None:
                    pred_label_index = pair["prediction_index"]
                    pred_type = pred_labels_types[pred_label_index] if pred_label_index < len(pred_labels_types) else "?"
                    tally.type_confusion[f"{row['relationship']}->{pred_type}"] += 1
            elif row["relationship"] == "extra" and index not in lcs_gold:
                bucket = "aligner_extra_identity"
            else:
                bucket = "aligner_wrong_score_location"
        tally.loss[bucket] += loss
        tally.loss_notes[bucket] += 1
        tally.example(bucket, {
            "clip": name,
            "index": index,
            "gold_pitch": row["pitch"],
            "sounding": row["sounding"],
            "pred_pitch": None if pred_index is None else pred_pitch[pred_index],
            "duration_ms": round(row["duration"] * 1000, 1),
            "relationship": row["relationship"],
            "ornament": row["ornament"],
            "gold_context": _window(gold_pitch, index),
            "pred_context": _window(pred_pitch, pred_index or 0),
        })

    # Gold score deletions and predicted deletions are past the rendered notes.
    n_rendered = len(gold_rows)
    n_predicted_events = len(remapped)
    pred_credit = {pair["prediction_index"]: pair["credit"] for pair in detail["pairs"]}
    tally.score_deletions["gold"] += detail["gold"] - n_rendered
    tally.score_deletions["predicted"] += detail["predicted"] - n_predicted_events
    for gold_index in range(n_rendered, detail["gold"]):
        credit = float(credit_of.get(gold_index, 0.0))
        if credit >= 1.0:
            tally.score_deletions["gold_full"] += 1
        loss = 1.0 - credit
        if loss <= 0:
            continue
        tally.loss["aligner_missed_a_score_deletion"] += loss
        tally.loss_notes["aligner_missed_a_score_deletion"] += 1
    for pred_index in range(n_predicted_events, detail["predicted"]):
        if float(pred_credit.get(pred_index, 0.0)) >= 1.0:
            tally.score_deletions["predicted_full"] += 1

    tally.samples.append(metric_sample(clip, remapped, result.deletions))


def _official(samples: list[Any]) -> dict[str, Any]:
    if not samples:
        return {}
    return evaluate_joint_dataset(samples, tolerances_sec=())["aggregate"]["official_note_wise"]


def _type_samples(samples: list[Any], kind: str) -> list[JointMetricSample]:
    output = []
    for sample in samples:
        if kind == "missed_note":
            output.append(JointMetricSample(
                predicted=(), target=(), source=sample.source,
                predicted_deletions=sample.predicted_deletions,
                target_deletions=sample.target_deletions,
                score_event_count=sample.score_event_count,
            ))
            continue
        output.append(JointMetricSample(
            predicted=tuple(
                event for event in sample.predicted
                if ("copy" if event.is_copy else event.relationship) == kind
            ),
            target=tuple(
                event for event in sample.target
                if ("copy" if event.is_copy else event.relationship) == kind
            ),
            source=sample.source,
            score_event_count=sample.score_event_count,
        ))
    return output


def _confidence_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, np.float64)
    return {
        "n": int(array.size),
        "mean": round(float(array.mean()), 3),
        "p50": round(float(np.median(array)), 3),
        "frac_below_drop_0.85": round(float((array < 0.85).mean()), 3),
    }


def _dataset_report(tally: Tally) -> dict[str, Any]:
    precision = tally.tx_matched / max(tally.tx_predicted, 1)
    recall = tally.tx_matched / max(tally.tx_gold, 1)
    combined = _official(tally.samples)
    per_type = {}
    if tally.samples:
        for kind in ("match", "copy", "substitute", "extra", "missed_note"):
            official = _official(_type_samples(tally.samples, kind))
            per_type[kind] = {
                "f1": official.get("f1"),
                "precision": official.get("precision"),
                "recall": official.get("recall"),
                "gold": official.get("gold"),
                "predicted": official.get("predicted"),
            }
    total_loss = sum(tally.loss.values())
    return {
        "clips": len(tally.samples) + sum(tally.exceptions.values()),
        "scored_clips": len(tally.samples),
        "exceptions": dict(tally.exceptions),
        "lineage_pitch_mismatches": tally.pitch_sequence_mismatches,
        "transcription": {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "predicted": tally.tx_predicted,
            "gold": tally.tx_gold,
        },
        "combined_note_wise": {
            "f1": combined.get("f1"),
            "precision": combined.get("precision"),
            "recall": combined.get("recall"),
            "per_type_f1": per_type,
        },
        "deletion_rates": _rate_table(tally.deletion_hits, tally.deletion_support),
        "substitution_rates": _rate_table(tally.substitution_hits, tally.substitution_support),
        "insertion_counts": dict(tally.insertions),
        "insertion_confidence": {key: _confidence_summary(value) for key, value in tally.insertion_confidence.items()},
        "matched_confidence": _confidence_summary(tally.matched_confidence),
        "substitution_kinds": dict(tally.sub_kinds),
        "deletion_mechanisms": dict(tally.deletion_mechanism),
        "recall_loss": {
            "total": total_loss,
            "by_cause": {
                key: {"notes": tally.loss_notes[key], "lost_credit": round(tally.loss[key], 2),
                      "share": round(tally.loss[key] / total_loss, 4) if total_loss else 0.0}
                for key in sorted(tally.loss, key=tally.loss.get, reverse=True)
            },
        },
        "score_deletion_labels": dict(tally.score_deletions),
        "aligner_type_confusion_on_correct_pitch": dict(tally.type_confusion.most_common(20)),
        "examples": {key: value for key, value in tally.examples.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _payload = load_dual_checkpoint(Path(candidate["checkpoint"]), device)
    model.eval()
    decoder = candidate["decoder"]
    costs = candidate["aligner_costs"]
    report: dict[str, Any] = {
        "checkpoint": candidate["checkpoint"],
        "split": "val",
        "seed": args.seed,
        "limit": args.limit,
        "device": str(device),
        "datasets": {},
    }
    for dataset_name in args.datasets:
        dataset = candidate["datasets"][dataset_name]
        root = Path(dataset["root"])
        freeze = json.loads(Path(dataset["aligner_freeze"]).read_text(encoding="utf-8"))
        names = list(freeze["eligible"]["val"])
        random.Random(args.seed).shuffle(names)
        names = names[:args.limit]
        tally = Tally()
        print(f"dataset={dataset_name} clips={len(names)} device={device}", flush=True)
        for position, name in enumerate(names, 1):
            try:
                audio = load_audio_mono(root / name / "performance_audio.wav", 22050)
                mel, _rates = extract_dual_mel(audio, device)
                with torch.inference_mode():
                    outputs = infer_dual_outputs(model, np.asarray(mel, np.float32), device)
                decoded = rich_decode(outputs["ctc"], model.config.midi_min, **decoder)
                gold_rows = _load_gold_rows(root, name)
                clip = load_clip(root, name, Path(freeze["lineage_dir"]))
                audit_clip(tally, name, decoded, gold_rows, clip, costs)
            except Exception as error:  # noqa: BLE001
                tally.exceptions[f"{type(error).__name__}: {str(error)[:180]}"] += 1
            if position % 20 == 0 or position == len(names):
                print(f"  {dataset_name} {position}/{len(names)} exceptions={sum(tally.exceptions.values())}", flush=True)
        report["datasets"][dataset_name] = _dataset_report(tally)
        tx = report["datasets"][dataset_name]["transcription"]
        combined = report["datasets"][dataset_name]["combined_note_wise"]
        print(json.dumps({
            "dataset": dataset_name,
            "transcription_f1": tx["f1"],
            "combined_f1": combined["f1"],
            "per_type_f1": {key: value["f1"] for key, value in combined["per_type_f1"].items()},
            "top_loss": list(report["datasets"][dataset_name]["recall_loss"]["by_cause"].items())[:8],
        }, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
