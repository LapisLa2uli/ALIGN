"""Cause-level diagnosis of frozen v3 stack errors on synthetic validation clips.

For each clip the gold written-pitch sequence is forced onto the model's CTC
posteriors (standard CTC Viterbi). Every missed gold note then has a frame span
on which its own posterior and the competing tokens can be read. Same-pitch
boundaries and false splits are also measured on the audio RMS envelope, which
does not depend on the model. Degraded and clean renders are both decoded, and
the aligner is also run on the perfect (gold) pitch sequence.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numba
import numpy as np
import torch

from alignmodel.joint.metrics import evaluate_joint_dataset
from alignmodel.joint.robust_dp_aligner_v2 import RobustDPCosts, align_robust
from alignmodel.transcription.ctc_decode_v2 import greedy_tokens, lcs_pairs, rich_decode
from alignmodel.transcription.mel_ctc_v3 import extract_dual_mel, infer_dual_outputs, load_dual_checkpoint
from alignmodel.transcription.mel_v1 import load_audio_mono
from audit_stack_v3_val_errors import sequence_ops
from realistic92_aligner_common import load_clip, metric_sample


SR = 22050
HOP_SAMPLES = 256
HOP = HOP_SAMPLES / SR
UNPAIRED = 1_000_000


@numba.njit(cache=True)
def _ctc_viterbi(logp, ext):
    frames, states = logp.shape[0], ext.shape[0]
    neg = -1e30
    dp = np.full((frames, states), neg)
    back = np.zeros((frames, states), np.int8)
    dp[0, 0] = logp[0, ext[0]]
    if states > 1:
        dp[0, 1] = logp[0, ext[1]]
    for t in range(1, frames):
        for s in range(states):
            best = dp[t - 1, s]
            step = 0
            if s >= 1 and dp[t - 1, s - 1] > best:
                best = dp[t - 1, s - 1]
                step = 1
            if s >= 2 and ext[s] != 0 and ext[s] != ext[s - 2] and dp[t - 1, s - 2] > best:
                best = dp[t - 1, s - 2]
                step = 2
            if best > neg:
                dp[t, s] = best + logp[t, ext[s]]
                back[t, s] = step
    path = np.full(frames, -1, np.int64)
    if states > 1 and dp[frames - 1, states - 2] > dp[frames - 1, states - 1]:
        s = states - 2
    else:
        s = states - 1
    if dp[frames - 1, s] <= neg:
        return path
    for t in range(frames - 1, -1, -1):
        path[t] = s
        s -= back[t, s]
    return path


def forced_spans(ctc: np.ndarray, tokens: list[int]) -> list[tuple[int, int]] | None:
    if not tokens:
        return []
    ext = np.zeros(2 * len(tokens) + 1, np.int64)
    ext[1::2] = tokens
    logp = np.log(np.maximum(np.asarray(ctc, np.float64), 1e-12))
    path = _ctc_viterbi(logp, ext)
    if path[0] < 0:
        return None
    spans: list[list[int]] = [[-1, -1] for _ in tokens]
    for t, s in enumerate(path.tolist()):
        if s % 2 == 1:
            k = (s - 1) // 2
            if spans[k][0] < 0:
                spans[k][0] = t
            spans[k][1] = t + 1
    if any(a < 0 for a, _b in spans):
        return None
    return [(a, b) for a, b in spans]


def rms_envelope(audio: np.ndarray, frames: int) -> np.ndarray:
    padded = np.pad(np.asarray(audio, np.float64), (HOP_SAMPLES, HOP_SAMPLES + 512))
    out = np.empty(frames)
    for t in range(frames):
        segment = padded[t * HOP_SAMPLES:t * HOP_SAMPLES + 512]
        out[t] = np.sqrt(np.mean(segment * segment) + 1e-12)
    return out


def boundary_dip(rms: np.ndarray, left: tuple[int, int], right: tuple[int, int], boundary: int) -> float:
    lo, hi = max(0, boundary - 3), min(len(rms), boundary + 2)
    if hi <= lo:
        return float("nan")
    left_level = np.median(rms[left[0]:max(left[1], left[0] + 1)])
    right_level = np.median(rms[right[0]:max(right[1], right[0] + 1)])
    level = max(left_level, right_level, 1e-9)
    return float(rms[lo:hi].min() / level)


def _gold_rows(root: Path, name: str) -> list[dict[str, Any]]:
    rendered = json.loads((root / name / "note_map.json").read_text(encoding="utf-8"))["rendered_notes"]
    rows = []
    for index, row in enumerate(rendered):
        start, end = float(row["start_sec"]), float(row["end_sec"])
        next_start = float(rendered[index + 1]["start_sec"]) if index + 1 < len(rendered) else None
        renderer_only = str(row.get("relationship") or "") == "extra" and not row.get("performed_indices")
        overlaps = next_start is not None and end - next_start > 0.05
        if renderer_only and overlaps and end - start >= 1.0:
            subtype = "grace_overlap_long"
        elif renderer_only:
            subtype = "ornament_expansion"
        else:
            subtype = "score_note"
        rows.append({
            "pitch": int(row["pitch_midi_written"]),
            "duration": end - start,
            "subtype": subtype,
            "overlaps_next": overlaps,
        })
    return rows


def _duration_bin(duration: float) -> str:
    if duration < 0.05:
        return "lt50"
    if duration < 0.08:
        return "50to80"
    if duration < 0.12:
        return "80to120"
    if duration < 0.2:
        return "120to200"
    return "ge200"


def _notes_for_aligner(decoded: list[dict[str, Any]]) -> list[list[float]]:
    starts = [note["frame"] * HOP for note in decoded]
    rows = []
    for k, (note, start) in enumerate(zip(decoded, starts)):
        end = starts[k + 1] if k + 1 < len(starts) else start + 0.1
        rows.append([note["pitch"], start, max(start + 0.01, end), note["confidence"], int(note["optional"]),
                     note["alternative_pitch"], note["alternative_confidence"]])
    return rows


class Stats:
    def __init__(self) -> None:
        self.c: collections.Counter[str] = collections.Counter()
        self.lists: dict[str, list[float]] = collections.defaultdict(list)
        self.examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        self.samples_real: list[Any] = []
        self.samples_perfect: list[Any] = []
        self.samples_clean: list[Any] = []

    def add(self, key: str, value: int = 1) -> None:
        self.c[key] += value

    def put(self, key: str, value: float) -> None:
        if np.isfinite(value):
            self.lists[key].append(float(value))

    def example(self, key: str, payload: dict[str, Any], cap: int = 5) -> None:
        if len(self.examples[key]) < cap:
            self.examples[key].append(payload)


def _align(notes, clip, costs):
    result = align_robust(notes, clip.index.events, clip.score_path, RobustDPCosts(**costs))
    events = list(result.events)
    gold_pitch = np.asarray([event.pitch for event in clip.rendered], np.int64)
    pred_pitch = np.asarray([event.pitch for event in sorted(events, key=lambda e: e.rendered_index)], np.int64)
    pairs = ({int(i): int(j) for i, j in lcs_pairs(pred_pitch, gold_pitch)}
             if len(pred_pitch) and len(gold_pitch) else {})
    remapped = [replace(e, rendered_index=pairs.get(e.rendered_index, UNPAIRED + e.rendered_index)) for e in events]
    return result, remapped


def _location(event) -> tuple:
    if event.score_span is not None:
        return ("score", event.score_span, event.copy_pass)
    return ("extra", event.rendered_index)


def _decode_with_evidence(model, audio, device, decoder):
    mel, _ = extract_dual_mel(audio, device)
    outputs = infer_dual_outputs(model, np.asarray(mel, np.float32), device)
    decoded = rich_decode(outputs["ctc"], model.config.midi_min, **decoder)
    return outputs, decoded


def diagnose_clip(stats: Stats, name: str, root: Path, clip, model, device, decoder, costs, clean: bool) -> None:
    midi_min = int(model.config.midi_min)
    vocabulary = int(model.vocabulary)
    gold = _gold_rows(root, name)
    gold_pitch = [row["pitch"] for row in gold]
    if [int(e.pitch) for e in clip.rendered] != gold_pitch:
        stats.add("skip_lineage_pitch_mismatch")
        return
    audio = load_audio_mono(root / name / "performance_audio.wav", SR)
    outputs, decoded = _decode_with_evidence(model, audio, device, decoder)
    ctc = np.asarray(outputs["ctc"], np.float32)
    frames = ctc.shape[0]
    committed = [n for n in decoded if not n["optional"]]
    pred_pitch = [int(n["pitch"]) for n in committed]
    ops = sequence_ops(pred_pitch, gold_pitch)
    gold_op: dict[int, tuple[str, int | None]] = {}
    pred_op: dict[int, tuple[str, int | None]] = {}
    for kind, i, j in ops:
        if j is not None:
            gold_op[j] = (kind, i)
        if i is not None:
            pred_op[i] = (kind, j)

    # Gold note categories and vocabulary range.
    for j, row in enumerate(gold):
        kind = gold_op[j][0]
        tags = [f"subtype:{row['subtype']}", f"dur:{_duration_bin(row['duration'])}"]
        if not (midi_min <= row["pitch"] <= midi_min + vocabulary - 2):
            tags.append("out_of_vocabulary")
        if row["overlaps_next"]:
            tags.append("overlaps_next_note")
        for tag in tags:
            stats.add(f"support|{tag}")
            if kind == "del":
                stats.add(f"deleted|{tag}")

    # Forced CTC alignment of gold, with out-of-vocabulary notes clipped to the nearest token.
    tokens = [int(np.clip(p - midi_min + 1, 1, vocabulary - 1)) for p in gold_pitch]
    spans = forced_spans(ctc, tokens)
    if spans is None:
        stats.add("forced_alignment_infeasible")
        return
    stats.add("forced_alignment_ok")
    scaled = ctc.copy()
    scaled[:, 0] *= float(decoder["blank_scale"])
    winners = scaled.argmax(axis=1)
    rms = rms_envelope(audio, frames)

    for j, row in enumerate(gold):
        kind, _i = gold_op[j]
        if kind != "del":
            continue
        a, b = spans[j]
        lo, hi = max(0, a - 2), min(frames, b + 2)
        token = tokens[j]
        peak = float(ctc[lo:hi, token].max())
        wins = bool((winners[a:b] == token).any())
        nonblank = ctc[lo:hi, 1:].copy()
        nonblank[:, token - 1] = -1
        rival = float(nonblank.max())
        blank = float(ctc[lo:hi, 0].max())
        same_prev = j > 0 and gold_pitch[j - 1] == row["pitch"]
        same_next = j + 1 < len(gold) and gold_pitch[j + 1] == row["pitch"]
        family = (
            "out_of_vocabulary" if not (midi_min <= row["pitch"] <= midi_min + vocabulary - 2)
            else "grace_overlap_long" if row["subtype"] == "grace_overlap_long"
            else "same_pitch_repeat" if (same_prev or same_next)
            else "ornament_expansion" if row["subtype"] == "ornament_expansion"
            else f"score_note_{_duration_bin(row['duration'])}"
        )
        if family == "same_pitch_repeat":
            # The CTC path must pass a blank between the two same-pitch tokens.
            k = j if same_next else j - 1
            gap_lo, gap_hi = spans[k][1], spans[k + 1][0]
            gap = ctc[gap_lo:max(gap_hi, gap_lo + 1), 0]
            blank_gap = float(gap.max()) if gap.size else 0.0
            token_gap = float(ctc[gap_lo:max(gap_hi, gap_lo + 1), token].max())
            dip = boundary_dip(rms, spans[k], spans[k + 1], spans[k + 1][0])
            stats.put("merge_dip", dip)
            stats.put("merge_blank_in_gap", blank_gap)
            separated = bool((winners[gap_lo:max(gap_hi, gap_lo + 1)] == 0).any())
            if not wins:
                bucket = ("own_attack_never_wins_no_evidence" if peak < 0.05
                          else "own_attack_never_wins_weak" if peak < 0.3
                          else "own_attack_never_wins_strong")
            elif not separated:
                bucket = ("no_blank_between_repeats_blank_lt0.3" if blank_gap < 0.3
                          else "no_blank_between_repeats_blank_suppressed_by_scale")
            else:
                bucket = "both_visible_lost_in_collapse"
            stats.add(f"cause|same_pitch_repeat|{bucket}")
            stats.put("merge_peak", peak)
            _ = token_gap
            stats.add(f"cause|same_pitch_repeat|dip_{'lt0.5' if dip < 0.5 else '0.5to0.8' if dip < 0.8 else 'ge0.8'}")
        else:
            bucket = (
                "no_evidence_lt0.05" if peak < 0.05
                else "weak_0.05to0.3" if peak < 0.3
                else "wins_somewhere_but_not_kept" if wins
                else "strong_but_outranked"
            )
            stats.add(f"cause|{family}|{bucket}")
            stats.put(f"peak|{family}", peak)
            stats.put(f"rival|{family}", rival)
            stats.put(f"blank|{family}", blank)
            stats.put(f"span_frames|{family}", b - a)
        stats.example(f"del|{family}", {
            "clip": name, "gold_index": j, "pitch": row["pitch"], "dur_ms": round(row["duration"] * 1000, 1),
            "subtype": row["subtype"], "forced_frames": [a, b], "peak": round(peak, 3), "rival": round(rival, 3),
            "context": gold_pitch[max(0, j - 3):j + 4],
        })

    # Correctly separated same-pitch repeats: reference dip distribution.
    for j in range(len(gold) - 1):
        if gold_pitch[j] != gold_pitch[j + 1]:
            continue
        if gold_op[j][0] == "eq" and gold_op[j + 1][0] == "eq":
            stats.put("repeat_ok_dip", boundary_dip(rms, spans[j], spans[j + 1], spans[j + 1][0]))
        stats.add("repeat_pairs")
    # Pitch-change boundaries: generic dip reference.
    for j in range(0, len(gold) - 1, 7):
        if gold_pitch[j] != gold_pitch[j + 1]:
            stats.put("pitch_change_dip", boundary_dip(rms, spans[j], spans[j + 1], spans[j + 1][0]))

    # Insertions: false same-pitch splits and other extras.
    for i, note in enumerate(committed):
        kind, _j = pred_op[i]
        if kind != "ins":
            continue
        previous = pred_pitch[i - 1] if i else None
        following = pred_pitch[i + 1] if i + 1 < len(pred_pitch) else None
        frame = int(note["frame"])
        left_start = int(committed[i - 1]["frame"]) if i else max(0, frame - 10)
        right_end = int(committed[i + 1]["frame"]) if i + 1 < len(committed) else min(frames, frame + 10)
        dip = boundary_dip(rms, (left_start, frame), (frame, right_end), frame)
        onset = float(np.max(outputs["onset"][max(0, frame - 2):frame + 3])) if frames else 0.0
        if previous == note["pitch"] or following == note["pitch"]:
            label = "split_same_pitch"
        elif previous is not None and previous == following:
            label = "flicker_ABA"
        else:
            label = "other_insertion"
        stats.add(f"insert|{label}")
        stats.put(f"insert_dip|{label}", dip)
        stats.put(f"insert_onset|{label}", onset)
        stats.put(f"insert_conf|{label}", float(note["confidence"]))
        stats.put(f"insert_len_frames|{label}", (right_end - frame) if following is not None else float("nan"))

    # Aligner on transcription, on perfect transcription, and attribution of wrong locations.
    real_result, real_events = _align(_notes_for_aligner(decoded), clip, costs)
    perfect_notes = [[int(e.pitch), float(e.start), float(e.end), 1.0, 0, -1, 0.0] for e in clip.rendered]
    perfect_result, perfect_events = _align(perfect_notes, clip, costs)
    stats.samples_real.append(metric_sample(clip, real_events, real_result.deletions))
    stats.samples_perfect.append(metric_sample(clip, perfect_events, perfect_result.deletions))
    gold_events = clip.rendered
    perfect_by_gold = {e.rendered_index: e for e in perfect_events if e.rendered_index < UNPAIRED}
    real_by_gold = {e.rendered_index: e for e in real_events if e.rendered_index < UNPAIRED}
    error_positions = [j for j, (kind, _i) in gold_op.items() if kind in ("del", "sub")]
    error_positions += [j for i, (kind, j) in pred_op.items() if kind == "ins" and j is not None]
    insertion_anchor = []
    for kind, i, j in ops:
        if kind == "ins":
            insertion_anchor.append(i)
    error_array = np.asarray(sorted(set(error_positions)), np.int64)
    for j, target in enumerate(gold_events):
        predicted = real_by_gold.get(j)
        if predicted is None or gold_op[j][0] != "eq":
            continue
        if _location(predicted) == _location(target):
            continue
        perfect = perfect_by_gold.get(j)
        intrinsic = perfect is None or _location(perfect) != _location(target)
        distance = int(np.abs(error_array - j).min()) if error_array.size else 10_000
        if target.score_span is not None and predicted.score_span == target.score_span:
            shape = "same_event_wrong_copy_pass"
        elif target.score_span is not None and predicted.score_span is not None:
            offset = predicted.score_span[0] - target.score_span[0]
            in_run = (j > 0 and gold_pitch[j - 1] == gold_pitch[j]) or (
                j + 1 < len(gold_pitch) and gold_pitch[j + 1] == gold_pitch[j])
            shape = ("shift_inside_same_pitch_run" if in_run and abs(offset) <= 4
                     else "small_shift_1to4" if abs(offset) <= 4 else "large_jump")
        elif target.score_span is None:
            shape = "gold_extra_pred_linked"
        else:
            shape = "gold_linked_pred_extra"
        origin = (
            "aligner_intrinsic" if intrinsic
            else "propagated_tx_error_within_3" if distance <= 3
            else "propagated_tx_error_within_10" if distance <= 10
            else "propagated_far_or_repeat_hypothesis"
        )
        stats.add(f"wrong_loc|{origin}")
        stats.add(f"wrong_loc_shape|{shape}")
        stats.add(f"wrong_loc|{origin}|{shape}")
        stats.example(f"wrong_loc|{origin}|{shape}", {
            "clip": name, "gold_index": j, "pitch": gold_pitch[j],
            "gold": [target.relationship, target.score_span, target.copy_pass],
            "pred": [predicted.relationship, predicted.score_span, predicted.copy_pass],
            "nearest_tx_error": distance,
        }, cap=3)
    gold_copies = max((e.copy_pass for e in gold_events), default=0)
    stats.add("repeat_hypothesis_copies_" + ("match" if real_result.copies == gold_copies else "differ"))
    stats.add("perfect_repeat_hypothesis_copies_" + ("match" if perfect_result.copies == gold_copies else "differ"))

    # Predicted score deletions.
    gold_deleted = clip.index.deleted_event_indices
    gold_op_by_event: dict[int, str] = {}
    for j, target in enumerate(gold_events):
        if target.score_span is not None and target.copy_pass == 0:
            for event_index in range(*target.score_span):
                gold_op_by_event[event_index] = gold_op[j][0]
    for index in real_result.deletions:
        if index in gold_deleted:
            label = "correct_gold_deletion"
        elif gold_op_by_event.get(index) == "del":
            label = "played_but_transcriber_missed_it"
        elif gold_op_by_event.get(index) == "sub":
            label = "played_transcribed_wrong_pitch"
        elif index in gold_op_by_event:
            label = "played_and_transcribed_aligner_deleted"
        else:
            label = "event_only_in_copy_pass_or_other"
        stats.add(f"pred_deletion|{label}")
    for index in gold_deleted:
        stats.add("gold_deletion|" + ("found" if index in real_result.deletions else "missed"))
        if index not in real_result.deletions:
            filled = any(
                e.score_span is not None and e.score_span[0] <= index < e.score_span[1] and e.copy_pass == 0
                for e in real_events
            )
            stats.add("gold_deletion_missed|" + ("filled_by_a_transcribed_note" if filled else "left_empty_no_deletion"))

    if clean:
        clean_path = root / name / "performance_audio_clean.wav"
        if clean_path.is_file():
            clean_audio = load_audio_mono(clean_path, SR)
            _clean_outputs, clean_decoded = _decode_with_evidence(model, clean_audio, device, decoder)
            clean_pitch = [int(n["pitch"]) for n in clean_decoded if not n["optional"]]
            for kind, _i, j in sequence_ops(clean_pitch, gold_pitch):
                if kind == "del" and j is not None:
                    row = gold[j]
                    same = (j > 0 and gold_pitch[j - 1] == row["pitch"]) or (
                        j + 1 < len(gold) and gold_pitch[j + 1] == row["pitch"])
                    stats.add("clean_deleted|all")
                    stats.add(f"clean_deleted|subtype:{row['subtype']}")
                    stats.add(f"clean_deleted|dur:{_duration_bin(row['duration'])}")
                    if same:
                        stats.add("clean_deleted|same_pitch_neighbor")
                elif kind == "ins":
                    stats.add("clean_inserted|all")
            for j, row in enumerate(gold):
                if gold_op[j][0] == "del":
                    same = (j > 0 and gold_pitch[j - 1] == row["pitch"]) or (
                        j + 1 < len(gold) and gold_pitch[j + 1] == row["pitch"])
                    stats.add("degraded_deleted_on_clean_clips|all")
                    stats.add(f"degraded_deleted_on_clean_clips|subtype:{row['subtype']}")
                    stats.add(f"degraded_deleted_on_clean_clips|dur:{_duration_bin(row['duration'])}")
                    if same:
                        stats.add("degraded_deleted_on_clean_clips|same_pitch_neighbor")
            stats.add("clean_clips")
            stats.add("clean_gold_notes", len(gold))
            stats.add("clean_predicted_notes", len(clean_pitch))


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values)
    return {"n": int(array.size), "p10": round(float(np.percentile(array, 10)), 3),
            "p50": round(float(np.median(array)), 3), "p90": round(float(np.percentile(array, 90)), 3),
            "mean": round(float(array.mean()), 3)}


def _f1(samples) -> dict[str, float] | None:
    if not samples:
        return None
    value = evaluate_joint_dataset(samples, tolerances_sec=())["aggregate"]["official_note_wise"]
    return {key: round(float(value[key]), 4) for key in ("f1", "precision", "recall")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _payload = load_dual_checkpoint(Path(candidate["checkpoint"]), device)
    model.eval()
    report: dict[str, Any] = {"split": "val", "seed": args.seed, "limit": args.limit, "datasets": {}}
    for dataset_name in args.datasets:
        dataset = candidate["datasets"][dataset_name]
        root = Path(dataset["root"])
        freeze = json.loads(Path(dataset["aligner_freeze"]).read_text(encoding="utf-8"))
        names = list(freeze["eligible"]["val"])
        random.Random(args.seed).shuffle(names)
        names = names[:args.limit]
        stats = Stats()
        for position, name in enumerate(names, 1):
            try:
                clip = load_clip(root, name, Path(freeze["lineage_dir"]))
                with torch.inference_mode():
                    diagnose_clip(stats, name, root, clip, model, device, candidate["decoder"],
                                  candidate["aligner_costs"], args.clean)
            except Exception as error:  # noqa: BLE001
                stats.add(f"exception|{type(error).__name__}: {str(error)[:120]}")
            if position % 40 == 0 or position == len(names):
                print(f"{dataset_name} {position}/{len(names)}", flush=True)
        report["datasets"][dataset_name] = {
            "counts": dict(sorted(stats.c.items())),
            "distributions": {key: _summary(value) for key, value in sorted(stats.lists.items())},
            "aligner_f1_real_transcription": _f1(stats.samples_real),
            "aligner_f1_perfect_transcription": _f1(stats.samples_perfect),
            "examples": stats.examples,
        }
        print(json.dumps({
            "dataset": dataset_name,
            "real": report["datasets"][dataset_name]["aligner_f1_real_transcription"],
            "perfect": report["datasets"][dataset_name]["aligner_f1_perfect_transcription"],
        }), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
