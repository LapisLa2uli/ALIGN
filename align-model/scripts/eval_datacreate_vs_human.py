"""Score stack alignments on DataCreate against the human (GUI "manual") labels.

Population: takes whose labels.json holds at least one manual label, plus takes
explicitly saved by a human annotator with no labels. Gold is the manual labels
only. Each stack alignment (stack schema: events with score_span/relationship/
copy_pass plus missed_score_event_indices) is turned into DataCreate labels:
one extra_note label per score gap (extras between the same two score notes are
one label), one missed_note per missed score event, one wrong_note per
substitute, one repetition per repeated passage.

Two scores per error type:
* official: exclusive note-wise metric on canonical label locations
  (score_part incl. pads), as used for the agent-vs-human comparisons;
* lenient: a prediction is correct when a gold label of the same type has a
  core score range within one note of the prediction's core range.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Sequence

from alignmodel.melody import labels_with_canonical_locations, load_bundle_notes, micro_note_wise, official_label_metrics
from datacreate.melody import extra_neighbor_core
from datacreate.transcription_labeling import _base_label, _score_fields

ERROR_TYPES = ("extra_note", "missed_note", "wrong_note", "repetition")
HUMAN_EMPTY_ANNOTATORS = {"annotator01"}


def human_population(samples: Path) -> list[tuple[Path, list[dict[str, Any]]]]:
    rows = []
    for sample in sorted(p for p in samples.iterdir() if p.is_dir()):
        path = sample / "labels.json"
        if not path.is_file() or not (sample / "verified_score.musicxml").is_file():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        manual = [dict(label) for label in document.get("labels") or [] if label.get("source") == "manual"]
        if manual or (not document.get("labels") and document.get("annotator_id") in HUMAN_EMPTY_ANNOTATORS):
            rows.append((sample, manual))
    return rows


def labels_from_stack_alignment(alignment: dict[str, Any], score) -> list[dict[str, Any]]:
    events = sorted(alignment["events"], key=lambda event: event["note_index"])
    labels: list[dict[str, Any]] = []
    if not score:
        return labels
    last_linked = None
    extras_by_anchor: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    pending: list[dict[str, Any]] = []
    for event in events:
        span = event.get("score_span")
        if span is not None and int(event.get("copy_pass") or 0) == 0:
            for item in pending:
                anchor = (last_linked if last_linked is not None else max(0, int(span[0]) - 1))
                extras_by_anchor[anchor].append(item)
            pending = []
            last_linked = int(span[1]) - 1
            if event["relationship"] == "substitute":
                label = _base_label(f"stack_wrong_{span[0]:04d}", "wrong_note", event["start"], event["end"],
                                    f"heard MIDI {event['pitch']}")
                label.update(_score_fields(score, int(span[0]), int(span[0]) + 1))
                labels.append(label)
        elif span is None and event["relationship"] == "extra":
            pending.append(event)
    for item in pending:
        anchor = last_linked if last_linked is not None else 0
        extras_by_anchor[anchor].append(item)
    for anchor, items in sorted(extras_by_anchor.items()):
        core_start, core_end = extra_neighbor_core(score, anchor)
        label = _base_label(f"stack_extra_{anchor:04d}", "extra_note", items[0]["start"], items[-1]["end"],
                            f"{len(items)} extra note(s): {[item['pitch'] for item in items]}")
        label.update(_score_fields(score, core_start, core_end))
        labels.append(label)
    for index in sorted(alignment.get("missed_score_event_indices") or []):
        if 0 <= int(index) < len(score):
            label = _base_label(f"stack_missed_{index:04d}", "missed_note", 0.0, 0.05, "missed score note")
            label.update(_score_fields(score, int(index), int(index) + 1))
            labels.append(label)
    hypothesis = alignment.get("repeat_hypothesis") or {}
    copies = int(hypothesis.get("extra_copies") or 0)
    source = hypothesis.get("source_span")
    if copies > 0 and source:
        start, end = int(source[0]), min(int(source[1]), len(score))
        if end > start:
            label = _base_label("stack_repetition_000", "repetition", 0.0, 0.05, "replayed score passage")
            label.update(_score_fields(score, start, end))
            label["extra_copies"] = copies
            labels.append(label)
    return labels


def _core(label: dict[str, Any]) -> tuple[int, int] | None:
    part = label.get("score_part") or {}
    start = part.get("core_start_note_index")
    end = part.get("core_end_note_index")
    if start is None or end is None:
        pad = int(part.get("pad_notes") or 0)
        if part.get("start_note_index") is None:
            return None
        start = int(part["start_note_index"]) + pad
        end = int(part["end_note_index"]) - pad
    return int(start), max(int(start), int(end))


def lenient(gold: Sequence[dict[str, Any]], predicted: Sequence[dict[str, Any]], kind: str) -> dict[str, int]:
    gold_cores = [_core(label) for label in gold if label.get("type") == kind]
    gold_cores = [core for core in gold_cores if core is not None]
    used = set()
    hits = 0
    predicted_cores = [_core(label) for label in predicted if label.get("type") == kind]
    for core in predicted_cores:
        if core is None:
            continue
        for index, other in enumerate(gold_cores):
            if index in used:
                continue
            if core[0] <= other[1] + 1 and other[0] <= core[1] + 1:
                used.add(index)
                hits += 1
                break
    return {"predicted": len(predicted_cores), "gold": len(gold_cores), "correct": hits}


def evaluate(samples: Path, alignment_for, *, name: str) -> dict[str, Any]:
    official_rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    lenient_totals: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    per_sample = []
    missing = []
    for sample, manual in human_population(samples):
        alignment = alignment_for(sample)
        if alignment is None:
            missing.append(sample.name)
            continue
        notes = load_bundle_notes(sample)
        predicted = labels_from_stack_alignment(alignment, notes)
        gold = labels_with_canonical_locations(manual, notes)
        predicted = labels_with_canonical_locations(predicted, notes)
        row = {"sample": sample.name, "gold": collections.Counter(l["type"] for l in gold),
               "predicted": collections.Counter(l["type"] for l in predicted), "lenient": {}}
        for kind in ERROR_TYPES:
            metric = official_label_metrics(
                [l for l in gold if l.get("type") == kind],
                [l for l in predicted if l.get("type") == kind],
                score_event_count=len(notes) or None,
            )["official_note_wise"]
            official_rows[kind].append(metric)
            counts = lenient(gold, predicted, kind)
            lenient_totals[kind].update(counts)
            row["lenient"][kind] = counts
        row["gold"] = dict(row["gold"])
        row["predicted"] = dict(row["predicted"])
        per_sample.append(row)
    summary = {}
    for kind in ERROR_TYPES:
        micro = micro_note_wise(official_rows[kind])
        totals = lenient_totals[kind]
        summary[kind] = {
            "official": {key: micro.get(key) for key in ("precision", "recall", "f1", "predicted", "gold")},
            "lenient": {
                "predicted": totals["predicted"], "gold": totals["gold"], "correct": totals["correct"],
                "precision": totals["correct"] / totals["predicted"] if totals["predicted"] else None,
                "recall": totals["correct"] / totals["gold"] if totals["gold"] else None,
            },
        }
    return {"name": name, "population": len(per_sample), "missing_alignments": missing,
            "summary": summary, "per_sample": per_sample}


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=root / "DataCreate" / "samples")
    parser.add_argument("--alignments", type=Path, required=True,
                        help="directory with <sample>.json stack alignments")
    parser.add_argument("--name", default="stack")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def alignment_for(sample: Path):
        path = args.alignments / f"{sample.name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    report = evaluate(args.samples, alignment_for, name=args.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"population": report["population"], "missing": report["missing_alignments"],
                      "summary": report["summary"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
