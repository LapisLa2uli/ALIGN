"""Attribute end-to-end note-wise losses to transcription vs alignment (9.2 val)."""

from __future__ import annotations

import argparse
import collections
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from alignmodel.joint.metrics import _note_wise_event_label
from alignmodel.transcription.ctc_decode_v2 import lcs_pairs
from datacreate.melody import canonical_note_location
from e2e_eval_realistic92 import UNPAIRED_OFFSET, align_notes
from realistic92_aligner_common import load_clip


_STATE: dict[str, Any] = {}


def _init(root: str, lineage_dir: str) -> None:
    _STATE.update(root=Path(root), lineage_dir=Path(lineage_dir))


def _location(label: dict[str, Any], count: int):
    return canonical_note_location(label, score_event_count=count)


def _run(payload) -> dict[str, int]:
    name, raw = payload
    clip = load_clip(_STATE["root"], name, _STATE["lineage_dir"])
    notes = [(int(p), float(s), float(e)) for p, s, e in raw]
    counts: collections.Counter[str] = collections.Counter()
    try:
        events, deletions = align_notes(notes, clip, {}, "dp_v1")
    except Exception:  # noqa: BLE001
        counts["aligner_exception"] += 1
        return dict(counts)
    gold_pitch = np.asarray([e.pitch for e in clip.rendered], np.int64)
    pred_pitch = np.asarray([p for p, _s, _e in notes], np.int64)
    pairs = {int(i): int(j) for i, j in lcs_pairs(pred_pitch, gold_pitch)} if len(pred_pitch) and len(gold_pitch) else {}
    paired_gold = set(pairs.values())
    events = [replace(e, rendered_index=pairs.get(e.rendered_index, UNPAIRED_OFFSET + e.rendered_index)) for e in events]
    count = len(clip.index.events)
    gold_by_rendered = {e.rendered_index: e for e in clip.rendered}
    pred_labels = {}
    for event in events:
        label = _note_wise_event_label(event)
        pred_labels.setdefault(_location(label, count), []).append(label["type"])
    for index in deletions:
        pred_labels.setdefault(("score_events", (index,), 0), []).append("missed_note")
    # Gold side.
    for gold in clip.rendered:
        label = _note_wise_event_label(gold)
        location = _location(label, count)
        types = pred_labels.get(location, [])
        if label["type"] in types:
            counts["gold_full"] += 1
        elif types:
            counts[f"gold_half:{'transcription_miss' if gold.rendered_index not in paired_gold else 'aligner'}:{label['type']}->{types[0]}"] += 1
        else:
            cause = "transcription_miss" if gold.rendered_index not in paired_gold else "aligner"
            counts[f"gold_zero:{cause}:{label['type']}"] += 1
    for index in clip.index.deleted_event_indices:
        types = pred_labels.get(("score_events", (index,), 0), [])
        counts["gold_missed_full" if "missed_note" in types else "gold_missed_notfull"] += 1
    # Predicted side: unpaired transcription notes.
    for event in events:
        if event.rendered_index >= UNPAIRED_OFFSET:
            counts[f"pred_fp_transcription:{event.relationship}"] += 1
    gold_deleted = set(clip.index.deleted_event_indices)
    for index in deletions:
        if index not in gold_deleted:
            location = ("score_events", (index,), 0)
            gold_types = [
                g.relationship for g in clip.rendered
                if g.score_span is not None and _location(_note_wise_event_label(g), count) == location
            ]
            counts["pred_deletion_on_played_note(half)" if gold_types else "pred_deletion_fp"] += 1
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--aligner-freeze", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()
    freeze = json.loads(args.aligner_freeze.read_text(encoding="utf-8"))
    notes = json.loads(args.notes.read_text(encoding="utf-8"))["notes"]
    names = [n for n in freeze["eligible"]["val"] if n in notes][::2][:args.limit]
    totals: collections.Counter[str] = collections.Counter()
    with ProcessPoolExecutor(max_workers=6, initializer=_init,
                             initargs=(str(args.root), freeze["lineage_dir"])) as pool:
        for counts in pool.map(_run, [(n, notes[n]) for n in names], chunksize=4):
            totals.update(counts)
    print(json.dumps({"clips": len(names), **dict(totals.most_common(40))}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
