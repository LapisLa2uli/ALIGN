"""Class-aware onset metrics against converted label MIDIs; no source paths needed."""
import argparse
import json
from pathlib import Path

from eval_bridge import (Warner, read_pred_midi_tracks, class_from_name, note_prf,
                         prf, _micro_block, _mean_block)
from prepare_dataset import output_paths
from check_labels import load_notes


def evaluate(root, pred_dir, allow_unclassified=False):
    ids = json.loads((pred_dir / "evaluated_ids.json").read_text())
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate evaluation selection")
    existing = {p.parent.name for p in pred_dir.glob("*/mix.mid")}
    if existing != set(ids):
        raise ValueError(f"Prediction coverage mismatch: missing={set(ids)-existing}, "
                         f"unexpected={existing-set(ids)}")
    manifest = json.loads((root / "manifest.json").read_text())
    real = {bool(manifest["tracks"][tid].get("real_test", False)) for tid in ids}
    if len(real) != 1:
        raise ValueError("Do not mix labelled synthetic data and unlabelled real data")
    classes = ("extra", "missing", "correct")
    rows = {key: [] for key in (*classes, "all", "class_aware")}
    counts, pieces, unclassified_counts = {}, {}, {}
    for tid in ids:
        tracks, _, _ = read_pred_midi_tracks(str(pred_dir / tid / "mix.mid"), Warner())
        pred, unclassified = {key: [] for key in classes}, []
        for track in tracks:
            cls = class_from_name(track['name'])
            if cls is not None:
                pred[cls].extend(track['notes'])
            elif track['notes']:
                # A decoder can emit notes before its first valid class token.
                # Never infer their class from track position or missing names.
                if not allow_unclassified or track['name'].strip():
                    raise ValueError(f"Unclassified prediction track in {tid}: {track['name']!r}")
                unclassified.extend(track['notes'])
        unclassified_counts[tid] = len(unclassified)
        counts[tid] = {key: len(value) for key, value in pred.items()}
        counts[tid]['unclassified'] = len(unclassified)
        if real == {True}:
            continue
        paths = output_paths(root, tid)
        gt = {key: load_notes(str(paths["removed" if key == "missing" else key]))
              for key in ("extra", "missing", "correct")}
        pieces[tid] = {}
        for key in (*classes, "all"):
            truth = sum(gt.values(), []) if key == "all" else gt[key]
            estimate = sum(pred.values(), []) + unclassified if key == "all" else pred[key]
            score = note_prf(truth, estimate, "hz")
            rows[key].append(score)
            pieces[tid][key] = score
        tp = sum(pieces[tid][key]['tp'] for key in classes)
        ng = sum(len(notes) for notes in gt.values())
        npred = sum(len(notes) for notes in pred.values()) + len(unclassified)
        precision, recall, f1 = prf(tp, npred, ng)
        score = dict(P=precision, R=recall, F1=f1, tp=tp, n_gt=ng, n_pred=npred)
        rows['class_aware'].append(score)
        pieces[tid]['class_aware'] = score
    return {
        "n_pieces": len(ids), "real_test": real == {True},
        "protocol": "mir_eval onset-only, 50 ms, 50 cents, classes by MIDI track name",
        "note": "Real label MIDIs are placeholders; note metrics are unavailable." if real == {True}
                else "All is class-agnostic transcription; use per-class F1 for error detection.",
        "unclassified_policy": "Unnamed notes receive no class-match credit and count as false positives in class_aware; included in class-agnostic all. Nonempty unknown track names still fail.",
        "unclassified_notes": sum(unclassified_counts.values()),
        "pieces_with_unclassified_notes": sum(n > 0 for n in unclassified_counts.values()),
        "micro": {key: _micro_block(value) for key, value in rows.items()} if real == {False} else None,
        "per_piece_mean": {key: _mean_block(value) for key, value in rows.items()} if real == {False} else None,
        "prediction_counts": counts, "per_piece": pieces,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-unclassified", action="store_true",
                        help="Retain unnamed decoder notes as invalid-class false positives; never guess their class")
    args = parser.parse_args()
    report = evaluate(args.data, args.pred_dir, allow_unclassified=args.allow_unclassified)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("per_piece", "prediction_counts")}, indent=2))
