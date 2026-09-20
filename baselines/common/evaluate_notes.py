"""Class-aware note metrics against converted label MIDIs; no source paths needed."""
import argparse
import json
import sys
from pathlib import Path

from eval_bridge import (Warner, read_pred_midi_tracks, class_from_name, note_prf,
                         prf, _micro_block, _mean_block)
from prepare_dataset import output_paths
from check_labels import load_notes


def _official_note_wise(gold_by_class, pred_by_class, score_notes):
    """ALIGN comparison metric; mir_eval remains the authors' legacy protocol."""
    try:
        root = Path(__file__).resolve().parents[2]
        for path in (root / "align-model" / "src", root / "DataCreate" / "src"):
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)
        from alignmodel.joint.score_location_adapter import (
            evaluate_class_notes_note_wise,
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"official note-wise adapter unavailable: {exc}",
        }
    return evaluate_class_notes_note_wise(
        gold_by_class, pred_by_class, score_notes
    )


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
    official_rows = []
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
        score_path = paths["score_mid"]
        score_notes = load_notes(str(score_path)) if score_path.is_file() else None
        official = _official_note_wise(gt, pred, score_notes)
        pieces[tid]["official_note_wise"] = official
        official_rows.append(official)
    legacy = {
        "protocol": "mir_eval onset-only, 50 ms, 50 cents, classes by MIDI track name",
        "micro": {key: _micro_block(value) for key, value in rows.items()} if real == {False} else None,
        "per_piece_mean": {key: _mean_block(value) for key, value in rows.items()} if real == {False} else None,
    }
    official_report = None
    if official_rows and all(row.get("status") == "available" for row in official_rows):
        credit = sum(float(row["credit"]) for row in official_rows)
        predicted = sum(int(row["predicted"]) for row in official_rows)
        gold = sum(int(row["gold"]) for row in official_rows)
        precision = credit / predicted if predicted else 0.0
        recall = credit / gold if gold else 0.0
        if not predicted and not gold:
            precision = recall = 1.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        official_report = {
            "schema_version": "align-note-wise-score-event-metric-v1",
            "status": "available",
            "type_mismatch_credit": 0.5,
            "credit": credit,
            "predicted": predicted,
            "gold": gold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    elif official_rows:
        official_report = {
            "status": "unavailable",
            "reason": official_rows[0].get("reason") or "official note-wise unavailable",
        }
    return {
        "n_pieces": len(ids), "real_test": real == {True},
        "protocol": "official_note_wise canonical score-event identity; "
                    "legacy_mir_eval_onset_50ms is the authors' native protocol",
        "note": "Real label MIDIs are placeholders; note metrics are unavailable." if real == {True}
                else "Headline F1 is official_note_wise. All is class-agnostic transcription "
                     "under the legacy mir_eval protocol.",
        "unclassified_policy": "Unnamed notes receive no class-match credit and count as false positives in class_aware; included in class-agnostic all. Nonempty unknown track names still fail.",
        "unclassified_notes": sum(unclassified_counts.values()),
        "pieces_with_unclassified_notes": sum(n > 0 for n in unclassified_counts.values()),
        "official_note_wise": official_report if real == {False} else None,
        "legacy_mir_eval_onset_50ms": legacy if real == {False} else None,
        "micro": legacy["micro"],
        "per_piece_mean": legacy["per_piece_mean"],
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
