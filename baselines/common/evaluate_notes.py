"""Class-aware onset metrics against converted label MIDIs; no source paths needed."""
import argparse
import json
from pathlib import Path

from eval_bridge import Warner, load_pred, note_prf, _micro_block, _mean_block
from prepare_dataset import output_paths
from check_labels import load_notes


def evaluate(root, pred_dir):
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
    rows = {key: [] for key in ("extra", "missing", "correct", "all")}
    counts, pieces = {}, {}
    for tid in ids:
        pred, info, _ = load_pred(str(pred_dir / tid / "mix.mid"), "name", Warner(), tid)
        if info["source"] in ("unreadable", "missing") or info.get("ambiguous"):
            raise ValueError(f"Unreadable or ambiguous prediction: {tid}")
        if info["source"] != "name" and any(info.get("track_sizes", [])):
            raise ValueError(f"Nonempty prediction has no class track names: {tid}")
        counts[tid] = {key: len(value) for key, value in pred.items()}
        if real == {True}:
            continue
        paths = output_paths(root, tid)
        gt = {key: load_notes(str(paths["removed" if key == "missing" else key]))
              for key in ("extra", "missing", "correct")}
        pieces[tid] = {}
        for key in rows:
            truth = sum(gt.values(), []) if key == "all" else gt[key]
            estimate = sum(pred.values(), []) if key == "all" else pred[key]
            score = note_prf(truth, estimate, "hz")
            rows[key].append(score)
            pieces[tid][key] = score
    return {
        "n_pieces": len(ids), "real_test": real == {True},
        "protocol": "mir_eval onset-only, 50 ms, 50 cents, classes by MIDI track name",
        "note": "Real label MIDIs are placeholders; note metrics are unavailable." if real == {True}
                else "All is class-agnostic transcription; use per-class F1 for error detection.",
        "micro": {key: _micro_block(value) for key, value in rows.items()} if real == {False} else None,
        "per_piece_mean": {key: _mean_block(value) for key, value in rows.items()} if real == {False} else None,
        "prediction_counts": counts, "per_piece": pieces,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.data, args.pred_dir)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("per_piece", "prediction_counts")}, indent=2))
