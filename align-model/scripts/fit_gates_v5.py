"""Fit extra/missed gates on all val clips with clip-grouped cross-validation.

Out-of-fold scores pick the lowest threshold whose precision reaches the
target in every source group (9.2 Weber1, 9.2 procedural, fast procedural),
so real-score clips cannot be carried by procedural ones. The final model is
refitted on all val rows. Test data is never read.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np

from alignmodel.joint.robust_dp_aligner_v3 import EXTRA_VECTORS, MISSED_VECTORS
from fit_gates_v4 import LogisticRegression


def _group(dataset: str, clip: str) -> str:
    return f"{dataset}:{clip.rsplit('_', 1)[0]}"


def _load(paths: list[Path]):
    rows, totals = [], {}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        for dataset, data in report["datasets"].items():
            for kind, key in (("extra", "extra"), ("missed", "missed_note")):
                metric = data["per_type"][key]
                previous = totals.get((dataset, kind), (0.0, 0, 0))
                totals[(dataset, kind)] = (previous[0] + metric["precision"] * metric["predicted"],
                                           previous[1] + metric["predicted"], previous[2] + metric["gold"])
            for row in data["features"]:
                rows.append((dataset, row))
    return rows, totals


def _fold(clip: str, folds: int) -> int:
    return zlib.crc32(clip.encode("utf-8")) % folds


def _fit(x, y):
    mean, scale = x.mean(0), x.std(0) + 1e-6
    model = LogisticRegression(C=1.0).fit((x - mean) / scale, y)
    return model, mean, scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", nargs="+", type=Path)
    parser.add_argument("--target-extra", type=float, default=0.97)
    parser.add_argument("--target-missed", type=float, default=0.97)
    parser.add_argument("--missed-features", default="v2")
    parser.add_argument("--extra-features", default="v1")
    parser.add_argument("--never-ornamented", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-by", choices=("clip", "group"), default="clip",
                        help="group = leave one source score (or procedural set) out")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, totals = _load(args.features)
    gate = {"enabled": True, "missed_edge_runs": False, "missed_features": args.missed_features,
            "extra_features": args.extra_features, "missed_never_ornamented": bool(args.never_ornamented)}
    report = {}
    for kind, target in (("extra", args.target_extra), ("missed", args.target_missed)):
        selected = [(dataset, row) for dataset, row in rows if row["kind"] == kind and not (
            kind == "missed" and (row["edge"] or (args.never_ornamented and (row["ornamented"] or row.get("ornamented_neighbor")))))]
        vector = EXTRA_VECTORS[args.extra_features] if kind == "extra" else MISSED_VECTORS[args.missed_features]
        x = np.array([vector(row) for _d, row in selected])
        y = np.array([1 if row["credit"] >= 1 else 0 for _d, row in selected])
        if args.fold_by == "group":
            names = sorted({_group(d, row["clip"]) for d, row in selected})
            folds = np.array([names.index(_group(d, row["clip"])) for d, row in selected])
            fold_ids = range(len(names))
        else:
            folds = np.array([_fold(row["clip"], args.folds) for _d, row in selected])
            fold_ids = range(args.folds)
        oof = np.zeros(len(selected))
        for fold in fold_ids:
            train, held = folds != fold, folds == fold
            if not held.any() or train.sum() < 20 or len(set(y[train].tolist())) < 2:
                continue
            model, mean, scale = _fit(x[train], y[train])
            oof[held] = model.predict_proba((x[held] - mean) / scale)[:, 1]
        groups = sorted({_group(d, row["clip"]) for d, row in selected})
        curve = []
        chosen = None
        for threshold in np.round(np.arange(0.0, 1.0, 0.01), 2):
            per_group = {}
            for group in groups:
                members = [(row, score) for (d, row), score in zip(selected, oof) if _group(d, row["clip"]) == group]
                kept = [row for row, score in members if score >= threshold]
                credit = sum(1.0 if row["credit"] >= 1 else 0.0 for row in kept)
                if kind == "extra":
                    withheld = [row for row, score in members if score < threshold]
                    precision_known = sum(1.0 if row["credit"] >= 1 else 0.0 for row in kept) / max(len(kept), 1)
                    per_group[group] = {"precision_gated_rows": round(precision_known, 4), "kept": len(kept),
                                        "withheld": len(withheld)}
                else:
                    per_group[group] = {"precision": round(credit / max(len(kept), 1), 4), "kept": len(kept),
                                        "positives_kept": int(credit)}
            per_dataset = {}
            for dataset in sorted({d for d, _r in selected}):
                credit_total, predicted, gold = totals[(dataset, kind)]
                members = [(row, score) for (d, row), score in zip(selected, oof) if d == dataset]
                if kind == "extra":
                    withheld = [row for row, score in members if score < threshold]
                    kept = predicted - len(withheld)
                    credit = credit_total - sum(row["credit"] for row in withheld)
                else:
                    kept_rows = [row for row, score in members if score >= threshold]
                    kept = len(kept_rows)
                    credit = sum(1.0 for row in kept_rows if row["credit"] >= 1)
                per_dataset[dataset] = {"precision": round(credit / max(kept, 1), 4),
                                        "recall": round(credit / max(gold, 1), 4), "kept": kept}
            point = {"threshold": float(threshold), "per_dataset": per_dataset, "per_group": per_group}
            curve.append(point)
            group_key = "precision_gated_rows" if kind == "extra" else "precision"
            if chosen is None and all(v["precision"] >= target for v in per_dataset.values()) and (
                    kind == "extra" or all(v[group_key] >= target for v in per_group.values() if v["kept"] >= 20)):
                chosen = point
        model, mean, scale = _fit(x, y)
        gate[f"{kind}_model"] = {"mean": mean.tolist(), "scale": scale.tolist(),
                                 "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])}
        gate[f"{kind}_threshold"] = chosen["threshold"] if chosen else 0.99
        report[kind] = {"chosen": chosen, "curve": curve, "rows": len(selected)}
        print(kind, "chosen", json.dumps(chosen))
    args.output.write_text(json.dumps({"gate": gate, "report": report}, indent=1) + "\n", encoding="utf-8")
    args.output.with_suffix(".gate.json").write_text(json.dumps(gate) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
