"""Fit extra/missed call scorers on the tuning half; read precision/recall on the check half.

Scorers are logistic regressions on a small fixed feature set (portable as
JSON coefficients) and, for comparison, a shallow gradient-boosted model.
Thresholds are chosen on the tuning half to reach a target per-type
precision; the check half is only read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from alignmodel.joint.robust_dp_aligner_v3 import extra_vector, missed_vector


class LogisticRegression:
    """L2-regularised logistic regression fitted by Newton steps (numpy only)."""

    def __init__(self, C: float = 1.0, max_iter: int = 50, **_unused) -> None:
        self.C = C
        self.max_iter = max_iter

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        design = np.hstack([x, np.ones((len(x), 1))])
        weights = np.zeros(design.shape[1])
        penalty = np.full(design.shape[1], 1.0 / self.C)
        penalty[-1] = 0.0
        for _ in range(self.max_iter):
            p = 1.0 / (1.0 + np.exp(-design @ weights))
            gradient = design.T @ (p - y) + penalty * weights
            hessian = (design * (p * (1 - p))[:, None]).T @ design + np.diag(penalty + 1e-9)
            step = np.linalg.solve(hessian, gradient)
            weights -= step
            if np.abs(step).max() < 1e-8:
                break
        self.coef_ = weights[None, :-1]
        self.intercept_ = weights[-1:]
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-(x @ self.coef_[0] + self.intercept_[0])))
        return np.stack([1 - p, p], axis=1)

EXTRA_FEATURES = ("ioi", "previous_ioi", "same_pitch_neighbor", "aba", "confidence", "onset",
                  "paired_with_missed", "previous_step", "next_step", "ornament")
MISSED_FEATURES = ("pitch_probability", "window_voiced", "slot_voiced", "level_drop_db", "slot_onset",
                   "expected_duration", "run_length", "edge", "ornamented", "nearby_similar",
                   "nearby_unexplained", "gap_ratio")


def _transform(kind: str, row: dict) -> list[float]:
    return extra_vector(row) if kind == "extra" else missed_vector(row)


def _rows(report: dict, kind: str):
    output = []
    for dataset, data in report["datasets"].items():
        for row in data["features"]:
            if row["kind"] == kind:
                output.append((dataset, row))
    return output


def _type_totals(report: dict, kind: str) -> dict[str, tuple[float, int, int]]:
    key = "extra" if kind == "extra" else "missed_note"
    return {dataset: (data["per_type"][key]["precision"] * data["per_type"][key]["predicted"],
                      data["per_type"][key]["predicted"], data["per_type"][key]["gold"])
            for dataset, data in report["datasets"].items()}


def _metrics(kind: str, rows, scores, threshold, totals) -> dict[str, dict[str, float]]:
    output = {}
    for dataset in totals:
        credit, predicted, gold = totals[dataset]
        selected = [(row, score) for (name, row), score in zip(rows, scores) if name == dataset]
        withheld = [row for row, score in selected if score < threshold]
        if kind == "extra":
            kept = predicted - len(withheld)
            kept_credit = credit - sum(row["credit"] for row in withheld)
        else:
            kept_rows = [row for row, score in selected if score >= threshold]
            kept = len(kept_rows)
            kept_credit = sum(1.0 for row in kept_rows if row["credit"] >= 1)
        precision = kept_credit / kept if kept else 1.0
        recall = kept_credit / gold if gold else 0.0
        output[dataset] = {"precision": round(precision, 4), "recall": round(recall, 4),
                           "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4), "kept": kept}
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", type=Path, required=True)
    parser.add_argument("--check", type=Path, required=True)
    parser.add_argument("--target", type=float, default=0.96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tune = json.loads(args.tune.read_text(encoding="utf-8"))
    check = json.loads(args.check.read_text(encoding="utf-8"))
    result = {"target_precision_on_tune": args.target}
    for kind, names in (("extra", EXTRA_FEATURES), ("missed", MISSED_FEATURES)):
        tune_rows, check_rows = _rows(tune, kind), _rows(check, kind)
        x_tune = np.array([_transform(kind, row) for _d, row in tune_rows])
        y_tune = np.array([1 if row["credit"] >= 1 else 0 for _d, row in tune_rows])
        x_check = np.array([_transform(kind, row) for _d, row in check_rows])
        mean, scale = x_tune.mean(0), x_tune.std(0) + 1e-6
        logistic = LogisticRegression(C=1.0, max_iter=200, solver="newton-cholesky").fit((x_tune - mean) / scale, y_tune)
        boosted = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08,
                                                 l2_regularization=1.0).fit(x_tune, y_tune)
        models = {
            "logistic": (logistic.predict_proba((x_tune - mean) / scale)[:, 1],
                         logistic.predict_proba((x_check - mean) / scale)[:, 1]),
            "boosted": (boosted.predict_proba(x_tune)[:, 1], boosted.predict_proba(x_check)[:, 1]),
        }
        totals_tune, totals_check = _type_totals(tune, kind), _type_totals(check, kind)
        result[kind] = {"features": list(names), "models": {}}
        for model_name, (score_tune, score_check) in models.items():
            curve = []
            chosen = None
            for threshold in np.round(np.arange(0.0, 1.0, 0.02), 2):
                tune_metrics = _metrics(kind, tune_rows, score_tune, threshold, totals_tune)
                check_metrics = _metrics(kind, check_rows, score_check, threshold, totals_check)
                curve.append({"threshold": float(threshold), "tune": tune_metrics, "check": check_metrics})
                if chosen is None and all(value["precision"] >= args.target for value in tune_metrics.values()):
                    chosen = curve[-1]
            result[kind]["models"][model_name] = {"chosen": chosen, "curve": curve}
            print(kind, model_name, "chosen", json.dumps(chosen))
        result[kind]["logistic_coefficients"] = {
            "mean": mean.tolist(), "scale": scale.tolist(),
            "coef": logistic.coef_[0].tolist(), "intercept": float(logistic.intercept_[0]),
        }
    gate = {"enabled": True}
    for kind, prefix in (("extra", "extra"), ("missed", "missed")):
        chosen = result[kind]["models"]["logistic"]["chosen"]
        gate[f"{prefix}_model"] = result[kind]["logistic_coefficients"]
        gate[f"{prefix}_threshold"] = chosen["threshold"] if chosen else 0.99
    result["gate_config"] = gate
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    (args.output.with_suffix(".gate.json")).write_text(json.dumps(gate) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
