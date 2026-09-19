"""Score the completed full stack on DataCreate 001-040 with exclusions.

Uses the hash-verified 2026-09-15 gold-blind freeze of the current completed
full model (joint decoder + error-heads-v3). Does not re-run audio inference:
the freeze already covers every requested sample with the same checkpoints.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ALIGN = ROOT / "align-model"
SCRIPTS = ALIGN / "scripts"
sys.path[:0] = [
    str(SCRIPTS),
    str(ALIGN / "src"),
    str(ROOT / "DataCreate" / "src"),
    str(ROOT / "synth-pipeline" / "src"),
]

import eval_datacreate_current as base
import eval_datacreate_error_heads_v3 as frozen_eval
import eval_datacreate_note_wise as note_eval
from alignmodel.melody import match_note_wise_labels_detail, parse_sounding_notes

OUT = Path(__file__).resolve().parent
FREEZE = (
    ALIGN
    / "runs"
    / "eval-datacreate-all94-agent-note-wise-20260915"
    / "freeze_manifest.json"
)
SAMPLES = ROOT / "DataCreate" / "samples"
EXCLUDE = {"005", "007", "010", "012", "020", "026", "030", "034", "036"}
KEEP = [f"{index:03d}" for index in range(1, 41) if f"{index:03d}" not in EXCLUDE]
MODELS = note_eval.MODELS
HEADLINE = "v3"
TYPE_ORDER = [
    "wrong_note",
    "extra_note",
    "missed_note",
    "rhythm_error",
    "repetition",
    "click",
    "bad_start",
    "bad_timbre",
    "squeak",
    "sliding",
]
TYPE_COLORS = {
    "wrong_note": "#c62828",
    "extra_note": "#ef6c00",
    "missed_note": "#6a1b9a",
    "rhythm_error": "#1565c0",
    "repetition": "#00838f",
    "click": "#546e7a",
    "bad_start": "#2e7d32",
    "bad_timbre": "#ad1457",
    "squeak": "#5d4037",
    "sliding": "#455a64",
}


def _span(label: dict) -> tuple[int, int] | None:
    part = label.get("score_part")
    if not isinstance(part, dict):
        return None
    try:
        first = int(part["start_note_index"])
        last = int(part["end_note_index"])
    except (KeyError, TypeError, ValueError):
        return None
    if last < first:
        return None
    return first, last


def _time_span(label: dict) -> tuple[float, float] | None:
    try:
        start = float(label["start_time"])
        end = float(label["end_time"])
    except (KeyError, TypeError, ValueError):
        return None
    if end <= start:
        return None
    return start, end


def inventory_and_audit(sample_ids: list[str]) -> dict:
    documents = {}
    provenance = {}
    audits = {}
    gold_full = {}
    gold_accepted = {}
    score_counts = {}
    notes_by_sample = {}
    for sample in sample_ids:
        path = SAMPLES / sample / "labels.json"
        document = base._json(path)
        documents[sample] = document
        provenance[sample] = note_eval._provenance(document)
        labels = [dict(value) for value in document.get("labels") or []]
        notes = parse_sounding_notes(SAMPLES / sample / "verified_score.musicxml")
        notes_by_sample[sample] = notes
        score_counts[sample] = len(notes)
        rows = [note_eval._audit_label(value, notes) for value in labels]
        accepted = sum(bool(row["accepted"]) for row in rows)
        if not labels:
            status = (
                "empty_unreviewed"
                if provenance[sample]["empty_unreviewed"]
                else "empty"
            )
        elif accepted == len(rows):
            status = "available"
        else:
            status = "rejected"
        audits[sample] = {
            "status": status,
            "accepted_rows": accepted,
            "rejected_rows": len(rows) - accepted,
            "label_count": len(labels),
            "rows": rows,
        }
        if labels and accepted == len(rows):
            gold_full[sample] = labels
        if accepted:
            gold_accepted[sample] = [
                label for label, row in zip(labels, rows) if row["accepted"]
            ]
    return {
        "documents": documents,
        "provenance": provenance,
        "audits": audits,
        "gold_full": gold_full,
        "gold_accepted": gold_accepted,
        "score_counts": score_counts,
        "notes_by_sample": notes_by_sample,
    }


def _clip_table(gold, documents, score_counts, sample_ids) -> list[dict]:
    rows = []
    for sample in sample_ids:
        targets = gold.get(sample)
        if targets is None:
            continue
        predicted = [dict(value) for value in documents[sample].get("labels") or []]
        detail = match_note_wise_labels_detail(
            targets,
            predicted,
            score_event_count=score_counts[sample],
        )
        rows.append(
            {
                "sample": sample,
                "gold": int(detail["gold"]),
                "predicted": int(detail["predicted"]),
                "credit": float(detail["credit"]),
                "precision": float(detail["precision"]),
                "recall": float(detail["recall"]),
                "f1": float(detail["f1"]),
                "full_credit_matches": int(detail["pair_counts"]["full_credit"]),
                "half_credit_matches": int(detail["pair_counts"]["half_credit"]),
                "gold_types": dict(Counter(str(item.get("type")) for item in targets)),
                "pred_types": dict(
                    Counter(str(item.get("type")) for item in predicted)
                ),
            }
        )
    return rows


def score_models(gold, predictions, score_counts):
    if not gold:
        return {
            model: {
                "status": "unavailable",
                "reason": "no verifiable nonempty gold documents in subset",
            }
            for model in MODELS
        }
    return {
        model: note_eval._model_metrics(predictions[model], gold, score_counts)
        for model in MODELS
    }


def _style_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def plot_model_f1(metrics: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = list(MODELS)
    values = [
        0.0
        if metrics[name].get("status") == "unavailable"
        else float(metrics[name]["f1"])
        for name in names
    ]
    bars = ax.bar(names, values, color="#1b4f72", width=0.62)
    for bar, name in zip(bars, names):
        item = metrics[name]
        if item.get("status") == "unavailable":
            label = "n/a"
        else:
            label = f"{item['f1']:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, max(0.15, max(values, default=0) * 1.35 + 0.05))
    _style_axes(
        ax,
        "Model",
        "Official note-wise F1",
        "DataCreate 001–040 subset — official note-wise F1",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_per_type(metrics: dict, path: Path) -> None:
    per_type = metrics.get("per_type") or {}
    kinds = [kind for kind in TYPE_ORDER if kind in per_type] + [
        kind for kind in sorted(per_type) if kind not in TYPE_ORDER
    ]
    if not kinds:
        return
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(kinds))
    f1 = [float(per_type[kind]["f1"]) for kind in kinds]
    gold = [int(per_type[kind]["gold"]) for kind in kinds]
    pred = [int(per_type[kind]["predicted"]) for kind in kinds]
    ax.bar(x - 0.2, f1, width=0.4, color="#1b4f72", label="F1")
    ax.set_ylabel("F1")
    ax.set_xticks(x)
    ax.set_xticklabels(kinds, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    _style_axes(
        ax,
        "Error type",
        "F1",
        "Headline v3 per-type F1 with gold / predicted support",
    )
    for index, kind in enumerate(kinds):
        ax.text(
            index,
            min(0.98, f1[index] + 0.04),
            f"g{gold[index]}/p{pred[index]}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_clip_counts(clip_rows: list[dict], path: Path) -> None:
    if not clip_rows:
        return
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    names = [row["sample"] for row in clip_rows]
    x = np.arange(len(names))
    ax.bar(x - 0.18, [row["gold"] for row in clip_rows], width=0.36, color="#5d6d7e", label="Gold")
    ax.bar(
        x + 0.18,
        [row["predicted"] for row in clip_rows],
        width=0.36,
        color="#1b4f72",
        label="v3 predicted",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend(frameon=False)
    _style_axes(
        ax,
        "Clip",
        "Label count",
        "Official audited clips — gold vs v3 predicted counts",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_coverage(manifest_rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    names = [row["sample"] for row in manifest_rows]
    coverage = [
        float((row.get("diagnostics") or {}).get("mapped_score_coverage") or 0.0)
        for row in manifest_rows
    ]
    extras = [
        int(((row.get("diagnostics") or {}).get("operations") or {}).get("EXTRA") or 0)
        for row in manifest_rows
    ]
    x = np.arange(len(names))
    ax.bar(x, coverage, color="#1b4f72", width=0.72)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    _style_axes(
        ax,
        "Clip",
        "Mapped score coverage",
        "Joint decoder mapped-score coverage (bar) with EXTRA count annotated",
    )
    for index, extra in enumerate(extras):
        if extra:
            ax.text(index, min(1.0, coverage[index] + 0.02), str(extra), ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_timelines(
    sample_ids: list[str],
    gold_docs: dict,
    pred_docs: dict,
    score_counts: dict,
    path: Path,
) -> None:
    if not sample_ids:
        return
    height = max(3.8, 1.15 * len(sample_ids) + 1.2)
    fig, axes = plt.subplots(
        len(sample_ids),
        1,
        figsize=(11.2, height),
        sharex=False,
        squeeze=False,
    )
    for axis, sample in zip(axes[:, 0], sample_ids):
        n_notes = int(score_counts[sample])
        gold = gold_docs.get(sample) or []
        pred = list(pred_docs.get(sample, {}).get("labels") or [])
        axis.set_xlim(-1, max(n_notes, 1))
        axis.set_ylim(-0.8, 1.8)
        axis.set_yticks([0, 1])
        axis.set_yticklabels(["gold", "v3"])
        axis.set_ylabel(sample, rotation=0, ha="right", va="center", labelpad=18)
        axis.axhline(0, color="#dddddd", linewidth=0.6)
        axis.axhline(1, color="#dddddd", linewidth=0.6)
        for labels, y in ((gold, 0.0), (pred, 1.0)):
            for label in labels:
                span = _span(label)
                if span is None:
                    continue
                kind = str(label.get("type"))
                axis.barh(
                    y,
                    span[1] - span[0] + 1,
                    left=span[0],
                    height=0.42,
                    color=TYPE_COLORS.get(kind, "#7f8c8d"),
                    alpha=0.9,
                    linewidth=0,
                )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlabel("Score note index" if sample == sample_ids[-1] else "")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TYPE_COLORS[kind], label=kind)
        for kind in TYPE_ORDER
        if any(
            str(label.get("type")) == kind
            for sample in sample_ids
            for label in (gold_docs.get(sample) or [])
            + list(pred_docs.get(sample, {}).get("labels") or [])
        )
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=min(5, len(handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
        )
    fig.suptitle(
        "Score-note spans: gold vs headline v3 predictions",
        y=1.04 if handles else 1.01,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_time_overlay(sample: str, gold, pred, duration: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 2.8))
    ax.set_xlim(0, max(duration, 0.1))
    ax.set_ylim(-0.7, 1.7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["gold", "v3"])
    for labels, y in ((gold, 0.0), (pred, 1.0)):
        for label in labels:
            span = _time_span(label)
            if span is None:
                continue
            kind = str(label.get("type"))
            ax.barh(
                y,
                span[1] - span[0],
                left=span[0],
                height=0.45,
                color=TYPE_COLORS.get(kind, "#7f8c8d"),
                alpha=0.9,
                linewidth=0,
            )
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{sample} time spans — gold vs v3")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    manifest, errors = frozen_eval._verify_freeze(FREEZE)
    if errors:
        raise SystemExit(f"Freeze integrity failed: {errors}")
    rows = {str(row["sample"]): row for row in manifest["samples"]}
    missing = [sample for sample in KEEP if sample not in rows]
    failed = [
        sample
        for sample in KEEP
        if rows.get(sample, {}).get("status") != "succeeded"
    ]
    if missing or failed:
        raise SystemExit(f"Freeze missing/failed: missing={missing} failed={failed}")
    predictions = {
        model: {
            sample: base._json(Path(rows[sample]["paths"][model]))
            for sample in KEEP
        }
        for model in MODELS
    }
    data = inventory_and_audit(KEEP)
    gold_ids = sorted(data["gold_full"])
    accepted_ids = sorted(data["gold_accepted"])
    official = score_models(
        {sample: data["gold_full"][sample] for sample in gold_ids},
        predictions,
        data["score_counts"],
    )
    accepted = score_models(
        {sample: data["gold_accepted"][sample] for sample in accepted_ids},
        predictions,
        data["score_counts"],
    )
    empty_gold = {
        sample: []
        for sample in KEEP
        if data["audits"][sample]["label_count"] == 0
    }
    empty_as_clean = score_models(
        {
            **{sample: data["gold_full"][sample] for sample in gold_ids},
            **empty_gold,
        },
        predictions,
        data["score_counts"],
    )
    clip_rows = _clip_table(
        data["gold_full"], predictions[HEADLINE], data["score_counts"], gold_ids
    )
    nonempty = [
        sample
        for sample in KEEP
        if data["audits"][sample]["label_count"] > 0
    ]
    pred_counts = {
        sample: len(predictions[HEADLINE][sample].get("labels") or [])
        for sample in KEEP
    }
    type_counts_gold = dict(
        Counter(
            str(label.get("type"))
            for sample in KEEP
            for label in data["documents"][sample].get("labels") or []
        )
    )
    type_counts_pred = dict(
        Counter(
            str(label.get("type"))
            for sample in KEEP
            for label in predictions[HEADLINE][sample].get("labels") or []
        )
    )
    plot_model_f1(official, figures / "official_f1_by_model.png")
    if official[HEADLINE].get("status") != "unavailable":
        plot_per_type(official[HEADLINE], figures / "v3_per_type_f1.png")
    plot_clip_counts(clip_rows, figures / "audited_clip_counts.png")
    plot_coverage([rows[sample] for sample in KEEP], figures / "joint_coverage.png")
    plot_timelines(
        nonempty,
        {
            sample: [dict(value) for value in data["documents"][sample].get("labels") or []]
            for sample in nonempty
        },
        predictions[HEADLINE],
        data["score_counts"],
        figures / "score_span_timelines.png",
    )
    for sample in nonempty[:6]:
        duration = float(rows[sample]["diagnostics"]["audio"]["duration_sec"])
        plot_time_overlay(
            sample,
            [dict(value) for value in data["documents"][sample].get("labels") or []],
            list(predictions[HEADLINE][sample].get("labels") or []),
            duration,
            figures / f"time_overlay_{sample}.png",
        )

    report = {
        "schema_version": "align-datacreate-001-040-excl-eval-v1",
        "created_from_freeze": str(FREEZE),
        "freeze_sha256": base._sha256(FREEZE),
        "audio_inference_rerun": False,
        "reason_predictions_reused": (
            "The completed full stack was already frozen gold-blind on all 94 "
            "DataCreate bundles, including every requested sample, using the "
            "same joint decoder and error-heads-v3 checkpoints. Identity-CRF "
            "full-v1 currently holds the CPU lease, so audio inference was not "
            "repeated."
        ),
        "model": {
            "headline": "error-heads-v3 on frozen joint decoder + Basic Pitch 0.4.0",
            "joint": manifest["model_selection"]["joint_aligner_decoder"]["path"],
            "error_heads": manifest["model_selection"]["error_heads_v2"]["path"],
            "v3_decode": manifest["model_selection"]["error_heads_v3"]["decode_config"],
            "comparisons": list(MODELS),
        },
        "subset": {
            "requested": "DataCreate samples 001-040",
            "excluded": sorted(EXCLUDE),
            "kept": KEEP,
            "kept_count": len(KEEP),
        },
        "inventory": {
            sample: {
                "provenance": data["provenance"][sample]["category"],
                "schema_version": data["provenance"][sample]["schema_version"],
                "label_count": data["audits"][sample]["label_count"],
                "audit_status": data["audits"][sample]["status"],
                "accepted_rows": data["audits"][sample]["accepted_rows"],
                "rejected_rows": data["audits"][sample]["rejected_rows"],
                "gold_types": dict(
                    Counter(
                        str(label.get("type"))
                        for label in data["documents"][sample].get("labels") or []
                    )
                ),
                "v3_predicted": pred_counts[sample],
                "v3_pred_types": dict(
                    Counter(
                        str(label.get("type"))
                        for label in predictions[HEADLINE][sample].get("labels") or []
                    )
                ),
                "mapped_score_coverage": rows[sample]["diagnostics"].get(
                    "mapped_score_coverage"
                ),
                "operations": rows[sample]["diagnostics"].get("operations"),
                "rejection_reasons": dict(
                    Counter(
                        reason
                        for row in data["audits"][sample]["rows"]
                        for reason in row.get("reasons") or []
                    )
                ),
            }
            for sample in KEEP
        },
        "gold_type_counts": type_counts_gold,
        "v3_predicted_type_counts": type_counts_pred,
        "official_note_wise": {
            "policy": (
                "exclusive one-to-one canonical score-event identity; "
                "same type=1.0, different type=0.5, wrong location=0; "
                "clip rejected if any gold row fails identity/pitch audit; "
                "empty unreviewed documents are not clean negatives"
            ),
            "audited_clips": gold_ids,
            "rejected_nonempty_clips": [
                sample
                for sample in nonempty
                if sample not in gold_ids
            ],
            "empty_unreviewed_clips": [
                sample
                for sample in KEEP
                if data["audits"][sample]["status"] == "empty_unreviewed"
            ],
            "models": official,
            "per_clip_v3": clip_rows,
        },
        "diagnostic_accepted_rows_only": {
            "warning": "not official; mixed clips contribute only validated rows",
            "clips": accepted_ids,
            "models": accepted,
        },
        "diagnostic_empty_as_clean": {
            "warning": (
                "not official; empty unreviewed files are treated as clean "
                "negatives and unmatched predictions count as false positives"
            ),
            "models": empty_as_clean,
        },
        "figures": sorted(str(path.name) for path in figures.glob("*.png")),
    }
    base._atomic_json(OUT / "report.json", report)
    headline = official[HEADLINE]
    print(json.dumps(
        {
            "kept": len(KEEP),
            "official_clips": gold_ids,
            "headline_v3": (
                headline
                if headline.get("status") == "unavailable"
                else {
                    "precision": headline["precision"],
                    "recall": headline["recall"],
                    "f1": headline["f1"],
                    "credit": headline["credit"],
                    "predicted": headline["predicted"],
                    "gold": headline["gold"],
                    "full": headline["full_credit_matches"],
                    "half": headline["half_credit_matches"],
                }
            ),
            "report": str(OUT / "report.json"),
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
