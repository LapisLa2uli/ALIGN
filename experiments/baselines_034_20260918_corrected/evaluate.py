"""Freeze baseline error spans, then score the two user-specified real test sets.

Use separate freeze/score processes. No parameters are selected on this test set.
The native note-to-span conversion is the existing eval_bridge default policy.
Timestamp results are diagnostics, not canonical score-location metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "baselines/common"))
from eval_bridge import Warner, class_from_name, notes_to_spans, read_pred_midi_tracks

FIVE_TYPES = ["wrong_note", "missed_note", "extra_note", "rhythm_error", "repetition"]
PARAMETERS = dict(tau_wrong=0.10, rep_gap=0.35, rep_gap_mode="onset",
                  rep_gap_ioi_mult=0.0, rep_break_on_correct=True, rep_trim=3,
                  rep_window_min=0.0, rep_min_run=3, rep_lcs=0.8,
                  rep_window_mult=2.0, rep_merge_gap=2.0)
CRITERIA = ["onset_50ms", "onset_100ms", "onset_200ms", "iou_0.3", "iou_0.5"]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_hashes():
    paths = [Path(__file__), ROOT / "baselines/common/eval_bridge.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def freeze(model):
    def deny_gold(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            name = str(args[0])
            if any(x in name for x in ("/bundles/", "labels.json", "label_audit.json", ".sqlite")):
                raise PermissionError(f"Gold access forbidden during prediction conversion: {name}")

    sys.addaudithook(deny_gold)
    dataset = read(HERE / "dataset_manifest.json")
    job = read(HERE / "inference_launch.json")["jobs"][model]
    source = Path(job["predictions"])
    ids = read(source / "evaluated_ids.json")
    assert len(ids) == len(set(ids)) == 40
    assert set(ids) == set(dataset["all_ids"]) == {p.parent.name for p in source.glob("*/mix.mid")}
    destination = HERE / (model + "_frozen")
    destination.mkdir(exist_ok=False)
    rows = []
    for sample in sorted(ids):
        midi = source / sample / "mix.mid"
        tracks, _, _ = read_pred_midi_tracks(str(midi), Warner())
        native = {kind: [] for kind in ("extra", "missing", "correct")}
        unknown = []
        for track in tracks:
            kind = class_from_name(track["name"])
            if kind is None and track["name"].strip() and track["notes"]:
                raise ValueError(f"Unknown named track: {model}/{sample}/{track['name']}")
            (native[kind] if kind else unknown).extend(track["notes"])
        diagnostic = {}
        spans = notes_to_spans(native, PARAMETERS, diagnostic)
        labels = [dict(type=t, start_time=float(s), end_time=float(e)) for t, s, e in spans]
        # Unclassified decoder events never receive class credit and stay in the denominator.
        labels.extend(dict(type="unclassified", start_time=float(s), end_time=float(e))
                      for s, e, _ in unknown)
        output = destination / (sample + ".json")
        write(output, dict(sample=sample, labels=labels, native_counts={k: len(v) for k, v in native.items()},
                           unclassified_count=len(unknown), conversion_diagnostics=diagnostic))
        rows.append(dict(sample=sample, midi_sha256=sha(midi), output_sha256=sha(output)))
    checkpoint = read(HERE / "checkpoint_selection.json")["models"][model]
    assert sha(Path(checkpoint["path"])) == checkpoint["sha256"]
    write(destination / "freeze_manifest.json", dict(model=model, checkpoint=checkpoint,
          parameters=PARAMETERS, code_sha256=code_hashes(), gold_access_guard="enabled",
          dataset_manifest_sha256=sha(HERE / "dataset_manifest.json"), rows=rows))
    print(f"Frozen {model}: {len(rows)} clips", flush=True)


def counts(gold, predicted, criterion, *, typed=True):
    """Maximum-cardinality exclusive matching; no partial type credit in timestamps."""
    eligible = np.zeros((len(predicted), len(gold)), dtype=np.int8)
    for i, pred in enumerate(predicted):
        for j, target in enumerate(gold):
            if typed and pred["type"] != target["type"]:
                continue
            if pred["type"] == "unclassified":
                continue
            if criterion.startswith("onset_"):
                tolerance = int(criterion.split("_")[1][:-2]) / 1000
                hit = abs(pred["start_time"] - target["start_time"]) <= tolerance + 1e-9
            else:
                overlap = max(0.0, min(pred["end_time"], target["end_time"]) -
                              max(pred["start_time"], target["start_time"]))
                union = max(pred["end_time"], target["end_time"]) - min(pred["start_time"], target["start_time"])
                hit = overlap / max(union, 1e-12) >= float(criterion.split("_")[1])
            eligible[i, j] = hit
    r, c = linear_sum_assignment(-eligible)
    return dict(matched=int(eligible[r, c].sum()), predicted=len(predicted), gold=len(gold))


def pool(rows):
    total = {k: sum(r[k] for r in rows) for k in ("matched", "predicted", "gold")}
    m, p, g = (total[k] for k in ("matched", "predicted", "gold"))
    return dict(**total, precision=m / p if p else 0.0, recall=m / g if g else 0.0,
                f1=2 * m / (p + g) if p + g else 0.0)


def bootstrap(rows):
    rng = np.random.default_rng(365)
    values = [pool([rows[int(i)] for i in rng.integers(0, len(rows), len(rows))])["f1"]
              for _ in range(2000)]
    return dict(unit="clip", seed=365, replicates=2000,
                lower_95=float(np.quantile(values, .025)), upper_95=float(np.quantile(values, .975)))


def score():
    dataset = read(HERE / "dataset_manifest.json")
    audit = {r["sample"]: r for r in read(HERE / "label_audit.json")["samples"]}
    docs = {sample: read(HERE / "bundles" / sample / "labels.json") for sample in dataset["all_ids"]}
    all_types = sorted({label["type"] for doc in docs.values() for label in doc["labels"]})
    result = dict(dataset=dataset, metric_status="timestamp error-event diagnostic; not canonical combined-pipeline F1",
                  matching="typed maximum-cardinality one-to-one; full credit only",
                  mapping_parameters=PARAMETERS, primary_taxonomy=FIVE_TYPES,
                  unsupported_native_types=[t for t in all_types if t not in ("wrong_note", "missed_note", "extra_note", "repetition")],
                  normal_notes_in_metric=False, gold_sha256={s: sha(HERE / "bundles" / s / "labels.json") for s in docs},
                  models={})
    for model in ("polytune", "laddersym"):
        folder = HERE / (model + "_frozen")
        manifest = read(folder / "freeze_manifest.json")
        assert manifest["code_sha256"] == code_hashes()
        assert manifest["dataset_manifest_sha256"] == sha(HERE / "dataset_manifest.json")
        predictions = {}
        for row in manifest["rows"]:
            path = folder / (row["sample"] + ".json")
            assert sha(path) == row["output_sha256"]
            predictions[row["sample"]] = read(path)["labels"]
        report = dict(checkpoint=manifest["checkpoint"], freeze_manifest_sha256=sha(folder / "freeze_manifest.json"), subsets={})
        for subset, key in (("full40", "all_ids"), ("filtered30", "filtered_ids")):
            ids = dataset[key]
            sub = dict(sample_ids=ids, count=len(ids), results={})
            # Audit only the five task types. No clips are removed from the requested metrics.
            invalid = {s: [r for r in audit[s]["labels"] if r["type"] in FIVE_TYPES and not r["passed"]]
                       for s in ids}
            invalid = {s: rows for s, rows in invalid.items() if rows}
            sub["canonical_note_wise"] = dict(status="unavailable", invalid_gold=invalid,
                reason="Some gold score locations fail the existing audit; no subset silently substituted.")
            sub["raw_gold_type_counts"] = dict(Counter(l["type"] for s in ids for l in docs[s]["labels"]))
            sub["empty_gold_clips"] = [s for s in ids if not docs[s]["labels"]]
            for taxonomy, kinds in (("five_type", FIVE_TYPES), ("shared_four_type", [t for t in FIVE_TYPES if t != "rhythm_error"]),
                                    ("all_annotated_types", all_types)):
                gold = {s: [l for l in docs[s]["labels"] if l["type"] in kinds] for s in ids}
                pred = {s: [l for l in predictions[s] if l["type"] in kinds or l["type"] == "unclassified"] for s in ids}
                metrics = {}
                for criterion in CRITERIA:
                    rows = {s: counts(gold[s], pred[s], criterion) for s in ids}
                    micro = pool(list(rows.values()))
                    per_type = {t: pool([counts([l for l in gold[s] if l["type"] == t],
                                                [l for l in pred[s] if l["type"] == t], criterion) for s in ids]) for t in kinds}
                    supported = [v["f1"] for v in per_type.values() if v["gold"]]
                    metrics[criterion] = dict(micro=micro, bootstrap=bootstrap(list(rows.values())),
                        macro_f1=float(np.mean(supported)) if supported else None,
                        macro_categories=[t for t in kinds if per_type[t]["gold"]], per_type=per_type, per_clip=rows)
                sub["results"][taxonomy] = metrics
            report["subsets"][subset] = sub
        result["models"][model] = report
    write(HERE / "results.json", result)
    rows = ["subset,model,taxonomy,criterion,clips,matched,predicted,gold,precision,recall,f1,ci_low,ci_high"]
    for model, report in result["models"].items():
        for subset, sub in report["subsets"].items():
            for taxonomy, metrics in sub["results"].items():
                for criterion, metric in metrics.items():
                    m, b = metric["micro"], metric["bootstrap"]
                    values = [subset, model, taxonomy, criterion, sub["count"], m["matched"], m["predicted"], m["gold"],
                              m["precision"], m["recall"], m["f1"], b["lower_95"], b["upper_95"]]
                    rows.append(",".join(map(str, values)))
    (HERE / "summary.csv").write_text("\n".join(rows) + "\n")
    print(HERE / "results.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "score"))
    parser.add_argument("--model", choices=("polytune", "laddersym"))
    args = parser.parse_args()
    if args.mode == "freeze":
        if not args.model:
            parser.error("freeze requires --model")
        freeze(args.model)
    else:
        score()
