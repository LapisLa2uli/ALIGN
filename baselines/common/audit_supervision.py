"""Conservatively exclude inconsistent legacy labels without inventing new truth.

Keeps split.json and all data files untouched; writes split.audited.json and a
reasoned exclusion report. Reuses each accepted track's original split.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from prepare_dataset import classes_from_note_labels


def issues_for_labels(labels):
    reasons = []
    try:
        classes, _ = classes_from_note_labels(labels)
    except (KeyError, TypeError, ValueError) as error:
        return [f"invalid_labels: {error}"]
    checks = labels.get("check", {})
    if not all(checks.get(key, {}).get("ok") is True for key in ("perf", "ref")):
        reasons.append("unverified_midi_consistency")
    references = labels["reference_notes"]
    n_reference = sum(not row.get("tie_prev", False) for row in references)
    if len(classes["correct"]) + len(classes["removed"]) != n_reference:
        reasons.append("reference_note_count_mismatch")
    if any(row.get("cls") == "correct" and not row.get("tie_prev")
           and row.get("perf_index") is None for row in references):
        reasons.append("correct_reference_without_performance")
    if any(row.get("cls") not in ("correct", "missed") for row in references):
        reasons.append("unknown_reference_class")
    # A tie partly removed/changed has no unambiguous three-class audible-note
    # interpretation here. Exclude instead of relabelling the whole tied event.
    for index, row in enumerate(references):
        if row.get("tie_prev") and (index == 0 or row.get("cls") != references[index - 1].get("cls")):
            reasons.append("mixed_class_reference_tie")
            break
    return sorted(set(reasons))


def audit(root, source_roots):
    original = (root / "split.json").read_bytes()
    split = json.loads(original)
    manifest = json.loads((root / "manifest.json").read_text())
    excluded, reason_counts, counts, accepted = {}, Counter(), Counter(), set()
    for key, filename in split["midi_filename"].items():
        tid = Path(filename).name.replace(".midi", "")
        rec = manifest["tracks"][tid]
        if rec.get("real_test"):
            raise ValueError("Audit supervised synthetic data only, not real-test placeholders")
        bundle = Path(rec["bundle"])
        if rec["set"] in source_roots:
            bundle = source_roots[rec["set"]] / bundle.name
        labels_path = bundle / "note_labels.json"
        # Missing sources are an audit error, not evidence that the track is bad.
        labels = json.loads(labels_path.read_text())
        reasons = issues_for_labels(labels)
        if not reasons:
            classes, _ = classes_from_note_labels(labels)
            if any(len(classes[k]) != rec["n_" + k] for k in classes):
                reasons.append("source_manifest_count_mismatch")
        if reasons:
            excluded[tid] = {"split": split["split"][key], "set": rec["set"], "reasons": reasons}
            reason_counts.update(reasons)
        else:
            accepted.add(key)
            counts[split["split"][key]] += 1
    if any(counts[key] == 0 for key in ("train", "validation", "test")):
        raise ValueError(f"Audited data must retain all three splits: {counts}")
    filtered = {name: {key: value for key, value in split[name].items() if key in accepted}
                for name in ("midi_filename", "split")}
    payload = (json.dumps(filtered, indent=1) + "\n").encode()
    report = {"original_tracks": len(split["split"]), "accepted_tracks": len(accepted),
              "excluded_tracks": len(excluded), "accepted_splits": dict(counts),
              "reasons": dict(reason_counts), "original_split_sha256": hashlib.sha256(original).hexdigest(),
              "audited_split_sha256": hashlib.sha256(payload).hexdigest(),
              "scope": "All source JSON semantics and manifest counts; does not re-render audio or reconstruct missing labels.",
              "excluded": excluded}
    (root / "split.audited.json").write_bytes(payload)
    (root / "supervision_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=BUNDLE_ROOT")
    args = parser.parse_args()
    roots = {name: Path(path) for name, path in (s.split("=", 1) for s in args.set)}
    result = audit(args.data.resolve(), roots)
    print(json.dumps({k: v for k, v in result.items() if k != "excluded"}, indent=2))
