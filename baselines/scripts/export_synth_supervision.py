"""Create audited legacy-format targets for current bundles without modifying them."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

COMMON = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(COMMON))
from synth_supervision import SupervisionError, export_bundle
from audit_supervision import issues_for_labels


def export_one(job):
    name, bundle_text, config_text, out_text = job
    bundle, config, out = Path(bundle_text), Path(config_text), Path(out_text)
    target = out / name / bundle.name
    label_path = target / "note_labels.json"
    implementation = hashlib.sha256((COMMON / "synth_supervision.py").read_bytes()).hexdigest()
    try:
        if label_path.exists():
            labels = json.loads(label_path.read_text())
            provenance = labels["provenance"]
            if provenance.get("implementation_sha256") != implementation or provenance["config_sha256"] != hashlib.sha256(config.read_bytes()).hexdigest():
                raise ValueError("Cached supervision implementation/config changed; use a new output root")
            if any(hashlib.sha256((bundle / f).read_bytes()).hexdigest() != sha for f, sha in provenance["source_sha256"].items()):
                raise ValueError("Cached supervision source changed; use a new output root")
        else:
            labels = export_bundle(bundle, config)
            labels["provenance"]["implementation_sha256"] = implementation
        reasons = issues_for_labels(labels)
        if reasons:
            raise ValueError(",".join(reasons))
        target.mkdir(parents=True, exist_ok=True)
        for source in bundle.iterdir():
            link = target / source.name
            if link.is_symlink():
                if link.resolve() != source.resolve():
                    raise ValueError(f"Unexpected source link: {link}")
            elif link.exists():
                raise ValueError(f"Unexpected existing file: {link}")
            else:
                link.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        temp = label_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(labels, indent=2) + "\n")
        temp.replace(label_path)
        return dict(set=name, sample=bundle.name, status="accepted",
                    correct=sum(p["cls"] == "correct" for p in labels["performance_notes"]),
                    extra=sum(p["cls"] == "extra" for p in labels["performance_notes"]),
                    missing=len(labels["missed_notes"]))
    except SupervisionError as exc:
        # Persisted accepted caches are never silently treated as exclusions.
        if label_path.exists():
            raise
        return dict(set=name, sample=bundle.name, status="excluded", reason=f"{type(exc).__name__}: {exc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", action="append", required=True, metavar="NAME=ROOT")
    parser.add_argument("--config", action="append", required=True, metavar="NAME=YAML")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    roots = {name: Path(path).resolve() for name, path in (s.split("=", 1) for s in args.set)}
    configs = {name: str(Path(path).resolve()) for name, path in (s.split("=", 1) for s in args.config)}
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(name, str(sample), configs[name], str(args.out.resolve()))
            for name, root in roots.items() for sample in sorted(root.glob("synth_*"))[:args.limit]
            if sample.is_dir()]
    rows, counts = [], Counter()
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.out / "export_progress.jsonl").open("w") as log:
        for i, row in enumerate(pool.map(export_one, jobs, chunksize=4), 1):
            rows.append(row)
            counts[row["status"]] += 1
            log.write(json.dumps(row) + "\n")
            if i % 100 == 0 or i == len(jobs):
                log.flush()
                progress = dict(processed=i, expected=len(jobs), counts=dict(counts), elapsed_seconds=time.time() - started)
                (args.out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                print(json.dumps(progress), flush=True)
    report = dict(expected=len(jobs), counts=dict(counts),
                  by_set={name: dict(Counter(row["status"] for row in rows if row["set"] == name)) for name in roots},
                  exclusion_reasons=dict(Counter(row["reason"] for row in rows if row["status"] == "excluded")),
                  samples=rows)
    (args.out / "supervision_export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()
