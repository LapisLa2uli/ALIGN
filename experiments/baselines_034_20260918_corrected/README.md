# Corrected 034.zip evaluation

This revision replaces full bundles 036–040 with `036-040.zip`. The previous run is preserved at `../baselines_034_20260918`. `RESULTS.md`, `results.json`, and `summary.csv` in this directory are the current results.

All model input files are byte-identical to the earlier run (160 SHA-256 comparisons). Only gold documents 036 and 039 changed. All 80 original MIDI predictions were hash-verified and reused; no inference rerun or test-set tuning was needed. The fixed checkpoint choices, conversion parameters, taxonomy, matching, and bootstrap protocol are unchanged.

The all-40 selection has 46 five-type gold events; the user-specified filtered-30 selection has 21. The excluded IDs remain 005, 007, 010, 012, 013, 020, 026, 030, 034, 036. The separate all-annotated-types view contains 63 / 32 gold events. User confirmation that empty label files are clean negatives continues to apply.

`inference_launch.json` records the reused original launches. `prediction_reuse_audit.json` proves prediction/input identity; `source_integrity.json` verifies every relevant bundle file against its correct archive. The revised label audit and all frozen converted predictions are retained locally. Canonical scores remain unavailable for both requested selections due to unchanged invalid gold locations; no clips are silently removed.

To reproduce scoring from the corrected frozen outputs:

```sh
align-model/runs/env/bin/python experiments/baselines_034_20260918_corrected/evaluate.py score
```

The prediction conversion script is a byte-identical copy of the previous version. No model or paper files were changed.
