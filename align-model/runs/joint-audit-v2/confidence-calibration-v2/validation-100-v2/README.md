# Candidate confidence calibration v2

Status: validation-only diagnostic. No model training ran, no production
default changed, and the locked test split was not read.

## Selection and method

The sweep used the audited manifest's first 100 validation rows. The manifest
documents deterministic corpus/source round-robin ordering, and this prefix is
balanced: 50 `raw2k_local` / soundfont rerenders and 50 `procedural12k` /
oscillator renders; 76 rows contain repetition. It covers 6,616 rendered
target notes. A full 1,871-row run was not launched while the existing
CPU/GPU jobs were active: the cache-only 100-row pass was sufficient to
reproduce the reference and resolve the direction of the tradeoff.

Each row's hash-validated Basic Pitch cache was loaded and decoded once with
candidate generator `align-joint-candidates-v2-short-rescue`; the four gates
were then evaluated against exact audited rendered lineage. Confidence
reliability labels and all timing matches use exact pitch plus onset.
Intervals are 2,000-replicate sample bootstraps.

## Main results

At the standard 50 ms tolerance:

| Gate | Precision | Recall | F1 | Pred/target | Unmatched-prediction FP | Copy coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.794965 | 0.892533 | 0.840929 | 1.122733 | 0.205035 | 0.888433 |
| 0.55 | 0.799918 | 0.889510 | 0.842339 | 1.112001 | 0.200082 | 0.882090 |
| 0.60 | 0.803422 | 0.887092 | 0.843187 | 1.104141 | 0.196578 | 0.878731 |
| 0.65 | 0.816854 | 0.879081 | 0.846826 | 1.076179 | 0.183146 | 0.864552 |

The 0.65 F1 is within 0.000370 of the prior validation-100 reference
0.847196. Its 95% bootstrap interval is 0.825982--0.870565. The intervals
overlap, but every lower gate has a lower aggregate 50 ms point estimate and
more unmatched predictions. Relative to 0.65, gate 0.50 gains 1.35 recall
points and 2.39 copy-coverage points, but loses 0.59 F1 points and adds 2.19
false-positive-rate points.

F1 at 20/50/100 ms respectively:

- 0.50: 0.529194 / 0.840929 / 0.933068
- 0.55: 0.529163 / 0.842339 / 0.937379
- 0.60: 0.529560 / 0.843187 / 0.938726
- 0.65: 0.531596 / 0.846826 / 0.929674

The 100 ms optimum at 0.60 does not carry to strict timing or the standard
50 ms metric. At 50 ms the soundfont/raw subset remains the limiting domain:
F1 is 0.771384 at gate 0.65 and 0.769818 at 0.50, while oscillator/procedural
F1 falls from 0.987094 to 0.974759.

## Coverage and calibration

Lower gates materially rescue the rare ultra-short audited events. Cumulative
recall for notes under 120 ms rises from 9/33 (0.272727) at 0.65 to 22/33
(0.666667) at 0.55 and 24/33 (0.727273) at 0.50. Under 180 ms, recall rises
from 0.831953 to 0.852655. Relationship recall at 0.50 is match 0.897180,
copy 0.888433, extra 0.879630, and substitute 0.881188.

The 0.45--0.65 reliability band is not monotonically calibrated. Exact 50 ms
empirical precision by confidence bin is:

- 0.45--0.50: 25/43 = 0.581395, mean confidence 0.478679.
- 0.50--0.55: 34/71 = 0.478873, mean confidence 0.527935.
- 0.55--0.60: 17/52 = 0.326923, mean confidence 0.575029.
- 0.60--0.65: 58/185 = 0.313514, mean confidence 0.637211.

This inversion means candidate confidence should not be treated as a globally
calibrated correctness probability. The detailed report includes cumulative
duration recall, all four relationship recalls, corpus/render/source
breakdowns, sample rows, and the oracle-mapping acoustic ceiling at every
tolerance.

## Recommendation to the optimizer

Keep the production admission threshold at 0.65 for now. Do not promote 0.50,
0.55, or 0.60 as a global hard gate: none improves standard 50 ms candidate
F1 or its oracle-mapping ceiling without extra-note inflation. For retraining,
admit or sample the 0.50--0.65 band and let a confidence-aware path score use
onset, duration, render/source, and structural evidence, with short-note
positive weighting and hard negatives. In particular, use 0.50/0.55 as
candidate-coverage ablations for sample 007, not production defaults.

## Artifacts

- `report.json`: full metrics, provenance, bootstrap intervals, breakdowns,
  reliability bins, and per-sample counts.
- `smoke-1.json`: one-row evaluator smoke test.
- `scripts/eval_candidate_confidence.py`: standalone cache-only evaluator.

Re-run from `align-model`:

```powershell
$env:PYTHONPATH = "src;..\synth-pipeline\src"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
.\.venv-amt-bench\Scripts\python.exe scripts\eval_candidate_confidence.py `
  --manifest data-audit\2026-09-14-v2\split.json --split val `
  --cache-root runs\joint-audit-v2\basic-pitch-cache `
  --output runs\joint-audit-v2\confidence-calibration-v2\validation-100-v2\report.json `
  --limit 100 --bootstrap-replicates 2000
```
