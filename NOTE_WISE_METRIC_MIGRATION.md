# Canonical Note-wise Metric Migration (2026-09-15)

## Policy

Official scoring now uses `align-note-wise-score-event-metric-v1`.
Location is canonical score-event identity, never pitch or time. Matching is
maximum-weight exclusive one-to-one. Exact location/type receives 1.0 credit,
exact location/different type receives 0.5, and wrong location receives 0.
Precision and recall divide summed credit by prediction and gold counts.

Canonical identities support exact event sets/ranges, collapsed tied spans,
audited EXTRA/rendered identities, score-linked copy passes, and repetition
source ranges with copy count. Predictions without an auditable identity remain
unmatched. Gold without one makes that evaluation formally unavailable.

## Migrated paths

- `DataCreate/src/datacreate/melody.py`: shared canonical identity projection
  and typed fractional-credit detail API.
- `align-model/src/alignmodel/melody.py`: shared API re-export; no second metric
  implementation.
- `align-model/src/alignmodel/joint/metrics.py`: official transcription,
  alignment, copy, EXTRA, and deletion scoring; onset tolerances are diagnostics.
- `align-model/src/alignmodel/eval_melodies.py`: CLI/smoke headline now uses
  canonical identity and fails closed for unprojected schema 1.1 gold; legacy
  pitch similarity remains explicitly named.
- `align-model/src/alignmodel/joint/outputraw_metrics.py`: official combined,
  Layer 2, rhythm, repetition, mapping, and clip bootstrap metrics.
- `align-model/src/alignmodel/joint/train.py` and
  `align-model/src/alignmodel/joint/end_to_end.py`: checkpoint selection,
  early stopping, bootstrap, and promotion input use note-wise F1.
- `align-model/src/alignmodel/joint/candidate_rescorer.py`: candidate threshold
  and checkpoint selection use audited candidate-event identity; 50 ms pairing
  is diagnostic.
- `align-model/scripts/eval_error_heads_integrated.py`,
  `train_error_heads_v2.py`, and `train_error_heads_v3.py`: threshold
  calibration and schema reporting use canonical locations. V4 and V5 call
  these shared calibration/report paths.
- `align-model/scripts/train_outputraw_full_pipeline.py`: report schema v2 and
  promotion gate consume note-wise combined F1 and note-wise bootstrap.
- `align-model/scripts/validate_outputraw_full_checkpoint.py`: backward reader
  accepts historical v1 reports while new scoring is note-wise.
- `DataCreate/src/datacreate/web/compare_eval.py` and `web/app.py`: comparison
  agreement uses note identities; timestamp IoU is shown only as a diagnostic.
- `DataCreate/scripts/audit_note_wise_labels.py`: read-only conversion audit;
  it validates ranges and pitches but never writes or invents identities.
- `align-model/scripts/rescore_datacreate_note_wise.py`: hash-verified,
  gold-isolated rescore of existing frozen DataCreate predictions.

Component-local losses and diagnostics (frame classification, acoustic onset
quality, duration regression, and UI alignment timing) remain optimization or
debugging signals, not official model scores. They cannot promote a model
without the migrated end-to-end note-wise gate.

## Report field migration

- Primary: `official_note_wise`, `combined_note_wise_f1`,
  `bootstrap_note_wise_f1`, `best_validation_note_wise_f1`.
- Diagnostic: `diagnostic_timestamp_tolerances`,
  `diagnostic_timestamp_*_50ms`, `diagnostic_inferred_score_range`.
- Legacy compatibility aliases such as `tolerances`, `current_metric_50ms`,
  `timestamp_event`, and `score_range` remain readable but are not selection
  or promotion inputs.
- The completed timestamp half-credit experiment is preserved under schema
  `legacy-align-datacreate-error-heads-v3-timestamp-half-credit-report-v1`.

## Historical comparability

All existing v1 outputRaw reports, v2--v5 error-head reports, timestamp
DataCreate reports, 20/50/100 ms results, and pitch-list similarity results are
legacy diagnostics. Their numeric values are not reinterpreted. Completed or
active artifacts and production weights were not rewritten. Any active run
started under the old metric must finish as a legacy run and be rerun before
promotion.

## DataCreate audit and rescore

The read-only corpus audit found 85/94 directories evaluable at the document
level (empty documents count as structurally evaluable, not as clean negatives) and 9 with at
least one unvalidated label identity. On the exact 18 explicitly agent-labelled
non-empty clips, 11 clips with 13 supported labels passed the stricter
score-range/pitch audit; seven are officially unavailable.

Using existing hash-verified frozen predictions, official note-wise F1 on the
11 auditable clips is 0 for v3, balanced v3, and v2. Rules receive F1 0.001718
from 0.5 total credit (precision 0.000879, recall 0.038462). These are
agent-label agreement results, not independent human accuracy. The other seven
clips require corrected/reviewed canonical ranges before official scoring.

Artifacts:

- `align-model/runs/note-wise-migration-20260915/datacreate-label-audit.json`
- `align-model/runs/eval-datacreate-error-heads-v3-agent-labels-20260915/note-wise-metric-v1/report.json`
- sibling `integrity.json`

## Verification commands

```powershell
# DataCreate
$env:PYTHONPATH='src'
..\align-model\.venv-amt-bench\Scripts\python.exe -m pytest -q

# align-model
$env:PYTHONPATH='src;scripts;..\DataCreate\src'
.\.venv-amt-bench\Scripts\python.exe -m pytest -q

# synth-pipeline
$env:PYTHONPATH='src'
..\align-model\.venv-amt-bench\Scripts\python.exe -m pytest -q
```

No lockbox targets, production weights, or production promotion state were
read or modified by this migration.

Verified results: DataCreate 72 passed; align-model 226 passed;
synth-pipeline 35 passed. Focused migration suites also passed, Python
compilation succeeded, IDE lint reported no errors, and `git diff --check`
reported no whitespace errors.
