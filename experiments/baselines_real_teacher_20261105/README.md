# Real-recording fine-tuning experiment

Compare independently fine-tuned PolyTune and prompted LadderSym at 0, 10, 20, 30, 40, and 50 real DataCreate recordings. Supervision and test references are AudioEval v9 + MuScriptor medium predictions. The experiment measures agreement with this teacher; the references are not human-verified ground truth.

**Completed outcome:** all ten fine-tunes and all 252 test decodes finished. Every fine-tuned condition scored below its original checkpoint on strict reference-score error-label F1. See `RESULTS.md` for the learning curve and `ANALYSIS.md` for the observed failure modes. Final integrity checks passed.

**Follow-up diagnosis:** `REGRESSION_CAUSES.md` documents missing musical correspondence in reference conditioning, a reproduced LadderSym generation loop, and the adapter's inability to assign ordinary extra-note locations. These findings qualify the interpretation of the negative result; the frozen results remain unchanged.

## Data and evaluation

- The test range is **100–120 inclusive**. Decode all 21 recordings for every condition.
- The teacher abstained on **101, 105, 114, and 117**. Exclude these four from supervised scoring, rather than treating their empty labels as correct performances. Keep all predictions.
- The 17 usable test references contain 76 labels: 49 wrong notes, 13 repetitions, 9 missed notes, and 5 extra-note labels. A label can cover multiple reference notes.
- Training uses nested subsets from 001–065, selected before test scoring. The selected 50 IDs and seed are frozen in `protocol.json`. Teacher abstentions and examples with unrepresentable supervision are excluded before sampling.
- The primary metric requires the exact canonical affected-score-note set **and error type**, with one-to-one label matching. Repetitions also require the correct copy count. Thus it measures exact location of an error label, not onset/offset tolerance. Partially overlapping ranges receive no strict credit. GUI context padding is stripped first. The existing half-credit-for-type-mismatch score is secondary.
- The native output adapter is fixed and cannot see teacher labels. Unlocated and unclassified predictions count as false positives.

## Training

Each nonzero condition starts from the original author checkpoint with a fresh optimizer. There is no sequential continuation from 10 to 20 examples. Each run gets 200 optimizer updates, four individual windows per update, AdamW at peak learning rate 1e-5, warmup over 20 steps, cosine decay to a floor of 10%, gradient clipping at 1, and the original model-specific weighted cross-entropy formula. All trainable parameters are fine-tuned; fixed positional embeddings remain fixed.

Sampling is uniform over recordings, then over cached 2.048-second windows. The 50-case set has 1,326 accepted windows per model. Every condition gets the same number of sampled windows, so larger datasets get fewer revisits per recording. This is a comparison at a fixed update budget, not a search for each dataset size's best possible result. The final update is evaluated; no epoch or hyperparameter is selected using the test set.

Training uses bf16 autocast with float32 parameters and optimizer state. Inference uses float32, batch size one, 1,024 generated-token limits, and a deterministic 1,024-token symbolic prompt for LadderSym. Use the existing compatible PolyTune environment for both baselines.

Teacher labels are converted to native Correct / Extra / Missing note targets. Wrong notes become Extra(heard pitch) plus Missing(expected pitch); replay notes become Extra. Only aligned matches become Correct. Windows containing withheld candidates or unassessed score regions are excluded. Missing-note timestamps are teacher estimates used for training, not the primary scoring target. All 6,567 whole-recording targets in the chosen 50 cases passed a pitch/timing MIDI round trip; some are outside accepted training windows.

## Input and label limitations

MuScriptor sometimes predicts events beyond the waveform. A uniform correction removed notes starting at/after the actual audio end and clipped note offsets; affected cached cases were re-aligned. See `teacher_bounds_audit.json`. This correction was applied before baseline training or scoring.

No identical audio, full canonical pitch sequence, or matching source filename was detected across training and test. Available source-score names are disjoint. Some earlier samples lack source metadata, so this does not independently prove that no shorter musical passage is shared. Duplicate pieces within the training pool are allowed; sample counts refer to recordings, not unique compositions.

Original rendered reference audio/MIDI is preserved. The rendered MIDI and canonical score-note parser do not always have the same number of notes: MIDI may expand repeats, ornaments, or chords, while the existing canonical parser collapses ties and ignores decorative events and chord objects. Examples in the test set are 100 (52 MIDI notes / 42 canonical notes), 104 (111 / 37), and 109 (126 / 63). This can affect conditioning and error localization. `data_audit.json` lists every pitch-count difference. No input was changed in response to a baseline's test output.

One subset ordering and training seed are used. Paired bootstrap intervals over the 17 scored recordings describe test-sample uncertainty, not variance across training seeds. Sparse or incorrect teacher decisions and the fixed three-class-to-error-label adapter remain limitations.

## Artifacts and reproduction

- `protocol.json`: frozen IDs, hyperparameters, source/checkpoint/data hashes.
- `run_campaign.py`: sequential, resumable training, inference, then scoring.
- `campaign_status.json`, `campaign_logs/`: current status and individual logs.
- `training/<model>/nNN/model.pt`: independent fine-tuned weights and training records.
- `predictions/<model>/nNN/<sample>/mix.mid`: all 252 test outputs, with hash/provenance sidecars.
- `converted/`: baseline outputs expressed as reference-score error labels.
- `teacher_gold/`: canonical teacher references, including explicit abstentions.
- `test_teacher_ui/`: manifest and backups for the requested 100–120 agent-label publication. Human labels are preserved.
- `results.json`, `RESULTS.md`, `learning_curve.png`: generated after every inference completes.
- `data_audit.json`, `final_audit.json`: integrity checks before and after the campaign.

From the repository root in PowerShell:

```powershell
$env:PYTHONNOUSERSITE='1'
& baselines/envs/polytune/Scripts/python.exe -u experiments/baselines_real_teacher_20261105/run_campaign.py
& baselines/envs/polytune/Scripts/python.exe -u experiments/baselines_real_teacher_20261105/audit_experiment.py --final
```

Completed checkpoints and predictions are hash-checked and reused on restart. Original model checkpoints and application defaults are preserved. The test teacher labels are published in the GUI; the experimental baseline variants are retained separately so they can be compared without overwriting each other.

The frozen scorer originally lacked a `json` import used only by its final console print, after saving all results. `score_entrypoint.py` supplies that standard-library global without modifying the frozen scoring source or any metric. The reporting step was rerun successfully with `run_campaign.py --score-only`; no training or inference was rerun for this fix.
