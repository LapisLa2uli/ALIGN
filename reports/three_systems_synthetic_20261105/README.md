# Synthetic benchmark

Same 30 synthetic clips; 135 gold content errors. Seed 20261105 selected 30 of 87 eligible audited dataset-9.2 test clips (12 WeberITAV, 10 Cruel Angel, 8 procedural).

| System | Content-error micro-F1 | 95% CI |
|---|---:|---:|
| AudioEval + MuScriptor medium | 36.24% | 22.12-54.28% |
| PolyTune | 2.26% | 0.84-4.46% |
| LadderSym | 2.71% | 0.93-6.48% |

This measures affected-score-note content-error detection, NOT the earlier pitch-sequence transcription F1. Wrong, missing, extra and repeated notes are included; timing and intonation are excluded. Exact location and type earn one credit; matching location with different type earns half credit. F1 = 2*credit/(predicted+gold). Duplicate matching is exclusive. Unlocated/unclassified predictions count in the denominator without credit. Context padding is stripped and the project's first-pass-only rule applies.

AudioEval uses its existing v9 mapper, acoustic evidence and gates, with hash-verified cached MuScriptor medium transcriptions. One inference failure (synth_WeberITAV_27392, empty acoustic feature slice) and one alignment abstention (synth_gen_22820) remain in the evaluation. No test-outcome tuning was applied.

PolyTune uses author-pretrained CocoChorales-E; LadderSym uses author-pretrained CocoChorales prompted. ALIGN-retrained checkpoints were unavailable. AudioEval includes dataset-trained components, so training exposure is not matched. Domain shift and native-output conversion may affect the baseline results; this small experiment does not establish general superiority. Baselines receive original audio resampled to mono 16 kHz; LadderSym also receives reference MIDI. Native outputs use the existing canonical adapter without gold-assisted alignment. Inference denies access to gold-label files.

95% percentile intervals use 2,000 clip-level bootstrap resamples (seed 20261105), with the same indices for every system. These quantify clip sampling, not generalization across independent musical works.

Full precision scores and per-clip counts: results.json. Inputs and checkpoint hashes: manifest.json. Canonical gold: gold.json. Native and converted predictions are retained. Original dataset files were not modified.

Reproduce from repository root using baselines/envs/polytune/Scripts/python.exe -u align-model/scripts/benchmark_three_systems_synthetic.py STAGE for stages prepare, audioeval, polytune, laddersym, score. Inference resumes from completed predictions. Generate PNG/SVG/PDF with python reports/three_systems_synthetic_20261105/plot_results.py.
