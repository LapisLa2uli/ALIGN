# Available baselines for ALIGN: architecture, software, and datasets

Research date: **24 October 2026**. Scope: ALIGN / MusicEval in this workspace, a score-informed music-performance error detector. This is a literature and source-code review, not a new benchmark run.

## Findings and recommended baseline set

**LadderSym and Polytune are the closest available end-to-end baselines.** Both compare a performance recording with reference-score audio and generate notes labeled **Correct, Missed, or Extra**. LadderSym additionally supports symbolic score prompting. They cover ALIGN's note-content assessment task, but do not natively reproduce its score-event identities, explicit repetition labels, rhythm-error labels, or intonation measurements. A wrong note is represented as a missed expected note plus an extra played note. [1, 2]

Prioritize **LadderSym prompted, LadderSym unprompted (Ladder), and Polytune**, evaluated on the same frozen data. Add **MT3 + explicit alignment** to distinguish transcription failures from comparison failures. For an alignment-component comparison, use **TheGlueNote or DualDTW through Parangonar** on the same transcribed notes. Basic Pitch is a useful low-cost transcription reference; Matchmaker is relevant to live score following rather than complete error detection.

| Candidate | Inputs and outputs | Fit to ALIGN | Availability verified |
|---|---|---|---|
| **LadderSym, prompted** | Performance WAV + reference WAV + reference MIDI -> class-labeled MIDI | Closest direct baseline when the clean score is available | Code; official piano and Coco checkpoints; datasets. Root license and checkpoint card: MIT. [1, 3, 5] |
| **Ladder / unprompted LadderSym** | Two WAV streams -> class-labeled MIDI | Separates symbolic prompting from the encoder change | Same repository; unprompted checkpoints for both datasets. [3, 5] |
| **Polytune** | Two WAV streams -> class-labeled MIDI | Direct predecessor and essential comparison | Code and official checkpoints. **Non-commercial restriction** in code license. [2, 4, 6] |
| **MT3 + DTW + note comparison** | Audio -> MIDI; align against score -> discrepancies | Explicit-alignment baseline with different failure modes | PyTorch implementation in Polytune; original T5X code/checkpoints separately. [2, 10, 11] |
| **TheGlueNote / Parangonar** | Two symbolic note sequences -> correspondences and unmatched notes | Alignment-component baseline for structural mismatch | Code and pretrained models; Apache-2.0. Needs an audio front end. [12, 13] |
| **Basic Pitch + alignment** | Audio -> notes/onsets/pitch contours -> external score comparison | Lightweight modular baseline | Code and pretrained model; Apache-2.0. [14, 15] |
| **Matchmaker** | Streaming audio/MIDI + score -> current score position | Online alignment comparison only | Installable Apache-2.0 library with multiple score followers. [16] |

“Available” means public source, documentation and asset listings were accessible. Large datasets and weights were not downloaded or executed. Polytune is source-available for non-commercial use; its additional restriction means it should not be described as unrestricted open-source software.

## 1. What is being compared with ALIGN?

The workspace has three cooperating packages: `DataCreate` for ingestion, annotation and review; `synth-pipeline` for controlled performances and exact note lineage; and `align-model` for transcription, score alignment and error labeling. `DOCUMENTATION.md` distinguishes the GUI's joint path CRF from the frozen stack-v6 experiment pipeline. An opt-in solo transcriber also extends coverage to 24 instruments; this alone does not establish equivalent error-detection performance across those instruments. [L1, L4]

```text
ALIGN:       recording -> note transcription -> score alignment -> score-linked error labels
Polytune:    recording + rendered score -> dual audio encoders -> late fusion -> error-note tokens
LadderSym:   recording + rendered score -> interleaved audio encoders --+
             symbolic clean score -> decoder prompt ------------------+-> error-note tokens
TheGlueNote: transcribed notes + score notes -> learned similarities -> explicit note matching
```

ALIGN's official metric uses canonical score-event identity, exclusive one-to-one matching and partial credit for the correct location with the wrong type. The papers use category-specific onset/pitch note F1. These cannot be combined into a leaderboard without harmonizing the protocol. Bb clarinet also requires consistent pitch space: ALIGN's written pitch is two semitones above the sounding audio. [L1, L2]

## 2. Polytune: software and model structure

**Paper:** Chou et al., *Detecting Music Performance Errors with Transformers*, AAAI 2025; full arXiv:2501.02030v1 PDF accessed. [2]

### Computation and representation

1. Obtain a reference audio rendition of the correct score and a performance recording. MIDI may synthesize the reference, but the main model receives audio rather than score tokens.
2. Compute spectrograms and divide both streams into short segments. The paper describes 2.145-second segments and 512 patch tokens per stream. The released frontend uses 16 kHz audio, a 2048-point FFT, a 128-sample hop, and 512 mel bins. Configuration `mel_length: 256` corresponds to a 2.048-second hop grid. Preserve this distinction rather than assuming manuscript and implementation timing are identical.
3. Apply separate AST/ViT-style encoders. Source parameter `modality_specific_depth=11` keeps the streams separate for the first 11 layers, then uses a joint encoder layer after concatenation.
4. Project encoder features from 768 to the T5 decoder's 512 dimensions.
5. Greedily decode MIDI-like events autoregressively: time, pitch, note-on/off, tie handling, sequence boundaries and error labels. The released decoder has eight layers, six attention heads and a 1,536-entry embedding vocabulary allocation. That is not the number of error classes.
6. Reassemble segments into MIDI with Extra, Missing and Correct tracks. [2, 4]

The main idea is **learned comparison with late fusion** rather than a hard alignment followed by comparison. It is still a chunked system: long tempo drift, restarts or repetitions can put corresponding notes outside the same context. Learned alignment is not a guarantee of arbitrary long-range score following.

### Code map

Paths are relative to upstream Polytune at `d2055bb21759d457c8f21c1cf2e47c79af6248f5`, also the public main-branch head observed during this review. [4]

| File or directory | Responsibility |
|---|---|
| `train_polytune.py` | Hydra-configured training entry point |
| `tasks/polytune_net.py` | Lightning training/validation, losses and optimizer integration |
| `models/ast.py` | Paired spectrogram encoder and fusion |
| `models/polytune.py` | AST encoder connected to modified T5 generation model |
| `dataset/dataset_2_random.py`, `dataset_2_random_coco.py` | Paired data discovery, segment selection and targets |
| `contrib/` | Spectrograms, event codec, vocabulary, MIDI/note conversion and run-length encoding |
| `config/` | Dataset, model, optimizer, trainer and inference settings |
| `inference_error.py`, `polytune*_test_inference.py` | Chunked generation and output reconstruction |
| `evaluate_errors*.py` | Error-note evaluation |
| `baseline/MT3_baseline/` | Separate transcription-plus-alignment baseline |

The stack is Python/PyTorch, PyTorch Lightning, Hydra, Hugging Face Transformers, audio processing and MIDI utilities. The paper reports PyTorch 2.3.0, Transformers 4.40.1 and an A100 80 GB GPU. Reusing AST/T5 architecture does not imply pretrained language-model weights. The paper favors supervised task training after finding little benefit from self-supervised pretraining, while the released MAESTRO config anticipates a CocoChorales warm start. Record initialization per experiment. [2, 4]

### Training and available weights

The paper's weighted token cross-entropy gives error tokens weight 10 and other tokens weight 1. Released configs instead contain `error_loss_weight: 8`, `lr: 2e-5`, and 350 epochs for MAESTRO or 800 for Coco. They are starting configurations, not a complete provenance record of the published run. [2, Section 4.4; 4]

Public weights include `CocoChorales-E/last.ckpt`, `MAESTRO-E/best.ckpt`, and `MAESTRO-E/robust-piano-v3/model.ckpt`. The robust variant adds fine-tuning on expanded piano synthesis with original-example replay; it is separate from the original paper checkpoint. [6]

## 3. LadderSym: software and model structure

**Paper:** Chou et al., *LadderSym: A Multimodal Interleaved Transformer for Music Practice Error Detection*, ICLR 2026; full arXiv:2510.08580v2, dated 4 March 2026, accessed. The OpenReview PDF endpoint returned HTTP 403; the arXiv PDF and HTML were accessible. [1]

### Changes relative to Polytune

**Ladder encoder.** Cross-attention, additive fusion and transformer processing alternate between streams. Information from one stream is aligned to the other and added to its representation before feature processing; the direction then reverses. This enables repeated comparison while preserving the different roles of reference and performance.

The code creates `depth=12` and iterates over the block list in pairs: one block processes the mistake stream and the next the reference stream. It should not casually be described as two independent 12-block encoders. Final stream features are concatenated and projected for the decoder. The paper reports approximately **172M parameters for LadderSym versus 192M for Polytune**; these counts were not independently remeasured. [1, 3]

**Symbolic prompt.** Clean-score MIDI is tokenized and prepended to the autoregressive decoder input. Prompt notes carry the Correct label. The decoder therefore receives explicit expected pitches alongside audio context, reducing ambiguity from overlapping harmonics. Prompted LadderSym still receives reference audio; it does not replace that stream with MIDI alone. [1]

**Unprompted variant.** Ladder without the symbolic prompt isolates the encoder improvement. The paper calls it Ladder, while the repository distributes it as an unprompted LadderSym checkpoint. This is a useful ablation for ALIGN.

### Code map

Paths refer to upstream `381179754cf6bcb435f9decf0d5e24eada6c68ec`, also the public main-branch head observed during this review. [3]

| File or directory | Responsibility |
|---|---|
| `models/ladder.py` | Patch streams; alternating alignment/fusion/encoder blocks; output projection |
| `models/ladder_backbone.py` | Alignment heads, fusion modules, stream embedders and supporting construction |
| `models/laddersym_t5.py` | Active Ladder encoder + T5 decoder integration and prompt-aware generation |
| `models/t5_prompt.py`, `models/t5.py` | Additional decoder code retained in the repository; distinguish from active wrapper imports |
| `tasks/laddersym_mt3_net.py` | Lightning task and prompt-aware training/validation |
| `tasks/laddersym_prompt_mt3_net.py` | Naming alias for the prompted task |
| `dataset/dataset_2_random*.py` | Audio/label loading and score-prompt preparation |
| `contrib/` | Audio features, MIDI/event codecs and target reconstruction |
| `config/` | Prompted/unprompted choices, datasets and hyperparameters |
| `train_laddersym.py`, `test_laddersym*.py`, `laddersym*_test_inference.py` | Training, evaluation and inference |
| `setup_laddersym_assets.py` | Dataset/checkpoint download and configuration helper |

The paper specifies 768-dimensional encoder features, projection to 512, and an eight-layer T5 decoder. Appendix A.4 reports 300 epochs, AdamW, learning-rate decay from `2e-4` to `1e-4`, error weight 10, token shuffling and bf16. Batch sizes are 48 spectrogram segments for MAESTRO and 96 for Coco on an A100 80 GB. However, released `config_maestro_prompted.yaml` specifies 1,600 epochs, `2e-5`, error weight 8 and different loader settings. Reproduction requires checking the actual checkpoint history rather than treating these as equivalent recipes. [1, 3]

Four official combinations are listed: prompted/unprompted x MAESTRO-E/CocoChorales-E, plus a separate robust-piano-v3 prompted model. Source root and checkpoint metadata advertise MIT. The paper acknowledges incorporated Polytune code under a non-commercial license, and files retain third-party notices; review component provenance before assuming the top-level license settles every downstream use. [3, 5]

## 4. Datasets used by Polytune and LadderSym

Both models use **MAESTRO-E and CocoChorales-E**, the same two error-augmented dataset families. LadderSym primarily contributes architecture and a small real beginner evaluation set, rather than an entirely different large training corpus. [1, 2]

| Dataset | Origin and construction | Role | Size evidence and limits |
|---|---|---|---|
| **MAESTRO-E** | MAESTRO piano MIDI modified to introduce mistakes and rendered into paired score/performance audio | Piano error-detection train/validation/test in both projects | Papers: 1,000+ tracks, 200+ hours, 200k+ errors. Downloaded MAESTRO-v3 metadata: 962 train / 137 validation / 177 test = 1,276 entries. These are metadata counts, not a check that every augmented file exists or loads. [1, 2, 7] |
| **CocoChorales-E** | CocoChorales material with injected errors and instrument rendering; loader uses individual stems | Train/validation/test across 13 non-piano instruments | Current card: 36,236 train / 4,504 validation / 4,284 test loader rows = 45,024 rows, from 11,256 selected track IDs. Polytune says 10k+ tracks; LadderSym says 40k+. Units/versions must be distinguished. [1, 2, 8] |
| **LadderSym beginner piano evaluation** | Genuine novice takes recorded through digital piano/VST, reference MIDI and human-verified errors | Held-out real-data evaluation without fine-tuning | Narrative and per-piece table describe 20 takes/pieces from three pianists; Table 8 says 10. Reported 75 wrong-note pairs, 51 extra notes and 35 missed notes total 161 error instances under that counting convention. Download of this specific corpus was not verified. [1, Appendix A.7] |
| **Benetos et al. historical benchmark** | Small real piano error dataset cited by both papers | Historical context for scarcity | 7 tracks, approximately 15 minutes, 40 errors. Not the large training corpus for these models. [1, 2] |

CocoChorales-E covers **flute, clarinet, oboe, bassoon, violin, viola, cello, trumpet, French horn, tuba, trombone, contrabass, and tenor saxophone**. Piano comes from MAESTRO-E, giving 14 instruments across evaluation. “Multi-instrument” means coverage across instruments here; it does not establish arbitrary simultaneous-ensemble error detection. Output tokens omit instrument identities, and the Coco loader excludes `mix.wav`/`mix.mid`. [1, 2, 3]

### Synthetic supervision and storage

The generator repository contains `create_maestro.py`, `create_cocochorales.py`, `midi_augmentation.py`, `mistake_augmentations.py`, `augment_config.yaml`, and `midi_ddsp_synthesize.py`. The flow is:

```text
source MIDI -> retain clean reference
            -> inject deletion / pitch change / time shift / insertion
            -> render clean and modified versions
            -> export Correct / Removed / Extra MIDI supervision
```

The papers describe an error-rate parameter sampled uniformly from 0.1 to 0.4 and sampled pitch/timing perturbations. These are controlled synthetic mistakes, rather than large-scale recordings of student errors. The papers describe MIDI-DDSP synthesis, while the released MAESTRO generator also enables FluidSynth and names a piano SoundFont by default. Record the actual renderer and sound bank rather than attributing every rendered instrument to MIDI-DDSP. [1, 2, 9]

The code-compatible filesystem is:

```text
mistake/<sample_or_track_id>/...                 performance WAV/MIDI
score/<sample_or_track_id>/...                   clean reference WAV/MIDI
label/extra_notes/<sample_or_track_id>/...        extra-note MIDI
label/removed_notes/<sample_or_track_id>/...      missing-note MIDI
label/correct_notes/<sample_or_track_id>/...      correct-note MIDI
maestro-v3.0.0.json  OR  split.json               split metadata
```

Loaders combine paired audio, class-specific target MIDI, and score MIDI for prompting. A missing note has no acoustic event: its time and pitch must come from the constructed supervision/reference convention. ALIGN conversion therefore needs exact lineage and a documented performance-time mapping after deletion or repetition. Equal-length audio padding does not provide alignment. [3, 7, 8, L2]

### Availability, counting units and provenance

Hugging Face offers the full code-compatible trees and a separate `viewer_preview` configuration. The preview is not the full corpus. Polytune's README also retains older Globus links. [4, 7, 8]

The downloaded Coco `split.json` contains **240,000 source IDs** (192,000 train / 24,000 valid / 24,000 test), while the dataset card describes a selected 11,256-ID subset and 45,024 loader rows. Counting split IDs alone therefore overstates the usable release. Save the actual loader's intersection of the requested split and complete paired files.

Both papers refer to a combined **4,401-track** test set. Current metadata/card counts do not independently reconstruct it: 177 MAESTRO metadata test entries + 4,284 Coco loader test rows = 4,461. This is an unresolved release/filtering/protocol question, not evidence that 4,461 audio files were verified or that the paper necessarily used an incorrect split.

Derivative dataset cards advertise CC BY 4.0, while original MAESTRO specifies **CC BY-NC-SA 4.0**. Preserve both provenance statements; a derivative card alone does not establish removal of upstream restrictions. [7, 8, 17]

## 5. What the published results support

These are **published F1 percentages**, not measurements on ALIGN. Values are **Correct / Missed / Extra**. Dataset-wide values come from LadderSym Table 2, cross-checked against Polytune Table 3; clarinet comes from LadderSym Table 10 and Polytune Table 4. Coco dataset-wide values are macro-averaged across instruments. [1, 2]

| Evaluation | LadderSym | Polytune | Explicit-alignment baseline |
|---|---:|---:|---:|
| MAESTRO-E piano | 94.4 / **54.7** / 86.4 | 90.1 / 26.8 / 72.0 | 43.5 / 6.6 / 39.9 |
| CocoChorales-E, 13 instruments | 97.7 / 61.7 / 61.4 | 95.4 / 51.3 / 46.8 | 36.7 / 7.7 / 23.5 |
| CocoChorales-E clarinet | 97.8 / 59.0 / 61.0 | 95.6 / 49.7 / 46.6 | 38.3 / 6.7 / 24.1 |

**Manuscript discrepancy:** LadderSym's abstract/introduction say 56.3% missed-note F1 on MAESTRO-E; the inspected PDF Tables 2 and 10 say 54.7%. This report uses the table value and records the conflict. Nearby prose also contains 26.3% where the Polytune table says 26.8%. Identify the version and table when quoting these comparisons.

The beginner piano evaluation reports mean missed-note F1 of 78.5% for LadderSym versus 63.9% for Polytune, and extra-note F1 of 81.6% versus 80.6%. These are small-sample piano results under clean direct recording conditions, not evidence for equivalent performance on phone-recorded clarinet. The model assisted initial annotation before two people corrected/verified labels; independently annotated tests would strengthen generalization evidence. [1, Appendix A.7]

The authors use category-specific `mir_eval` onset F1 with 50 ms onset tolerance and pitch agreement. ALIGN's score-event metric and partial-credit rule must be reported separately. High Correct-note scores should not obscure substantially lower error-specific scores.

## 6. Other useful available baselines and their datasets

### MT3 and Polytune's explicit-alignment baseline

**MT3: Multi-Task Multitrack Music Transcription** (ICLR 2022) uses a T5 encoder-decoder to map log-mel audio to timed MIDI-like events including instrument tokens. The paper uses approximately 60M-parameter T5-small. The original repository uses T5X/JAX and supplies inference checkpoints; training is less turnkey. [10, 11]

Its six training/evaluation dataset families are **MAESTRO v3, Slakh2100, Cerberus4, GuitarSet, MusicNet and URMP**. Cerberus4 derives from Slakh stems and guitar/bass/drums/piano combinations, so these are not six independent sources. They are transcription corpora, not annotated learner-error corpora. [10, Section 4.1]

Polytune's `baseline/MT3_baseline/` is a **separate PyTorch reimplementation**. Its README explicitly says original MT3 weights cannot be applied directly and it retrains from scratch. It updates the Benetos/Wang-style pipeline by replacing NMF transcription with MT3 and WTW with DTW. Thus “original pretrained MT3 + ALIGN alignment” and “Polytune's retrained MT3 baseline” are different experiments. [2, 4]

For a diagnostic baseline, retain the transcribed notes, alignment path and unmatched events separately. This distinguishes failure to hear a note from failure to map it to the score.

### TheGlueNote and Parangonar

**TheGlueNote: Learned Representations for Robust and Flexible Note Alignment** (ISMIR 2024) is a non-causal transformer over two symbolic sequences. Each note encodes relative onset, pitch, duration and velocity. Windows contain 512 notes from each sequence plus unmatched/default tokens. Learned representations produce a pairwise similarity matrix; matches are extracted directly, with a learned head, or with weighted DTW followed by pitch-separated onset matching. Long inputs use overlapping windows and aggregated similarities. [12]

**Training:** 1,032 valid MIDI files from **(n)ASAP**, using score/performance MIDI themselves rather than the dataset's existing alignments as targets. Synthetic copying and augmentation create exact correspondences, including tempo/onset changes, insertions, deletions, long repeats, skips and trills. **Evaluation:** public **Vienna 4x22** (4 pieces x 22 performances), plus challenging material inherited from prior work which the paper identifies as proprietary. Access to the package does not imply access to every original test corpus. [12, Section 4.1]

The repository has `src/` modules for configuration, datasets/augmentation, model, post-processing, evaluation and train/test scripts. `data/` contains small/mid checkpoints, (n)ASAP-based training material and Vienna test resources; a large checkpoint is separately linked. Parangonar exposes `TheGlueNoteMatcher`, non-neural `DualDTWNoteMatcher`, repeat identification, sub-part matching and other methods with Partitura I/O. [13]

This is useful for ALIGN because it exposes note correspondences. It still needs audio transcription and an error-label conversion layer. Training on piano MIDI and synthetic repeats does not establish detection of ALIGN's full repetition-event semantics.

### Basic Pitch

The ICASSP 2022 paper, *A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation*, describes a small fully convolutional model over a harmonically stacked constant-Q transform. It outputs onset, note-activity and fine-pitch posteriorgrams, followed by note-event post-processing. The paper model has 16,782 parameters; the maintained release explicitly says it has improved beyond the paper, so do not assume every release has identical architecture or preprocessing. [14, 15]

**Paper training data:** GuitarSet, MAESTRO, Slakh, iKala and MedleyDB. **Note-transcription evaluation:** Molina vocals, GuitarSet, MAESTRO, Slakh and PHENICX. **Additional multipitch evaluation:** Bach10 and Su. Some iKala/MedleyDB note labels are generated with `pyin-notes`; the Slakh evaluation selects non-percussive stems. [14, Table 1 and Section 4]

Basic Pitch provides a low-cost front end but neither consumes a clean score nor directly labels mistakes. Keeping it fixed while replacing ALIGN's aligner with DualDTW or TheGlueNote is a useful component study.

### Matchmaker

Matchmaker exposes online audio/MIDI-to-score alignment and current position in beats. Its implementation separates input streams, processing and alignment methods, with simulation and live-device modes, and integrates Parangonar trackers. It has no single universal training dataset: it packages multiple algorithms, including methods requiring no training. Its README was inspected, but a dedicated Matchmaker paper and unified benchmark corpus were not reviewed. It is an implementation option for streaming alignment, not a paper-backed replacement for the full error detector. [16]

## 7. What already exists inside this ALIGN workspace

The baseline integration already pins the same upstream versions reviewed here, with setup/training/evaluation scripts, configs, patches, converters, strict checkpoint loading and regression tests. Reuse it rather than building another independent wrapper. [L2, L3]

The documented frozen legacy corpus **`align_v1`** originally contained 12,000 synthetic tracks. The supervision audit excluded 86, leaving **11,914: 10,725 train / 594 validation / 595 test**. This is **ALIGN data**, not MAESTRO-E or CocoChorales-E. Later experiment-specific corpora and splits also exist; `align_v1` is not the only data ever used in this workspace. These counts come from project documentation and were not re-audited in this task. [L2]

The raw-score portion of the legacy corpus uses per-clip splitting, so the same score can appear across splits. It does not support unseen-piece generalization claims. Generator versions and label schemas also differ: `note_map.json` is not interchangeable with legacy `note_labels.json`, particularly for missing-note timing. The integration provides an audited supervision exporter for newer corpora. [L2, L3]

The opt-in solo corpus documented locally has 12,000 monophonic clips across 24 instruments, plus a separate 4,608-clip chromatic training supplement. It is a **transcription corpus without planted errors**, so it cannot directly replace the error-detection targets required by Polytune/LadderSym. Real DataCreate human labels, model-generated agent labels and synthetic exact labels must retain separate provenance. [L1, L4]

The local reproduction audit records fixes for EOS handling, prompted-generation prefixes, strict weight loading, class-name matching, scheduler accounting and other engineering problems. These support reusing the established integration, but passing smoke tests or retraining on ALIGN does not reproduce paper scores. [L3]

## 8. Recommended evaluation design

1. **Run the three direct baselines:** prompted LadderSym, unprompted LadderSym and Polytune. For clarinet pretrained evaluation, use the **CocoChorales-E** variants first. Keep official and robust-piano weights separate.
2. **Separate transfer from adaptation:** evaluate frozen official weights, then fine-tune eligible models on the same ALIGN split. Label from-scratch training as a third condition. Record checkpoint, code SHA, data manifest, prompt budget and configuration overrides.
3. **Group by held-out pieces:** split by source composition/melody before augmentation. Keep validation for model selection and thresholds. Respect existing restrictions on which real labels are verified gold.
4. **Standardize inputs:** produce clean reference MIDI/audio from MusicXML, retain note-ID mappings, and explicitly resolve written versus sounding pitch. If pre-warping is introduced, name it as a separate condition because it changes the model's inputs.
5. **Publish two metric families:** authors' Correct/Missed/Extra onset-pitch F1 with documented tolerance/aggregation, and ALIGN's canonical score-event metric with a documented conversion. Report unexpressed classes and the conversion's oracle ceiling rather than treating them as learned capabilities.
6. **Add a modular diagnostic:** fix the transcription front end and compare ALIGN alignment against DualDTW and TheGlueNote. This measures alignment separately from transcription. Original pretrained MT3 is a useful additional front end if acoustic capacity is also being studied.
7. **Test structural and acoustic shifts:** long restarts/repeats, tempo drift, same-pitch repeated attacks, phone noise, transposition and unseen timbres. Measure full-recording latency on target hardware; A100 encoder timings are not complete interactive latency.

The existing three-way integration is the most defensible starting point, supplemented by explicit alignment and careful dataset/metric conversion. LadderSym is the strongest direct candidate in the reviewed published comparisons; whether that advantage holds on ALIGN recordings remains empirical.

## Sources and audit trail

Five primary papers were downloaded and their full text accessed. Polytune's architecture page and LadderSym's architecture, result table and dataset table were rendered and visually checked. PDFs, source snapshots, dataset metadata and retrieval logs are stored in `sources/` beside this report. `source_manifest.json` records SHA-256 hashes. Search-engine summaries were used only for discovery.

### Papers, code and datasets

1. Chou et al. **LadderSym**, ICLR 2026. [Paper v2](https://arxiv.org/abs/2510.08580v2), [PDF](https://arxiv.org/pdf/2510.08580v2), [OpenReview](https://openreview.net/forum?id=cizuvfyQXs). Sections 3-4; Tables 2, 3, 7, 8, 10; Appendices A.2-A.8. Saved as `sources/laddersym.pdf`.
2. Chou et al. **Detecting Music Performance Errors with Transformers**, AAAI 2025. [Paper](https://arxiv.org/abs/2501.02030), [PDF](https://arxiv.org/pdf/2501.02030). Sections 3.2-3.4, 4.1, 4.4; Tables 2-4. Saved as `sources/polytune.pdf`.
3. [LadderSym inspected source](https://github.com/ben2002chou/LadderSym/tree/381179754cf6bcb435f9decf0d5e24eada6c68ec), [license](https://github.com/ben2002chou/LadderSym/blob/381179754cf6bcb435f9decf0d5e24eada6c68ec/LICENSE).
4. [Polytune inspected source](https://github.com/ben2002chou/Polytune/tree/d2055bb21759d457c8f21c1cf2e47c79af6248f5), [license](https://github.com/ben2002chou/Polytune/blob/d2055bb21759d457c8f21c1cf2e47c79af6248f5/LICENSE), [explicit baseline](https://github.com/ben2002chou/Polytune/tree/d2055bb21759d457c8f21c1cf2e47c79af6248f5/baseline/MT3_baseline).
5. [LadderSym checkpoints](https://huggingface.co/ben2002chou/laddersym-checkpoints).
6. [Polytune checkpoints](https://huggingface.co/ben2002chou/Polytune/tree/main).
7. [MAESTRO-E dataset/card](https://huggingface.co/datasets/ben2002chou/MAESTRO-E), [split metadata](https://huggingface.co/datasets/ben2002chou/MAESTRO-E/blob/main/maestro-v3.0.0.json).
8. [CocoChorales-E dataset/card](https://huggingface.co/datasets/ben2002chou/CocoChorales-E), [split metadata](https://huggingface.co/datasets/ben2002chou/CocoChorales-E/blob/main/split.json).
9. [Error-dataset generator](https://github.com/ben2002chou/CocoChorales-E_MAESTRO-E). Inspected README, file tree, `create_maestro.py` and `create_cocochorales.py`.
10. Gardner et al. **MT3: Multi-Task Multitrack Music Transcription**, ICLR 2022. [Paper](https://arxiv.org/abs/2111.03017), [PDF](https://arxiv.org/pdf/2111.03017). Sections 3-4 and Appendix B. Saved as `sources/mt3.pdf`.
11. [Original MT3 implementation](https://github.com/magenta/mt3).
12. Peter and Widmer. **TheGlueNote: Learned Representations for Robust and Flexible Note Alignment**, ISMIR 2024. [Paper](https://arxiv.org/abs/2408.04309), [PDF](https://arxiv.org/pdf/2408.04309). Sections 3-4. Saved as `sources/thegluenote.pdf`.
13. [TheGlueNote source/checkpoints](https://github.com/sildater/thegluenote), [Parangonar](https://github.com/sildater/parangonar), [algorithm references](https://github.com/sildater/parangonar/blob/main/refs.md), [(n)ASAP](https://github.com/CPJKU/asap-dataset).
14. Bittner et al. **A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation**, ICASSP 2022. [Paper](https://arxiv.org/abs/2203.09893), [PDF](https://arxiv.org/pdf/2203.09893). Section 3, Table 1 and Section 4. Saved as `sources/basic_pitch.pdf`.
15. [Basic Pitch source, model and release caveat](https://github.com/spotify/basic-pitch).
16. [Matchmaker implementation and documentation](https://github.com/pymatchmaker/matchmaker).
17. [Original MAESTRO dataset and license](https://magenta.tensorflow.org/datasets/maestro).

### Local project evidence

- **L1:** [ALIGN documentation](D:/stuff/Audio Evaluation/ALIGN/DOCUMENTATION.md), [root README](D:/stuff/Audio Evaluation/ALIGN/README.md), [model README](D:/stuff/Audio Evaluation/ALIGN/align-model/README.md).
- **L2:** [Baseline integration and data](D:/stuff/Audio Evaluation/ALIGN/baselines/README.md), [metric specification](D:/stuff/Audio Evaluation/ALIGN/methodology.md).
- **L3:** [Reproduction audit](D:/stuff/Audio Evaluation/ALIGN/baselines/docs/REPRODUCTION_AUDIT.md).
- **L4:** [Opt-in solo multi-instrument corpus/model](D:/stuff/Audio Evaluation/ALIGN/align-model/SOLO_MULTI_INSTRUMENT_V1.md).

Limitations: no model inference, dataset-wide file audit, training, checkpoint compatibility test or numerical reproduction was performed. The beginner piano download and exact published 4,401-item test manifest remain unverified. Existing project source edits and experiments were left unchanged.
