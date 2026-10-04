# ALIGN project documentation

This is the guide to the ALIGN / MusicEval workspace: what each part does, which model the GUI actually shows, which model the latest experiments use, and how to read every JSON the project writes.

[`methodology.md`](methodology.md) remains the specification for the label schema and the official score. This document is the operator's map. Where an older README still describes pitch-list matching or the contextual aligner as the live system, use this file and the code it cites.

The workspace is three Python packages plus the recordings they consume:

| Directory | Role |
|---|---|
| [`DataCreate/`](DataCreate/) | Turn a MusicXML score and a recording into a sample folder, and review it in the browser. |
| [`synth-pipeline/`](synth-pipeline/) | Generate synthetic clarinet performances whose errors and note lineage are known exactly. |
| [`align-model/`](align-model/) | Transcribe the performance, align those notes to the clean score, and emit error labels. |
| `RawData/` | Uploaded scores and practice recordings. Not a package. |

## Two systems are live, and they are not the same

**The annotation GUI** reads files inside each sample folder. Alignment and the transcription staff come from `note_alignment_v2.json`. That file is produced by the joint path CRF at `align-model/runs/joint-audit-v2/end-to-end-v2/weak-note-continuation-optimized/joint_decoder.pt`, configured in `DataCreate/config/default.yaml` as `paths.note_alignment_checkpoint`. Agent labels in `labels_agent.json` come from an earlier inference pass: the mel transcriber `candidate-epoch-018.pt`, that same joint decoder, and error-heads v5. Re-align in the GUI reruns this joint bridge. It does not run stack v6.

**Stack v6** is the current experiment checkpoint, frozen in [`align-model/runs/precision-v4/CANDIDATE_STACK_V6.json`](align-model/runs/precision-v4/CANDIDATE_STACK_V6.json). It is a dual-resolution mel CTC transcriber (`runs/precision-v4/v5-dclike/best.pt`), the robust DP aligner v3, a note-presence verifier, and logistic gates (extra threshold 0.85, missed threshold 0.96). Its alignments for the real takes live under `align-model/runs/precision-v4/dc-v6/alignments/<id>.json`. They have not been copied into `DataCreate/samples/`, so the GUI does not show them.

An earlier production bundle, `align-model/runs/contextual-aligner-outputRaw_sf-1k/weights/`, is the fallback if the joint checkpoint file is missing. It is Basic Pitch plus a contextual GRU aligner trained on dataset 6.0. The root README used to describe that bundle as current. The GUI has since moved to the joint decoder, and the sealed tests have since moved to stack v6.

## Pitch

MusicXML, `labels.json`, transcriptions, and model targets use **written** MIDI. Bb clarinet audio sounds two semitones lower: sounding pitch = written pitch − 2. Loaders must use `effective_audio_transpose` (or `sounding_transpose`) from `metadata.json`. They must not guess the pitch space from a folder name.

Take 001 is the exception used in annotation: `001.musicxml` is a concert-pitch demo scale, so concert pitch equals written pitch. Takes 002 and later are written for Bb clarinet.

## What a sample folder contains

A real take lives at `DataCreate/samples/<id>/`. A synthetic clip has the same shape, plus the supervision files marked below.

| File | What it is | Gold? |
|---|---|---|
| `verified_score.musicxml` | The clean score. Inference and evaluation read only this score. | Yes, as notation |
| `full_score.musicxml` | The uncut score, when the take was sliced to a measure range. | Source of the slice |
| `performance_audio.wav` | The recording or the degraded synthetic performance. All label times are on this file. | The audio being judged |
| `reference_audio.wav` | A metronomic render of the clean score. Alignment aid, not a second performance. | No |
| `performance_score.musicxml` | Synthetic only. The score that was actually rendered, including planted errors. | Supervision only. Never an inference input. |
| `performance_audio_clean.wav` | Synthetic only. Muse Sounds render before phone-style degradation. | No |
| `note_map.json` | Synthetic only. Exact performed-note to clean-note lineage. | Yes, for synthetic training |
| `labels.json` | Human gold on real takes (`source: manual`). Synthetic gold on generated clips (`source: synthetic`). | Yes, with the rules in the next section |
| `labels_agent.json` | Automatic labels from the mel + joint + error-heads v5 pass. | No. A model output. |
| `candidates.json` | Older automatic candidates (`source: auto`). | No, until a person confirms them |
| `note_alignment_v2.json` | Alignment the GUI draws. | No |
| `transcription_notes.json` | Note list from an older transcription pass. | No |
| `transcription_mel_v1.json`, `note_alignment_mel_v1.json` | Artifacts from the 2026-09-21 agent-label pass. | No |
| `alignment.npz` | Compact warping path kept so older code still loads. | No |
| `performance_mel.npy`, `reference_mel.npy` | Log-mel spectrograms. | Features |
| `metadata.json` | Sample rate, score slice, trim, and which agent files were written. | Provenance |

Times in every label file are seconds on `performance_audio.wav` after any trim. A trim is recorded in `metadata.json` under `performance_trim` and is already applied to the WAV, so do not add `trim_start` back onto label times.

## How to read `labels.json`

Validated by `LabelsDocument` in `DataCreate/src/datacreate/models.py` and `datacreate-validate`. Schemas **1.1** and **1.2** both validate. The default config still writes `schema_version: "1.1"` for human saves. Schema 1.2 is the same document plus a score melody on each label. Many human files are marked 1.1 and still carry `score_part` and `pitches`; those fields are readable either way. A 1.1 label that has no score projection cannot be scored by the official metric.

### Document

```json
{
  "schema_version": "1.1",
  "audio_reference": "performance_audio.wav",
  "annotator_id": "ai_f0_align",
  "self_reported": [],
  "labels": []
}
```

| Field | Meaning |
|---|---|
| `schema_version` | `"1.1"` or `"1.2"`. |
| `audio_reference` | Always the performance WAV. |
| `annotator_id` | Who saved the file. Human GUI saves from the reviewed Weber/Mozart pass use `ai_f0_align`. An empty document with `annotator_id: annotator01` is a human save that found nothing to mark (take 004). |
| `self_reported` | The player's own suspected regions. Never merged into `labels`. Each entry is `{start_time, end_time, comment}`. |
| `labels` | The error list. Empty is a valid "no errors" document. |

`labels_agent.json` uses this same label list, and adds an `agent_labeling` object above it describing the checkpoints that produced the file. Read `agent_labeling.mel_checkpoint`, `joint_checkpoint`, and `error_heads_checkpoint` before treating those labels as any particular model.

### One label

Required fields: `id`, `source`, `start_time`, `end_time`, `type`. `end_time` must be greater than `start_time`.

| Field | Meaning |
|---|---|
| `id` | Stable id. Human ids look like `lbl_<ms>_<random>`. Synthetic ids look like `syn_###`. Automatic ids look like `cand_###`. |
| `source` | One of `auto`, `auto_confirmed`, `auto_edited`, `auto_rejected`, `agent`, `manual`, `synthetic`. Train and evaluate real takes on `manual` only. `auto` has not been reviewed. |
| `start_time`, `end_time` | Seconds on `performance_audio.wav`. This is what the waveform shows. It is not the official location. |
| `type` | An error type from the taxonomy below. |
| `severity` | Optional 1–5. The reviewed human pass uses 3 unless the error is extreme. |
| `deviation_cents`, `deviation_ms` | Optional measurements. Often null. |
| `measure_number` | Score measure, when the annotator or the projector filled it in. |
| `note_id` | A single clean-score id such as `note_0079`, when the label was attached to one note. Often null once `note_ids` is present. |
| `comment` | Free text. A comment containing `repeated pass` or `(pass N)` marks a replay copy. Those copies are dropped before scoring. A comment containing `first pass` is kept. |
| `repeats_label_range` | For `repetition` only: `{start_time, end_time}` of the earlier span being replayed. This range is in seconds, not score indices. |
| `extra_copies` | For `repetition` only. `1` means the passage was played twice (one extra copy). `2` means three plays. Official repetition identity includes this count. A repetition label without it has no canonical location. |

### Score location: `score_part`, `pitches`, `note_ids`

These fields name **where on the clean score** the error sits. Indices count sounding notes on `verified_score.musicxml` after tied notes are collapsed, in written order, starting at 0. `note_0007` is index 7. Rests are not sounding notes.

`score_part.start_note_index` and `score_part.end_note_index` are **inclusive**. A label with start 77 and end 81 covers notes 77, 78, 79, 80, and 81. That matches `note_ids` of length 5 and `pitches` of length 5.

The stored range includes padding. `pad_notes` is how many clean notes were added on each side of the actual fault, usually 1 or 2, clamped at the ends of the score. The fault itself is the inner span:

```text
core_start = start_note_index + pad_notes
core_end   = end_note_index   - pad_notes
```

both inclusive. When the file also stores `core_start_note_index` and `core_end_note_index`, those should equal that arithmetic. A single-note fault has `core_start_note_index == core_end_note_index`.

`pitches` is the written MIDI of `note_ids[0] ... note_ids[-1]`, in that order, including the padding notes. It checks that the range still refers to the same score. It does not identify the error. Two labels with the same pitches at different indices are different errors.

`core_note_ids`, when present, should be the unpadded note ids. On some human saves it disagrees with `core_start_note_index` (take 020's first label stores core index 79 but `core_note_ids: ["note_0080"]`). Trust the numeric `score_part` indices and the `note_ids` list that has the same length as `pitches`. Treat a disagreeing `core_note_ids` entry as a projection leftover.

Worked example, take 020, first label:

```text
type: wrong_note
time: 12.558–12.665 s on performance_audio.wav
score_part: notes 77–81 inclusive, pad 2, measures 119–120
core: note 79
pitches: [74, 67, 72, 67, 71]
```

The official location of this label is the inclusive event tuple `(77, 78, 79, 80, 81)`, not "the note at 12.6 seconds" and not "written MIDI 72".

### What each type means

Closed taxonomy in `DataCreate/config/default.yaml`. `wrong_pitch` is accepted only so old files still validate; new labels use `wrong_note`.

| Type | On a real take | On a synthetic clip | Core of `score_part` |
|---|---|---|---|
| `wrong_note` | A melody note at the wrong pitch, about 2 semitones or more. Not an inner chord tone the pitch tracker dropped. | Written pitch shifted ±1 or ±2 semitones, or replaced by a squeak. | The written note that changed. |
| `missed_note` | An obvious skipped melody note. Not a short pedal or arpeggio tone inside a fast figure. | That written note became a rest of the same duration. Family 11 also has a mode that holds the previous note instead. | The written note that did not sound. |
| `extra_note` | A clear extra attack, including a re-attack of the pitch just played. | A note inserted by splitting a written note. | The clean notes immediately before and after the insert. The extra pitch is not on the clean score. An insert after the last note uses that last note only. |
| `repetition` | A practice restart of a figure the score does not ask for. Written repeats are not this type. Requires `repeats_label_range`. | A replay of the measures that contain the planted errors, or a standalone replay, after a 0.2–1.0 s silence. | The written notes of the passage that was replayed. `extra_copies` is required for scoring. |
| `rhythm_error` | A stall or a figure that falls apart, such as a cadence note held instead of moving on. | Late/early start or end, a short tempo change, or uneven durations inside a bar. | The written notes whose timing changed. |
| `intonation_error` | Right pitch class, clearly out of tune. Do not use it for a noisy pitch track. | MIDI pitch bend of 40–80 cents, sometimes across up to 4 notes. | The detuned written notes. |
| `sliding` | Rolled or smeared through a note. Human listening only. | Not planted. | The note that was smeared, when a score span was stored. |
| `bad_start` | Failed attack: air, tongue, no clean onset. Human listening only. | Not planted. | — |
| `bad_timbre` | Stuffy, airy, or unfocused tone. Human listening only. | Not planted. | — |
| `squeak` | Unintended harmonic. Human listening only. Synthetic squeaks are stored as `wrong_note` or `extra_note`, not as this type. | — | — |
| `stylistic_choice` | A deliberate departure the annotator does not want scored as a mistake. | Not planted. | — |
| `misc` | Anything else. | Not planted. | — |

Stack v6 emits only `extra_note`, `missed_note`, `wrong_note`, and `repetition`. A human `sliding` or `bad_start` has no counterpart in that output.

### Repeated passes

Synthetic clips write the content error once for the first pass and again, in the comment, for each replay. Evaluation drops any label whose comment contains `repeated pass` or `(pass N)`. The first-pass label and the single `repetition` label stay. Do not delete those copies from the file; the scorer ignores them.

### Which labels are gold

| File and source | Use |
|---|---|
| `labels.json` and `source: manual` | Human gold for that take. |
| `labels.json` and `source: synthetic` | Gold for that synthetic clip, after dropping repeated-pass copies. |
| `labels.json` and `source: auto*` | Unreviewed or reviewed candidates. Confirmed and edited rows are human decisions; raw `auto` is not training gold. |
| `labels_agent.json` | Model output from the 2026-09-21 mel pass. Compare it to `labels.json`. Do not train on it as if it were human. |
| `candidates.json` | Same label shape, `source: auto`. Intermediate. |

On takes 001–094, the non-empty human files are 001, 002, 003, 005, 006, 008, 010, 011, 012, 013, 014, 016, 017, 019, 020, 029, 030, 033, 034, 035, 036, 039, 041, and 043. Take 004 is an empty human save. Annotation practice for those takes is written in [`DataCreate/docs/annotation_rules.md`](DataCreate/docs/annotation_rules.md).

## How to read the other JSON files

### `note_alignment_v2.json` (what the GUI draws)

`format_version` 2, `engine: "align-joint"`. `summary.checkpoint` names the weights. The current GUI files point at `joint_decoder.pt`.

Each `events[]` row is one aligned performance note:

| Field | Meaning |
|---|---|
| `sounding_index`, `score_index` | Position in the clean score's sounding-note list. |
| `note_id` | `note_NNNN` for that score index. |
| `midi` | Written MIDI. |
| `pitch` | Pitch name for that MIDI. |
| `measure`, `duration_ql` | From the score. |
| `ref_start`, `ref_end` | Seconds on the reference render. |
| `perf_start`, `perf_end` | Seconds on `performance_audio.wav`. This is the span the staff and waveform use. |
| `alignment_kind` | `match`, `substitute`, `extra`, `delete`, or `copy`, depending on the decoder. |
| `is_repetition` | True when this note is part of a detected replay. |

`summary` counts transcribed notes, mapped notes, and the path score. `repetitions` lists detected replays. This file is an inference result. Re-align overwrites it.

### Stack v6 alignment (`runs/precision-v4/dc-v6/alignments/<id>.json`)

`schema_version: "align-datacreate-stack-v4-alignment"`. This is the file to read when comparing the latest model to gold. The GUI does not load it.

| Field | Meaning |
|---|---|
| `aligner` | `robust_dp_aligner_v3`. |
| `aligner_config` | `artifact_ioi: 0`, `timing_weight: 0.2`, `copy_after_replay: true` for v6. |
| `gate_info` | How many extra and missed calls the logistic gate withheld. |
| `score_event_count` | Sounding notes in the clean score. |
| `missed_score_event_indices` | Clean-score indices the aligner marked deleted. Each one becomes one `missed_note` label. |
| `repeat_hypothesis.source_span` | Half-open `[start, end)` score range of the passage that was replayed. |
| `repeat_hypothesis.extra_copies` | Same meaning as on a label: 1 = one extra play. |
| `events[]` | One transcribed note. |

Each event:

| Field | Meaning |
|---|---|
| `note_index` | Index in the transcription, not in the score. |
| `pitch` | Written MIDI. |
| `start`, `end` | Seconds on the performance. |
| `relationship` | `match`, `substitute` (wrong pitch on a score note), `extra` (no score note), or `copy` (replay of a score span). |
| `score_span` | Half-open `[start, end)` into the clean score. Null for an extra. A `substitute` uses a one-note span. |
| `copy_pass` | 0 on the first pass. 1 on the first replay, and so on. |
| `confidence` | Decoder confidence in `[0, 1]`. |

To turn this file into labels, use `labels_from_stack_alignment` in `align-model/scripts/eval_datacreate_vs_human.py`. It emits one `extra_note` per gap (several extras between the same two score notes collapse to one label), one `missed_note` per missed index, one `wrong_note` per substitute, and one `repetition` from `repeat_hypothesis`. The comparison against human gold for takes 001–094 is `align-model/runs/precision-v4/dc-v6-vs-gold-001-094.json`.

`score_span` here is half-open. `score_part` inside `labels.json` is inclusive. A stack span `[206, 207)` is score note 206, which a label stores as `start_note_index: 206, end_note_index: 206` before padding. The official metric then pads by 2, so the identity becomes notes 204–208. That padding is why a prediction and a gold label can describe the same note and still fail official matching when their stored ranges differ.

### `note_map.json` (synthetic lineage)

`kind: "synth_note_lineage"`. This is the training target for synthetic clips. Human takes do not have it.

| Field | Meaning |
|---|---|
| `clean_notes[]` | One row per clean sounding note: `clean_index`, pitch, onset, duration, measure, and `deleted: true` when the performance never plays it. |
| `deleted_clean_notes` | The `clean_index` values that were planted as misses. |
| `performed_notes[]` | One row per note in `performance_score.musicxml`. |
| `performed_notes[].clean_index` | Clean note this performance note comes from, or null for an insertion. |
| `performed_notes[].relationship` | `match` (same written pitch), `substitute` (wrong pitch), `extra` (inserted), or `copy` (a later pass of a note already heard). |
| `performed_notes[].origin_relationship` | The first-pass relationship. A copied extra keeps `clean_index: null` and `origin_relationship: extra`. |
| `performed_notes[].copy_pass` | 0 on the first time that identity is heard. |
| `rendered_notes[]` | What the MIDI renderer actually emitted. One rendered event can cover several tied written notes via `clean_indices`. |
| `rendered_notes[].pitch_midi_written` | Written pitch. `pitch_midi_sounding` is written − 2 for Bb clarinet. |
| `rendered_notes[].rendered_index` | The audio-event id. Unpaired extras in the official metric use this id, offset by 1_000_000, so they cannot collide with score indices. |

Do not treat MuseScore's `note_map` timestamps as the times heard in a Muse Sounds WAV. The written pitch order is reliable. The clock on those events is not, which is why alignment is redone from audio.

### `metadata.json`

Human takes store `sample_rate` (22050), `mel_params`, the score slice (`score_segment.start_measure`, `end_measure`, `start_beat`), and `performance_trim`. `pipeline_run_at` is when DataCreate last built the folder, not when a model ran. `agent_transcriber` and `agent_note_alignment` name the sidecar files from the agent-label pass.

Synthetic takes add `dataset_version` (for example `9.2`), `audio_render` (`soundfont_v1`, `musesounds_v1`, `oscillator_v1`), `audio_degrade` when degradation ran, `midi_pitch_space`, `sounding_transpose`, and `effective_audio_transpose`.

### `candidates.json`

Same label objects as `labels.json`, wrapped as `{schema_version, labels}`, with `source: auto`. The joint-decoder GUI path does not refresh this file on Re-align.

### `alignment.npz`

`warping_path` (or `wp`), `frame_residuals`, dummy or real `ref_features` and `perf_features`, `hop_length`, `sample_rate`, and `engine`. The joint bridge writes a path derived from note times so old DTW readers still open. It is not the note alignment. Read `note_alignment_v2.json` instead.

## Official score

Implemented in `DataCreate/src/datacreate/melody.py` as `match_note_wise_labels_detail`. Schema name: `align-note-wise-score-event-metric-v1`.

Location is the canonical score-event identity from `canonical_note_location`:

1. `score_event_indices`, if the label has an explicit list.
2. Otherwise the inclusive range `score_part.start_note_index` … `end_note_index`. This range **includes padding**.
3. Otherwise the integers parsed from `note_ids`.
4. A `repetition` identity is that range plus `extra_copies`.
5. An extra that has no clean-score neighbors can match on `rendered_index` / `extra_identity` instead.

Matching is exclusive and one-to-one. At the same identity, the same type scores 1.0 and a different type scores 0.5. A different identity scores 0. Precision is total credit divided by the number of predictions. Recall is total credit divided by the number of gold labels. Headline F1 is the harmonic mean of those two. Empty against empty scores 1.

If any gold label has no identity, the whole comparison is `status: unavailable`. Schema 1.1 times without a score projection are in that category. Do not substitute a timestamp score.

Timestamp intersection-over-union and onset tolerances at 20/50/100 ms are diagnostics. They must not choose a checkpoint or appear as the headline. Older reports that led with those numbers stay historical; they are not recomputed into the note-wise scale.

A second, looser comparison used for DataCreate listening checks treats two labels as a hit when they have the same type and their **core** ranges overlap within one note. That number is not the official score. Report it beside the official one and name it lenient.

## Synthetic datasets

`synth-pipeline/datasets.yaml` is the registry. Refer to a dataset as `9.2`, not by folder name. The major number is a new generation (new scores, new planted errors). The minor number is the same bundles re-rendered or degraded. Folders keep historical names because frozen split files point at them.

The clean `verified_score.musicxml` stays correct. Errors are written only into the performance. Content errors do not overlap in time. Repetition is applied after those errors, not as one of the content draws.

| Version | Folder | Status | What changed, and why |
|---|---|---|---|
| 1.0 | `synth-pipeline/1000dataexport` | on disk | One planted error per procedural clip. Schema 1.1. Used by the earliest models. |
| 2.0 | `E:/output` | missing | 1–8 errors per clip and exact `note_map` lineage, because one-error clips could not train a multi-error detector. |
| 3.0 / 3.1 | `output_2k_rawdata` / `E:/output_2k_rawdata` | 3.0 on disk, 3.1 pruned | Real-score snippets, because procedural melodies missed the repertoire. 3.1 stripped ornaments after the MIDI player held ornament notes under the melody. |
| 6.0 | `E:/outputRaw_sf_10k` | pruned | Clarinet SoundFont, score 001 held out. Trained the contextual aligner. Audio was removed; the frozen split can no longer be rerun. |
| 7.1 | `E:/outputRaw_orn_10k` | pruned | Muse Sounds replaced the SoundFont because the SoundFont timbre was far from a clarinet. Ornaments kept. |
| 8.1 | `E:/outputRaw_fast_1k` | pruned | Sixteenth-note etudes, because fast passages were missing. |
| 9.2 | `E:/outputRaw_realistic_10k` | on disk | Same idea as 7.x, but procedural density near 4.8 notes/s (7.x was near 1.8, DataCreate is near 5) and phone-style degradation `realistic_v1`. Clean Muse Sounds was perfectly separable from the phone recordings. |
| 10.2 | `E:/outputRaw_fast16_focus_5k` | on disk | Sixteenths aimed at the written register that the transcriber misses, with `realistic_v2` (harder dereverb, no extra room). 9.2's reverb tail was longer than the phone recordings. |
| 11.2 | `E:/outputRaw_dclike_11` | on disk | Ornaments written out as ordinary notes, because 9.x/10.x labels followed a MIDI ornament expansion Muse Sounds does not play. Missed notes can be a held previous note instead of a rest. `note_map` no longer folds repeated same-pitch notes into one event and no longer lets a repeat gap shift the timed match. Degradation is `realistic_v3`. |

Versions 3.1, 5.x, 6.0, 7.x, and 8.x were pruned on 2026-10-02. Each pruned folder keeps a few bundles, `_config/`, and `PRUNED.json`. About 1 TB was freed. Do not expect to retrain a model that reads those WAVs.

Family 11 is still separable from DataCreate by a mel-statistics classifier (AUC 0.998). Quiet passages have 7–9 dB less mid-band noise than the phone recordings. That gap is the next degradation change, and it has not been made.

Generation details, rhythm kinds, and render markers are in [`synth-pipeline/README.md`](synth-pipeline/README.md).

## Models, in the order they replaced each other

Each later system was built because the previous one failed a measurement. A later system is the one to use only on the task that measurement covered.

**Contextual aligner (dataset 6.0).** Basic Pitch transcription, a small repetition scorer, and a GRU aligner. It raised note-mapping F1 on 200 held-out SoundFont clips from 0.450 (deterministic edit alignment) to 0.509 after calibration. It remains the fallback weights. It is not what the GUI runs, and it was not retrained for Muse Sounds or phone audio.

**Joint path CRF v2.** The GUI's aligner. A sparse lattice over match, substitute, extra, noise, delete, and repeat, conditioned on the score, with a weak-note continuation fix. On its own validation it reached note F1 0.864 and joint F1 0.705. It replaced the contextual aligner in DataCreate because the contextual model had been trained on SoundFont snippets and the joint decoder was the checkpoint wired to `paths.note_alignment_checkpoint`. Re-align still runs this checkpoint.

**Mel transcriber plus error-heads v5 (the agent labels).** Basic Pitch split repeated notes and missed boundaries. On 358 synthetic validation clips, feeding the frozen joint aligner a repository-trained mel transcriber instead of Basic Pitch moved downstream alignment F1 from 0.632 to 0.708. That comparison is synthetic only. The human labels are too sparse for the same claim, so this pass was an inference switch for `labels_agent.json`, not a statement that the mel checkpoint won on real takes. The run is `align-model/runs/datacreate-agent-labels-mel-v1-20260921-192439/`. The previous agent labels were archived under `datacreate-agent-labels-pre-mel-20260921-192105/`.

**Stack v3, then v6 (current sealed tests).** Stack v3 is the dual-resolution mel CTC transcriber plus robust DP aligner v2. Its combined test F1 was already high (0.967 on 9.2, 0.990 on 10.2), and missed-note F1 was the weak type. Error audits on the validation split found that many "missed ornaments" were notes the MuseScore MIDI expansion wrote and Muse Sounds never played, that same-pitch merges were mostly alignment artifacts, and that false extras were short transition blips with confidence near 1, so a confidence threshold could not drop them. Stack v4 and v5 added a timing pass and logistic gates. On 9.2 test, extra precision rose to 0.980 then 0.988, but missed precision stayed at 0.932 then 0.902, because the gates were fit on scores that did not include the test scores and the features came from the transcriber. Stack v6 fine-tuned the transcriber on 9.2, 10.2, and 11.x with realism augmentation, added a note-presence CNN (validation AUC 0.992) so a missed-note decision can look at the audio, and refit the gates leaving one score group out. Thresholds were chosen on validation only. The sealed tests, each read once, are:

| Test | Clips | Extra precision | Extra recall | Missed precision | Missed recall | Combined F1 |
|---|---:|---:|---:|---:|---:|---:|
| 9.2 | 1836 | 0.983 | 0.824 | 0.984 | 0.435 | 0.967 |
| 10.2 | 597 | 0.994 | 0.923 | 1.000 | 0.501 | 0.989 |
| 11.2 | 260 | 1.000 | 0.412 | 1.000 | 0.333 | 0.985 |

Combined F1 did not move past v3. The precision comes from withholding calls, which is why recall of extra and missed notes is lower. The full write-up is [`align-model/runs/precision-v4/RESULT_SUMMARY.md`](align-model/runs/precision-v4/RESULT_SUMMARY.md).

Against the human labels, v6 is not at that precision. On the 33-take set used during development, v6 predicted 33 extra notes and 3 matched under the lenient rule; 9 missed notes and 1 matched. On takes 001–094 only, official extra precision is 0.065 (2 of 31) and missed precision is 0.125 (1 of 8). Those human-label figures were not the promotion test, and the extra threshold was not tuned on them.

## Annotation GUI

```powershell
conda activate MusicEval
cd DataCreate
datacreate serve
```

Open http://127.0.0.1:8765/. The page is `DataCreate/src/datacreate/web/templates/annotate.html`. Port 8765 must be free. A leftover `python -m http.server 8765` produces Windows error 10048 (`WSAEADDRINUSE`); stop that process and start `datacreate serve` again.

The **Label set** control switches between `labels.json` (human) and `labels_agent.json` (model). Saving writes only the selected file. Alignment mode loads `GET /api/samples/<id>/note-alignment`, which reads `note_alignment_v2.json` and draws the transcription staff above the score. **Re-align** reruns the joint decoder. **Re-label** rebuilds `labels_agent.json` from the alignment currently in the sample folder. Neither button runs stack v6.

`/compare` scores human labels against agent labels with the note-wise metric. Timestamp overlap on that page is a diagnostic.

## Commands that match the current tree

Environment setup is in the root [`README.md`](README.md). The align-model scripts that need torch and numba use `align-model/.venv-amt-bench`. The MusicEval conda env is what `datacreate` and `synth-pipeline` expect. Set `PYTHONPATH` to `align-model/src`, `align-model/scripts`, `DataCreate/src`, and `synth-pipeline/src` when running a script directly.

```powershell
# Annotate
datacreate serve

# Validate a label file against the taxonomy and schema
datacreate-validate DataCreate/samples/020/labels.json

# List dataset versions
synth-pipeline datasets list

# One-shot score of a frozen stack on a sealed split.
# Refuses to score the same candidate and dataset a second time.
python align-model/scripts/eval_stack_v4_test.py --help
```

Do not rerun the sealed 9.2, 10.2, or 11.2 tests for stack v6. The freeze file records that those tests were already read.

Training flags and layer hyperparameters for the older four-stage pipeline are in [`align-model/TRAINING.md`](align-model/TRAINING.md) and [`align-model/HYPERPARAMETERS.md`](align-model/HYPERPARAMETERS.md). Those documents describe Model A and Model B. They do not describe stack v6.

## Rules that keep the numbers meaningful

- Official location is score-event identity. Pitch lists and timestamps never substitute for it.
- Match one prediction to one gold label. One transcribed note cannot satisfy two gold labels.
- Inference reads `performance_audio.wav` and `verified_score.musicxml` only.
- `performance_score.musicxml` and `note_map.json` are supervision.
- Train and test membership come from a frozen `align-model/runs/*/split.json`, not from a directory listing.
- A dataset minor version can overwrite audio in place. Read `datasets.yaml` before assuming a folder still contains the render a paper table used.
- Human gold for real takes is `labels.json` with `source: manual`. Stack v6 output is a different file, in the run directory, until someone copies it into the sample.
