# MusicEval / ALIGN — Methodology

This document describes the ALIGN / MusicEval methodology: how score + performance audio become labeled training bundles, how **synthetic** gold is planted, and how model output is scored. The current focus is **clarinet** practice against notated MusicXML (or PDF→OMR). Real takes use human review as ground truth. Synth volume comes from `synth-pipeline` (schema **1.2** score-part melodies). Official evaluation is exclusive one-to-one canonical score-event identity F1. Pitch coincidence and timestamps do not establish location.

---

## 1. Goal and design principles

**Goal.** Produce per-sample training bundles containing:

1. Time-stamped error labels on the *performer’s* audio timeline (annotator, crop trainers)
2. A **score-part melody** on the clean MusicXML for each first-pass fault (`score_part`, `pitches`)
3. Log-mel spectrograms of performance and score-derived reference audio
4. Verified MusicXML used as the musical ground truth for synthesis and note mapping
5. Alignment artifacts used for auto-candidates and UI visualization

**Principles.**

| Principle | Implication |
|-----------|-------------|
| Alignment is an aid, not labels | Stage 5 writes `candidates.json` with `source: "auto"`. Humans confirm, edit, reclassify, or reject in the annotator UI. |
| Closed taxonomy | Error types live in `config/default.yaml`; schemas **1.1** (times only) and **1.2** (times plus a score-part melody) both validate. |
| Reference vs performance | Alignment compares real performance to a **synthesized** rendering of the verified score—not another human recording. |
| Pitch features for DTW | Chroma/CQT for alignment (timbre-robust); log-mel for the training features bundle. |
| Partial takes supported | Measure-range score segmentation + waveform trim + re-align for incomplete recordings. |
| Train/eval on score entities, not clocks | Official synth gold is a **contiguous run of clean-score notes** (`score_part` + `pitches`), not the wall-clock interval of the fault. Times stay on the file for the annotator and for crop trainers. |

---

## 2. End-to-end pipeline overview

```
Score (.musicxml / .mxl / .pdf)
        │
        ├─ PDF → Audiveris OMR → draft MusicXML → human verify
        └─ MusicXML → validate (music21) → verified_score.musicxml
                          │
          ┌───────────────┴───────────────┐
          │                               │
  Performance audio              MuseScore → MIDI → SoundFont
  (ingest / resample)            → reference_audio.wav
          │                               │
          └───────────────┬───────────────┘
                          │
        Basic Pitch note transcription
                          │
       Repetition-aware note-to-score mapping
                          │
              Auto candidates (pitch / rhythm / miss / extra)
                          │
              Annotation UI (confirm / edit / manual)
                          │
              Log-mel features + labels.json + metadata
                          ▼
                     Sample bundle
```

**Stages (implementation map)**

| Stage | Role |
|------:|------|
| 1 | Score ingest / validation |
| 2 | OMR (Audiveris) + manual correction gate |
| 3 | Reference audio synthesis |
| 4 | Performance audio ingest |
| 5 | Basic Pitch transcription + repetition-aware note alignment + candidate detection |
| 6 | Web annotation UI (`datacreate serve`) |
| 7 | Log-mel feature extraction |
| 8 | Bundle metadata / labels template |
| 9 | Synthetic corruption samples (optional) |

Batch mode runs stages **1–5 and 7** without the UI. Annotation (stage 6) produces final `labels.json`.

---

## 3. Inputs, audio standards, and sample layout

### 3.1 Inputs

- **Score:** MusicXML / MXL / XML, or PDF (OMR path).
- **Performance:** WAV/MP3/M4A/FLAC/etc.; resampled to pipeline sample rate.
- **Shared-score batch:** one MusicXML + many numbered takes (`batch-range`).

### 3.2 Audio standards (`config/default.yaml`)

- Sample rate: **22050 Hz**, mono
- Mel: `n_fft=2048`, `hop_length=512`, `n_mels=128`, `fmin=30`

### 3.3 Per-sample directory

```
samples/<sample_id>/
├── verified_score.musicxml
├── performance_audio.wav
├── reference_audio.wav
├── performance_mel.npy / reference_mel.npy
├── *_mel_preview.png
├── alignment.npz
├── candidates.json
├── labels.json
└── metadata.json
```

All label times are relative to **`performance_audio.wav`**.

---

## 4. Score path (Stages 1–2)

### 4.1 MusicXML ingest

Parse with **music21**. Invalid/corrupt files fail loudly. Canonical file: `verified_score.musicxml`.

### 4.2 PDF / OMR

**Audiveris** batch export → draft MusicXML. Human corrects (typically MuseScore GUI) before Stage 3. Verified MusicXML is the only score used for reference synthesis and note-ID / measure references.

---

## 5. Reference synthesis (Stage 3)

Ideal reference = metronomic, in-tune rendering of the verified score.

**Windows practical path:** MuseScore CLI WAV export is unreliable from Python; the pipeline:

1. Exports **MIDI** via MuseScore 4.2+
2. Renders WAV with **tinysoundfont** + configured SoundFont (default MS Basic)
3. Rejects silent renders via RMS/peak checks

Optional **measure-range segmentation** extracts a contiguous measure span from a full score, writes a sliced MusicXML, and regenerates reference audio so partial recordings can be aligned without padding the whole piece.

---

## 6. Performance ingest (Stage 4)

Performance is resampled/normalized to pipeline audio settings and stored as `performance_audio.wav`. Optional **trim** removes leading/trailing silence; re-alignment is run after trim.

Optional **self-reported** marks (performer’s own suspected mistake regions) are stored separately from expert `labels` and never merged automatically.

---

## 7. Alignment and candidate detection (Stage 5)

The current path transcribes performance notes once, detects repeated note phrases, aligns the first pass to the clean score, reapplies repeat links, and derives note-error candidates. It writes `note_alignment_v2.json` and a compact compatibility `alignment.npz`. These are **not** ground truth. ALIGN can subsequently run Layer 3 rhythm detection, but the current DataCreate subprocess bridge requests Layers 1–2 only.

The DTW method below is the legacy fallback retained for old bundles and UI compatibility. New DataCreate alignment and re-alignment requests use the note-first bridge.

### 7.1 Features for alignment

From both performance and reference:

- Default: **chroma** (`librosa` chroma-CQT)
- Optional: CQT chroma with configurable bin count

Features are sanitized (e.g. zero-norm columns) before DTW. Mel spectrograms are **not** used for DTW (timbre confounds alignment).

### 7.2 Dynamic time warping

- Library DTW with **cosine** cost
- **Sakoe–Chiba** band: `dtw_band_ratio` (default 0.1 of sequence length)
- Full-sequence alignment (`subseq=False`)

Outputs stored in `alignment.npz`:

- `ref_features`, `perf_features`
- `warping_path`, `dtw_cost`
- `frame_residuals` (L2 of chroma vectors along the path)
- `hop_length`, `sample_rate`

**Note:** Frame residuals are stored for analysis/UI; they are **not** emitted as `rhythm_error` (rhythm uses duration ratios below).

### 7.3 Score-event mapping

MusicXML notes/rests are extracted (per part) with:

- Quarter-length offset/duration → reference seconds via tempo (MetronomeMark, else 120 BPM)
- DTW path → median performance frame per reference frame, with interpolation
- Each event gains `perf_start` / `perf_end` (and optional residual mean)

Shared by Stage 5 rhythm detection and the annotate UI (`build_note_alignment`).

### 7.4 Onset refinement (post-DTW)

DTW often places note starts in **leading silence**. After mapping:

1. Compute a global **RMS** envelope of the performance
2. For each **non-rest** event, search `[perf_start − lookback, min(perf_end − ε, perf_start + max_shift)]`, clipped so search does not steal the previous event’s body
3. Floor = 20th percentile of window RMS; threshold = floor × 10^(rise_db/20)
4. Snap `perf_start` to first rising edge above threshold; keep original as `perf_start_dtw`

Defaults: lookback 0.15 s, max shift 0.6 s, rise 8 dB. Rests unchanged. This improves EWMA rhythm features and staff/waveform spans in the UI.

### 7.5 Auto candidate types

#### Pitch / intonation (along warping path)

For successive path steps with advancing reference:

- **wrong_note:** chroma peak pitch-class differs and both peaks are strong enough
- **intonation_error:** same class path but chroma-vector “cents” proxy `|arccos(cos sim)| × 1200/π` exceeds `cents_tolerance` (default 20)

#### Rhythm (score-event duration ratios)

After onset refine (and merging consecutive rests on the same part):

\[
r_i = \frac{\mathrm{perf\_dur}_i}{\mathrm{ref\_dur}_i}
\]

Interpretation: \(r > 1\) → slower than reference; \(r < 1\) → faster.

1. **EWMA jump:** maintain EWMA of recent ratios (`α = rhythm_ewma_alpha`). Flag if
   \(|\log(r_i / \mathrm{EWMA})| >\) `rhythm_ewma_log_threshold` (~0.25 ≈ 28% jump).
2. **Far-window drift:** compare \(r_i\) to median of a lagged window (`far_gap` / `far_window`); flag if
   \(|\log(r_i / \mathrm{median})| >\) `rhythm_far_log_threshold`.

Candidates are `rhythm_error` with comments distinguishing EWMA vs far-window. Soft onsets / rubato / ornaments may false-positive; expected to be cleaned in review.

#### Structural miss / extra

- Reference frames never visited by the warping path → `missed_note`
- Performance frames never visited → `extra_note`

Short regions are expanded to `min_candidate_duration_sec` (default 0.15 s). Overlapping same-type candidates are merged.

### 7.6 Output

`candidates.json`: same label shape as final labels, `source: "auto"`, typically without severity/human comments until Stage 6.

---

## 8. Human annotation (Stage 6)

Local FastAPI app + browser UI (`datacreate serve`):

- Waveform timeline (wavesurfer), zoom/scrub, region drag
- Score view (OpenSheetMusicDisplay); alignment mode overlays mapped note spans / EWMA strip
- Candidate workflow: confirm → `auto_confirmed`, edit → `auto_edited`, reject → `auto_rejected`
- Manual regions → `source: "manual"`
- Score segment + performance trim + re-align for partial pieces
- Optional inter-annotator compare (`review.sampling_rate`)

**Policy:** Do not train on raw `source: "auto"` without human confirm/reject.

---

## 9. Features and bundle (Stages 7–8)

- Log-mel (power → dB) for performance and reference → `*_mel.npy` + preview PNGs
- `labels.json` template / human-filled labels
- `metadata.json`: sample id, schema, mel/alignment params, mode flags

Validation: `datacreate-validate` against Pydantic models (`schema_version`, non-zero durations, taxonomy).

---

## 10. Label schema

`labels.json` accepts **1.1** and **1.2**. Times are always required and are always on `performance_audio.wav`. Schema **1.2** adds an optional score-part melody compiled from the clean `verified_score.musicxml`.

### 10.1 Fields

| Field | Role |
|-------|------|
| `id` | Stable id (`cand_###`, `syn_###`, or UI-assigned) |
| `source` | `auto` / `auto_confirmed` / `auto_edited` / `auto_rejected` / `manual` / `synthetic` / `pipeline` |
| `start_time`, `end_time` | Seconds on performance audio |
| `type` | Taxonomy string |
| `severity` | Optional ordinal (human) |
| `deviation_cents`, `deviation_ms` | Optional numeric aids |
| `measure_number`, `note_id`, `comment` | Optional provenance |
| `repeats_label_range` | Required for `repetition`: the first-pass span that was replayed |
| `score_part` | Inclusive clean-score note indices (`start_note_index` … `end_note_index`), measures, and `pad_notes` |
| `pitches` | MIDI pitch list of that span, in written order |
| `note_ids` | Matching `note_0000`-style ids on the clean score |
| `extra_copies` | On `repetition` only: `1` = played twice, `2` = three plays |

Document-level: `schema_version`, `audio_reference`, `annotator_id`, `self_reported[]`, `labels[]`.

**Taxonomy (default):** `wrong_note`, `intonation_error`, `missed_note`, `extra_note`, `rhythm_error`, `repetition`, `stylistic_choice`. Synth generation also plants clarinet **squeaks** as `wrong_note` or `extra_note` (MIDI C6–A7), not as a separate gold type.

### 10.2 Gold melody (schema 1.2)

The labelled melody is a **contiguous slice of the clean score**, then 1–2 notes of padding on each side. Padding **clamps** at the first and last sounding notes; it does not wrap.

| Planted fault | Core (before padding) |
|---------------|------------------------|
| `wrong_note`, `missed_note`, `intonation_error`, `rhythm_error` | The written note(s) that were altered |
| `repetition` | The written notes in the replayed measure(s) |
| `extra_note` | The clean notes **immediately before and after** the insert. The extra itself is not on the clean score. If the insert is after the last written note, only that last note is the core. |

Then expand `[core_i0, core_i1)` by `pad_notes` ∈ `{1, 2}` on each side and store `score_part`, `pitches`, and `note_ids`.

**Repeated-pass copies are not gold.** Labels whose comment contains `repeated pass` or `(pass N)` for N > 1 are skipped at eval. The first pass and the single `repetition` label are kept.

### 10.3 Official note-wise evaluation

Official synth metric: `align-model eval-melodies` (also the headline of `align-model smoke`).

Matching is **exclusive** (Hungarian 1-1) on canonical score-event identity. Canonical event indices, tied spans, repeated-pass/copy identity, and audited EXTRA identity define location. Equal pitch lists at different score locations never match. At the same canonical location the same error type scores 1 and a different type scores 0.5; a different location scores 0.

- Precision = sum of pair credits / number of predictions
- Recall = sum of pair credits / number of golds
- Headline = F1 of those two

Empty gold and empty prediction scores 1. Schema 1.1 labels without a validated score-event projection are officially unavailable. Timestamp IoU and 20/50/100 ms onset scores are retained only under `legacy_*` or `diagnostic_*` fields and cannot select checkpoints or pass promotion gates. Historical timestamp and pitch-similarity headlines remain historical diagnostics; they require recomputation and are never reinterpreted as note-wise results.

---

## 11. Synthetic data

There are two generators. Neither replaces human-reviewed real takes.

### 11.1 DataCreate Stage 9

Programmatic corruptions of an **existing** MusicXML in place, then render and align. Labels are `source: "synthetic"`. Useful for a few fixtures; not the volume path.

### 11.2 synth-pipeline (volume path)

A **separate package**. The clean `verified_score.musicxml` stays correct. Errors are written only into `performance_score.musicxml`, which is rendered as clarinet `performance_audio.wav`. Labels are known at plant time (`source: "synthetic"`, schema **1.2**).

Default single-error config: `synth-pipeline/config/default.yaml`. Multi-error 10k config: `synth-pipeline/config/multi_error_10k.yaml`.

#### Score generation

Procedural clarinet etudes (or `--score` to corrupt existing MusicXML): 8–16 measures; meters 4/4, 3/4, 2/4, 6/8; major/minor keys; tempo 72–112 BPM; written range E3–C6; occasional rests and syncopation.

#### Planted errors (1–8 per clip)

`per_clip_min` / `per_clip_max` draw how many **content** errors to plant. The five types below have **equal weight**. They do not overlap in time on the same clip.

| Type | What is planted |
|------|-----------------|
| `wrong_note` | Shift ±1 or ±2 semitones inside the written range, **or** (when `squeak.prob` is enabled; it is 0 in the current checked-in configs) replace the pitch with a high squeak MIDI **C6–A7**. |
| `extra_note` | Split a note and insert a neighbor (±1–2 semitones) **or** a C6–A7 squeak in the second half. Gold core is the clean notes before and after that insert. |
| `missed_note` | Replace a written note with a rest of the same duration. |
| `intonation_error` | Keep the written pitch class; detune the render with MIDI pitch bend, 40–80 cents, sometimes a run of up to 4 notes. |
| `rhythm_error` | One of: **late start** (rest inserted before the note, offset unchanged), **early start** (onset steals a preceding rest), **late end** (offset steals a following rest), **early end** (note shortened, rest fills the tail), **sudden tempo change** (MetronomeMark 0.68–0.82× or 1.2–1.4× for 1–2 measures, then restore), **uneven rhythm** (3–4 note durations in a measure reweighted so the bar still sums). |

#### Repetition

Repetition is **not** one of the 1–8 planted draws. After the content errors:

1. With `repetition_prob` (0.80 in the 10k config) replay the measure(s) that contain those errors. This is the common case.
2. If that did not fire, with `standalone_repetition_prob` (0.20) replay a random measure that has notes, even if the error is elsewhere.

`repeat_extra_copies_weights`: `1` → two plays (70%), `2` → three plays (30%).

**Adjustment rest.** Immediately before the replayed copy, insert a silent gap of **0.2–1.0 s** (`repeat_gap_seconds`). The gap is not part of the gold melody. `repeats_label_range` stays the first-pass span; the `repetition` label covers the restart after the rest.

First-pass content labels keep a score part. Repeated-pass copies of those labels do not.

#### Backfill

Existing 1.1 bundles (times only) are converted in place:

```powershell
synth-pipeline convert-labels --root ./output --force --pad-random --workers 8
```

`--force` is required after the extra-neighbor rule changed. `--pad-random` picks pad ∈ {1, 2} per bundle unless the file already stored `score_part.pad_notes`. Extra-note conversion maps the extra’s **performance time** onto the clean score (it does not use the inserted MIDI from the comment) and then expands to the neighbors.

---

## 12. Configuration surface

Real-data pipeline: `DataCreate/config/default.yaml`.

| Block | Controls |
|-------|----------|
| `paths` | MuseScore, SoundFont, Audiveris, samples/raw roots |
| `audio` / `mel` | Resample and spectrogram |
| `alignment` | Feature, DTW band, cents, EWMA/far-window, onset refine |
| `taxonomy` | Closed label enum |
| `musescore` / `omr` / `review` / `synthetic` | Tooling and secondary modes |

Synth volume path: `synth-pipeline/config/default.yaml` (one error) and `synth-pipeline/config/multi_error_10k.yaml` (1–8 errors, squeaks, rhythm kinds, standalone repetition, 0.2–1 s restart gap).

---

## 13. Methodological limitations and known gaps

Current design accepts these tradeoffs (see also `SHORTPLANS.md`):

- Chroma peak / cosine “cents” are **proxies**, not F0 trackers—weak for soft/noisy clarinet attacks.
- Missed/extra from unmatched **frames** can be noisy vs true note events.
- Rhythm uses **local duration ratios vs tempo memory**, not absolute DTW slope; breaths, phrase gaps, ornaments, and soft onsets still FP.
- Onset refine assumes a clear energy rise; very soft attacks may still distort ratios (often labeled as rhythm by design for clarinet).
- Early release, dedicated residual/match-quality labels, ornament/trill types, and score-relative IOI pattern detectors are **deferred**.
- Training should use human-reviewed sources only.

---

## 14. Intended use of outputs

- **Annotator / crop trainers:** `start_time` / `end_time` on `performance_audio.wav`.
- **Note-wise eval and score-informed training:** canonical `score_part` event indices on the clean `verified_score.musicxml`. Official scoring is exclusive one-to-one identity F1 (`align-model eval-melodies`) with 1.0 exact-type and 0.5 wrong-type credit at the same location. `pitches` validate a range but never identify it.
- Alignment NPZ and `candidates.json` are intermediate. Real takes should still be human-reviewed; synth labels are known by construction.

---

## References in repo

- Spec / original requirements: `DataCollectionPipelinePrompt.md`
- Operator docs: `DataCreate/README.md`, `synth-pipeline/README.md`, `align-model/README.md`
- Deferred ideas: `SHORTPLANS.md`
- Core code: `DataCreate/src/datacreate/melody.py`, `DataCreate/src/datacreate/models.py`, `synth-pipeline/src/synthpipeline/errors.py`, `align-model/src/alignmodel/eval_melodies.py`
