# MusicEval / ALIGN — Methodology

This document describes the methodology of the **MusicEval Data Creation** pipeline (`DataCreate`): how score + performance audio become labeled training bundles for music performance error detection. The current focus is **clarinet** practice takes against notated MusicXML (or PDF→OMR), with human review as the source of ground truth.

---

## 1. Goal and design principles

**Goal.** Produce per-sample training bundles containing:

1. Time-stamped error labels on the *performer’s* audio timeline
2. Log-mel spectrograms of performance and score-derived reference audio
3. Verified MusicXML used as the musical ground truth for synthesis and note mapping
4. Alignment artifacts used for auto-candidates and UI visualization

**Principles.**

| Principle | Implication |
|-----------|-------------|
| Alignment is an aid, not labels | Stage 5 writes `candidates.json` with `source: "auto"`. Humans confirm, edit, reclassify, or reject in the annotator UI. |
| Closed taxonomy | Error types live in `config/default.yaml`; schema versioned (`1.1`). |
| Reference vs performance | Alignment compares real performance to a **synthesized** rendering of the verified score—not another human recording. |
| Pitch features for DTW | Chroma/CQT for alignment (timbre-robust); log-mel for the training features bundle. |
| Partial takes supported | Measure-range score segmentation + waveform trim + re-align for incomplete recordings. |

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
              Chroma/CQT DTW alignment
                          │
         Score-event mapping + onset refine
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
| 5 | DTW alignment + candidate detection |
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

This stage produces (1) a warping path and residuals, (2) score-event ↔ performance time mapping, and (3) auto candidates. It is **not** ground truth.

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

## 10. Label schema (summary)

Per label (schema `1.1`):

| Field | Role |
|-------|------|
| `id` | Stable id (`cand_###` or UI-assigned) |
| `source` | `auto` / `auto_confirmed` / `auto_edited` / `auto_rejected` / `manual` / `synthetic` |
| `start_time`, `end_time` | Seconds on performance audio |
| `type` | Taxonomy string |
| `severity` | Optional ordinal (human) |
| `deviation_cents`, `deviation_ms` | Optional numeric aids |
| `measure_number`, `note_id`, `comment` | Optional provenance |
| `repeats_label_range` | For `repetition` |

Document-level: `schema_version`, `audio_reference`, `annotator_id`, `self_reported[]`, `labels[]`.

**Taxonomy (default):** `wrong_note`, `intonation_error`, `missed_note`, `extra_note`, `rhythm_error`, `repetition`, `stylistic_choice`.

---

## 11. Synthetic data (Stage 9)

Programmatic MusicXML corruptions (pitch shift, timing, delete/insert note, duration change) → corrupted score rendered as “performance,” clean score as reference → full alignment + known `labels.json` with `source: "synthetic"`. Used to stress detectors and bootstrap volume; not a substitute for real annotated takes.

---

## 12. Configuration surface

Central file: `DataCreate/config/default.yaml`.

| Block | Controls |
|-------|----------|
| `paths` | MuseScore, SoundFont, Audiveris, samples/raw roots |
| `audio` / `mel` | Resample and spectrogram |
| `alignment` | Feature, DTW band, cents, EWMA/far-window, onset refine |
| `taxonomy` | Closed label enum |
| `musescore` / `omr` / `review` / `synthetic` | Tooling and secondary modes |

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

Downstream error-detection models consume each bundle’s performance (and optionally reference) log-mel plus time-stamped labels. Alignment NPZ and candidates are intermediate; **reviewed `labels.json`** is the supervision signal.

---

## References in repo

- Spec / original requirements: `DataCollectionPipelinePrompt.md`
- Operator docs: `DataCreate/README.md`
- Deferred ideas: `SHORTPLANS.md`
- Core code: `stages/stage5_alignment.py`, `note_alignment.py`, `pipeline.py`, `models.py`, `config/default.yaml`
