# Spoken performance feedback

`datacreate-feedback` takes ALIGN label JSON, asks a configured chat-completions
API for an explanation, and sends narration to Fish Audio or Qwen Audio for MP3.
It also exposes `datacreate.feedback.run_feedback` for integration. It does not
rerun detection or change labels, and is independent of the GUI alignment model.

For speech synthesis without a Fish API key, see [local Fish setup](local_fish.md)
and use `DataCreate/config/feedback.local.yaml` after starting the local server.

For a synchronized 480p, 24 fps score animation, see [animated score feedback](feedback_video.md).
Use `--all-labels` when generating audio for a video that covers every retained label.

## Setup

### Qwen Audio 3.1 TTS Flash (China / Beijing)

Use `DataCreate/config/feedback.ssstoken.qwen.yaml`. This keeps the existing
ssstoken `gpt-6-luna` narration and replaces speech with `qwen-audio-3.1-tts-flash`.
No local Fish server or provider SDK is required. The Qwen-Audio HTTP protocol
is different from the older Qwen3-TTS API; do not substitute their endpoints.

1. Sign in to Alibaba Cloud China and open Model Studio / the Qwen API console.
2. Select **China (Beijing)**, activate the service if prompted, and create a
   Model Studio API key in the workspace you will use. This is not a RAM AccessKey.
3. Copy that workspace's **workspace ID** from its details page.
4. Enable the model and sufficient account balance/billing in the console.
   The key itself is not a subscription purchase: synthesis usage is billed.

Set these in the terminal that starts the pipeline:

```powershell
$env:DASHSCOPE_API_KEY = "your-Beijing-Model-Studio-key"
$env:DASHSCOPE_WORKSPACE_ID = "your-workspace-id"
$env:ALIGN_FEEDBACK_CONFIG = (Resolve-Path DataCreate/config/feedback.ssstoken.qwen.yaml).Path
```

Alternatively, save the first two as Windows **user environment variables**;
the pipeline reads those dedicated settings even if its shell is already open.
Keep actual keys out of YAML, source control, and chat. `SSSTOKEN_API_KEY` remains
the separate credential for the narration LLM. Studio still defaults to local
Fish unless `ALIGN_FEEDBACK_CONFIG` selects the Qwen config; restart Studio with
that override to switch.

The starting English voice is `Abby_v3.1`, with `qwen_rate: 0.95` and a warm
teacher instruction in `qwen_instruction`. For Mandarin change both
`language: Chinese` and `qwen_voice: xieshurou_v3.1`. Premium English presets
support English only, and premium Chinese presets support Mandarin only.
Choose a documented compatible voice before requesting mixed-language output.

Test saved text without another LLM request:

```powershell
$env:PYTHONPATH = 'DataCreate/src'
python -m datacreate.feedback --text path/to/plain-narration.txt `
  --config DataCreate/config/feedback.ssstoken.qwen.yaml --output feedback/qwen-trial
```

For narration with music examples, reuse its `playback_plan.json` via `--plan`
instead of `--text`. This regenerates speech, preserves the music clips, matches
loudness, and rebuilds `timeline.json` from the new durations. Render the video
again from this new feedback directory; do not reuse the previous MP4 timing.

Each generated speech MP3 has a `.tts.json` sidecar containing model, voice, and
provider-reported token usage. Signed audio URLs and API keys are never saved.
The published Beijing list price checked October 5, 2026 is CNY 1.5 per million
input tokens and CNY 12 per million output tokens (not a per-character price).
Failed calls are not automatically retried; saved narration allows explicit
recovery without paying for a second LLM request.

Official references: [HTTP API](https://help.aliyun.com/en/model-studio/qwen-audio-tts-http-api),
[voice list](https://help.aliyun.com/en/model-studio/qwen-audio-tts-voice-list),
[pricing](https://help.aliyun.com/en/model-studio/qwen-audio-3-1-tts-flash).

### Fish Audio

#### Latest hosted Fish alternative (Drama 3 Preview)

`DataCreate/config/feedback.fish.yaml` keeps the existing ssstoken
`gpt-6-luna` narrator and calls `https://api.fish.audio/v1/tts` with the exact
`model: drama-3-preview` header. Fish's September 23, 2026 release is its newest
documented TTS preview as of October 5, 2026. Its behavior and availability may
change. The latest recommended production model is `s2.1-pro`; select that by
editing `fish_model` in this config. The pipeline makes no automatic fallback
or retry. It rejects unknown model IDs on the official endpoint because Fish
otherwise silently falls back to S2.1 Pro.

Create a Fish API key and select a hosted voice model in your Fish library.
The voice ID is separate from the synthesis engine ID. The local `teacher_lj`
folder is not a hosted voice; pick a conversational teacher-like English voice
to begin. This configuration does not upload local reference recordings.

```powershell
$env:FISH_AUDIO_API_KEY = "your-Fish-API-key"
$env:FISH_AUDIO_REFERENCE_ID = "your-hosted-voice-model-id"
$env:ALIGN_FEEDBACK_CONFIG = (Resolve-Path DataCreate/config/feedback.fish.yaml).Path

# Replay an existing comparison plan without another LLM request.
$env:PYTHONPATH = 'DataCreate/src'
python -m datacreate.feedback `
  --plan feedback/sample-001-sounding-pitch-corrected/playback_plan.json `
  --config DataCreate/config/feedback.fish.yaml `
  --output feedback/sample-001-fish-drama3
```

Both Fish variables can also be saved as Windows user environment variables.
`SSSTOKEN_API_KEY` is needed only when generating new narration. To switch
Studio, start it with the `ALIGN_FEEDBACK_CONFIG` override above. Qwen and local
Fish configurations remain independently selectable.

Hosted calls use `latency: normal` (quality-focused), 44.1 kHz MP3, loudness
normalization, and previous-chunk conditioning. `fish_speed: 1.0` requests the
voice's normal pace. `speech_min_wpm: 140` gently accelerates slower English
paragraphs during comparison assembly with pitch-preserving FFmpeg `atempo`,
capped at 1.25x. It leaves short cues (under 12 words), faster speech, and other
languages unchanged. Set this to zero to disable correction. Original speech
MP3s are retained beside adjusted `*-paced.wav` files, and the timeline records
the applied speed factors. Music slowdown is independent. These hosted controls are not sent to local
Fish 1.5. Music clips, loudness matching, silence margins, and reverb are
preserved, and the timeline is rebuilt for the new voice before video export.

Pricing checked October 5, 2026: S2.1 Pro is $15 per million **UTF-8 bytes**;
1,000 ASCII English characters cost about $0.015, while 1,000 typical Chinese
characters use about 3,000 bytes ($0.045). Drama 3 Preview is not separately
listed in the public pricing table; check your Fish console before a paid run.
The documented `s2.1-pro-free` evaluation option uses the same S2.1 model under
fair-use limits, currently advertised through November 30, 2026. This is Fish's
direct hosted API; mainland-China inference location has not been verified.

Sources: [releases](https://docs.fish.audio/developer-guide/getting-started/changelog),
[API](https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech),
[pricing](https://docs.fish.audio/developer-guide/models-pricing/pricing-and-rate-limits),
[API-key setup](https://docs.fish.audio/developer-guide/getting-started/api-key).

#### Original 302.AI + Fish configuration

From the repository root, in your Python environment:

```powershell
python -m pip install -e ./DataCreate

# Set locally or through a secret manager; never commit actual values.
$env:API_302_KEY = "your-302-ai-key"
$env:FISH_AUDIO_API_KEY = "your-fish-audio-key"
$env:FISH_AUDIO_REFERENCE_ID = "your-selected-fish-voice-model-id"
```

Choose a voice from your Fish Audio library. The reference ID selects the voice;
`fish_model` selects the synthesis engine. The default engine is `s2.1-pro`;
the documented developer-tier option is `s2.1-pro-free`.

Edit `DataCreate/config/feedback.yaml` to choose a chat-completions model enabled
on your 302.AI account, language, instrument, timeout, or alternate HTTPS URL.
The default LLM is `gpt-4o-mini`. Account access and billing depend on the provider.
Another chat-completions service can use its own environment-variable name via
`llm_api_key_env`; credentials are never implicitly reused across providers.

## Run

Select the exact file to explain: `labels.json` normally holds human/synthetic
gold, while `labels_agent.json` holds predictions. Documents with a `labels`
array and bare label arrays are accepted, including pipeline/melody predictions.

```powershell
# Inspect the outbound prompt offline, without keys or charges.
datacreate-feedback --labels DataCreate/samples/020/labels_agent.json `
  --config DataCreate/config/feedback.yaml --dry-run

# Generate narration and audio.
datacreate-feedback --labels DataCreate/samples/020/labels_agent.json `
  --config DataCreate/config/feedback.yaml --output feedback/take-020
```

You can also use `python -m datacreate.feedback` with the same flags and
`DataCreate/src` on `PYTHONPATH`. This config is separate from `default.yaml`.

The pipeline automatically reads `verified_score.musicxml` beside the labels.
For a score stored elsewhere, supply `--score path/to/matching.musicxml`.
It resolves score indices locally into locations such as "the third to fifth
notes of bar 8", so the LLM does not calculate or guess note positions.
Use the same single-part score that produced the labels. Counting follows ALIGN's
sounding-note convention: rests and decorative notes are excluded, and tied
continuations count as part of the original note. Note numbers restart in each
bar; bar numbers use the enumeration in the full score. `full_score.musicxml`
and `metadata.json`'s `score_segment` map renumbered selections to their original
bars without adding an offset twice. For a selection that starts mid-bar, the
full score supplies the count of preceding notes. If that score is missing,
feedback uses the bar alone rather than inventing the within-bar ordinal.
Without a score, feedback uses the known bar or "the marked passage"; it never
invents within-bar positions.

Without `--output`, each invocation creates a unique run below the input's
`feedback/` directory. An explicit output directory must not already exist.

## Performance examples in the feedback

Musical examples are synthesized locally, in this order: **full-score bar and
reference cue → synthesized reference → performance cue → synthesized
transcription → diagnosis and practice advice**. Both use the reference MIDI's
instrument. The performance example preserves the transcription's pitches,
relative onsets, note lengths, and pauses, including extra notes and wrong
pitches; missed notes are not filled in. Ignored transcription events are excluded.
This is an approximation of notes and timing, not the student's tone, breath,
or dynamics. Decoder note ends may be estimates rather than acoustic offsets.
ALIGN's written-pitch transcription is converted to sounding pitch before
playback (B-flat clarinet: minus two semitones), using its declared pitch
convention or the validated score/MIDI transposition. This preserves wrong-note
intervals as well as correct notes. A transcription explicitly marked
`pitch_space: sounding` bypasses conversion to avoid transposing it twice.

Every selected error expands to the complete bar or inclusive range of bars
containing its core notes. Context padding does not expand the error itself.
The narrator uses full-score bar numbers instead of note ordinals, and does not
imply every note in the example is wrong. At most three teaching points are
selected; neighboring errors may share one comparison.

Examples require `verified_score.musicxml`, its matching `reference_audio.mid`,
and `note_alignment_v2.json` beside the labels. The alignment's embedded
`transcribed_notes` is used so notes match the alignment that located them.
Override with `--score`, `--reference-midi`, `--note-alignment`, or
`--transcription` (a note array or an object containing `transcribed_notes`).
An override transcription must use the same time origin as the alignment.
For concert-pitch overrides, use an object with `pitch_space: sounding` and
`transcribed_notes`; a bare array inherits the alignment's pitch convention.
A SoundFont is required; configure `paths.soundfont` in the pipeline config.
On Windows, MuseScore's installed MS Basic SoundFont is detected automatically.

The reference MIDI is checked against the selected score, including its
written-to-sounding transposition. Rendered ornaments and tempo changes are
preserved. `full_score.musicxml` and selection metadata provide full-score
numbering and reference notes outside partial-bar selections. Only the recorded
portion of a partial selection can be synthesized as performance; the narrator
acknowledges this. Aligned notes locate performance bars, with boundary rests or
missing edge notes estimated from local timing. Internal transcription timing
is never quantized. Missing score/alignment data fails clearly; use
`--no-excerpts` for narration alone.

Both versions receive exactly the same time multiplier. A brief example targets
at least three seconds; fast notes target a 10th-percentile note duration/onset
interval of 0.22 seconds. The larger required slowdown is used, capped at 4× so
short transcription glitches cannot create excessively long examples. This cap
means unusually brief events can remain shorter than the target. Pitches stay
unchanged. The narrator mentions when both versions are slowed. The student's
relative tempo differences remain audible.

The original recording is not embedded or uploaded. If available, it is used
only to validate label time bounds (`audio_reference`, `performance_audio.wav`,
or `--performance`). No reference WAV is needed; the legacy `--reference` flag
only locates its paired MIDI. `excerpt_padding_seconds` applies to legacy crop
helpers, not these whole-bar examples. The LLM receives label facts and example
availability/bar/slowdown information; all music and note-event data stay local.

Assembly matches speech and music with gated, frequency-weighted loudness
(normally -20 LUFS). One gain per segment preserves internal dynamics. The
shared target is lowered if needed to keep estimated peaks below -2 dBFS or
avoid boosts above 30 dB; silent examples remain silent.

Music snippets receive short edge fades and subtle room reverb (8% wet amplitude,
0.25-second tail). Speech stays dry for clarity. There is 0.5 seconds of silence
before each snippet and another 0.5 seconds after its reverb tail, so narration
does not interrupt the decay. The finished MP3 is 44.1 kHz stereo. Labels may
use score indices without timestamps when the note alignment provides timing.

Use `--no-excerpts` or `include_performance: false` for narration alone.
Without a recording or alignment, the plain narration workflow continues to work. The excerpts mode also works with `--text-only`: it saves
the plan and selected clips for review without synthesizing speech.

| File | Contents |
|---|---|
| `report.json` | Normalized observations, counts, excluded-label counts. |
| `request.json` | LLM model and messages, without authentication headers. |
| `feedback.txt` | Spoken text; excerpt runs also include readable `[Performance excerpt]` markers that are not sent to Fish. |
| `feedback.mp3` | MP3 published after a successful response and header check. |
| `feedback.json` | Run status, model names, input/text hashes, output filenames. |
| `playback_plan.json` | Selected labels, spoken cues/corrections, and verified reference/performance clip files; old performance-only plans remain replayable. |
| `excerpt-NNN.wav` | Synthesized performance transcription covering the selected bars. |
| `reference-NNN.wav` | Synthesized reference covering the same bars and slowdown. |
| `*.notes.json` | Exact synthesized pitch/onset/end events, source bounds, bars, slowdown, and alignment hash. |
| `speech-NN-*.mp3` | Individual synthesized introduction, performance cue, and feedback segments. |
| `timeline.json` | Segment order, final MP3 positions, source excerpt times, loudness/gains, reverb tails, and silence margins. |

A dry run writes only the report, request, and manifest. Status is `dry_run`,
`text_ready`, `complete`, `llm_failed`, or `tts_failed`, with `prepared` during
the initial request. MP3 header checks detect common invalid responses; they
are not a full audio decode or a guarantee of speech quality.

## Text review and speech retries

```powershell
datacreate-feedback --labels path/to/pipeline_pred.json --text-only `
  --config DataCreate/config/feedback.yaml --output feedback/review

# Review/edit the saved text, then synthesize without another LLM request.
datacreate-feedback --text feedback/review/feedback.txt `
  --config DataCreate/config/feedback.yaml --output feedback/spoken
```

If Fish fails, narration remains available. Retry with `--text` into a new run.
For a run containing performance excerpts, retry with its saved plan instead:

```powershell
datacreate-feedback --plan feedback/review/playback_plan.json `
  --config DataCreate/config/feedback.ssstoken.teacher.yaml `
  --output feedback/spoken-with-excerpts
```

This makes no LLM request and preserves the saved musical clips. Legacy plans
remain replayable with their original clips; regenerate from labels to adopt synthesis. Keep the
plan beside its excerpt WAV files; hashes are checked before reuse. The practice
studio also uses this plan when retrying a failed speech stage.
Full runs check both credentials and the voice ID before calling the LLM.
There are no automatic paid-request retries: a timeout may follow a billable
generation. Errors report status without printing provider bodies or API keys.

## Interpretation and scope

The prompt asks for a music teacher speaking directly to the student, with
2-4 sentences, roughly 40-80 English words, and at most three useful points.
It uses bar and note positions instead of timestamps, with a short correction
and practical advice. A single issue usually needs only two sentences.
English integer bar numbers and numeric ordinals are expanded into spoken words
before saving `feedback.txt` and sending it to Fish (for example, `bar 2` becomes
`bar two`). This also applies when retrying saved narration with `--text`.
The prompt grounds claims in supplied labels, keeps automatic findings tentative,
distinguishes core notes from padding, and prohibits claims of having listened.
Counts represent regions, not necessarily individual notes. Empty labels mean
no issues were marked, not proof of a perfect performance. Rejected candidates
and repeated-pass content copies are excluded; first-pass content and repetition
labels remain. These prompt constraints do not
mechanically guarantee factual accuracy; use text review when needed.

Only selected label fields reach 302.AI. Checkpoint paths, annotator IDs, score
files, and performance audio are not uploaded. Fish receives only narration and
the chosen voice ID. The outbound labels contain only type, source, derived score
location, repetition count, and synthesized example context when available. Timestamps, global note indices,
MIDI pitches, and comments are omitted from the LLM request. The local report
retains normalized fields, including comments, and should be treated as private
performance data.

Raw stack alignment files (`events`, `score_span`, `missed_score_event_indices`)
are rejected. Convert them with the existing `labels_from_stack_alignment`
adapter in `align-model/scripts/eval_datacreate_vs_human.py` first. Its current
miss/repetition timestamps are placeholders: omit them before narration and
retain score locations.

Oversized reports fail explicitly. Narration prioritizes at most three useful
points. Narration exceeding `max_speech_chars` fails before TTS; the shorter
word and sentence targets are prompt instructions, not a hard output validator.
There is no automatic GUI action or extra charge during ordinary annotation.

## Provider references and testing

- [302.AI documentation](https://doc.302.ai/):
  `https://api.302.ai/v1/chat/completions`, bearer authentication.
- [Fish Audio TTS](https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech):
  `https://api.fish.audio/v1/tts`, JSON body with `text`, `reference_id`,
  `format: mp3`; bearer authentication and engine in the `model` header.

Tests mock HTTP to check request contracts, filtering, empty reports, validation,
failure recovery, MP3 streaming, and secret handling. They do not establish that
any particular account/model/voice is live.
