# Spoken performance feedback

`datacreate-feedback` takes ALIGN label JSON, asks 302.AI for an English
explanation, and sends the narration to Fish Audio directly to produce an MP3.
It also exposes `datacreate.feedback.run_feedback` for integration. It does not
rerun detection or change labels, and is independent of the GUI alignment model.

For speech synthesis without a Fish API key, see [local Fish setup](local_fish.md)
and use `DataCreate/config/feedback.local.yaml` after starting the local server.

## Setup

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

When a matching recording and timed labels are available, feedback now plays
each selected point as **full-score bar and reference cue → reference excerpt →
performance cue → actual performance excerpt → diagnosis and practice advice**.
With a reference example, the narrator uses the bar number and lets the music
identify the notes instead of saying "seventh" or "eighth". The model chooses
at most three labeled points and returns `intro`, `performance_intro`, and
`feedback` sentences.
Audio markers and JSON are never spoken. The LLM receives only label facts and
an indication that an excerpt is available; the recording stays local.

The recording comes from the label document's `audio_reference`, or from
`performance_audio.wav` beside the labels if no reference is specified.
Use `--performance path/to/recording.wav` to select it explicitly (MP3 is also
supported). Label times must refer to that exact recording: DataCreate labels
usually address the trimmed `performance_audio.wav`, not the original full
recording. Out-of-range times fail before a paid LLM request.

Reference examples use `reference_audio.wav` and its paired `reference_audio.mid`.
Use `--reference` and `--reference-midi` to override these paths. The MIDI must
be the file used to render the selected score's reference, including its tempo
changes. Sounding note count and pitch sequence are checked against MusicXML,
allowing the constant written-to-sounding transposition used by clarinet.
The snippet covers the label's score span, including supplied context notes;
it does not imply that every note in the example is wrong. Release tails stop
before the next note. No duration scaling or performance-to-reference timestamp
guessing is used. A stale or mismatched MIDI fails before any paid request.

When no reference recording is available, the performance-only format remains
available. In that fallback, spoken note positions use "the nth note of bar x"
and still refer to the full score. If a reference recording exists but its score
or matching MIDI is missing, supply those files or use `--no-excerpts`.

Clips include 0.25 seconds of context on either side, bounded by the recording.
Set `excerpt_padding_seconds` in YAML to change this (0–2 seconds). Assembly
keeps the original pitch and tempo, adjusts each excerpt's level with one bounded
gain, adds short edge fades, and leaves 0.3-second pauses between speech and music.
The finished MP3 is 44.1 kHz stereo. Missing label times produce spoken feedback
without an excerpt for that point; no times are guessed from score positions.

Use `--no-excerpts` or `include_performance: false` for narration alone.
Without a matching recording or any timed labels, the plain narration workflow
continues to work. The excerpts mode also works with `--text-only`: it saves
the plan and selected clips for review without synthesizing speech.

| File | Contents |
|---|---|
| `report.json` | Normalized observations, counts, excluded-label counts. |
| `request.json` | LLM model and messages, without authentication headers. |
| `feedback.txt` | Spoken text; excerpt runs also include readable `[Performance excerpt]` markers that are not sent to Fish. |
| `feedback.mp3` | MP3 published after a successful response and header check. |
| `feedback.json` | Run status, model names, input/text hashes, output filenames. |
| `playback_plan.json` | Selected labels, spoken cues/corrections, and verified reference/performance clip files; old performance-only plans remain replayable. |
| `excerpt-NNN.wav` | Exact selected recording range with context, before final mix gain and fades. |
| `reference-NNN.wav` | Corresponding passage from the rendered reference audio. |
| `speech-NN-*.mp3` | Individual synthesized introduction, performance cue, and feedback segments. |
| `timeline.json` | Segment order, final MP3 positions, source excerpt times, and mix gains. |

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

This makes no LLM request and preserves the selected recording clips. Keep the
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
location, and repetition count when available. Timestamps, global note indices,
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
