# Practice studio

Open **http://127.0.0.1:8765/studio** after starting `datacreate serve` (or
`datacreate-serve`). There is also a Practice studio link in the annotation UI.

If port 8765 is already in use, the GUI may already be running. Open the printed
URL instead of launching another copy. For a separate instance, use
`datacreate serve --port 8767` (or `datacreate-serve --port 8767`), choosing an
unused port. To reload changed Python code, stop the existing server with Ctrl+C
in its terminal and start it again. The launcher reserves the chosen port before
starting the app and reports a clear message when the port is occupied.

1. Select a score from the configured `paths.raw_data_score` directory or upload
   MusicXML (`.musicxml`, `.xml`, `.mxl`, up to 10 MB). Use the score for the passage
   you intend to play; this page does not select measure excerpts.
2. Start recording and allow microphone access. Play for 1–300 seconds.
3. Press **Stop & get feedback**. The take uploads and the pipeline starts
   automatically. At five minutes recording stops and submits automatically.
4. Follow the actual processing stages, then play or download the feedback MP3.
   The narration is also available as text. Your original take remains playable
   in the page until you record another take or reload.

Alternatively, select **Upload audio** below the recording controls. Choose a
WAV, MP3, M4A, or another audio format supported by your browser (1–300 seconds,
up to 60 MB). Listen to the preview, select the matching score, then click
**Get my feedback**. No microphone permission is needed for uploads. The browser
decodes the file, mixes its channels to mono, and converts it to 48 kHz PCM WAV
before sending it through the same analysis and local Fish speech pipeline.
Unsupported or overlong files show an error and leave your previous take intact.

During processing, the detailed progress bar shows **Transcriber → Aligner →
Label analysis → Narration → Speech audio**. Updates come from the model
subprocess and feedback service. The bar counts completed steps, not elapsed
time; transcription, alignment, and local speech can take different amounts of
time. Failed runs keep the stopped step visible, and all five steps are marked
complete only after the MP3 is saved.

## Server configuration

Use the existing DataCreate environment with the score renderer and ALIGN model
installed. The studio uses `config/default.yaml` (or the configuration passed to
the server), including its alignment Python, checkpoint, device, and timeout.
It runs the configured GUI alignment pipeline; the current default is experimental stack v9.
Its predicted `candidates.json` is passed to the spoken-feedback service;
the empty human label template is never used as the feedback source.

The browser submits to `/api/studio/takes`; the worker calls the existing
`datacreate.feedback.run_feedback` entry point used by `datacreate-feedback`.
Report preparation, score locations, ssstoken narration, local Fish synthesis,
and MP3 assembly all stay in that shared pipeline. Matching performance audio,
reference audio, MusicXML, and reference MIDI are discovered in the generated
sample directory, enabling the pipeline's reference/performance examples.
Speech retries reuse `playback_plan.json` when present, preserving the excerpts;
plain narration retries reuse `feedback.txt`. The page plays the resulting
`feedback.mp3` directly.

The studio selects the installed `teacher_lj` voice and enables reference and
performance snippets. They require detected labels with usable locations and
times. If no issues are marked, the page and narration explicitly explain why
no targeted clips are available. New output manifests record the feedback
pipeline revision and selected voice; the config API exposes the loaded revision.
See [the Interpretation Pipeline audit](interpretation_studio_audit.md).

Configure the providers as described in [spoken_feedback.md](spoken_feedback.md).
`ALIGN_FEEDBACK_CONFIG` optionally selects a feedback YAML file; without it,
`DataCreate/config/feedback.local.yaml` is used (ssstoken narration with
`gpt-6-luna`, and local Fish Speech
at `http://127.0.0.1:8081`). No Fish API key or hosted voice ID is needed.

```powershell
conda activate MusicEval
./DataCreate/scripts/start_fish_local.ps1
# SSSTOKEN_API_KEY must be available for narration generation.
datacreate serve
```

Only the LLM key is required in local Fish mode. To explicitly use hosted Fish,
set `ALIGN_FEEDBACK_CONFIG` to `DataCreate/config/feedback.yaml` and configure its
Fish credentials. Restart the studio after changing environment variables.
The page reports missing server
environment variables; credentials are never sent to the browser. Configuration
readiness does not prove the provider, renderer, or model is available.

The studio uses the dedicated `SSSTOKEN_API_KEY` variable, so another app's
`OPENAI_API_KEY` cannot override the ssstoken credential. When the key is saved
as a Windows user environment variable, run `datacreate serve` directly. If the
process has not inherited `SSSTOKEN_API_KEY`, both readiness checks and provider
requests read it directly from your Windows user environment. No terminal
application restart is required. Explicit process values take precedence, and
an explicitly empty value disables the fallback. Restart the GUI once after
updating its Python code.

If ssstoken returns HTTP 401, check that its saved credential is still valid.
As an alternative to the persistent user variable, stop the studio and launch
it with the existing provider-and-key file explicitly:

```powershell
conda activate MusicEval
./DataCreate/scripts/start_studio.ps1
```

This launcher reads the ignored repository-root `llmauth.txt`, verifies that it
names ssstoken, and loads only its key into the server process. It prints no
credentials, restores the parent shell's environment on exit, and runs in the
foreground. Optional flags: `-Port 8767`, `-Python path/to/python.exe`, and
`-KeyFile path/to/provider-and-key.txt`. A revoked or expired key will still need
replacement; the launcher does not establish provider authorization.
Microphone recording still works while providers are unconfigured; a failed
submission keeps the take in the page for another attempt.

The server is intended for **one local user**, on loopback, with **one Uvicorn
worker**. Do not expose this unauthenticated annotation server publicly. Only one
studio job runs at a time to bound GPU and provider work; concurrent submissions
receive a busy response and retain their browser recording for resubmission.

## Audio and recovery

The browser captures mono 16-bit PCM WAV through an AudioWorklet, with browser
echo cancellation, noise suppression and automatic gain control requested off.
Use localhost or HTTPS in a current Chrome, Edge, Firefox, or Safari browser.
The microphone graph is muted to avoid feedback and releases tracks on stop or
page exit. Capture is bounded inside the audio worklet even in background tabs.
An analyser FFT feeds a 96-band triangular mel filter bank; log-power
energies animate the decorative bars. Idle bars are decorative. This live visual
is independent of the pipeline's full-resolution saved mel features.

Takes and generated sample artifacts live below `DataCreate/work/studio/` by
default, separate from annotation/training samples. They remain on disk until
you remove them. The current job ID stays in browser session storage, so a reload
can resume status and MP3 playback. A restarted server marks unfinished work as
interrupted rather than replaying requests. Failed feedback can be retried
explicitly; saved narration is reused when speech failed, avoiding a second LLM
request. Retrying a timed-out provider request can incur another provider charge.

HTTP integration tests stub GPU/rendering and speech providers. They verify
orchestration, upload limits, recovery, and artifact delivery, but do not establish
real model accuracy or provider availability.

## Zero labels and inconclusive analysis

Studio checks the model status in `note_alignment_v2.json` before narration.
An `alignment_uncertain` result means the detector withheld feedback; it is not
a successful evaluation with no errors. Studio stops at label analysis, explains
that the recording and selected score need checking, and disables speech-only
retry. The same check applies to older saved jobs and direct narration of
`candidates.json`. Old audio remains on disk but is not offered as valid feedback.

An `ok` result can legitimately have zero labels. The status API exposes
`assessment`, including label count, withheld extra/missed-note count, and the
model's match fraction. This fraction is an alignment diagnostic, not a grade.
The page explains when candidates were filtered by confidence. Human annotation
templates (`labels.json`) are not the prediction source.
