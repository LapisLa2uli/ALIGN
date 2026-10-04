# Practice studio

Open **http://127.0.0.1:8765/studio** after starting `datacreate serve` (or
`datacreate-serve`). There is also a Practice studio link in the annotation UI.

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
It runs the configured GUI alignment pipeline, not experimental stack v9.
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

Configure the providers as described in [spoken_feedback.md](spoken_feedback.md).
`ALIGN_FEEDBACK_CONFIG` optionally selects a feedback YAML file; without it,
`DataCreate/config/feedback.local.yaml` is used (ssstoken narration with
`gpt-6-luna`, and local Fish Speech
at `http://127.0.0.1:8081`). No Fish API key or hosted voice ID is needed.

```powershell
conda activate MusicEval
./DataCreate/scripts/start_fish_local.ps1
# Set OPENAI_API_KEY to your ssstoken key in this shell for narration generation.
datacreate serve
```

Only the LLM key is required in local Fish mode. To explicitly use hosted Fish,
set `ALIGN_FEEDBACK_CONFIG` to `DataCreate/config/feedback.yaml` and configure its
Fish credentials. Restart the studio after changing environment variables.
The page reports missing server
environment variables; credentials are never sent to the browser. Configuration
readiness does not prove the provider, renderer, or model is available.
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
