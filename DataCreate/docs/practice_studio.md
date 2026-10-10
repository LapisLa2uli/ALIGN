# Practice studio

For hardware gateways, see [the robot integration contract](robot_integration.md)
for durable idempotency, crash reconciliation, score selection and job-specific media.

Open **http://127.0.0.1:8765/studio** after starting `datacreate serve` (or
`datacreate-serve`). There is also a Practice studio link in the annotation UI.

For the dedicated upload flow, open **http://127.0.0.1:8765/upload**. Select audio
and its matching score, then submit. The form is replaced by the centered Studio
progress panel as the only page content. Once the shared pipeline finishes,
the page displays **feedback uploaded!** with a video player to preview the generated
feedback MP4. If this take has no MP4, the preview uses the newest completed video
with identical WAV audio samples and the same score and selected passage, and
identifies it as previously generated feedback. WAV header metadata is ignored
when matching. This also works for older saved takes. If no matching video exists,
the page explains why this take has no preview.
Feedback files are saved on this
server; this message does not imply delivery to an external service. Failed jobs show an error with a
retry option when supported; they never display the success message. Reloading
the tab reconnects to its take using a separate session key from the full Studio.

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
4. Follow the actual processing stages, then watch or download the feedback MP4.
   An MP3 download is also available.
   The narration is also available as text. Your original take remains playable
   in the page until you record another take or reload.

Alternatively, select **Upload audio** below the recording controls. Choose a
WAV, MP3, M4A, or another audio format supported by your browser (1–300 seconds,
up to 60 MB). Listen to the preview, select the matching score, then click
**Get my feedback**. No microphone permission is needed for uploads. The browser
decodes the file, mixes its channels to mono, and converts it to 48 kHz PCM WAV
before sending it through the same analysis and configured speech pipeline.
Unsupported or overlong files show an error and leave your previous take intact.

When several score passages fit the recording similarly, analysis assumes the
earliest occurrence in the score. Candidates qualify when their retrieval
similarity is within 0.04 and their normalized alignment rank is within 0.08 of
the best match, and they pass the alignment match-fraction threshold. A clearly
better later match still wins. Feedback and audio examples use the selected
occurrence's original score locations; repeated passages alone no longer cause
all labels to be withheld. Other confidence checks still apply.

During processing, the detailed progress bar shows **Transcriber → Score passage →
Aligner → Label analysis → Narration → Speech audio → Score animation**. Updates come from the model
subprocess and feedback service. The bar counts completed steps, not elapsed
time; transcription, alignment, and local speech can take different amounts of
time. Failed runs keep the stopped step visible. Completion waits for the MP4
to be saved. Audio-only feedback without a comparison plan (including takes with
no marked issues) skips animation and explains why in the page.

The centered progress panel groups work into Prepare, Analyze, Write feedback,
Create speech, and Animate score. Each stage lists its substeps, with an active
task description. Narration points, voice segments, engraved comparisons, and
video frames report actual counts where available. The thick overall bar counts
resolved milestones (completed or skipped), not elapsed time or an ETA. The
panel scrolls into view once per run and keeps failed tasks visible. Retries
reuse saved work and restart only the relevant task. Legacy stage progress is
retained in the API for older clients.

If feedback detects a mismatched reference MIDI, it automatically regenerates
the reference MIDI and audio with the configured renderer, validates them, and
retries example preparation once. The page remains on progress with a
"Regenerating the reference to match your score" message. Originals are retained
in a `reference-before-rebuild-*` directory. Notated octave chords are validated
without changing the detector's note indices. An unrecoverable rendering or
validation failure still stops the take rather than using an incorrect example.

## Server configuration

Use the existing DataCreate environment with the score renderer and ALIGN model
installed. The studio uses `config/default.yaml` (or the configuration passed to
the server), including its alignment Python, checkpoint, device, and timeout.
It runs the configured GUI alignment pipeline; the current default is experimental stack v9.
Its predicted `candidates.json` is passed to the spoken-feedback service;
the empty human label template is never used as the feedback source.

The browser submits to `/api/studio/takes`; the worker calls the existing
`datacreate.feedback.run_feedback` entry point used by `datacreate-feedback`.
Report preparation, score locations, ssstoken narration, Fish synthesis,
and MP3 assembly all stay in that shared pipeline. Matching performance audio,
reference audio, MusicXML, and reference MIDI are discovered in the generated
sample directory, enabling the pipeline's reference/performance examples.
Speech retries reuse `playback_plan.json` when present, preserving the excerpts;
plain narration retries reuse `feedback.txt`. New takes request a narrated
comparison for every retained label (`all_labels=True`, one LLM call per label).
After MP3 assembly, `datacreate.feedback_video.render_video` creates
`jobs/<id>/video/feedback.mp4` from that same audio, report, playback plan, and
saved note events. The page plays it via `/api/studio/takes/<id>/video`, with
range requests supported for seeking; `/audio` remains the MP3 download.

If animation fails, the MP3 stays playable. **Retry score animation** reuses the
completed feedback files without transcription, alignment, narration, or speech
generation. Partial video attempts are archived. Speech retries also preserve
the label report and synthesized note events needed by the animation renderer.
Existing completed audio-only takes are not automatically regenerated.

Install video dependencies in the same Python environment used to run Studio:
`python -m pip install -e "./DataCreate[video]"`. Restart Studio after upgrading
the server code. Video rendering uses the optional Verovio, resvg-py, Pillow,
and imageio-ffmpeg dependencies; it does not call another external API.

The studio selects the saved `FISH_AUDIO_REFERENCE_ID` voice and enables reference and
performance snippets. They require detected labels with usable locations and
times. If no issues are marked, the page and narration explicitly explain why
no targeted clips are available. New output manifests record the feedback
pipeline revision and selected voice; the config API exposes the loaded revision.
See [the Interpretation Pipeline audit](interpretation_studio_audit.md).

Configure the providers as described in [spoken_feedback.md](spoken_feedback.md).
`ALIGN_FEEDBACK_CONFIG` optionally selects a feedback YAML file; without it,
`DataCreate/config/feedback.fish.yaml` is used: ssstoken narration with
`gpt-6-luna`, and hosted Fish at `https://api.fish.audio` with `drama-3-preview`.
This matches Interpretation Pipeline, including normal Fish speed (1.0) and
pitch-preserving correction of slow English narration toward 140 words/minute.
Music timing is unchanged. Set `FISH_AUDIO_API_KEY` and `FISH_AUDIO_REFERENCE_ID`
alongside `SSSTOKEN_API_KEY`; saved Windows user variables are supported.
`/api/studio/config` exposes the selected provider, mode, model, and pacing
threshold without exposing credentials or the voice ID.

```powershell
conda activate MusicEval
# Save the ssstoken and Fish credentials before starting.
datacreate serve
```

To use local Fish instead, select `DataCreate/config/feedback.local.yaml` through
`ALIGN_FEEDBACK_CONFIG` or the launcher's `-FeedbackConfig` parameter and start
`DataCreate/scripts/start_fish_local.ps1`. Only the LLM key is required in local
mode. The launcher preserves an explicit configuration override; its default is
hosted Fish. Restart the studio after changing process environment variables.
For Qwen Audio 3.1 TTS Flash, set `ALIGN_FEEDBACK_CONFIG` to
`DataCreate/config/feedback.ssstoken.qwen.yaml`, and set `DASHSCOPE_API_KEY` and
`DASHSCOPE_WORKSPACE_ID` for the Beijing workspace. No Fish credentials or local
speech server are needed. See the Qwen setup in [spoken_feedback.md](spoken_feedback.md).
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
