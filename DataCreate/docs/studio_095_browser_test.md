# Studio browser verification with sample 095 — 2026-10-05

Used the actual `/studio` page, uploaded
`DataCreate/samples/095/verified_score.musicxml` and `performance_audio.wav`
through its file choosers, and clicked **Get my feedback**. The browser decoded
the recording to mono 48 kHz WAV, then the normal pipeline ran fresh v9 inference.
The human labels were inspected only for evaluation; they were never submitted
as predictions or used to tune the detector. External narration requests used
ssstoken with the user's explicit permission; Fish speech remained local.

## Reproduced and fixed

- The sandboxed test process could not open an external socket to ssstoken
  (Windows error 10013). Restarting the test server with approved network access
  let **Retry spoken feedback** recover the saved analysis without rerunning it.
- The v9 publisher did not write progress milestones. It now atomically reports
  transcription, alignment, and label generation to Studio.
- Provider failures previously showed a generic instruction to check the log.
  Studio now gives actionable, sanitized messages for network, credential,
  malformed-plan, and speech failures.
- The launcher could hang during model imports with an unwritable Numba profile
  cache. It now defaults to a workspace cache and restores the caller's previous
  environment on exit.
- Mel preview rendering attempted to start a Matplotlib GUI in the job thread.
  It now uses a file-only Agg canvas.
- The real LLM returned a 123-word plan despite the 100-word prompt limit, and
  another plan repeated the same advice for neighboring wrong notes in bar 12.
  Plan validation now retains one example per issue type/bar range and removes
  whole lower-priority points until the English narration fits 100 words. It
  preserves complete sentences and reference/performance clip pairs, without an
  extra paid request. A single overlong point is rejected before synthesis.

## Results

The browser-created recovery test `083ba51317a34802953a60fdf7b22214` completed
with four model labels and a playable 63.111-second MP3 containing three paired
examples. The browser player advanced normally with `readyState=4` and no media
error. The source clip hashes matched the plan, all samples were finite, peak
amplitude was 0.661, and every assembled segment measured approximately
−20 LUFS. Its script contained 99 spoken words.

Subsequent fresh browser runs verified full processing without retry and exposed
the length/duplicate issues described above. The final run after those fixes is
`96c1b32224bf45cb8940bb22c5bd4f84`; see its `feedback` directory under
`DataCreate/work/studio/jobs/` for the provider response, normalized plan,
transcript, individual speech clips, examples, final MP3, and timeline.

90 focused tests passed across Studio, feedback, audio assembly, and the model
bridge. This includes regressions for safe actionable errors, complete-point
word budgeting, and same-bar duplicate advice.

## Accuracy limits

The model found the annotated opening replay and the replay around bars 15–16.
Its two wrong-note predictions identify score notes in bar 12, but their playback
times (about 31.8 and 32.3 seconds) differ from the human wrong-note annotation
(about 29.5–30.3 seconds). It also missed the human-annotated replay around
31–32.7 seconds. These are model alignment/recall limitations, not an empty-label
frontend bug; the detector, frozen thresholds, and original sample were not
changed to force a pass. Playback and signal checks do not constitute a human
listening assessment of Fish pronunciation.
