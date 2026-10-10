# ALIGN Studio robot integration contract (studio-robot-v1)

This contract connects a trusted gateway running on the same PC to ALIGN Studio.
The supported deployment is **one Studio process / one worker**, bound to
`127.0.0.1:8765` by `datacreate serve`. It is not an authenticated LAN API.
Do not change the Studio bind address to expose it to ESP32 devices. Device
transport, access control, buffering, speech recognition and media conversion
belong to the gateway. Keep the Studio work directory on a persistent local disk.

## Recording and submission boundary

Xiaozhi recognizes `enter music training mode` and `upload audio`. The controller
can stream mono PCM16 little-endian, 48,000 Hz to the PC during recording, but
ALIGN receives **no request that starts evaluation until `upload audio` is
recognized**. The gateway finalizes a RIFF/WAVE file and submits it then. Studio
does not implement streaming capture or PC voice recognition. Head tracking
remains the device/gateway's responsibility.

- Recommended input: mono PCM16 WAV, 48 kHz, 1–300 seconds (96,000 PCM bytes/sec).
- Server accepts mono, uncompressed 16-bit WAV at 8–96 kHz. The validator permits
  up to 301 seconds for existing GUI rounding; clients must cap at 300 seconds.
- File limit: 60,000,000 bytes for audio; uploaded score limit: 10,000,000 bytes.
- Use multipart `audio` (required) plus `score_id` OR uploaded `score` (.musicxml,
  .xml, .mxl). Uploaded score takes precedence if both are supplied.
- The HTTP API does not decode MP3/M4A; the GUI decodes these to WAV in the browser.

## Discovery and default score

`GET /api/studio/config` provides `ready`, `message`, `max_seconds`, `scores`
(`id`, `name`), `feedback_pipeline_revision`, `integration_contract`, and
`default_score_id`. `integration_contract` is `studio-robot-v1` for this API.
`default_score_id` is the sole score's ID when exactly one is available, otherwise
null. Prefer an explicitly configured gateway score ID and verify it is listed;
otherwise use this default. When null, require score selection/configuration or
upload a matching score. Never silently choose the alphabetically first score.
POST always requires an explicit score ID or file; discovery does not implicitly
select it. Keyed submissions copy a selected score into the accepted job so that
later changes to the score library cannot alter that job.

## Durable idempotency and crash reconciliation

Before submission the gateway must persist a random, globally unique key per
recording (e.g. a UUID), the exact WAV, and its selected/uploaded score identity.
Do not use secrets as keys. Keys may appear in local HTTP access logs.

```http
POST /api/studio/takes
Idempotency-Key: musebot-550e8400-e29b-41d4-a716-446655440000
Content-Type: multipart/form-data; boundary=...
```

Keys are case-sensitive, 1–128 ASCII letters/digits or `._:-`. The header is
optional: existing GUI submissions without it retain their original behavior.
Idempotency does not apply retroactively to unkeyed jobs.

New acceptance returns **202**:

```json
{"id":"<32 lowercase hex characters>","status_url":"/api/studio/takes/<id>","replayed":false}
```

An identical accepted request returns **200** with the same ID/URL and
`replayed:true`. No processing is scheduled, including for completed, failed,
interrupted, or currently processing jobs. Replays work while another take is
busy and even if provider credentials are temporarily unavailable.

Identity uses SHA-256 of exact WAV file bytes, exact score file bytes, and the
score extension (.xml and .musicxml are distinct formats for this purpose).
Multipart boundaries, audio filename and score filename stem do not matter.
Re-encoding a WAV, changing its metadata/header, or changing a selected score on
disk changes identity. Keep originals for replay. A selected score and an uploaded
copy with identical bytes and extension have the same identity.

- **409**, `detail.code = idempotency_conflict`: key already accepted with different
  content. Do not retry under a new key automatically; reconcile the saved session.
- **409**, `detail.code = studio_busy`, `Retry-After: 2`: a different take is running.
  Key was not consumed; wait and retry the same key and data. There is no queue.
- **422** invalid key/audio/score, **413** oversized upload, **503** server not ready:
  no new acceptance record and no inference/provider calls.

After a timeout, disconnect, or gateway restart, call:

```http
GET /api/studio/requests/musebot-550e8400-e29b-41d4-a716-446655440000
```

- **200** returns the same receipt with `replayed:true`; poll `status_url`.
- **404** means no acceptance record yet. The original request might still be
  uploading; retry POST with the **same key and exact data**, never a fresh key.
- **410** means the key was accepted but its job status was removed. It remains
  reserved; it will not trigger another evaluation. Require operator intervention.

Receipts are stored transactionally in `work/studio/requests.sqlite3` before
background processing is scheduled. Key hashes are stored instead of raw keys.
Inputs and initial status are saved before the receipt is committed. A crash
before commit cannot have started processing. A crash after commit but before
processing leaves a recoverable receipt pointing to an interrupted job. Incoming
upload directories or unindexed job directories can remain after a hard crash;
they are never automatically scheduled. No key expiration or automatic cleanup
is implemented. Preserve the database AND job directories when moving/backing
up Studio. Deleting the database discards deduplication history.

This prevents duplicate job creation, not exactly-once billing at external
providers. A provider may have completed a request before a network failure;
explicit feedback retries can repeat provider work. Never automatically invoke
`/retry` merely because the POST acceptance response was lost.

## Progress, failures and retry

Poll `GET /api/studio/takes/{id}` roughly every two seconds until `status` is
`complete` or `failed`. An accepted job starts as `processing`, stage `queued`.
Use `detailed_progress` (phases, substeps, completed/total milestones, message,
optional units) for display. Milestones are not elapsed-time percentages or ETAs.
`analysis_progress` is retained for older clients. Ignore unknown additive fields.

After a Studio restart an unfinished job reports `failed`, with a restart message.
Lookup/replay still returns its original ID; neither resumes processing. If
`can_retry:true`, `POST /api/studio/takes/{id}/retry` explicitly resumes saved
feedback (or video when `retry_kind:video`). This endpoint is not covered by the
creation idempotency key. Reconcile its timeout by polling that job, never by
submitting a new take. Provider costs may recur on explicit retries. If
`can_retry:false`, the gateway must report the issue and await an operator/new
user submission; it must not silently create another job.

## Assessment and job-specific media

### Original-audio quality gate

New Studio takes (GUI and keyed gateway submissions) first run the
`stationary-broadband-v1` check on the accepted original WAV, before score
rendering, resampling, transcription, alignment or provider calls. A short
`stage:input_quality` preflight has its own progress display; normal pipeline
milestones follow afterward. This change needs a server restart to become active.

The job persists `input_quality` with `revision`, `status`, `reason`,
`sample_rate`, `duration_seconds`, `thresholds`, `metrics` and `message`:

- `rejected`, reason `stationary_broadband_noise`: terminal `status:failed`,
  `stage:input_quality`, `can_retry:false`, `assessment:null`; no feedback media
  or prior-run preview is exposed. Show the message asking the user to check the
  microphone and record again. A freshly recorded performance uses a new key.
- `passed`, reason `noise_pattern_not_detected`: this particular noise signature
  was not detected; it does **not** establish music, a working microphone, or
  successful alignment. Processing continues normally.
- `not_assessed`: original sample rate below 32 kHz or duration below five seconds.
  Existing processing continues; this is not a quality certification.

Acceptance still returns 202 before this asynchronous check. Repeating an
accepted key or looking it up returns the original job even when quality failed;
it never starts another check, inference or provider request. The same-byte
`/retry` is rejected. Old jobs are not retroactively reclassified, and existing
speech/video retries for pre-gate jobs retain their original behavior.

The detector requires stationary RMS, high spectral flatness, substantial power
above 10 kHz, and absence of strong tonal peaks, all together. It does not reject
merely for low volume. A final overlapping second checks the fractional ending
and vetoes rejection if it has changed level or tonal content. Silence and very
weak music hidden by noise are not reliably distinguished by this narrow gate;
do not describe its result as a hardware diagnosis. Implementation/validation:
`docs/input_quality_investigation.md`.

`assessment.status != ok` represents uncertain/unreliable alignment. A failed
alignment is **not** a zero-error result. Show `assessment.message` or `message`,
do not announce successful evaluation or play unrelated feedback. Some older
runs have `assessment:null`; do not infer a positive assessment from that alone.

A reliable analysis with zero retained labels may complete with **audio only**:
`feedback_details.label_count == 0`, `no_issues_marked:true`, `audio_url` present,
`video_status:unavailable`, and no `video_url`. Zero labels do not establish an
error-free performance; preserve the narration's uncertainty wording. Video may
also be unavailable when no comparison playback plan exists.

Only use `audio_url` and `video_url` returned for the requested job. Audio is MP3
(`audio/mpeg`); video is MP4 (`video/mp4`). Verify relative media URLs identify the
same `/api/studio/takes/{id}/` before fetching. A 404 means that media is not ready
or unavailable. Audio can be available during video generation; successful video
completion is indicated by `status:complete` plus `video_url`.

**Ignore `preview_video_url` and `preview_video_source_id` for robot playback**:
these are GUI-only conveniences for a different prior run, even if inputs match.
Never pair that video with the current job's narration or announce it as current.
The gateway may create a clearly labeled narrated-summary video using only this
job's MP3 when no MP4 exists, but must not claim it is an annotated score video.

No hardware profile, MJPEG/PCM transcoder, stream transport or ESP32 H.264 decoder
assumption is introduced here. MUSEBOT owns conversion to the actual screen and
speaker formats (1024x600 screen, no SD). Studio's source media stays unchanged.

## Deployment and verification

Reload Python changes by restarting Studio only when no job is active. Do not
run multiple workers or two Studio instances against the same work directory:
the processing busy slot is process-local, as in the original GUI. SQLite
serializes keyed acceptance, but does not make inference scheduling distributed.
Uploads are validated before busy/replay decisions, so reconnect via lookup to
avoid repeatedly transferring the WAV. Error bodies may be strings or structured
`detail` objects; branch on status and the documented code, not English text.

Regression coverage in `tests/test_studio.py` checks deduplication, payload
conflicts, score snapshots, concurrent submissions, restart before the worker,
busy behavior, unavailable credentials, removed artifacts and storage failure,
as well as the existing GUI, assessment, feedback and media behavior. These tests
use local pipeline/provider doubles and make no paid API requests.
