# MuScriptor take 002: diagnosis and repair

Date: 2026-11-06. Fixed pipeline revision: `muscriptor-medium-align-v9-v2`.

DataCreate was transcribing a trimmed WAV, while the user submitted the original recording to the MuScriptor website. The local model reproduces the reported problem on the trimmed input and removes it when given the preserved full recording. The failure originates in transcription and causes the aligner to infer a repetition; it is not caused by adding a chunk offset twice or appending the same decoded event twice.

MuScriptor divides audio into five-second chunks. Sequential decoding carries unfinished-note context across chunk boundaries. Starting the input 5.20839 seconds later changes both the chunk boundaries and the preceding context. With this take, the crop produces a spurious phrase at approximately 55.25-57.65 seconds on DataCreate's timeline. The aligner then classifies ten notes in the subsequent real phrase as copies of the earlier phrase.

The original-upload file is preserved at `DataCreate/samples/002/performance_audio_original/002.m4a`. The tested full WAV is `performance_audio_full.wav` (76.308 seconds), and the current crop is `performance_audio.wav` (67.6601 seconds). The crop is exactly the full WAV's samples `[114845:1606750]` at 22050 Hz: start 5.208390022675737 seconds, end 72.86848072562358 seconds, maximum sample difference zero. This provides a controlled comparison without changing the score or audio samples within the selected passage. No raw output from the hosted website was available for a byte-for-byte comparison.

| Input and precision | Transcribed notes | Observation |
| --- | ---: | --- |
| Saved cropped input, CUDA float16 | 127 | Spurious phrase followed by apparent repetition |
| Same crop, CUDA float32 | 127 | Exactly the same note sequence as saved float16 output |
| Preserved full recording, CUDA float32 | 117 | No spurious phrase in the pause |
| Repaired bridge, full recording, CUDA float16 | 117 | Same absence of the spurious phrase; correct crop clock |

All comparisons use MuScriptor medium weights with SHA-256 `ac80adbdf85d87231735fd948af7013441c0afced316c4e9067fd5d8a7fb97ec`, from the local MuScriptor 0.4.0 checkout at `8a1ef88`. The precision control rules out float16 as the cause for this take. The local upstream web server's sequential inference settings were inspected; no audio was uploaded externally during this investigation.

The repair verifies that the current audio is a slice of the preserved full recording, transcribes that full recording, then clips note intervals to the selected range and subtracts its exact sample origin once. All 117 full-recording notes survive this take's projection, with zero clipped or discarded notes. There is no score-driven deletion or heuristic deduplication. Genuine repeated attacks remain intact.

The bridge records full-source and crop provenance, invalidates old cached results, and rejects inputs changed during processing. Repeated trims retain their absolute origin in the original recording. The adapter explicitly uses `batch_size=1` and `prelude_forcing=True`, pairs end events with their originating start events, and validates note identities and intervals. The publisher rejects transcripts carrying stale audio context.

| Alignment outcome | Before | After |
| --- | ---: | ---: |
| Transcribed notes | 127 | 117 |
| Raw match relationships | 113 | 116 |
| Raw copy relationships | 10 | 0 |
| Raw substitute relationships | 2 | 0 |
| Raw extra relationships | 2 | 1 |
| Content feedback labels | 5 | 0 |
| Repetition feedback labels | 1 | 0 |

The five previous content feedback labels were two wrong notes, one extra note, one missed note, and one repetition. The remaining raw extra relationship does not pass the existing content-feedback gates. Fourteen rhythm labels remain from the separately added rhythm classifier; absence of content feedback does not establish that the entire performance is error-free.

Verification completed:

- 58 focused tests passed across DataCreate bridge/transcriber coverage and ALIGN passage, publication, and rhythm coverage, including ten new full-context regression tests. One unrelated Starlette/httpx deprecation warning was emitted.
- Python compilation and `git diff --check` passed.
- The updated local server returned HTTP 200 for `POST /api/samples/002/re-align` with `{"transcriber":"muscriptor"}`. Both GUI servers returned the corrected revision and 117 notes through the note-alignment API.
- Hashes confirm that human `labels.json`, both audio WAVs, `verified_score.musicxml`, and `metadata.json` remain unchanged after regeneration and the HTTP check.

The existing server on port 8765 had the old bridge module loaded. Its Re-align request was correctly rejected by the new publisher because the old transcript lacked full-audio provenance. Automatic approval review blocked restarting that server with the message "blocked by policy". It remains running and can display the regenerated result. An updated server is running at <http://127.0.0.1:8766/>; use it for Re-align until the original server is restarted.

Evidence: `before_alignment.json`, `after_alignment.json`, `verification.json` (protected hashes and initial regeneration), `live_realign.json` (HTTP verification and final protected hashes), `live_realign_stale_server.json`, `002-float32.json`, and `002-float32-full.json`. `reproduce.py` supports the controlled inference comparison. `regenerate_002.py` performs a single-sample repair and captures before/after files; rerunning it overwrites those captures, so preserve this report directory first if the original evidence is needed.

This fixes the demonstrated context mismatch for take 002 and applies the same source-selection policy to future MuScriptor runs. Samples without a verifiable full recording fall back to their current audio and record the reason; the original chunk context cannot be reconstructed reliably from a crop alone. This change does not guarantee perfect MuScriptor transcription or recalibrate ALIGN's feedback thresholds.
