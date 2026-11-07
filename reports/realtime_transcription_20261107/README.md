# Realtime MuScriptor transcription

Implemented 2026-11-07 in the nested `muscriptor` repository.

The new `TranscriptionModel.transcribe_realtime` consumes 16 kHz mono PCM tensors incrementally. It buffers only enough samples to form MuScriptor's existing five-second model window, runs the normal sequential generation with the same tie-prelude forcing, and keeps one decoder state across windows. A short final window is padded using the existing batch rule. It emits `NoteStartEvent`/`NoteEndEvent` immediately after each completed window and a `RealtimeProgressEvent` with the available audio frontier.

The FastAPI server exposes `WebSocket /transcribe/realtime`. The client sends a JSON `start` message, binary little-endian float32 mono frames, then JSON `stop`. The server applies bounded input backpressure (16 frame handoff buffer), rejects non-16 kHz input to prevent resampling differences at chunk boundaries, enforces the existing 15-minute limit, and returns the same raw MIDI payload shape at completion. Five seconds is the minimum model-context delay; inference time adds only if the hardware cannot keep up with capture.

The endpoint does not run beat-grid detection while the stream is open. Clients needing tempo-aware MIDI or sheet corrections can run the existing finalized upload/export path afterward; note events and raw MIDI are available without waiting for that post-processing.

Quality rationale: chunk boundaries, 5-second padding, sequential order, prelude forcing, and `OpenNoteTracker` semantics are shared with finalized inference. The model-level test verifies the first full window uses a following-boundary cutoff and the final partial window uses an end-of-stream cutoff. The WebSocket test verifies streaming events, MIDI completion, instrument forwarding, and prelude forcing.

Validation: `259 passed, 24 skipped` in the full MuScriptor suite. No real model weights were present in the local checkout, so an end-to-end GPU latency/bit-for-bit benchmark remains to be run in the deployment environment.
