# Studio waveform — original source export

This package contains byte-for-byte copies of the current Studio frontend source. The original JavaScript, HTML, and CSS have not been extracted, rewritten, or simplified. manifest.json lists their source paths and SHA-256 hashes.

## Where the waveform lives

- static/studio-audio.js: melFilters() creates 96 triangular mel filters over the FFT power spectrum; melEnergy() produces normalized energy values.
- static/studio.js: startRecording() connects the microphone to a Web Audio AnalyserNode. The ResizeObserver and draw(time) block near the end render the waveform on canvas. Its initial variables are at the beginning of the file.
- static/studio-worklet.js: original PCM recording AudioWorklet. It captures audio; the AnalyserNode and canvas code create the visual.
- static/studio.css: original visual container, canvas sizing, and complete Studio styles.
- templates/studio.html: original page, including canvas#spectrum inside .visual.

## Original signal and rendering behavior

Microphone -> AudioContext (requested 48 kHz) -> AnalyserNode (FFT size 4096, analyser smoothing disabled) -> FFT power -> 96 mel bands from 30 Hz to min(12 kHz, Nyquist) -> normalized energy -> animated canvas bars.

The display smooths energy by 0.22 per animation frame, applies the original decorative envelope, colors, and rounded bars, and resizes for device pixel ratio (capped at 2). The idle animation respects the reduced-motion preference.

This is a decorative display of live mel-frequency energy, not a scrolling spectrogram or a time-domain waveform. In the original implementation, live energy is displayed only while microphone recording is active. Uploaded audio is decoded for submission but does not drive the visualization.

## Integration notes

The source uses native browser JavaScript modules, Web Audio, AudioWorklet, Canvas 2D, and ResizeObserver. There is no frontend build step or visualization library. Use a modern browser and serve over localhost or HTTPS for microphone access; grant permission after pressing the recording button.

The included studio.js is the COMPLETE original Studio controller. It expects the DOM elements in studio.html, serves the recorder from /static/studio-worklet.js, and calls /api/studio/config and the Studio take endpoints. Serve static/ at /static/ and templates/studio.html at /studio in the existing DataCreate application to run the complete page. Opening the HTML directly or serving these files alone will not provide the score/feedback API.

For reuse of only the waveform in another application, use melFilters(), melEnergy(), the analyser setup, canvas state, ResizeObserver, draw(time), and the matching CSS/HTML. Wire the recording state and audio lifecycle to that application's controls. This is an integration guide; the supplied source files remain unchanged.

No recordings, model weights, server configuration, or credentials are included. The waveform itself requires no model or external API.
