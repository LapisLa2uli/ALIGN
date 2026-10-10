# Stationary broadband noise: offline investigation

Implementation follow-up: the authorized gate is now prepared in
`src/datacreate/input_quality.py`, called by Studio before any pipeline work.
It was deployed after all live jobs became terminal and gateway admissions were
paused. Live keyed probe `1167b3c61a1d480fa7ef4d977acc7464` was rejected at
`input_quality`, with no sample/feedback artifacts or provider work. Lookup and
same-key replay returned the same ID; retry returned 409 and media returned 404.
The initial investigation below remains as historical rationale. The shipped rule also
examines an overlapping final full second to protect fractional endings.

The production function was validated offline against the faulty GPIO6 capture
(`rejected`), healthy GPIO7 capture `63fd5d0b87cf4a54af2c6c2d520ccfda` (`passed`),
and 12 original music controls (`passed`). GPIO7: 50.04 s at 48 kHz, RMS 35.772,
RMS spread 17.537 dB, broadband flatness p10 0.01490, high-frequency fraction
p10 0.00622, band flatness p10 0.03500, tonal peak ratio 4759.78. The low level
does not cause rejection. Results are saved in
`work/noise-quality-gate-validation/results.json`; reproduce with
`work/validate_noise_quality_gate.py`. No additional provider requests were made.

Tests cover stationary noise, quiet/sustained/harmonic/brief tones, fractional
endings, changing noise levels, silence, short/lower-rate inputs, malformed WAVs,
pre-pipeline rejection, busy-slot release, same-key replay/restart, forbidden
retry, and suppression of stale media. One healthy capture is not broad hardware
qualification: retain the limitations below and collect more real captures.

Scope: read-only inspection of Studio job `3cd2e206c80947339c130cd91ab1ddfd`
and offline signal measurements. No live gate, restart, cancellation, retry or
paid evaluation was performed. The running service is unchanged.

## Job findings

The job is progressing, not stuck at the observed writing stage. Live substep
counts advanced from 11/35 through 21/35 to 26/35 completed narration points.
The first 13 response artifacts were saved roughly 11–19 seconds apart. Studio
runs `all_labels=True`; `run_feedback` makes one sequential narration request per
retained label. Thirty-five requests at the observed rate imply roughly 8–10
minutes for writing alone, before example synthesis, speech and video. This is
an estimate, not a promise about the provider or remaining duration.

The backend assessment is `ok`, match_fraction 0.636, 35 retained labels,
2 withheld labels and 231 transcribed notes, against a 49-note supplied excerpt.
These alignment statistics do not establish that the original input was music.
The failure here is acceptance of nonperformance audio, not a detected provider
failure. Do not deliver the resulting advice as a trustworthy evaluation of the
noise take. No active work was altered during investigation.

## Reproducible measurements

Run `DataCreate/work/investigate_noise_quality.py` in MusicEval. It reads the
supplied MUSEBOT WAV, the prior hardware fixture, original RawData/Audio 001–012,
and synthetic controls. It writes only local metrics, never calls Studio or a
provider. Results: `DataCreate/work/noise-quality-investigation/metrics.json`.
The fixture duplicates sample001 and is not an independent music control.

Measurements use nonoverlapping one-second blocks. Each block's PSD uses SciPy
Welch, Hann windows, 2048 samples, 1024 overlap, default constant detrending.
Whole-recording RMS is diagnostic only, not a low-volume rejection rule.

| Metric | Faulty GPIO6 recording |
| --- | ---: |
| Duration / rate | 25.460 s / 48 kHz |
| RMS in PCM16 units | 278.229 |
| RMS spread, 20 log10(p95/p05) | 0.0815 dB |
| Flatness above 80 Hz, p10 across blocks | 0.9651 |
| Fraction of power at/above 10 kHz, p10 | 0.5282 |
| Flatness within 300–8000 Hz, p10 | 0.9873 |
| Maximum 300–8000 Hz PSD peak/median ratio | 1.6678 |

## Candidate conservative rule (NOT deployed)

All conditions must hold:

- Original sample rate at least 32 kHz, at least five full seconds.
- Nonzero signal (RMS above 1 PCM16 unit; near-silence needs separate treatment).
- Per-second RMS p95/p05 spread below 1 dB.
- p10 spectral flatness above 80 Hz greater than 0.90.
- p10 power fraction at/above 10 kHz greater than 0.40.
- p10 flatness in 300–8000 Hz greater than 0.95.
- No one-second block with a 300–8000 Hz PSD peak/median ratio of 5 or more.

The last condition conservatively spares takes with detected tonal evidence.
This rule flags the supplied hardware noise and synthetic stationary white noise.
It does not flag any of the 12 native-48kHz music controls, a very quiet sustained
440 Hz tone (RMS 6.95), a harmonic tone, or the amplitude-modulated-noise control.
It also spares the actual noise mixed with a 440 Hz tone at -20, -10, 0 and +10 dB
signal-to-noise ratios. At -30 dB it still flags the mixture: absence of a
reliably detected tone cannot prove absence of an instrument. Silence does not
pass this particular detector and should not be reported as valid music.

These are exploratory checks, not a calibrated false-positive rate. Only one
faulty microphone capture was available; no healthy GPIO7 microphone captures
were included. The 12 music controls are compressed source recordings and may
have less high-frequency sensor noise than a real raw microphone. Stationary
breath/noise techniques, very weak music under hiss, other microphones, room
noise and gain settings need held-out testing. The prototype ignores the last
fractional second; a deployed version needs a conservative treatment of that
boundary. Short or lower-rate inputs are unassessed by this rule, not proven good.

## Recommended integration

1. Run the check on the finalized **original 48 kHz WAV before resampling**.
   22.05 kHz pipeline audio removes much of the identifying high-frequency noise.
2. Keep it before transcription, alignment, and any provider calls. For keyed
   Studio submissions, the cleanest job flow is after durable acceptance, at the
   beginning of `StudioJobs.process`, before preparing the score/reference.
3. Persist a distinct `input_quality` result with the detector version, metrics
   and reason `stationary_broadband_noise`. On rejection use a failed input-stage
   job with `can_retry:false`; do not report alignment `ok`, zero labels, or
   reuse another job's preview. Same-key lookup/replay must still return that job.
4. User message: “This recording contains mostly steady broadband noise. Check
   the microphone connection and record again.” Do not diagnose a particular
   GPIO or microphone solely from audio. A new user recording gets a new key;
   retries of the same bytes should not invoke inference or providers.
5. Begin with diagnostics/opt-in for the hardware profile, test healthy GPIO7
   quiet/loud sustained notes, rests and ambient noise, then enable the strict
   rule for that profile. A gateway preflight can avoid upload too, but do not
   independently tune duplicate rules in ALIGN and MUSEBOT.

A separate policy could cap/prioritize narration points for cost and latency,
but it would change the existing all-label feedback requirement. It is not a
substitute for rejecting unusable input and was not changed in this investigation.
