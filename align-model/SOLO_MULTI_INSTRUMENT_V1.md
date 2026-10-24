# Solo multi-instrument transcriber

This opt-in model extends the repository's dual-resolution mel CTC transcriber
to sounding MIDI 28–108. It predicts pitches without an instrument ID or a score.
The original v9/v10 checkpoints and default configurations remain available.

## Data and evaluation protocol

`synthpipeline.solo` reuses the existing procedural score generator and cached
SoundFont renderer. The configured corpus has 500 clips for each of 24 solo
instruments: flute, piccolo, oboe, English horn, bassoon, clarinet, soprano/alto/
tenor/baritone saxophone, trumpet, muted trumpet, French horn, trombone, tuba,
violin, viola, cello, contrabass, nylon/steel guitar, electric bass, recorder,
and pan flute. Every MIDI gate is monophonic; acoustic release tails can overlap.

The corpus uses MuseScore MS Basic and FluidR3 GM. FluidR3's source and MIT
license are preserved in `E:/ALIGN_Data/soundfonts/FluidR3`. Each generated sample
includes concert-pitch MusicXML, exact sounding MIDI, WAV, per-attack targets,
and provenance. This is a transcription corpus, not a planted-error dataset.

Train/validation/test partitions are 80/10/10, grouped by melody seed across
every instrument. Random evaluation samples use a recorded seed. Rehearsal uses
only existing train partitions from realistic, fast-note, and DataCreate-like
synthetic clarinet datasets; DataCreate recordings are never training data.
Labels from DataCreate 095 onwards are never treated as truth.

Primary transcription metric: one-to-one ordered pitch-note F1, counting every
repeated attack separately. Fast-note, repeated-note, and single-note recall
are reported independently. Onset/pitch F1 within 50 ms is a secondary diagnostic.
These transcription metrics are not the final error-label accuracy metric.
Real-take feedback is evaluated on canonical affected reference-note identities.

Audio QA checks clipping, silence, finite samples, and harmonic agreement with
the intended MIDI pitch. Audition WAVs and their source IDs are saved. These
checks do not establish perceptual realism or accuracy on real instruments.

The completed corpus contains **12,000 clips, 147,474 notes, and 17.28 hours**.
All clips pass the full audibility/finite/clipping audit. Bank-range probes
identified silent upper-key regions in FluidR3 contrabass, and full rendering
also caught silent repeated-note windows. In total, 91 clips use the alternate
bank while retaining their original melody, score, and split. Both banks remain
represented; actual bank assignments and fallback reasons are in each sample's
metadata. The generator now checks both bank coverage and rendered audibility.
No perceptual listening claim is made: this environment provides no reliable
listening tool. The audition montage is available for human review.

A separate **4,608-clip / 3.04-hour training-only chromatic supplement** covers
every configured pitch with sustained notes, repeated attacks, and semitone
contrasts. It uses independent seeds and leaves the musical validation/test
splits unchanged. This addresses sparse and absent pitch classes found in the
initial musical training distribution. Core corpus plus training supplement:
**16,608 clips / 20.32 hours**. Its configuration is
`synth-pipeline/config/solo_chromatic_support_v1.yaml`.
The independent register validation set adds 288 clips with new seeds and
rhythms, including low/mid/high pitches. It is never training data. Total newly
generated audio including this diagnostic: **16,896 clips / 20.51 hours**.

## Training

The old encoder weights are retained. Its pitch/CTC classes are shifted from
written B-flat clarinet to sounding pitch, and the heads are extended at both
ends. The short-window branch remains unchanged for fast attacks. Experiments
compare plain transfer training with gain/EQ/noise/band augmentation, clarinet
rehearsal, and teacher consistency on clarinet rehearsal examples only.
No old teacher predictions supervise other instruments or out-of-range notes.
The third experiment adds the chromatic supplement while retaining the musical
corpus, augmentation, teacher consistency, and clarinet rehearsal. Candidates
are compared on the same full validation population before selection.

Results, source hashes, validation selection, and final test metrics are stored
under `align-model/runs/solo-v1`. The validation winner is selected before test
evaluation. No synthetic result alone warrants a claim to support every real
instrument or playing technique.

The frozen candidate is `runs/solo-v1/solo-transcriber.pt`, selected in
`runs/solo-v1/CANDIDATE_SOLO_V1.json`. It combines augmentation/rehearsal,
chromatic refinement, and a validation-selected +1.5 logit adjustment to
extended pitch classes (below sounding MIDI 50 or above 98). The adjustment
is already in the saved weights; do not apply it again. Blank scale stays 0.5.
Compared with the augmentation-only candidate, musical validation macro F1
changes from 0.92973 to 0.92858, register validation from 0.81801 to 0.85023,
and single-passage F1 from 0.92199 to 0.97015. Fast-passage F1 decreases slightly
from 0.93529 to 0.93214. This is an explicit tradeoff for wider register and
isolated-note coverage. The untouched musical test is reported separately in
[`runs/solo-v1/RESULTS.md`](runs/solo-v1/RESULTS.md).

Final evaluation uses **1,200 held-out clips (50 per instrument)**. Mean
instrument note F1 improves from **0.6133 to 0.9171**; fast-passage F1 from
0.598 to 0.856, repeated-passage F1 from 0.620 to 0.964, and single-passage F1
from 0.641 to 1.000. Every instrument improves. French horn (0.800), contrabass
(0.814), and tuba (0.817) remain the weakest musical-test instruments.
The original/new paired bootstrap improvement interval is +0.276 to +0.334.

The paired real-clarinet **error-label** evaluation does not justify replacing
v10: on 25 eligible human takes, wrong/missed-note F1 is zero for the shared
model, extra-note F1 is 0.074 versus v10's 0.093, and repetition F1 is zero for
both. These are strict affected-reference-note scores, not transcription F1.
The candidate remains opt-in; broader real-instrument recordings and
multi-instrument error-gate calibration are still needed.

The real DataCreate gold annotations are recovered read-only from
`DataCreate/archives/user-labels-before-agent-seed-20261006T074109Z-763e1b7d` when
the live label files contain copied agent predictions. Archive checksums and the
corresponding score/audio hashes must match. Only explicit manual labels and an
explicitly human-saved empty annotation qualify; 25 takes qualify in this corpus.
Samples 007 and 095 are additional inspection cases, not human-labeled gold.

Experiment entry points (run inside MusicEval):

```powershell
python align-model/scripts/run_solo_campaign.py
python -m synthpipeline.solo --config synth-pipeline/config/solo_chromatic_support_v1.yaml --extra-bank E:/ALIGN_Data/soundfonts/FluidR3/FluidR3_GM.sf2 --workers 4
python align-model/scripts/solo_experiment.py prepare --data E:/ALIGN_Data/solo_chromatic_support_v1 --output E:/ALIGN_Data/solo_chromatic_cache_v1 --device cpu
python align-model/scripts/run_solo_refinement.py
```

The refinement runner waits for the main training run and supplementary cache.
It ends after full validation; `freeze_solo_candidate.py` records the reviewed
choice and hashes before `solo_experiment.py eval --split test` is run.
`evaluate_solo_datacreate.py` performs the paired score-note feedback comparison.
`report_solo_results.py` builds the final report and per-instrument plot.

## Run

Use the MusicEval environment for every command. From the repository root:

```powershell
conda activate MusicEval
$env:PYTHONNOUSERSITE = '1'
python -m synthpipeline.solo --config synth-pipeline/config/solo_multinstrument_v1.yaml --extra-bank E:/ALIGN_Data/soundfonts/FluidR3/FluidR3_GM.sf2 --workers 4
python align-model/scripts/solo_experiment.py prepare --data E:/ALIGN_Data/solo_multinstrument_v1 --output E:/ALIGN_Data/solo_multi_cache_v1 --rehearsal
python align-model/scripts/run_solo_transcriber.py --audio performance.wav --output transcription.json
```

To align to a score, add `--score reference.musicxml`. MusicXML `<transpose>`
metadata determines sounding versus written pitch. XML without this metadata
is interpreted as concert pitch. For transposing notation that omits metadata,
pass `--written-to-sounding -2` for B-flat clarinet/trumpet, `-9` for E-flat alto
sax, `-14` for tenor sax, etc. This is a notation conversion, not an automatic
instrument classifier. The score-note indices remain those of the original XML.

The DataCreate opt-in server is:

```powershell
cd DataCreate
datacreate --config config/solo.yaml serve --port 8769
```

An optional `alignment.written_to_sounding` override in that YAML also participates
in cache validity, so changing it invalidates saved model feedback. Rendering,
score-aware same-pitch repair, and error-confidence calibration outside clarinet
have different generalization requirements. The solo configuration currently
uses learned CTC boundaries directly and the existing bounded score aligner;
its inherited error gates are explicitly marked experimental outside clarinet.

For the existing DataCreate clarinet recordings, add `written_to_sounding: -2`
under `alignment` in the solo YAML: those historical XML files lack transposition
metadata. Keep that override absent for general uploads that contain correct
MusicXML transpose metadata or use concert-pitch notation. Inference runs in a
child process; importing the UI does not load PyTorch or retain the model there.

Frozen-candidate checks can be reproduced without editing live samples:

```powershell
python align-model/scripts/smoke_solo_candidate.py --sample align-model/runs/solo-v1/smoke-trial/sample --output align-model/runs/solo-v1/smoke-new
```
