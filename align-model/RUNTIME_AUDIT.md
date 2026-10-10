# Alignment runtime audit — 2026-10-05

The current `v9-passage-v1` pipeline spends more time starting a fresh Python
process and preparing score metadata than running numerical alignment on the
three measured inputs. Optimize reuse and template construction before reducing
the search space or moving the DP to a GPU.

## Measurements

MusicEval Python, existing CUDA configuration, unchanged candidate/checkpoint.
The real DataCreate publisher ran against isolated input copies. Timings use
lightweight function wrappers; no cProfile overhead. First-call results include
import/model/library initialization with existing on-disk caches, not a deliberately
empty-cache benchmark. One verified fresh run and one warm alignment per input;
these are observations, not latency percentiles or guaranteed speedups.

| Input | Full publisher incl. imports | Imports | Detailed aligner | Warm passage alignment + gates |
|---|---:|---:|---:|---:|
| 007: 238 score notes | 8.92 s | 5.78 s | 0.68 s | 0.57 s |
| 095: 63 score notes | 7.63 s | 5.33 s | 0.18 s | 0.19 s |
| Studio full score: 3,934 notes | 14.93 s | 5.30 s | 0.86 s | 0.68 s |

The warm call reuses decoded notes, canonical index, ornament cache and loaded
verifier; it excludes audio transcription, initial score parsing and UI export.
It is not an estimate of complete warm request latency. The full alignment
result, events, deletion identities and diagnostics were identical on all three
warm comparisons. Original audio, MusicXML and human-label hashes were unchanged.

Reports: `DataCreate/work/runtime-audit-20261005/007-verified/timings.json`,
`095-verified/timings.json`, `full-score/timings.json`. Preliminary `007` and `095`
directories are excluded: the initial audit compared diagnostics after the caller
mutated them with transcription fields. The profiler now snapshots before that
mutation; the verified runs above pass the exact comparison.

## Where the time goes

1. **A new subprocess for every realignment.** `align_bridge._run_v9_alignment`
   starts the publisher each time. Imports cost 5.3–5.8 seconds here, 35–70% of
   the measured total. Library initialization and loading models add further
   overhead. The aligner's DP is CPU/Numba code; the CUDA option principally
   accelerates acoustic models. Measured transcription inference was 0.17–0.18 s;
   mel extraction was 0.44–0.55 s. Switching DP hardware is not the first priority.

2. **Repeated whole-score preparation.** The full-score run calls music21 parsing
   three times, totaling 2.71 s inside parse calls alone. Canonical indexing,
   ornament patterns and the GUI's sounding-note adapter each traverse score
   structures. Ornament preparation takes 2.40 s, and GUI conversion 1.59 s.
   These are inclusive timings and overlap the parsing total: do not add them.
   The source score is still prepared in full even though only 172 canonical
   notes enter detailed alignment. Passage retrieval itself takes only 0.016 s.

3. **Template objects are constructed before the length rejection.**
   `align_v5` already computes `template_size`, but expands the full Python
   dataclass template before checking its length against the transcription.
   On the full score it expands 852 hypotheses but runs DP on only 362; 490
   expansions are rejected afterward. For 007 the figures are 379 versus 256.
   Template expansion takes 0.53 s on the full score versus 0.13 s in untimed DP;
   for 007 they take 0.27 s and 0.26 s respectively.

4. **Full DP matrices for every hypothesis.** Each untimed candidate allocates
   two cost layers plus backtrace and predecessor-layer tables across the entire
   performance/template grid. The first search performs 19.38 million grid cells
   for 007, 3.69 million for 095 and 12.16 million for the full-score passage,
   each with two state layers. Only the winning candidate's backtrace is used.
   Search complexity is approximately O(H × N × M) per passage, with H repeat
   hypotheses, N transcribed candidates and M expanded template units. Multiple
   candidate passages multiply this work. The measured inputs each use one
   passage; these timings do not characterize the configured worst case.

5. **Smaller costs.** Publisher hash calls cost 0.24–0.31 s. Local repeat proposal
   generation and the final timing-aware DP were minor in these samples.
   Presence verification is already batched within each query call; repeated
   small calls may be consolidated, but this is lower priority.

Studio additionally renders reference audio before alignment, builds display
features afterward and may generate spoken feedback. Those stages are outside
this publisher benchmark. They should receive separate timing fields so their
latency is not attributed to note alignment.

## Recommended implementation order

### 1. Reuse one bounded inference worker

Load Python dependencies, both neural models and Numba kernels once. Queue jobs
through the worker; keep one model instance instead of starting multiple heavy
processes. Preserve the same inference mode, numerical precision and settings.
Offer explicit cache invalidation/reload when the checkpoint or runtime changes.
This targets the largest repeated cost, without changing alignment decisions.
It retains a resident memory footprint between requests; bound caches and avoid
duplicating workers. Measure complete warm requests before promising a speedup.

### 2. Cache score preparation by content

Use a score bundle containing canonical events, ornament patterns and validated
GUI note metadata. Key it by MusicXML content hash and parser/runtime revision,
not by path or sample ID. Preserve the canonical/UI identity consistency check
on bundle creation. Reuse it for subsequent takes and pass the parsed structure
between first-request stages where practical. Avoid a large unbounded cache of
music21 trees; compact immutable arrays/metadata are preferable.

### 3. Move existing rejection before template expansion

Use the already computed exact expanded length for the existing length test,
before creating template units. Preserve hypothesis order, strict/relaxed passes,
all accepted hypotheses and tie-breaking. Then build numeric pitch/link/merge
arrays for scoring, materializing rich template objects only for the winner.
Prefix sums of per-note expansion sizes can also avoid recounting repeated spans.
These changes remove overhead without narrowing the search.

### 4. Score hypotheses with rolling DP rows; backtrace only the winner

The recurrence needs the previous row and the current row's left neighbor,
including the same-row merge transition. Use identical floating-point costs and
operation order to find the best hypothesis with O(M) working storage, then run
the existing backtrace DP once for that winner, followed by the current timed
refinement. Preserve the existing first-wins tie rule. This adds one DP run for
the winner while removing full-matrix allocation/writes for all losing candidates.
It should be benchmarked; its end-to-end benefit is smaller than worker reuse on
these inputs. Mathematical equivalence still requires differential testing.

### 5. Reuse acoustic results for unchanged inputs

For reruns on the same recording, cache the exact probability arrays used by the
pipeline, decoded/repaired candidates and any evidence needed by the gates.
Key by audio hash, checkpoint, frontend, decoder, repair configuration and runtime
revision. Preserve the existing float16-to-float32 round trip. Score-only changes
should not require another transcription. Existing Re-label already reuses valid
final feedback; retain its provenance checks instead of bypassing them.

## Changes that do not meet the no-regression requirement by themselves

Do not simply reduce hypothesis/passage limits, drop weak short-note candidates,
disable alternative pitches, remove the verifier or timing pass, or use a narrow
fixed DTW band. Those can lose fast notes, repetitions and displaced alignments.
Beam search/heuristic early exits change search behavior. Exact lower-bound
pruning is possible only with a proof accounting for free optional drops,
ornaments, alternative pitches and merge operations; simple length-based bounds
can be invalid for this recurrence.

Acceptance should compare old/new canonical score spans, error types, deletion
sets, repetition ranges/counts, uncertainty decisions and deterministic ties,
alongside output playback fields and costs. Use fast 007, repetition 095, the
full-score take, ambiguous passages, ornaments and synthetic cases, then the
broader corpus. DataCreate >=095 can check output equivalence, not human-label
accuracy. Preserve frozen v9 sources by introducing versioned optimized modules.

## Reproduce

```powershell
& 'C:\Users\Hank\.conda\envs\MusicEval\python.exe' `
  align-model/scripts/profile_alignment_runtime.py `
  --sample DataCreate/samples/007 --output DataCreate/work/new-runtime-audit-007
```

Only the audit script and this report were added; production inference was not
changed by this analysis.
