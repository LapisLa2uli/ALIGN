# Studio / Interpretation Pipeline audit — 2026-10-04

Compared the user requests and completed changes in the chat **Interpretation
Pipeline** with the live studio workflow and saved take
`2f6745699b1347da88144d3a23c1ec48`.

## Why this take had no examples

The run used stack v9 and returned `status: ok`, but **zero labels** in
`feedback_v9.json`, `labels_agent.json`, and `candidates.json`. The pipeline log
also says `ALIGN v9 wrote 0 feedback labels`. Its narration report therefore had
`label_count: 0`; the final manifest records both excerpt flags as false.
The detector withheld two potential extra notes. Those withheld candidates are
not accepted errors and must not be narrated as mistakes.

The saved request contains the current teacher-style prompt. This was not the
old 302.AI prompt or an obsolete audio mixer. No timed labels meant no passages
could enter the reference/performance excerpt branch. The LLM nevertheless
returned vague advice about a "marked passage". That wording was misleading.

## Requirements checked

| Request from Interpretation Pipeline | Studio/shared-pipeline behavior |
| --- | --- |
| English music-teacher style, concise, specific correction and practice advice | `feedback.py` teacher prompt; 1–3 labeled examples, complete spoken sentences; deliberately concise, per the chat request |
| ssstoken, lightweight GPT-6 model | `https://api.ssstoken.net/v1`, `gpt-6-luna`, dedicated `SSSTOKEN_API_KEY` |
| Fish 1.5 with natural human reference voice | Reference files are installed as `teacher_lj`; studio default corrected from unspecified voice to `teacher_lj` |
| Whole-score bars, within-bar note positions | `feedback_score.py` and `add_score_locations`; examples replace spoken note ordinals when reference clips are available |
| Reference example → performance example → mistake and improvement | Shared playback plan and `render_playback_plan`; reference MIDI supplies exact reference timing |
| Speak numbers fully; avoid clipping “bar two” | Spoken-number normalization; local Fish wrapper uses `chunk_length=300` for short speech segments |
| Similar speech/music loudness | Shared feasible loudness target, nominally −20 LUFS, with peak headroom |
| Slight reverb and ~0.5-second space around snippets | 8% room reverb, 0.25-second tail, 0.5-second gaps around music snippets |
| Retry speech without repeating LLM generation | Studio replays saved `playback_plan.json` and copies its clips; plain text is used only when no plan exists |

## Fixes and evidence

- Studio configuration explicitly enables examples and selects `teacher_lj`.
- Empty English reports now produce a truthful fixed explanation, without an
  unnecessary LLM call or invented marked passage. No example errors are added.
- The browser explains why no snippets exist, including for older saved reports.
- New manifests record pipeline revision, label count, empty-report reason, and
  reference voice. `/api/studio/config` reports the loaded feedback revision.
- Studio integration tests run the real shared feedback and audio assembly code
  with generated performance/reference fixtures. Only model inference and
  external provider HTTP are mocked. They verify request destinations, voice,
  reference/performance order, loudness, reverb, silence, MP3 delivery and retries.

Existing MP3s are historical artifacts and are not changed by editing code or
configuration. Restart the GUI after Python edits. Fresh detected labels are
required for issue-specific explanations and corresponding audio examples;
this audit does not change model thresholds or fabricate detections.
