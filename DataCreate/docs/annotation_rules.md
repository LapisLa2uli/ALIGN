# DataCreate annotation rules (Weber clarinet takes)

Lessons from human review of **005** and **006**, plus earlier 003/004 work.
F0/onset is the only ear available here. It is wrong often. Prefer fewer labels.

## Instrument and score

- Takes **002–045** are Bb clarinet vs `RawData/Score/WeberITAV.musicxml`.
- Takes **046–093** are new practice snippets (古龙路 54–82 → 046–074, Keyibao 1–18 → 075–092, Wanyuan Road → 093). Some are Weber; some are Mozart K.622.
- Mozart score: `RawData/Score/MozartClConcertoA.musicxml` (Gaudreau, **Clarinette en Si♭**). Written C major sounds concert **Bb** (piece transposed down from A to Bb). Same rule: **concert MIDI = written MIDI − 2**.
- Mozart measure map (this edition, continuous numbering): **mvt 1 Allegro 4/4 mm 1–358** (clarinet entry **57**); **mvt 2 Adagio 3/4 mm 359–465**; **mvt 3 Rondo 6/8 mm 466–813**.
- **Concert MIDI = written MIDI − 2**. Always compare performance F0 to concert pitch.
- **001** is the demo scale (`001.musicxml`), concert = written.
- Time signatures in Weber: **4/4** through m20, **2/4** from m21, **6/8** from m189.
- Annotator `start_beat` is **time-signature beats** (quarters in 2/4), not eighths.
  - User “94:4” on a 2/4 bar means **eighth-beat 4** = written G3 pickup at offset 1.5 ql.
  - Closest integer extract is `start_beat=2` (from quarter-beat 2). Clip leftover cadence notes (e.g. m94 B4) if the take starts on the pickup.
- User “90:2” is valid 2/4: rest + G3 pickup into the **first strain** of Variation 2.

## Do not trust auto measure matching

The batch F0 matcher was often wrong (005/006 were placed on mm 66–82).
Re-identify the excerpt from **opening concert pitches** and **structural pickups**, not from DTW/greedy note alignment.

Known practice windows (2/4 unless noted):

| Window | What |
|--------|------|
| 2:1–20 (4/4) | Introduction |
| 20:4–36 | Theme (pickup G4) |
| 45:1–61 | Variation 1 |
| 66:1–82 | Fast arpeggio variation (not var. 2) |
| **90:2–106** | **Variation 2, first strain + repeat + second half** (006) |
| **94:2–106** | Variation 2 from the **second-strain pickup** only (005) |
| 108–113 | Rests / gap |
| **114:2–124** | G-pedal variation (G4 written between almost every note) |
| 114:2–130 | Longer slice of the same variation |
| 140:1–146 | Slow lyrical (C6–B5–A5 half notes) |
| 159:1–177 | F#–G–A neighbor / 16th-note variation |
| 188:1–204 | 6/8 variation |
| 208–241 | Late / coda |

If a take is 30–40s of looping, it can still be a ~12–17 bar window: players restart phrases.

## Written repeats are not `repetition` errors

Variation 2 (and similar binary variations):

- First strain (e.g. m90 pickup–m94 first ending) then **written** second strain (m95–98) is **in the score**.
- Do **not** label the second C-arpeggio / G-arpeggio as `repetition` of the first.
- `repetition` is only for **practice restarts**: stopping, holding, then playing the same figure again when the score does not ask for it.
- Always set `repeats_label_range` to the earlier span being copied.

005-style `repetition`s are **local 0.5–2s figures** (a stalled cadence note, a 1-bar arpeggio retry), not 6–8s chunks of a written repeat.

## F0 cannot be trusted on inner arpeggio / pedal tones

**006 correction:** “missing” written G4s inside C and G arpeggios were **there**, just very quiet / short. F0 skipped them and jumped E4→C5.

**007:** the G-pedal variation writes G4 on almost every 16th. F0 shows only the skeleton (C5–E4–C5…) and drops the G4s. Same rule.

Do **not** label `missed_note` when:

- The missing pitch is a **chord tone or pedal** inside a fast arpeggio/ostinato.
- The gap is &lt; ~80 ms between two clearly correct neighbors.
- A later written repeat of the same figure **does** show the note (F0 caught it the second time). That is evidence the first time was played too, not skipped.

Only label `missed_note` for an obvious skipped **melody** note (e.g. jumping over a whole figure), not inner 16ths.

A weak F0 blip **on** the expected inner pitch (007’s midi 66.6 between F4 and Bb4) is the pedal leaking through — **not** `extra_note`.

## Extra notes that re-hit the previous pitch

**006 correction:** around 12–13s there is an **extra concert D6** (written E6) after the real m97 E6, then that extra is a **repetition of the previous E6**.

Pattern:

1. Score has pitch P once.
2. Player sounds P, then attacks P (or the same concert pitch) again.
3. Label `extra_note` on the extra attack.
4. Also label `repetition` on the extra attack with `repeats_label_range` = the previous P.

User-saved 006 extra at **12.57–13.08** is this D6. Do not rely on greedy aligner “EXTRA” dumps: they mix split onsets with real extras. Prefer a **clear second attack** of a high note already just played.

Tiny extra blips (005/006 concert B4 of ~0.15s in a stalled cadence) are also `extra_note`. Very short extras (~20 ms) appear in 005; keep them if the human labeled that style, otherwise prefer ≥0.08s.

## Cadence stalls

If the player **holds** the cadence note instead of moving (006 m94 C5), label `rhythm_error` on the hold.

If they then **re-attack the same held pitch** before continuing, that re-attack is `repetition` of the hold — **unless** the human deleted it. On 006 the human **kept** the m94 `rhythm_error` and the B4 `extra_note`, and **removed** the C5 `repetition` and the G4 `missed_note`s. Prefer:

- `rhythm_error` for the stall
- `extra_note` for a foreign pitch during the stall
- `repetition` only when the restart is a **whole figure**, not a single held note (unless the take clearly re-plays a 1-bar cell)

006 human also **removed** the recap m103 C5-vs-D5 `wrong_note` (another inner-arpeggio F0 miss). Do not label ±1 chord-tone swaps inside arpeggios.

## What to label (keep the set small)

Target **about 6–12 labels** per take, like 005 (~11) and reviewed 006 (~6–8), not the batch auto pass (20–30).

Use:

| Type | When |
|------|------|
| `extra_note` | Clear extra attack; re-hit of previous note; post-excerpt noodling |
| `repetition` | Practice restart of a figure **not** written as a repeat; always with `repeats_label_range` |
| `rhythm_error` | Stall / collapse of a figure (held cadence, broken 16ths that fall apart) |
| `wrong_note` | Stable wrong pitch **≥ ~2 semitones** on a melody note, not an inner chord tone |
| `missed_note` | Skipped obvious melody note only |
| `intonation_error` | Right pitch class, clearly off (don’t use for noisy F0) |
| `bad_start` / `bad_timbre` / `squeak` | Ear-only; **do not guess** from F0 |
| `sliding` | Rolling over a note too fast (smear / incomplete articulation) |

`source`: `manual`. `annotator_id`: `ai_f0_align`. `severity`: 3 unless extreme. Times are on **trimmed** `performance_audio.wav`.

Do **not** emit batch `extra_note`s for every unmatched onset. Do **not** copy labels or times from another take.

## Alignment method

1. Confirm / correct `score_segment` (measures + start beat).
2. Synthesize reference; do not overwrite human `labels.json`.
3. Transcribe with pyin + onsets, but treat it as a **skeleton**.
4. Map **structural pickups** (var. 2: concert F3 = written G3) to score pickups (m90, m92, m94, m96, m99, m102, m104).
5. Greedy MIDI align **over-labels** MISS/EXTRA on fast music. Use it only as a hint.
6. Write few labels; put a short `comment` with written pitch + measure so a human can check in the GUI.

## Reviewed samples (do not clobber)

- **003** Theme 20:4–36
- **004** Var. 1 45–61
- **005** Var. 2 from 94:2–106 (human labels; some may still reflect the old 66–82 score)
- **006** Var. 2 **90:2–106** (human-edited after AI pass)

## 006 human labels after review (ground truth style)

Kept: m94 hold `rhythm_error`; m94 B4 `extra_note`; m98 extra chromatics `extra_note`; m102 collapse `rhythm_error`; extra D6 ~12.57–13.08; extra ~19.58–19.74.

Removed: inner-arpeggio G4 `missed_note`s; C5 stall `repetition`; recap C5/D5 `wrong_note`.
