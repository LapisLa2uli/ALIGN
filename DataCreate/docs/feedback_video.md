# Animated score feedback

`datacreate-feedback-video` exports an **854 × 480, 24 fps MP4** with H.264
video and AAC audio. It uses the saved feedback mix and note timings, with no
LLM or Fish calls and no music regeneration. The B-flat clarinet sounding-pitch
correction, shared slowdown, loudness, reverb, and silence margins are retained.
The presentation uses a black background, white staff lines and notation,
bright error colors, and high-contrast captions and cursor highlights.

Install the optional local engraving/video dependencies:

```powershell
python -m pip install -e './DataCreate[video]'
```

For an existing run containing a comparison for every retained label:

```powershell
python -m datacreate.feedback_video `
  --sample DataCreate/samples/001 `
  --feedback feedback/sample-001-sounding-pitch-corrected `
  --output feedback/sample-001-video
```

`DataCreate/src` must be on `PYTHONPATH` unless DataCreate is installed. The output
directory must be new. The sample directory supplies the selected/full MusicXML,
reference MIDI, metadata, and `note_alignment_v2.json` that produced the audio.
The saved clip hashes and alignment hash are checked before rendering.

The ordinary audio workflow chooses up to three teaching points. To narrate
**every label**, first generate a run with `--all-labels`:

```powershell
python -m datacreate.feedback --labels DataCreate/samples/001/labels.json `
  --config DataCreate/config/feedback.ssstoken.teacher.yaml `
  --all-labels --output feedback/sample-001-every-label

python -m datacreate.feedback_video --sample DataCreate/samples/001 `
  --feedback feedback/sample-001-every-label --output feedback/sample-001-video
```

This audio mode makes one LLM request per retained label, with three speech
segments per comparison. Each label keeps its own concise teaching point;
same-bar labels are not deduplicated. The speech-length limit applies per point.
Saved plans retain all points when retried with `--plan`. Video export refuses
to silently omit labels without a narrated comparison. Rejected detector labels
remain excluded by the normal feedback filtering policy.

For each label the film:

1. Shows its original reference measure(s), with the affected region shaded.
2. Follows reference synthesis note events with a blue cursor.
3. Changes to engraved performance transcription for the performed synthesis.
   Added notes are orange-red, wrong pitches rose, and missing notes amber cue
   notes. Missing cue notes are visual reminders and never receive playback
   cursors. Longer transcriptions turn pages as playback advances.
4. Shows the mistake label and shades its region throughout that label's
   diagnosis/advice segment. Subtitles show the corresponding narration.

Both scores display written instrument pitches; their audio uses sounding
pitches. Performance notation rounds note values for readability and is labeled
as approximate. Cursor onsets/ends use the exact unrounded synthesis seconds,
with a maximum display resolution of one frame (41.7 ms). A missing note cannot
be timed as a performed note, so it is placed before the following aligned note.
Highlighting is tied to the teaching point, not individual spoken-word alignment.
The renderer currently targets the project's single-part melodic scores.

The output includes `feedback.mp4`, `video.json`, exact `animation-events.json`,
representative preview PNGs, and `score-assets/` containing editable MusicXML,
SVG, and PNG notation. Audio is re-encoded to AAC for MP4 compatibility; it is
not resynthesized. No media or scores leave the computer during video export.
