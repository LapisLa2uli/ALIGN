import copy
import random
import unittest

from music21 import meter, note, stream, tempo

from synthpipeline.config import SynthConfig
from synthpipeline.errors import inject_error
from synthpipeline.note_map import (
    _sequence_alignment,
    build_note_map,
    clean_note_index,
    sounding_notes,
    tag_clean_notes,
    tag_extra_note,
)


def _score(n_measures: int = 2) -> stream.Score:
    score = stream.Score()
    part = stream.Part()
    part.append(tempo.MetronomeMark(number=100))
    part.append(meter.TimeSignature("4/4"))
    pitches = ("C4", "D4", "E4", "F4")
    for measure_index in range(n_measures):
        measure = stream.Measure(number=measure_index + 1)
        for pitch_name in pitches:
            measure.append(note.Note(pitch_name, quarterLength=1.0))
        part.append(measure)
    score.append(part)
    return score


def _extra_repeat_config() -> SynthConfig:
    return SynthConfig(
        errors={
            "weights": {
                "wrong_note": 0.0,
                "missed_note": 0.0,
                "extra_note": 1.0,
                "rhythm_error": 0.0,
                "intonation_error": 0.0,
            },
            "per_clip_min": 1,
            "per_clip_max": 1,
            "repetition_prob": 1.0,
            "standalone_repetition_prob": 0.0,
            "repeat_gap_seconds": [0.0, 0.0],
            "repeat_extra_copies_weights": {1: 1.0},
        },
        generation={"pitch_min": "E3", "pitch_max": "C6"},
    )


class NoteMapTests(unittest.TestCase):
    def test_rendered_event_alignment_keeps_tied_span(self) -> None:
        mapping, deleted = _sequence_alignment(
            [60, 62, 64],
            [60, 62, 62, 64],
        )
        self.assertEqual(mapping, [0, 2, 3])
        self.assertEqual(deleted, [1])

    def test_clean_identity_survives_deepcopy(self) -> None:
        clean = _score(1)
        self.assertEqual(tag_clean_notes(clean), 4)
        cloned = copy.deepcopy(clean)
        self.assertEqual(
            [clean_note_index(item) for item in sounding_notes(cloned)],
            [0, 1, 2, 3],
        )
        first = sounding_notes(cloned)[0]
        delattr(first, "_synthpipeline_clean_note_index")
        self.assertEqual(clean_note_index(first), 0)

    def test_match_substitute_extra_and_miss_lineage(self) -> None:
        clean = _score(1)
        tag_clean_notes(clean)
        performed = copy.deepcopy(clean)
        performed_notes = sounding_notes(performed)

        performed_notes[1].pitch.nameWithOctave = "D#4"
        missing = performed_notes[2]
        missing.activeSite.remove(missing)
        inserted = note.Note("G4", quarterLength=0.5)
        tag_extra_note(inserted, performed)
        performed.parts[0].measure(1).insert(2.5, inserted)

        payload = build_note_map(clean, performed)
        by_clean = {
            entry["clean_index"]: entry
            for entry in payload["performed_notes"]
            if entry["clean_index"] is not None
        }
        extras = [
            entry
            for entry in payload["performed_notes"]
            if entry["clean_index"] is None
        ]

        self.assertEqual(by_clean[0]["relationship"], "match")
        self.assertEqual(by_clean[1]["relationship"], "substitute")
        self.assertEqual(by_clean[3]["relationship"], "match")
        self.assertEqual(payload["deleted_clean_notes"], [2])
        self.assertEqual(len(extras), 1)
        self.assertEqual(extras[0]["relationship"], "extra")
        self.assertEqual(extras[0]["copy_pass"], 0)

    def test_repeat_tracks_clean_and_extra_copy_pass(self) -> None:
        clean = _score(2)
        tag_clean_notes(clean)
        result = inject_error(
            copy.deepcopy(clean),
            random.Random(5),
            _extra_repeat_config(),
        )
        self.assertTrue(result.repeated)
        payload = build_note_map(clean, result.score)

        original_extras = [
            entry
            for entry in payload["performed_notes"]
            if entry["relationship"] == "extra"
        ]
        copied_extras = [
            entry
            for entry in payload["performed_notes"]
            if entry["relationship"] == "copy"
            and entry["origin_relationship"] == "extra"
        ]
        copied_clean = [
            entry
            for entry in payload["performed_notes"]
            if entry["relationship"] == "copy"
            and entry["clean_index"] is not None
        ]

        self.assertEqual(len(original_extras), 1)
        self.assertEqual(len(copied_extras), 1)
        self.assertEqual(copied_extras[0]["copy_pass"], 1)
        self.assertIsNone(copied_extras[0]["clean_index"])
        self.assertTrue(copied_clean)
        self.assertTrue(all(entry["copy_pass"] == 1 for entry in copied_clean))
        self.assertEqual(payload["deleted_clean_notes"], [])


if __name__ == "__main__":
    unittest.main()
