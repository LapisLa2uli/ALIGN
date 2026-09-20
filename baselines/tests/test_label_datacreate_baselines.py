from pathlib import Path
import json
import sys

import pretty_midi

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "baselines" / "scripts"), str(ROOT / "baselines" / "common")]
from label_datacreate_baselines import document_from_midi, deny_gold, write_sample_document
from eval_bridge import class_from_name


def _score(path: Path) -> None:
    import music21

    score = music21.stream.Score()
    part = music21.stream.Part()
    for pitch in (60, 62, 64):
        part.append(music21.note.Note(pitch, quarterLength=1))
    score.append(part)
    score.write("musicxml", fp=path)


def _midi(path: Path, tracks: dict[str, list[tuple[float, float, int]]]) -> None:
    pm = pretty_midi.PrettyMIDI()
    for name, notes in tracks.items():
        instrument = pretty_midi.Instrument(program=71, name=name)
        for start, end, pitch in notes:
            instrument.notes.append(
                pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=end)
            )
        pm.instruments.append(instrument)
    pm.write(str(path))


def test_wrong_missed_extra_and_repetition_conversion(tmp_path):
    score = tmp_path / "verified_score.musicxml"
    midi = tmp_path / "mix.mid"
    _score(score)
    _midi(
        midi,
        {
            "extra": [
                (1.0, 1.4, 65),
                (3.0, 3.3, 58),
                (3.3, 3.6, 60),
                (3.6, 3.9, 62),
            ],
            "missing": [(1.05, 1.4, 60), (2.0, 2.4, 62)],
            "correct": [(0.0, 0.4, 58)],
        },
    )
    document = document_from_midi(
        midi, score, model="polytune", sample_id="001"
    )
    types = [label["type"] for label in document["labels"]]
    assert "wrong_note" in types
    assert "missed_note" in types
    assert "extra_note" in types or "repetition" in types
    assert document["schema_version"] == "1.2"
    assert document["baseline_labeling"]["model"] == "polytune"
    located = [label for label in document["labels"] if label.get("score_part")]
    assert located
    assert all(label["source"] == "agent" for label in document["labels"])


def test_unlocated_extra_has_no_invented_identity(tmp_path):
    score = tmp_path / "verified_score.musicxml"
    midi = tmp_path / "mix.mid"
    _score(score)
    _midi(midi, {"extra": [(8.0, 8.4, 90)], "missing": [], "correct": [(0.0, 0.4, 58)]})
    document = document_from_midi(
        midi, score, model="laddersym", sample_id="002"
    )
    extras = [label for label in document["labels"] if label["type"] == "extra_note"]
    assert extras
    assert extras[0]["score_part"] is None
    assert document["baseline_labeling"]["unlocated_events"] >= 1


def test_writer_does_not_touch_human_labels(tmp_path):
    sample = tmp_path / "001"
    sample.mkdir()
    score = sample / "verified_score.musicxml"
    midi = tmp_path / "mix.mid"
    human = sample / "labels.json"
    human.write_text(json.dumps({"schema_version": "1.1", "labels": [{"id": "keep"}]}))
    _score(score)
    _midi(midi, {"missing": [(0.0, 0.4, 58)], "extra": [], "correct": []})
    document = document_from_midi(midi, score, model="polytune", sample_id="001")
    path = write_sample_document(sample, "polytune", document)
    assert path.name == "labels_polytune.json"
    assert json.loads(human.read_text())["labels"][0]["id"] == "keep"


def test_gold_open_is_denied():
    raised = False
    try:
        deny_gold("open", ("D:/samples/001/labels.json",))
    except PermissionError:
        raised = True
    assert raised


def test_midi_class_names_decode_to_native_tracks():
    assert class_from_name("Extra") == "extra"
    assert class_from_name("Missing") == "missing"
    assert class_from_name("Missed") == "missing"
    assert class_from_name("Correct") == "correct"
    assert class_from_name("unknown-track") is None
