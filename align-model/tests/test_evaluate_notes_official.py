import json
import sys
from pathlib import Path

import pretty_midi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines" / "common"))
from evaluate_notes import evaluate
from prepare_dataset import output_paths, write_label_midi


def test_evaluate_notes_keeps_legacy_mir_eval_and_official_note_wise(tmp_path: Path):
    root = tmp_path / "data"
    pred = tmp_path / "pred"
    root.mkdir()
    pred.mkdir()
    ids = ["all_classes", "only_correct"]
    (pred / "evaluated_ids.json").write_text(json.dumps(ids))
    (root / "manifest.json").write_text(
        json.dumps({"tracks": {tid: {"real_test": False} for tid in ids}})
    )
    for tid in ids:
        midi = pretty_midi.PrettyMIDI()
        for index, name in enumerate(("extra", "missing", "correct")):
            notes = (
                []
                if tid == "only_correct" and name != "correct"
                else [{"start": 1.0, "end": 1.5, "pitch": 60 + index}]
            )
            write_label_midi(
                output_paths(root, tid)["removed" if name == "missing" else name],
                notes,
            )
            inst = pretty_midi.Instrument(71, name=name)
            inst.notes = [
                pretty_midi.Note(90, note["pitch"], note["start"], note["end"])
                for note in notes
            ]
            midi.instruments.append(inst)
        (pred / tid).mkdir()
        midi.write(str(pred / tid / "mix.mid"))
    result = evaluate(root, pred)
    assert result["legacy_mir_eval_onset_50ms"]["micro"]["all"]["F1"] == 1.0
    assert result["official_note_wise"]["status"] == "available"
    assert result["official_note_wise"]["f1"] == 1.0
    assert result["official_note_wise"]["schema_version"] == (
        "align-note-wise-score-event-metric-v1"
    )
