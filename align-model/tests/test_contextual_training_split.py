from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from alignmodel.contextual_align_train import _training_paths


class FrozenTrainingSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "split.json"

    def write_manifest(self, train, val):
        document = {}
        for split, names in (("train", train), ("val", val)):
            document[split] = []
            for name in names:
                path = self.root / f"{name}.json"
                path.write_text("{}", encoding="utf-8")
                document[split].append(
                    {"corpus": "procedural12k", "note_map": str(path)}
                )
        self.manifest.write_text(json.dumps(document), encoding="utf-8")

    def test_insufficient_training_never_borrows_validation(self):
        self.write_manifest(["train"], ["v1", "v2", "v3"])
        original = self.manifest.read_bytes()
        with self.assertRaisesRegex(ValueError, "frozen train split"):
            _training_paths(self.manifest, 2, 1)
        self.assertEqual(self.manifest.read_bytes(), original)

    def test_selection_preserves_validation_membership(self):
        self.write_manifest(["t1", "t2"], ["v1", "v2"])
        train, val = _training_paths(self.manifest, 1, 2)
        self.assertEqual([path.stem for path in train], ["t1"])
        self.assertEqual([path.stem for path in val], ["v1", "v2"])

    def test_overlapping_maps_are_rejected_even_outside_subset(self):
        self.write_manifest(["t1", "shared"], ["v1", "shared"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            _training_paths(self.manifest, 1, 1)

    def test_duplicate_maps_are_rejected(self):
        self.write_manifest(["t1", "t1"], ["v1"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _training_paths(self.manifest, 1, 1)

    def test_validation_cannot_be_silently_shortened(self):
        self.write_manifest(["t1"], ["v1"])
        with self.assertRaisesRegex(ValueError, "frozen val split"):
            _training_paths(self.manifest, 1, 2)

    def test_missing_map_cannot_be_silently_dropped(self):
        self.write_manifest(["t1", "t2"], ["v1"])
        (self.root / "t1.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing train note map"):
            _training_paths(self.manifest, 1, 1)

    def target_row(self, record, database="targets.sqlite", **metadata):
        return {
            "corpus": "audited",
            "sample_dir": str(self.root / f"sample_{record}"),
            "target_db": str(self.root / database),
            "target_record": record,
            **metadata,
        }

    def test_validated_targets_preserve_rows_and_split_membership(self):
        document = {
            "train": [self.target_row(0), self.target_row(1)],
            "val": [self.target_row(2), self.target_row(3)],
        }
        self.manifest.write_text(json.dumps(document), encoding="utf-8")
        train, val = _training_paths(self.manifest, 1, 2)
        self.assertEqual(train, document["train"][:1])
        self.assertEqual(val, document["val"])

    def test_duplicate_target_records_ignore_path_spelling_and_metadata(self):
        for split in ("train", "val"):
            with self.subTest(split=split):
                document = {
                    "train": [self.target_row(0)],
                    "val": [self.target_row(1)],
                }
                record = document[split][0]["target_record"]
                document[split].append(
                    self.target_row(str(record), "nested/../targets.sqlite", priority=2)
                )
                self.manifest.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, f"Duplicate.*{split}"):
                    _training_paths(self.manifest, 1, 1)

    def test_overlapping_target_records_are_rejected_outside_subset(self):
        document = {
            "train": [self.target_row(0), self.target_row(2)],
            "val": [self.target_row(1), self.target_row("2", priority=3)],
        }
        self.manifest.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "overlap"):
            _training_paths(self.manifest, 1, 1)

    def test_target_record_numbers_are_scoped_to_each_database(self):
        document = {
            "train": [self.target_row(0, "train.sqlite")],
            "val": [self.target_row(0, "val.sqlite")],
        }
        self.manifest.write_text(json.dumps(document), encoding="utf-8")
        train, val = _training_paths(self.manifest, 1, 1)
        self.assertEqual(train, document["train"])
        self.assertEqual(val, document["val"])

    def test_target_records_never_borrow_from_the_other_split(self):
        document = {
            "train": [self.target_row(0)],
            "val": [self.target_row(1), self.target_row(2)],
        }
        self.manifest.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen train split"):
            _training_paths(self.manifest, 2, 1)

    def test_note_maps_and_target_records_can_share_a_manifest(self):
        self.write_manifest(["t1"], ["v1"])
        document = json.loads(self.manifest.read_text(encoding="utf-8"))
        target = self.target_row(0)
        document["train"].append(target)
        self.manifest.write_text(json.dumps(document), encoding="utf-8")
        train, val = _training_paths(self.manifest, 2, 1)
        self.assertEqual(train, [(self.root / "t1.json").resolve(), target])
        self.assertEqual(val, [(self.root / "v1.json").resolve()])


if __name__ == "__main__":
    unittest.main()
