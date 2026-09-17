from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import torch

from alignmodel.joint.packed_data import sha256_file
from alignmodel.transcription.mel_v1 import (
    CACHE_SCHEMA_VERSION,
    MelDecodeConfig,
    MelFrontendConfig,
    MelNoteTranscriber,
    MelTranscriberConfig,
    _local_peaks,
    decode_mel_notes,
    load_mel_checkpoint,
    make_mel_targets,
    mel_transcriber_loss,
)
from alignmodel.transcription.mel_v1_data import (
    MelPackedCache,
    _canonical_json,
)
from alignmodel.transcription.mel_v1_train import (
    MelTrainConfig,
    _split_train_fold,
    train_mel_transcriber,
)


def _probabilities(frames: int, classes: int = 49) -> dict[str, np.ndarray]:
    return {
        "voiced": np.full(frames, 0.02, np.float32),
        "pitch": np.full((frames, classes), 1e-5, np.float32),
        "onset": np.full(frames, 0.02, np.float32),
        "boundary": np.full(frames, 0.02, np.float32),
        "rearticulation": np.full(frames, 0.02, np.float32),
        "confidence": np.full(frames, 0.02, np.float32),
    }


def _make_cache(root: Path, *, records: int = 3) -> Path:
    root.mkdir()
    frontend = MelFrontendConfig(hop_length=256)
    frames_per = 32
    values = np.full(
        (records * frames_per, frontend.n_mels), -0.25, dtype="<f2"
    )
    shard = root / "shard-00000.mel.float16.bin"
    shard.write_bytes(values.tobytes())
    index = root / "index.sqlite"
    connection = sqlite3.connect(index)
    connection.execute(
        "CREATE TABLE records("
        "ordinal INTEGER PRIMARY KEY, sample TEXT UNIQUE NOT NULL, "
        "split TEXT NOT NULL, source TEXT NOT NULL, shard INTEGER NOT NULL, "
        "frame_offset INTEGER NOT NULL, frame_count INTEGER NOT NULL, "
        "duration_sec REAL NOT NULL, audio_render TEXT NOT NULL, "
        "effective_audio_transpose INTEGER NOT NULL, audio_sha256 TEXT NOT NULL, "
        "mel_sha256 TEXT NOT NULL, normalization TEXT NOT NULL, target BLOB NOT NULL)"
    )
    target = [
        {"pitch": 60, "start_sec": 0.03, "end_sec": 0.12},
        {"pitch": 60, "start_sec": 0.12, "end_sec": 0.24},
    ]
    raw_record = values[:frames_per].tobytes()
    for index_value in range(records):
        connection.execute(
            "INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                index_value,
                f"clip-{index_value}",
                "train" if index_value < records - 1 else "val",
                "Mozart",
                0,
                index_value * frames_per,
                frames_per,
                frames_per * frontend.hop_sec,
                "soundfont_v1",
                2,
                "a" * 64,
                hashlib.sha256(raw_record).hexdigest(),
                "{}",
                sqlite3.Binary(zlib.compress(_canonical_json(target))),
            ),
        )
    connection.commit()
    connection.close()
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "frontend_config": frontend.to_dict(),
        "dtype": "float16",
        "layout": "time_major_contiguous",
        "shard_rows": 64,
        "split_counts": {"train": records - 1, "val": 1},
        "record_count": records,
        "source": {
            "release": "joint-outputraw-full-v1",
            "pack_id": "release-pack",
            "locked_test_materialized": False,
        },
        "index": {
            "name": "index.sqlite",
            "bytes": index.stat().st_size,
            "sha256": sha256_file(index),
        },
        "shards": [{
            "name": shard.name,
            "bytes": shard.stat().st_size,
            "sha256": sha256_file(shard),
        }],
    }
    metadata["pack_id"] = hashlib.sha256(_canonical_json(metadata)).hexdigest()
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


class MelTargetAndLossTests(unittest.TestCase):
    def test_weak_short_note_gets_duration_weight(self) -> None:
        target = make_mel_targets(
            [{"pitch": 60, "start_sec": 0.05, "end_sec": 0.10}],
            frames=24,
            hop_sec=256 / 22050,
            midi_min=52,
            midi_max=100,
        )
        active = target["voiced"] > 0
        self.assertTrue(active.any())
        self.assertTrue(np.all(target["duration_weight"][active] == 4.0))
        self.assertEqual(int(target["pitch"][active][0]), 8)

    def test_same_pitch_rearticulation_target_differs_from_sustain(self) -> None:
        sustained = make_mel_targets(
            [{"pitch": 60, "start_sec": 0.0, "end_sec": 0.30}],
            frames=40, hop_sec=0.01, midi_min=52, midi_max=100,
        )
        repeated = make_mel_targets(
            [
                {"pitch": 60, "start_sec": 0.0, "end_sec": 0.15},
                {"pitch": 60, "start_sec": 0.15, "end_sec": 0.30},
            ],
            frames=40, hop_sec=0.01, midi_min=52, midi_max=100,
        )
        self.assertEqual(float(sustained["rearticulation"].sum()), 0.0)
        self.assertEqual(float(repeated["rearticulation"].sum()), 1.0)
        self.assertEqual(float(repeated["rearticulation"][15]), 1.0)

    def test_written_bb_pitch_is_target_not_sounding_pitch(self) -> None:
        # Immutable metadata says effective_audio_transpose=+2.  Exact lineage
        # already supplies written 60 even though the rendered clarinet F0 is 58.
        target = make_mel_targets(
            [{"pitch": 60, "start_sec": 0.0, "end_sec": 0.1}],
            frames=12, hop_sec=0.01, midi_min=52, midi_max=100,
        )
        self.assertEqual(set(target["pitch"][target["voiced"] > 0]), {60 - 52})

    def test_frame_and_segment_loss_is_finite_and_differentiable(self) -> None:
        model = MelNoteTranscriber(MelTranscriberConfig(
            conv_channels=8, temporal_dim=16, spectral_blocks=2,
            temporal_blocks=1,
        ))
        mel = torch.randn(2, 128, 32)
        target_np = make_mel_targets(
            [
                {"pitch": 60, "start_sec": 0.03, "end_sec": 0.10},
                {"pitch": 62, "start_sec": 0.10, "end_sec": 0.22},
            ],
            frames=32, hop_sec=0.01, midi_min=52, midi_max=100,
        )
        batch = {
            key: torch.from_numpy(value).repeat(2, 1)
            for key, value in target_np.items()
        }
        batch["frame_mask"] = torch.ones(2, 32, dtype=torch.bool)
        loss, parts = mel_transcriber_loss(model(mel), batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(parts["segment"]), 0.0)
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))


class MelDecoderTests(unittest.TestCase):
    def test_weak_short_note_survives_high_resolution_decode(self) -> None:
        probability = _probabilities(20)
        probability["voiced"][3:8] = 0.72
        probability["pitch"][3:8, 60 - 52] = 0.75
        probability["onset"][3] = 0.55
        probability["boundary"][3] = 0.55
        probability["confidence"][3:8] = 0.65
        notes = decode_mel_notes(
            probability, midi_min=52, hop_sec=256 / 22050
        )
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].pitch, 60)
        self.assertLess(notes[0].end - notes[0].start, 0.080)

    def test_train_selected_confidence_gate_filters_weak_note(self) -> None:
        probability = _probabilities(20)
        probability["voiced"][3:8] = 0.72
        probability["pitch"][3:8, 60 - 52] = 0.75
        probability["onset"][3] = 0.55
        probability["boundary"][3] = 0.55
        probability["confidence"][3:8] = 0.65
        notes = decode_mel_notes(
            probability,
            midi_min=52,
            hop_sec=256 / 22050,
            config=MelDecodeConfig(min_confidence=0.8),
        )
        self.assertEqual(notes, [])

    def test_sustain_merges_but_strong_second_onset_is_preserved(self) -> None:
        probability = _probabilities(40)
        probability["voiced"][2:34] = 0.95
        probability["pitch"][2:34, 60 - 52] = 0.98
        probability["confidence"][2:34] = 0.90
        probability["onset"][2] = probability["boundary"][2] = 0.9
        probability["boundary"][18] = 0.50
        merged = decode_mel_notes(
            probability, midi_min=52, hop_sec=0.01
        )
        self.assertEqual(len(merged), 1)
        probability["onset"][18] = 0.90
        probability["rearticulation"][18] = 0.92
        repeated = decode_mel_notes(
            probability, midi_min=52, hop_sec=0.01
        )
        self.assertEqual(len(repeated), 2)
        self.assertEqual([note.pitch for note in repeated], [60, 60])

    def test_local_peak_contract(self) -> None:
        self.assertEqual(_local_peaks(np.array([0.1, 0.8, 0.2]), 0.5), [1])


class MelCacheAndCheckpointTests(unittest.TestCase):
    def test_zero_calibration_fraction_uses_every_training_row(self) -> None:
        records = [object(), object(), object()]
        training, calibration = _split_train_fold(records, 0.0, 365)
        self.assertEqual(training, records)
        self.assertEqual(calibration, [])

    def test_cache_integrity_and_target_free_inference_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _make_cache(Path(temporary) / "cache")
            cache = MelPackedCache(root, deep=True)
            records = cache.records("val", include_targets=False)
            self.assertEqual(records[0].target, ())
            self.assertEqual(records[0].effective_audio_transpose, 2)
            cache.close()
            shard = root / "shard-00000.mel.float16.bin"
            data = bytearray(shard.read_bytes())
            data[0] ^= 1
            shard.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "checksum"):
                MelPackedCache(root, deep=True)

    def test_checkpoint_load_and_atomic_mid_epoch_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cache = _make_cache(base / "cache", records=3)
            output = base / "run"
            model_config = MelTranscriberConfig(
                conv_channels=8, temporal_dim=16, spectral_blocks=2,
                temporal_blocks=1,
            )
            config = MelTrainConfig(
                output_dir=output,
                epochs=1,
                batch_size=1,
                crop_frames=32,
                crops_per_clip=2,
                workers=0,
                calibration_fraction=0.5,
                calibration_max_clips=1,
                calibration_every_epochs=1,
                checkpoint_every_steps=1,
                amp="none",
                model=model_config,
            )
            train_mel_transcriber(cache, config, device="cpu")
            mid = output / "mid_epoch_checkpoint.pt"
            self.assertTrue(mid.exists())
            self.assertTrue((output / "best.pt").exists())
            mid_payload = torch.load(mid, map_location="cpu", weights_only=False)
            self.assertTrue(mid_payload["progress"]["optimizer_boundary"])
            self.assertGreater(mid_payload["progress"]["position"], 0)
            train_mel_transcriber(cache, config, device="cpu", resume=mid)
            last = output / "last.pt"
            model, frontend, _decode, payload = load_mel_checkpoint(last)
            self.assertIsInstance(model, MelNoteTranscriber)
            self.assertEqual(frontend.hop_length, 256)
            self.assertEqual(payload["progress"]["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
