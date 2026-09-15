from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from alignmodel.joint.data import JointTrainingExample
from alignmodel.joint.index import JointEvent, ScoreEvent
from alignmodel.joint.lattice import (
    JointCandidate,
    LatticeConfig,
    SparseJointLattice,
)
from alignmodel.joint.outputraw_full import (
    FullJointPipelineModel,
    full_pipeline_loss,
)
from alignmodel.joint.outputraw_train import (
    collate_local_samples,
    prepare_local_sample,
)
from alignmodel.joint.packed_data import (
    ExistingShardMismatch,
    PackedCursor,
    PackedJointDataset,
    PackedJointWriter,
    canonical_target,
    validate_split_manifest,
)
from alignmodel.joint.prepared_local import (
    PreparedLocalDataset,
    PreparedLocalWriter,
)
from alignmodel.transcription.basic_pitch import BasicPitchFeatures


def _features(seed: int, frames: int = 7) -> BasicPitchFeatures:
    rng = np.random.default_rng(seed)
    return BasicPitchFeatures(
        note=rng.random((frames, 88), dtype=np.float32),
        onset=rng.random((frames, 88), dtype=np.float32),
        contour=rng.random((frames, 264), dtype=np.float32),
        frame_times=np.arange(frames, dtype=np.float64) / 86.0,
        metadata={
            "cache_schema_version": 1,
            "frontend_version": "test",
            "wav_sha256": f"{seed:064x}",
            "pitch_policy": {"audio_to_written_shift": 2},
        },
    )


def _example(sample: str) -> JointTrainingExample:
    score = (
        ScoreEvent(0, 60, 0.0, 2.0, (0, 1), 1),
        ScoreEvent(1, 62, 2.0, 3.0, (2,), 2),
        ScoreEvent(2, 64, 3.0, 4.0, (3,), 2),
    )
    candidates = (
        JointCandidate(60, 0.0, 1.0, 0.9, (0,), (0.9,) * 5),
        JointCandidate(63, 1.0, 2.0, 0.8, (1,), (0.8,) * 5),
    )
    targets = (
        JointEvent(60, 0.0, 1.0, (0, 1), "match", source_indices=(0, 1)),
        JointEvent(
            60,
            1.0,
            2.0,
            (0, 1),
            "copy",
            copy_pass=1,
            source_indices=(0, 1),
        ),
        JointEvent(63, 2.0, 3.0, (1, 2), "substitute", source_indices=(2,)),
    )
    return JointTrainingExample(
        sample=sample,
        source="Mozart",
        candidates=candidates,
        score=score,
        gold_spans=((0, 1), (1, 2)),
        gold_keep_unlinked=(True, True),
        target_events=targets,
        target_deletions=frozenset({2}),
    )


def _row(sample: str, split: str, seed: int) -> dict:
    return {
        "sample": sample,
        "sample_dir": f"C:/source/{sample}",
        "source": "Mozart",
        "split": split,
        "source_hashes": {"performance_audio.wav": f"{seed:064x}"},
        "clean_fingerprint": f"clean-{sample}",
        "leakage_group": f"group-{sample}",
    }


def _write_pack(root: Path) -> tuple[dict[str, BasicPitchFeatures], list[dict]]:
    originals = {}
    rows = []
    with PackedJointWriter(
        root,
        manifest_sha256="a" * 64,
        candidate_version="test-candidates",
        shard_rows=2,
    ) as writer:
        for index in range(5):
            split = "train" if index < 4 else "val"
            sample = f"sample-{index}"
            row = _row(sample, split, index + 1)
            features = _features(index + 1, frames=5 + index)
            example = _example(sample)
            target = canonical_target(
                example,
                valid_labels=[
                    {
                        "type": "rhythm_error",
                        "start_time": 1.5,
                        "end_time": 2.5,
                    }
                ],
            )
            writer.add(row, features, example, target)
            originals[sample] = features
            rows.append(row)
        writer.finalize()
    return originals, rows


def test_packed_arrays_targets_and_tie_projection_are_exact(tmp_path: Path) -> None:
    root = tmp_path / "packed"
    originals, _rows = _write_pack(root)
    with PackedJointDataset(root, manifest_sha256="a" * 64) as dataset:
        assert len(dataset) == 5
        saved = dataset[0]
        original = originals[saved.sample]
        assert np.array_equal(saved.features.note, original.note)
        assert np.array_equal(saved.features.onset, original.onset)
        assert np.array_equal(saved.features.contour, original.contour)
        assert np.array_equal(saved.features.frame_times, original.frame_times)
        assert saved.training_example() == _example(saved.sample)
        assert saved.target["score"][0]["tie_chain"] is True
        assert saved.target["score"][1]["tie_chain"] is False
        assert saved.target["layer1_repeats"] == [
            {
                "copy_count": 1,
                "copy_pass": 1,
                "rendered_event_span": [1, 2],
                "resume_event": 1,
                "source_span": [0, 1],
            }
        ]
        assert saved.target["masks"]["intonation"] is False
        assert dataset.validate(deep=True)["record_checks"] == 5


def test_order_and_resume_cursor_are_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "packed"
    _write_pack(root)
    with PackedJointDataset(root) as dataset:
        first = dataset.deterministic_order("train", epoch=3, seed=365)
        assert first == dataset.deterministic_order("train", epoch=3, seed=365)
        assert first != dataset.deterministic_order("train", epoch=4, seed=365)
        initial = dataset.cursor("train", epoch=3, seed=365)
        consumed = list(dataset.iter_from_cursor(initial))
        resume = consumed[1][0]
        remaining = list(dataset.iter_from_cursor(resume, workers=2, prefetch=3))
        assert [sample.ordinal for _cursor, sample in remaining] == first[2:]
        round_trip = PackedCursor.from_dict(
            json.loads(json.dumps(resume.to_dict()))
        )
        assert round_trip == resume


def test_mmap_cache_is_bounded_by_shard_count(tmp_path: Path) -> None:
    root = tmp_path / "packed"
    _write_pack(root)
    with PackedJointDataset(root, max_open_shards=1) as dataset:
        for index in range(len(dataset)):
            dataset[index]
            assert len(dataset._maps) <= 4


def test_metadata_only_training_read_skips_frontend_arrays(tmp_path: Path) -> None:
    root = tmp_path / "packed"
    _write_pack(root)
    with PackedJointDataset(root) as full:
        expected = full[0]
    with PackedJointDataset(
        root,
        verify_records=False,
        load_feature_arrays=False,
    ) as metadata_only:
        actual = metadata_only[0]
        assert actual.features is None
        assert actual.candidates == expected.candidates
        assert actual.target == expected.target
        assert actual.training_example() == expected.training_example()
        assert not metadata_only._maps
    with pytest.raises(ValueError, match="requires frontend feature arrays"):
        PackedJointDataset(root, load_feature_arrays=False)


def test_prepared_local_mmap_is_exact_and_resumes_committed_prefix(
    tmp_path: Path,
) -> None:
    root = tmp_path / "packed"
    _write_pack(root)
    destination = tmp_path / "prepared"
    config = LatticeConfig(max_options_per_candidate=8, max_states=32)
    lattice = SparseJointLattice(FullJointPipelineModel(), config)
    with PackedJointDataset(
        root,
        verify_records=False,
        load_feature_arrays=False,
    ) as source:
        pack_id = str(source.metadata["pack_id"])
        expected = {
            ordinal: prepare_local_sample(source[ordinal], lattice)
            for ordinal in source.ordinals("train")
        }
    writer = PreparedLocalWriter(
        destination,
        pack_id=pack_id,
        lattice_config=config,
        shard_rows=2,
        checkpoint_rows=1,
    )
    for ordinal in (0, 1):
        writer.add(
            source_ordinal=ordinal,
            sample=expected[ordinal].sample,
            prepared=expected[ordinal],
        )
    staging = writer.staging
    writer.preserve()
    with PreparedLocalWriter(
        destination,
        pack_id=pack_id,
        lattice_config=config,
        shard_rows=2,
        checkpoint_rows=1,
        staging=staging,
    ) as resumed:
        resumed.validate_prefix(
            [(ordinal, expected[ordinal].sample) for ordinal in range(4)]
        )
        for ordinal in (2, 3):
            resumed.add(
                source_ordinal=ordinal,
                sample=expected[ordinal].sample,
                prepared=expected[ordinal],
            )
        resumed.finalize()
    with PreparedLocalDataset(
        destination,
        pack_id=pack_id,
        lattice_config=config,
        max_open_shards=1,
    ) as prepared:
        assert prepared.validate(deep=True)["records"] == 4
        loaded = []
        for ordinal, wanted in expected.items():
            actual = prepared[ordinal]
            loaded.append(actual)
            assert torch.equal(actual.edge_features, wanted.edge_features)
            assert actual.groups == wanted.groups
            for name in (
                "keep",
                "boundary",
                "split",
                "emission",
                "structure",
                "layer2",
                "rhythm",
                "duration_target",
                "duration_weight",
                "rearticulation_weight",
            ):
                assert torch.equal(getattr(actual, name), getattr(wanted, name))
            assert len(prepared._maps) <= 4
    expected_batch = collate_local_samples(list(expected.values()))
    actual_batch = collate_local_samples(loaded)
    torch.manual_seed(47)
    reference_model = FullJointPipelineModel()
    cached_model = copy.deepcopy(reference_model)
    torch.manual_seed(48)
    reference_loss = full_pipeline_loss(
        reference_model,
        expected_batch.edge_features,
        expected_batch.targets,
    ).total
    torch.manual_seed(48)
    cached_loss = full_pipeline_loss(
        cached_model,
        actual_batch.edge_features,
        actual_batch.targets,
    ).total
    assert torch.equal(reference_loss, cached_loss)
    reference_loss.backward()
    cached_loss.backward()
    for reference, cached in zip(
        reference_model.parameters(), cached_model.parameters()
    ):
        assert torch.equal(reference.grad, cached.grad)


def test_hash_and_corruption_checks_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "packed"
    _write_pack(root)
    with pytest.raises(ValueError, match="different manifest"):
        PackedJointDataset(root, manifest_sha256="b" * 64)

    shard = next(root.glob("shard-00000.note.*.bin"))
    with shard.open("r+b") as stream:
        byte = stream.read(1)
        stream.seek(0)
        stream.write(bytes([byte[0] ^ 0xFF]))
    with PackedJointDataset(root) as dataset:
        with pytest.raises(ValueError, match="checksum mismatch"):
            dataset[0]


def test_manifest_rejects_leakage_and_materialized_test() -> None:
    clean = {
        "train": [
            {
                "sample": "train",
                "leakage_group": "a",
                "clean_fingerprint": "a",
            }
        ],
        "val": [
            {
                "sample": "val",
                "leakage_group": "b",
                "clean_fingerprint": "b",
            }
        ],
        "test": [
            {
                "sample": "test",
                "leakage_group": "c",
                "clean_fingerprint": "c",
            }
        ],
    }
    validate_split_manifest(clean)
    leaked = json.loads(json.dumps(clean))
    leaked["val"][0]["clean_fingerprint"] = "a"
    with pytest.raises(ValueError, match="leakage"):
        validate_split_manifest(leaked)
    materialized = json.loads(json.dumps(clean))
    materialized["test"][0]["target_record"] = 3
    with pytest.raises(ValueError, match="materialized"):
        validate_split_manifest(materialized)


def test_repeated_extra_has_no_fake_score_span() -> None:
    base = _example("repeated-extra")
    repeated_extra = JointEvent(
        70,
        3.0,
        3.5,
        None,
        "copy",
        copy_pass=1,
        origin_relationship="extra",
    )
    example = JointTrainingExample(
        sample=base.sample,
        source=base.source,
        candidates=base.candidates,
        score=base.score,
        gold_spans=base.gold_spans,
        gold_keep_unlinked=base.gold_keep_unlinked,
        target_events=(*base.target_events, repeated_extra),
        target_deletions=base.target_deletions,
    )
    target = canonical_target(example)
    assert target["layer1_repeats"][-1]["source_span"] is None
    assert target["layer2_operations"][3]["operation"] == "extra"


def test_extra_with_neighbor_span_remains_extra_operation() -> None:
    base = _example("spanned-extra")
    spanned_extra = JointEvent(70, 3.0, 3.5, (1, 2), "extra")
    example = JointTrainingExample(
        sample=base.sample,
        source=base.source,
        candidates=base.candidates,
        score=base.score,
        gold_spans=base.gold_spans,
        gold_keep_unlinked=base.gold_keep_unlinked,
        target_events=(*base.target_events, spanned_extra),
        target_deletions=base.target_deletions,
    )
    assert canonical_target(example)["layer2_operations"][3]["operation"] == "extra"


def test_crash_staging_resumes_from_committed_cursor(tmp_path: Path) -> None:
    root = tmp_path / "packed"
    writer = PackedJointWriter(
        root,
        manifest_sha256="a" * 64,
        candidate_version="test-candidates",
        shard_rows=2,
        checkpoint_rows=1,
    )
    rows = [_row(f"sample-{index}", "train", index + 1) for index in range(4)]
    for index, row in enumerate(rows[:3]):
        features = _features(index + 1, frames=5 + index)
        example = _example(row["sample"])
        writer.add(row, features, example, canonical_target(example))
    staging = writer.staging
    writer.preserve()

    # Simulate bytes written beyond the last committed cursor.
    note = staging / "shard-00001.note.float32.bin"
    with note.open("ab") as stream:
        stream.write(b"partial-crash-tail")

    with PackedJointWriter(
        root,
        manifest_sha256="a" * 64,
        candidate_version="test-candidates",
        shard_rows=2,
        staging=staging,
        checkpoint_rows=1,
    ) as resumed:
        resumed.validate_committed_prefix(rows)
        assert resumed.committed_records == 3
        features = _features(4, frames=8)
        example = _example(rows[3]["sample"])
        with pytest.raises(ExistingShardMismatch):
            resumed.reuse(
                rows[3],
                features,
                example,
                canonical_target(example),
            )
        resumed.begin_append()
        resumed.add(rows[3], features, example, canonical_target(example))
        resumed.finalize()

    with PackedJointDataset(root) as dataset:
        assert len(dataset) == 4
        assert np.array_equal(dataset[3].features.note, _features(4, 8).note)

