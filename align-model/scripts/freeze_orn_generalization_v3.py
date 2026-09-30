"""Audit and seal a freshly generated ORN v3 lockbox.

Every accepted generated row is assigned to the lockbox; there is no
outcome-dependent membership selection. Public audio/score inputs are copied,
corrected lineage targets are AES-GCM encrypted, and the temporary source root
can be deleted after the freeze verifies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import freeze_orn_generalization_v2 as v2
from alignmodel.joint.packed_data import sha256_file


SCHEMA_VERSION = "align-orn-generalization-v3"


def _row_id(sample: str, leakage_group: str) -> str:
    return hashlib.sha256(
        f"{sample}\0{leakage_group}".encode("utf-8")
    ).hexdigest()[:24]


def _prior_identities(
    release: Mapping[str, Any],
    lockbox_manifest: Mapping[str, Any],
) -> dict[str, set[str]]:
    output = {
        "leakage_group": set(),
        "audio_sha256": set(),
        "score_sha256": set(),
    }
    for split_rows in release["splits"]["development"].values():
        for row in split_rows:
            output["leakage_group"].add(str(row["leakage_group"]))
            hashes = row.get("source_hashes") or {}
            output["audio_sha256"].add(
                str(hashes.get("performance_audio.wav") or "")
            )
            output["score_sha256"].add(
                str(hashes.get("verified_score.musicxml") or "")
            )
    for row in lockbox_manifest["inputs"]:
        output["leakage_group"].add(str(row["leakage_group"]))
        output["audio_sha256"].add(str(row["audio"]["sha256"]))
        output["score_sha256"].add(str(row["score"]["sha256"]))
    return output


def _public_audit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"_lineage", "sample_dir"}
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-name",
        default="orn-generalization-v3",
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--v2-release", type=Path, required=True)
    parser.add_argument("--expected-v2-release-sha256", required=True)
    parser.add_argument("--v2-lockbox-manifest", type=Path, required=True)
    parser.add_argument("--expected-v2-lockbox-sha256", required=True)
    parser.add_argument(
        "--additional-prior-lockbox-manifest",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--expected-additional-prior-lockbox-sha256",
        action="append",
        default=[],
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--lockbox-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delete-source-after-freeze", action="store_true")
    args = parser.parse_args(argv)

    args.source_root = args.source_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.lockbox_key = args.lockbox_key.resolve()
    try:
        args.lockbox_key.relative_to(Path(__file__).resolve().parents[2])
    except ValueError:
        pass
    else:
        raise ValueError("Lockbox key must remain outside the repository")
    if sha256_file(args.config) != args.expected_config_sha256:
        raise ValueError("Generation config changed after protocol freeze")
    if sha256_file(args.v2_release) != args.expected_v2_release_sha256:
        raise ValueError("ORN v2 release mismatch")
    if (
        sha256_file(args.v2_lockbox_manifest)
        != args.expected_v2_lockbox_sha256
    ):
        raise ValueError("ORN v2 public lockbox manifest mismatch")
    if sha256_file(args.protocol) != args.expected_protocol_sha256:
        raise ValueError("ORN v3 protocol mismatch")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    release_schema = f"align-{args.release_name}"
    if protocol.get("release_name") != args.release_name:
        raise ValueError("Protocol release name mismatch")
    if protocol["freezer_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("Freezer changed after protocol freeze")
    if int(protocol["seed"]) != args.seed:
        raise ValueError("Protocol seed mismatch")
    if int(protocol["requested_rows"]) != args.expected_count:
        raise ValueError("Protocol row count mismatch")
    minimum_rows = int(protocol["minimum_strict_audit_rows"])
    freeze_path = args.output_dir / "FREEZE.json"
    if freeze_path.exists():
        raise FileExistsError(f"Refusing to overwrite {freeze_path}")

    samples = sorted(
        path
        for path in args.source_root.glob("synth_gen_*")
        if path.is_dir()
    )
    expected_names = {
        f"synth_gen_{value:04d}"
        for value in range(args.seed, args.seed + args.expected_count)
    }
    if len(samples) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} generated rows, found {len(samples)}"
        )
    if {path.name for path in samples} != expected_names:
        raise ValueError("Generated membership differs from seed/count protocol")

    hash_cache: dict[str, dict[str, Any]] = {}
    audited = []
    for position, sample_dir in enumerate(samples, 1):
        row = v2._audit_row(sample_dir, args.source_root, hash_cache)
        audited.append(row)
        if position == 1 or position % 10 == 0 or position == len(samples):
            print(
                f"audit={position}/{len(samples)} "
                f"eligible={sum(bool(value.get('eligible')) for value in audited)}",
                flush=True,
            )
    failures = [
        {
            "sample": row["sample"],
            "reasons": row.get("exclusion_reasons") or [],
        }
        for row in audited
        if not row.get("eligible")
    ]
    if failures:
        v2.v1._atomic_json(
            args.output_dir / "AUDIT_FAILURES.json",
            {
                "schema_version": f"{release_schema}-audit-failures",
                "failures": failures,
            },
        )
    eligible = [row for row in audited if row.get("eligible")]
    if len(eligible) < minimum_rows:
        raise ValueError(
            f"Only {len(eligible)} rows passed strict audit; "
            f"protocol requires at least {minimum_rows}"
        )
    groups = v2.v1._group_records(eligible)
    for group_id, rows in groups.items():
        for row in rows:
            row["leakage_group"] = group_id

    release = json.loads(args.v2_release.read_text(encoding="utf-8"))
    old_lockbox = json.loads(
        args.v2_lockbox_manifest.read_text(encoding="utf-8")
    )
    prior = _prior_identities(release, old_lockbox)
    if len(args.additional_prior_lockbox_manifest) != len(
        args.expected_additional_prior_lockbox_sha256
    ):
        raise ValueError("Additional prior manifest/hash counts differ")
    empty_release = {"splits": {"development": {}}}
    for path, expected_sha in zip(
        args.additional_prior_lockbox_manifest,
        args.expected_additional_prior_lockbox_sha256,
    ):
        if sha256_file(path) != expected_sha:
            raise ValueError(f"Additional prior lockbox mismatch: {path}")
        additional = _prior_identities(
            empty_release,
            json.loads(path.read_text(encoding="utf-8")),
        )
        for key, values in additional.items():
            prior[key].update(values)
    leakage_failures = []
    seen_groups = set()
    seen_audio = set()
    seen_scores = set()
    for row in eligible:
        group = str(row["leakage_group"])
        audio = str(row["source_hashes"]["performance_audio.wav"])
        score = str(row["source_hashes"]["verified_score.musicxml"])
        reasons = []
        if group in prior["leakage_group"]:
            reasons.append("prior_leakage_group")
        if audio in prior["audio_sha256"]:
            reasons.append("prior_audio")
        if score in prior["score_sha256"]:
            reasons.append("prior_score")
        if group in seen_groups:
            reasons.append("within_v3_leakage_group")
        if audio in seen_audio:
            reasons.append("within_v3_audio")
        if score in seen_scores:
            reasons.append("within_v3_score")
        if reasons:
            leakage_failures.append(
                {"sample": row["sample"], "reasons": reasons}
            )
        seen_groups.add(group)
        seen_audio.add(audio)
        seen_scores.add(score)
    if leakage_failures:
        raise ValueError(f"Fresh lockbox leakage detected: {leakage_failures[:3]}")

    key = v2.v1._create_or_load_key(args.lockbox_key)
    lockbox_dir = args.output_dir / "lockbox"
    input_dir = lockbox_dir / "inputs"
    public_rows = []
    private_rows = []
    for ordinal, row in enumerate(eligible):
        row_id = _row_id(str(row["sample"]), str(row["leakage_group"]))
        audio = input_dir / f"{row_id}.wav"
        score = input_dir / f"{row_id}.musicxml"
        v2.v1._atomic_copy(
            Path(row["sample_dir"]) / "performance_audio.wav", audio
        )
        v2.v1._atomic_copy(
            Path(row["sample_dir"]) / "verified_score.musicxml", score
        )
        public_rows.append(
            {
                "ordinal": ordinal,
                "row_id": row_id,
                "leakage_group": row["leakage_group"],
                "duration_sec": row["duration_sec"],
                "audio": v2.v1._artifact(audio),
                "score": v2.v1._artifact(score),
            }
        )
        private_rows.append(
            {
                "ordinal": ordinal,
                "row_id": row_id,
                "representation_version": v2.REPAIR_VERSION,
                "target_lineage_sha256": row["target_lineage_sha256"],
                "lineage": row["_lineage"],
            }
        )
    encrypted = v2._encrypt(private_rows, key)
    if v2._decrypt(encrypted, key) != private_rows:
        raise RuntimeError("ORN v3 encryption round trip failed")
    encrypted_path = lockbox_dir / "targets.aes256gcm"
    v2.v1._atomic_bytes(encrypted_path, encrypted)
    manifest_path = lockbox_dir / "manifest.json"
    v2.v1._atomic_json(
        manifest_path,
        {
            "schema_version": f"{release_schema}-lockbox-manifest",
            "created_utc": v2.v1._utc(),
            "rows": len(public_rows),
            "seed": args.seed,
            "generation_config_sha256": args.expected_config_sha256,
            "protocol_sha256": args.expected_protocol_sha256,
            "inputs": public_rows,
            "target_envelope": {
                **v2.v1._artifact(encrypted_path),
                "algorithm": "AES-256-GCM",
                "aad": v2.AAD.decode(),
                "key_sha256": hashlib.sha256(key).hexdigest(),
                "key_stored_outside_repository": True,
            },
            "target_content_public": False,
            "source_bundle_paths_public": False,
        },
    )
    audit_path = args.output_dir / "AUDIT.json"
    v2.v1._atomic_json(
        audit_path,
        {
            "schema_version": f"{release_schema}-audit",
            "rows": len(audited),
            "eligible": len(eligible),
            "failures": len(failures),
            "prior_overlap_failures": 0,
            "gold_path_coverage": 1.0,
            "exact_identity_round_trip": 1.0,
            "records": [_public_audit_row(row) for row in audited],
        },
    )
    v2.v1._atomic_json(
        freeze_path,
        {
            "schema_version": f"{release_schema}-freeze",
            "frozen_utc": v2.v1._utc(),
            "membership_rule": "all strict-audit-passing rows from fixed seed/count",
            "membership_selected_without_model_outcomes": True,
            "seed": args.seed,
            "requested_rows": args.expected_count,
            "minimum_strict_audit_rows": minimum_rows,
            "frozen_rows": len(public_rows),
            "protocol": v2.v1._artifact(args.protocol),
            "config": v2.v1._artifact(args.config),
            "audit": v2.v1._artifact(audit_path),
            "lockbox_manifest": v2.v1._artifact(manifest_path),
            "lockbox_targets_encrypted": v2.v1._artifact(encrypted_path),
            "key_sha256": hashlib.sha256(key).hexdigest(),
            "key_stored_outside_repository": True,
            "lockbox_opened": False,
            "model_development_after_freeze_allowed": True,
        },
    )
    if args.delete_source_after_freeze:
        shutil.rmtree(args.source_root)
        if args.source_root.exists():
            raise RuntimeError("Temporary source root deletion failed")
    print(
        json.dumps(
            {
                "rows": len(public_rows),
                "freeze": str(freeze_path),
                "manifest_sha256": sha256_file(manifest_path),
                "encrypted_targets_sha256": sha256_file(encrypted_path),
                "source_deleted": not args.source_root.exists(),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
