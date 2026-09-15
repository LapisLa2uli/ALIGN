from __future__ import annotations

import json
from pathlib import Path

import pytest

from alignmodel.training_resources import (
    STATUS_SCHEMA_VERSION,
    ResourceBusyError,
    claim_resource,
    release_resource,
)


def test_resource_claim_is_atomic_exclusive_and_releasable(
    tmp_path: Path,
) -> None:
    status = tmp_path / "TRAINING_RESOURCE_STATUS.json"
    lease = claim_resource(
        status,
        "gpu",
        track="fixture",
        command=("python", "train.py"),
    )
    document = json.loads(status.read_text(encoding="utf-8"))
    assert document["schema_version"] == STATUS_SCHEMA_VERSION
    assert document["leases"]["gpu"]["lease_id"] == lease
    assert not status.with_suffix(status.suffix + ".lock").exists()

    with pytest.raises(ResourceBusyError, match="gpu is owned"):
        claim_resource(status, "gpu", track="duplicate")

    release_resource(status, "gpu", "not-the-owner")
    assert json.loads(status.read_text(encoding="utf-8"))["leases"]["gpu"][
        "lease_id"
    ] == lease
    release_resource(status, "gpu", lease)
    assert "gpu" not in json.loads(status.read_text(encoding="utf-8"))["leases"]
