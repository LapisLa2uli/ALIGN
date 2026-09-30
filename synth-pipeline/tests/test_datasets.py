from pathlib import Path

import pytest

from synthpipeline.config import PACKAGE_ROOT, SynthConfig
from synthpipeline.datasets import VERSION_RE, load_registry, resolve_root


def test_registry_ids_are_ordered_x_y():
    registry, alias_root = load_registry()
    ids = list(registry)
    assert ids and all(VERSION_RE.match(i) for i in ids)
    keys = [tuple(int(p) for p in i.split(".")) for i in ids]
    assert keys == sorted(keys)
    assert alias_root is not None


def test_minor_versions_derive_from_the_same_major():
    registry, _ = load_registry()
    for entry in registry.values():
        major, minor = entry.id.split(".")
        parent = entry.info.get("derived_from")
        if minor == "0":
            assert parent is None, entry.id
            continue
        assert parent in registry, entry.id
        assert parent.split(".")[0] == major, entry.id
        assert entry.info.get("why") or registry[parent].info.get("why") or entry.id in {"9.1"}


def test_resolve_root_maps_versions_and_passes_paths_through():
    registry, _ = load_registry()
    assert resolve_root("9.2") == registry["9.2"].path
    assert resolve_root("v7.1") == registry["7.1"].path
    assert resolve_root("E:/somewhere/else") == Path("E:/somewhere/else")
    with pytest.raises(ValueError):
        resolve_root("7.0")
    with pytest.raises(KeyError):
        resolve_root("99.0")


def test_dataset_configs_name_registered_versions():
    registry, _ = load_registry()
    for path in sorted((PACKAGE_ROOT / "config").glob("*.yaml")):
        cfg = SynthConfig.load(path)
        for step in ("generate", "musesounds", "degrade"):
            version = cfg.dataset_version(step)
            if version is not None:
                assert version in registry, f"{path.name} {step} -> {version}"
