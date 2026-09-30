"""Dataset version registry (``datasets.yaml``): x.y IDs mapped to bundle folders."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from synthpipeline.config import PACKAGE_ROOT

REGISTRY_PATH = PACKAGE_ROOT / "datasets.yaml"
VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)$")
LINKED_STATUSES = ("on_disk", "in_progress")


@dataclass
class DatasetVersion:
    id: str
    name: str
    path: Path
    status: str
    info: dict = field(default_factory=dict)

    @property
    def alias(self) -> str:
        return f"{self.id}_{self.name}"


def _sort_key(version_id: str) -> tuple[int, int]:
    major, minor = VERSION_RE.match(version_id).groups()
    return int(major), int(minor)


def load_registry(path: Path = REGISTRY_PATH) -> tuple[dict[str, DatasetVersion], Path | None]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    base = Path(path).resolve().parent
    out: dict[str, DatasetVersion] = {}
    for raw_id, entry in (doc.get("datasets") or {}).items():
        version_id = str(raw_id)
        if not VERSION_RE.match(version_id):
            raise ValueError(f"dataset id {version_id!r} is not x.y")
        target = Path(str(entry["path"]))
        if not target.is_absolute():
            target = (base / target).resolve()
        out[version_id] = DatasetVersion(
            id=version_id,
            name=str(entry["name"]),
            path=target,
            status=str(entry.get("status", "on_disk")),
            info=dict(entry),
        )
    ordered = dict(sorted(out.items(), key=lambda kv: _sort_key(kv[0])))
    alias_root = doc.get("alias_root")
    return ordered, Path(str(alias_root)) if alias_root else None


def resolve_root(value: str | Path) -> Path:
    """``"9.2"`` / ``"v9.2"`` -> the registered folder; anything else is a plain path."""
    text = str(value)
    match = VERSION_RE.match(text)
    if not match:
        return Path(text)
    version_id = f"{int(match.group(1))}.{int(match.group(2))}"
    registry, _ = load_registry()
    if version_id not in registry:
        raise KeyError(f"dataset {version_id} is not in {REGISTRY_PATH}")
    entry = registry[version_id]
    if entry.status == "superseded_in_place":
        raise ValueError(
            f"dataset {version_id} was overwritten in place by a later minor version; "
            f"its folder now holds a newer revision ({entry.path})"
        )
    return entry.path


def _make_link(alias: Path, target: Path) -> Path:
    """Directory junction/symlink; a Windows shortcut where the volume has no links (exFAT)."""
    alias.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        alias.symlink_to(target, target_is_directory=True)
        return alias
    proc = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return alias
    lnk = alias.with_name(alias.name + ".lnk")
    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:ALIGN_LNK); "
        "$s.TargetPath = $env:ALIGN_TARGET; $s.Save()"
    )
    env = dict(os.environ, ALIGN_LNK=str(lnk), ALIGN_TARGET=str(target))
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return lnk


def link_aliases(alias_root: Path | None = None, *, dry_run: bool = False) -> list[tuple[str, Path, Path, str]]:
    """Create ``<alias_root>/<x.y>_<name>`` links for every version on disk, plus INDEX.txt."""
    registry, default_root = load_registry()
    root = Path(alias_root or default_root or "")
    if not str(root):
        raise ValueError("no alias_root configured")
    rows = []
    for entry in registry.values():
        if entry.status not in LINKED_STATUSES:
            continue
        alias = root / entry.alias
        shortcut = alias.with_name(alias.name + ".lnk")
        if not entry.path.is_dir():
            action = "target_missing"
        elif alias.exists() or shortcut.exists():
            action = "exists"
            alias = alias if alias.exists() else shortcut
        else:
            action = "would_link" if dry_run else "linked"
            if not dry_run:
                alias = _make_link(alias, entry.path)
        rows.append((entry.id, alias, entry.path, action))
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
        lines = [
            "ALIGN synthetic datasets by version (source: synth-pipeline/datasets.yaml).",
            "Folders keep historical names; use the version ID in commands, e.g. --root 9.2.",
            "",
        ]
        for entry in registry.values():
            lines.append(f"{entry.id:>5}  {entry.status:20s} {entry.name:28s} {entry.path}")
        (root / "INDEX.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows
