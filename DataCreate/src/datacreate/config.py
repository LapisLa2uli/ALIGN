from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineConfig:
    schema_version: str = "1.1"
    paths: dict[str, Any] = field(default_factory=dict)
    audio: dict[str, Any] = field(default_factory=dict)
    mel: dict[str, Any] = field(default_factory=dict)
    alignment: dict[str, Any] = field(default_factory=dict)
    taxonomy: list[str] = field(default_factory=list)
    musescore: dict[str, Any] = field(default_factory=dict)
    omr: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] = field(default_factory=dict)
    synthetic: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str | None = None) -> PipelineConfig:
        if path is None:
            path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(**data)

    def path(self, key: str, default: str | None = None) -> Path | None:
        value = self.paths.get(key, default)
        return Path(value) if value else None

    def resolved_path(self, key: str, default: str | None = None) -> Path | None:
        path = self.path(key, default)
        if path is None:
            return None
        if path.is_absolute():
            return path
        base = Path(__file__).resolve().parents[2]
        return (base / path).resolve()

    def sample_rate(self) -> int:
        return int(self.audio.get("sample_rate", 22050))

    def taxonomy_set(self) -> set[str]:
        return set(self.taxonomy)
