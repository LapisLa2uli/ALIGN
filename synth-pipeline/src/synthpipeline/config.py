from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "default.yaml"


@dataclass
class SynthConfig:
    schema_version: str = "1.1"
    paths: dict[str, Any] = field(default_factory=dict)
    audio: dict[str, Any] = field(default_factory=dict)
    musescore: dict[str, Any] = field(default_factory=dict)
    render: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, Any] = field(default_factory=dict)
    _config_path: Path | None = field(default=None, repr=False)

    @classmethod
    def load(cls, path: Path | str | None = None) -> SynthConfig:
        config_path = Path(path) if path else DEFAULT_CONFIG
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        known = {
            "schema_version",
            "paths",
            "audio",
            "musescore",
            "render",
            "generation",
            "errors",
        }
        payload = {k: v for k, v in data.items() if k in known}
        cfg = cls(**payload)
        cfg._config_path = config_path.resolve()
        return cfg

    @property
    def config_dir(self) -> Path:
        if self._config_path is not None:
            return self._config_path.parent
        return DEFAULT_CONFIG.parent

    def resolved_path(self, key: str, default: str | None = None) -> Path | None:
        value = self.paths.get(key, default)
        if not value:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.config_dir.parent / path).resolve()

    def sample_rate(self) -> int:
        return int(self.audio.get("sample_rate", 22050))

    def clarinet_program(self) -> int:
        from synthpipeline.soundfonts import CATALOG, soundfont_id

        if "clarinet_program" in self.musescore:
            return int(self.musescore["clarinet_program"])
        sid = soundfont_id(self)
        return int(CATALOG[sid]["program"])

    def midi_backend(self) -> str:
        value = str(self.render.get("midi_backend") or "music21").lower()
        if value not in {"music21", "musescore"}:
            raise ValueError(f"render.midi_backend must be music21 or musescore, got {value!r}")
        return value

    def output_root(self) -> Path:
        return self.resolved_path("output_root") or (PACKAGE_ROOT / "output")

    def to_datacreate_config(self):
        from datacreate.config import PipelineConfig

        dc_path = self.resolved_path("datacreate_config")
        if dc_path is not None and dc_path.exists():
            cfg = PipelineConfig.load(dc_path)
        else:
            cfg = PipelineConfig.load()

        from synthpipeline.soundfonts import resolve_soundfont

        musescore = self.resolved_path("musescore")
        if musescore is not None:
            cfg.paths["musescore"] = str(musescore)
        if self.musescore:
            cfg.musescore.update(self.musescore)
        preset = resolve_soundfont(self)
        cfg.paths["soundfont"] = str(preset.path)
        if "clarinet_program" not in self.musescore:
            cfg.musescore["clarinet_program"] = preset.program
        cfg.audio["sample_rate"] = self.sample_rate()
        cfg.audio["mono"] = bool(self.audio.get("mono", True))
        cfg.schema_version = self.schema_version
        return cfg
