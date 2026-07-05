from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datacreate.config import PipelineConfig


def setup_sample_logger(sample_dir: Path, name: str = "datacreate") -> logging.Logger:
    sample_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{name}.{sample_dir.name}")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.FileHandler(sample_dir / "pipeline.log", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
    return logger


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_binary(config: PipelineConfig, key: str, fallbacks: list[str]) -> Path | None:
    configured = config.paths.get(key)
    if configured:
        path = Path(configured)
        if path.exists():
            return path
    for candidate in fallbacks:
        found = Path(candidate)
        if found.exists():
            return found
    return None
