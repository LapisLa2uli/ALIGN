from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path

from datacreate.config import PipelineConfig
from datacreate.utils import resolve_binary


def find_musescore(config: PipelineConfig) -> Path | None:
    fallbacks = []
    if platform.system() == "Windows":
        fallbacks = [
            r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
            r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
        ]
    elif platform.system() == "Darwin":
        fallbacks = ["/Applications/MuseScore 4.app/Contents/MacOS/mscore"]
    else:
        fallbacks = ["/usr/bin/mscore", "/usr/local/bin/mscore"]
    return resolve_binary(config, "musescore", fallbacks)


def parse_version(text: str) -> tuple[int, ...]:
    digits: list[int] = []
    for part in text.replace("_", ".").split("."):
        chunk = "".join(ch for ch in part if ch.isdigit())
        if chunk:
            digits.append(int(chunk))
    return tuple(digits or [0])


def check_musescore_version(config: PipelineConfig, logger: logging.Logger) -> str | None:
    binary = find_musescore(config)
    if binary is None:
        logger.warning("MuseScore binary not found; configure paths.musescore in config")
        return None
    result = subprocess.run(
        [str(binary), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    version_text = (result.stdout or result.stderr or "").strip()
    logger.info("MuseScore version output: %s", version_text)
    min_version = parse_version(str(config.musescore.get("min_version", "4.2.0")))
    current = parse_version(version_text)
    if current and current < min_version:
        logger.warning(
            "MuseScore %s is below recommended %s; CLI export may produce silent audio",
            ".".join(map(str, current)),
            ".".join(map(str, min_version)),
        )
    return version_text


def render_score_to_wav(
    config: PipelineConfig,
    score_path: Path,
    output_wav: Path,
    logger: logging.Logger,
) -> None:
    binary = find_musescore(config)
    if binary is None:
        raise RuntimeError(
            "MuseScore not configured. Set paths.musescore in config/default.yaml"
        )

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "-o", str(output_wav), str(score_path)]
    soundfont = config.paths.get("soundfont")
    if soundfont:
        cmd.extend(["-S", str(soundfont)])

    if platform.system() == "Linux":
        cmd = ["xvfb-run", "-a"] + cmd

    logger.info("Running MuseScore: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    logger.info("MuseScore stdout:\n%s", result.stdout)
    if result.stderr:
        logger.warning("MuseScore stderr:\n%s", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"MuseScore export failed with code {result.returncode}")
    if not output_wav.exists():
        raise RuntimeError(f"MuseScore did not produce {output_wav}")
