from __future__ import annotations

import logging
import platform
import subprocess
import webbrowser
from pathlib import Path

from datacreate.config import PipelineConfig
from datacreate.utils import resolve_binary


def find_audiveris(config: PipelineConfig) -> Path | None:
    fallbacks = []
    if platform.system() == "Windows":
        fallbacks = [
            r"C:\Program Files\Audiveris\Audiveris.bat",
            r"C:\Program Files\Audiveris\bin\Audiveris.bat",
        ]
    elif platform.system() == "Darwin":
        fallbacks = ["/Applications/Audiveris.app/Contents/MacOS/Audiveris"]
    else:
        fallbacks = ["/usr/bin/Audiveris", "/usr/local/bin/Audiveris"]
    return resolve_binary(config, "audiveris", fallbacks)


def run_omr(
    config: PipelineConfig,
    pdf_path: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> Path:
    binary = find_audiveris(config)
    if binary is None:
        raise RuntimeError(
            "Audiveris not configured. Set paths.audiveris in config/default.yaml"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "-batch", "-export", "-output", str(output_dir), str(pdf_path)]
    logger.info("Running Audiveris: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    logger.info("Audiveris stdout:\n%s", result.stdout)
    if result.stderr:
        logger.warning("Audiveris stderr:\n%s", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Audiveris failed with code {result.returncode}")

    candidates = sorted(output_dir.rglob("*.mxl")) + sorted(output_dir.rglob("*.musicxml"))
    if not candidates:
        raise RuntimeError(f"Audiveris produced no MusicXML in {output_dir}")
    draft = candidates[0]
    logger.info("OMR draft score: %s", draft)
    return draft


def open_for_manual_correction(score_path: Path, config: PipelineConfig) -> None:
    if not config.omr.get("open_in_gui", True):
        return
    if platform.system() == "Windows":
        subprocess.run(["start", "", str(score_path)], shell=True, check=False)
    elif platform.system() == "Darwin":
        subprocess.run(["open", str(score_path)], check=False)
    else:
        webbrowser.open(score_path.as_uri())
