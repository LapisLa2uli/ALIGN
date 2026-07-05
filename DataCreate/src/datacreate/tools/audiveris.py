from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path

from datacreate.config import PipelineConfig
from datacreate.utils import resolve_binary

WINDOWS_INSTALL_ROOT = Path(r"C:\Program Files\Audiveris")
WINDOWS_EXE_NAMES = ("Audiveris.exe", "audiveris.exe")


def _exe_candidates(install_root: Path) -> list[Path]:
    return [install_root / name for name in WINDOWS_EXE_NAMES]


def _resolve_from_path(path: Path) -> Path | None:
    if path.is_file() and path.suffix.lower() == ".exe":
        return path
    if path.is_dir():
        for candidate in _exe_candidates(path):
            if candidate.exists():
                return candidate
    return None


def find_audiveris(config: PipelineConfig) -> Path | None:
    configured = config.paths.get("audiveris")
    if configured:
        resolved = _resolve_from_path(Path(configured))
        if resolved:
            return resolved

    home = config.paths.get("audiveris_home")
    if home:
        resolved = _resolve_from_path(Path(home))
        if resolved:
            return resolved

    fallbacks: list[str] = []
    if platform.system() == "Windows":
        fallbacks.extend(str(p) for p in _exe_candidates(WINDOWS_INSTALL_ROOT))
        fallbacks.extend(
            str(p)
            for p in _exe_candidates(Path(r"C:\Program Files (x86)\Audiveris"))
        )
    elif platform.system() == "Darwin":
        fallbacks = [
            "/Applications/Audiveris.app/Contents/MacOS/Audiveris",
            "/Applications/Audiveris.app/Contents/MacOS/Audiveris.exe",
        ]
    else:
        fallbacks = [
            "/usr/bin/Audiveris",
            "/usr/local/bin/Audiveris",
            "/usr/bin/audiveris",
        ]
    return resolve_binary(config, "audiveris", fallbacks)


def audiveris_install_root(config: PipelineConfig) -> Path | None:
    home = config.paths.get("audiveris_home")
    if home:
        root = Path(home)
        if root.is_dir():
            return root

    exe = find_audiveris(config)
    if exe is None:
        return None
    parent = exe.parent
    if (parent / "app" / "audiveris.jar").exists():
        return parent
    return parent


def run_omr(
    config: PipelineConfig,
    pdf_path: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> Path:
    binary = find_audiveris(config)
    if binary is None:
        raise RuntimeError(
            "Audiveris not found. Set paths.audiveris or paths.audiveris_home "
            "in config/default.yaml (e.g. C:/Program Files/Audiveris/Audiveris.exe)."
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


def open_for_manual_correction(
    score_path: Path, config: PipelineConfig, logger: logging.Logger | None = None
) -> None:
    if not config.omr.get("open_in_gui", True):
        return

    binary = find_audiveris(config)
    if binary is None:
        if logger:
            logger.warning(
                "Audiveris not found; open %s manually for correction.", score_path
            )
        return

    cmd = [str(binary), str(score_path)]
    if logger:
        logger.info("Opening score in Audiveris GUI: %s", " ".join(cmd))
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
