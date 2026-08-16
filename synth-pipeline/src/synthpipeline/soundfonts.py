from __future__ import annotations

import logging
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from typing import TYPE_CHECKING

from synthpipeline.config import PACKAGE_ROOT

if TYPE_CHECKING:
    from synthpipeline.config import SynthConfig

SOUNDFONT_ROOT = PACKAGE_ROOT / "soundfonts"

# Clarinet-only banks use preset 0. MS Basic is a GM bank (clarinet = 71).
CATALOG: dict[str, dict] = {
    "freepats": {
        "label": "FreePats Clarinet (recorded, CC0)",
        "filename": "Clarinet-20190818.sf2",
        "relpath": "freepats/Clarinet-20190818.sf2",
        "program": 0,
        "url": "http://freepats.zenvoid.org/Reed/Clarinet1/Clarinet-SF2-20190818.tar.xz",
        "archive": True,
        "license": "CC0 1.0",
        "source": "http://freepats.zenvoid.org/Reed/clarinet.html",
    },
    "u220": {
        "label": "Roland U220 Winds clarinet",
        "filename": "u220_clarinet.sf2",
        "relpath": "u220/u220_clarinet.sf2",
        "program": 0,
        "url": "https://www.polyphone.io/en/soundfonts/reeds/219-roland-u220-winds-clarinet",
        "archive": False,
        "manual": True,
        "license": "check Polyphone page",
        "source": "https://www.polyphone.io/en/soundfonts/reeds/219-roland-u220-winds-clarinet",
    },
    "mcb": {
        "label": "Maestro Clarinet Base (Mats Helgesson)",
        "filename": "mcb.sf2",
        "relpath": "mcb/mcb.sf2",
        "program": 0,
        "url": "https://musical-artifacts.com/artifacts/2135",
        "archive": False,
        "manual": True,
        "license": "CC BY 3.0 (attribute Mats Helgesson)",
        "source": "https://musical-artifacts.com/artifacts/2135",
    },
    "msbasic": {
        "label": "MuseScore MS Basic (GM clarinet 71)",
        "filename": "MS Basic.sf3",
        "relpath": None,
        "program": 71,
        "url": None,
        "archive": False,
        "system_path_keys": ("soundfont",),
        "default_windows": r"C:\Program Files\MuseScore 4\sound\MS Basic.sf3",
        "license": "MuseScore",
        "source": "MuseScore 4 install",
    },
}

SOUNDFONT_IDS = tuple(CATALOG.keys())


@dataclass(frozen=True)
class SoundfontPreset:
    id: str
    label: str
    path: Path
    program: int
    license: str
    source: str


def soundfont_id(config: SynthConfig, override: str | None = None) -> str:
    value = (override or config.render.get("soundfont") or "freepats").lower()
    if value not in CATALOG:
        raise ValueError(
            f"Unknown soundfont {value!r}. Choose one of: {', '.join(SOUNDFONT_IDS)}"
        )
    return value


def resolve_soundfont(config: SynthConfig, override: str | None = None) -> SoundfontPreset:
    sid = soundfont_id(config, override)
    spec = CATALOG[sid]
    path = _resolve_path(config, sid, spec)
    if path is None or not path.exists():
        raise FileNotFoundError(_missing_message(sid, spec))
    return SoundfontPreset(
        id=sid,
        label=str(spec["label"]),
        path=path,
        program=int(spec["program"]),
        license=str(spec.get("license") or ""),
        source=str(spec.get("source") or ""),
    )


def list_soundfonts(config: SynthConfig) -> list[dict]:
    rows = []
    for sid, spec in CATALOG.items():
        path = _resolve_path(config, sid, spec)
        rows.append(
            {
                "id": sid,
                "label": spec["label"],
                "program": spec["program"],
                "path": str(path) if path else "",
                "installed": bool(path and path.exists()),
                "source": spec.get("source"),
            }
        )
    return rows


def fetch_soundfonts(
    ids: list[str] | None,
    logger: logging.Logger,
) -> list[Path]:
    wanted = ids or [sid for sid, spec in CATALOG.items() if spec.get("url") and not spec.get("manual")]
    fetched: list[Path] = []
    for sid in wanted:
        if sid not in CATALOG:
            raise ValueError(f"Unknown soundfont {sid!r}")
        spec = CATALOG[sid]
        dest = SOUNDFONT_ROOT / spec["relpath"] if spec.get("relpath") else None
        if dest is not None and dest.exists():
            logger.info("Already present: %s (%s)", sid, dest)
            fetched.append(dest)
            continue
        if spec.get("manual") or not spec.get("url"):
            logger.warning(_missing_message(sid, spec))
            continue
        dest = _download(sid, spec, logger)
        fetched.append(dest)
    return fetched


def _resolve_path(config: SynthConfig, sid: str, spec: dict) -> Path | None:
    if spec.get("relpath"):
        return SOUNDFONT_ROOT / spec["relpath"]
    for key in spec.get("system_path_keys") or ():
        path = config.resolved_path(key)
        if path is not None:
            return path
    default_win = spec.get("default_windows")
    if default_win:
        path = Path(default_win)
        if path.exists():
            return path
    return None


def _download(sid: str, spec: dict, logger: logging.Logger) -> Path:
    dest = SOUNDFONT_ROOT / spec["relpath"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = spec["url"]
    logger.info("Downloading %s from %s", sid, url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        if spec.get("archive"):
            _extract_sf2(tmp, dest, logger)
            tmp.unlink(missing_ok=True)
        else:
            if not _looks_like_soundfont(tmp):
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Download for {sid} was not an SF2/SF3 file. "
                    f"Download it manually from {spec.get('source')}"
                )
            tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    if not dest.exists() or dest.stat().st_size < 1000:
        raise RuntimeError(f"Failed to install soundfont {sid} at {dest}")
    logger.info("Installed %s -> %s (%d bytes)", sid, dest, dest.stat().st_size)
    return dest


def _extract_sf2(archive: Path, dest: Path, logger: logging.Logger) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    extract_dir = dest.parent / "_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive) as tf:
            tf.extractall(extract_dir)
        matches = list(extract_dir.rglob("*.sf2")) + list(extract_dir.rglob("*.sf3"))
        if not matches:
            raise RuntimeError(f"No SF2/SF3 inside {archive.name}")
        shutil.copy2(matches[0], dest)
        logger.info("Extracted %s", matches[0].name)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def _looks_like_soundfont(path: Path) -> bool:
    if path.stat().st_size < 1000:
        return False
    header = path.read_bytes()[:4]
    return header in {b"RIFF", b"sfbk"}


def _missing_message(sid: str, spec: dict) -> str:
    dest = SOUNDFONT_ROOT / spec["relpath"] if spec.get("relpath") else spec.get("filename")
    source = spec.get("source") or spec.get("url")
    return (
        f"SoundFont '{sid}' is not installed. "
        f"Download {spec.get('filename')} from {source} "
        f"and save it as {dest}. "
        f"Or run: synth-pipeline fetch-soundfonts --soundfont {sid}"
    )
