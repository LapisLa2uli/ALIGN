"""Render ALIGN MusicXML through MuseScore's Muse Sounds clarinet library.

MuseSamplerCoreLib only initializes inside a Muse Hub SDK host, so audio is
exported with MuseScore Studio ``--sound-profile MuseSounds`` (the Muse
Woodwinds Bb clarinet mapping), then ffmpeg converts MP3 to the bundle WAV.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

MUSESOUNDS_RENDER_MARK = "musesounds_v1"
DEFAULT_MUSESCORE = Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe")
DEFAULT_FFMPEG = Path(r"C:\Users\Hank\.conda\envs\MusicEval\Library\bin\ffmpeg.exe")
INSTRUMENT_SOUND = "wind.reed.clarinet.bflat"
INSTRUMENT_ID = "wind.reed.clarinet"
PART_NAME = "Clarinet in Bb"
PART_ABBR = "Bb Cl."
_STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_PC_TO_STEP = {
    0: ("C", 0),
    1: ("D", -1),
    2: ("D", 0),
    3: ("E", -1),
    4: ("E", 0),
    5: ("F", 0),
    6: ("G", -1),
    7: ("G", 0),
    8: ("A", -1),
    9: ("A", 0),
    10: ("B", -1),
    11: ("B", 0),
}


def find_musescore() -> Path:
    configured = os.environ.get("MUSESCORE_PATH")
    if configured:
        path = Path(configured)
        if path.is_file():
            return path
    if DEFAULT_MUSESCORE.is_file():
        return DEFAULT_MUSESCORE
    found = shutil.which("MuseScore4") or shutil.which("mscore")
    if found:
        return Path(found)
    raise FileNotFoundError("MuseScore 4 not found; set MUSESCORE_PATH")


def find_ffmpeg() -> Path:
    configured = os.environ.get("FFMPEG_PATH")
    if configured:
        path = Path(configured)
        if path.is_file():
            return path
    if DEFAULT_FFMPEG.is_file():
        return DEFAULT_FFMPEG
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    raise FileNotFoundError("ffmpeg not found; set FFMPEG_PATH")


def _win_junction(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
        check=True,
        capture_output=True,
        text=True,
    )


def musescore_worker_env(work_home: Path) -> dict[str, str]:
    """Isolated USERPROFILE so several MuseScore CLIs can run at once.

    MuseSampler and Muse Hub stay shared via directory junctions; MuseScore's
    own settings are copied so each process has a private session.
    """
    work_home = Path(work_home)
    local = work_home / "AppData" / "Local"
    roaming = work_home / "AppData" / "Roaming"
    local.mkdir(parents=True, exist_ok=True)
    roaming.mkdir(parents=True, exist_ok=True)
    real_local = Path(os.environ.get("LOCALAPPDATA") or "")
    if real_local.is_dir():
        for name in ("MuseSampler", "Muse Hub"):
            src = real_local / name
            if src.exists():
                _win_junction(src, local / name)
        src_ms = real_local / "MuseScore"
        dst_ms = local / "MuseScore"
        if src_ms.exists() and not dst_ms.exists():
            shutil.copytree(
                src_ms,
                dst_ms,
                ignore=shutil.ignore_patterns("cache", "cloud_scores", "logs"),
            )
    env = os.environ.copy()
    env["USERPROFILE"] = str(work_home)
    env["HOME"] = str(work_home)
    env["LOCALAPPDATA"] = str(local)
    env["APPDATA"] = str(roaming)
    env.setdefault("QT_LOGGING_RULES", "*.debug=false")
    return env


def _local_tag(tag: str) -> str:
    return tag.split("}", 1)[-1]


def prepare_bb_clarinet_xml(
    src: Path,
    dest: Path,
    *,
    sounding_shift: int = -2,
) -> Path:
    """Rewrite a copy so MuseScore maps the part to Muse Woodwinds Bb clarinet.

    Bundle MusicXML on disk is left unchanged. MuseScore plays these MusicXML
    pitches as concert pitch even with a Bb-clarinet mapping, so the export
    copy is shifted by ``sounding_shift`` (written C → sounding Bb).
    """
    tree = ET.parse(src)
    root = tree.getroot()
    for el in root.iter():
        tag = _local_tag(el.tag)
        if tag == "part-name":
            el.text = PART_NAME
        elif tag == "part-abbreviation":
            el.text = PART_ABBR
        elif tag == "instrument-name":
            el.text = PART_NAME
        elif tag == "instrument-abbreviation":
            el.text = PART_ABBR
        elif tag == "score-instrument":
            _ensure_child(el, "instrument-sound", INSTRUMENT_SOUND)
            _ensure_child(el, "instrument-id", INSTRUMENT_ID)
    if sounding_shift:
        _shift_xml_pitches(root, int(sounding_shift))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dest, encoding="utf-8", xml_declaration=True)
    return dest


def _shift_xml_pitches(root: ET.Element, semis: int) -> None:
    for pitch in root.iter():
        if _local_tag(pitch.tag) != "pitch":
            continue
        step_el = alter_el = octave_el = None
        for child in list(pitch):
            tag = _local_tag(child.tag)
            if tag == "step":
                step_el = child
            elif tag == "alter":
                alter_el = child
            elif tag == "octave":
                octave_el = child
        if step_el is None or step_el.text is None or octave_el is None or octave_el.text is None:
            continue
        step = step_el.text.strip()
        if step not in _STEP:
            continue
        alter = float(alter_el.text) if alter_el is not None and alter_el.text else 0.0
        midi = 12 * (int(octave_el.text) + 1) + _STEP[step] + int(round(alter)) + int(semis)
        midi = max(0, min(127, midi))
        pc = midi % 12
        new_step, new_alter = _PC_TO_STEP[pc]
        new_oct = midi // 12 - 1
        step_el.text = new_step
        octave_el.text = str(new_oct)
        if new_alter:
            if alter_el is None:
                alter_el = ET.SubElement(pitch, "alter")
            alter_el.text = str(new_alter)
        elif alter_el is not None:
            pitch.remove(alter_el)


def _ensure_child(parent: ET.Element, tag: str, text: str) -> None:
    for child in list(parent):
        if _local_tag(child.tag) == tag:
            child.text = text
            return
    ET.SubElement(parent, tag).text = text


def export_musesounds_mp3(
    xml_path: Path,
    mp3_path: Path,
    *,
    musescore: Path | None = None,
    bitrate: int = 320,
    timeout_sec: float = 180.0,
    env: dict[str, str] | None = None,
) -> dict:
    binary = musescore or find_musescore()
    mp3_path = Path(mp3_path)
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    mp3_path.unlink(missing_ok=True)
    cmd = [
        str(binary),
        "--sound-profile",
        "MuseSounds",
        "-f",
        "-b",
        str(int(bitrate)),
        "-o",
        str(mp3_path),
        str(Path(xml_path)),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=float(timeout_sec),
        env=env,
    )
    info = {
        "cmd": cmd,
        "exit": proc.returncode,
        "bytes": mp3_path.stat().st_size if mp3_path.is_file() else 0,
        "stderr_tail": (proc.stderr or "")[-800:],
    }
    if proc.returncode != 0 or not mp3_path.is_file() or mp3_path.stat().st_size < 256:
        raise RuntimeError(
            f"MuseSounds export failed exit={proc.returncode} "
            f"bytes={info['bytes']}: {info['stderr_tail']}"
        )
    return info


def export_musesounds_job(
    jobs: list[tuple[Path, Path]],
    *,
    musescore: Path | None = None,
    bitrate: int = 320,
    timeout_sec: float = 0.0,
    work_dir: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    """One MuseScore process, many MP3s. ``jobs`` is ``(in_xml, out_mp3)``."""
    if not jobs:
        return {"exit": 0, "n": 0}
    binary = musescore or find_musescore()
    payload = [{"in": str(src), "out": str(dst)} for src, dst in jobs]
    for _src, dst in jobs:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).unlink(missing_ok=True)
    tmp = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="musesounds_job_"))
    tmp.mkdir(parents=True, exist_ok=True)
    job_path = tmp / "job.json"
    job_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    cmd = [
        str(binary),
        "--sound-profile",
        "MuseSounds",
        "-f",
        "-b",
        str(int(bitrate)),
        "-j",
        str(job_path),
    ]
    kwargs: dict = {
        "capture_output": True,
        "text": True,
        "check": False,
    }
    if env is not None:
        kwargs["env"] = env
    if timeout_sec and timeout_sec > 0:
        kwargs["timeout"] = float(timeout_sec)
    proc = subprocess.run(cmd, **kwargs)
    missing = [str(dst) for _src, dst in jobs if not Path(dst).is_file() or Path(dst).stat().st_size < 256]
    info = {
        "cmd": cmd,
        "exit": proc.returncode,
        "n": len(jobs),
        "missing": missing,
        "stderr_tail": (proc.stderr or "")[-800:],
        "job_path": str(job_path),
    }
    if proc.returncode != 0 or missing:
        raise RuntimeError(
            f"MuseSounds job failed exit={proc.returncode} missing={len(missing)}: "
            f"{info['stderr_tail']}"
        )
    return info


def mp3_to_wav(
    mp3_path: Path,
    wav_path: Path,
    *,
    sample_rate: int = 22050,
    ffmpeg: Path | None = None,
) -> Path:
    binary = ffmpeg or find_ffmpeg()
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "-y",
        "-i",
        str(mp3_path),
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not wav_path.is_file() or wav_path.stat().st_size < 256:
        raise RuntimeError(
            f"ffmpeg failed exit={proc.returncode}: {(proc.stderr or '')[-500:]}"
        )
    return wav_path
