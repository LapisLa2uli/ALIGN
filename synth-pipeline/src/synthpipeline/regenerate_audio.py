from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from datacreate.config import PipelineConfig
from synthpipeline.render import render_midi_clarinet
from synthpipeline.soundfonts import CATALOG, SOUNDFONT_ROOT
from synthpipeline.transpose_audio import _rewrite_mel, discover_bundles

AUDIO_RENDER_MARK = "soundfont_rerender"
_STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_PAIRS = (
    ("performance_audio.mid", "performance_audio.wav", "performance_score.musicxml"),
    ("reference_audio.mid", "reference_audio.wav", "verified_score.musicxml"),
)


def resolve_bundle_soundfont(meta: dict) -> tuple[Path, int]:
    candidates: list[Path] = []
    if meta.get("soundfont_path"):
        candidates.append(Path(str(meta["soundfont_path"])))
    raw = meta.get("soundfont")
    if raw:
        path = Path(str(raw))
        if path.suffix.lower() in {".sf2", ".sf3"}:
            candidates.append(path)
        spec = CATALOG.get(str(raw).lower())
        if spec:
            rel = spec.get("relpath")
            if rel:
                candidates.append(SOUNDFONT_ROOT / str(rel))
            default_win = spec.get("default_windows")
            if default_win:
                candidates.append(Path(str(default_win)))
    for cand in candidates:
        if cand.exists() and cand.is_file():
            program = meta.get("clarinet_program")
            if program is None:
                program = (meta.get("musescore") or {}).get("clarinet_program")
            if program is None:
                program = 71 if cand.suffix.lower() == ".sf3" else 0
            return cand, int(program)
    raise FileNotFoundError(f"No SoundFont found from metadata keys {raw!r}")


def _midi_keys(path: Path, limit: int = 24) -> list[int]:
    from tinysoundfont.midi import NoteOn, load

    keys: list[int] = []
    for ev in load(str(path), persistent=False):
        if isinstance(ev.action, NoteOn):
            keys.append(int(ev.action.key))
            if len(keys) >= limit:
                break
    return keys


def _xml_written_pitches(path: Path, limit: int = 24) -> list[int]:
    tree = ET.parse(path)
    root = tree.getroot()
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    pitches: list[int] = []
    for pitch in root.iter(f"{ns}pitch"):
        step = pitch.findtext(f"{ns}step")
        octave = pitch.findtext(f"{ns}octave")
        if step is None or octave is None or step not in _STEP:
            continue
        alter = pitch.findtext(f"{ns}alter") or "0"
        pitches.append(12 * (int(octave) + 1) + _STEP[step] + int(round(float(alter))))
        if len(pitches) >= limit:
            break
    return pitches


def _collapse_repeats(pitches: list[int]) -> list[int]:
    out: list[int] = []
    for pitch in pitches:
        if not out or out[-1] != pitch:
            out.append(pitch)
    return out


def midi_note_transpose(midi_path: Path, xml_path: Path, target: int = -2) -> int:
    """How many semitones to apply now so MIDI sounds at ``target`` vs the written score."""
    mid = _collapse_repeats(_midi_keys(midi_path))
    xml = _collapse_repeats(_xml_written_pitches(xml_path) if xml_path.exists() else [])
    if not mid:
        return int(target)
    if xml:
        n = min(16, len(mid), len(xml))
        diffs = sorted(mid[i] - xml[i] for i in range(n))
        current = int(round(diffs[n // 2]))
    else:
        current = 0
    return int(target) - current


def _dc_config(meta: dict, soundfont: Path, sample_rate: int) -> PipelineConfig:
    musescore = dict(meta.get("musescore") or {})
    cfg = PipelineConfig(
        paths={"soundfont": str(soundfont)},
        audio={"sample_rate": int(sample_rate), "mono": True},
        musescore={
            "synthesizer_gain_db": musescore.get("synthesizer_gain_db", -6),
            "tail_seconds": musescore.get("tail_seconds", 2.0),
            "render_chunk_size": musescore.get("render_chunk_size", 4096),
        },
    )
    return cfg


def regenerate_bundle(
    sample_dir: Path,
    semitones: int = -2,
    *,
    force: bool = False,
    sample_rate: int = 22050,
) -> str:
    sample_dir = Path(sample_dir)
    meta_path = sample_dir / "metadata.json"
    if not (sample_dir / "performance_audio.mid").exists() or not (
        sample_dir / "reference_audio.mid"
    ).exists():
        return "skip_missing"
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not force and meta.get("audio_render") == AUDIO_RENDER_MARK:
            return "skip_done"

    soundfont, program = resolve_bundle_soundfont(meta)
    cfg = _dc_config(meta, soundfont, sample_rate)
    logger = logging.getLogger("synthpipeline.rerender")
    logger.setLevel(logging.WARNING)

    for midi_name, wav_name, xml_name in _PAIRS:
        midi_path = sample_dir / midi_name
        wav_path = sample_dir / wav_name
        shift = midi_note_transpose(midi_path, sample_dir / xml_name, semitones)
        render_midi_clarinet(
            midi_path,
            wav_path,
            cfg,
            logger,
            clarinet_program=program,
            note_transpose=shift,
        )
        audio, _ = _load_mono(wav_path, sample_rate)
        _rewrite_mel(audio, sample_rate, sample_dir / f"{wav_name.split('_')[0]}_mel.npy")

    meta["sounding_transpose"] = int(semitones)
    meta["audio_render"] = AUDIO_RENDER_MARK
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return "converted"


def _load_mono(path: Path, sample_rate: int):
    from datacreate.audio_utils import load_audio

    return load_audio(path, sample_rate, mono=True)


def _one(payload: tuple[str, int, bool, int]) -> str:
    sample, semis, force, sr = payload
    try:
        return regenerate_bundle(Path(sample), semis, force=force, sample_rate=sr)
    except Exception as exc:
        print(f"failed {sample}: {type(exc).__name__}: {exc}", flush=True)
        return "failed"


def regenerate_root(
    root: Path,
    semitones: int = -2,
    *,
    force: bool = False,
    workers: int = 8,
    sample_rate: int = 22050,
) -> dict[str, int]:
    dirs = discover_bundles(root)
    counts = {
        "converted": 0,
        "skip_done": 0,
        "skip_missing": 0,
        "failed": 0,
        "n_bundles": len(dirs),
    }
    jobs = [(str(p), int(semitones), bool(force), int(sample_rate)) for p in dirs]
    workers = max(1, int(workers))
    if workers == 1:
        statuses = [_one(job) for job in jobs]
    else:
        statuses = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_one, job) for job in jobs]
            for i, fut in enumerate(as_completed(futs), start=1):
                statuses.append(fut.result())
                if i % 50 == 0 or i == len(futs):
                    print(f"  {root}: {i}/{len(futs)}", flush=True)
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return counts
