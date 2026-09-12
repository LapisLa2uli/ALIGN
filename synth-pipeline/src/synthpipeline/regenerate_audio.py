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

AUDIO_RENDER_MARK = "oscillator_v1"
BARE_RENDER_MARK = "oscillator_v1_bare"
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
    try:
        cfg = PipelineConfig.load()
    except Exception:
        cfg = PipelineConfig()
    cfg.paths["soundfont"] = str(soundfont)
    cfg.audio["sample_rate"] = int(sample_rate)
    cfg.audio["mono"] = True
    cfg.musescore.update(
        {
            "synthesizer_gain_db": musescore.get("synthesizer_gain_db", -6),
            "tail_seconds": musescore.get("tail_seconds", 2.0),
            "render_chunk_size": musescore.get("render_chunk_size", 4096),
        }
    )
    if meta.get("mel_params"):
        cfg.mel.update(meta["mel_params"])
    if meta.get("alignment_params"):
        cfg.alignment.update(meta["alignment_params"])
    return cfg


def _pitch_bends_from_labels(sample_dir: Path) -> list[dict]:
    path = sample_dir / "labels.json"
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    bends: list[dict] = []
    for lab in doc.get("labels") or []:
        if lab.get("type") != "intonation_error":
            continue
        cents = lab.get("deviation_cents")
        if cents is None:
            continue
        bends.append(
            {
                "cents": float(cents),
                "ql_start": float(lab["start_time"]),
                "ql_end": float(lab["end_time"]),
            }
        )
    return bends


def _rewrite_scores_and_midi(sample_dir: Path, semitones: int, logger: logging.Logger) -> None:
    from music21 import converter

    from synthpipeline.midi_player import strip_ornaments
    from synthpipeline.render import export_score_to_midi_music21
    from synthpipeline.scoregen import write_musicxml

    for midi_name, _wav_name, xml_name in _PAIRS:
        xml_path = sample_dir / xml_name
        if not xml_path.exists():
            continue
        score = converter.parse(str(xml_path))
        strip_ornaments(score)
        write_musicxml(score, xml_path)
        export_score_to_midi_music21(
            converter.parse(str(xml_path)),
            xml_path,
            sample_dir / midi_name,
            logger,
            sounding_transpose=int(semitones),
        )


def regenerate_bundle(
    sample_dir: Path,
    semitones: int = -2,
    *,
    force: bool = False,
    sample_rate: int = 22050,
    strip_ornaments: bool = False,
) -> str:
    sample_dir = Path(sample_dir)
    meta_path = sample_dir / "metadata.json"
    if not (sample_dir / "performance_audio.mid").exists() or not (
        sample_dir / "reference_audio.mid"
    ).exists():
        return "skip_missing"
    meta: dict = {}
    mark = BARE_RENDER_MARK if strip_ornaments else AUDIO_RENDER_MARK
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not force and meta.get("audio_render") == mark:
            return "skip_done"

    try:
        soundfont, program = resolve_bundle_soundfont(meta)
    except FileNotFoundError:
        soundfont, program = Path("unused.sf2"), int(meta.get("clarinet_program") or 0)
    cfg = _dc_config(meta, soundfont, sample_rate)
    logger = logging.getLogger("synthpipeline.rerender")
    logger.setLevel(logging.WARNING)

    if strip_ornaments:
        _rewrite_scores_and_midi(sample_dir, semitones, logger)

    bends = _pitch_bends_from_labels(sample_dir) if strip_ornaments else []
    for midi_name, wav_name, xml_name in _PAIRS:
        midi_path = sample_dir / midi_name
        wav_path = sample_dir / wav_name
        shift = 0 if strip_ornaments else midi_note_transpose(
            midi_path, sample_dir / xml_name, semitones
        )
        render_midi_clarinet(
            midi_path,
            wav_path,
            cfg,
            logger,
            clarinet_program=program,
            note_transpose=shift,
            pitch_bends=bends if midi_name.startswith("performance") else None,
            bpm=60.0 if bends and midi_name.startswith("performance") else None,
        )
        audio, _ = _load_mono(wav_path, sample_rate)
        _rewrite_mel(audio, sample_rate, sample_dir / f"{wav_name.split('_')[0]}_mel.npy")

    from synthpipeline.pitch_convention import (
        annotate_pitch_metadata,
        infer_midi_pitch_space,
    )

    inferred_space, _offset = infer_midi_pitch_space(sample_dir, meta)
    meta["sounding_transpose"] = int(semitones)
    meta.update(
        annotate_pitch_metadata(
            meta,
            midi_space=inferred_space,
            audio_space="sounding",
            effective_audio_shift=-int(semitones),
            audio_render=mark,
        )
    )
    if strip_ornaments:
        meta["ornaments_stripped"] = True
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return "converted"


def _load_mono(path: Path, sample_rate: int):
    from datacreate.audio_utils import load_audio

    return load_audio(path, sample_rate, mono=True)


def _one(payload: tuple[str, int, bool, int, bool]) -> str:
    sample, semis, force, sr, strip = payload
    try:
        return regenerate_bundle(
            Path(sample),
            semis,
            force=force,
            sample_rate=sr,
            strip_ornaments=strip,
        )
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
    strip_ornaments: bool = False,
) -> dict[str, int]:
    dirs = discover_bundles(root)
    counts = {
        "converted": 0,
        "skip_done": 0,
        "skip_missing": 0,
        "failed": 0,
        "n_bundles": len(dirs),
    }
    jobs = [
        (str(p), int(semitones), bool(force), int(sample_rate), bool(strip_ornaments))
        for p in dirs
    ]
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
