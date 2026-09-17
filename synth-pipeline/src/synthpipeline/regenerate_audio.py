from __future__ import annotations

import json
import logging
import shutil
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from datacreate.config import PipelineConfig
from synthpipeline.render import render_midi_clarinet
from synthpipeline.soundfonts import CATALOG, SOUNDFONT_ROOT
from synthpipeline.transpose_audio import _rewrite_mel, discover_bundles

AUDIO_RENDER_MARK = "oscillator_v1"
BARE_RENDER_MARK = "oscillator_v1_bare"
MUSESOUNDS_RENDER_MARK = "musesounds_v1"
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
    backend: str = "soundfont",
) -> str:
    sample_dir = Path(sample_dir)
    if str(backend).lower() == "musesounds":
        if strip_ornaments:
            raise ValueError("Muse Sounds re-render keeps ornaments; do not pass strip_ornaments")
        return regenerate_bundle_musesounds(
            sample_dir, semitones=semitones, force=force, sample_rate=sample_rate
        )
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

    # Intonation labels are wall-clock seconds; at the synthetic 60 BPM
    # regeneration clock they map directly to ql positions. Preserve them in
    # both ordinary and ornament-stripped rerenders.
    bends = _pitch_bends_from_labels(sample_dir)
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


def regenerate_bundle_musesounds(
    sample_dir: Path,
    semitones: int = -2,
    *,
    force: bool = False,
    sample_rate: int = 22050,
    work_dir: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    from synthpipeline.musesounds import (
        MUSESOUNDS_RENDER_MARK as mark,
        export_musesounds_mp3,
        mp3_to_wav,
        prepare_bb_clarinet_xml,
    )

    sample_dir = Path(sample_dir)
    verified = sample_dir / "verified_score.musicxml"
    performance = sample_dir / "performance_score.musicxml"
    if not verified.is_file() or not performance.is_file():
        return "skip_missing"
    meta_path = sample_dir / "metadata.json"
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not force and meta.get("audio_render") == mark:
            return "skip_done"

    tmp_root = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="musesounds_"))
    tmp = tmp_root / sample_dir.name
    tmp.mkdir(parents=True, exist_ok=True)
    pairs = (
        (verified, sample_dir / "reference_audio.wav", "reference"),
        (performance, sample_dir / "performance_audio.wav", "performance"),
    )
    try:
        for xml_src, wav_path, stem in pairs:
            xml_copy = tmp / f"{stem}.musicxml"
            mp3_path = tmp / f"{stem}.mp3"
            prepare_bb_clarinet_xml(xml_src, xml_copy)
            export_musesounds_mp3(xml_copy, mp3_path, env=env)
            mp3_to_wav(mp3_path, wav_path, sample_rate=sample_rate)
        _finalize_musesounds_bundle(sample_dir, meta, meta_path, semitones, sample_rate)
    finally:
        if work_dir is None:
            shutil.rmtree(tmp_root, ignore_errors=True)
    return "converted"


def _finalize_musesounds_bundle(
    sample_dir: Path,
    meta: dict,
    meta_path: Path,
    semitones: int,
    sample_rate: int,
) -> None:
    from datacreate.stages.stage5_alignment import run_alignment, write_candidates
    from datacreate.stages.stage7_features import extract_mels
    from synthpipeline.musesounds import MUSESOUNDS_RENDER_MARK
    from synthpipeline.pitch_convention import (
        annotate_pitch_metadata,
        infer_midi_pitch_space,
    )

    dummy_sf = Path(str(meta.get("soundfont_path") or "unused.sf2"))
    cfg = _dc_config(meta, dummy_sf, sample_rate)
    logger = logging.getLogger("synthpipeline.rerender")
    logger.setLevel(logging.WARNING)
    from datacreate.audio_utils import save_wav

    ref_wav = sample_dir / "reference_audio.wav"
    perf_wav = sample_dir / "performance_audio.wav"
    for wav_path in (ref_wav, perf_wav):
        audio, _ = _load_mono(wav_path, sample_rate)
        mag = float(np.max(np.abs(audio))) if audio.size else 0.0
        if mag < 1e-5:
            raise RuntimeError(f"silent Muse Sounds render: {wav_path}")
        audio = (audio * min(0.89 / mag, 4.0)).astype(np.float32)
        save_wav(wav_path, audio, sample_rate)
    extract_mels(perf_wav, ref_wav, sample_dir, cfg, logger)
    alignment = run_alignment(perf_wav, ref_wav, sample_dir, cfg, logger)
    schema = "1.2"
    labels_path = sample_dir / "labels.json"
    if labels_path.is_file():
        schema = str(json.loads(labels_path.read_text(encoding="utf-8")).get("schema_version") or schema)
    write_candidates(alignment.candidates, sample_dir, schema)
    inferred_space, _offset = infer_midi_pitch_space(sample_dir, meta)
    meta["sounding_transpose"] = int(semitones)
    meta["sound_profile"] = "MuseSounds"
    meta["soundfont"] = "muse_woodwinds_clarinet"
    meta.update(
        annotate_pitch_metadata(
            meta,
            midi_space=inferred_space,
            audio_space="sounding",
            effective_audio_shift=-int(semitones),
            audio_render=MUSESOUNDS_RENDER_MARK,
        )
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _one(payload: tuple) -> str:
    sample, semis, force, sr, strip, backend = payload
    try:
        return regenerate_bundle(
            Path(sample),
            semis,
            force=force,
            sample_rate=sr,
            strip_ornaments=strip,
            backend=str(backend),
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
    backend: str = "soundfont",
    batch_size: int = 16,
) -> dict[str, int]:
    backend = str(backend or "soundfont").lower()
    if backend == "musesounds":
        return regenerate_root_musesounds(
            root,
            semitones=semitones,
            force=force,
            sample_rate=sample_rate,
            batch_size=batch_size,
            workers=workers,
        )
    dirs = discover_bundles(root)
    counts = {
        "converted": 0,
        "skip_done": 0,
        "skip_missing": 0,
        "failed": 0,
        "n_bundles": len(dirs),
    }
    jobs = [
        (str(p), int(semitones), bool(force), int(sample_rate), bool(strip_ornaments), "soundfont")
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


def _split_shards(items: list, workers: int) -> list[list]:
    n = max(1, int(workers))
    if not items:
        return [[] for _ in range(n)]
    n = min(n, len(items))
    shards: list[list] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        shards[i % n].append(item)
    return shards


def _musesounds_run_pending(payload: dict) -> dict[str, int]:
    from synthpipeline.musesounds import (
        export_musesounds_job,
        mp3_to_wav,
        musescore_worker_env,
        prepare_bb_clarinet_xml,
    )

    pending = [Path(p) for p in payload["pending"]]
    semitones = int(payload["semitones"])
    sample_rate = int(payload["sample_rate"])
    batch_n = max(1, int(payload["batch_size"]))
    work_root = Path(payload["work_root"])
    label = str(payload.get("label") or "w")
    home = payload.get("home")
    env = musescore_worker_env(Path(home)) if home else None
    counts = {"converted": 0, "failed": 0, "n": len(pending)}
    work_root.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pending), batch_n):
        batch = pending[start : start + batch_n]
        jobs: list[tuple[Path, Path]] = []
        prepared: list[tuple[Path, Path, Path, Path, Path]] = []
        for sample_dir in batch:
            tmp = work_root / sample_dir.name
            tmp.mkdir(parents=True, exist_ok=True)
            ref_xml = tmp / "reference.musicxml"
            perf_xml = tmp / "performance.musicxml"
            ref_mp3 = tmp / "reference.mp3"
            perf_mp3 = tmp / "performance.mp3"
            prepare_bb_clarinet_xml(sample_dir / "verified_score.musicxml", ref_xml)
            prepare_bb_clarinet_xml(sample_dir / "performance_score.musicxml", perf_xml)
            jobs.append((ref_xml, ref_mp3))
            jobs.append((perf_xml, perf_mp3))
            prepared.append(
                (
                    sample_dir,
                    ref_mp3,
                    perf_mp3,
                    sample_dir / "reference_audio.wav",
                    sample_dir / "performance_audio.wav",
                )
            )
        try:
            export_musesounds_job(
                jobs, work_dir=work_root / f"job_{start}", env=env
            )
        except Exception as exc:
            print(f"  {label} batch {start} job failed ({exc}); retrying one-by-one", flush=True)
            for sample_dir, _ref_mp3, _perf_mp3, _ref_wav, _perf_wav in prepared:
                try:
                    regenerate_bundle_musesounds(
                        sample_dir,
                        semitones=semitones,
                        force=True,
                        sample_rate=sample_rate,
                        work_dir=work_root / f"retry_{sample_dir.name}",
                        env=env,
                    )
                    counts["converted"] += 1
                except Exception as one_exc:
                    print(
                        f"failed {sample_dir}: {type(one_exc).__name__}: {one_exc}",
                        flush=True,
                    )
                    counts["failed"] += 1
            done = min(start + len(batch), len(pending))
            print(f"  {label}: {done}/{len(pending)}", flush=True)
            continue

        for sample_dir, ref_mp3, perf_mp3, ref_wav, perf_wav in prepared:
            try:
                mp3_to_wav(ref_mp3, ref_wav, sample_rate=sample_rate)
                mp3_to_wav(perf_mp3, perf_wav, sample_rate=sample_rate)
                meta_path = sample_dir / "metadata.json"
                meta = (
                    json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta_path.exists()
                    else {}
                )
                _finalize_musesounds_bundle(
                    sample_dir, meta, meta_path, semitones, sample_rate
                )
                counts["converted"] += 1
            except Exception as exc:
                print(f"failed {sample_dir}: {type(exc).__name__}: {exc}", flush=True)
                counts["failed"] += 1
        done = min(start + len(batch), len(pending))
        print(f"  {label}: {done}/{len(pending)}", flush=True)
    return counts


def regenerate_root_musesounds(
    root: Path,
    *,
    semitones: int = -2,
    force: bool = False,
    sample_rate: int = 22050,
    batch_size: int = 16,
    workers: int = 4,
) -> dict[str, int]:
    """Re-render WAVs with Muse Woodwinds clarinet. Scores, MIDI, and labels stay put."""
    from synthpipeline.musesounds import MUSESOUNDS_RENDER_MARK

    dirs = discover_bundles(root)
    counts = {
        "converted": 0,
        "skip_done": 0,
        "skip_missing": 0,
        "failed": 0,
        "n_bundles": len(dirs),
    }
    pending: list[Path] = []
    for sample_dir in dirs:
        verified = sample_dir / "verified_score.musicxml"
        performance = sample_dir / "performance_score.musicxml"
        if not verified.is_file() or not performance.is_file():
            counts["skip_missing"] += 1
            continue
        meta_path = sample_dir / "metadata.json"
        if meta_path.exists() and not force:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("audio_render") == MUSESOUNDS_RENDER_MARK:
                counts["skip_done"] += 1
                continue
        pending.append(sample_dir)

    n_workers = max(1, int(workers))
    batch_n = max(1, int(batch_size))
    print(
        f"  {root}: musesounds pending {len(pending)}/{len(dirs)} workers={n_workers}",
        flush=True,
    )
    if not pending:
        return counts

    work_root = Path(tempfile.mkdtemp(prefix="musesounds_root_"))
    try:
        if n_workers == 1:
            part = _musesounds_run_pending(
                {
                    "pending": [str(p) for p in pending],
                    "semitones": semitones,
                    "sample_rate": sample_rate,
                    "batch_size": batch_n,
                    "work_root": str(work_root / "w0"),
                    "home": None,
                    "label": "w0",
                }
            )
            counts["converted"] += part["converted"]
            counts["failed"] += part["failed"]
            return counts

        shards = _split_shards(pending, n_workers)
        jobs = []
        for i, shard in enumerate(shards):
            if not shard:
                continue
            jobs.append(
                {
                    "pending": [str(p) for p in shard],
                    "semitones": semitones,
                    "sample_rate": sample_rate,
                    "batch_size": batch_n,
                    "work_root": str(work_root / f"w{i}"),
                    "home": str(work_root / f"home_w{i}"),
                    "label": f"w{i}",
                }
            )
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            futs = [pool.submit(_musesounds_run_pending, job) for job in jobs]
            for fut in as_completed(futs):
                part = fut.result()
                counts["converted"] += part["converted"]
                counts["failed"] += part["failed"]
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    return counts
