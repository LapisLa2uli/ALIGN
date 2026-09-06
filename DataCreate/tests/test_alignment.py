from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from music21 import converter, meter, note, stream, tempo

from datacreate.audio_utils import save_wav, sounding_span
from datacreate.config import PipelineConfig
from datacreate.note_alignment import (
    _audio_time_for_ql,
    _extract_score_events,
    _fill_ref_to_perf,
    _ql_to_sec,
    _redistribute_crushed_phrases,
    _snap_phrases_to_voiced,
    _tempo_map,
    align_score_events,
)
from datacreate.score_segment import (
    extract_measure_range,
    measures_in_written_order,
)
from datacreate.stages.stage5_alignment import (
    _align_phrases_sequential,
    _densify_warping_path,
    _score_phrase_spans,
    run_alignment,
    silence_keep_mask,
)


def _pickup_score(path: Path) -> Path:
    score = stream.Score()
    part = stream.Part()
    part.insert(0, tempo.MetronomeMark(number=60))
    part.insert(0, meter.TimeSignature("4/4"))

    m1 = stream.Measure(number=1)
    # Measure-level mark survives MusicXML write; part-level marks often do not.
    m1.insert(0, tempo.MetronomeMark(number=60))
    m1.append(note.Note("C4", quarterLength=4.0))

    m2 = stream.Measure(number=2)
    m2.append(note.Note("C4", quarterLength=2.0))
    m2.append(note.Rest(quarterLength=1.0))
    m2.append(note.Rest(quarterLength=0.5))
    m2.insert(3.5, note.Note("G4", quarterLength=0.5))
    m2.insert(3.5, tempo.MetronomeMark(number=72))

    m3 = stream.Measure(number=3)
    m3.insert(0, meter.TimeSignature("2/4"))
    m3.append(note.Note("C5", quarterLength=1.0))
    m3.append(note.Note("E5", quarterLength=1.0))

    part.append(m1)
    part.append(m2)
    part.append(m3)
    score.insert(0, part)
    score.write("musicxml", fp=str(path))
    return path


def test_pickup_extract_keeps_measure_order_and_packs(tmp_path):
    src = _pickup_score(tmp_path / "src.musicxml")
    dest = tmp_path / "excerpt.musicxml"
    extract_measure_range(src, dest, 2, 3, logging.getLogger("test"), start_beat=4)

    parsed = converter.parse(str(dest))
    part = parsed.parts[0]
    measures = list(part.getElementsByClass(stream.Measure))
    assert [m.number for m in measures] == [2, 3]
    assert measures_in_written_order(parsed)
    offsets = [float(part.elementOffset(m)) for m in measures]
    assert offsets[0] == 0.0
    assert abs(offsets[1] - 1.0) < 1e-6

    m2 = measures[0]
    pitches = [
        el.nameWithOctave
        for el in m2.notesAndRests
        if isinstance(el, note.Note)
    ]
    assert pitches == ["G4"]
    assert all(float(el.offset) < 1.0 + 1e-6 for el in m2.notesAndRests)

    marks = [
        (float(el.getOffsetInHierarchy(parsed)), float(el.number))
        for el in parsed.flatten().getElementsByClass(tempo.MetronomeMark)
        if el.number
    ]
    opening = [bpm for off, bpm in marks if off < 1e-4]
    assert opening
    # Excerpt starts on the leftover rest (still 60); Andante 72 lands on the G4.
    assert opening[0] == 60.0
    later = [bpm for off, bpm in marks if off > 1e-4]
    assert 72.0 in later


def test_score_events_use_hierarchy_offsets_not_measure_local(tmp_path):
    src = _pickup_score(tmp_path / "src.musicxml")
    events = _extract_score_events(src)
    by_measure = {}
    for ev in events:
        by_measure.setdefault(ev["measure"], []).append(ev)
    m1_end = max(ev["ref_start"] for ev in by_measure[1])
    m2_start = min(ev["ref_start"] for ev in by_measure[2])
    m3_start = min(ev["ref_start"] for ev in by_measure[3])
    assert m2_start >= m1_end - 1e-6
    assert m3_start > m2_start
    g4 = next(ev for ev in events if ev["pitch"] == "G4")
    assert g4["offset_ql"] == 7.5


def test_tempo_map_integrates_mid_score_change():
    score = stream.Score()
    part = stream.Part()
    part.insert(0, tempo.MetronomeMark(number=60))
    part.insert(4, tempo.MetronomeMark(number=120))
    score.insert(0, part)
    mapping = _tempo_map(score)
    assert abs(_ql_to_sec(4.0, mapping) - 4.0) < 1e-6
    assert abs(_ql_to_sec(6.0, mapping) - 5.0) < 1e-6


def test_fill_mapping_holds_silence_edges():
    mapping = np.array([np.nan, np.nan, 10.0, 12.0, np.nan], dtype=np.float64)
    filled = _fill_ref_to_perf(mapping)
    assert filled[0] == 0.0
    assert 0.0 < filled[1] < 10.0
    assert filled[-1] == 12.0
    assert np.all(np.diff(filled) >= -1e-9)


def test_sounding_span_trims_ends_only():
    sr = 22050
    audio = np.zeros(sr * 3, dtype=np.float32)
    start = int(0.8 * sr)
    end = int(1.8 * sr)
    t = np.arange(end - start) / sr
    audio[start:end] = 0.25 * np.sin(2 * np.pi * 440 * t)
    s0, s1 = sounding_span(audio, sr, top_db=30, pad_sec=0.05)
    assert 0.2 * sr < s0 < start
    assert end < s1 < 2.7 * sr


def _tone(sr: int, freq: float, seconds: float) -> np.ndarray:
    n = int(sr * seconds)
    t = np.arange(n) / sr
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_leading_silence_mismatch_does_not_shift_note(tmp_path):
    sr = 22050
    rest = 0.5
    note_dur = 0.8
    tail = 0.4
    ref = np.concatenate(
        [np.zeros(int(sr * rest), dtype=np.float32), _tone(sr, 440, note_dur), np.zeros(int(sr * tail), dtype=np.float32)]
    )
    perf_lead = 1.2
    perf = np.concatenate(
        [
            np.zeros(int(sr * perf_lead), dtype=np.float32),
            _tone(sr, 440, note_dur),
            np.zeros(int(sr * 0.15), dtype=np.float32),
        ]
    )
    save_wav(tmp_path / "reference_audio.wav", ref, sr)
    save_wav(tmp_path / "performance_audio.wav", perf, sr)

    score = stream.Score()
    part = stream.Part()
    part.insert(0, tempo.MetronomeMark(number=120))
    part.append(note.Rest(quarterLength=1.0))
    part.append(note.Note("A4", quarterLength=1.6))
    score.insert(0, part)
    score.write("musicxml", fp=str(tmp_path / "verified_score.musicxml"))

    config = PipelineConfig.load()
    config.audio["sample_rate"] = sr
    logger = logging.getLogger("test_align")
    run_alignment(
        tmp_path / "performance_audio.wav",
        tmp_path / "reference_audio.wav",
        tmp_path,
        config,
        logger,
    )
    data = np.load(tmp_path / "alignment.npz")
    events = align_score_events(
        tmp_path / "verified_score.musicxml",
        data["warping_path"],
        int(data["ref_features"].shape[1]),
        float(data["hop_length"]) / float(data["sample_rate"]),
        onset_refine=False,
    )
    sounding = next(ev for ev in events if not ev["is_rest"])
    assert sounding["perf_start"] > perf_lead - 0.25
    assert sounding["perf_start"] < perf_lead + 0.35
    rest_ev = next(ev for ev in events if ev["is_rest"])
    assert rest_ev["perf_end"] <= sounding["perf_start"] + 1e-3


def test_intro_extract_stamps_opening_tempo_not_late_andante(tmp_path):
    src = _pickup_score(tmp_path / "src.musicxml")
    dest = tmp_path / "intro.musicxml"
    extract_measure_range(src, dest, 1, 2, logging.getLogger("test"), start_beat=1)
    parsed = converter.parse(str(dest))
    marks = [
        (float(el.getOffsetInHierarchy(parsed)), float(el.number))
        for el in parsed.flatten().getElementsByClass(tempo.MetronomeMark)
        if el.number
    ]
    opening = [bpm for off, bpm in marks if off < 1e-4]
    assert opening[0] == 60.0
    assert any(off > 3.0 and bpm == 72.0 for off, bpm in marks)


def test_audio_time_follows_quarter_length_not_wrong_tempo():
    # 76 ql rendered into 38 s of audio (MIDI at 120), even if MusicXML says 72.
    assert abs(_audio_time_for_ql(0.0, 76.0, 38.0) - 0.0) < 1e-9
    assert abs(_audio_time_for_ql(38.0, 76.0, 38.0) - 19.0) < 1e-9
    assert abs(_audio_time_for_ql(76.0, 76.0, 38.0) - 38.0) < 1e-9


def test_silence_keep_mask_collapses_long_rests():
    hop_sec = 0.02
    energy = np.zeros(200)
    energy[20:40] = 1.0
    energy[160:180] = 1.0
    keep = silence_keep_mask(energy, hop_sec, silence_thresh=0.1, keep_silence_sec=0.08)
    assert int(np.count_nonzero(keep[40:160])) <= 12
    assert keep[20] and keep[39]
    assert keep[160] and keep[179]


def test_score_phrases_split_on_long_rests(tmp_path):
    score = stream.Score()
    part = stream.Part()
    part.insert(0, tempo.MetronomeMark(number=60))
    part.append(note.Note("C4", quarterLength=1.0))
    part.append(note.Rest(quarterLength=2.0))
    part.append(note.Note("E4", quarterLength=1.0))
    part.append(note.Rest(quarterLength=0.5))
    part.append(note.Note("G4", quarterLength=1.0))
    score.insert(0, part)
    path = tmp_path / "verified_score.musicxml"
    score.write("musicxml", fp=str(path))
    # 5 ql at 60 BPM → 5 s of "audio"
    phrases = _score_phrase_spans(path, audio_dur=5.0, min_rest_ql=1.5)
    assert len(phrases) == 2
    assert phrases[0][0] < 1.1
    assert phrases[1][0] >= 2.5


def test_redistribute_spreads_crushed_arpeggio():
    events = []
    t = 0.0
    for i, ql in enumerate([1.0, 1.0, 0.25, 0.25, 0.25, 0.25, 2.0]):
        events.append(
            {
                "is_rest": False,
                "duration_ql": ql,
                "ref_start": t,
                "ref_end": t + ql,
                "perf_start": 10.0 if i < 2 else 12.0,
                "perf_end": 11.0 if i < 2 else 12.001,
            }
        )
        t += ql
    _redistribute_crushed_phrases(events)
    assert events[2]["perf_start"] > events[1]["perf_end"] - 1e-6
    assert events[-1]["perf_end"] - events[-1]["perf_start"] > 0.8
    assert events[2]["perf_end"] - events[2]["perf_start"] > 0.05


def test_snap_phrases_moves_notes_off_silence():
    sr = 22050
    hop_sec = 512 / sr
    note = _tone(sr, 440, 0.6)
    rest = np.zeros(int(sr * 1.4), dtype=np.float32)
    audio = np.concatenate([note, rest, _tone(sr, 554, 0.6)])
    events = [
        {
            "is_rest": False,
            "duration_ql": 1.0,
            "perf_start": 0.05,
            "perf_end": 0.55,
        },
        {
            "is_rest": True,
            "duration_ql": 2.0,
            "perf_start": 0.55,
            "perf_end": 1.9,
        },
        {
            "is_rest": False,
            "duration_ql": 1.0,
            # DTW spilled the second phrase into the rest.
            "perf_start": 0.9,
            "perf_end": 1.8,
        },
    ]
    _snap_phrases_to_voiced(events, audio, sr, hop_sec)
    second = events[2]
    assert second["perf_start"] > 1.85
    assert second["perf_end"] < 2.65
    assert second["perf_start"] < 2.15


def test_short_rest_splits_phrases(tmp_path):
    score = stream.Score()
    part = stream.Part()
    part.insert(0, tempo.MetronomeMark(number=60))
    part.append(note.Note("C4", quarterLength=1.0))
    part.append(note.Rest(quarterLength=0.25))
    part.append(note.Note("E4", quarterLength=1.0))
    score.insert(0, part)
    path = tmp_path / "verified_score.musicxml"
    score.write("musicxml", fp=str(path))
    phrases = _score_phrase_spans(path, audio_dur=2.25, min_rest_ql=0.25)
    assert len(phrases) == 2
    assert phrases[1][0] >= 1.2


def test_snap_preserves_dtw_when_already_voiced():
    sr = 22050
    hop_sec = 512 / sr
    audio = _tone(sr, 440, 1.2)
    events = [
        {
            "is_rest": False,
            "duration_ql": 0.5,
            "ref_start": 0.0,
            "ref_end": 0.5,
            "perf_start": 0.10,
            "perf_end": 0.55,
        },
        {
            "is_rest": False,
            "duration_ql": 0.5,
            "ref_start": 0.5,
            "ref_end": 1.0,
            "perf_start": 0.55,
            "perf_end": 1.00,
        },
    ]
    _snap_phrases_to_voiced(events, audio, sr, hop_sec)
    assert abs(events[0]["perf_start"] - 0.10) < 0.08
    assert abs(events[1]["perf_start"] - 0.55) < 0.08


def test_sequential_phrases_do_not_skip_to_later_repeat():
    """A late global path must not keep the first figure on the second repeat."""
    hop = 512
    sr = 22050
    fts = hop / sr
    n = 40
    silent = 12

    def block(bin_idx: int, frames: int, energy: float = 1.0) -> np.ndarray:
        feat = np.zeros((13, frames), dtype=np.float64)
        for i in range(frames):
            feat[(bin_idx + i // 8) % 12, i] = 1.0
            feat[12, i] = energy
        return feat

    ref = np.hstack(
        [block(0, n), np.zeros((13, silent)), block(5, n), np.zeros((13, silent)), block(0, n)]
    )
    perf = np.hstack(
        [block(0, n), np.zeros((13, silent)), block(5, n), np.zeros((13, silent)), block(0, n)]
    )
    # Global path shifted so early ref frames land on the later repeat.
    shift = 2 * n + 2 * silent
    wp = np.column_stack(
        [
            np.arange(ref.shape[1], dtype=np.int32),
            np.clip(np.arange(ref.shape[1]) + shift, 0, perf.shape[1] - 1).astype(np.int32),
        ]
    )
    phrases = [
        (0.0, n * fts),
        ((n + silent) * fts, (2 * n + silent) * fts),
        ((2 * n + 2 * silent) * fts, (3 * n + 2 * silent) * fts),
    ]
    config = PipelineConfig.load()
    out = _align_phrases_sequential(
        wp, phrases, ref, perf, hop, sr, config, logging.getLogger("test_seq")
    )
    first = out[out[:, 0] < n]
    last = out[out[:, 0] >= 2 * n + 2 * silent]
    assert first.size and last.size
    assert int(first[:, 1].max()) < n + silent
    assert int(last[:, 1].min()) >= 2 * n + silent


def test_densify_path_fills_rest_jumps():
    wp = np.array([[0, 0], [2, 1], [20, 4]], dtype=np.int32)
    dense = _densify_warping_path(wp)
    assert dense[0, 0] == 0
    assert dense[-1, 0] == 20
    assert 10 in set(dense[:, 0])


def test_interior_rest_mismatch_keeps_second_note(tmp_path):
    sr = 22050
    note_dur = 0.45
    ref = np.concatenate(
        [
            _tone(sr, 440, note_dur),
            np.zeros(int(sr * 2.0), dtype=np.float32),
            _tone(sr, 554, note_dur),
        ]
    )
    perf = np.concatenate(
        [
            np.zeros(int(sr * 0.4), dtype=np.float32),
            _tone(sr, 440, note_dur),
            np.zeros(int(sr * 0.25), dtype=np.float32),
            _tone(sr, 554, note_dur),
            np.zeros(int(sr * 0.2), dtype=np.float32),
        ]
    )
    save_wav(tmp_path / "reference_audio.wav", ref, sr)
    save_wav(tmp_path / "performance_audio.wav", perf, sr)

    score = stream.Score()
    part = stream.Part()
    part.insert(0, tempo.MetronomeMark(number=120))
    part.append(note.Note("A4", quarterLength=0.9))
    part.append(note.Rest(quarterLength=4.0))
    part.append(note.Note("C#5", quarterLength=0.9))
    score.insert(0, part)
    score.write("musicxml", fp=str(tmp_path / "verified_score.musicxml"))

    config = PipelineConfig.load()
    config.audio["sample_rate"] = sr
    run_alignment(
        tmp_path / "performance_audio.wav",
        tmp_path / "reference_audio.wav",
        tmp_path,
        config,
        logging.getLogger("test_rests"),
    )
    data = np.load(tmp_path / "alignment.npz")
    events = align_score_events(
        tmp_path / "verified_score.musicxml",
        data["warping_path"],
        int(data["ref_features"].shape[1]),
        float(data["hop_length"]) / float(data["sample_rate"]),
        onset_refine=False,
    )
    notes = [ev for ev in events if not ev["is_rest"]]
    assert len(notes) == 2
    assert notes[0]["perf_start"] < 1.1
    # Second note must land on the later tone, not get eaten by the long rest.
    assert notes[1]["perf_start"] > notes[0]["perf_end"] + 0.1
    assert notes[1]["perf_start"] > 0.9


def test_opening_rest_and_repeat_do_not_lag_first_figure(tmp_path):
    """Opening rest + a later repeat of the first pitch must not drag notes late."""
    sr = 22050
    ref = np.concatenate(
        [
            np.zeros(int(sr * 0.35), dtype=np.float32),
            _tone(sr, 440, 0.45),
            np.zeros(int(sr * 0.30), dtype=np.float32),
            _tone(sr, 554, 0.45),
        ]
    )
    perf = np.concatenate(
        [
            np.zeros(int(sr * 0.12), dtype=np.float32),
            _tone(sr, 440, 0.45),
            np.zeros(int(sr * 0.20), dtype=np.float32),
            _tone(sr, 554, 0.45),
            np.zeros(int(sr * 0.20), dtype=np.float32),
            _tone(sr, 440, 0.55),
        ]
    )
    save_wav(tmp_path / "reference_audio.wav", ref, sr)
    save_wav(tmp_path / "performance_audio.wav", perf, sr)

    score = stream.Score()
    part = stream.Part()
    part.insert(0, tempo.MetronomeMark(number=120))
    part.append(note.Rest(quarterLength=0.7))
    part.append(note.Note("A4", quarterLength=0.9))
    part.append(note.Rest(quarterLength=0.6))
    part.append(note.Note("C#5", quarterLength=0.9))
    score.insert(0, part)
    score.write("musicxml", fp=str(tmp_path / "verified_score.musicxml"))

    config = PipelineConfig.load()
    config.audio["sample_rate"] = sr
    run_alignment(
        tmp_path / "performance_audio.wav",
        tmp_path / "reference_audio.wav",
        tmp_path,
        config,
        logging.getLogger("test_lag"),
    )
    data = np.load(tmp_path / "alignment.npz")
    events = align_score_events(
        tmp_path / "verified_score.musicxml",
        data["warping_path"],
        int(data["ref_features"].shape[1]),
        float(data["hop_length"]) / float(data["sample_rate"]),
        onset_refine=False,
    )
    notes = [ev for ev in events if not ev["is_rest"]]
    assert notes[0]["perf_start"] < 0.45
    assert notes[1]["perf_start"] < 1.15
    assert notes[1]["perf_start"] > notes[0]["perf_end"]


def test_practice_restart_does_not_stretch_last_phrase(tmp_path):
    """Stop, replay the first figure, then continue — last notes stay on the continuation."""
    sr = 22050
    ref = np.concatenate(
        [
            _tone(sr, 440, 0.50),
            np.zeros(int(sr * 0.35), dtype=np.float32),
            _tone(sr, 554, 0.50),
        ]
    )
    perf = np.concatenate(
        [
            _tone(sr, 440, 0.50),
            np.zeros(int(sr * 0.45), dtype=np.float32),
            _tone(sr, 440, 0.50),
            np.zeros(int(sr * 0.45), dtype=np.float32),
            _tone(sr, 554, 0.50),
        ]
    )
    save_wav(tmp_path / "reference_audio.wav", ref, sr)
    save_wav(tmp_path / "performance_audio.wav", perf, sr)

    score = stream.Score()
    part = stream.Part()
    part.insert(0, tempo.MetronomeMark(number=120))
    part.append(note.Note("A4", quarterLength=1.0))
    part.append(note.Rest(quarterLength=0.7))
    part.append(note.Note("C#5", quarterLength=1.0))
    score.insert(0, part)
    score.write("musicxml", fp=str(tmp_path / "verified_score.musicxml"))

    config = PipelineConfig.load()
    config.audio["sample_rate"] = sr
    run_alignment(
        tmp_path / "performance_audio.wav",
        tmp_path / "reference_audio.wav",
        tmp_path,
        config,
        logging.getLogger("test_restart"),
    )
    data = np.load(tmp_path / "alignment.npz")
    events = align_score_events(
        tmp_path / "verified_score.musicxml",
        data["warping_path"],
        int(data["ref_features"].shape[1]),
        float(data["hop_length"]) / float(data["sample_rate"]),
        onset_refine=False,
    )
    notes = [ev for ev in events if not ev["is_rest"]]
    # First play of A4, then a restart of A4 at ~0.95s, continuation C#5 at ~1.90s.
    assert notes[0]["perf_end"] < 0.85
    assert notes[1]["perf_start"] > 1.70
    assert notes[1]["perf_end"] < 2.55
    assert notes[1]["perf_end"] - notes[1]["perf_start"] < 1.1
