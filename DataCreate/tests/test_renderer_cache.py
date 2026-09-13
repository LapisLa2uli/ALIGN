from __future__ import annotations

import logging
import sys
import types
import numpy as np

from datacreate.config import PipelineConfig
from datacreate.note_alignment import _normalize_transcribed_notes
from datacreate.tools import musescore


class FakeSynth:
    loads = 0

    def __init__(self, **_kwargs):
        pass

    def sfload(self, _path):
        type(self).loads += 1

    def program_change(self, *_args):
        pass

    def generate(self, count):
        return np.zeros(count * 2, dtype=np.float32).tobytes()

    def notes_off(self):
        pass

    def sounds_off(self):
        pass


class FakeEvent:
    def __init__(self, t=0.05):
        self.t = t


class FakeSequencer:
    def __init__(self, _synth):
        pass

    def add(self, _events):
        pass


def _reset_renderer():
    musescore._synth_cache.clear()
    musescore._set_status(state="idle", error=None, soundfont=None, loaded_sec=None)
    FakeSynth.loads = 0


def _install_fake_tinysoundfont(monkeypatch):
    midi = types.ModuleType("tinysoundfont.midi")
    midi.load = lambda _path, persistent=False: [FakeEvent(0.05)]
    seq = types.ModuleType("tinysoundfont.sequencer")
    seq.Sequencer = FakeSequencer
    tsf = types.ModuleType("tinysoundfont")
    tsf.Synth = FakeSynth
    tsf.midi = midi
    tsf.sequencer = seq
    monkeypatch.setitem(sys.modules, "tinysoundfont", tsf)
    monkeypatch.setitem(sys.modules, "tinysoundfont.midi", midi)
    monkeypatch.setitem(sys.modules, "tinysoundfont.sequencer", seq)


def test_normalize_transcribed_notes_keeps_unmapped_extras():
    notes = _normalize_transcribed_notes(
        {
            "transcribed_notes": [
                {"pitch": 60, "start": 0.0, "end": 0.2},
                {"pitch": 64, "start": 0.2, "end": 0.4},
            ],
            "note_mapping": [0, None],
        }
    )
    assert notes[0]["score_index"] == 0
    assert notes[0]["pitch"] == "C4"
    assert notes[1]["score_index"] is None
    assert notes[1]["pitch"] == "E4"


def test_render_reuses_cached_synth(tmp_path, monkeypatch):
    _reset_renderer()
    _install_fake_tinysoundfont(monkeypatch)
    soundfont = tmp_path / "MS Basic.sf3"
    soundfont.write_bytes(b"sf")
    monkeypatch.setattr(musescore, "find_soundfont", lambda _config: soundfont)
    config = PipelineConfig(
        paths={"soundfont": str(soundfont)},
        audio={"sample_rate": 22050},
        musescore={"tail_seconds": 0.01, "render_chunk_size": 64},
    )
    logger = logging.getLogger("test-renderer")
    midi = tmp_path / "in.mid"
    midi.write_bytes(b"mid")
    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    musescore.render_midi_to_wav(midi, first, config, logger)
    musescore.render_midi_to_wav(midi, second, config, logger)
    assert FakeSynth.loads == 1
    assert first.exists() and second.exists()
    assert musescore.renderer_status()["state"] == "ready"


def test_warmup_loads_soundfont_once(tmp_path, monkeypatch):
    _reset_renderer()
    _install_fake_tinysoundfont(monkeypatch)
    soundfont = tmp_path / "MS Basic.sf3"
    soundfont.write_bytes(b"sf")
    monkeypatch.setattr(musescore, "find_soundfont", lambda _config: soundfont)
    config = PipelineConfig(
        paths={"soundfont": str(soundfont)},
        audio={"sample_rate": 22050},
    )
    musescore.warmup_synth(config, logging.getLogger("test-warmup"))
    musescore.warmup_synth(config, logging.getLogger("test-warmup"))
    assert FakeSynth.loads == 1
    assert musescore.renderer_status()["state"] == "ready"
    assert musescore.renderer_status()["soundfont"] == "MS Basic.sf3"
