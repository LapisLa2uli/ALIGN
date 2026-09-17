"""In-process host for MuseSamplerCoreLib (Muse Sounds).

MuseScore Studio loads this DLL at runtime and feeds it note events. This
module does the same without spawning the GUI: init, pick Bb clarinet, schedule
notes (including per-note cents), and pull stereo audio via the offline
processor.

The C ABI is the MuseScore 4.7 ``apitypes.h`` / ``libhandler.h`` wrapper
(note event 6 = ``ms_NoteEvent_5``). Struct layouts follow MSVC x64 packing.
"""

from __future__ import annotations

import os
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    byref,
    c_bool,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_int16,
    c_longlong,
    c_uint64,
    c_void_p,
    cdll,
)
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MS_OK = 0
DEFAULT_DLL = Path(os.environ.get("LOCALAPPDATA", "")) / "MuseSampler" / "lib" / "MuseSamplerCoreLib.dll"
DEFAULT_HUB_LIB = Path(os.environ.get("LOCALAPPDATA", "")) / "Muse Hub" / "lib"
DEFAULT_SR = 44100
DEFAULT_BLOCK = 1024
AUDIO_CHANNELS = 2

# logging callback must outlive the DLL; keep on the host instance
_LogCallback = CFUNCTYPE(None, c_int16, c_char_p)


class OutputBuffer(Structure):
    _fields_ = [
        ("_channels", POINTER(POINTER(c_float))),
        ("_num_data_pts", c_int),
        ("_num_channels", c_int),
    ]


class NoteEvent5(Structure):
    _fields_ = [
        ("_voice", c_int),
        ("_location_us", c_longlong),
        ("_duration_us", c_longlong),
        ("_pitch", c_int),
        ("_tempo", c_double),
        ("_offset_cents", c_int),
        ("_articulation", c_uint64),
        ("_articulation_2", c_uint64),
        ("_notehead", c_int16),
    ]


class DynamicsEvent2(Structure):
    _fields_ = [
        ("_location_us", c_longlong),
        ("_value", c_double),
    ]


@dataclass(frozen=True)
class Instrument:
    instrument_id: int
    name: str
    category: str
    pack: str
    vendor: str
    musicxml_sound: str
    mpe_sound: str
    online: bool
    presets: tuple[str, ...]


@dataclass(frozen=True)
class SamplerNote:
    pitch: int
    start_us: int
    duration_us: int
    cents: int = 0
    tempo: float = 120.0
    voice: int = 0


class MuseSamplerError(RuntimeError):
    pass


def _cstr(value) -> str:
    if not value:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class MuseSamplerHost:
    def __init__(
        self,
        dll_path: Path | str | None = None,
        sample_rate: int = DEFAULT_SR,
        block_size: int = DEFAULT_BLOCK,
        log_lines: list[str] | None = None,
    ) -> None:
        self.dll_path = Path(dll_path) if dll_path else DEFAULT_DLL
        if not self.dll_path.is_file():
            raise MuseSamplerError(f"MuseSampler DLL not found: {self.dll_path}")
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.log_lines = log_lines if log_lines is not None else []
        self._lib = None
        self._sampler = None
        self._track = None
        self._log_cb = None
        self._offline = False
        self._left = None
        self._right = None
        self._chan_ptrs = None
        self._bus = OutputBuffer()

    def __enter__(self) -> "MuseSamplerHost":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        os.add_dll_directory(str(self.dll_path.parent))
        if DEFAULT_HUB_LIB.is_dir():
            os.add_dll_directory(str(DEFAULT_HUB_LIB))
        lib = cdll.LoadLibrary(str(self.dll_path))
        self._bind(lib)
        self._lib = lib

        def _on_log(level, msg):
            text = _cstr(msg)
            self.log_lines.append(f"{int(level)} {text}")

        self._log_cb = _LogCallback(_on_log)
        lib.ms_set_logging_callback(self._log_cb)

        rc = lib.ms_init_2()
        if rc != MS_OK:
            rc = lib.ms_init()
        if rc != MS_OK:
            raise MuseSamplerError(
                f"ms_init failed: {rc}. MuseSampler 0.105 only initializes inside a "
                "Muse Hub SDK-enabled host (MuseScore Studio). A standalone Python "
                "process is rejected immediately; use MuseScore --sound-profile MuseSounds "
                "or run this host from an SDK-enabled binary."
            )

    def close(self) -> None:
        lib = self._lib
        if lib is None:
            return
        try:
            if self._sampler:
                if self._offline:
                    lib.ms_MuseSampler_stop_offline_mode(self._sampler)
                    self._offline = False
                lib.ms_MuseSampler_destroy(self._sampler)
        except Exception:
            pass
        self._sampler = None
        self._track = None
        try:
            lib.ms_deinit()
        except Exception:
            pass
        self._lib = None

    def version(self) -> str:
        lib = self._require_lib()
        parts = [
            lib.ms_get_version_major(),
            lib.ms_get_version_minor(),
            lib.ms_get_version_revision(),
            lib.ms_get_version_build_number(),
        ]
        text = ""
        try:
            raw = lib.ms_get_version_string()
            text = _cstr(raw)
        except Exception:
            pass
        return text or ".".join(str(p) for p in parts)

    def list_instruments(self) -> list[Instrument]:
        lib = self._require_lib()
        handle = lib.ms_get_instrument_list()
        instruments: list[Instrument] = []
        while True:
            info = lib.ms_InstrumentList_get_next(handle)
            if not info:
                break
            presets: list[str] = []
            plist = lib.ms_Instrument_get_preset_list(info)
            while True:
                preset = lib.ms_PresetList_get_next(plist)
                if not preset:
                    break
                presets.append(_cstr(preset))
            instruments.append(
                Instrument(
                    instrument_id=int(lib.ms_Instrument_get_id(info)),
                    name=_cstr(lib.ms_Instrument_get_name(info)),
                    category=_cstr(lib.ms_Instrument_get_category(info)),
                    pack=_cstr(lib.ms_Instrument_get_pack_name(info)),
                    vendor=_cstr(lib.ms_Instrument_get_vendor_name(info)),
                    musicxml_sound=_cstr(lib.ms_Instrument_get_musicxml_sound(info)),
                    mpe_sound=_cstr(lib.ms_Instrument_get_mpe_sound(info)),
                    online=bool(lib.ms_Instrument_is_online(info)),
                    presets=tuple(presets),
                )
            )
        return instruments

    def find_clarinet_bb(self, instruments: list[Instrument] | None = None) -> Instrument:
        items = instruments if instruments is not None else self.list_instruments()
        scored: list[tuple[int, Instrument]] = []
        for inst in items:
            blob = " ".join([inst.name, inst.pack, inst.category, inst.mpe_sound]).lower()
            if "clarinet" not in blob:
                continue
            score = 0
            if "clarinet in bb" in inst.name.lower() or inst.name.lower() == "clarinet in bb":
                score += 10
            if "woodwind" in inst.pack.lower() or "woodwind" in inst.category.lower():
                score += 3
            if "bb" in inst.name.lower() or "b-flat" in blob or "bflat" in blob:
                score += 2
            if not inst.online:
                score += 1
            scored.append((score, inst))
        if not scored:
            raise MuseSamplerError("No clarinet instrument found in MuseSampler pack list")
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[0][1]

    def create_sampler(self, instrument_id: int, preset: str | None = None) -> None:
        lib = self._require_lib()
        if self._sampler:
            lib.ms_MuseSampler_destroy(self._sampler)
            self._sampler = None
            self._track = None
            self._offline = False
        sampler = lib.ms_MuseSampler_create()
        if not sampler:
            raise MuseSamplerError("ms_MuseSampler_create returned NULL")
        rc = lib.ms_MuseSampler_init_2(
            sampler, c_double(self.sample_rate), c_int(self.block_size), c_int(AUDIO_CHANNELS)
        )
        if rc != MS_OK:
            rc = lib.ms_MuseSampler_init(
                sampler, c_double(self.sample_rate), c_int(self.block_size), c_int(AUDIO_CHANNELS)
            )
        if rc != MS_OK:
            lib.ms_MuseSampler_destroy(sampler)
            raise MuseSamplerError(f"ms_MuseSampler_init failed: {rc}")
        lib.ms_MuseSampler_set_lazy_render(sampler, False)
        lib.ms_MuseSampler_set_auto_render_interval(sampler, c_double(-1.0))
        track = lib.ms_MuseSampler_add_track(sampler, c_int(instrument_id))
        if not track:
            lib.ms_MuseSampler_destroy(sampler)
            raise MuseSamplerError(f"add_track failed for instrument_id={instrument_id}")
        if preset:
            change = lib.ms_MuseSampler_create_preset_change(sampler, track, c_longlong(0))
            lib.ms_MuseSampler_add_preset(sampler, track, change, preset.encode("utf-8"))
        self._sampler = sampler
        self._track = track
        self._prepare_bus(self.block_size)

    def clear_track(self) -> None:
        lib, sampler, track = self._require_session()
        lib.ms_MuseSampler_clear_track(sampler, track)

    def add_dynamics(self, location_us: int = 0, value: float = 0.62) -> None:
        lib, sampler, track = self._require_session()
        evt = DynamicsEvent2(_location_us=int(location_us), _value=float(value))
        rc = lib.ms_MuseSampler_add_track_dynamics_event_2(sampler, track, evt)
        if rc != MS_OK:
            raise MuseSamplerError(f"add dynamics failed: {rc}")

    def add_note(self, note: SamplerNote) -> int:
        lib, sampler, track = self._require_session()
        evt = NoteEvent5(
            _voice=int(note.voice),
            _location_us=int(note.start_us),
            _duration_us=int(max(1, note.duration_us)),
            _pitch=int(note.pitch),
            _tempo=float(note.tempo),
            _offset_cents=int(note.cents),
            _articulation=0,
            _articulation_2=0,
            _notehead=0,
        )
        event_id = c_longlong(0)
        rc = lib.ms_MuseSampler_add_track_note_event_6(sampler, track, evt, byref(event_id))
        if rc != MS_OK:
            raise MuseSamplerError(
                f"add note failed: pitch={note.pitch} t={note.start_us} rc={rc}"
            )
        return int(event_id.value)

    def finalize_track(self) -> None:
        lib, sampler, track = self._require_session()
        rc = lib.ms_MuseSampler_finalize_track(sampler, track)
        if rc != MS_OK:
            raise MuseSamplerError(f"finalize_track failed: {rc}")
        lib.ms_MuseSampler_trigger_render(sampler)

    def ready_to_play(self) -> bool:
        lib, sampler, _track = self._require_session()
        return bool(lib.ms_MuseSampler_ready_to_play(sampler))

    def wait_ready(self, timeout_sec: float = 45.0) -> bool:
        import time

        deadline = time.perf_counter() + float(timeout_sec)
        while time.perf_counter() < deadline:
            if self.ready_to_play():
                return True
            time.sleep(0.02)
        return self.ready_to_play()

    def render(self, duration_sec: float, tail_sec: float = 1.5) -> np.ndarray:
        """Return stereo float32 of shape (N, 2)."""
        total_sec = max(0.05, float(duration_sec) + float(tail_sec))
        total = int(round(total_sec * self.sample_rate))
        try:
            return self._render_offline(total)
        except MuseSamplerError:
            return self._render_realtime(total)

    def render_notes(
        self,
        notes: list[SamplerNote],
        instrument_id: int,
        preset: str | None = None,
        dynamics: float = 0.62,
        tail_sec: float = 1.5,
        reuse_sampler: bool = True,
    ) -> np.ndarray:
        if not notes:
            raise MuseSamplerError("No notes to render")
        if not reuse_sampler or self._sampler is None or self._track is None:
            self.create_sampler(instrument_id, preset=preset)
        else:
            self.clear_track()
        self.add_dynamics(0, dynamics)
        end_us = 0
        for note in notes:
            self.add_note(note)
            end_us = max(end_us, note.start_us + note.duration_us)
        self.finalize_track()
        self.wait_ready()
        return self.render(end_us / 1_000_000.0, tail_sec=tail_sec)

    def _render_offline(self, total_samples: int) -> np.ndarray:
        lib, sampler, _track = self._require_session()
        if not self._offline:
            rc = lib.ms_MuseSampler_start_offline_mode(sampler, c_double(self.sample_rate))
            if rc != MS_OK:
                raise MuseSamplerError(f"start_offline_mode failed: {rc}")
            self._offline = True
        lib.ms_MuseSampler_set_position(sampler, c_longlong(0))
        lib.ms_MuseSampler_set_playing(sampler, c_int(1))
        chunks: list[np.ndarray] = []
        remaining = int(total_samples)
        while remaining > 0:
            n = min(self.block_size, remaining)
            self._prepare_bus(n)
            rc = lib.ms_MuseSampler_process_offline(sampler, self._bus)
            if rc != MS_OK:
                raise MuseSamplerError(f"process_offline failed: {rc}")
            chunks.append(self._read_bus(n))
            remaining -= n
        lib.ms_MuseSampler_set_playing(sampler, c_int(0))
        return np.concatenate(chunks, axis=0)

    def _render_realtime(self, total_samples: int) -> np.ndarray:
        lib, sampler, _track = self._require_session()
        if self._offline:
            lib.ms_MuseSampler_stop_offline_mode(sampler)
            self._offline = False
        lib.ms_MuseSampler_set_position(sampler, c_longlong(0))
        lib.ms_MuseSampler_set_playing(sampler, c_int(1))
        chunks: list[np.ndarray] = []
        pos = 0
        remaining = int(total_samples)
        while remaining > 0:
            n = min(self.block_size, remaining)
            self._prepare_bus(n)
            rc = lib.ms_MuseSampler_process(sampler, self._bus, c_longlong(pos))
            if rc != MS_OK:
                raise MuseSamplerError(f"process failed: {rc} pos={pos}")
            chunks.append(self._read_bus(n))
            pos += n
            remaining -= n
        lib.ms_MuseSampler_set_playing(sampler, c_int(0))
        return np.concatenate(chunks, axis=0)

    def _prepare_bus(self, n: int) -> None:
        if self._left is None or len(self._left) < n:
            self._left = (c_float * max(n, self.block_size))()
            self._right = (c_float * max(n, self.block_size))()
            self._chan_ptrs = (POINTER(c_float) * AUDIO_CHANNELS)(self._left, self._right)
            self._bus._channels = self._chan_ptrs
            self._bus._num_channels = AUDIO_CHANNELS
        self._bus._num_data_pts = int(n)

    def _read_bus(self, n: int) -> np.ndarray:
        left = np.ctypeslib.as_array(self._left)[:n].copy()
        right = np.ctypeslib.as_array(self._right)[:n].copy()
        return np.stack([left, right], axis=1)

    def _bind(self, lib) -> None:
        lib.ms_get_version_major.restype = c_int
        lib.ms_get_version_minor.restype = c_int
        lib.ms_get_version_revision.restype = c_int
        lib.ms_get_version_build_number.restype = c_int
        lib.ms_get_version_string.restype = c_char_p
        lib.ms_init.restype = c_int
        lib.ms_init_2.restype = c_int
        lib.ms_deinit.restype = c_int
        lib.ms_set_logging_callback.argtypes = [_LogCallback]
        lib.ms_set_logging_callback.restype = None
        lib.ms_get_instrument_list.restype = c_void_p
        lib.ms_InstrumentList_get_next.argtypes = [c_void_p]
        lib.ms_InstrumentList_get_next.restype = c_void_p
        lib.ms_Instrument_get_id.argtypes = [c_void_p]
        lib.ms_Instrument_get_id.restype = c_int
        lib.ms_Instrument_get_name.argtypes = [c_void_p]
        lib.ms_Instrument_get_name.restype = c_char_p
        lib.ms_Instrument_get_category.argtypes = [c_void_p]
        lib.ms_Instrument_get_category.restype = c_char_p
        lib.ms_Instrument_get_pack_name.argtypes = [c_void_p]
        lib.ms_Instrument_get_pack_name.restype = c_char_p
        lib.ms_Instrument_get_vendor_name.argtypes = [c_void_p]
        lib.ms_Instrument_get_vendor_name.restype = c_char_p
        lib.ms_Instrument_get_musicxml_sound.argtypes = [c_void_p]
        lib.ms_Instrument_get_musicxml_sound.restype = c_char_p
        lib.ms_Instrument_get_mpe_sound.argtypes = [c_void_p]
        lib.ms_Instrument_get_mpe_sound.restype = c_char_p
        lib.ms_Instrument_is_online.argtypes = [c_void_p]
        lib.ms_Instrument_is_online.restype = c_bool
        lib.ms_Instrument_get_preset_list.argtypes = [c_void_p]
        lib.ms_Instrument_get_preset_list.restype = c_void_p
        lib.ms_PresetList_get_next.argtypes = [c_void_p]
        lib.ms_PresetList_get_next.restype = c_char_p
        lib.ms_MuseSampler_create.restype = c_void_p
        lib.ms_MuseSampler_destroy.argtypes = [c_void_p]
        lib.ms_MuseSampler_init.argtypes = [c_void_p, c_double, c_int, c_int]
        lib.ms_MuseSampler_init.restype = c_int
        lib.ms_MuseSampler_init_2.argtypes = [c_void_p, c_double, c_int, c_int]
        lib.ms_MuseSampler_init_2.restype = c_int
        lib.ms_MuseSampler_add_track.argtypes = [c_void_p, c_int]
        lib.ms_MuseSampler_add_track.restype = c_void_p
        lib.ms_MuseSampler_clear_track.argtypes = [c_void_p, c_void_p]
        lib.ms_MuseSampler_clear_track.restype = c_int
        lib.ms_MuseSampler_finalize_track.argtypes = [c_void_p, c_void_p]
        lib.ms_MuseSampler_finalize_track.restype = c_int
        lib.ms_MuseSampler_add_track_note_event_6.argtypes = [
            c_void_p,
            c_void_p,
            NoteEvent5,
            POINTER(c_longlong),
        ]
        lib.ms_MuseSampler_add_track_note_event_6.restype = c_int
        lib.ms_MuseSampler_add_track_dynamics_event_2.argtypes = [
            c_void_p,
            c_void_p,
            DynamicsEvent2,
        ]
        lib.ms_MuseSampler_add_track_dynamics_event_2.restype = c_int
        lib.ms_MuseSampler_create_preset_change.argtypes = [c_void_p, c_void_p, c_longlong]
        lib.ms_MuseSampler_create_preset_change.restype = c_int
        lib.ms_MuseSampler_add_preset.argtypes = [c_void_p, c_void_p, c_int, c_char_p]
        lib.ms_MuseSampler_add_preset.restype = c_int
        lib.ms_MuseSampler_start_offline_mode.argtypes = [c_void_p, c_double]
        lib.ms_MuseSampler_start_offline_mode.restype = c_int
        lib.ms_MuseSampler_stop_offline_mode.argtypes = [c_void_p]
        lib.ms_MuseSampler_stop_offline_mode.restype = c_int
        lib.ms_MuseSampler_process_offline.argtypes = [c_void_p, OutputBuffer]
        lib.ms_MuseSampler_process_offline.restype = c_int
        lib.ms_MuseSampler_process.argtypes = [c_void_p, OutputBuffer, c_longlong]
        lib.ms_MuseSampler_process.restype = c_int
        lib.ms_MuseSampler_set_position.argtypes = [c_void_p, c_longlong]
        lib.ms_MuseSampler_set_position.restype = None
        lib.ms_MuseSampler_set_playing.argtypes = [c_void_p, c_int]
        lib.ms_MuseSampler_set_playing.restype = None
        lib.ms_MuseSampler_ready_to_play.argtypes = [c_void_p]
        lib.ms_MuseSampler_ready_to_play.restype = c_bool
        lib.ms_MuseSampler_trigger_render.argtypes = [c_void_p]
        lib.ms_MuseSampler_trigger_render.restype = None
        lib.ms_MuseSampler_set_lazy_render.argtypes = [c_void_p, c_bool]
        lib.ms_MuseSampler_set_lazy_render.restype = None
        lib.ms_MuseSampler_set_auto_render_interval.argtypes = [c_void_p, c_double]
        lib.ms_MuseSampler_set_auto_render_interval.restype = None
        lib.ms_MuseSampler_all_notes_off.argtypes = [c_void_p]
        lib.ms_MuseSampler_all_notes_off.restype = c_int

    def _require_lib(self):
        if self._lib is None:
            raise MuseSamplerError("Host is not open")
        return self._lib

    def _require_session(self):
        lib = self._require_lib()
        if self._sampler is None or self._track is None:
            raise MuseSamplerError("Sampler/track not created")
        return lib, self._sampler, self._track


def midi_to_sampler_notes(midi_path: Path | str, tempo: float = 120.0) -> list[SamplerNote]:
    from tinysoundfont.midi import load

    from synthpipeline.midi_player import events_to_notes

    events = load(str(midi_path), persistent=False)
    if not events:
        raise MuseSamplerError(f"No MIDI events in {midi_path}")
    played = events_to_notes(events, tail_seconds=0.0)
    notes: list[SamplerNote] = []
    for item in played:
        dur = max(0.03, float(item.end) - float(item.start))
        notes.append(
            SamplerNote(
                pitch=int(item.key),
                start_us=int(round(item.start * 1_000_000)),
                duration_us=int(round(dur * 1_000_000)),
                cents=int(round(item.cents)),
                tempo=float(tempo),
            )
        )
    return notes


def peak_normalize(stereo: np.ndarray, peak: float = 0.89) -> np.ndarray:
    mag = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if mag <= 1e-8:
        return stereo
    gain = min(peak / mag, 4.0)
    return (stereo * gain).astype(np.float32)
