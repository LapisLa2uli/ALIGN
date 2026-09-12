from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .data import FRAME_HOP_SEC
from .model import NoteFrameNet, NoteFrameNetConfig


@dataclass(frozen=True)
class TransNote:
    pitch: int
    start: float
    end: float
    confidence: float
    cents: float = 0.0
    pitch_candidates: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DecodeConfig:
    voiced_threshold: float = 0.50
    onset_threshold: float = 0.45
    offset_threshold: float = 0.40
    min_note_sec: float = 0.055
    max_gap_frames: int = 1
    pitch_change_frames: int = 8

    def to_dict(self) -> dict:
        return asdict(self)


def _close_short_gaps(active: np.ndarray, max_gap: int) -> np.ndarray:
    out = active.copy()
    if max_gap <= 0:
        return out
    i = 0
    while i < len(out):
        if out[i]:
            i += 1
            continue
        j = i
        while j < len(out) and not out[j]:
            j += 1
        if i > 0 and j < len(out) and j - i <= max_gap:
            out[i:j] = True
        i = j
    return out


def _local_peaks(values: np.ndarray, threshold: float) -> list[int]:
    if len(values) == 0:
        return []
    peaks = []
    for i, value in enumerate(values):
        left = values[i - 1] if i else -np.inf
        right = values[i + 1] if i + 1 < len(values) else -np.inf
        if value >= threshold and value >= left and value >= right:
            peaks.append(i)
    return peaks


def decode_notes(
    probabilities: dict[str, np.ndarray],
    *,
    midi_min: int = 36,
    hop_sec: float = FRAME_HOP_SEC,
    config: DecodeConfig | None = None,
) -> list[TransNote]:
    """Convert merged frame probabilities into monophonic written notes."""

    cfg = config or DecodeConfig()
    voiced = np.asarray(probabilities["voiced"], dtype=np.float32)
    pitch_probs = np.asarray(probabilities["pitch"], dtype=np.float32)
    onset = np.asarray(probabilities["onset"], dtype=np.float32)
    offset = np.asarray(probabilities["offset"], dtype=np.float32)
    if pitch_probs.ndim != 2 or pitch_probs.shape[0] != len(voiced):
        raise ValueError("pitch probabilities must have shape [T, n_pitches]")
    if not (len(onset) == len(offset) == len(voiced)):
        raise ValueError("all probability heads must have the same frame count")

    active = _close_short_gaps(voiced >= cfg.voiced_threshold, cfg.max_gap_frames)
    pitch_idx = pitch_probs.argmax(axis=-1)
    onset_peaks = _local_peaks(onset, cfg.onset_threshold)
    min_frames = max(1, int(round(cfg.min_note_sec / hop_sec)))
    notes: list[TransNote] = []
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        run_start = i
        while i < len(active) and active[i]:
            i += 1
        run_end = i
        splits = {run_start, run_end}
        for peak in onset_peaks:
            if run_start + min_frames <= peak <= run_end - min_frames:
                splits.add(peak)

        # A sustained pitch change can recover legato notes with weak onsets.
        p = run_start + 1
        while p + cfg.pitch_change_frames <= run_end:
            previous = int(pitch_idx[p - 1])
            candidate = int(pitch_idx[p])
            if (
                candidate != previous
                and np.all(pitch_idx[p : p + cfg.pitch_change_frames] == candidate)
                and p - run_start >= min_frames
            ):
                splits.add(p)
                p += cfg.pitch_change_frames
            else:
                p += 1

        boundaries = sorted(splits)
        for start_i, nominal_end in zip(boundaries[:-1], boundaries[1:]):
            if nominal_end - start_i < min_frames:
                continue
            weights = voiced[start_i:nominal_end, None]
            pitch_score = (pitch_probs[start_i:nominal_end] * weights).sum(axis=0)
            cls = int(pitch_score.argmax())

            end_i = nominal_end
            search_lo = max(start_i + min_frames, nominal_end - 3)
            search_hi = min(len(offset), nominal_end + 4)
            if search_hi > search_lo:
                rel = int(offset[search_lo:search_hi].argmax())
                candidate_end = search_lo + rel
                if offset[candidate_end] >= cfg.offset_threshold:
                    end_i = max(start_i + 1, candidate_end)
            if end_i <= start_i:
                continue
            voice_conf = float(voiced[start_i:nominal_end].mean())
            pitch_conf = float(pitch_probs[start_i:nominal_end, cls].mean())
            onset_conf = float(onset[start_i])
            confidence = float(
                np.clip(0.45 * voice_conf + 0.45 * pitch_conf + 0.10 * onset_conf, 0, 1)
            )
            cents_value = 0.0
            if "cents" in probabilities:
                cents_arr = np.asarray(probabilities["cents"], dtype=np.float32)
                cents_value = float(cents_arr[start_i:nominal_end].mean())
            ranked = np.argsort(pitch_score)[::-1][:3]
            candidates = tuple(int(midi_min + idx) for idx in ranked if pitch_score[idx] > 0)
            if end_i <= start_i:
                continue
            notes.append(
                TransNote(
                    pitch=midi_min + cls,
                    start=round(start_i * hop_sec, 6),
                    end=round(max((start_i + 1) * hop_sec, end_i * hop_sec), 6),
                    confidence=round(confidence, 6),
                    cents=round(cents_value, 2),
                    pitch_candidates=candidates or (midi_min + cls,),
                )
            )
    return _valid_note_sequence(notes, hop_sec=hop_sec, min_note_sec=cfg.min_note_sec)


def _valid_note_sequence(
    notes: list[TransNote],
    *,
    hop_sec: float,
    min_note_sec: float,
) -> list[TransNote]:
    """Keep onset/offset order, minimum duration, and non-overlapping notes."""

    ordered = sorted(
        (note for note in notes if note.end > note.start),
        key=lambda note: (note.start, note.end),
    )
    cleaned: list[TransNote] = []
    for note in ordered:
        if cleaned and note.start < cleaned[-1].end:
            previous = cleaned[-1]
            clipped_end = max(previous.start + hop_sec, note.start)
            if clipped_end - previous.start < min_note_sec:
                cleaned.pop()
            else:
                cleaned[-1] = TransNote(
                    pitch=previous.pitch,
                    start=previous.start,
                    end=round(clipped_end, 6),
                    confidence=previous.confidence,
                    cents=previous.cents,
                    pitch_candidates=previous.pitch_candidates,
                )
        if note.end - note.start >= min_note_sec:
            if cleaned and note.start < cleaned[-1].end:
                note = TransNote(
                    pitch=note.pitch,
                    start=round(cleaned[-1].end, 6),
                    end=note.end,
                    confidence=note.confidence,
                    cents=note.cents,
                    pitch_candidates=note.pitch_candidates,
                )
            if note.end - note.start >= min_note_sec:
                cleaned.append(note)
    return cleaned


@torch.no_grad()
def infer_probabilities(
    model: NoteFrameNet,
    mel: np.ndarray,
    device: torch.device | str,
    *,
    window_frames: int = 1024,
    overlap_frames: int = 256,
    batch_size: int = 4,
    f0: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Sliding-window inference with tapered overlap averaging."""

    arr = np.asarray(mel, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D mel, got {arr.shape}")
    if arr.shape[0] != model.config.n_mels and arr.shape[1] == model.config.n_mels:
        arr = arr.T
    if arr.shape[0] != model.config.n_mels:
        raise ValueError(
            f"Expected {model.config.n_mels}xT mel, got {arr.shape}"
        )
    total = arr.shape[1]
    empty = {
        "voiced": np.zeros(0, np.float32),
        "pitch": np.zeros((0, model.config.n_pitches), np.float32),
        "onset": np.zeros(0, np.float32),
        "offset": np.zeros(0, np.float32),
    }
    if model.config.predict_cents:
        empty["cents"] = np.zeros(0, np.float32)
    if total == 0:
        return empty
    if overlap_frames >= window_frames:
        raise ValueError("overlap_frames must be less than window_frames")
    stride = window_frames - overlap_frames
    starts = list(range(0, max(total - window_frames, 0) + 1, stride))
    last = max(0, total - window_frames)
    if not starts or starts[-1] != last:
        starts.append(last)

    sums = {
        "voiced": np.zeros(total, np.float64),
        "pitch": np.zeros((total, model.config.n_pitches), np.float64),
        "onset": np.zeros(total, np.float64),
        "offset": np.zeros(total, np.float64),
        "cents": np.zeros(total, np.float64),
    }
    f0_arr = None
    if f0 is not None:
        f0_arr = np.asarray(f0, dtype=np.float32)
        if f0_arr.ndim == 1:
            f0_arr = np.stack([f0_arr, np.ones_like(f0_arr)], axis=0)
        if f0_arr.shape[0] != 2 and f0_arr.shape[-1] == 2:
            f0_arr = f0_arr.T
    weight_sum = np.zeros(total, np.float64)
    taper = np.hanning(window_frames + 2)[1:-1].astype(np.float32)
    taper = np.maximum(taper, 0.05)
    model.eval()
    device = torch.device(device)
    for batch_start in range(0, len(starts), batch_size):
        group = starts[batch_start : batch_start + batch_size]
        windows = np.full(
            (len(group), model.config.n_mels, window_frames), -80.0, np.float32
        )
        f0_windows = np.zeros((len(group), 2, window_frames), np.float32)
        lengths = []
        for b, start in enumerate(group):
            length = min(window_frames, total - start)
            windows[b, :, :length] = arr[:, start : start + length]
            if f0_arr is not None:
                take = min(length, f0_arr.shape[1] - start)
                if take > 0:
                    f0_windows[b, :, :take] = f0_arr[:, start : start + take]
            lengths.append(length)
        f0_t = (
            torch.from_numpy(f0_windows).to(device)
            if getattr(model.config, "use_f0", False)
            else None
        )
        out = model(torch.from_numpy(windows).to(device), f0_t)
        probs = {
            "voiced": out["voiced_logits"].sigmoid().cpu().numpy(),
            "pitch": out["pitch_logits"].softmax(dim=-1).cpu().numpy(),
            "onset": out["onset_logits"].sigmoid().cpu().numpy(),
            "offset": out["offset_logits"].sigmoid().cpu().numpy(),
        }
        if "cents" in out:
            probs["cents"] = out["cents"].cpu().numpy()
        for b, (start, length) in enumerate(zip(group, lengths)):
            sl = slice(start, start + length)
            weight = taper[:length]
            weight_sum[sl] += weight
            sums["voiced"][sl] += probs["voiced"][b, :length] * weight
            sums["onset"][sl] += probs["onset"][b, :length] * weight
            sums["offset"][sl] += probs["offset"][b, :length] * weight
            sums["pitch"][sl] += probs["pitch"][b, :length] * weight[:, None]
            if "cents" in probs:
                sums["cents"][sl] += probs["cents"][b, :length] * weight
    denom = np.maximum(weight_sum, 1e-8)
    result = {
        "voiced": (sums["voiced"] / denom).astype(np.float32),
        "pitch": (sums["pitch"] / denom[:, None]).astype(np.float32),
        "onset": (sums["onset"] / denom).astype(np.float32),
        "offset": (sums["offset"] / denom).astype(np.float32),
    }
    if "cents" in sums:
        result["cents"] = (sums["cents"] / denom).astype(np.float32)
    return result


def infer_full_clip(
    model: NoteFrameNet,
    mel_or_path: np.ndarray | Path | str,
    device: torch.device | str,
    *,
    decode_config: DecodeConfig | None = None,
    window_frames: int = 2048,
    overlap_frames: int = 512,
    batch_size: int = 4,
    f0: np.ndarray | None = None,
) -> list[TransNote]:
    sample_dir = None
    if isinstance(mel_or_path, (str, Path)):
        path = Path(mel_or_path)
        sample_dir = path.parent if path.name == "performance_mel.npy" else path
        mel = np.load(
            path if path.name.endswith(".npy") else path / "performance_mel.npy",
            mmap_mode="r",
        )
        if f0 is None and sample_dir is not None:
            from .data import load_fine_pitch

            f0 = load_fine_pitch(sample_dir, int(np.asarray(mel).shape[-1]))
    else:
        mel = mel_or_path
    probs = infer_probabilities(
        model,
        np.asarray(mel),
        device,
        window_frames=window_frames,
        overlap_frames=overlap_frames,
        batch_size=batch_size,
        f0=f0,
    )
    return decode_notes(
        probs,
        midi_min=model.config.midi_min,
        config=decode_config,
    )


def _written_pitch_shift(sample_dir: Path) -> int:
    """Undo the clarinet transpose for pitch measured from the WAV."""

    from synthpipeline.pitch_convention import (
        audio_to_written_shift,
        load_bundle_metadata,
    )

    return audio_to_written_shift(load_bundle_metadata(sample_dir))


def spectral_pitch_frames(
    wav_path: Path | str,
    *,
    sample_rate: int = 22050,
    hop_length: int = 512,
    written_shift: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one robust monophonic MIDI candidate and strength per frame."""
    import librosa

    from alignmodel.audio import load_mono

    audio = load_mono(Path(wav_path), sample_rate)
    pitches, magnitudes = librosa.piptrack(
        y=audio,
        sr=sample_rate,
        n_fft=4096,
        hop_length=hop_length,
        fmin=50.0,
        fmax=4200.0,
    )
    best = magnitudes.argmax(axis=0)
    columns = np.arange(pitches.shape[1])
    hz = pitches[best, columns]
    strength = magnitudes[best, columns].astype(np.float32)
    midi = np.rint(librosa.hz_to_midi(np.maximum(hz, 1.0))).astype(np.int16)
    midi += int(written_shift)
    return midi, strength


def apply_spectral_pitch(
    notes: list[TransNote],
    frame_pitch: np.ndarray,
    frame_strength: np.ndarray,
    *,
    hop_sec: float = FRAME_HOP_SEC,
    min_confidence: float = 0.55,
    only_low_confidence: bool = True,
) -> list[TransNote]:
    """Optionally replace low-confidence neural pitch inside neural boundaries."""
    output: list[TransNote] = []
    n_frames = len(frame_pitch)
    for value in notes:
        if only_low_confidence and value.confidence >= min_confidence:
            output.append(value)
            continue
        i0 = max(0, min(n_frames, int(value.start / hop_sec) + 1))
        i1 = max(i0 + 1, min(n_frames, int(np.ceil(value.end / hop_sec)) - 1))
        pitches = frame_pitch[i0:i1]
        strengths = frame_strength[i0:i1]
        if pitches.size and np.any(strengths > 0):
            values, counts = np.unique(pitches, return_counts=True)
            chosen = int(values[int(np.argmax(counts))])
            spectral_conf = float(np.max(counts) / max(len(pitches), 1))
            if spectral_conf < 0.45:
                output.append(value)
                continue
            confidence = float(
                np.clip(0.65 * value.confidence + 0.35 * spectral_conf, 0.0, 1.0)
            )
        else:
            chosen = value.pitch
            confidence = value.confidence
        output.append(
            TransNote(
                pitch=int(chosen),
                start=value.start,
                end=value.end,
                confidence=round(confidence, 6),
                cents=value.cents,
                pitch_candidates=value.pitch_candidates or (int(chosen),),
            )
        )
    return output


def infer_sample_notes(
    model: NoteFrameNet,
    sample_dir: Path | str,
    device: torch.device | str,
    *,
    decode_config: DecodeConfig | None = None,
    fusion: str = "neural",
    window_frames: int = 2048,
    overlap_frames: int = 512,
    batch_size: int = 4,
) -> list[TransNote]:
    """Canonical frontend: calibrated neural notes, optional conservative fusion."""

    sample_dir = Path(sample_dir)
    notes = infer_full_clip(
        model,
        sample_dir / "performance_mel.npy",
        device,
        decode_config=decode_config,
        window_frames=window_frames,
        overlap_frames=overlap_frames,
        batch_size=batch_size,
    )
    mode = str(fusion or "neural").lower()
    if mode in {"neural", "none", ""}:
        return notes
    wav = sample_dir / "performance_audio.wav"
    if mode == "hybrid":
        return infer_sample_notes_hybrid(
            model, sample_dir, device, decode_config=decode_config
        )
    if not wav.exists():
        return notes
    frame_pitch, frame_strength = spectral_pitch_frames(
        wav, written_shift=_written_pitch_shift(sample_dir)
    )
    return apply_spectral_pitch(
        notes,
        frame_pitch,
        frame_strength,
        only_low_confidence=True,
    )


def infer_sample_notes_hybrid(
    model: NoteFrameNet,
    sample_dir: Path | str,
    device: torch.device | str,
    *,
    decode_config: DecodeConfig | None = None,
) -> list[TransNote]:
    """Legacy Librosa-onset decoder, kept only for evaluation ablations."""

    sample_dir = Path(sample_dir)
    from .data import load_fine_pitch

    mel = np.load(sample_dir / "performance_mel.npy", mmap_mode="r")
    probabilities = infer_probabilities(
        model,
        np.asarray(mel),
        device,
        f0=load_fine_pitch(sample_dir, int(np.asarray(mel).shape[-1])),
    )
    wav = sample_dir / "performance_audio.wav"
    if not wav.exists():
        return decode_notes(
            probabilities,
            midi_min=model.config.midi_min,
            config=decode_config,
        )

    import librosa

    from alignmodel.audio import load_mono

    audio = load_mono(wav, 22050)
    pitches, magnitudes = librosa.piptrack(
        y=audio,
        sr=22050,
        n_fft=4096,
        hop_length=512,
        fmin=50.0,
        fmax=4200.0,
    )
    best = magnitudes.argmax(axis=0)
    columns = np.arange(pitches.shape[1])
    frame_pitch = np.rint(
        librosa.hz_to_midi(np.maximum(pitches[best, columns], 1.0))
    ).astype(np.int16)
    frame_pitch += _written_pitch_shift(sample_dir)
    onset_envelope = librosa.onset.onset_strength(
        y=audio, sr=22050, hop_length=512
    )
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=22050,
        hop_length=512,
        units="frames",
        delta=0.05,
        wait=4,
        pre_max=1,
        post_max=1,
        pre_avg=3,
        post_avg=3,
    )
    n_frames = min(len(frame_pitch), len(probabilities["voiced"]))
    boundaries = sorted(
        set(int(value) for value in onsets if 0 <= int(value) < n_frames)
        | {n_frames}
    )
    notes: list[TransNote] = []
    for start_i, end_i in zip(boundaries[:-1], boundaries[1:]):
        if end_i - start_i < 3:
            continue
        values = frame_pitch[start_i + 1 : max(start_i + 2, end_i - 1)]
        if not values.size:
            continue
        unique, counts = np.unique(values, return_counts=True)
        chosen_i = int(np.argmax(counts))
        chosen = int(unique[chosen_i])
        spectral_conf = float(counts[chosen_i] / max(len(values), 1))
        voiced_conf = float(probabilities["voiced"][start_i:end_i].mean())
        onset_conf = float(
            probabilities["onset"][
                max(0, start_i - 1) : min(n_frames, start_i + 2)
            ].max(initial=0.0)
        )
        confidence = float(
            np.clip(
                0.45 * spectral_conf + 0.40 * voiced_conf + 0.15 * onset_conf,
                0.0,
                1.0,
            )
        )
        notes.append(
            TransNote(
                pitch=chosen,
                start=round(start_i * FRAME_HOP_SEC, 6),
                end=round(end_i * FRAME_HOP_SEC, 6),
                confidence=round(confidence, 6),
            )
        )
    if notes:
        return notes
    return decode_notes(
        probabilities,
        midi_min=model.config.midi_min,
        config=decode_config,
    )


def load_note_transcriber(
    checkpoint: Path | str, device: torch.device | str = "cpu"
) -> tuple[NoteFrameNet, DecodeConfig]:
    payload = torch.load(Path(checkpoint), map_location=torch.device(device), weights_only=False)
    model = NoteFrameNet(NoteFrameNetConfig.from_dict(payload.get("model_config") or {}))
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    calibration = payload.get("calibration") or {}
    allowed = {item.name for item in fields(DecodeConfig)}
    return model, DecodeConfig(
        **{key: value for key, value in calibration.items() if key in allowed}
    )
