"""Fine-tune the Basic Pitch 0.4.0 TensorFlow heads on exact synth targets.

The input manifest is deliberately mandatory: this script never discovers
bundles by scanning a dataset root.  A typical CPU smoke run is:

  python scripts/finetune_basic_pitch.py --manifest splits/frozen.json \
    --out runs/basic-pitch-head --procedural-only --smoke

TensorFlow uses a visible CUDA GPU automatically.  ``--smoke`` uses CPU when
``--device auto`` so the same command remains small and deterministic on
Windows.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

# Keep these local so target/manifest tests do not import TensorFlow through
# basic_pitch.  They are the public constants in Basic Pitch 0.4.0.
BASIC_PITCH_VERSION = "0.4.0"
AUDIO_SAMPLE_RATE = 22050
AUDIO_N_SAMPLES = 43844
ANNOTATIONS_FPS = 86
ANNOT_N_FRAMES = 172
N_FREQ_BINS_NOTES = 88
N_FREQ_BINS_CONTOURS = 264
MIDI_OFFSET = 21
CONTOUR_BINS_PER_SEMITONE = 3


@dataclass(frozen=True)
class WindowRecord:
    sample_dir: Path
    row: Mapping[str, Any]
    start_sample: int


def _manifest_rows(document: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(document, list):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for value in document:
            row = dict(value) if isinstance(value, dict) else {"sample": value}
            grouped.setdefault(str(row.get("split", "train")), []).append(row)
        return grouped
    if not isinstance(document, dict):
        raise ValueError("Frozen manifest must be a JSON object, list, or JSONL")
    grouped = {}
    for split, values in document.items():
        if split in {"roots", "version", "metadata", "policy", "distribution", "discovered"}:
            continue
        if isinstance(values, list):
            grouped[str(split)] = [
                dict(value) if isinstance(value, dict) else {"sample": value}
                for value in values
            ]
    return grouped


def _looks_raw2k(row: Mapping[str, Any]) -> bool:
    fields = (
        row.get("corpus"),
        row.get("root"),
        row.get("sample_dir"),
        row.get("path"),
        row.get("audio_path"),
        row.get("note_map"),
    )
    normalized = " ".join(str(value).replace("\\", "/").lower() for value in fields if value is not None)
    return "raw2k" in normalized or "output_2k_rawdata" in normalized


def load_frozen_manifest(path: Path | str, *, procedural_only: bool = False) -> dict[str, Any]:
    """Load a fixed manifest without touching any path named by its rows."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        document: Any = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        document = json.loads(text)
    rows = _manifest_rows(document)
    if "train" not in rows or not rows["train"]:
        raise ValueError("Frozen manifest has no non-empty train split")
    validation = rows.get("val", rows.get("validation", []))
    if not validation:
        raise ValueError("Frozen manifest has no non-empty val/validation split")
    if procedural_only:
        raw = [
            (split, index, row)
            for split, split_rows in rows.items()
            for index, row in enumerate(split_rows)
            if _looks_raw2k(row)
        ]
        if raw:
            split, index, row = raw[0]
            raise ValueError(
                "--procedural-only rejected raw2k manifest row "
                f"{split}[{index}]: {row}"
            )
    return {
        "path": path,
        "document": document if isinstance(document, dict) else {},
        "splits": rows,
    }


def _parse_root_overrides(values: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--root must be NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name.strip() or not raw_path.strip():
            raise ValueError(f"--root must be NAME=PATH, got {value!r}")
        roots[name.strip()] = Path(raw_path)
    return roots


def resolve_sample_dir(
    row: Mapping[str, Any],
    manifest_path: Path,
    manifest_roots: Mapping[str, Any],
    root_overrides: Mapping[str, Path],
) -> Path:
    raw = row.get("sample_dir", row.get("path", row.get("sample", row.get("id"))))
    if raw is None:
        raise ValueError(f"Manifest row lacks sample_dir/path/sample/id: {row}")
    candidate = Path(str(raw))
    if candidate.is_absolute():
        return candidate

    root_hint = row.get("root", row.get("corpus"))
    roots = dict(manifest_roots)
    roots.update({key: str(value) for key, value in root_overrides.items()})
    if root_hint is not None and str(root_hint) in roots:
        return Path(str(roots[str(root_hint)])) / candidate
    if len(root_overrides) == 1:
        return next(iter(root_overrides.values())) / candidate
    return manifest_path.parent / candidate


def _row_key(row: Mapping[str, Any]) -> str:
    return str(row.get("sample_dir", row.get("path", row.get("sample", row.get("id", row)))))


def deterministic_limit(
    rows: Sequence[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    return sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{_row_key(row)}".encode("utf-8")).digest(),
            _row_key(row),
        ),
    )[:limit]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _explicit_audio_transpose(
    row: Mapping[str, Any], metadata: Mapping[str, Any], metadata_path: Path
) -> int:
    if "effective_audio_transpose" in row:
        raw = row["effective_audio_transpose"]
    elif "effective_audio_transpose" in metadata:
        raw = metadata["effective_audio_transpose"]
    else:
        raise ValueError(
            "Missing explicit effective_audio_transpose in manifest row or "
            f"{metadata_path}; no pitch-space default is permitted"
        )
    value = float(raw)
    if not value.is_integer():
        raise ValueError(f"effective_audio_transpose must be whole semitones, got {raw!r}")
    return int(value)


def load_target_documents(
    sample_dir: Path | str, row: Mapping[str, Any] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Load only exact rendered events, labels, and explicit pitch metadata."""

    sample_dir = Path(sample_dir)
    row = row or {}
    note_map_path = Path(str(row.get("note_map", sample_dir / "note_map.json")))
    labels_path = Path(str(row.get("labels_path", sample_dir / "labels.json")))
    metadata_path = Path(str(row.get("metadata_path", sample_dir / "metadata.json")))
    note_map = _json(note_map_path)
    if note_map.get("kind") != "synth_note_lineage":
        raise ValueError(f"{note_map_path} is not an exact synth_note_lineage map")
    rendered = note_map.get("rendered_notes")
    if not isinstance(rendered, list) or not rendered:
        raise ValueError(f"{note_map_path} has no exact rendered_notes")
    labels_doc = _json(labels_path)
    labels = labels_doc.get("labels") or []
    if not isinstance(labels, list):
        raise ValueError(f"{labels_path} labels must be a list")
    metadata = _json(metadata_path)
    transpose = _explicit_audio_transpose(row, metadata, metadata_path)
    return [dict(note) for note in rendered], [dict(label) for label in labels], transpose


def _label_cents(
    labels: Sequence[Mapping[str, Any]], time_sec: float, note: Mapping[str, Any]
) -> float:
    for label in labels:
        if str(label.get("type")) != "intonation_error" or label.get("deviation_cents") is None:
            continue
        start = float(label.get("start_time", 0.0))
        end = max(float(label.get("end_time", start)), start)
        if not (start <= time_sec < end):
            continue
        label_pitch = label.get("midi_pitch")
        if label_pitch is not None and int(label_pitch) != int(note["pitch_midi_written"]):
            continue
        return float(label["deviation_cents"])
    return 0.0


def build_window_targets(
    rendered_notes: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    effective_audio_transpose: int,
    *,
    window_start_sec: float = 0.0,
) -> dict[str, np.ndarray]:
    """Rasterize exact events to Basic Pitch's sounding-pitch output axes.

    ``effective_audio_transpose`` is the corpus convention's audio-to-written
    shift, so it is subtracted from written note-map pitches here.
    """

    note_target = np.zeros((ANNOT_N_FRAMES, N_FREQ_BINS_NOTES), dtype=np.float32)
    onset_target = np.zeros_like(note_target)
    contour_target = np.zeros(
        (ANNOT_N_FRAMES, N_FREQ_BINS_CONTOURS), dtype=np.float32
    )
    frame_times = window_start_sec + np.arange(ANNOT_N_FRAMES) / ANNOTATIONS_FPS
    window_end = window_start_sec + ANNOT_N_FRAMES / ANNOTATIONS_FPS

    for item in rendered_notes:
        if "pitch_midi_written" not in item:
            raise ValueError("Every rendered note must include pitch_midi_written")
        written_pitch = int(item["pitch_midi_written"])
        sounding_pitch = written_pitch - int(effective_audio_transpose)
        if item.get("pitch_midi_sounding") is not None:
            mapped = int(item["pitch_midi_sounding"])
            if mapped != sounding_pitch:
                raise ValueError(
                    "rendered_notes pitch-space mismatch: "
                    f"written {written_pitch} - effective_audio_transpose "
                    f"{effective_audio_transpose} != sounding {mapped}"
                )
        start = float(item["start_sec"])
        end = max(float(item["end_sec"]), start + 0.001)
        if end <= window_start_sec or start >= window_end:
            continue

        note_bin = sounding_pitch - MIDI_OFFSET
        active = np.flatnonzero((frame_times >= start) & (frame_times < end))
        if 0 <= note_bin < N_FREQ_BINS_NOTES:
            note_target[active, note_bin] = 1.0
            onset_frame = int(round((start - window_start_sec) * ANNOTATIONS_FPS))
            if 0 <= onset_frame < ANNOT_N_FRAMES:
                onset_target[onset_frame, note_bin] = 1.0
        for frame in active:
            cents = _label_cents(labels, float(frame_times[frame]), item)
            contour_bin = int(
                round(
                    (sounding_pitch - MIDI_OFFSET + cents / 100.0)
                    * CONTOUR_BINS_PER_SEMITONE
                )
            )
            if 0 <= contour_bin < N_FREQ_BINS_CONTOURS:
                contour_target[frame, contour_bin] = 1.0
    return {"note": note_target, "onset": onset_target, "contour": contour_target}


def slice_audio_window(audio: np.ndarray, start_sample: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    window = np.zeros((AUDIO_N_SAMPLES, 1), dtype=np.float32)
    start = max(0, int(start_sample))
    take = audio[start : start + AUDIO_N_SAMPLES]
    window[: len(take), 0] = take
    return window


@lru_cache(maxsize=8)
def _load_mono_audio(path_text: str) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sample_rate = sf.read(path_text, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if int(sample_rate) != AUDIO_SAMPLE_RATE:
        divisor = int(np.gcd(int(sample_rate), AUDIO_SAMPLE_RATE))
        mono = resample_poly(
            mono, AUDIO_SAMPLE_RATE // divisor, int(sample_rate) // divisor
        ).astype(np.float32)
    return np.clip(mono, -1.0, 1.0)


def _audio_path(record: WindowRecord) -> Path:
    return Path(str(record.row.get("audio_path", record.sample_dir / "performance_audio.wav")))


def _window_starts(n_samples: int, hop_samples: int) -> list[int]:
    if hop_samples <= 0:
        raise ValueError("window hop must be positive")
    return list(range(0, max(1, int(n_samples)), hop_samples))


def _audio_length(path: Path) -> int:
    import soundfile as sf

    info = sf.info(str(path))
    return int(round(info.frames * AUDIO_SAMPLE_RATE / info.samplerate))


def build_window_records(
    rows: Sequence[dict[str, Any]],
    *,
    manifest_path: Path,
    manifest_roots: Mapping[str, Any],
    root_overrides: Mapping[str, Path],
    hop_samples: int,
    max_windows: int,
    seed: int,
) -> list[WindowRecord]:
    records: list[WindowRecord] = []
    for row in rows:
        sample_dir = resolve_sample_dir(
            row, manifest_path, manifest_roots, root_overrides
        )
        audio_path = Path(str(row.get("audio_path", sample_dir / "performance_audio.wav")))
        for start in _window_starts(_audio_length(audio_path), hop_samples):
            records.append(WindowRecord(sample_dir, row, start))
    if max_windows > 0 and len(records) > max_windows:
        records = sorted(
            records,
            key=lambda record: hashlib.sha256(
                f"{seed}:{_row_key(record.row)}:{record.start_sample}".encode("utf-8")
            ).digest(),
        )[:max_windows]
    return records


def example_for_record(record: WindowRecord) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    audio = slice_audio_window(
        _load_mono_audio(str(_audio_path(record))), record.start_sample
    )
    rendered, labels, transpose = load_target_documents(record.sample_dir, record.row)
    targets = build_window_targets(
        rendered,
        labels,
        transpose,
        window_start_sec=record.start_sample / AUDIO_SAMPLE_RATE,
    )
    return audio, targets


def make_dataset(
    tf: Any,
    records: Sequence[WindowRecord],
    *,
    batch_size: int,
    training: bool,
    seed: int,
) -> Any:
    if not records:
        raise ValueError("Cannot construct a dataset with no windows")

    def generate() -> Iterator[tuple[np.ndarray, dict[str, np.ndarray]]]:
        for record in records:
            yield example_for_record(record)

    signature = (
        tf.TensorSpec((AUDIO_N_SAMPLES, 1), tf.float32),
        {
            "note": tf.TensorSpec((ANNOT_N_FRAMES, N_FREQ_BINS_NOTES), tf.float32),
            "onset": tf.TensorSpec((ANNOT_N_FRAMES, N_FREQ_BINS_NOTES), tf.float32),
            "contour": tf.TensorSpec(
                (ANNOT_N_FRAMES, N_FREQ_BINS_CONTOURS), tf.float32
            ),
        },
    )
    dataset = tf.data.Dataset.from_generator(generate, output_signature=signature)
    if training:
        dataset = dataset.shuffle(len(records), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options).batch(batch_size).prefetch(1)


def _as_tensors(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        out: list[Any] = []
        for item in value:
            out.extend(_as_tensors(item))
        return out
    return [value]


def _source_layer(tensor: Any) -> Any | None:
    history = getattr(tensor, "_keras_history", None)
    if history is None:
        return None
    if hasattr(history, "layer"):
        return history.layer
    if hasattr(history, "operation"):
        return history.operation
    return history[0]


def _ancestors(tensor: Any) -> set[Any]:
    found: set[Any] = set()

    def visit(current: Any) -> None:
        layer = _source_layer(current)
        if layer is None or layer in found:
            return
        found.add(layer)
        try:
            inputs = layer.input
        except (AttributeError, RuntimeError):
            inputs = None
        for input_tensor in _as_tensors(inputs):
            visit(input_tensor)

    visit(tensor)
    return found


def _nearest_weighted_layers(tensor: Any) -> set[Any]:
    found: set[Any] = set()
    visited: set[Any] = set()

    def visit(current: Any) -> None:
        layer = _source_layer(current)
        if layer is None or layer in visited:
            return
        visited.add(layer)
        if getattr(layer, "weights", ()):
            found.add(layer)
            return
        try:
            inputs = layer.input
        except (AttributeError, RuntimeError):
            inputs = None
        for input_tensor in _as_tensors(inputs):
            visit(input_tensor)

    visit(tensor)
    return found


def _model_outputs(model: Any) -> dict[str, Any]:
    output = model.output
    if isinstance(output, dict):
        outputs = dict(output)
    else:
        tensors = _as_tensors(output)
        names = list(getattr(model, "output_names", ()))
        outputs = dict(zip(names, tensors))
    missing = {"note", "onset", "contour"} - set(outputs)
    if missing:
        raise ValueError(
            f"Model outputs must be note/onset/contour; missing {sorted(missing)}, "
            f"found {sorted(outputs)}"
        )
    return outputs


def discover_layer_groups(model: Any) -> dict[str, set[Any]]:
    """Discover output branches from graph ancestry instead of generated names."""

    outputs = _model_outputs(model)
    ancestors = {name: _ancestors(tensor) for name, tensor in outputs.items()}
    heads = set().union(
        *(_nearest_weighted_layers(outputs[name]) for name in ("note", "onset", "contour"))
    )
    contour_head = _nearest_weighted_layers(outputs["contour"])
    branches = heads | (
        (ancestors["note"] | ancestors["onset"]) - ancestors["contour"]
    )
    branches = {layer for layer in branches if getattr(layer, "weights", ())}

    harmonic_layers = {
        layer
        for layer in getattr(model, "layers", ())
        if "harmonic" in str(getattr(layer, "name", "")).lower()
        or "harmonic" in layer.__class__.__name__.lower()
    }
    frontend: set[Any] = set()
    for layer in harmonic_layers:
        frontend.add(layer)
        try:
            for tensor in _as_tensors(layer.input):
                frontend.update(_ancestors(tensor))
        except (AttributeError, RuntimeError):
            pass
    for layer in getattr(model, "layers", ()):
        identity = (
            str(getattr(layer, "name", "")) + " " + layer.__class__.__name__
        ).lower()
        if any(token in identity for token in ("cqt", "harmonic", "normalizedlog", "flattenaudio")):
            frontend.add(layer)

    weighted = {
        layer for layer in getattr(model, "layers", ()) if getattr(layer, "weights", ())
    }
    return {
        "heads": heads,
        "branches": branches,
        "trunk": weighted - frontend,
        "frontend": frontend,
        "contour_trunk": (
            {layer for layer in ancestors["contour"] if getattr(layer, "weights", ())}
            - contour_head
        ),
    }


def set_trainable_phase(model: Any, phase: str) -> dict[str, list[str]]:
    groups = discover_layer_groups(model)
    if phase not in {"heads", "branches", "trunk"}:
        raise ValueError(f"Unknown phase {phase!r}")
    selected = groups[phase]
    for layer in getattr(model, "layers", ()):
        layer.trainable = layer in selected
    return {
        key: sorted(str(getattr(layer, "name", layer.__class__.__name__)) for layer in value)
        for key, value in groups.items()
    }


def _import_tensorflow(device: str, seed: int) -> Any:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow could not be imported. Basic Pitch 0.4.0 requires a "
            "TensorFlow-compatible NumPy build (for TF 2.15, use NumPy < 2)."
        ) from exc

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    gpus = tf.config.list_physical_devices("GPU")
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        gpus = []
    elif device == "cuda" and not gpus:
        raise RuntimeError("--device cuda requested, but TensorFlow sees no CUDA GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(
        f"TensorFlow {tf.__version__}; device={'CUDA' if gpus else 'CPU'}; "
        f"visible_gpus={len(gpus)}",
        flush=True,
    )
    return tf


def official_saved_model_path() -> Path:
    try:
        version = importlib.metadata.version("basic-pitch")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("basic-pitch==0.4.0 is not installed") from exc
    if version != BASIC_PITCH_VERSION:
        raise RuntimeError(
            f"This path requires basic-pitch=={BASIC_PITCH_VERSION}, found {version}"
        )
    spec = importlib.util.find_spec("basic_pitch")
    if spec is None or spec.origin is None:
        raise RuntimeError("Could not locate the basic_pitch package")
    return Path(spec.origin).parent / "saved_models" / "icassp_2022" / "nmp"


def load_official_model(tf: Any, model_path: Path | str | None = None) -> Any:
    """Load the official ICASSP Keras SavedModel through the required API."""

    from basic_pitch import nn
    from basic_pitch.layers import nnaudio, signal

    path = Path(model_path) if model_path is not None else official_saved_model_path()
    if not (path / "saved_model.pb").is_file():
        raise FileNotFoundError(f"Basic Pitch TensorFlow SavedModel not found: {path}")
    custom_objects = {
        "CQT": nnaudio.CQT,
        "NormalizedLog": signal.NormalizedLog,
        "HarmonicStacking": nn.HarmonicStacking,
        "FlattenAudioCh": nn.FlattenAudioCh,
        "FlattenFreqCh": nn.FlattenFreqCh,
    }
    model = tf.keras.models.load_model(
        str(path), custom_objects=custom_objects, compile=False
    )
    _model_outputs(model)
    return model


def _save_saved_model(model: Any, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing SavedModel: {path}")
    try:
        model.save(str(path), save_format="tf", include_optimizer=False)
    except (TypeError, ValueError):
        if not hasattr(model, "export"):
            raise
        model.export(str(path))


def export_onnx_if_requested(
    tf: Any, model: Any, output_path: Path, requested: bool
) -> dict[str, Any]:
    if not requested:
        return {"status": "not_requested"}
    try:
        import tf2onnx
    except ImportError:
        message = "ONNX export skipped: tf2onnx is not installed"
        print(message, flush=True)
        return {"status": "skipped", "reason": "tf2onnx is not installed"}
    signature = (tf.TensorSpec((None, AUDIO_N_SAMPLES, 1), tf.float32, name="audio"),)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_keras(
        model, input_signature=signature, opset=17, output_path=str(output_path)
    )
    print(f"ONNX export: {output_path}", flush=True)
    return {"status": "exported", "path": str(output_path)}


def _compile(tf: Any, model: Any, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss={
            "note": tf.keras.losses.BinaryCrossentropy(),
            "onset": tf.keras.losses.BinaryCrossentropy(),
            "contour": tf.keras.losses.BinaryCrossentropy(),
        },
        loss_weights={"note": 1.0, "onset": 1.0, "contour": 0.5},
    )


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_frozen_manifest(
        args.manifest, procedural_only=args.procedural_only
    )
    document = manifest["document"]
    roots = document.get("roots", {}) if isinstance(document, dict) else {}
    overrides = _parse_root_overrides(args.root)
    train_rows = manifest["splits"]["train"]
    val_rows = manifest["splits"].get(
        "val", manifest["splits"].get("validation", [])
    )

    device = "cpu" if args.smoke and args.device == "auto" else args.device
    max_train = min(args.max_train_windows or 2, 2) if args.smoke else args.max_train_windows
    max_val = min(args.max_val_windows or 1, 1) if args.smoke else args.max_val_windows
    batch_size = 1 if args.smoke else args.batch_size
    phase_epochs = {
        "heads": 1 if args.smoke else args.head_epochs,
        "branches": 1 if args.smoke else args.branch_epochs,
        "trunk": 1 if args.smoke else args.trunk_epochs,
    }
    train_records = build_window_records(
        train_rows,
        manifest_path=Path(args.manifest),
        manifest_roots=roots,
        root_overrides=overrides,
        hop_samples=args.window_hop_samples,
        max_windows=max_train,
        seed=args.seed,
    )
    val_records = build_window_records(
        val_rows,
        manifest_path=Path(args.manifest),
        manifest_roots=roots,
        root_overrides=overrides,
        hop_samples=args.window_hop_samples,
        max_windows=max_val,
        seed=args.seed + 1,
    )

    tf = _import_tensorflow(device, args.seed)
    model = load_official_model(tf, args.model)
    train_ds = make_dataset(
        tf, train_records, batch_size=batch_size, training=True, seed=args.seed
    )
    val_ds = make_dataset(
        tf, val_records, batch_size=batch_size, training=False, seed=args.seed
    )
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    report: dict[str, Any] = {
        "basic_pitch_version": BASIC_PITCH_VERSION,
        "tensorflow_version": tf.__version__,
        "device": "cuda" if tf.config.list_physical_devices("GPU") and device != "cpu" else "cpu",
        "manifest": str(Path(args.manifest)),
        "procedural_only": bool(args.procedural_only),
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "train_windows": len(train_records),
        "val_windows": len(val_records),
        "phases": [],
    }
    learning_rates = {
        "heads": args.head_lr,
        "branches": args.branch_lr,
        "trunk": args.trunk_lr,
    }
    for phase in ("heads", "branches", "trunk"):
        epochs = int(phase_epochs[phase])
        if epochs <= 0:
            continue
        groups = set_trainable_phase(model, phase)
        _compile(tf, model, learning_rates[phase])
        checkpoint = checkpoint_dir / f"{phase}.best.keras"
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint), monitor="val_loss", save_best_only=True
            )
        ]
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks,
        )
        report["phases"].append(
            {
                "name": phase,
                "epochs": epochs,
                "learning_rate": learning_rates[phase],
                "checkpoint": str(checkpoint),
                "trainable_layers": groups[phase],
                "frozen_frontend": groups["frontend"],
                "history": {
                    key: [float(value) for value in values]
                    for key, values in history.history.items()
                },
            }
        )

    final_weights = checkpoint_dir / "final.weights.h5"
    model.save_weights(str(final_weights))
    saved_model = output_dir / "saved_model"
    _save_saved_model(model, saved_model)
    report["saved_model"] = str(saved_model)
    report["final_checkpoint"] = str(final_weights)
    report["onnx"] = export_onnx_if_requested(
        tf, model, output_dir / "basic_pitch_finetuned.onnx", args.export_onnx
    )
    history_path = output_dir / "history.json"
    report["history"] = str(history_path)
    history_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Required frozen JSON/JSONL split")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="ICASSP TensorFlow SavedModel override")
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override one named manifest root; repeat as needed",
    )
    parser.add_argument("--procedural-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=365)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--window-hop-samples", type=int, default=AUDIO_N_SAMPLES)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--branch-epochs", type=int, default=4)
    parser.add_argument("--trunk-epochs", type=int, default=2)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--branch-lr", type=float, default=1e-4)
    parser.add_argument("--trunk-lr", type=float, default=3e-5)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Deterministic CPU run: one epoch/phase, two train and one val windows",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export ONNX when tf2onnx is installed; otherwise report a skip",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_training(args)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
