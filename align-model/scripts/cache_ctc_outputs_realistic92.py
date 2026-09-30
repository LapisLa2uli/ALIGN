"""Cache CTC softmax and frame heads for a 9.2 split (inference only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from alignmodel.joint.packed_data import sha256_file
from alignmodel.transcription.mel_ctc_v1 import infer_ctc_outputs, load_ctc_checkpoint
from alignmodel.transcription.mel_v1 import extract_log_mel, load_audio_mono


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True, help="JSON list of clip names")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-name", default="performance_audio.wav")
    args = parser.parse_args()

    names = json.loads(args.names.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, frontend, _payload = load_ctc_checkpoint(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cache_info.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "midi_min": model.config.midi_min,
        "hop_sec": frontend.hop_sec,
        "clips": len(names),
    }, indent=2), encoding="utf-8")
    for position, name in enumerate(names, 1):
        path = args.output_dir / f"{name}.npz"
        if path.exists():
            continue
        audio = load_audio_mono(args.root / name / args.audio_name, frontend.sample_rate)
        mel, _ = extract_log_mel(audio, frontend, device=device)
        outputs = infer_ctc_outputs(model, np.asarray(mel, np.float32), device)
        np.savez_compressed(path, **{key: value.astype(np.float16) for key, value in outputs.items()})
        if position % 200 == 0 or position == len(names):
            print(f"cached={position}/{len(names)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
