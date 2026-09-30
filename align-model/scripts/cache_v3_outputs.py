"""Cache v3 (dual-mel) CTC softmax and frame heads for a list of clips (inference only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from alignmodel.joint.packed_data import sha256_file
from alignmodel.transcription.mel_ctc_v3 import extract_dual_mel, infer_dual_outputs, load_dual_checkpoint
from alignmodel.transcription.mel_v1 import load_audio_mono


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    names = json.loads(args.names.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _payload = load_dual_checkpoint(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cache_info.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "midi_min": model.config.midi_min,
        "hop_sec": 256 / 22050,
        "clips": len(names),
    }, indent=2), encoding="utf-8")
    for position, name in enumerate(names, 1):
        path = args.output_dir / f"{name}.npz"
        if not path.exists():
            audio = load_audio_mono(args.root / name / "performance_audio.wav", 22050)
            mel, _ = extract_dual_mel(audio, device)
            outputs = infer_dual_outputs(model, np.asarray(mel, np.float32), device)
            np.savez_compressed(path, **{key: value.astype(np.float16) for key, value in outputs.items()})
        if position % 200 == 0 or position == len(names):
            print(f"cached={position}/{len(names)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
