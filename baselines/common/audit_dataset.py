"""Check a portable converted dataset without opening every audio waveform."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import soundfile as sf
from prepare_dataset import output_paths


def audit(root):
    selected = root / ("split.audited.json" if (root / "split.audited.json").exists() else "split.json")
    split_bytes = selected.read_bytes()
    split = json.loads(split_bytes)
    manifest = json.loads((root / "manifest.json").read_text())
    ids = []
    for key, filename in split["midi_filename"].items():
        tid = Path(filename).name.replace(".midi", "")
        ids.append(tid)
        if split["split"][key] not in ("train", "validation", "test"):
            raise ValueError(f"Invalid split: {key}")
        rec = manifest["tracks"][tid]
        if rec.get("real_test") and split["split"][key] != "test":
            raise ValueError(f"Unlabelled real track in supervised split: {tid}")
        if rec.get("note_labels_check_ok") is False or rec.get("excluded"):
            raise ValueError(f"Failed or excluded labels in split: {tid}")
        paths = output_paths(root, tid)
        for path in paths.values():
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        audio = [sf.info(str(paths[name])) for name in ("mistake_wav", "score_wav")]
        if any(info.samplerate != 16000 or info.channels != 1 for info in audio):
            raise ValueError(f"Expected 16 kHz mono audio: {tid}")
        if audio[0].frames != audio[1].frames:
            raise ValueError(f"Audio lengths differ: {tid}")
    if not ids or len(ids) != len(set(ids)) or set(split["midi_filename"]) != set(split["split"]):
        raise ValueError("Empty, duplicate, or inconsistent split.json")
    return {"tracks": len(ids), "split_file": selected.name, "splits": dict(Counter(split["split"].values())),
            "split_sha256": hashlib.sha256(split_bytes).hexdigest(), "status": "OK"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.data.resolve()), indent=2))
