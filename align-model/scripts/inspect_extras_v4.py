"""List wrong extra calls with their predicted and gold context (val features only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from alignmodel.joint.robust_dp_aligner_v2 import RobustDPCosts
from alignmodel.joint.robust_dp_aligner_v3 import AlignerV3Config, align_v3
from precision_harness_v4 import ALIGN, decode_rows, remap
from realistic92_aligner_common import load_clip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--dataset", default="dclike11")
    parser.add_argument("--root", type=Path, default=Path("E:/outputRaw_dclike_11"))
    parser.add_argument("--freeze", type=Path, default=ALIGN / "runs/dclike11-v1/ALIGNER_FREEZE.json")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--clips", type=int, default=6)
    parser.add_argument("--all", action="store_true", help="every wrong extra, not only plausible-looking ones")
    parser.add_argument("--group", help="clip-name prefix filter")
    args = parser.parse_args()
    candidate = json.loads((ALIGN / "runs/realistic92-stack-v2/CANDIDATE_STACK_V3.json").read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    report = json.loads(args.features.read_text(encoding="utf-8"))
    rows = [row for row in report["datasets"][args.dataset]["features"]
            if row["kind"] == "extra" and row["credit"] < 1 and (args.all or (
                row["ioi"] >= 0.12 and not row["same_pitch_neighbor"] and row["confidence"] >= 0.99))
            and (args.group is None or row["clip"].startswith(args.group))]
    print("plausible-looking wrong extras:", len(rows))
    clips = []
    for row in rows:
        if row["clip"] not in clips:
            clips.append(row["clip"])
    for name in clips[:args.clips]:
        clip = load_clip(args.root, name, Path(freeze["lineage_dir"]))
        cache = np.load(args.cache / f"{name}.npz")
        notes = decode_rows(cache["ctc"].astype(np.float32), 52, candidate["decoder"])
        alignment = align_v3(notes, clip.index.events, clip.score_path,
                             AlignerV3Config(costs=RobustDPCosts(**candidate["aligner_costs"]), artifact_ioi=0,
                                             timing_weight=0.2, copy_after_replay=True))
        events = remap(alignment.events, clip)
        gold = clip.rendered
        for extra in alignment.extras:
            event = events[extra.event_position]
            j = event.rendered_index
            if j >= 1_000_000:
                description = "unpaired: transcriber false note"
                gold_context = []
            else:
                target = gold[j]
                description = f"gold {target.relationship} span {target.score_span}"
                gold_context = [(g.pitch, g.relationship[:3]) for g in gold[max(0, j - 2):j + 3]]
            if j < 1_000_000 and gold[j].relationship == "extra" and gold[j].score_span is None:
                continue
            position = extra.event_position
            context = [(events[k].pitch, events[k].relationship[:3])
                       for k in range(max(0, position - 2), min(len(events), position + 3))]
            print(name, "extra", event.pitch, f"{event.start:.2f}s", "->", description, "| pred", context,
                  "| gold", gold_context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
