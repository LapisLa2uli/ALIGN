"""Write stored v3 eval predictions into DataCreate labels_agent.json.

Archives the previous agent documents, then replaces labels_agent.json on the
31 clips from the 001-040 exclusion eval. Human labels.json is not touched.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ALIGN = ROOT / "align-model"
sys.path[:0] = [str(ALIGN / "scripts"), str(ALIGN / "src"), str(ROOT / "DataCreate" / "src")]

import eval_datacreate_current as base
from datacreate.models import LabelsDocument
from label_datacreate_agent import agent_label_from_prediction

OUT = Path(__file__).resolve().parent
SAMPLES = ROOT / "DataCreate" / "samples"
PRED_ROOT = (
    ALIGN
    / "runs"
    / "eval-datacreate-all94-agent-note-wise-20260915"
    / "predictions"
    / "v3"
)
KEEP = [
    f"{index:03d}"
    for index in range(1, 41)
    if f"{index:03d}"
    not in {"005", "007", "010", "012", "020", "026", "030", "034", "036"}
]
SCORE_PART_KEYS = (
    "start_note_index",
    "end_note_index",
    "pad_notes",
    "start_measure",
    "end_measure",
    "core_start_note_index",
    "core_end_note_index",
)
COMMENT = (
    "ALIGN error-heads v3 prediction from the DataCreate 001-040 exclusion eval."
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _score_part(raw: dict | None) -> dict | None:
    if not isinstance(raw, dict):
        return None
    ordered = {key: raw[key] for key in SCORE_PART_KEYS if key in raw}
    for key, value in raw.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def agent_document(sample: str, prediction: dict) -> dict:
    labels = []
    for index, raw in enumerate(prediction.get("labels") or []):
        label = agent_label_from_prediction(raw, index)
        label["comment"] = COMMENT
        label["score_part"] = _score_part(label.get("score_part"))
        labels.append(label)
    return {
        "schema_version": "1.2",
        "audio_reference": "performance_audio.wav",
        "annotator_id": "cursor_agent_error_heads_v3",
        "self_reported": [],
        "labels": labels,
        "agent_labeling": {
            "method": "align_error_heads_v3",
            "transcriber": "basic-pitch-frozen",
            "uses_project_alignment_or_error_models": True,
            "replaced_previous_agent_labels": True,
            "source_prediction": str(PRED_ROOT / f"{sample}.json"),
            "source_eval": str(OUT / "report.json"),
            "kept_counts_by_type": dict(Counter(label["type"] for label in labels)),
            "pipeline": dict(prediction.get("pipeline") or {}),
            "written_utc": _utc(),
        },
    }


def main() -> None:
    archive = OUT / "archived_labels_agent_v5"
    written_dir = OUT / "written_labels_agent_v3"
    archive.mkdir(parents=True, exist_ok=True)
    written_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in KEEP:
        sample_dir = SAMPLES / sample
        dest = sample_dir / "labels_agent.json"
        pred_path = PRED_ROOT / f"{sample}.json"
        if not pred_path.is_file():
            raise FileNotFoundError(pred_path)
        previous = None
        if dest.is_file():
            archived = archive / f"{sample}.json"
            shutil.copy2(dest, archived)
            old = json.loads(dest.read_text(encoding="utf-8"))
            previous = {
                "archived": str(archived),
                "annotator_id": old.get("annotator_id"),
                "label_count": len(old.get("labels") or []),
            }
        prediction = json.loads(pred_path.read_text(encoding="utf-8"))
        document = agent_document(sample, prediction)
        LabelsDocument.model_validate(
            {
                key: document[key]
                for key in (
                    "schema_version",
                    "audio_reference",
                    "annotator_id",
                    "self_reported",
                    "labels",
                )
            }
        )
        base._atomic_json(dest, document)
        base._atomic_json(written_dir / f"{sample}.json", document)
        rows.append(
            {
                "sample": sample,
                "path": str(dest),
                "label_count": len(document["labels"]),
                "counts_by_type": document["agent_labeling"]["kept_counts_by_type"],
                "previous": previous,
                "human_labels_json_touched": False,
            }
        )
    manifest = {
        "written_utc": _utc(),
        "model": "error-heads-v3",
        "annotator_id": "cursor_agent_error_heads_v3",
        "samples": KEEP,
        "human_labels_unchanged": True,
        "excluded_untouched": ["005", "007", "010", "012", "020", "026", "030", "034", "036"],
        "rows": rows,
        "total_labels": sum(row["label_count"] for row in rows),
    }
    base._atomic_json(OUT / "labels_agent_write_manifest.json", manifest)
    print(json.dumps(
        {
            "clips": len(rows),
            "total_labels": manifest["total_labels"],
            "manifest": str(OUT / "labels_agent_write_manifest.json"),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
