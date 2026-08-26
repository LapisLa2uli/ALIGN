from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from alignmodel.pipeline import run_pipeline, write_prediction


def find_repetition_sample(root: Path) -> Path:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    for sample in sorted(p for p in root.iterdir() if p.is_dir()):
        meta_path = sample / "metadata.json"
        labels_path = sample / "labels.json"
        wav = sample / "performance_audio.wav"
        score = sample / "verified_score.musicxml"
        if not (wav.exists() and score.exists() and labels_path.exists()):
            continue
        repeated = False
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            repeated = bool(meta.get("repeated"))
        if not repeated:
            labels = json.loads(labels_path.read_text(encoding="utf-8")).get("labels", [])
            repeated = any(lab.get("type") == "repetition" for lab in labels)
        if repeated:
            return sample
    raise FileNotFoundError(f"No repetition-labeled bundle under {root}")


def gold_spans(sample_dir: Path, kind: str) -> list[tuple[float, float]]:
    labels = json.loads((sample_dir / "labels.json").read_text(encoding="utf-8")).get(
        "labels", []
    )
    return [
        (float(lab["start_time"]), float(lab["end_time"]))
        for lab in labels
        if lab.get("type") == kind
    ]


def span_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def best_mean_iou(
    pred: list[tuple[float, float]], gold: list[tuple[float, float]]
) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    scores = []
    for g in gold:
        if not pred:
            scores.append(0.0)
            continue
        scores.append(max(span_iou(g, p) for p in pred))
    return float(sum(scores) / len(scores))


def smoke_pipeline(
    data_root: Path,
    out_path: Path | None = None,
    timbre: bool = False,
    device: str = "cuda",
) -> dict:
    sample = find_repetition_sample(data_root)
    state = run_pipeline(sample, timbre=timbre, device=device)
    pred_path = out_path or Path("align-model/runs/pipeline-smoke/pipeline_pred.json")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    write_prediction(state, pred_path)
    pred_rep = [
        (lab.start_time, lab.end_time) for lab in state.labels if lab.type == "repetition"
    ]
    gold_rep = gold_spans(sample, "repetition")
    counts = Counter(lab.type for lab in state.labels)
    report = {
        "sample": str(sample),
        "pred_path": str(pred_path),
        "device": state.device,
        "stages_run": state.stages_run,
        "n_boundaries": len(state.boundaries),
        "n_segments": len(state.segments),
        "n_pairs": len(state.pairs),
        "pred_counts": dict(counts),
        "gold_repetition_spans": gold_rep,
        "pred_repetition_spans": [(round(a, 4), round(b, 4)) for a, b in pred_rep],
        "repetition_iou": round(best_mean_iou(pred_rep, gold_rep), 3),
        "n_labels": len(state.labels),
    }
    return report
