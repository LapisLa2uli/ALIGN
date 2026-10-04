"""Run the v3 transcriber + aligner v3 (+ gate) on DataCreate takes into a run directory.

Transcriber outputs are cached per take (CTC + frame heads + level envelope),
so aligner/gate variants rerun without the GPU. Alignments are written in the
stack schema used by eval_datacreate_vs_human.py. Sample folders are not
modified.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from alignmodel.joint.index import ScoreEventIndex
from alignmodel.joint.robust_dp_aligner_v2 import RobustDPCosts
from alignmodel.joint.robust_dp_aligner_v3 import AlignerV3Config, GateConfig, align_v3, gate_alignment
from precision_harness_v4 import HOP, decode_rows, rms_db

ALIGN = Path(__file__).resolve().parents[1]
CANDIDATE = ALIGN / "runs/realistic92-stack-v2/CANDIDATE_STACK_V3.json"
_STATE: dict[str, Any] = {}


def cache_outputs(samples: list[Path], cache: Path, checkpoint: Path | None = None) -> None:
    import torch
    from alignmodel.transcription.mel_ctc_v3 import extract_dual_mel, infer_dual_outputs, load_dual_checkpoint
    from alignmodel.transcription.mel_v1 import load_audio_mono

    todo = [sample for sample in samples if not (cache / f"{sample.name}.npz").is_file()]
    if not todo:
        return
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _payload = load_dual_checkpoint(checkpoint or Path(candidate["checkpoint"]), device)
    model.eval()
    cache.mkdir(parents=True, exist_ok=True)
    for position, sample in enumerate(todo, 1):
        audio = load_audio_mono(sample / "performance_audio.wav", 22050)
        with torch.inference_mode():
            mel, _ = extract_dual_mel(audio, device)
            outputs = infer_dual_outputs(model, np.asarray(mel, np.float32), device)
        frames = len(outputs["ctc"])
        np.savez_compressed(
            cache / f"{sample.name}.npz",
            ctc=outputs["ctc"].astype(np.float16), voiced=outputs["voiced"].astype(np.float16),
            onset=outputs["onset"].astype(np.float16), rms_db=rms_db(audio, frames),
        )
        if position % 25 == 0:
            print(f"cached {position}/{len(todo)}", flush=True)


def _init(payload: dict[str, Any]) -> None:
    _STATE.update(payload)
    if payload.get("verifier"):
        import torch

        from alignmodel.joint.presence_verifier_v1 import load_verifier

        torch.set_num_threads(1)
        _STATE["verifier_model"] = load_verifier(Path(payload["verifier"]), "cpu")


def _run(sample_dir: str) -> tuple[str, dict[str, Any] | None, str | None]:
    sample = Path(sample_dir)
    try:
        cache = np.load(Path(_STATE["cache"]) / f"{sample.name}.npz")
        ctc = cache["ctc"].astype(np.float32)
        evidence = {"ctc": ctc, "voiced": cache["voiced"].astype(np.float32),
                    "onset": cache["onset"].astype(np.float32), "rms_db": cache["rms_db"],
                    "hop": HOP, "midi_min": 52}
        if _STATE.get("verifier_model") is not None:
            from alignmodel.joint.presence_verifier_v1 import score_presence
            from alignmodel.transcription.mel_ctc_v3 import extract_dual_mel
            from alignmodel.transcription.mel_v1 import load_audio_mono

            mel, _ = extract_dual_mel(load_audio_mono(sample / "performance_audio.wav", 22050), "cpu")
            mel = np.asarray(mel, np.float32)
            model = _STATE["verifier_model"]
            evidence["presence"] = lambda queries: score_presence(model, mel, queries, "cpu")
        rows = decode_rows(ctc, 52, _STATE["decoder"])
        score_path = sample / "verified_score.musicxml"
        index = ScoreEventIndex.from_musicxml(score_path)
        aligner = AlignerV3Config(costs=RobustDPCosts(**_STATE["costs"]), **_STATE["aligner"])
        alignment = align_v3(rows, index.events, score_path, aligner)
        events, deletions, info = gate_alignment(alignment, index.events, evidence, GateConfig(**_STATE["gate"]))
        return sample.name, {
            "schema_version": "align-datacreate-stack-v4-alignment",
            "sample": sample.name,
            "aligner": "robust_dp_aligner_v3",
            "aligner_config": _STATE["aligner"],
            "gate_info": info,
            "score_event_count": len(index.events),
            "match_fraction": alignment.match_fraction,
            "repeat_hypothesis": {"source_span": list(alignment.source_span) if alignment.source_span else None,
                                  "extra_copies": alignment.copies},
            "missed_score_event_indices": sorted(deletions),
            "events": [
                {"note_index": event.rendered_index, "pitch": event.pitch, "start": event.start, "end": event.end,
                 "relationship": event.relationship,
                 "score_span": list(event.score_span) if event.score_span else None,
                 "copy_pass": event.copy_pass, "confidence": event.confidence}
                for event in events
            ],
        }, None
    except Exception as error:  # noqa: BLE001
        return sample.name, None, f"{type(error).__name__}: {error}"


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=root / "DataCreate" / "samples")
    parser.add_argument("--cache", type=Path, default=ALIGN / "runs/precision-v4/dc-cache")
    parser.add_argument("--aligner", default="{}")
    parser.add_argument("--gate", default='{"enabled": false}')
    parser.add_argument("--gate-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--checkpoint", type=Path, help="transcriber checkpoint (default: stack v3 candidate)")
    parser.add_argument("--verifier", type=Path, help="note-presence verifier checkpoint")
    args = parser.parse_args()
    samples = sorted(path for path in args.samples.iterdir() if path.is_dir()
                     and (path / "performance_audio.wav").is_file() and (path / "verified_score.musicxml").is_file())
    cache_outputs(samples, args.cache, args.checkpoint)
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    gate = json.loads(args.gate_file.read_text(encoding="utf-8")) if args.gate_file else json.loads(args.gate)
    payload = {"cache": str(args.cache), "decoder": candidate["decoder"], "costs": candidate["aligner_costs"],
               "aligner": json.loads(args.aligner), "gate": gate,
               "verifier": str(args.verifier) if args.verifier else None}
    out = args.output / "alignments"
    out.mkdir(parents=True, exist_ok=True)
    errors = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(payload,)) as pool:
        for name, alignment, error in pool.map(_run, [str(sample) for sample in samples]):
            if error:
                errors[name] = error
                continue
            (out / f"{name}.json").write_text(json.dumps(alignment) + "\n", encoding="utf-8")
    (args.output / "run_config.json").write_text(json.dumps({**payload, "errors": errors}, indent=1) + "\n",
                                                 encoding="utf-8")
    print(f"wrote {len(samples) - len(errors)} alignments, errors: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
