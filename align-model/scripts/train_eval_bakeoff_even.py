"""Wait for shared ES, then train/eval Model B even bakeoff versions V2 → V4 → V6."""

from __future__ import annotations

import json
import time
from pathlib import Path

from alignmodel.bakeoff.runner import run_one

ROOT = Path(__file__).resolve().parents[2]
ES_READY = ROOT / "align-model" / "runs" / "melody-bakeoff" / "es_ready"
RUNS = ROOT / "align-model" / "runs"
DATA = ROOT / "synth-pipeline" / "output"
SUMMARY = RUNS / "melody-bakeoff" / "even_summary.json"


def wait_es_ready(path: Path, timeout_sec: int = 1800, interval_sec: int = 30) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if path.is_file():
            print(f"es_ready present: {path}", flush=True)
            return True
        print(f"waiting for {path} sleep={interval_sec}s", flush=True)
        time.sleep(interval_sec)
    return path.is_file()


def main() -> None:
    if not wait_es_ready(ES_READY):
        from alignmodel.melody_train import MelodyTrainConfig, step_ema_is_plateau, step_ema_update

        cfg = MelodyTrainConfig()
        has_es = (
            hasattr(cfg, "es_min_steps")
            and hasattr(cfg, "es_plateau_steps")
            and callable(step_ema_update)
            and callable(step_ema_is_plateau)
        )
        if not has_es:
            raise RuntimeError("es_ready missing and shared ES is not in melody_train.py")
        print("es_ready missing after timeout; shared ES already in melody_train.py — proceeding", flush=True)

    results = []
    for variant in ("v2", "v4", "v6"):
        metrics = run_one(
            variant,
            data_root=DATA,
            runs_root=RUNS,
            epochs=8,
            batch_size=2,
            device="cuda",
            seed=365,
            n_eval=100,
        )
        results.append(metrics)
        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(json.dumps({"results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
