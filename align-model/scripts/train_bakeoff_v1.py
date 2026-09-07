"""Train Model B V1 control (current MelodyFirst + shared early stop)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = {
    (ROOT / "align-model" / "runs" / "melody-random12k-set").resolve(),
    (ROOT / "align-model" / "runs" / "melody-raw2k-set").resolve(),
}


def main() -> None:
    from alignmodel.melody_train import MelodyTrainConfig, train_melody

    out = (ROOT / "align-model" / "runs" / "melody-bakeoff" / "melody-b-v1-es").resolve()
    if out in FORBIDDEN:
        raise RuntimeError(f"refusing to overwrite {out}")
    data = ROOT / "synth-pipeline" / "output"
    last_err = None
    for batch in (2, 1):
        print(f"V1 train batch={batch} out={out}", flush=True)
        try:
            train_melody(
                MelodyTrainConfig(
                    data_root=data,
                    output_dir=out,
                    epochs=8,
                    batch_size=batch,
                    lr=2e-4,
                    device="cuda",
                    seed=365,
                    variant="v1",
                )
            )
            print(f"V1 done batch={batch}", flush=True)
            return
        except RuntimeError as exc:
            last_err = exc
            if "out of memory" not in str(exc).lower() or batch == 1:
                raise
            print("V1 OOM at batch 2; retrying batch=1", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise last_err  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
