"""Inference-only DataCreate decoding with the frozen epoch-30 retrained baselines.

Does not train, resume training, or select checkpoints from test results. Loads
the already-selected ``training/<model>/best.pt`` weights (or ``--ckpt``) with
the 2026-09-18 protocol: bf16 autocast, batch size 1, 1024-token budget,
prompted deterministic LadderSym, and this experiment's source snapshots.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PROTO = json.loads((RUN / "protocol.json").read_text(encoding="utf-8"))
REMOTE_CANDIDATES = (
    "/home/weixi/projects/ALIGN/experiments/baselines_retrain_20260918",
    "/root/ALIGN/experiments/baselines_retrain_20260918",
    "/srv/ALIGN/experiments/baselines_retrain_20260918",
)
SSH_OPTS = (
    "-4",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=20",
    "-o",
    "StrictHostKeyChecking=accept-new",
)
DUMMY_OPTIM = {
    "error_loss_weight": 8,
    "lr": 2e-5,
    "warmup_steps": 4000,
    "num_epochs": int(PROTO["epochs"]),
    "min_lr": 0.5,
    "num_steps_per_epoch": 1,
}


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ssh(host: str, command: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *SSH_OPTS, host, command],
        check=check,
        capture_output=True,
        text=True,
    )


def wait_for_host(host: str, attempts: int, delay: int) -> None:
    for attempt in range(1, attempts + 1):
        result = _ssh(host, "hostname && echo READY", check=False)
        if result.returncode == 0 and "READY" in (result.stdout or ""):
            print((result.stdout or "").strip(), flush=True)
            return
        detail = ((result.stderr or result.stdout) or "").strip().replace("\n", " ")
        print(f"restore attempt {attempt}/{attempts} failed: {detail[:240]}", flush=True)
        if attempt < attempts:
            time.sleep(delay)
    raise ConnectionError(f"{host} did not become reachable")


def restore_artifacts(host: str, flavor: str) -> Path:
    dest_dir = RUN / "training" / flavor
    dest_dir.mkdir(parents=True, exist_ok=True)
    remote_root = None
    for candidate in REMOTE_CANDIDATES:
        probe = _ssh(
            host,
            f'test -f "{candidate}/training/{flavor}/best.pt" && echo FOUND',
            check=False,
        )
        if probe.returncode == 0 and "FOUND" in (probe.stdout or ""):
            remote_root = candidate
            break
    if remote_root is None:
        raise FileNotFoundError(
            f"No {flavor} best.pt under {REMOTE_CANDIDATES} on {host}"
        )
    print(f"restoring {flavor} from {host}:{remote_root}", flush=True)
    for name in ("best.pt", "best.json", "config.json"):
        remote = f"{remote_root}/training/{flavor}/{name}"
        local = dest_dir / name
        subprocess.run(
            ["scp", *SSH_OPTS, f"{host}:{remote}", str(local)],
            check=True,
        )
    ckpt = dest_dir / "best.pt"
    if not ckpt.is_file() or ckpt.stat().st_size < 1024:
        raise FileNotFoundError(f"Restored checkpoint missing or empty: {ckpt}")
    return ckpt


def _load_model_config(flavor: str):
    config_json = RUN / "training" / flavor / "config.json"
    if config_json.is_file():
        payload = json.loads(config_json.read_text(encoding="utf-8"))
        model_cfg = OmegaConf.create(payload["model"])
        optim_cfg = OmegaConf.create(payload.get("optim") or DUMMY_OPTIM)
        return model_cfg, optim_cfg
    if flavor == "polytune":
        model_cfg = OmegaConf.load(RUN / "source" / "Polytune" / "config" / "model" / "polytune.yaml").config
        return model_cfg, OmegaConf.create(DUMMY_OPTIM)
    model_cfg = OmegaConf.load(
        RUN / "source" / "LadderSym" / "config" / "model" / "laddersym_MT3Net.yaml"
    ).config
    model_cfg.use_prompt = True
    return model_cfg, OmegaConf.create(DUMMY_OPTIM)


def build_model(flavor: str, ckpt: Path):
    sys.path[:0] = [
        str(RUN / "source" / ("Polytune" if flavor == "polytune" else "LadderSym")),
        str(ROOT / "baselines" / "common"),
    ]
    from align_runtime import load_model_weights

    model_cfg, optim_cfg = _load_model_config(flavor)
    if flavor == "polytune":
        from tasks.polytune_net import polytune

        module = polytune(model_cfg, optim_cfg)
    else:
        from tasks.laddersym_mt3_net import laddersym_MT3Net

        model_cfg.use_prompt = True
        module = laddersym_MT3Net(model_cfg, optim_cfg)
    load_model_weights(module.model, ckpt)
    return module.model


def infer(model, flavor: str, root: Path, ids: list[str], dest: Path) -> dict:
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    write(dest / "evaluated_ids.json", ids)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["tracks"]
    from inference_error import InferenceHandler

    handler = InferenceHandler(model=model, device=model.device, mel_norm=True)
    prior_mode = model.training
    model.eval()
    rows = []
    with (
        (dest / "inference.log").open("w", encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()),
    ):
        for index, sample_id in enumerate(ids):
            paths = manifest[sample_id]["outputs"]
            perf, sr = sf.read(paths["mistake_wav"], dtype="float32")
            ref, rr = sf.read(paths["score_wav"], dtype="float32")
            assert sr == rr == 16000
            kwargs = dict(
                mistake_audio=perf,
                score_audio=ref,
                audio_path=sample_id,
                outpath=str(dest / sample_id / "mix.mid"),
                batch_size=1,
                max_length=1024,
            )
            if flavor == "laddersym":
                kwargs["prompt_path"] = paths["score_mid"]
            handler.inference(**kwargs)
            midi = dest / sample_id / "mix.mid"
            if not midi.is_file():
                raise FileNotFoundError(f"No prediction written for {sample_id}")
            rows.append(
                {
                    "sample": sample_id,
                    "midi": str(midi),
                    "sha256": digest(midi),
                }
            )
            write(
                dest / "progress.json",
                {"completed": index + 1, "total": len(ids), "updated_at": time.time()},
            )
    model.train(prior_mode)
    return {"ids": ids, "rows": rows}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["polytune", "laddersym"], required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "baselines" / "data" / "datacreate_all_20260920",
    )
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--restore-host", default="i-2.gpushare.com")
    parser.add_argument("--restore-attempts", type=int, default=180)
    parser.add_argument("--restore-delay", type=int, default=20)
    parser.add_argument("--no-restore", action="store_true")
    args = parser.parse_args(argv)

    flavor = args.model
    os.environ["LADDERSYM_DETERMINISTIC_PROMPT"] = "1"
    os.environ["LADDERSYM_PROMPT_LENGTH"] = "1024"
    os.environ["LADDERSYM_USE_CACHE"] = "1"
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False

    ckpt = args.ckpt or (RUN / "training" / flavor / "best.pt")
    if not ckpt.is_file():
        if args.no_restore:
            raise FileNotFoundError(ckpt)
        wait_for_host(args.restore_host, args.restore_attempts, args.restore_delay)
        ckpt = restore_artifacts(args.restore_host, flavor)
    selected = {}
    best_json = ckpt.with_name("best.json")
    if best_json.is_file():
        selected = json.loads(best_json.read_text(encoding="utf-8"))
        if selected.get("sha256") and selected["sha256"] != digest(ckpt):
            raise ValueError(f"Checkpoint hash mismatch for {ckpt}")
        if selected.get("epoch") not in (None, int(PROTO["epochs"])):
            print(f"warning: selected epoch {selected.get('epoch')} vs protocol {PROTO['epochs']}", flush=True)

    data = args.data.resolve()
    ids = args.ids
    if ids is None:
        split = json.loads((data / "split.json").read_text(encoding="utf-8"))
        ids = [
            Path(path).name.replace(".midi", "")
            for key, path in split["midi_filename"].items()
            if split["split"].get(key) == "test"
        ]
        ids.sort()
    dest = (
        args.out.resolve()
        if args.out
        else ROOT
        / "experiments"
        / "baselines_datacreate_all_20260920"
        / "predictions"
        / flavor
    )
    if dest.exists():
        if (dest / "inference_manifest.json").is_file():
            raise FileExistsError(dest)
        shutil.rmtree(dest)

    sys.path[:0] = [
        str(RUN / "source" / ("Polytune" if flavor == "polytune" else "LadderSym")),
        str(ROOT / "baselines" / "common"),
    ]
    model = build_model(flavor, ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.device = device
    report = infer(model, flavor, data, ids, dest)
    write(
        dest / "inference_manifest.json",
        {
            "model": flavor,
            "checkpoint": str(ckpt.resolve()),
            "checkpoint_sha256": digest(ckpt),
            "selected": selected,
            "protocol_epochs": PROTO["epochs"],
            "precision": "bf16 autocast" if device.type == "cuda" else "float32-cpu",
            "batch_size": 1,
            "max_length": 1024,
            "prompted": flavor == "laddersym",
            "source": str(RUN / "source" / ("Polytune" if flavor == "polytune" else "LadderSym")),
            "source_hashes": str(RUN / "source_hashes.json"),
            "hydra_config": str(
                RUN
                / "source"
                / ("Polytune" if flavor == "polytune" else "LadderSym")
                / "config"
                / ("config_align.yaml" if flavor == "polytune" else "config_align_prompted.yaml")
            ),
            "hydra_config_sha256": digest(
                RUN
                / "source"
                / ("Polytune" if flavor == "polytune" else "LadderSym")
                / "config"
                / ("config_align.yaml" if flavor == "polytune" else "config_align_prompted.yaml")
            ),
            "data": str(data),
            "n_ids": len(ids),
            "rows": report["rows"],
        },
    )
    print("COMPLETE", flavor, len(ids), dest, flush=True)


if __name__ == "__main__":
    main()
