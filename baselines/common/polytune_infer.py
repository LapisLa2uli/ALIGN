#!/usr/bin/env python
"""Run Polytune's InferenceHandler on one (mistake, score) wav pair.

Standalone driver (no Hydra chdir): builds the model from Polytune's config via
``hydra.compose``, loads a Lightning ``.ckpt`` or a plain ``.pt``/``.pth`` state dict,
picks a device (POLYTUNE_DEVICE / --device, else cuda > mps > cpu) and writes ONE
predicted MIDI whose tracks are named ``extra`` / ``missing`` / ``correct``
(error classes 1 / 2 / 3, see the ``inference_error.py`` patch).

    python polytune_infer.py --ckpt <last.pt|x.ckpt> --mistake perf.wav --score ref.wav --out pred.mid

Must be run with the Polytune venv (baselines/envs/polytune/bin/python); wrapped by
baselines/scripts/polytune_infer.sh.
"""
import argparse
import contextlib
import os
import sys
import time

DEFAULT_REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Polytune")
SAMPLE_RATE = 16000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Lightning .ckpt or plain state_dict .pt/.pth")
    p.add_argument("--mistake", required=True, help="performance ('mistake') wav")
    p.add_argument("--score", required=True, help="reference ('score') wav, same absolute timeline")
    p.add_argument("--out", required=True, help="output MIDI path")
    p.add_argument("--repo", default=DEFAULT_REPO, help="Polytune repo (added to sys.path)")
    p.add_argument("--config-name", default="config_align", help="Hydra config in <repo>/config")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--batch-size", type=int, default=1, help="2.048 s segments per generate() call")
    p.add_argument("--max-length", type=int, default=1024, help="max decoder tokens per segment")
    p.add_argument("--no-mel-norm", action="store_true", help="disable log-mel normalisation (MT3 weights only)")
    p.add_argument("--strict", action="store_true", default=True, help="strict checkpoint loading (always enabled)")
    p.add_argument("--verbose", action="store_true", help="keep InferenceHandler's stdout (default: <out>.log)")
    return p.parse_args()


def compose_cfg(repo, config_name):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(repo, "config"), version_base=None):
        return compose(config_name=config_name)


def build_model(cfg, ckpt, strict):
    import torch
    from tasks.polytune_net import polytune

    if ckpt.endswith(".ckpt"):
        module = polytune.load_from_checkpoint(
            ckpt, map_location="cpu", config=cfg.model.config, optim_cfg=cfg.optim
        )
        print(f"[polytune_infer] loaded Lightning checkpoint {ckpt}")
        return module.model

    module = polytune(config=cfg.model.config, optim_cfg=cfg.optim)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    # train_polytune.py already strips the Lightning "model." prefix; handle both forms.
    state = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in state.items()}
    missing, unexpected = module.model.load_state_dict(state, strict=strict)
    print(
        f"[polytune_infer] loaded state dict {ckpt} "
        f"({len(state)} tensors, {len(missing)} missing, {len(unexpected)} unexpected keys)"
    )
    if missing or unexpected:
        print("[polytune_infer]   missing:", missing[:10], "..." if len(missing) > 10 else "")
        print("[polytune_infer]   unexpected:", unexpected[:10], "..." if len(unexpected) > 10 else "")
    return module.model


def load_audio(path):
    import librosa

    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    print(f"[polytune_infer] {path}: {len(audio) / SAMPLE_RATE:.2f} s @ {SAMPLE_RATE} Hz")
    return audio


def summarise_midi(path):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(path)
    print(f"[polytune_infer] wrote {path}")
    for inst in pm.instruments:
        onsets = [n.start for n in inst.notes]
        span = f"{min(onsets):.2f}-{max(onsets):.2f} s" if onsets else "-"
        print(f"[polytune_infer]   track '{inst.name or '?'}': {len(inst.notes)} notes ({span})")
    if not pm.instruments:
        print("[polytune_infer]   (no notes predicted)")


def main():
    args = parse_args()
    repo = os.path.abspath(args.repo)
    if not os.path.isfile(os.path.join(repo, "inference_error.py")):
        sys.exit(f"[polytune_infer] Polytune repo not found at {repo}")
    for f in (args.ckpt, args.mistake, args.score):
        if not os.path.isfile(f):
            sys.exit(f"[polytune_infer] file not found: {f}")
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, repo)

    import torch  # noqa: F401  (import after sys.path so repo modules resolve)
    from inference_error import InferenceHandler, resolve_device

    device = resolve_device(None if args.device == "auto" else args.device)
    print(f"[polytune_infer] device: {device}")

    cfg = compose_cfg(repo, args.config_name)
    model = build_model(cfg, os.path.abspath(args.ckpt), strict=args.strict)
    model.eval()

    mistake_audio = load_audio(args.mistake)
    score_audio = load_audio(args.score)

    handler = InferenceHandler(model=model, device=device, mel_norm=not args.no_mel_norm)
    if os.path.exists(out):
        os.remove(out)
    t0 = time.time()
    log_path = out + ".log"
    if args.verbose:
        ctx = contextlib.nullcontext()
    else:
        print(f"[polytune_infer] InferenceHandler stdout -> {log_path}")
        ctx = contextlib.redirect_stdout(open(log_path, "w"))
    with ctx:
        handler.inference(
            mistake_audio=mistake_audio,
            score_audio=score_audio,
            audio_path=args.mistake,
            outpath=out,
            batch_size=args.batch_size,
            max_length=args.max_length,
            verbose=True,
        )
    print(f"[polytune_infer] inference took {time.time() - t0:.1f} s")
    if not os.path.isfile(out):
        # InferenceHandler.inference swallows exceptions (traceback on stderr) -> fail loudly here.
        sys.exit(f"[polytune_infer] no output written to {out}; see traceback above" + ("" if args.verbose else f" and {log_path}"))
    summarise_midi(out)


if __name__ == "__main__":
    main()
