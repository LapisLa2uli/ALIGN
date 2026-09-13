"""Check imports and execute a real device forward/backward before a long run."""
import argparse
import importlib
import json
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cu121", "cu118", "cuda", "cpu", "mac", "mps"], default="cuda")
    args = parser.parse_args()
    versions = {}
    for module in ("torch", "torchaudio", "torchvision", "transformers", "pytorch_lightning",
                   "hydra", "note_seq", "pretty_midi", "librosa", "numpy", "scipy", "soundfile"):
        obj = importlib.import_module(module)
        versions[module] = getattr(obj, "__version__", "import OK")
    import torch
    device = "cuda" if args.device.startswith("cu") else args.device
    if device == "mac":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable: check NVIDIA driver and the selected PyTorch wheel")
    x = torch.randn(64, 64, device=device, requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
        raise RuntimeError("Device forward/backward produced nonfinite values")
    report = {"python": sys.version, "platform": platform.platform(), "versions": versions,
              "device": device, "forward_backward": "OK"}
    if device == "cuda":
        report.update(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                      cuda=torch.version.cuda, bf16=torch.cuda.is_bf16_supported(),
                      vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
