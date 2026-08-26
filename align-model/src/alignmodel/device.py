from __future__ import annotations

import torch


def resolve_device(name: str) -> torch.device:
    key = (name or "auto").strip().lower()
    if key == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(key)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{device.index or 0} ({torch.cuda.get_device_name(device)})"
    return str(device)
