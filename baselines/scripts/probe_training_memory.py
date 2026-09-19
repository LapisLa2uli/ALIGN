"""Measure CUDA memory for two real full-budget training updates (no checkpoint)."""
import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--flavor", choices=("polytune", "laddersym"), required=True)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
baseline = Path(__file__).resolve().parents[1]
repo = baseline / ("Polytune" if args.flavor == "polytune" else "LadderSym")
sys.path.insert(0, str(baseline / "common"))
sys.path.insert(0, str(repo))
os.environ.setdefault("MPLBACKEND", "Agg")

import torch
from omegaconf import OmegaConf
from dataset.dataset_2_random import Dataset, collate_fn

torch.manual_seed(365)
cfg_path = repo / "config/model" / ("polytune.yaml" if args.flavor == "polytune" else "laddersym_MT3Net.yaml")
cfg = OmegaConf.load(cfg_path)
optim = OmegaConf.create(dict(lr=2e-5, error_loss_weight=8))
kwargs = dict(root_dir=str(args.data.resolve()), split_json_path=str(args.data.resolve()/"split.json"),
              split="train", mel_length=256, event_length=1024, num_rows_per_batch=1,
              split_frame_length=2000, is_deterministic=True, is_randomize_tokens=False,
              is_random_alignment_shift_augmentation=False, shuffle=False)
if args.flavor == "polytune":
    from tasks.polytune_net import polytune
    model = polytune(cfg.config, optim)
else:
    from tasks.laddersym_mt3_net import laddersym_MT3Net
    cfg.config.use_prompt = True
    model = laddersym_MT3Net(cfg.config, optim)
    kwargs.update(use_prompt=True, prompt_length=1024, audio_filename="mix.wav")
dataset = Dataset(**kwargs)
torch.cuda.reset_peak_memory_stats()
model = model.cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
losses = []
for i in range(2):
    batch = [value.cuda() for value in collate_fn([dataset[i]])]
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model.training_step(batch, i)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
torch.cuda.synchronize()
result = dict(flavor=args.flavor, batch_size=1, event_length=1024,
              prompt_length=1024 if args.flavor=="laddersym" else None,
              max_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
              max_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2,
              free_MiB=torch.cuda.mem_get_info()[0]/1024**2, losses=losses)
args.out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result), flush=True)
