# Polytune environment — deviations from `requirements.txt`

- Venv: `/Users/tong172/Desktop/projects/ALIGN/baselines/envs/polytune`
- Repo: `/Users/tong172/Desktop/projects/ALIGN/baselines/Polytune` (commit `d2055bb`)
- Interpreter: CPython 3.11.4 (`/usr/local/bin/python3.11`), macOS arm64 (Apple Silicon), no CUDA, no conda
- Tool: `uv 0.12.5`; created 2026-09-08
- Lock file (exact `uv pip freeze`): `/Users/tong172/Desktop/projects/ALIGN/baselines/envs/requirements.polytune.lock.txt` (114 packages)

## Explicit pins

All 22 packages pinned in `requirements.txt` installed at **exactly** the pinned version
(absl-py 2.1.0, einops 0.7.0, hydra-core 1.3.2, immutabledict 4.2.0, librosa 0.10.1, matplotlib 3.8.4,
mir-eval 0.7, note-seq 0.0.5, numpy 1.26.4, omegaconf 2.3.0, pandas 2.2.2, pretty-midi 0.2.10,
pytorch-lightning 2.2.3, scipy 1.13.0, timm 0.9.16, torch 2.3.0, torchaudio 2.3.0, torchmetrics 1.3.2,
transformers 4.40.1, tqdm 4.66.2, wandb 0.16.6, yacs 0.1.8). No pin had to be relaxed; every wheel was
available for macOS arm64 / cp311. No CUDA-only package was installed (torch/torchaudio/torchvision are the
CPU+MPS macOS wheels).

## Deviations (transitive dependencies only)

| package    | pinned in requirements.txt | resolver picked | installed | reason |
|------------|----------------------------|-----------------|-----------|--------|
| setuptools | not pinned (transitive, required only by `wandb==0.16.6`, unconstrained) | 84.0.0 | **80.10.2** (constraint `setuptools<81`) | `lightning_fabric/__init__.py:41` (dependency of `pytorch-lightning==2.2.3`) executes `__import__("pkg_resources").declare_namespace(__name__)` at import time. `pkg_resources` is no longer shipped by setuptools 84.0.0 (the `site-packages/pkg_resources` directory is absent), so `import pytorch_lightning` failed with `ModuleNotFoundError: No module named 'pkg_resources'`. Downgrading setuptools is the smallest fix; it touches no explicitly pinned package. setuptools 80.x still ships `pkg_resources` (with a DeprecationWarning). |

Command used for the fix:

    uv pip install --python envs/polytune/bin/python "setuptools<81"

Nothing else was added, removed, or changed after `uv pip install -r requirements.txt`.

## Notable transitive versions (chosen by the resolver, not pinned upstream)

- torchvision 0.18.0 (pulled in by timm 0.9.16; matches torch 2.3.0)
- numba 0.67.0 / llvmlite 0.49.0 (librosa, note-seq) — imports fine with numpy 1.26.4
- protobuf 4.25.9 (note-seq, wandb)
- huggingface-hub 0.36.2, tokenizers 0.19.1, safetensors 0.8.0 (transformers 4.40.1)
- bokeh 3.9.2, ipython 9.17.1 (note-seq's plotting deps; unused by Polytune code paths)

## Verification results (run from inside the repo with the venv python)

1. `python -c "import torch, pytorch_lightning, transformers, note_seq, hydra, librosa, mir_eval, pretty_midi, timm, einops, wandb, omegaconf; print(torch.__version__, transformers.__version__, torch.backends.mps.is_available())"`
   -> `2.3.0 4.40.1 True`  (MPS backend available)
2. `python -c "import tasks.polytune_net, dataset.dataset_2_random, contrib.vocabularies, models.polytune"`
   -> OK (no `__init__.py` in `tasks/`, `models/`, `dataset/`; they work as namespace packages, including the
   relative imports `from .ast import ASTEncoder` / `from .pos_embed import ...` inside `models/polytune.py`).
3. Model construction on CPU (config/model/polytune.yaml -> `config` node -> `T5Config.from_dict` ->
   `models.polytune.T5ForConditionalGeneration`):
   - total parameters 191,757,056 (191.8M), all trainable
     - encoder `ASTEncoder`: 164,743,424 (11+11 modality-specific ViT-B blocks + 1 shared block, embed_dim 768)
     - decoder `T5Stack`: 25,965,056 (8 layers, d_model 512, d_ff 1024, 6 heads)
     - decoder_embed_tokens 786,432; lm_head 786,432; proj 262,144
   - construction time ~0.9 s on CPU
   - **No pretrained weights are downloaded at construction time.** There is no `timm.create_model(...)`
     call anywhere; `models/ast.py` only imports timm building blocks (`Attention`, `Mlp`, `LayerScale`,
     `DropPath`, `to_2tuple`) and builds the encoder from scratch with sin-cos positional init. A socket
     guard that raised on any outbound `connect()` recorded zero attempts; `HF_HUB_OFFLINE=1` was also set.
   - Extra (not required): a dummy CPU forward pass with `mistake_inputs`/`score_inputs` of shape
     (1, 256, 512) and 16 label tokens returned logits of shape (1, 16, 1536).

Warnings emitted (harmless):

- `lightning_fabric/__init__.py:41: pkg_resources is deprecated as an API ...` — consequence of the setuptools<81 fix.
- `pydub/utils.py:170: RuntimeWarning: Couldn't find ffmpeg or avconv` — `note_seq` imports `pydub`; ffmpeg is
  not on PATH. Only affects mp3 decoding via pydub; wav loading goes through librosa/soundfile and is unaffected.

## Side effects / gotchas to know about

- `import models.ast` sets `os.environ["TORCH_HOME"] = "./pretrained_models"` (relative to the CWD) at import
  time. Nothing is downloaded, and no directory is created, but any later `torch.hub` use would cache there.
- `models/ast.py` shadows the stdlib `ast` module if `models/` itself is ever put on `sys.path` (it is not when
  running from the repo root).
- `config/config_maestro.yaml` is written for a CUDA box: `trainer.accelerator: gpu`, `precision: bf16-mixed`,
  `strategy: ddp_find_unused_parameters_false`, `dataloader.*.num_workers: 32`. These need Hydra overrides on
  macOS (e.g. `trainer.accelerator=cpu` or `mps`, `trainer.precision=32`, `trainer.strategy=auto`,
  `dataloader.train.num_workers=0`). Not a package deviation — noted for whoever runs inference/training.
- `torch.backends.mps.is_available()` is True, but MPS support in torch 2.3.0 has gaps (e.g. some ops fall back
  or are unsupported); set `PYTORCH_ENABLE_MPS_FALLBACK=1` if trying MPS.
