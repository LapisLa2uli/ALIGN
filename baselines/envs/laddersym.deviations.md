# LadderSym environment: deviations from `LadderSym/requirements.txt`

- Venv: `/Users/tong172/Desktop/projects/ALIGN/baselines/envs/laddersym` (created with `uv venv --python /usr/local/bin/python3.11`, CPython 3.11.4)
- Machine: macOS arm64 (Apple Silicon), no CUDA, no conda. `torch.backends.mps.is_available()` -> `True`.
- Lock file (`uv pip freeze`): `/Users/tong172/Desktop/projects/ALIGN/baselines/envs/requirements.laddersym.lock.txt`
- `uv pip check`: "All installed packages are compatible" (108 packages).

## Deviations (package, pinned, installed, reason)

| package | pinned in requirements.txt | installed | reason |
|---|---|---|---|
| `transformers` | `4.18.0` | `4.40.1` | Known blocker: 4.18.0 requires `tokenizers<0.13`, which has no cp311 wheel. 4.40.1 is the version Polytune uses. Every symbol LadderSym imports exists in 4.40.1: `T5Config`, `T5PreTrainedModel` (`models/laddersym_t5.py:19`, `models/t5_prompt.py:3`, `models/t5.py:4`, `tasks/laddersym_mt3_net.py:6`); `Seq2SeqLMOutput`, `BaseModelOutput`, `BaseModelOutputWithPastAndCrossAttentions`, `T5LayerNorm`, `T5Block` from `transformers.models.t5.modeling_t5`; `transformers.utils.logging`. `T5Block(config, has_relative_attention_bias=...)` and its `forward(...)` kwargs are unchanged between 4.18 and 4.40, so the repo's copied `T5Stack` works. Verified by construction + forward + backward + `generate` (see below). Do NOT move to transformers >= 4.46: `T5Block` gained `layer_idx` / `Cache` objects there and the repo's hand-copied `T5Stack` would break. |
| `hydra-core` | `1.2.0` | `1.3.0` | 1.2.0 fails at `import hydra` on Python 3.11: `ValueError: mutable default <class 'hydra.conf.JobConf.JobConfig.OverrideDirname'> for field override_dirname is not allowed: use default_factory` (Python 3.11 dataclass rule). 1.3.0 is the first Hydra release supporting Python 3.11 and still requires `omegaconf>=2.2,<2.4`, so `omegaconf==2.2.3` stays exactly pinned. The repo's `@hydra.main(..., version_base="1.1")` calls (`train_laddersym.py:132`, `test_laddersym.py:207`, `test_laddersym_coco.py:366`, `laddersym_test_inference.py:117`) are accepted by 1.3.x. Verified: `compose(config_name="config_maestro_prompted")` works. |
| `protobuf` | not pinned (transitive via `note-seq`/`wandb`; resolver chose `4.25.9`) | `3.20.3` | `import note_seq` fails under protobuf 4.x: `TypeError: Descriptors cannot be created directly. If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.` (note-seq 0.0.3 ships old `_pb2.py` files). 3.20.3 has no cp311 binary wheel, so it runs the pure-Python implementation (`api_implementation.Type() == 'python'`): slower proto parsing, functionally fine. Alternative not taken: keep 4.x and export `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` on every run. |
| `setuptools` | not pinned (uv default `84.0.0`) | `80.10.2` (`setuptools<81`) | setuptools >= 81 removed `pkg_resources`; `lightning_fabric/__init__.py:41` (pytorch-lightning 2.2.3) does `__import__("pkg_resources").declare_namespace(__name__)` -> `ModuleNotFoundError: No module named 'pkg_resources'`, which breaks `import pytorch_lightning`. A deprecation warning is still printed at import; harmless. |

## Pins kept exactly as in requirements.txt (all import and smoke-test OK on Py 3.11 / numpy 1.26.4)

torch 2.3.0, torchaudio 2.3.0, librosa 0.9.1, note-seq 0.0.3, pretty-midi 0.2.9, einops 0.4.1,
pytorch-lightning 2.2.3, wandb 0.16.6, immutabledict 4.2.0, timm 0.9.16, matplotlib 3.8.4, mir_eval 0.7,
huggingface_hub 0.23.0, numpy 1.26.4, scipy 1.13.0, soundfile 0.12.1, tqdm 4.66.2, PyYAML 6.0.1,
omegaconf 2.2.3, absl-py 2.1.0.

Runtime smoke tests that passed: `librosa.feature.melspectrogram / stft / resample / power_to_db / filters.mel`
(repo uses `librosa.load`, `librosa.midi_to_hz`, `librosa.util.frame`, `librosa.resample`);
`note_seq.NoteSequence` -> `note_sequence_to_pretty_midi` -> `midi_to_note_sequence` round trip;
`mir_eval.transcription.precision_recall_f1_overlap`; `pretty_midi.PrettyMIDI`; hydra compose of `config/config_maestro_prompted.yaml`.

Notable transitive versions chosen by the resolver (unpinned upstream): torchvision 0.18.0 (from timm), tokenizers 0.19.1,
safetensors 0.8.0, torchmetrics 1.9.0, pandas 3.0.5, numba 0.67.0, llvmlite 0.49.0, scikit-learn 1.9.0, ipython 9.17.1 (from note-seq).
No CUDA-only packages were installed (torch 2.3.0 macOS arm64 wheel = CPU + MPS).

## Verification (run from `/Users/tong172/Desktop/projects/ALIGN/baselines/LadderSym`)

```
PY=/Users/tong172/Desktop/projects/ALIGN/baselines/envs/laddersym/bin/python
$PY -c "import torch, pytorch_lightning, transformers, note_seq, hydra, librosa, mir_eval, pretty_midi, timm, einops, wandb, omegaconf, soundfile; print(torch.__version__, transformers.__version__, torch.backends.mps.is_available())"
# -> 2.3.0 4.40.1 True
$PY -c "import tasks.laddersym_mt3_net, dataset.dataset_2_random, contrib.vocabularies, models.laddersym_t5, models.ladder, models.ladder_backbone, models.t5_prompt"
# -> OK (no repo code changes were needed)
```

Harmless warnings printed at import: `pkg_resources is deprecated as an API` (lightning_fabric) and
`pydub ... Couldn't find ffmpeg or avconv` (note_seq imports pydub; ffmpeg is not on PATH; only matters if pydub decodes non-WAV audio).

## Model construction on CPU (no data, no downloads)

`OmegaConf.load("config/model/laddersym_MT3Net.yaml")`, `cfg.config.use_prompt = True`,
`T5Config.from_dict(OmegaConf.to_container(cfg.config))`, `models.laddersym_t5.T5ForConditionalGeneration(t5cfg, use_prompt=True)`:

- Construction OK in ~1.5 s on CPU.
- Parameters: total 172,434,640 (all trainable); encoder (`models.ladder.LadderSymEncoder`, ViT-B/16 ladder, in_chans=1) 145,421,008;
  decoder (`T5Stack`, 8 layers, d_model 512, d_ff 1024, 6 heads) 25,965,056; `lm_head` 786,432; `proj` + `decoder_embed_tokens` 262,144.
- Pretrained weights: construction does NOT try to download anything. `models/laddersym_t5.py:71` hard-codes
  `laddersym_base_patch16_224(pretrained=False)`, so timm's `build_model_with_cfg` never calls `load_pretrained`.
  Verified with a socket-level tripwire plus wrappers on `torch.hub.load_state_dict_from_url`,
  `timm.models._builder.load_pretrained`, `timm.models._hub.download_cached_file`/`load_state_dict_from_hf`,
  `huggingface_hub.hf_hub_download`: 0 calls, 0 connection attempts. No `./pretrained_models` directory was created
  (note `models/vit.py:3` sets `os.environ["TORCH_HOME"] = "./pretrained_models"` at import time as a side effect).
- Extra runtime check of the transformers substitution: dummy batch (`mistake_inputs`/`score_inputs` of shape (1, 256 frames, 512 mel bins),
  8 prompt tokens, 32 target tokens) -> logits (1, 40, 1536), finite; CE after slicing prompt 7.69 (~ln 1536 = 7.34);
  train-mode forward+backward gives grads to 738/738 parameters; `model.generate(..., max_length=4)` returns a (1, 5) tensor.

## Reproduce

```
uv venv --python /usr/local/bin/python3.11 /Users/tong172/Desktop/projects/ALIGN/baselines/envs/laddersym
uv pip install --python /Users/tong172/Desktop/projects/ALIGN/baselines/envs/laddersym/bin/python \
  -r /Users/tong172/Desktop/projects/ALIGN/baselines/envs/requirements.laddersym.lock.txt
```
(or install `LadderSym/requirements.txt` with `transformers==4.40.1`, `hydra-core==1.3.0`, then `protobuf==3.20.3 "setuptools<81"`).


## Added 2026-09-08 (baseline wiring)

| package | pinned | installed | reason |
|---|---|---|---|
| pretty-midi | 0.2.9 | **0.2.10** | 0.2.9 calls `np.int` (removed in numpy 1.24+), so `note_seq.midi_file_to_note_sequence` raised `MIDIConversionError` on every MIDI file with the pinned numpy 1.26.4 — this broke the LadderSym loader, prompt construction and evaluation. 0.2.10 is a pure-Python bugfix release (API-identical; the Polytune venv already uses it). Installed with `uv pip install --no-deps pretty_midi==0.2.10`. |
