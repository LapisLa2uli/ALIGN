# ALIGN RUMAA-lite

Score-informed clarinet error detector for ALIGN bundles. Architecture follows **RUMAA** (Chang, Dixon, Benetos, WASPAA 2025), scaled to this project's MusicXML + log-mel samples rather than pretrained M3 / YourMT3+ weights.

## What it predicts

| Head | Stream | ALIGN types |
|------|--------|-------------|
| Score edit | one label per note in `verified_score.musicxml` | `match`, `miss`, `wrong`, `rhythm`, `intonation` |
| Insert | audio frames | `extra_note`, `repetition` spans |
| Repeat | clip | `metadata.repeated` / `repetition` labels |

Hierarchical fusion: score notes cross-attend to performance mel, then self-attend (RUMAA's audio-then-score order). Audio frames also attend back to the score for extras.

## Setup

Same conda env as DataCreate (`MusicEval`):

```powershell
cd "D:\stuff\Audio Evaluation\ALIGN"
conda activate MusicEval
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ./align-model
```

CPU works for a smoke run (`--overfit 8`). CUDA wheels are large; if `pip install torch --index-url https://download.pytorch.org/whl/cu124` times out, install the CPU build first:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Train

```powershell
align-model train --data ".\synth-pipeline\1000dataexport" --out ".\align-model\runs\rumaa-lite" --epochs 20 --batch-size 4
align-model train --data ".\synth-pipeline\1000dataexport" --overfit 8 --epochs 30 --batch-size 2
```

Writes `best.pt`, `last.pt`, and `history.json`.

## Infer

```powershell
align-model infer --ckpt ".\align-model\runs\rumaa-lite\best.pt" --sample ".\synth-pipeline\1000dataexport\synth_gen_0010"
```

Each sample needs `verified_score.musicxml`, `performance_mel.npy`, and `labels.json` (same layout as synth-pipeline output).
