---
pretty_name: MAESTRO-E
license: cc-by-4.0
task_categories:
- audio-classification
- audio-to-audio
tags:
- music
- maestro
- piano
- error-detection
- multimodal
configs:
- config_name: default
  data_dir: viewer_preview
---

> Viewer note: `default` uses `viewer_preview/` for responsive audio playback.
> Full training/evaluation files remain available in the original folder structure.

# MAESTRO-E

MAESTRO-E dataset for music practice error detection with paired performance/reference inputs and note-level error labels.

## Paired Inputs for Error Detection

The model takes **paired inputs**:

- `mistake`: performance audio/MIDI containing musical errors
- `score`: paired reference score audio/MIDI (target/correct context)

Error supervision is provided with labels:

- `label/extra_notes`
- `label/removed_notes`
- `label/correct_notes`

## Full Structure (Code-Compatible)

This repo preserves the filesystem layout expected by the training loader:

- `mistake/<sample_id>/...`
- `score/<sample_id>/...`
- `label/{extra_notes,removed_notes,correct_notes}/<sample_id>/...`
- `maestro-v3.0.0.json` (included at repo root)

## Quick Start (Directly with Existing Code)

```python
from huggingface_hub import snapshot_download

root = snapshot_download("ben2002chou/maestro-e", repo_type="dataset")
# set root_dir = root
# set split_json_path = f"{root}/maestro-v3.0.0.json"
```

## Project Link

- https://github.com/ben2002chou/LadderSYM

## Citation

```bibtex
@inproceedings{
chou2026laddersym,
title={LadderSym: A Multimodal Interleaved Transformer for Music Practice Error Detection},
author={Benjamin Shiue-Hal Chou and Purvish Jajal and Nicholas John Eliopoulos and James C. Davis and George K Thiruvathukal and Kristen Yeon-Ji Yun and Yung-Hsiang Lu},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=cizuvfyQXs}
}
```
