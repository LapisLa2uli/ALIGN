---
pretty_name: CocoChorales-E
license: cc-by-4.0
task_categories:
- audio-classification
- audio-to-audio
tags:
- music
- cocochorales
- error-detection
- multimodal
configs:
- config_name: default
  data_dir: viewer_preview
---

> Viewer note: `default` uses `viewer_preview/` for responsive audio playback.
> Full training/evaluation files remain available in the original folder structure.

# CocoChorales-E

CocoChorales-E subset used by the LadderSym training pipeline.

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

- `mistake/<track_id>/...`
- `score/<track_id>/...`
- `label/{extra_notes,removed_notes,correct_notes}/<track_id>/...`
- `split.json` (included at repo root)

## Split/Size Summary

Rows from cache logic:

- train: 36,236
- validation: 4,504
- test: 4,284

Selected track IDs: 11,256

## Quick Start (Directly with Existing Code)

```python
from huggingface_hub import snapshot_download

root = snapshot_download("ben2002chou/CocoChorales-E", repo_type="dataset")
# set root_dir = root
# set split_json_path = f"{root}/split.json"
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
