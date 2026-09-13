"""Small, shared safeguards used by the pinned upstream entry points."""
import json
from pathlib import Path


def load_model_weights(model, checkpoint_path):
    """Load raw, DDP, or Lightning weights into the inner model, strictly.

    Does not write to the source checkpoint or restore optimizer/epoch state.
    A checkpoint for another architecture must fail, not leave random weights.
    """
    import torch

    state = torch.load(checkpoint_path, map_location="cpu")
    for key in ("state_dict", "model_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Not a model state dict: {checkpoint_path}")
    for prefix in ("module.", "model."):
        if all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    print(f"Loaded all {len(state)} weight tensors from {checkpoint_path}", flush=True)


def validate_eval_dataset(dataset, root, split_json, split):
    with open(split_json) as handle:
        manifest = json.load(handle)
    expected = {
        Path(path).name.replace(".midi", ""): path
        for key, path in manifest["midi_filename"].items()
        if manifest["split"][key] == split
    }
    actual = {row["track_id"] for row in dataset}
    if not expected or actual != set(expected) or len(actual) != len(dataset):
        raise ValueError(f"Incomplete {split} dataset: missing={sorted(set(expected)-actual)}, "
                         f"unexpected={sorted(actual-set(expected))}")
    for row in dataset:
        row["split_filename"] = expected[row["track_id"]]
        for key in ("mistake_audio", "score_audio", "extra_notes_midi",
                    "removed_notes_midi", "correct_notes_midi", "prompt"):
            if key in row and not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])


def select_eval_subset(dataset, first_n):
    dataset = sorted(dataset, key=lambda row: row["track_id"])
    if first_n is not None:
        if int(first_n) < 1:
            raise ValueError("--first-n must be positive")
        dataset = dataset[:int(first_n)]
    return dataset


def write_eval_selection(directory, dataset):
    directory = Path(directory)
    (directory / "evaluated_ids.json").write_text(json.dumps(
        [row["track_id"] for row in dataset], indent=2))
    (directory / "evaluated_split.json").write_text(json.dumps({
        "midi_filename": {str(i): row["split_filename"] for i, row in enumerate(dataset)},
        "split": {str(i): "test" for i in range(len(dataset))},
    }, indent=2))
