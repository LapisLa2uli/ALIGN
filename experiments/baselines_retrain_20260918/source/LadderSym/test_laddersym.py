from align_runtime import load_model_weights, validate_eval_dataset, select_eval_subset, write_eval_selection
import os
import torch
import inference_error
from inference_error import InferenceHandler
from glob import glob
from tqdm import tqdm, tqdm as std_tqdm
import librosa
import hydra
import numpy as np
import json

from evaluate_errors import evaluate_main


def _pick_device():
    """cuda -> mps -> cpu, or whatever LADDERSYM_DEVICE names (e.g. 'cpu' to force CPU on a Mac)."""
    forced = os.environ.get("LADDERSYM_DEVICE", "").strip()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _validate_eval_inputs(cfg):
    help_msg = (
        "Set dataset paths via env vars:\n"
        "  LADDERSYM_MAESTRO_ROOT, LADDERSYM_MAESTRO_SPLIT_JSON\n"
        "or pass Hydra overrides like:\n"
        "  dataset.test.root_dir=/path dataset.test.split_json_path=/path/to/split.json"
    )
    if cfg.dataset.test.root_dir is None or str(cfg.dataset.test.root_dir).strip() == "":
        raise ValueError(f"dataset.test.root_dir is empty.\n{help_msg}")
    if cfg.dataset.test.split_json_path is None or str(cfg.dataset.test.split_json_path).strip() == "":
        raise ValueError(f"dataset.test.split_json_path is empty.\n{help_msg}")
    if not os.path.isdir(cfg.dataset.test.root_dir):
        raise FileNotFoundError(
            f"dataset.test.root_dir does not exist or is not a directory: {cfg.dataset.test.root_dir}"
        )
    if not os.path.isfile(cfg.dataset.test.split_json_path):
        raise FileNotFoundError(
            "dataset.test.split_json_path does not exist or is not a file: "
            f"{cfg.dataset.test.split_json_path}"
        )
    if cfg.path is None or str(cfg.path).strip() == "":
        raise ValueError("path is empty. Provide a checkpoint path via path=/path/to/model.ckpt")
    if not os.path.isfile(cfg.path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {cfg.path}")


def get_scores(
    model,
    mistakes_audio_dir=None,
    scores_audio_dir=None,
    mel_norm=True,
    eval_dataset="MAESTRO",
    exp_tag_name="test_midis",
    ground_truth=None,
    prompt_dir=None,
    verbose=True,
    contiguous_inference=False,
    batch_size=1,
    max_length=1024,
):
    if os.path.exists(exp_tag_name):
        raise FileExistsError(f"Prediction directory already exists: {exp_tag_name}; choose a new --tag")
    os.makedirs(exp_tag_name)
    write_eval_selection(exp_tag_name, ground_truth)
    device = _pick_device()
    print(f"Inference device: {device}", flush=True)
    handler = InferenceHandler(
        model=model,
        device=device,
        mel_norm=mel_norm,
        contiguous_inference=contiguous_inference,
    )

    def func(fname):
        audio, _ = librosa.load(fname, sr=16000)
        print(f"audio_len in seconds: {len(audio)/16000}")
        return audio
    
    if verbose:
        print("Total mistake audio files:", len(mistakes_audio_dir))
        print("Total score audio files:", len(scores_audio_dir))


    print(f"batch_size: {batch_size}")
    
    print("mistakes_audio_dir:", mistakes_audio_dir, flush=True)
    print("scores_audio_dir:", scores_audio_dir, flush=True)
    print("prompt_dir:", prompt_dir, flush=True)
    # for mistake_file, score_file in tqdm(zip(mistakes_audio_dir, scores_audio_dir), total=len(mistakes_audio_dir)):
    for mistake_file, score_file, prompt_file in tqdm(
        zip(mistakes_audio_dir, scores_audio_dir, prompt_dir),
        total=len(mistakes_audio_dir),
        dynamic_ncols=True,
        leave=False,
        mininterval=0.1,
        maxinterval=0.25,
        ncols=100
    ):
    
        # Process each file pair here
        print("Processing:", mistake_file, "and", score_file, "and", prompt_file)
        mistake_audio = func(mistake_file)
        score_audio = func(score_file)

        fname = mistake_file.split("/")[-2]
        outpath = os.path.join(exp_tag_name, fname, "mix.mid") 

        handler.inference(
            mistake_audio=mistake_audio,
            score_audio=score_audio,
            audio_path=fname,
            prompt_path=prompt_file,
            outpath=outpath,
            batch_size=batch_size,
            max_length=max_length,
            verbose=verbose,
        )

    if verbose:
        print("Evaluating...")
    current_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    scores, mean_scores = evaluate_main(
        dataset_name=eval_dataset,
        test_midi_dir=os.path.join(current_dir, exp_tag_name),
        ground_truth=ground_truth,
    )

    if verbose:
        for key in sorted(list(mean_scores)):
            print("{}: {:.4}".format(key, mean_scores[key]))

    return scores, mean_scores

def _load_MAESTRO_split_info(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        midi_filename_to_number = {
            os.path.basename(path).replace(".midi", ""): str(number)
            for number, path in data["midi_filename"].items()
        }
        split_to_numbers = {split: set() for split in set(data["split"].values())}
        for number, split in data["split"].items():
            split_to_numbers[split].add(str(number))
        return midi_filename_to_number, split_to_numbers

def _build_dataset(root_dir, json_path, split):
    # Load the mapping and splits
    midi_filename_to_number, split_to_numbers = _load_MAESTRO_split_info(json_path)
    desired_file_numbers = split_to_numbers[split]

    df = []
    mistakes_audio_dir = []
    scores_audio_dir = []

    # Patterns for file discovery
    extra_notes_pattern = os.path.join(root_dir, "label", "extra_notes", "**", "*.mid")
    removed_notes_pattern = os.path.join(root_dir, "label", "removed_notes", "**", "*.mid")
    correct_notes_pattern = os.path.join(root_dir, "label", "correct_notes", "**", "*.mid")
    mistake_pattern = os.path.join(root_dir, "mistake", "**", "mix.*")
    score_pattern = os.path.join(root_dir, "score", "**", "mix.*")

    # Find all file paths using the glob patterns and parse identifiers
    extra_notes_files = {
        os.path.normpath(f).split(os.sep)[-3]: f
        for f in glob(extra_notes_pattern, recursive=True)
    }
    removed_notes_files = {
        os.path.normpath(f).split(os.sep)[-3]: f
        for f in glob(removed_notes_pattern, recursive=True)
    }
    correct_notes_files = {
        os.path.normpath(f).split(os.sep)[-3]: f
        for f in glob(correct_notes_pattern, recursive=True)
    }
    mistake_files = {
        os.path.normpath(f).split(os.sep)[-2]: f
        for f in glob(mistake_pattern, recursive=True)
    }
    score_files = {
        os.path.normpath(f).split(os.sep)[-2]: f
        for f in glob(score_pattern, recursive=True)
    }

    # Match files based on the common identifier
    for track_id in sorted(extra_notes_files.keys()):  # Sorting ensures consistency
        file_number = midi_filename_to_number.get(track_id)
        if (
            file_number in desired_file_numbers
            and track_id in removed_notes_files
            and track_id in correct_notes_files
            and track_id in mistake_files
            and track_id in score_files
        ):
            df.append(
                {
                    "track_id": track_id,
                    "extra_notes_midi": extra_notes_files[track_id],
                    "removed_notes_midi": removed_notes_files[track_id],
                    "correct_notes_midi": correct_notes_files[track_id],
                    "prompt": score_files[track_id].replace(".wav", ".mid"),
                    "mistake_audio": mistake_files[track_id].replace(".mid", ".wav"),
                    "score_audio": score_files[track_id].replace(".mid", ".wav"),
                }
            )
            mistakes_audio_dir.append(mistake_files[track_id].replace(".mid", ".wav"))
            scores_audio_dir.append(score_files[track_id].replace(".mid", ".wav"))

    assert len(df) > 0, "No matching files found. Check the dataset directory."

    # Sort mistakes_audio_dir and scores_audio_dir for alignment
    mistakes_audio_dir = sorted(mistakes_audio_dir)
    scores_audio_dir = sorted(scores_audio_dir)

    return df, mistakes_audio_dir, scores_audio_dir

    


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    # keep the inference prompt padding identical to training (cfg.prompt_length -> Dataset.prompt_length)
    inference_error.dataset.prompt_length = int(cfg.get("prompt_length", inference_error.dataset.prompt_length))
    _validate_eval_inputs(cfg)
    assert (
        cfg.path.endswith(".pt")
        or cfg.path.endswith("pth")
        or cfg.path.endswith("ckpt")
    ), "Only .pt, .pth, .ckpt files are supported."
    assert cfg.eval.exp_tag_name
    cfg.model.config.use_prompt = cfg.use_prompt
    
    # Build the dataset and get file directories
    dataset, mistakes_audio_dir, scores_audio_dir = _build_dataset(
        root_dir=cfg.dataset.test.root_dir,
        json_path=cfg.dataset.test.split_json_path,
        split="test",
    )
    
    # Extract ground truth prompts
    prompt_dir = [entry["prompt"] for entry in dataset]

    # Load model
    validate_eval_dataset(dataset, cfg.dataset.test.root_dir,
                          cfg.dataset.test.split_json_path, "test")
    pl = hydra.utils.instantiate(cfg.model, optim_cfg=cfg.optim)
    print(f"Loading weights from: {cfg.path}")
    if cfg.path.endswith(".ckpt"):
        # Load lightning module from checkpoint
        model_cls = hydra.utils.get_class(cfg.model._target_)
        print("torch.cuda.device_count():", torch.cuda.device_count())

        pl = model_cls.load_from_checkpoint(
            cfg.path,
            map_location="cpu",
            config=cfg.model.config,
            optim_cfg=cfg.optim,
        )
        model = pl.model
        if os.environ.get("LADDERSYM_COMPILE", "") == "1":
            print("Compiling the model with torch.compile (LADDERSYM_COMPILE=1)...", flush=True)
            model = torch.compile(model)
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        num_params = count_parameters(model)
        print(f"Trainable parameters: {num_params:,}", flush=True)
    else:
        model = pl.model
        load_model_weights(model, cfg.path)

    model.eval()

    # Subset selection
    dataset = select_eval_subset(dataset, cfg.eval.eval_first_n_examples)
    mistakes_audio_dir = [entry["mistake_audio"] for entry in dataset]
    scores_audio_dir = [entry["score_audio"] for entry in dataset]
    prompt_dir = [entry["prompt"] for entry in dataset]

    mel_norm = True

    # Call the get_scores function
    get_scores(
        model,
        mistakes_audio_dir=mistakes_audio_dir,
        scores_audio_dir=scores_audio_dir,
        mel_norm=mel_norm,
        eval_dataset=cfg.eval.eval_dataset,
        exp_tag_name=cfg.eval.exp_tag_name,
        ground_truth=dataset,  # Pass aligned ground truth
        prompt_dir=prompt_dir,
        contiguous_inference=cfg.eval.contiguous_inference,
        batch_size=cfg.eval.batch_size,
        # eval.max_length (absent upstream -> 1024): new decoder tokens per 2.048 s segment
        # (on top of the prompt).  Smoke runs on CPU pass a small value; no KV cache is used.
        max_length=int(cfg.eval.get("max_length", None) or 1024),
    )


if __name__ == "__main__":
    main()
