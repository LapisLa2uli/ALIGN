"""Run once per baseline environment; uses actual patched modules, small tensors."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

RUN = Path(__file__).resolve().parents[1]
B = RUN / "source"
sys.path.insert(0, str(RUN.parents[1] / "baselines/common"))
sys.path.insert(0, str(B / os.environ.get("BASELINE_FLAVOR", "Polytune")))
from align_runtime import load_model_weights, select_eval_subset, validate_eval_dataset
from prepare_dataset import classes_from_note_labels, assign_splits, is_bundle
from eval_bridge import gt_notes_from_note_labels
from evaluate_notes import evaluate
from prepare_dataset import output_paths, write_label_midi
from audit_supervision import issues_for_labels
import pretty_midi
import inference_error
from contrib.event_codec import Event


class CheckpointTests(unittest.TestCase):
    def test_plain_and_lightning_load_all_weights_without_modifying_input(self):
        source = torch.nn.Linear(3, 2)
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / ("weights.ckpt" if wrapped else "weights.pt")
                state = source.state_dict()
                if wrapped:
                    state = {"state_dict": {"model." + k: v for k, v in state.items()}}
                torch.save(state, path)
                digest = hashlib.sha256(path.read_bytes()).digest()
                target = torch.nn.Linear(3, 2)
                load_model_weights(target, path)
                self.assertTrue(torch.equal(source.weight, target.weight))
                self.assertEqual(digest, hashlib.sha256(path.read_bytes()).digest())
                self.assertEqual(len(list(Path(tmp).iterdir())), 1)

    def test_wrong_architecture_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong.pt"
            torch.save({"wrong.weight": torch.zeros(1)}, path)
            with self.assertRaises(RuntimeError):
                load_model_weights(torch.nn.Linear(3, 2), path)


class DecodingTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("BASELINE_FLAVOR") == "LadderSym", "LadderSym prompt path")
    def test_cached_prompt_decoder_matches_full_prefix_with_padding(self):
        from transformers import T5Config
        from models.laddersym_t5 import T5Stack
        torch.manual_seed(17)
        cfg = T5Config(d_model=32, d_kv=8, d_ff=64, num_layers=1, num_heads=4,
                       vocab_size=32, is_decoder=True, is_encoder_decoder=False,
                       dropout_rate=0.0, use_cache=True, pad_token_id=0,
                       decoder_start_token_id=0)
        decoder = T5Stack(cfg, torch.nn.Embedding(32, 32), 'decoder', use_prompt=True).eval()
        prefix = torch.tensor([[5, 6, 0, 0, 0], [7, 8, 9, 0, 0]])
        mask = torch.tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 1]])
        context = torch.randn(2, 3, 32)
        with torch.no_grad():
            cached = decoder(input_ids=prefix, attention_mask=mask,
                             encoder_hidden_states=context, initial_prompt_lengths=[2, 3],
                             use_cache=True, return_dict=True)
            for token in (11, 12, 13):
                new_token = torch.full((2, 1), token)
                prefix = torch.cat([prefix, new_token], dim=1)
                mask = torch.cat([mask, torch.ones(2, 1, dtype=torch.long)], dim=1)
                full = decoder(input_ids=prefix, attention_mask=mask,
                               encoder_hidden_states=context, initial_prompt_lengths=[2, 3],
                               use_cache=False, return_dict=True)
                cached = decoder(input_ids=new_token, attention_mask=mask,
                                 encoder_hidden_states=context, initial_prompt_lengths=[2, 3],
                                 past_key_values=cached.past_key_values,
                                 use_cache=True, return_dict=True)
                torch.testing.assert_close(cached.last_hidden_state[:, -1],
                                           full.last_hidden_state[:, -1], rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(os.environ.get("BASELINE_FLAVOR") == "LadderSym", "LadderSym prompt path")
    def test_prompted_generation_starts_after_padded_prompt_like_training(self):
        from models.laddersym_t5 import T5ForConditionalGeneration
        prompt = torch.tensor([[8, 9, 0, 0], [7, 8, 9, 0]])
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        expected_ids = torch.cat([prompt, torch.zeros(2, 1, dtype=torch.long)], dim=1)
        expected_mask = torch.cat([mask, torch.ones(2, 1, dtype=torch.long)], dim=1)

        def decoder(**kwargs):
            self.assertTrue(torch.equal(kwargs['input_ids'], expected_ids))
            self.assertTrue(torch.equal(kwargs['attention_mask'], expected_mask))
            self.assertEqual(kwargs['initial_prompt_lengths'], [2, 3])
            logits = torch.zeros(2, 5, 10)
            logits[:, -1, 1] = 1  # EOS on the first generated event.
            return (logits,)

        model = types.SimpleNamespace(
            use_prompt=True, device=torch.device('cpu'),
            config=types.SimpleNamespace(pad_token_id=0, decoder_start_token_id=0, eos_token_id=1),
            encoder=lambda **kwargs: torch.zeros(2, 3, 4), decoder=decoder,
            lm_head=lambda hidden: hidden,
        )
        output = T5ForConditionalGeneration.generate(
            model, torch.zeros(2, 3, 4), torch.zeros(2, 3, 4),
            decoder_input_ids=prompt, decoder_attention_mask=mask, max_length=2,
        )
        self.assertEqual(output.tolist(), [[0, 1], [0, 1]])

    def test_length_capped_valid_tokens_decode_to_real_classified_notes(self):
        handler = inference_error.InferenceHandler(model=None, device="cpu")
        for cls in (1, 2, 3):
            events = [Event("tie", 0), Event("error_class", cls), Event("velocity", 1),
                      Event("pitch", 60), Event("shift", 50), Event("velocity", 0), Event("pitch", 60)]
            tokens = np.array([[handler.codec.encode_event(event) for event in events]])
            ns = handler._to_event([tokens], [np.array([[0.0]])])
            self.assertEqual(len(ns.notes), 1)
            self.assertEqual((ns.notes[0].pitch, ns.notes[0].instrument), (60, cls))
            self.assertAlmostEqual(ns.notes[0].start_time, 0.0)
            self.assertAlmostEqual(ns.notes[0].end_time, 0.5)

    @unittest.skipUnless(os.environ.get("BASELINE_FLAVOR") == "LadderSym", "LadderSym prompt path")
    def test_last_partial_prompt_preserves_its_event_boundaries(self):
        handler = inference_error.InferenceHandler(model=None, device="cpu")
        frames = np.zeros((400, 128), dtype=np.float32)
        times = np.arange(400) / 125
        features = {"targets": np.arange(401), "input_event_start_indices": np.arange(400),
                    "input_event_end_indices": np.arange(400) + 1,
                    "input_state_event_indices": np.zeros(400), "state_events": np.array([0])}
        *_, result = handler._split_token_into_length(
            frames, frames, times, times, features, return_prompt_row=True)
        self.assertEqual(result["prompt_event_start_indices"][-1][0], 128)
        self.assertEqual(result["prompt_event_end_indices"][-1][-1], 400)

    def test_no_eos_keeps_predictions_and_eos_trims(self):
        handler = inference_error.InferenceHandler.__new__(inference_error.InferenceHandler)
        handler.codec = types.SimpleNamespace(steps_per_second=100)
        for tokens, expected in (([5, 6, 7], [5, 6, 7]), ([5, -1, 7], [5]), ([-1], [])):
            with self.subTest(tokens=tokens):
                captured = []
                def decode(predictions, **kwargs):
                    captured.extend(predictions[0]["est_tokens"].tolist())
                    return {"est_ns": "decoded"}
                with patch.object(inference_error.metrics_utils, "event_predictions_to_ns", decode):
                    result = handler._to_event([np.array([tokens])], [np.array([[0.0]])])
                self.assertEqual(result, "decoded")
                self.assertEqual(captured, expected)

    def test_inference_exception_propagates(self):
        handler = inference_error.InferenceHandler.__new__(inference_error.InferenceHandler)
        handler.model = types.SimpleNamespace(config=types.SimpleNamespace(use_prompt=False))
        with patch.object(handler, "_preprocess", side_effect=RuntimeError("bad audio")):
            with self.assertRaisesRegex(RuntimeError, "bad audio"):
                handler.inference(np.zeros(160), np.zeros(160), "piece", outpath="unused.mid")


class LabelTests(unittest.TestCase):
    def test_unmapped_correct_reference_and_mixed_tie_are_excluded(self):
        labels = self.labels()
        labels["check"] = {"perf": {"ok": True}, "ref": {"ok": True}}
        labels["reference_notes"] = [
            {"cls": "correct", "perf_index": None, "tie_prev": False},
            {"cls": "missed", "perf_index": None, "tie_prev": True},
        ]
        reasons = issues_for_labels(labels)
        self.assertIn("correct_reference_without_performance", reasons)
        self.assertIn("mixed_class_reference_tie", reasons)
        self.assertIn("reference_note_count_mismatch", reasons)

    def labels(self):
        return {"schema_version": "1.0", "performance_notes": [], "missed_notes": [], "reference_notes": []}

    def test_missed_ties_match_converter_and_metrics(self):
        labels = self.labels()
        labels["missed_notes"] = [
            {"onset": 0, "offset": 1, "sounding_pitch": 60},
            {"onset": 1, "offset": 2, "sounding_pitch": 60, "tie_prev": True},
            {"onset": 4, "offset": 5, "sounding_pitch": 60, "copy": 1},
        ]
        classes, _ = classes_from_note_labels(labels)
        gt = gt_notes_from_note_labels(labels)
        self.assertEqual(len(classes["removed"]), 1)
        self.assertEqual(gt["missing"], [(0, 2, 60)])

    def test_unknown_class_fails_instead_of_disappearing(self):
        labels = self.labels()
        labels["performance_notes"] = [{"onset": 0, "offset": 1, "sounding_pitch": 60, "cls": "typo"}]
        with self.assertRaises(ValueError):
            classes_from_note_labels(labels)

    def test_note_map_is_not_a_supervised_note_labels_file(self):
        with self.assertRaises(ValueError):
            classes_from_note_labels({"schema_version": "1.0", "performed_notes": []})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("labels.json", "performance_audio.wav", "reference_audio.wav", "reference_audio.mid", "note_map.json"):
                (root / name).touch()
            with self.assertRaisesRegex(ValueError, "note_labels.json"):
                is_bundle(root, False)

    def test_source_split_never_falls_back_to_clip_split(self):
        records = [{"track_id": str(i), "set": "raw", "source": "one-score", "real_test": False} for i in range(6)]
        with self.assertRaises(ValueError):
            assign_splits(records, .2, .2, 365, ["raw"])


class EvalTests(unittest.TestCase):
    def test_unclassified_notes_are_not_guessed_or_silently_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, pred = Path(tmp) / 'data', Path(tmp) / 'pred'
            root.mkdir(); (pred / 'clip').mkdir(parents=True)
            (pred / 'evaluated_ids.json').write_text(json.dumps(['clip']))
            (root / 'manifest.json').write_text(json.dumps({'tracks': {'clip': {'real_test': False}}}))
            for name, pitch in [('extra', None), ('missing', 64), ('correct', 60)]:
                notes = [] if pitch is None else [dict(start=1.0, end=1.5, pitch=pitch)]
                write_label_midi(output_paths(root, 'clip')['removed' if name == 'missing' else name], notes)
            midi = pretty_midi.PrettyMIDI()
            for name, pitch in [('extra', None), ('correct', 60), ('', 64)]:
                track = pretty_midi.Instrument(0, name=name)
                if pitch is not None:
                    track.notes.append(pretty_midi.Note(90, pitch, 1.0, 1.5))
                midi.instruments.append(track)
            midi.write(str(pred / 'clip/mix.mid'))
            with self.assertRaisesRegex(ValueError, 'Unclassified'):
                evaluate(root, pred)
            result = evaluate(root, pred, allow_unclassified=True)
            self.assertEqual(result['unclassified_notes'], 1)
            self.assertEqual(result['micro']['missing']['F1'], 0)
            self.assertEqual(result['micro']['correct']['F1'], 1)
            self.assertEqual(result['micro']['all']['F1'], 1)
            self.assertEqual(result['micro']['class_aware']['F1'], 0.5)
            self.assertEqual(result['micro']['class_aware']['n_pred'], 2)
            midi.instruments[-1].name = 'unknown_typo'
            midi.write(str(pred / 'clip/mix.mid'))
            with self.assertRaisesRegex(ValueError, 'Unclassified'):
                evaluate(root, pred, allow_unclassified=True)

    def test_oracle_metrics_keep_classes_when_first_tracks_are_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            pred = Path(tmp) / "pred"
            root.mkdir(); pred.mkdir()
            ids = ["all_classes", "only_correct"]
            (pred / "evaluated_ids.json").write_text(json.dumps(ids))
            (root / "manifest.json").write_text(json.dumps({"tracks": {
                tid: {"real_test": False} for tid in ids}}))
            for tid in ids:
                pm = pretty_midi.PrettyMIDI()
                for i, name in enumerate(("extra", "missing", "correct")):
                    notes = [] if tid == "only_correct" and name != "correct" else [
                        {"start": 1.0, "end": 1.5, "pitch": 60 + i}]
                    write_label_midi(output_paths(root, tid)["removed" if name == "missing" else name], notes)
                    inst = pretty_midi.Instrument(71, name=name)
                    inst.notes = [pretty_midi.Note(90, n["pitch"], n["start"], n["end"]) for n in notes]
                    pm.instruments.append(inst)
                (pred / tid).mkdir()
                pm.write(str(pred / tid / "mix.mid"))
            result = evaluate(root, pred)
            for name in ("extra", "missing", "correct", "all"):
                self.assertEqual(result["micro"][name]["F1"], 1.0)
            (pred / "only_correct" / "mix.mid").unlink()
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                evaluate(root, pred)

    def test_subset_is_stable_and_sorted(self):
        rows = [{"track_id": "b"}, {"track_id": "a"}]
        self.assertEqual(select_eval_subset(rows, 1), [{"track_id": "a"}])

    def test_missing_split_track_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            path.write_text(json.dumps({"midi_filename": {"0": "set/a.midi"}, "split": {"0": "test"}}))
            with self.assertRaisesRegex(ValueError, "Incomplete test dataset"):
                validate_eval_dataset([], tmp, path, "test")


if __name__ == "__main__":
    unittest.main()
