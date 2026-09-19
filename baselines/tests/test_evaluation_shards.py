import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('evaluation', Path(__file__).parents[1] / 'scripts/evaluate_validation.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationShardsTests(unittest.TestCase):
    def test_partition_covers_each_validation_clip_once(self):
        durations = dict(a=100, b=80, c=20, d=15, e=5)
        shards = evaluation.balanced_shards(list(durations), durations, 2)
        self.assertEqual(sorted(sum(shards, [])), sorted(durations))
        self.assertFalse(set(shards[0]) & set(shards[1]))
        self.assertEqual([sum(durations[k] for k in s) for s in shards], [115, 105])

    def test_merge_rejects_missing_and_overlapping_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / 'a'; a.mkdir()
            (a / 'evaluated_ids.json').write_text(json.dumps(['clip']))
            with self.assertRaises(ValueError):
                evaluation.merge_predictions(['clip'], [a], root / 'out')
            (a / 'clip').mkdir(); (a / 'clip/mix.mid').touch()
            with self.assertRaises(ValueError):
                evaluation.merge_predictions(['clip'], [a, a], root / 'out')
            evaluation.merge_predictions(['clip'], [a], root / 'out')
            self.assertTrue((root / 'out/clip/mix.mid').is_file())


if __name__ == '__main__':
    unittest.main()
