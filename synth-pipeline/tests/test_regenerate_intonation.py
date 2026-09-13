from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from synthpipeline.regenerate_audio import regenerate_bundle


class RegenerateIntonationTests(unittest.TestCase):
    def test_ordinary_rerender_preserves_labeled_pitch_bends(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sample = Path(temp)
            for name in (
                "performance_audio.mid",
                "reference_audio.mid",
                "performance_score.musicxml",
                "verified_score.musicxml",
            ):
                (sample / name).write_bytes(b"x")
            (sample / "metadata.json").write_text(
                json.dumps({"sounding_transpose": -2}),
                encoding="utf-8",
            )
            (sample / "labels.json").write_text(
                json.dumps(
                    {
                        "labels": [
                            {
                                "type": "intonation_error",
                                "start_time": 1.0,
                                "end_time": 1.5,
                                "deviation_cents": -55.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch(
                    "synthpipeline.regenerate_audio.resolve_bundle_soundfont",
                    return_value=(Path("unused.sf2"), 0),
                ),
                patch(
                    "synthpipeline.regenerate_audio.render_midi_clarinet"
                ) as render,
                patch(
                    "synthpipeline.regenerate_audio.midi_note_transpose",
                    return_value=0,
                ),
                patch(
                    "synthpipeline.regenerate_audio._load_mono",
                    return_value=(np.zeros(32, np.float32), 22050),
                ),
                patch("synthpipeline.regenerate_audio._rewrite_mel"),
                patch(
                    "synthpipeline.pitch_convention.infer_midi_pitch_space",
                    return_value=("sounding", -2),
                ),
            ):
                result = regenerate_bundle(
                    sample, force=True, strip_ornaments=False
                )
            self.assertEqual(result, "converted")
            performance_call = next(
                call
                for call in render.call_args_list
                if Path(call.args[0]).name.startswith("performance")
            )
            bends = performance_call.kwargs["pitch_bends"]
            self.assertEqual(len(bends), 1)
            self.assertEqual(bends[0]["cents"], -55.0)


if __name__ == "__main__":
    unittest.main()
