from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alignmodel.transcription.canonical import (
    infer_note_decoder,
    load_note_decoder,
)
from alignmodel.transcription.decode import TransNote


class CanonicalNoteDecoderTests(unittest.TestCase):
    def test_basic_pitch_manifest_loads_and_uses_explicit_shift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sample = root / "sample"
            sample.mkdir()
            (sample / "performance_audio.wav").write_bytes(b"wav")
            manifest = root / "note_decoder.json"
            manifest.write_text(
                json.dumps(
                    {
                        "kind": "basic-pitch",
                        "format_version": 3,
                        "effective_audio_transpose": 4,
                        "corpus": "procedural12k",
                        "basic_cache_root": str(root / "cache"),
                    }
                ),
                encoding="utf-8",
            )
            decoder = load_note_decoder(manifest)
            expected = [TransNote(60, 0.1, 0.4, 0.9)]
            with (
                patch(
                    "alignmodel.transcription.canonical.extract_basic_pitch_features",
                    return_value=object(),
                ) as extract,
                patch(
                    "alignmodel.transcription.canonical.decode_basic_pitch_features",
                    return_value=expected,
                ),
                patch(
                    "alignmodel.transcription.canonical.sanitize_basic_pitch_notes",
                    return_value=expected,
                ),
            ):
                result = infer_note_decoder(decoder, sample)
            self.assertEqual(result, expected)
            metadata = extract.call_args.kwargs["source_metadata"]
            self.assertEqual(metadata["effective_audio_transpose"], 4)
            self.assertIn("procedural12k", str(extract.call_args.kwargs["cache_path"]))


if __name__ == "__main__":
    unittest.main()
