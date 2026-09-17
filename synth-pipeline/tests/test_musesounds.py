from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from synthpipeline.musesounds import prepare_bb_clarinet_xml
from synthpipeline.regenerate_audio import _split_shards, regenerate_bundle


MIN_XML = """<?xml version="1.0" encoding="utf-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name>Clarinet</part-name>
      <part-abbreviation>Cl</part-abbreviation>
      <score-instrument id="P1-I1">
        <instrument-name>Clarinet</instrument-name>
      </score-instrument>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
    </measure>
  </part>
</score-partwise>
"""


def test_split_shards_round_robin() -> None:
    shards = _split_shards(list(range(10)), 4)
    assert [len(s) for s in shards] == [3, 3, 2, 2]
    assert sorted(x for shard in shards for x in shard) == list(range(10))


def test_prepare_bb_clarinet_xml_maps_woodwinds(tmp_path: Path) -> None:
    src = tmp_path / "in.musicxml"
    dest = tmp_path / "out.musicxml"
    src.write_text(MIN_XML, encoding="utf-8")
    prepare_bb_clarinet_xml(src, dest)
    text = dest.read_text(encoding="utf-8")
    assert "Clarinet in Bb" in text
    assert "wind.reed.clarinet.bflat" in text
    assert "<step>B</step>" in text
    assert "<alter>-1</alter>" in text
    assert "<octave>3</octave>" in text
    assert src.read_text(encoding="utf-8") == MIN_XML


def test_musesounds_rerender_skips_when_already_marked(tmp_path: Path) -> None:
    sample = tmp_path / "bundle"
    sample.mkdir()
    (sample / "verified_score.musicxml").write_text(MIN_XML, encoding="utf-8")
    (sample / "performance_score.musicxml").write_text(MIN_XML, encoding="utf-8")
    (sample / "metadata.json").write_text(
        json.dumps({"audio_render": "musesounds_v1"}), encoding="utf-8"
    )
    assert regenerate_bundle(sample, backend="musesounds") == "skip_done"


def test_musesounds_rerender_writes_wavs_and_mark(tmp_path: Path) -> None:
    sample = tmp_path / "bundle"
    sample.mkdir()
    (sample / "verified_score.musicxml").write_text(MIN_XML, encoding="utf-8")
    (sample / "performance_score.musicxml").write_text(MIN_XML, encoding="utf-8")
    (sample / "metadata.json").write_text(
        json.dumps({"sounding_transpose": -2, "audio_render": "soundfont_v1"}),
        encoding="utf-8",
    )
    (sample / "labels.json").write_text(
        json.dumps({"schema_version": "1.2", "labels": []}), encoding="utf-8"
    )

    def fake_export(xml_path, mp3_path, **kwargs):
        Path(mp3_path).write_bytes(b"mp3")
        return {"exit": 0}

    def fake_wav(mp3_path, wav_path, **kwargs):
        Path(wav_path).write_bytes(b"RIFF")
        return Path(wav_path)

    with (
        patch("synthpipeline.musesounds.export_musesounds_mp3", side_effect=fake_export),
        patch("synthpipeline.musesounds.mp3_to_wav", side_effect=fake_wav),
        patch("synthpipeline.regenerate_audio._finalize_musesounds_bundle"),
    ):
        result = regenerate_bundle(sample, backend="musesounds", force=True)
    assert result == "converted"
    assert (sample / "reference_audio.wav").is_file()
    assert (sample / "performance_audio.wav").is_file()
