from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "import_rawdata_mel_batch.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "import_rawdata_mel_batch", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
importer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = importer
SPEC.loader.exec_module(importer)


def test_source_number_uses_original_filename_number() -> None:
    assert importer.source_number(Path("古龙路 101(1).m4a")) == 101
    assert importer.source_number(Path("094.m4a")) is None


def test_plan_orders_source_numbers_and_allocates_following_ids(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "audio"
    samples = tmp_path / "samples"
    audio.mkdir()
    (samples / "093").mkdir(parents=True)
    (samples / "074").mkdir()
    (samples / "074" / "metadata.json").write_text(
        json.dumps(
            {
                "source_origin": "gulonglu",
                "source_origin_number": 82,
            }
        ),
        encoding="utf-8",
    )
    for name in ("古龙路 101(1).m4a", "古龙路 84.m4a", "古龙路 83.m4a"):
        (audio / name).write_bytes(b"audio")
    rows = importer.plan_import(audio, samples)
    assert [row["source_number"] for row in rows] == [83, 84, 101]
    assert [row["sample_id"] for row in rows] == ["094", "095", "096"]
    assert [Path(str(row["renamed_path"])).name for row in rows] == [
        "094.m4a",
        "095.m4a",
        "096.m4a",
    ]
