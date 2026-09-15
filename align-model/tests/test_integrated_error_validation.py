from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from eval_error_heads_integrated import (  # noqa: E402
    REQUESTED_TYPES,
    _deny_target_column,
    _schema_counts,
)


def test_inference_sqlite_authorizer_denies_target_column() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE records(candidates BLOB,target BLOB)")
    connection.execute("INSERT INTO records VALUES(?,?)", (b"candidate", b"gold"))
    connection.set_authorizer(_deny_target_column)
    assert connection.execute("SELECT candidates FROM records").fetchone() == (
        b"candidate",
    )
    try:
        connection.execute("SELECT target FROM records").fetchone()
    except sqlite3.DatabaseError:
        pass
    else:
        raise AssertionError("target column read was not denied")
    connection.close()


def test_requested_schema_metric_excludes_repetition_context() -> None:
    labels = [
        {
            "type": "wrong_note",
            "pitches": [60, 62, 64],
            "score_part": {"start_note_index": 3, "end_note_index": 5},
        },
        {
            "type": "repetition",
            "pitches": [60, 62, 64],
            "score_part": {"start_note_index": 3, "end_note_index": 5},
            "extra_copies": 1,
        },
    ]
    counts = _schema_counts(labels, labels, types=REQUESTED_TYPES)
    assert counts == {"correct": 1.0, "predicted": 1, "target": 1}
