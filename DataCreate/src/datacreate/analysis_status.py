"""Keep model abstention distinct from a completed evaluation with no labels."""
from __future__ import annotations

import json
from pathlib import Path


UNCERTAIN_MESSAGE = (
    "The recording could not be aligned reliably with the selected score. "
    "Error labels were withheld; this is not a zero-error result. "
    "Check that the score matches the recorded passage and uses written B-flat "
    "clarinet pitches, then submit the take again."
)


def document_status(document: dict) -> str | None:
    """Recognize raw feedback, UI alignment, and agent-label document status."""
    for container in (document, document.get("summary"), document.get("agent_labeling"),
                      document.get("diagnostics")):
        if isinstance(container, dict) and container.get("status"):
            return container["status"]
    return None


def sample_assessment(sample: Path) -> dict | None:
    path = sample / "note_alignment_v2.json"
    if not path.is_file():
        return None  # Older engines do not publish an assessment status.
    document = json.loads(path.read_text(encoding="utf-8"))
    status = document_status(document)
    if status is None:
        return None
    summary = document.get("summary") or {}
    diagnostics = document.get("diagnostics") or {}
    location = diagnostics.get("passage_location") or summary.get("passage_location") or {}
    withheld = sum(diagnostics.get(key, 0) for key in ("extras_withheld", "missed_withheld"))
    count = len(document.get("labels") or [])
    if status != "ok":
        message = UNCERTAIN_MESSAGE
        if location.get("status") == "ambiguous":
            message = ("Several score passages match this recording equally well. "
                       "Error labels were withheld. Select or upload the intended score excerpt and try again.")
    elif count == 0 and withheld:
        message = (f"No issues passed the detector's confidence checks. {withheld} possible "
                   "extra or missed notes were withheld as uncertain. This does not establish "
                   "an error-free performance.")
    elif count == 0:
        message = "No specific issues were marked. This does not establish an error-free performance."
    else:
        message = f"The detector marked {count} possible issues."
    return {"status": status, "label_count": count, "withheld_count": withheld,
            "passage_location": location,
            "match_fraction": diagnostics.get("match_fraction", summary.get("match_fraction")),
            "minimum_match_fraction": diagnostics.get("minimum_match_fraction"),
            "transcribed_note_count": summary.get("transcribed_note_count"),
            "score_event_count": summary.get("score_event_count"), "message": message}
