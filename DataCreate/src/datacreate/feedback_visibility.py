"""Explain final model feedback without promoting raw alignment hypotheses."""
from __future__ import annotations

import json
import math
from pathlib import Path


def has_playback(label: dict) -> bool:
    return all(isinstance(label.get(k), (int, float)) and math.isfinite(label[k])
               for k in ('start_time', 'end_time'))


def score_only_feedback(payload: dict, sample: Path | None = None) -> list[dict]:
    """Recover accepted score labels whose playback time was unavailable.

    Older exports omitted these rows. Read only the matching versioned feedback,
    with identical input/candidate provenance; never borrow another model's labels.
    """
    status = (payload.get('diagnostics') or {}).get('status', (payload.get('summary') or {}).get('status'))
    if status != 'ok':
        return []
    if 'score_only_labels' in payload:
        return list(payload['score_only_labels'] or [])
    if sample is None:
        return []
    provenance = payload.get('provenance') or {}
    method = (payload.get('label_generation') or {}).get('method')
    name = {'align_stack_v9': 'feedback_v9.json', 'align_stack_v10': 'feedback_v10.json'}.get(method)
    if name is None or not (sample / name).is_file():
        return []
    try:
        feedback = json.loads((sample / name).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    source = feedback.get('provenance') or {}
    keys = ('candidate_sha256', 'audio_sha256', 'score_sha256')
    if any(not provenance.get(k) or source.get(k) != provenance[k] for k in keys):
        return []
    if source.get('pipeline_revision') != provenance.get('pipeline_revision') or feedback.get('status') != 'ok':
        return []
    return [r for r in feedback.get('labels', []) if not has_playback(r)]


def feedback_review(payload: dict, sample: Path | None = None) -> dict | None:
    from datacreate.transcription_labeling import _has_model_feedback
    if not _has_model_feedback(payload):
        return None
    diagnostics = payload.get('diagnostics') or {}
    summary = payload.get('summary') or {}
    return {
        'status': diagnostics.get('status', summary.get('status')),
        'agent_label_count': len(payload.get('labels') or []),
        'extras_withheld': diagnostics.get('extras_withheld', 0),
        'missed_withheld': diagnostics.get('missed_withheld', 0),
        'missed_inferred': diagnostics.get('missed_inferred', 0),
        'score_only_labels': score_only_feedback(payload, sample),
        'ignored_candidates': sum(bool(n.get('ignored')) for n in payload.get('transcribed_notes', [])),
    }
