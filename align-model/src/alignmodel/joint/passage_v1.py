"""Cheap score subsequence retrieval and bounded, lazy restart expansion.

All spans are half-open canonical score-note indices. Retrieval uses rolling
rows (O(score notes) memory), never a full score x performance backtrace.
"""
from dataclasses import dataclass, asdict
from heapq import nsmallest

import numba
import numpy as np

from .restarts_v4 import local_hypotheses, schedule_restarts

REVISION = "v9-passage-v2"


@dataclass(frozen=True)
class PassageConfig:
    full_score_limit: int = 128
    max_score_notes: int = 512
    max_candidates: int = 4
    padding_notes: int = 8
    minimum_notes: int = 8
    minimum_similarity: float = 0.52
    ambiguity_margin: float = 0.04
    max_locator_cells: int = 20_000_000


@numba.njit(cache=True)
def _subsequence_costs(heard, written, confidence):
    """Free score prefix/suffix, but every heard note participates in the cost."""
    m = len(written)
    previous = np.zeros(m + 1, np.float64)
    starts = np.arange(m + 1)
    matches = np.zeros(m + 1, np.int64)
    for i in range(len(heard)):
        row = np.empty(m + 1, np.float64)
        origins = np.empty(m + 1, np.int64)
        exact = np.empty(m + 1, np.int64)
        weight = max(0.35, min(1.0, confidence[i]))
        row[0] = previous[0] + 0.8 * weight
        origins[0] = 0
        exact[0] = 0
        for j in range(1, m + 1):
            equal = heard[i] == written[j - 1]
            diagonal = previous[j - 1] + (0.0 if equal else weight)
            insert = previous[j] + 0.8 * weight
            delete = row[j - 1] + 0.8
            if diagonal <= insert and diagonal <= delete:
                row[j] = diagonal
                origins[j] = starts[j - 1]
                exact[j] = matches[j - 1] + int(equal)
            elif insert <= delete:
                row[j] = insert
                origins[j] = starts[j]
                exact[j] = matches[j]
            else:
                row[j] = delete
                origins[j] = origins[j - 1]
                exact[j] = exact[j - 1]
        previous, starts, matches = row, origins, exact
    return previous, starts, matches


def locate_passages(notes, score, config=PassageConfig()):
    """Return distinct plausible passages, or an explicit unassessed result."""
    reliable = [n for n in notes if not n.optional and n.confidence >= 0.2]
    info = {"revision": REVISION, "score_notes": len(score),
            "heard_notes": len(reliable), "config": asdict(config), "candidates": []}
    if not reliable or not score:
        return [], {**info, "status": "insufficient_evidence"}
    if len(score) <= config.full_score_limit:
        return [(0, len(score))], {**info, "status": "provided_excerpt"}
    if len(reliable) < config.minimum_notes:
        return [], {**info, "status": "insufficient_evidence"}
    if len(reliable) * len(score) > config.max_locator_cells:
        return [], {**info, "status": "search_budget_exceeded"}
    costs, starts, matches = _subsequence_costs(
        np.array([n.pitch for n in reliable], np.int64),
        np.array([e.pitch for e in score], np.int64),
        np.array([n.confidence for n in reliable], np.float64),
    )
    scale = sum(max(.35, min(1., n.confidence)) for n in reliable)
    selected = []
    for end in np.argsort(costs[1:], kind="stable") + 1:
        end = int(end)
        start = int(starts[end])
        similarity = max(0., 1. - float(costs[end]) / scale)
        if similarity < config.minimum_similarity:
            break
        if end - start < config.minimum_notes or matches[end] < config.minimum_notes:
            continue
        # Neighboring endpoints for one passage are not independent locations.
        if any(max(0, min(end, b) - max(start, a)) >= .5 * min(end-start, b-a)
               for a, b in selected):
            continue
        lo, hi = max(0, start-config.padding_notes), min(len(score), end+config.padding_notes)
        if hi - lo > config.max_score_notes:
            continue
        selected.append((start, end))
        info["candidates"].append({"start": lo, "end": hi,
            "core_start": start, "core_end": end, "similarity": similarity,
            "exact_matches": int(matches[end]),
            "start_measure": score[start].measure, "end_measure": score[end-1].measure})
        if len(selected) == config.max_candidates:
            break
    info["status"] = "located" if selected else "no_confident_bounded_passage"
    return [(c["start"], c["end"]) for c in info["candidates"]], info


def bounded_hypotheses(score, notes, primary_count, *, max_hypotheses=1024, max_local_proposals=12):
    """Retain small span descriptors; construct each expanded schedule on demand.

    Local acoustic replay proposals receive priority. The remaining budget goes
    to measure spans whose expanded length is closest to the transcription.
    No full-score grammar_hypotheses() call or list of expanded schedules.
    """
    if max_hypotheses < 1:
        raise ValueError("max_hypotheses must be positive")
    yield schedule_restarts(score, ())
    remaining = max_hypotheses - 1
    seen = set()
    for h in local_hypotheses(score, notes, max_proposals=max_local_proposals):
        if not remaining:
            return
        key = h.restart_regions
        if key in seen:
            continue
        seen.add(key)
        remaining -= 1
        yield h
    # Group by contiguous measure/part runs, retaining canonical order.
    boundaries = [0]
    for i in range(1, len(score)):
        if (score[i].part, score[i].measure) != (score[i-1].part, score[i-1].measure):
            boundaries.append(i)
    boundaries.append(len(score))
    def descriptors():
        for left in range(len(boundaries)-1):
            for right in range(left+1, len(boundaries)):
                a, b = boundaries[left], boundaries[right]
                for copies in (1, 2):
                    key = ((a, b, copies),)
                    if key not in seen:
                        yield (abs(len(score)+(b-a)*copies-primary_count), a, b, copies)
    for _, a, b, copies in nsmallest(remaining, descriptors()):
        yield schedule_restarts(score, ((a, b, copies),))
