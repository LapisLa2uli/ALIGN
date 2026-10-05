"""V9 acoustics with passage retrieval and bounded detailed alignment.

The source score is never cropped or renumbered on disk. Only internal windows
use local indices; every published event/label maps back to the full index.
"""
from dataclasses import replace

from .index import ScoreEventIndex
from .ornament_mapper_v1 import score_ornament_patterns
from .passage_v1 import PassageConfig, locate_passages, REVISION
from .robust_dp_aligner_v2 import RobustDPCosts, _as_notes
from .robust_dp_aligner_v3 import GateConfig
from .robust_dp_aligner_v4 import gate_v4
from .robust_dp_aligner_v5 import AlignerV5Config, AlignmentV5, align_v5
from .stack_v7 import HOP
from .stack_v9 import feedback
from ..transcription.transition_v7 import DecodeV7Config, decode_v7
from ..transcription.same_pitch_v2 import SamePitchConfig, repair_same_pitch


def _global_alignment(alignment, offset):
    span = lambda value: (value[0]+offset, value[1]+offset) if value is not None else None
    return replace(alignment,
        events=tuple(replace(e, score_span=span(e.score_span)) for e in alignment.events),
        deletions=frozenset(i+offset for i in alignment.deletions),
        missed=tuple(replace(m, score_index=m.score_index+offset) for m in alignment.missed),
        source_span=span(alignment.source_span),
        restart_regions=tuple((a+offset, b+offset, c) for a,b,c in alignment.restart_regions))


def align_notes(rows, index, score_path, candidate, *, evidence, config=None, progress=None):
    """Shared deployable path and test seam for an already decoded transcription."""
    config = dict(config or {})
    locator = PassageConfig(**config.get("passage", {}))
    notes = _as_notes(rows)
    windows, location = locate_passages(notes, index.events, locator)
    if progress:
        progress("aligner")
    limits = AlignerV5Config(costs=RobustDPCosts(**candidate['aligner_costs']),
        **(candidate['aligner_v3'] | config.get("alignment_limits", {})))
    results = []
    # Extract score ornaments once, not once per candidate passage.
    patterns = score_ornament_patterns(score_path, index.events) if windows else ()
    for position, (a, b) in enumerate(windows):
        local_score = tuple(replace(e, index=i) for i, e in enumerate(index.events[a:b]))
        try:
            alignment = align_v5(rows, local_score, score_path, limits, patterns=patterns[a:b])
        except ValueError as error:
            location.setdefault("rejected", []).append({"start": a, "end": b, "reason": str(error)})
            continue
        retrieval = location['candidates'][position]['similarity'] if location['candidates'] else 1.
        rank = alignment.cost / max(1, len(notes)) + .4 * (1. - retrieval)
        results.append((rank, a, b, alignment, local_score, retrieval))
        if location['candidates']:
            location['candidates'][position].update(alignment_cost=alignment.cost,
                match_fraction=alignment.match_fraction, rank=rank)
    location['limits'] = {k: getattr(limits, k) for k in
        ('max_score_notes', 'max_hypotheses', 'max_dp_cells', 'max_template_notes')}
    if not results:
        empty = AlignmentV5(events=(), deletions=frozenset(), cost=0., source_span=None,
            copies=0, kept_note_indices=tuple(range(len(notes))), extras=(), missed=(),
            notes=tuple(notes), artifact_notes=frozenset(), seconds_per_ql=0., match_fraction=0.)
        return empty, (), frozenset(), {'status':'alignment_uncertain', 'clip_abstained':True,
            'match_fraction':0., 'passage_location':location, 'pipeline_revision':REVISION}
    results.sort(key=lambda row: row[0])
    rank, a, b, alignment, local_score, retrieval = results[0]
    ambiguous = (len(results) > 1 and abs(results[1][5]-retrieval) <= locator.ambiguity_margin
                 and results[1][0]-rank <= .08)
    location.update(selected_start=a, selected_end=b,
        status='ambiguous' if ambiguous else location['status'])
    events, deletions, info = gate_v4(alignment, local_score, evidence,
        GateConfig(**(candidate['gate'] | config.get('gate', {}))),
        minimum_match_fraction=config.get('minimum_match_fraction', .45))
    if location['candidates']:
        # Padding assists boundary matching; it is not evidence that those notes
        # were attempted. Do not emit missing-note errors outside audible coverage.
        covered = [i for e in alignment.events if e.score_span for i in range(*e.score_span)]
        if covered:
            lo, hi = min(covered), max(covered)+1
            deletions = frozenset(i for i in deletions if lo <= i < hi)
            location.update(assessed_start=lo+a, assessed_end=hi+a,
                start_measure=index.events[lo+a].measure, end_measure=index.events[hi+a-1].measure)
    if ambiguous:
        events, deletions = (), frozenset()
        info.update(status='alignment_uncertain', clip_abstained=True)
    info.update(passage_location=location, pipeline_revision=REVISION)
    return (_global_alignment(alignment, a),
        tuple(replace(e, score_span=(e.score_span[0]+a,e.score_span[1]+a) if e.score_span else None) for e in events),
        frozenset(i+a for i in deletions), info)


def align_outputs(outputs, score_path, candidate, *, mel, audio=None, presence=None,
                  config=None, progress=None):
    config = dict(config or {})
    rows, decode_info = decode_v7(outputs, config=DecodeV7Config(**config.get('decoder', candidate['decoder'])))
    rows, repair = repair_same_pitch(rows, outputs, mel, audio=audio,
                                    config=SamePitchConfig(**config.get('same_pitch', {})))
    if progress:
        progress("locator")
    index = ScoreEventIndex.from_musicxml(score_path)
    evidence = {**outputs, 'hop':HOP, 'midi_min':52}
    if presence is not None:
        evidence['presence'] = presence
    alignment, events, deletions, info = align_notes(rows, index, score_path, candidate,
        evidence=evidence, config=config, progress=progress)
    info.update(decode_info)
    info['same_pitch_repair'] = repair
    return index, alignment, events, deletions, info
