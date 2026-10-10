"""V10 keeps v9 acoustics/passage alignment, adds learned same-pitch protection."""
from .index import ScoreEventIndex
from .stack_v9_passage import align_notes
from .stack_v9_passage import align_outputs as v9_align_outputs
from .stack_v7 import HOP, feedback as feedback_v7
from ..transcription.transition_v7 import DecodeV7Config, decode_v7
from ..transcription.same_pitch_v2 import SamePitchConfig
from ..transcription.same_pitch_v3 import BoundaryConfig, repair_same_pitch

REVISION = 'v10-boundary-v1'


def supported_regions(rows, index, alignment, repair):
    """Only existing CTC boundaries mapped across distinct same-pitch notes."""
    supported, regions = set(), {}
    for event in alignment.events:
        if event.score_span is None or event.rendered_index is None:
            continue
        row = alignment.kept_note_indices[event.rendered_index]
        group = repair['source_groups'][row]
        a, b = event.score_span
        if len(group) < 2 or b-a != len(group):
            continue
        if any(e.pitch != rows[group[0]][0] for e in index.events[a:b]):
            continue
        if any(rows[i][3] < .5 or (len(rows[i]) > 4 and rows[i][4]) for i in group):
            continue
        for i in group[1:]:
            supported.add(i)
            regions[i] = (a, b)
    return supported, regions


def outside_signature(alignment, repair, changed_sources):
    result = []
    for e in alignment.events:
        row = alignment.kept_note_indices[e.rendered_index]
        sources = tuple(repair['source_groups'][row])
        if not changed_sources.intersection(sources):
            result.append((sources, e.pitch, e.start, e.end, e.score_span, e.relationship, e.copy_pass, e.origin_relationship))
    return result


def label_signature(doc, changed_score):
    return sorted((r['type'], tuple(r['score_event_indices']), r.get('extra_copies', 0))
                  for r in doc['labels'] if not changed_score.intersection(r['score_event_indices']))


def align_outputs(outputs, score_path, candidate, *, mel, boundary_model, audio=None,
                  presence=None, config=None, progress=None, baseline=None):
    config = dict(config or {})
    baseline = baseline or v9_align_outputs(outputs, score_path, candidate, mel=mel, audio=audio,
                                           presence=presence, config=config, progress=progress)
    index, old_alignment, old_events, old_deletions, old_info = baseline
    rows, decode_info = decode_v7(outputs, config=DecodeV7Config(**config.get('decoder', candidate['decoder'])))
    supported, regions = supported_regions(rows, index, old_alignment, old_info['same_pitch_repair'])
    if old_info['status'] != 'ok':
        supported = set()
    repaired, repair = repair_same_pitch(rows, outputs, mel, model=boundary_model,
        config=BoundaryConfig(**config.get('boundary', {})), v9_config=SamePitchConfig(**config.get('same_pitch', {})),
        supported_boundaries=supported, baseline_audit=old_info['same_pitch_repair'])
    restored = [d['source_indices'][1] for d in repair['decisions']
                if d['reason'] == 'score_supported_rearticulation']
    audit = {'supported_boundaries': sorted(supported), 'proposed_restorations': restored,
             'accepted_restorations': [], 'outside_region_preserved': True}
    if not restored:
        return index, old_alignment, old_events, old_deletions, {
            **old_info, 'pipeline_revision': REVISION, 'same_pitch_repair': repair, 'boundary_refinement': audit}
    changed_score = {i for at in restored for i in range(*regions[at])}
    changed_sources = {i for group in old_info['same_pitch_repair']['source_groups']
                       if any(at in group for at in restored) for i in group}
    evidence = {**outputs, 'hop': HOP, 'midi_min': 52}
    if presence is not None:
        evidence['presence'] = presence
    alignment, events, deletions, info = align_notes(repaired, index, score_path, candidate,
        evidence=evidence, config=config, progress=progress)
    info.update(decode_info, same_pitch_repair=repair, pipeline_revision=REVISION)
    original = feedback_v7(index, old_alignment, old_events, old_deletions, old_info)
    updated = feedback_v7(index, alignment, events, deletions, info)
    preserved = (info['status'] == 'ok'
        and outside_signature(old_alignment, old_info['same_pitch_repair'], changed_sources)
            == outside_signature(alignment, repair, changed_sources)
        and old_deletions-changed_score == deletions-changed_score
        and label_signature(original, changed_score) == label_signature(updated, changed_score))
    # Restored sources must actually map to separate score notes, not become extras.
    mapping = {}
    for e in alignment.events:
        for source in repair['source_groups'][alignment.kept_note_indices[e.rendered_index]]:
            mapping[source] = e.score_span
    preserved = preserved and all(mapping.get(at-1) and mapping.get(at)
        and mapping[at-1][1] == mapping[at][0]
        and regions[at][0] <= mapping[at-1][0] < mapping[at][1] <= regions[at][1] for at in restored)
    if not preserved:
        audit.update(rejected_reason='realignment_changed_protected_output_or_failed_boundary',
                     proposed_audit=repair)
        return index, old_alignment, old_events, old_deletions, {
            **old_info, 'pipeline_revision': REVISION, 'boundary_refinement': audit}
    audit['accepted_restorations'] = restored
    info['boundary_refinement'] = audit
    return index, alignment, events, deletions, info


def feedback(index, alignment, events, deletions, info):
    result = feedback_v7(index, alignment, events, deletions, info)
    result['schema_version'] = 'align-score-feedback-v10'
    for label in result['labels']:
        label['id'] = label['id'].replace('v7_', 'v10_', 1)
    return result
