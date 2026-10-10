from dataclasses import replace
from itertools import islice
from types import SimpleNamespace
import numpy as np
import pytest

from alignmodel.joint.index import ScoreEvent, ScoreEventIndex
from alignmodel.joint.passage_v1 import PassageConfig, locate_passages, bounded_hypotheses
from alignmodel.joint.robust_dp_aligner_v2 import TranscribedNote
from alignmodel.joint.robust_dp_aligner_v5 import align_v5, AlignerV5Config
from alignmodel.joint.stack_v9_passage import align_notes, feedback


def score_rows(pitches):
    return tuple(ScoreEvent(i, int(p), float(i), float(i+1), (i,), i//4+1)
                 for i,p in enumerate(pitches))


def heard(pitches):
    return [TranscribedNote(int(p), i*.3, (i+1)*.3, .99) for i,p in enumerate(pitches)]


def test_locator_handles_wrong_missing_extra_and_replay_notes():
    pitches = np.random.default_rng(42).integers(60, 84, 1000)
    passage = list(pitches[440:480])
    passage[12] = 85
    del passage[18]
    passage.insert(25, 86)
    passage[6:6] = passage[:6]
    windows, info = locate_passages(heard(passage), score_rows(pitches))
    assert windows
    best = info['candidates'][0]
    assert best['core_start'] == 440
    assert best['core_end'] == 480
    assert best['similarity'] > .7
    assert len(windows) <= 4
    assert all(b-a <= 512 for a,b in windows)


def test_identical_passages_are_retained_as_separate_locations():
    pitches = np.random.default_rng(3).integers(60, 84, 600)
    pitches[400:432] = pitches[100:132]
    windows, info = locate_passages(heard(pitches[100:132]), score_rows(pitches))
    assert len(windows) >= 2
    assert {c['core_start'] for c in info['candidates'][:2]} == {100, 400}
    assert info['candidates'][0]['similarity'] == info['candidates'][1]['similarity']


def test_no_evidence_and_oversized_search_abstain():
    score = score_rows([60,62,64,65]*100)
    assert not locate_passages([], score)[0]
    assert not locate_passages(heard([90]*24), score)[0]
    windows, info = locate_passages(heard([60]*20), score, PassageConfig(max_locator_cells=100))
    assert not windows and info['status'] == 'search_budget_exceeded'


def test_repeat_search_is_capped_and_lazy():
    score = score_rows(np.random.default_rng(4).integers(60,84,512))
    notes = heard([60,62,64,65]*4)
    proposals = bounded_hypotheses(score, notes, 16, max_hypotheses=32)
    assert iter(proposals) is proposals
    assert len(list(proposals)) <= 32
    assert len(next(bounded_hypotheses(score, notes, 16)).units) == 512


def _write_score(tmp_path, pitches):
    from music21 import stream, note
    path = tmp_path/'score.musicxml'
    stream.Stream([note.Note(int(p), quarterLength=1) for p in pitches]).write('musicxml', fp=path)
    return path, ScoreEventIndex.from_musicxml(path)


def _candidate():
    return {'aligner_costs':{}, 'aligner_v3':{'artifact_ioi':0}, 'gate':{'enabled':False}}


def test_full_score_identity_and_unplayed_notes_are_unassessed(tmp_path):
    pitches = np.random.default_rng(6).integers(60,84,400)
    path,index = _write_score(tmp_path,pitches)
    played = list(pitches[200:240]); played[15] = 90
    alignment, events, deletions, info = align_notes(heard(played), index, path, _candidate(), evidence={})
    doc = feedback(index, alignment, events, deletions, info)
    assert doc['status'] == 'ok'
    wrong = [l for l in doc['labels'] if l['type']=='wrong_note']
    assert len(wrong)==1 and wrong[0]['score_event_indices']==[215]
    assert wrong[0]['note_ids']==['note_0215']
    assert all(200<=i<240 for l in doc['labels'] for i in l['score_event_indices'])
    assert set(range(200)) | set(range(240,400)) <= set(doc['unassessed_score_event_indices'])


def test_ambiguous_locations_use_first_occurrence_and_keep_labels(tmp_path):
    pitches = np.random.default_rng(6).integers(60,84,450)
    pitches[300:340] = pitches[200:240]
    path,index = _write_score(tmp_path,pitches)
    played = list(pitches[200:240])
    played[15] = 90
    alignment, events, deletions, info = align_notes(heard(played), index, path, _candidate(), evidence={})
    doc = feedback(index, alignment, events, deletions, info)
    location = info['passage_location']
    assert location['status'] == 'located'
    assert location['ambiguity_resolved']
    assert location['assessed_start'] == 200
    assert location['start_measure'] == index.events[200].measure
    assert doc['status'] == 'ok'
    wrong = [label for label in doc['labels'] if label['type'] == 'wrong_note']
    assert len(wrong) == 1 and wrong[0]['score_event_indices'] == [215]
    assert wrong[0]['note_ids'] == ['note_0215']
    assert set(range(300, 340)) <= set(doc['unassessed_score_event_indices'])


@pytest.mark.parametrize('costs,similarities,fractions,expected,status', [
    ([.07, .02, 0.], [1., 1., 1.], [1., 1., 1.], 100, 'ok'),
    ([.2, .15, 0.], [1., 1., 1.], [1., 1., 1.], 300, 'ok'),
    ([.01, .01, 0.], [.9, .9, 1.], [1., 1., 1.], 300, 'ok'),
    ([.01, .01, 0.], [1., 1., 1.], [.2, .2, 1.], 300, 'ok'),
    ([.01, .01, 0.], [1., 1., 1.], [.2, .2, .2], 300, 'alignment_uncertain'),
])
def test_passage_preference_preserves_ranking_and_confidence(
        monkeypatch, costs, similarities, fractions, expected, status):
    from alignmodel.joint import stack_v9_passage as pipeline
    from alignmodel.joint.robust_dp_aligner_v5 import AlignmentV5

    notes = heard([60] * 8)
    score = score_rows([60] * 400)
    starts = [100, 200, 300]
    # Retrieval order is deliberately reversed; the earliest near tie is third.
    monkeypatch.setattr(pipeline, 'locate_passages', lambda *args: (
        [(a, a+8) for a in reversed(starts)], {'status': 'located', 'candidates': [
            {'similarity': similarity} for similarity in reversed(similarities)]}))
    monkeypatch.setattr(pipeline, 'score_ornament_patterns', lambda *args: (None,) * 400)

    def fake_align(rows, local_score, *args, **kwargs):
        position = starts.index(int(local_score[0].ql_start))
        return AlignmentV5(events=(), deletions=frozenset(),
            cost=costs[position]*len(notes), source_span=(0, 8), copies=0,
            kept_note_indices=tuple(range(8)), extras=(), missed=(), notes=tuple(notes),
            artifact_notes=frozenset(), seconds_per_ql=.3, match_fraction=fractions[position])

    monkeypatch.setattr(pipeline, 'align_v5', fake_align)
    alignment, events, deletions, info = align_notes(
        notes, SimpleNamespace(events=score), None, _candidate(), evidence={})
    assert info['passage_location']['selected_start'] == expected
    assert alignment.source_span == (expected, expected+8)
    assert info['status'] == status
    assert info['clip_abstained'] == (status != 'ok')


def test_dp_budget_checked_before_allocating(tmp_path):
    path,index = _write_score(tmp_path,[60,62,64,65]*8)
    with pytest.raises(ValueError, match='memory budget'):
        align_v5(heard([60,62]*20),index.events,path,AlignerV5Config(max_dp_cells=10))


def test_bounded_aligner_preserves_two_repetitions(tmp_path):
    pitches = [60,62,64,65,67,69,71,72]
    path,index = _write_score(tmp_path,pitches)
    alignment,events,deletions,info=align_notes(
        heard([60,62,64,62,64,65,67,69,71,69,71,72]),index,path,_candidate(),evidence={})
    doc=feedback(index,alignment,events,deletions,info)
    repeats=[l for l in doc['labels'] if l['type']=='repetition']
    assert {tuple(l['score_event_indices']) for l in repeats}=={(1,2),(5,6)}
