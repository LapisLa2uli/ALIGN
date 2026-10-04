import numpy as np
import pytest
from alignmodel.transcription.same_pitch_v1 import repair_same_pitch, waveform_levels

HOP = 256/22050


def fixture():
    frames = 180
    mel = np.tile(np.linspace(-.5, .5, 192)[:, None], (1, frames))
    outputs = {'onset': np.zeros(frames), 'rearticulation': np.zeros(frames),
               'voiced': np.ones(frames), 'rms_db': np.full(frames, -20.)}
    rows = [[70, .2, .7, .8, 0, 71, .1], [70, .7, 1.2, .9, 0, 69, .1],
            [70, 1.2, 1.6, .7, 0, -1, 0]]
    return rows, outputs, mel


def test_continuous_chain_and_lineage_without_mutating_input():
    rows, outputs, mel = fixture()
    result, audit = repair_same_pitch(rows, outputs, mel)
    assert len(result) == 1 and result[0][:4] == [70, .2, 1.6, .9]
    assert result[0][5:] == [-1, 0.]
    assert audit['source_groups'] == [[0, 1, 2]] and audit['pairs_checked'] == 2
    assert rows[0][2] == .7


@pytest.mark.parametrize('kind', ['onset', 'rearticulation', 'energy', 'silence', 'spectral', 'ambiguous'])
def test_real_repetition_or_ambiguous_evidence_retains_boundary(kind):
    rows, outputs, mel = fixture(); frame = round(.7/HOP)
    if kind in ('onset', 'rearticulation'):
        outputs[kind][frame-3] = .9  # attack can precede CTC emission
    elif kind == 'energy': outputs['rms_db'][frame-1] -= 8
    elif kind == 'silence': outputs['rms_db'][frame-1] = -90
    elif kind == 'spectral': mel[170:, frame:] += .3
    else: outputs['onset'][frame] = .27
    result, audit = repair_same_pitch(rows, outputs, mel)
    assert audit['decisions'][0]['decision'] == 'retain'
    assert result[0][2] == .7


def test_different_pitches_and_original_boundaries():
    rows, outputs, mel = fixture(); rows[1][0] = 71
    assert repair_same_pitch(rows, outputs, mel)[1]['pairs_checked'] == 0
    rows[1][0] = 70; rows[1][1] = .25
    result, audit = repair_same_pitch(rows, outputs, mel)
    assert audit['decisions'][0]['reason'] == 'insufficient_context'


@pytest.mark.parametrize('damage', ['missing', 'nan', 'edge', 'wrong_mel'])
def test_incomplete_evidence_abstains(damage):
    rows, outputs, mel = fixture()
    if damage == 'missing': outputs.pop('voiced')
    if damage == 'nan': mel[:, round(.7/HOP)] = np.nan
    if damage == 'edge': mel = mel[:, :65]
    if damage == 'wrong_mel': mel = mel[:128]
    assert repair_same_pitch(rows, outputs, mel)[1]['decisions'][0]['decision'] == 'retain'


def test_waveform_tone_and_brief_silence():
    rows, outputs, mel = fixture()
    time = np.arange(round(180*HOP*22050))/22050
    audio = .2*np.sin(2*np.pi*440*time)
    assert repair_same_pitch(rows, outputs, mel, audio=audio)[1]['merged_boundaries'] == 2
    audio[(time>.68)&(time<.71)] = 0
    audit = repair_same_pitch(rows, outputs, mel, audio=audio)[1]
    assert audit['decisions'][0]['decision'] == 'retain'
    assert audit['energy_source'] == 'waveform_12ms'
    assert np.isnan(waveform_levels([], 2)).all()


def test_malformed_and_empty_inputs():
    assert repair_same_pitch([], {}, None)[0] == []
    with pytest.raises(ValueError): repair_same_pitch([[70, .5, .2, 1]], {}, None)
    with pytest.raises(ValueError): repair_same_pitch([[70, .5, .6, 1], [70, .1, .6, 1]], {}, None)


def test_stack_repairs_before_alignment_and_keeps_reference_identity(tmp_path, monkeypatch):
    from music21 import stream, note
    from alignmodel.joint import stack_v8
    from alignmodel.joint.robust_dp_aligner_v2 import RobustDPCosts
    from dataclasses import asdict
    rows, outputs, mel = fixture()
    # Two CTC emissions for the first reference note, followed by another pitch.
    rows[2][0] = 72
    path = tmp_path/'score.musicxml'
    stream.Stream([note.Note(70), note.Note(72)]).write('musicxml', fp=path)
    monkeypatch.setattr(stack_v8, 'decode_v7', lambda *a, **kw: (rows, {'transition_candidates':0}))
    candidate = {'decoder':{}, 'aligner_costs':asdict(RobustDPCosts()),
                 'aligner_v3':{}, 'gate':{'enabled':False}}
    index, alignment, events, deletions, info = stack_v8.align_outputs(outputs, path, candidate, mel=mel)
    document = stack_v8.feedback(index, alignment, events, deletions, info)
    assert info['same_pitch_repair']['merged_boundaries'] == 1
    assert [e.score_span for e in events] == [(0,1), (1,2)]
    assert not deletions and not document['labels']
    assert document['schema_version'] == 'align-score-feedback-v8'
