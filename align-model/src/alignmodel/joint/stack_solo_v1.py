"""Opt-in sounding-pitch solo transcriber with the bounded score aligner.

The acoustic model is shared across instruments. Existing error gates remain
experimental outside clarinet; transcription evaluation is reported separately.
"""
from datacreate.score_transposition import score_transposition
from .index import ScoreEventIndex
from .stack_v9_passage import align_notes
from .stack_v7 import feedback as feedback_v7
from ..transcription.transition_v7 import decode_v7, DecodeV7Config

REVISION='solo-multi-v1'


def feedback(index, alignment, events, deletions, info):
    result = feedback_v7(index, alignment, events, deletions, info)
    result['schema_version'] = 'align-score-feedback-solo-v1'
    for label in result['labels']:
        label['id'] = label['id'].replace('v7_', 'solo_', 1)
    return result


def align_outputs(outputs,score_path,candidate,*,midi_min,written_to_sounding=None,progress=None):
    shift=score_transposition(score_path,written_to_sounding)
    written_min=midi_min-shift
    rows,decode_info=decode_v7(outputs,midi_min=written_min,
        config=DecodeV7Config(**candidate.get('decoder',{'blank_scale':.5,'candidate_threshold':.08})))
    if progress:
        progress('locator')
    index=ScoreEventIndex.from_musicxml(score_path)
    for a,b in zip(index.events,index.events[1:]):
        if a.ql_end>b.ql_start+1e-5:
            raise ValueError('Solo transcription requires a monophonic reference score')
    evidence={**outputs,'hop':256/22050,'midi_min':written_min}
    alignment,events,deletions,info=align_notes(rows,index,score_path,candidate,evidence=evidence,progress=progress)
    info.update(decode_info,pipeline_revision=REVISION,written_to_sounding=shift,
                gate_calibration_scope='Inherited clarinet gates; multi-instrument feedback is experimental',
                same_pitch_repair={'schema_version':'solo-ctc-boundaries-v1',
                    'source_groups':[[i] for i in range(len(rows))],'merged_boundaries':0,'pairs_checked':0})
    return index,alignment,events,deletions,info
