"""v7 plus acoustic same-pitch repair after decoding and before score alignment."""
from .stack_v7 import HOP, feedback as feedback_v7
from .index import ScoreEventIndex
from .robust_dp_aligner_v2 import RobustDPCosts
from .robust_dp_aligner_v3 import GateConfig
from .robust_dp_aligner_v4 import AlignerV4Config, align_v4, gate_v4
from ..transcription.transition_v7 import DecodeV7Config, decode_v7
from ..transcription.same_pitch_v1 import SamePitchConfig, repair_same_pitch


def align_outputs(outputs, score_path, candidate, *, mel, audio=None, presence=None, config=None):
    config = dict(config or {})
    rows, decode_info = decode_v7(outputs, config=DecodeV7Config(**config.get('decoder', candidate['decoder'])))
    rows, repair = repair_same_pitch(rows, outputs, mel, audio=audio,
                                    config=SamePitchConfig(**config.get('same_pitch', {})))
    index = ScoreEventIndex.from_musicxml(score_path)
    alignment = align_v4(rows, index.events, score_path, AlignerV4Config(
        costs=RobustDPCosts(**candidate['aligner_costs']), **candidate['aligner_v3']))
    evidence = {**outputs, 'hop': HOP, 'midi_min': 52}
    if presence is not None:
        evidence['presence'] = presence
    events, deletions, info = gate_v4(alignment, index.events, evidence,
        GateConfig(**(candidate['gate'] | config.get('gate', {}))),
        minimum_match_fraction=config.get('minimum_match_fraction', .45))
    info.update(decode_info)
    info['same_pitch_repair'] = repair
    return index, alignment, events, deletions, info


def feedback(index, alignment, events, deletions, info):
    result = feedback_v7(index, alignment, events, deletions, info)
    result['schema_version'] = 'align-score-feedback-v8'
    for label in result['labels']:
        label['id'] = label['id'].replace('v7_', 'v8_', 1)
    return result
