import numpy as np
import torch
from alignmodel.transcription.solo_v1 import ordered_matches,metrics,initialize_solo
from alignmodel.transcription.mel_ctc_v3 import DualMelConfig,DualMelCTCTranscriber,save_dual_checkpoint


def test_repeated_notes_are_not_collapsed_in_identity_metric():
    assert len(ordered_matches([60,62],[60,60,62]))==2
    report=metrics([{'pitch':60,'start':0},{'pitch':62,'start':.3}],
                   [{'pitch':60,'start':0,'end':.1},{'pitch':60,'start':.1,'end':.2},
                    {'pitch':62,'start':.3,'end':.4}])
    assert report['f1']==.8 and report['onset50_matched']==2
    assert report['repeat_matched'] == 0 and report['repeat_target'] == 1
    assert metrics([],[])['f1']==0


def test_widening_moves_written_clarinet_classes_to_sounding_pitch(tmp_path):
    model=DualMelCTCTranscriber(DualMelConfig(temporal_dim=32,conv_channels=16,temporal_blocks=1))
    path=tmp_path/'old.pt';save_dual_checkpoint(path,model,{})
    widened,_=initialize_solo(path)
    for pitch in [52,70,100]:
        before=pitch-model.config.midi_min
        after=pitch-2-widened.config.midi_min
        assert torch.equal(model.pitch_head.weight[before],widened.pitch_head.weight[after])
        assert torch.equal(model.ctc_head[2].weight[before+1],widened.ctc_head[2].weight[after+1])
    assert torch.equal(model.ctc_head[2].weight[0],widened.ctc_head[2].weight[0])
    assert widened.config.midi_min==28 and widened.config.midi_max==108


def test_score_transposition_is_explicit_and_includes_octaves(tmp_path):
    from alignmodel.joint.stack_solo_v1 import score_transposition
    path=tmp_path/'score.musicxml'
    path.write_text('<score-partwise><part><measure><attributes><transpose><chromatic>-2</chromatic><octave-change>-1</octave-change></transpose></attributes></measure></part></score-partwise>')
    assert score_transposition(path)==-14
    assert score_transposition(path,0)==0
    path.write_text('<score-partwise/>')
    assert score_transposition(path)==0


def test_wide_range_frame_and_ctc_losses_support_extremes_and_repeats():
    from alignmodel.transcription.mel_v1 import make_mel_targets,mel_transcriber_loss
    from alignmodel.transcription.mel_ctc_v1 import ctc_loss
    model=DualMelCTCTranscriber(DualMelConfig(midi_min=28,midi_max=108,temporal_dim=32,conv_channels=16,temporal_blocks=1))
    notes=[{'pitch':p,'start':s,'end':e} for p,s,e in [(28,.05,.15),(28,.16,.25),(108,.3,.4)]]
    batch={k:torch.from_numpy(v)[None] for k,v in make_mel_targets(notes,frames=48,hop_sec=256/22050,midi_min=28,midi_max=108).items()}
    batch['frame_mask']=torch.ones(1,48,dtype=torch.bool)
    output=model(torch.randn(1,192,48))
    frame,_=mel_transcriber_loss(output,batch)
    seq=ctc_loss(output['ctc_logits'],batch['frame_mask'],torch.tensor([[1,1,81]]),torch.tensor([3]))
    loss=frame+seq
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.ctc_head[2].weight.grad).all()
