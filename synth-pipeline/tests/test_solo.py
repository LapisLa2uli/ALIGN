from synthpipeline.config import SynthConfig
from synthpipeline.solo import INSTRUMENTS,create_notes,split_for,write_exact_midi


def test_solo_notes_are_in_range_monophonic_and_deterministic(tmp_path):
    import mido
    from datacreate.melody import parse_sounding_notes
    from synthpipeline.scoregen import write_musicxml
    cfg=SynthConfig(generation={'measures_min':2,'measures_max':2})
    for inst in [INSTRUMENTS[1],INSTRUMENTS[4],INSTRUMENTS[18]]:
        score,rows,_=create_notes(cfg,inst,1234)
        assert rows==create_notes(cfg,inst,1234)[1]
        assert all(inst.low <= r['pitch'] <= inst.high for r in rows)
        assert all(a['end']<=b['start'] for a,b in zip(rows,rows[1:]))
        path=tmp_path/f'{inst.name}.musicxml';write_musicxml(score,path)
        assert [n.pitch for n in parse_sounding_notes(path)]==[r['pitch'] for r in rows]
        path=tmp_path/f'{inst.name}.mid';write_exact_midi(path,rows,inst.program)
        midi=mido.MidiFile(path)
        assert [m.note for m in midi if m.type=='note_on' and m.velocity]==[r['pitch'] for r in rows]


def test_grouped_split_is_balanced():
    roles=[split_for(i,500,20261017) for i in range(500)]
    assert {k:roles.count(k) for k in set(roles)}=={'train':400,'val':50,'test':50}


def test_missing_soundfont_key_region_uses_audible_alternative():
    import pytest
    from synthpipeline.solo import choose_supported_bank
    banks=[('primary','p.sf2'),('fallback','f.sf2')]
    coverage={'primary':{'bass':[60,61]},'fallback':{'bass':[]}}
    assert choose_supported_bank([{'pitch':60}], 'bass', banks[0], banks, coverage)==banks[1]
    assert choose_supported_bank([{'pitch':40}], 'bass', banks[0], banks, coverage)==banks[0]
    coverage['fallback']['bass']=[60]
    with pytest.raises(ValueError,match='No SoundFont'):
        choose_supported_bank([{'pitch':60}], 'bass', banks[0], banks, coverage)


def test_chromatic_supplement_guarantees_every_pitch_including_rare_classes():
    from synthpipeline.solo import SoloInstrument
    cfg=SynthConfig(dataset={'mode':'chromatic_coverage','seed':30000})
    inst=SoloInstrument('test',73,101,108)
    pitches=set()
    for index in range(16):
        _,rows,style=create_notes(cfg,inst,30000+index*1009)
        assert style=='coverage'
        assert rows[0]['pitch']==101+index//2
        assert all(a['end']<=b['start'] for a,b in zip(rows,rows[1:]))
        pitches.update(n['pitch'] for n in rows)
    assert pitches==set(range(101,109))
