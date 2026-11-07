"""Production policy-3 integration and input/cache safety checks."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pretty_midi
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'baselines/scripts'), str(ROOT/'baselines/common'), str(ROOT/'DataCreate/src')]
import label_datacreate_baselines_v3 as current
import label_datacreate_baselines as legacy
from datacreate.melody import canonical_note_location
from transcription_label_alignment import TimeScoreLookup, VERSION


@pytest.fixture
def sample(tmp_path):
    import music21
    folder = tmp_path/'samples/001'
    folder.mkdir(parents=True)
    score = music21.stream.Score()
    part = music21.stream.Part()
    for pitch in (60,62,64,65,67):
        part.append(music21.note.Note(pitch,quarterLength=1))
    score.append(part)
    score_path = folder/'verified_score.musicxml'
    score.write('musicxml',fp=score_path)
    audio = folder/'performance_audio.wav'
    audio.write_bytes(b'cache-identity-test-audio')
    alignment = dict(sample_id='001',provenance=dict(transcriber='muscriptor',
        score_sha256=legacy.sha256(score_path),audio_sha256=legacy.sha256(audio)),
        transcribed_notes=[dict(start=1.,end=2.,pitch=70,score_span=[3,4]),
                           dict(start=3.,end=4.,pitch=67,score_span=[4,5])],
        labels=[{'type':'wrong_note','score_event_indices':[0]}],diagnostics={'status':'ok'})
    (folder/'note_alignment_v2.json').write_text(json.dumps(alignment),encoding='utf-8')
    midi_path=tmp_path/'predictions/001/mix.mid'
    midi_path.parent.mkdir(parents=True)
    midi=pretty_midi.PrettyMIDI()
    for name,pitch in [('extra',68),('missing',58)]:
        track=pretty_midi.Instrument(program=71,name=name)
        track.notes.append(pretty_midi.Note(pitch=pitch,velocity=90,start=1.1,end=1.8))
        midi.instruments.append(track)
    midi.write(str(midi_path))
    return folder,midi_path


@pytest.mark.parametrize('model',['polytune','laddersym'])
def test_default_converter_uses_transcript_and_preserves_predictions(sample,model):
    folder,midi=sample
    old=legacy.document_from_midi(midi,folder/'verified_score.musicxml',model=model,sample_id='001')
    doc=current.document_from_midi(midi,folder/'verified_score.musicxml',model=model,sample_id='001')
    assert current._signature(doc['labels'])==current._signature(old['labels'])
    assert doc['baseline_labeling']['alignment_policy']==3
    label=doc['labels'][0]
    assert label['score_event_indices']==[3]
    assert canonical_note_location(label)==('score_events',(3,),0)
    assert label['context_score_event_indices']==[1,2,3,4]
    assert label['core_note_ids']==['note_0003']
    assert doc['baseline_labeling']['alignment_lookup']['error_label_fields_used'] is False
    # Changing teacher errors cannot change projection.
    path=folder/'note_alignment_v2.json';alignment=current._read(path)
    alignment['labels']=[{'type':'repetition','score_event_indices':[4]}]
    alignment['transcribed_notes'][0]['relationship']='copy'
    path.write_text(json.dumps(alignment),encoding='utf-8')
    changed=current.document_from_midi(midi,folder/'verified_score.musicxml',model=model,sample_id='001')
    assert changed['labels']==doc['labels']


@pytest.mark.parametrize('fault',['missing','audio','score','transcriber','span'])
def test_missing_or_stale_lookup_never_falls_back(sample,fault):
    folder,midi=sample;path=folder/'note_alignment_v2.json';d=current._read(path)
    if fault=='missing':path.unlink()
    elif fault=='audio':(folder/'performance_audio.wav').write_bytes(b'changed')
    elif fault=='score':d['provenance']['score_sha256']='stale'
    elif fault=='transcriber':d['provenance']['transcriber']='align'
    else:d['transcribed_notes'][0]['score_span']=[0,999]
    if fault not in ('missing','audio'):path.write_text(json.dumps(d),encoding='utf-8')
    with pytest.raises((ValueError,FileNotFoundError)):
        current.document_from_midi(midi,folder/'verified_score.musicxml',model='polytune',sample_id='001')


def test_noncontiguous_repetition_core_is_not_filled_by_gui_context(sample):
    folder,midi=sample
    doc=legacy.document_from_midi(midi,folder/'verified_score.musicxml',model='polytune',sample_id='001')
    doc['labels'][0].update(type='repetition',start_time=1,end_time=4,extra_copies=2)
    lookup,score,metadata=current.load_lookup(folder/'verified_score.musicxml',folder/'performance_audio.wav',folder/'note_alignment_v2.json',sample_id='001')
    lookup.notes[0]['score_event_indices']=[0]
    lookup.notes[1]['score_event_indices']=[2]
    out=current.apply_lookup(doc,lookup,score,metadata)['labels'][0]
    assert out['score_event_indices']==[0,2]
    assert out['core_note_ids']==['note_0000','note_0002']
    assert canonical_note_location(out)==('score_events_with_copies',(0,2),2)


def test_writer_validates_before_replacing_existing_baseline(sample):
    folder,midi=sample
    human=folder/'labels.json';human.write_text('human file unchanged',encoding='utf-8')
    path=folder/'labels_polytune.json';path.write_text('previous baseline',encoding='utf-8')
    doc=current.document_from_midi(midi,folder/'verified_score.musicxml',model='polytune',sample_id='001')
    doc['labels'][0]['end_time']=doc['labels'][0]['start_time']
    with pytest.raises(ValueError):current.write_sample_document(folder,'polytune',doc)
    assert path.read_text()=='previous baseline'
    assert human.read_text()=='human file unchanged'


def test_established_cli_defaults_to_policy3_and_backs_up(sample,tmp_path):
    folder,midi=sample
    dest=folder/'labels_polytune.json';dest.write_text('previous baseline',encoding='utf-8')
    output=tmp_path/'run'
    command=[sys.executable,str(ROOT/'baselines/scripts/label_datacreate_baselines.py'),
             '--model','polytune','--pred-dir',str(midi.parents[1]),'--samples',str(folder.parent),'--output',str(output)]
    result=subprocess.run(command,env={**os.environ,'PYTHONNOUSERSITE':'1'},capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert current._read(dest)['baseline_labeling']['adapter_version']==VERSION
    assert (output/'backup/001/labels_polytune.json').read_text()=='previous baseline'
    assert current._read(output/'polytune_label_manifest.json')['alignment_policy']=='transcription'


def test_batch_missing_alignment_does_not_partially_publish(sample,tmp_path):
    import shutil
    folder,midi=sample
    other=folder.parent/'002';shutil.copytree(folder,other)
    (other/'note_alignment_v2.json').unlink()
    second=midi.parents[1]/'002/mix.mid';second.parent.mkdir();shutil.copy2(midi,second)
    dest=folder/'labels_laddersym.json';dest.write_text('keep',encoding='utf-8')
    result=subprocess.run([sys.executable,str(ROOT/'baselines/scripts/label_datacreate_baselines.py'),
        '--model','laddersym','--pred-dir',str(midi.parents[1]),'--samples',str(folder.parent),'--output',str(tmp_path/'run')],
        env={**os.environ,'PYTHONNOUSERSITE':'1'},capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Policy 3 requires' in result.stderr
    assert dest.read_text()=='keep'
    assert not (other/'labels_laddersym.json').exists()


def test_core_lookup_gap_repeat_and_silence():
    doc=dict(score=[dict(pitch=60,ql_start=i,ql_end=i+1) for i in range(4)],notes=[
        dict(start=1,end=2,score_event_indices=[0],ordinal=0,pitch=60),
        dict(start=3,end=4,score_event_indices=[2],ordinal=1,pitch=60),
        dict(start=5,end=6,score_event_indices=[0],ordinal=2,pitch=60)])
    labels=[dict(type=k,start_time=a,end_time=b,score_event_indices=[3]) for k,a,b in [
        ('missed_note',2,3),('wrong_note',5,6),('wrong_note',0,.5),('extra_note',4.1,4.9)]]
    out,_=TimeScoreLookup(doc).project(labels)
    assert [l['score_event_indices'] for l in out]==[[1],[0],[],[]]
