import wave
import numpy as np
import pytest
from datacreate.input_quality import assess_recording, REVISION


def write_wav(path, audio, sr=48000):
    with wave.open(str(path), 'wb') as recording:
        recording.setnchannels(1); recording.setsampwidth(2); recording.setframerate(sr)
        recording.writeframes(np.clip(audio*32768, -32768,32767).astype('<i2').tobytes())
    return path


def noise(seconds=6, sr=48000):
    return np.random.default_rng(567).normal(0,.0085,int(seconds*sr))


def test_stationary_broadband_noise_rejected_with_auditable_metrics(tmp_path):
    result=assess_recording(write_wav(tmp_path/'input.wav',noise(6.4)))
    assert result['status']=='rejected' and result['reason']=='stationary_broadband_noise'
    assert result['revision']==REVISION and result['sample_rate']==48000
    assert result['metrics']['tail_checked'] and result['metrics']['tail_consistent']
    assert result['metrics']['full_blocks']==6
    assert result['metrics']['broadband_flatness_p10']>.95
    assert result['metrics']['rms_pcm16']==pytest.approx(278,abs=2)


@pytest.mark.parametrize('case',['quiet_tone','harmonic','am_noise','silence','buried_tone','brief_tone','fractional_tail_tone','fractional_tail_silence'])
def test_conservative_gate_preserves_tones_and_variable_recordings(tmp_path,case):
    sr=48000
    a=noise(6.4);t=np.arange(len(a))/sr
    if case=='quiet_tone': a=.0003*np.sin(2*np.pi*440*t)
    elif case=='harmonic': a=sum(.005/k*np.sin(2*np.pi*440*k*t) for k in range(1,12))
    elif case=='am_noise': a*=.1+.9*np.sin(2*np.pi*.8*t)**2
    elif case=='silence': a*=0
    elif case=='buried_tone': a+=.0012*np.sin(2*np.pi*440*t) # roughly -20 dB SNR
    elif case=='brief_tone': a[2*sr:3*sr]+=.02*np.sin(2*np.pi*440*t[2*sr:3*sr])
    elif case=='fractional_tail_tone': a[6*sr:]+=.03*np.sin(2*np.pi*440*t[6*sr:])
    elif case=='fractional_tail_silence': a[6*sr:]=0
    result=assess_recording(write_wav(tmp_path/'input.wav',a))
    assert result['status']=='passed'
    assert result['reason']=='noise_pattern_not_detected'


@pytest.mark.parametrize('sr,seconds,reason',[(22050,6,'unsupported_sample_rate'),(48000,4.99,'too_short')])
def test_out_of_scope_is_explicit_not_assessed(tmp_path,sr,seconds,reason):
    result=assess_recording(write_wav(tmp_path/'input.wav',noise(seconds,sr),sr))
    assert result['status']=='not_assessed' and result['reason']==reason
    assert result['metrics']=={}


def test_invalid_pcm_or_truncated_data_cannot_pass(tmp_path):
    path=write_wav(tmp_path/'input.wav',noise())
    raw=path.read_bytes();path.write_bytes(raw[:-100])
    with pytest.raises(ValueError,match='Incomplete'):assess_recording(path)
    with wave.open(str(path),'wb') as recording:
        recording.setnchannels(2);recording.setsampwidth(2);recording.setframerate(48000)
        recording.writeframes(b'\0'*48000*4*6)
    with pytest.raises(ValueError,match='mono PCM16'):assess_recording(path)
