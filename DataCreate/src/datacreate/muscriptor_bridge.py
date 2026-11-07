"""MuScriptor transcription followed by the existing bounded ALIGN score mapper."""
import json
import os
from pathlib import Path
import subprocess
import uuid

from datacreate.score_transposition import score_transposition

REVISION = 'muscriptor-medium-align-v9-v2'


def written_shift(sample, config):
    metadata_path = sample / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    if metadata.get('effective_audio_transpose') is not None:
        return int(metadata['effective_audio_transpose'])
    if metadata.get('audio_pitch_space') == 'written':
        return 0
    return -score_transposition(sample / 'verified_score.musicxml', config.alignment.get('written_to_sounding'))


def candidate_path(config):
    from datacreate.align_bridge import ROOT, _v9_candidate
    if config.alignment.get('model_version', 'stack-v9') == 'stack-v9':
        return _v9_candidate(config)
    return ROOT / 'align-model/runs/stack-v9/CANDIDATE_STACK_V9.json'


def transcribe(sample, config, logger, output):
    from datacreate.align_bridge import ROOT, _sha256
    input_hashes = {key: _sha256(sample / filename) for key, filename in
                    [('audio_sha256', 'performance_audio.wav'), ('score_sha256', 'verified_score.musicxml')]}
    python = config.resolved_path('muscriptor_python') or ROOT / 'muscriptor/.venv/Scripts/python.exe'
    if not python.is_file():
        raise FileNotFoundError('MuScriptor is not installed. Run muscriptor/setup-local.ps1 or configure paths.muscriptor_python.')
    # Explicit acoustic metadata wins; otherwise use the score's instrument transpose.
    shift = written_shift(sample, config)
    from datacreate.muscriptor_audio import transcription_source, context_signature, file_hash
    source_audio, audio_context = transcription_source(sample)
    logger.info('MuScriptor audio context: %s (%s)', audio_context['mode'], audio_context['reason'])
    command = [str(python), str(ROOT / 'DataCreate/scripts/transcribe_muscriptor.py'),
               '--audio', str(source_audio), '--output', str(output),
               '--written-shift', str(int(shift)), '--device', str(config.alignment.get('note_alignment_device', 'cuda'))]
    if audio_context['crop_end'] is not None:
        command += ['--crop-start', str(audio_context['crop_start']), '--crop-end', str(audio_context['crop_end'])]
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONNOUSERSITE'] = '1'
    logger.info('Transcribing %s with MuScriptor medium', sample.name)
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                                encoding='utf-8', errors='replace',
                                timeout=float(config.alignment.get('note_alignment_timeout_sec', 900)))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('MuScriptor transcription timed out. Try a shorter recording or increase note_alignment_timeout_sec.') from exc
    if result.returncode:
        raise RuntimeError('MuScriptor transcription failed: ' + (result.stderr or result.stdout)[-4000:])
    payload = json.loads(output.read_text(encoding='utf-8'))
    if any(file_hash(sample / filename) != input_hashes[key] for key, filename in
           [('audio_sha256', 'performance_audio.wav'), ('score_sha256', 'verified_score.musicxml')]):
        raise RuntimeError('MuScriptor inputs changed during transcription')
    if any(audio_context[key] != value for key, value in context_signature(sample).items()):
        raise RuntimeError('MuScriptor source recording or trim changed during transcription')
    payload['provenance'].update(input_hashes)
    payload['provenance']['audio_context'] = audio_context
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return output


def current_feedback(sample, config):
    from datacreate.align_bridge import _sha256
    path = sample / 'note_alignment_v2.json'
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding='utf-8'))
    p = payload.get('provenance', {})
    from alignmodel.joint.rhythm_v1 import REVISION as RHYTHM_REVISION
    from datacreate.muscriptor_audio import context_signature
    context = p.get('transcription', {}).get('audio_context', {})
    if any(context.get(key) != value for key, value in context_signature(sample).items()):
        return False
    return (p.get('rhythm_revision') == RHYTHM_REVISION
            and p.get('pipeline_revision') == REVISION and p.get('transcriber') == 'muscriptor'
            and p.get('transcription', {}).get('written_shift') == written_shift(sample, config)
            and all(file.is_file() and p.get(key) == _sha256(file) for key, file in {
                'audio_sha256': sample / 'performance_audio.wav',
                'score_sha256': sample / 'verified_score.musicxml',
                'candidate_sha256': candidate_path(config)}.items()))


def run_alignment(sample, config, logger, detect_candidates):
    import numpy as np
    from datacreate.align_bridge import ROOT, _bridge_command_env, NoteFirstAlignmentResult
    from datacreate.models import Label
    work = config.resolved_path('work_dir') or ROOT / 'DataCreate/work'
    run = work / 'muscriptor-ui' / f'{sample.name}-{uuid.uuid4().hex}'
    run.mkdir(parents=True)
    (sample / 'alignment_progress.json').write_text(json.dumps({'step': 'transcriber'}))
    notes = transcribe(sample, config, logger, run / 'muscriptor_notes.json')
    python, env = _bridge_command_env(config)
    command = [str(python), str(ROOT / 'align-model/scripts/publish_datacreate_v9.py'),
               '--sample', str(sample.resolve()), '--output', str(run / 'publish'),
               '--candidate', str(candidate_path(config)), '--transcription', str(notes.resolve()),
               '--device', str(config.alignment.get('note_alignment_device', 'cuda')),
               '--sample-rate', str(config.sample_rate()), '--hop-length', str(config.mel.get('hop_length', 512))]
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                                timeout=float(config.alignment.get('note_alignment_timeout_sec', 900)))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('MuScriptor score alignment timed out. Try a shorter score excerpt.') from exc
    if result.returncode:
        raise RuntimeError('MuScriptor score alignment failed: ' + (result.stderr or result.stdout)[-4000:])
    if not current_feedback(sample, config):
        raise RuntimeError('MuScriptor alignment returned without matching input provenance')
    payload = json.loads((sample / 'note_alignment_v2.json').read_text(encoding='utf-8'))
    alignment_path = sample / 'alignment.npz'
    with np.load(alignment_path) as archive:
        wp = archive['wp'].copy()
    candidates = [Label(**{**r, 'source': 'auto'}) for r in payload['labels']] if detect_candidates else []
    return NoteFirstAlignmentResult(candidates, alignment_path, wp, wp, 'muscriptor + ALIGN')
