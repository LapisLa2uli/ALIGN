"""User-facing milestones based on observed pipeline events, not elapsed time."""

PHASES = (
    ('prepare', 'Prepare', (('score', 'Read and prepare score'),
                            ('reference', 'Render reference music'), ('audio', 'Prepare recording'))),
    ('analyze', 'Analyze', (('transcriber', 'Transcribe performed notes'),
                           ('locator', 'Find the score passage'), ('aligner', 'Align notes to score'),
                           ('labels', 'Identify possible issues'), ('features', 'Build audio features'))),
    ('narrate', 'Write feedback', (('prepare_feedback', 'Prepare teaching points'),
                                  ('write', 'Write advice for each point'), ('examples', 'Synthesize music examples'))),
    ('speech', 'Create speech', (('synthesize', 'Generate voice segments'),
                                ('mix', 'Match pacing, loudness and mix audio'))),
    ('video', 'Animate score', (('engrave', 'Draw reference and played notation'),
                               ('frames', 'Render synchronized frames'), ('finalize', 'Encode and save MP4'))),
)


def detailed_progress(state, analysis):
    stage = state.get('stage')
    if stage == 'input_quality':
        status = 'failed' if state.get('status') == 'failed' else 'active'
        return {'phases': [{'id': 'input_quality', 'label': 'Check recording', 'state': status,
                            'substeps': [{'id': 'input_quality', 'label': 'Check original audio quality',
                                         'state': status}]}],
                'completed': 0, 'total': 1, 'current': 'input_quality', 'units': None,
                'message': state.get('message') or 'Checking recording quality',
                'status': state.get('status')}
    detail = state.get('progress_detail') or {}
    step = (analysis or {}).get('current')
    if stage in {'queued', 'score', 'reference', 'audio'}:
        current = 'score' if stage == 'queued' else stage
    elif stage == 'alignment':
        current = step or 'transcriber'
    elif stage == 'features':
        current = 'features'
    elif stage == 'feedback':
        default = {'labels': 'prepare_feedback', 'narration': 'write', 'speech': 'synthesize'}
        current = detail.get('substep') or default.get(step, 'prepare_feedback')
    elif stage == 'video':
        current = detail.get('substep') or 'engrave'
    else:
        current = None
    keys = [key for _, _, steps in PHASES for key, _ in steps]
    finished = state.get('status') == 'complete'
    index = len(keys) if finished else keys.index(current) if current in keys else 0
    unavailable = state.get('video_status') == 'unavailable'
    skipped = set(state.get('progress_skipped') or [])
    if finished and unavailable:
        skipped.update(('engrave', 'frames', 'finalize'))
    phases, offset = [], 0
    for key, label, steps in PHASES:
        rows = []
        for position, (substep, title) in enumerate(steps, offset):
            status = ('skipped' if substep in skipped else 'complete' if position < index
                      else ('failed' if state.get('status') == 'failed' else 'active')
                      if position == index else 'pending')
            rows.append({'id': substep, 'label': title, 'state': status})
        statuses = {r['state'] for r in rows}
        status = next((s for s in ('failed', 'active', 'pending') if s in statuses),
                      'skipped' if statuses == {'skipped'} else 'complete')
        phases.append({'id': key, 'label': label, 'state': status, 'substeps': rows})
        offset += len(steps)
    completed = sum(r['state'] in {'complete', 'skipped'} for p in phases for r in p['substeps'])
    active = next((r for p in phases for r in p['substeps'] if r['id'] == current), None)
    message = 'Your feedback is ready' if finished else (active or {}).get('label', 'Preparing your take')
    if not finished and detail.get('substep') == current:
        message = detail.get('message') or message
    if state.get('status') == 'failed':
        message = state.get('message') or message
    units = None
    if not finished and detail.get('substep') == current and detail.get('total', 0) > 0:
        units = {'completed': detail.get('completed', 0), 'total': detail['total']}
    return {'phases': phases, 'completed': completed, 'total': len(keys),
            'current': current if not finished else None, 'message': message,
            'units': units, 'status': state.get('status')}
