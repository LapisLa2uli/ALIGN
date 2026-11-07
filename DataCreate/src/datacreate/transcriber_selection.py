"""Per-request transcription choices; never mutate the shared server config."""
from copy import deepcopy
from typing import Literal

Transcriber = Literal['align', 'muscriptor']


def selected_config(config, transcriber=None):
    choice = transcriber or config.alignment.get('transcriber', 'align')
    if choice not in ('align', 'muscriptor'):
        raise ValueError(f'Unknown transcriber: {choice}')
    result = deepcopy(config)
    result.alignment['transcriber'] = choice
    return result


def sample_config(sample, config):
    """Reuse the successfully published choice for automatic refresh/re-label."""
    import json
    # Trimming and passage changes invalidate alignment but retain transcription.
    for filename in ('note_alignment_v2.json', 'transcription_notes.json'):
        path = sample / filename
        if path.is_file():
            provenance = json.loads(path.read_text(encoding='utf-8')).get('provenance', {})
            return selected_config(config, provenance.get('transcriber', 'align'))
    return selected_config(config)
