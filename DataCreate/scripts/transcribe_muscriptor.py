"""Run inside MuScriptor's isolated environment; no DataCreate dependencies."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re


def load_local_model(device):
    # Existing workspace credential is read only in this short-lived process.
    token_path = Path(__file__).resolve().parents[2] / 'hftoken.txt'
    if not os.environ.get('HF_TOKEN') and token_path.is_file():
        match = re.search(r'hf_[A-Za-z0-9]+', token_path.read_text())
        if match:
            os.environ['HF_TOKEN'] = match.group()
    from huggingface_hub import hf_hub_download
    from muscriptor.transcription_model import TranscriptionModel
    weights = Path(hf_hub_download('MuScriptor/muscriptor-medium', 'model.safetensors'))
    dtype = 'float32' if device == 'cpu' else 'float16'
    model = TranscriptionModel.load_model(weights, device=device, dtype=dtype)
    with weights.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return model, dtype, digest


def project_notes(notes, crop_start=0.0, crop_end=None):
    """Clip full-recording intervals and convert to the current WAV's clock.

    These event times are already absolute; never add a five-second chunk offset.
    Keep genuine repeated attacks. Only notes outside the selected audio vanish.
    """
    if not math.isfinite(crop_start) or crop_start < 0 or (
            crop_end is not None and (not math.isfinite(crop_end) or crop_end <= crop_start)):
        raise ValueError('Invalid transcription crop interval')
    seen = set()
    projected = []
    clipped = 0
    for original in notes:
        index = original['index']
        if index in seen:
            raise ValueError(f'Duplicate MuScriptor note-end index: {index}')
        seen.add(index)
        start, end = original['start'], original['end']
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError(f'Invalid MuScriptor note interval: {index}')
        if end <= crop_start or (crop_end is not None and start >= crop_end):
            continue
        lo, hi = max(start, crop_start), min(end, crop_end) if crop_end is not None else end
        row = dict(original, start=lo-crop_start, end=hi-crop_start)
        if lo != start or hi != end:
            clipped += 1
        projected.append(row)
    projected.sort(key=lambda n: (n['start'], n['index']))
    return projected, dict(source_note_count=len(notes), retained_note_count=len(projected),
                           outside_crop_count=len(notes)-len(projected), clipped_note_count=clipped)


def transcribe_file(model, dtype, digest, audio, written_shift, *, crop_start=0.0, crop_end=None):
    from muscriptor.events import NoteEndEvent
    notes = []
    for event in model.transcribe(audio, batch_size=1, prelude_forcing=True):
        if isinstance(event, NoteEndEvent):
            start = event.start_event
            notes.append(dict(index=start.index, pitch=start.pitch + written_shift,
                              sounding_pitch=start.pitch, start=start.start_time,
                              end=event.end_time, instrument=start.instrument, confidence=1.0))
    notes, projection = project_notes(notes, crop_start, crop_end)
    if not notes:
        raise ValueError('MuScriptor detected no notes in the selected recording. Try a clearer recording.')
    return dict(engine='muscriptor-medium', transcribed_notes=notes,
                transcribed_notes_pitch_space='written',
                provenance=dict(transcriber='muscriptor', model='medium', model_sha256=digest,
                    written_shift=written_shift, dtype=dtype,
                    batch_size=1, prelude_forcing=True,
                    time_projection=dict(crop_start=crop_start, crop_end=crop_end, **projection),
                    confidence_policy='MuScriptor emits no note probabilities; confidence=1 is a placeholder'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--written-shift', type=int, required=True)
    parser.add_argument('--crop-start', type=float, default=0.0)
    parser.add_argument('--crop-end', type=float)
    args = parser.parse_args()
    model, dtype, digest = load_local_model(args.device)
    payload = transcribe_file(model, dtype, digest, args.audio, args.written_shift,
                              crop_start=args.crop_start, crop_end=args.crop_end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
