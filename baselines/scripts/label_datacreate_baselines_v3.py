"""Current baseline export: policy 3 using a verified stored MuScriptor alignment.

Both PolyTune and LadderSym retain their error types, times, and copy counts.
Only score locations change. Historical v1/v2 Python APIs remain reproducible.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'baselines/common'), str(ROOT/'baselines/scripts'), str(ROOT/'DataCreate/src')]

import label_datacreate_baselines as legacy
from datacreate.melody import padded_melody, parse_sounding_notes
from datacreate.validation import validate_labels_file
from transcription_label_alignment import TimeScoreLookup, VERSION, MIN_SHORTER_INTERVAL_OVERLAP

DEFAULT_ALIGNMENT_POLICY = 'transcription'
CONVERSION_VERSION = 'baseline-datacreate-gui-labels-v3'


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _signature(labels):
    return [(l.get('id'), l['type'], l['start_time'], l['end_time'], l.get('extra_copies')) for l in labels]


def load_lookup(score_path, audio_path, alignment_path, *, sample_id):
    """Check cache identity, then discard every field outside the note allowlist."""
    for path in (score_path, audio_path, alignment_path):
        if not path.is_file():
            raise FileNotFoundError(f'Policy 3 requires {path}. Generate/store the MuScriptor score alignment first.')
    doc = _read(alignment_path)
    provenance = doc.get('provenance') or {}
    if provenance.get('transcriber') != 'muscriptor':
        raise ValueError(f'{alignment_path}: policy 3 requires a MuScriptor alignment, not {provenance.get("transcriber")!r}')
    if str(doc.get('sample_id', sample_id)) != sample_id:
        raise ValueError(f'{alignment_path}: alignment belongs to a different sample')
    for key, path in [('score_sha256', score_path), ('audio_sha256', audio_path)]:
        if provenance.get(key) != legacy.sha256(path):
            raise ValueError(f'{alignment_path}: missing or stale {key}; regenerate the MuScriptor alignment for {path.name}')
    score = parse_sounding_notes(score_path)
    if not score:
        raise ValueError(f'{score_path}: no reference notes')
    raw_notes = doc.get('transcribed_notes')
    if not isinstance(raw_notes, list):
        raise ValueError(f'{alignment_path}: missing transcribed_notes array')
    notes = []
    for ordinal, note in enumerate(raw_notes):
        if note.get('ignored'):
            continue
        a, b = float(note['start']), float(note['end'])
        if not math.isfinite(a) or not math.isfinite(b) or a < 0 or b <= a:
            raise ValueError(f'{alignment_path}: invalid timing at transcript event {ordinal}')
        span = note.get('score_span')
        if span is None:
            indices = []
        else:
            if (not isinstance(span, list) or len(span) != 2
                    or any(type(i) is not int for i in span)
                    or not 0 <= span[0] < span[1] <= len(score)):
                raise ValueError(f'{alignment_path}: invalid score_span at transcript event {ordinal}')
            indices = list(range(*span))
        notes.append(dict(ordinal=ordinal, start=a, end=b, pitch=note['pitch'], score_event_indices=indices))
    sanitized = dict(sample=sample_id, notes=notes,
                     score=[dict(pitch=n.pitch, ql_start=n.ql_start, ql_end=n.ql_end) for n in score])
    metadata = dict(alignment_path=str(alignment_path.resolve()), alignment_sha256=legacy.sha256(alignment_path),
                    audio_sha256=provenance['audio_sha256'], score_sha256=provenance['score_sha256'],
                    transcriber='muscriptor', model=(provenance.get('transcription') or {}).get('model'),
                    transcript_events=len(notes), mapped_transcript_events=sum(bool(n['score_event_indices']) for n in notes),
                    alignment_status=(doc.get('diagnostics') or {}).get('status'),
                    consumed_fields=['transcribed_notes.start', 'transcribed_notes.end', 'transcribed_notes.pitch',
                                     'transcribed_notes.score_span', 'transcribed_notes.ignored', 'reference score rhythm'],
                    error_label_fields_used=False)
    return TimeScoreLookup(sanitized), score, metadata


def apply_lookup(document, lookup, score, metadata):
    """Export exact core identities separately from the GUI context envelope."""
    before = _signature(document['labels'])
    labels, traces = lookup.project(document['labels'])
    for label, trace in zip(labels, traces):
        indices = label['score_event_indices']
        label.update(score_part=None, note_id=None, note_ids=None, core_note_ids=None,
                     pitches=None, measure_number=None, context_score_event_indices=[])
        # The old span's display hints must not survive a different location.
        label['repeats_label_range'] = None
        if indices:
            first, last = indices[0], indices[-1]
            context = padded_melody(score, first, last+1, legacy.PAD_NOTES)
            label.update(context.as_fields())
            label['score_part'].update(core_start_note_index=first, core_end_note_index=last)
            label['context_score_event_indices'] = list(range(context.start_note_index, context.end_note_index+1))
            label['core_note_ids'] = [score[i].note_id for i in indices]
            label['note_id'] = label['core_note_ids'][0]
            label['measure_number'] = score[first].measure
        # Do not fill holes or replace these with padded context indices.
        label['score_event_indices'] = list(indices)
        label['location_evidence'] = trace
        label['comment'] = f"{document['baseline_labeling']['model']} prediction; score location from stored MuScriptor alignment (policy 3)."
    assert _signature(labels) == before
    document['labels'] = labels
    native_metadata = document['baseline_labeling']
    native_metadata.update(
        method=CONVERSION_VERSION, adapter_version=VERSION, alignment_policy=3,
        location_policy=DEFAULT_ALIGNMENT_POLICY,
        adapter_parameters={'min_shorter_interval_overlap': MIN_SHORTER_INTERVAL_OVERLAP,
                            'fallback_to_legacy': False, 'missing_note_time': 'interpolate_forward_score_gaps'},
        score_identity_field='score_event_indices', context_identity_field='context_score_event_indices',
        unlocated_events=sum(not l['score_event_indices'] for l in labels),
        alignment_lookup={**metadata, 'virtual_missing_intervals': len(lookup.virtual),
                          'backward_boundaries': len(lookup.jumps)},
        conversion_diagnostics={**native_metadata.get('conversion_diagnostics', {}),
                                'location_methods': dict(Counter(t['method'] for t in traces))})
    native_metadata.pop('adapter_unlocated_events', None)
    return document


def document_from_midi(midi_path, score_path, *, model, sample_id, provenance=None,
                       alignment_path=None, audio_path=None):
    """Default API for new evaluations and exports. Requires policy 3 inputs."""
    if model not in legacy.LABEL_FILES:
        raise ValueError(f'Unsupported baseline model: {model}')
    score_path = Path(score_path)
    audio_path = Path(audio_path) if audio_path else score_path.parent/'performance_audio.wav'
    alignment_path = Path(alignment_path) if alignment_path else score_path.parent/'note_alignment_v2.json'
    lookup, score, metadata = load_lookup(score_path, audio_path, alignment_path, sample_id=sample_id)
    document = legacy.document_from_midi(Path(midi_path), score_path, model=model, sample_id=sample_id, provenance=provenance)
    return apply_lookup(document, lookup, score, metadata)


def _validate_write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(document, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    try:
        errors = validate_labels_file(tmp)
        if errors:
            raise ValueError('; '.join(errors))
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_sample_document(sample_dir, model, document):
    if model not in legacy.LABEL_FILES:
        raise ValueError(f'Unsupported baseline model: {model}')
    path = Path(sample_dir)/legacy.LABEL_FILES[model]
    _validate_write(path, document)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=sorted(legacy.LABEL_FILES), required=True)
    parser.add_argument('--pred-dir', type=Path, required=True)
    parser.add_argument('--samples', type=Path, default=ROOT/'DataCreate/samples')
    parser.add_argument('--output', type=Path, required=True, help='Run archive and manifest directory')
    parser.add_argument('--alignment-root', type=Path, help='Sample folders containing note_alignment_v2.json; defaults to --samples')
    parser.add_argument('--labels-root', type=Path, help='Export sample folders; defaults to --samples')
    parser.add_argument('--alignment-policy', choices=['transcription', 'legacy-v1', 'legacy-v2'], default=DEFAULT_ALIGNMENT_POLICY,
                        help='Default: transcription (policy 3). Legacy options are for historical reproduction only.')
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--inference-manifest', type=Path)
    args = parser.parse_args(argv)
    sys.addaudithook(legacy.deny_gold)
    pred_root, samples, output = args.pred_dir.resolve(), args.samples.resolve(), args.output.resolve()
    alignment_root = args.alignment_root.resolve() if args.alignment_root else samples
    labels_root = args.labels_root.resolve() if args.labels_root else samples
    marker = output/f'{args.model}_label_manifest.json'
    if marker.exists():
        raise FileExistsError(f'{marker}: choose a new --output directory')
    ids_file = pred_root/'evaluated_ids.json'
    ids = _read(ids_file) if ids_file.exists() else sorted(p.parent.name for p in pred_root.glob('*/mix.mid'))
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise ValueError('Need a nonempty unique list of predicted sample IDs')
    if any(not isinstance(sid, str) or sid in ('.','..') or '/' in sid or '\\' in sid or ':' in sid for sid in ids):
        raise ValueError('Invalid sample IDs')
    provenance: dict[str, Any] = {'prediction_dir': str(pred_root), 'checkpoint_sha256': args.checkpoint_sha256}
    if args.inference_manifest:
        payload = _read(args.inference_manifest)
        provenance.update(inference_manifest=str(args.inference_manifest.resolve()),
                          inference_manifest_sha256=legacy.sha256(args.inference_manifest),
                          checkpoint=payload.get('checkpoint'),
                          checkpoint_sha256=payload.get('checkpoint_sha256') or args.checkpoint_sha256)
    staged = []
    # Prepare and validate every sample before publishing any live labels.
    for sid in ids:
        midi, score = pred_root/sid/'mix.mid', samples/sid/'verified_score.musicxml'
        info = {**provenance, 'prediction_midi_sha256': legacy.sha256(midi), 'score_sha256': legacy.sha256(score)}
        if args.alignment_policy == 'transcription':
            doc = document_from_midi(midi, score, model=args.model, sample_id=sid, provenance=info,
                                     alignment_path=alignment_root/sid/'note_alignment_v2.json')
        else:
            from label_datacreate_baselines_v2 import document_from_midi as v2
            convert = legacy.document_from_midi if args.alignment_policy == 'legacy-v1' else v2
            doc = convert(midi, score, model=args.model, sample_id=sid, provenance=info)
        archive = output/'documents'/args.model/f'{sid}.json'
        _validate_write(archive, doc)
        staged.append((sid, archive, doc))
    rows=[]
    for sid, archive, doc in staged:
        sample = labels_root/sid
        destination = sample/legacy.LABEL_FILES[args.model]
        backup = None
        if destination.exists():
            backup = output/'backup'/sid/destination.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                raise FileExistsError(backup)
            shutil.copy2(destination, backup)
        path = write_sample_document(sample, args.model, doc)
        rows.append(dict(sample=sid,path=str(path),archive=str(archive),backup=str(backup) if backup else None,
                         labels_sha256=legacy.sha256(path),label_count=len(doc['labels']),
                         unlocated_events=doc['baseline_labeling']['unlocated_events']))
    manifest=dict(schema_version=CONVERSION_VERSION,model=args.model,alignment_policy=args.alignment_policy,
                  n_samples=len(rows),rows=rows,human_labels_json_touched=False,
                  source_sha256={name:legacy.sha256(ROOT/name) for name in [
                      'baselines/scripts/label_datacreate_baselines_v3.py','baselines/common/transcription_label_alignment.py']},
                  **provenance)
    marker.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(marker)


if __name__ == '__main__':
    main()
