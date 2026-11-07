"""Audit refreshed transcription/alignment provenance for every DataCreate bundle."""
from __future__ import annotations
import csv, hashlib, json, re
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / 'DataCreate/samples'
OUT = ROOT / 'reports/muscriptor_corpus_20261107'

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def xml_info(path: Path):
    root = ET.parse(path).getroot()
    work_title = root.findtext('./work/work-title') or ''
    movement = root.findtext('./movement-title') or ''
    title = work_title or (movement if movement and movement != 'full_score.musicxml' else 'Untitled score')
    parts = []
    for part in root.findall('./part-list/score-part'):
        parts.append({'id': part.get('id'), 'name': part.findtext('part-name') or ''})
    return title, parts

def sample_dirs():
    return sorted((p for p in SAMPLES.iterdir() if p.is_dir()
                   and (p / 'note_alignment_v2.json').is_file()
                   and (p / 'verified_score.musicxml').is_file()), key=lambda p: p.name)

rows = []
for sample in sample_dirs():
    alignment = json.loads((sample / 'note_alignment_v2.json').read_text(encoding='utf-8'))
    metadata = json.loads((sample / 'metadata.json').read_text(encoding='utf-8')) if (sample / 'metadata.json').is_file() else {}
    summary = alignment.get('summary') or {}
    passage = summary.get('passage_location') or {}
    segment = metadata.get('score_segment') or {}
    title, parts = xml_info(sample / 'verified_score.musicxml')
    measures = [e.get('measure') for e in alignment.get('events', [])
                if isinstance(e.get('measure'), (int, float))]
    expected_start, expected_end = segment.get('start_measure'), segment.get('end_measure')
    actual_start, actual_end = (min(measures), max(measures)) if measures else (None, None)
    warnings = []
    if summary.get('status') != 'ok':
        warnings.append('alignment_status:' + str(summary.get('status')))
    if not measures:
        warnings.append('no_aligned_events')
    if expected_start is not None and actual_start is not None and actual_start < expected_start:
        warnings.append('events_before_expected_segment')
    if expected_end is not None and actual_end is not None and actual_end > expected_end:
        warnings.append('events_after_expected_segment')
    if len(parts) != 1:
        warnings.append('score_has_multiple_parts')
    context = (alignment.get('provenance') or {}).get('transcription', {}).get('audio_context') or {}
    rows.append({
        'sample': sample.name,
        'piece': title,
        'score_file': 'verified_score.musicxml',
        'score_parts': '; '.join(f"{p['id']}:{p['name']}" for p in parts),
        'part_count': len(parts),
        'expected_start_measure': expected_start,
        'expected_end_measure': expected_end,
        'aligned_start_measure': actual_start,
        'aligned_end_measure': actual_end,
        'passage_status': passage.get('status'),
        'selected_start_note': passage.get('selected_start'),
        'selected_end_note': passage.get('selected_end'),
        'score_notes': passage.get('score_notes'),
        'heard_notes': passage.get('heard_notes'),
        'plausible_starts': len(passage.get('plausible_starts') or []),
        'alignment_status': summary.get('status'),
        'match_fraction': summary.get('match_fraction'),
        'transcribed_notes': summary.get('transcribed_note_count'),
        'mapped_notes': summary.get('mapped_note_count'),
        'event_count': summary.get('event_count'),
        'audio_context': context.get('mode'),
        'audio_source': context.get('source_audio'),
        'warnings': ';'.join(warnings),
    })

manifest = json.loads((OUT / 'publish/manifest.json').read_text(encoding='utf-8'))
protected_ok = True
protected_failures = []
for sample, hashes in manifest.get('protected', {}).items():
    for name, expected in hashes.items():
        path = SAMPLES / sample / name
        actual = sha(path) if path.is_file() else None
        if actual != expected:
            protected_ok = False
            protected_failures.append({'sample': sample, 'file': name, 'expected': expected, 'actual': actual})
report = {
    'schema_version': 'muscriptor-corpus-alignment-audit-v1',
    'sample_count': len(rows),
    'numbered_sample_count': sum(bool(re.fullmatch(r'\d{3}', r['sample'])) for r in rows),
    'alignment_status_counts': {},
    'piece_counts': {},
    'audio_context_counts': {},
    'warning_samples': [r['sample'] for r in rows if r['warnings']],
    'protected_files_unchanged': protected_ok,
    'protected_file_failures': protected_failures,
    'rows': rows,
}
for row in rows:
    for key, dest in [('alignment_status', report['alignment_status_counts']),
                      ('piece', report['piece_counts']), ('audio_context', report['audio_context_counts'])]:
        value = row[key] or 'unknown'
        dest[value] = dest.get(value, 0) + 1
(OUT / 'alignment_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
fields = list(rows[0])
with (OUT / 'alignment_audit.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
print(json.dumps({k: report[k] for k in ('sample_count', 'numbered_sample_count', 'alignment_status_counts', 'piece_counts', 'audio_context_counts', 'warning_samples', 'protected_files_unchanged')}, indent=2, ensure_ascii=False))
