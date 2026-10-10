"""Archive active user labels, then copy exact agent files (--apply required).

Only live corpus and Studio samples are changed. Model provenance stays intact.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from datacreate.validation import validate_labels_file


def existing(path):
    return path.read_bytes() if path.exists() else None


def sha(data):
    return hashlib.sha256(data).hexdigest() if data is not None else None


def atomic_write(path, data):
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    roots = [p for p in (ROOT/'samples', ROOT/'work/studio/samples') if p.is_dir()]
    listing = subprocess.run(['rg', '--files', '--hidden', '--no-ignore', '-g', 'labels.json',
        '-g', 'labels_agent.json', *map(str, roots)], capture_output=True, text=True)
    if listing.returncode not in (0, 1) or listing.stderr:
        raise RuntimeError(f'Incomplete inventory: {listing.stderr}')
    folders = sorted({Path(line).parent for line in listing.stdout.splitlines()})
    rows, snapshots = [], {}
    for folder in folders:
        if not any(folder.resolve().is_relative_to(p.resolve()) for p in roots):
            raise ValueError(f'Path escaped active roots: {folder}')
        user, agent = folder/'labels.json', folder/'labels_agent.json'
        before, source = existing(user), existing(agent)
        if source is not None and (errors := validate_labels_file(agent)):
            raise ValueError(f'{agent}: {errors}')
        old = json.loads(before.decode('utf-8-sig')) if before is not None else {}
        new = json.loads(source.decode('utf-8-sig')) if source is not None else {}
        relative = folder.relative_to(ROOT).as_posix()
        rows.append({'sample': relative, 'user_file': f'{relative}/labels.json',
            'agent_file': f'{relative}/labels_agent.json', 'user_before_sha256': sha(before),
            'agent_sha256': sha(source), 'user_labels_before': len(old.get('labels', [])),
            'agent_labels': len(new.get('labels', [])), 'agent_annotator': new.get('annotator_id'),
            'action': 'copy_agent_to_user' if source is not None else 'leave_unchanged_no_agent'})
        snapshots[relative] = before, source
    summary = {'sample_directories': len(rows),
        'user_files_archived': sum(r['user_before_sha256'] is not None for r in rows),
        'user_files_seeded': sum(r['agent_sha256'] is not None for r in rows),
        'user_labels_before': sum(r['user_labels_before'] for r in rows),
        'agent_labels_copied': sum(r['agent_labels'] for r in rows),
        'missing_agent': [r['sample'] for r in rows if r['agent_sha256'] is None],
        'agent_annotators': dict(Counter(r['agent_annotator'] for r in rows if r['agent_sha256']))}
    if not args.apply:
        print(json.dumps(summary, indent=2))
        return
    now = datetime.now(timezone.utc)
    archive = ROOT/'archives'/f'user-labels-before-agent-seed-{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}'
    archive.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'archiving', 'created_utc': now.isoformat(), 'datacreate_root': str(ROOT),
        'scope': [str(p.relative_to(ROOT)) for p in roots], 'summary': summary, 'records': rows,
        'provenance_policy': 'Exact agent bytes; not marked human-reviewed. Pending manual filtering.'}
    def save():
        atomic_write(archive/'manifest.json', (json.dumps(manifest, indent=2)+'\n').encode('utf-8'))
    save()
    for name, key, position in [('user_labels_before.zip', 'user_file', 0), ('agent_labels_snapshot.zip', 'agent_file', 1)]:
        with ZipFile(archive/name, 'x', ZIP_DEFLATED) as z:
            for row in rows:
                data = snapshots[row['sample']][position]
                if data is not None:
                    z.writestr(row[key], data)
        with ZipFile(archive/name) as z:
            assert z.testzip() is None
            for row in rows:
                data = snapshots[row['sample']][position]
                if data is not None:
                    assert z.read(row[key]) == data
    manifest['archive_sha256'] = {p.name: sha(p.read_bytes()) for p in archive.glob('*.zip')}
    manifest['status'] = 'archived_and_verified'
    save()
    for row in rows:
        if sha(existing(ROOT/row['user_file'])) != row['user_before_sha256'] or sha(existing(ROOT/row['agent_file'])) != row['agent_sha256']:
            raise RuntimeError(f'Concurrent edit before copying: {row["sample"]}; archive preserved')
    completed = []
    try:
        manifest['status'] = 'copying'
        save()
        for row in rows:
            if row['agent_sha256'] is None:
                continue
            before, source = snapshots[row['sample']]
            if existing(ROOT/row['user_file']) != before or existing(ROOT/row['agent_file']) != source:
                raise RuntimeError(f'Concurrent edit: {row["sample"]}')
            atomic_write(ROOT/row['user_file'], source)
            completed.append(row)
        for row in rows:
            expected = row['agent_sha256'] if row['agent_sha256'] is not None else row['user_before_sha256']
            assert sha(existing(ROOT/row['user_file'])) == expected, row['sample']
            assert sha(existing(ROOT/row['agent_file'])) == row['agent_sha256'], row['sample']
        manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                        verified_user_files=len(rows), verified_agent_files=summary['user_files_seeded'])
    except BaseException as error:
        conflicts = []
        for row in reversed(completed):
            target = ROOT/row['user_file']
            if sha(existing(target)) != row['agent_sha256']:
                conflicts.append(row['sample'])
                continue
            before = snapshots[row['sample']][0]
            if before is not None:
                atomic_write(target, before)
            else:
                target.unlink()
        manifest.update(status='failed_rolled_back' if not conflicts else 'failed_concurrent_edits',
                        error=str(error), rollback_conflicts=conflicts)
        save()
        raise
    save()
    (archive/'README.md').write_text(
        '# User-label archive\n\n'
        'All prior active user-label files are in `user_labels_before.zip`, byte-for-byte. '
        'Entry paths are relative to DataCreate. `agent_labels_snapshot.zip` stores copied sources. '
        '`manifest.json` records all paths and SHA-256 hashes, including the missing-agent sample.\n\n'
        'To restore, close annotation editors, back up any newer manual edits, and extract '
        '`user_labels_before.zip` to DataCreate with its relative paths. A manifest row with '
        'user_before_sha256=null means no original user file existed.\n\n'
        'Seeded files retain agent provenance and are pending human review. '
        'This operation makes no claim about model accuracy or false-negative rates.\n', encoding='utf-8')
    print(json.dumps({'archive': str(archive), 'status': manifest['status'], **summary}, indent=2))


if __name__ == '__main__':
    main()
