"""Sequential, resumable GPU campaign; score only after every decode completes."""
import os
import argparse
import subprocess
import sys
import time

from common import ROOT, RUN, read, write, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score-only', action='store_true')
    options = parser.parse_args()
    protocol = read(RUN / 'protocol.json')
    # Fail before spending GPU time if the frozen implementation has changed.
    for relative, digest in protocol['source_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
    for relative, digest in protocol['experiment_code_sha256'].items():
        assert sha(RUN / relative) == digest, relative
    tasks = []
    for action, sizes in [('train', protocol['sizes'][1:]), ('infer', protocol['sizes'])]:
        for size in sizes:
            for model in ['polytune', 'laddersym']:
                tasks.append((f'{action}_{model}_n{size:02}', [str(RUN / 'run_model.py'), action, '--model', model, '--count', str(size)]))
    tasks.append(('score', [str(RUN / 'score_entrypoint.py')]))
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')
    status = {'protocol_sha256': sha(RUN / 'protocol.json'), 'started_unix': time.time(), 'completed': [], 'status': 'running'}
    if options.score_only:
        status = read(RUN / 'campaign_status.json')
        assert status['protocol_sha256'] == sha(RUN / 'protocol.json')
        completed = {row['name'] for row in status['completed']}
        assert all(name in completed for name, _ in tasks if name != 'score')
        status['completed'] = [row for row in status['completed'] if row['name'] != 'score']
        status.update(status='running')
        status.pop('returncode', None)
        tasks = [tasks[-1]]
    logs = RUN / 'campaign_logs'
    logs.mkdir(exist_ok=True)
    for index, (name, args) in enumerate(tasks, 1):
        print(f'START {index}/{len(tasks)} {name}', flush=True)
        status.update(current=name, current_started_unix=time.time())
        write(RUN / 'campaign_status.json', status)
        with (logs / f'{name}.log').open('a', encoding='utf-8') as handle:
            handle.write(f'\nCampaign invocation {time.ctime()}\n')
            handle.flush()
            result = subprocess.run([sys.executable, '-u', *args], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(status='failed', returncode=result.returncode)
            write(RUN / 'campaign_status.json', status)
            raise RuntimeError(f'{name} failed ({result.returncode}); see {logs / (name + ".log")}')
        status['completed'].append({'name': name, 'seconds': time.time() - status['current_started_unix']})
        write(RUN / 'campaign_status.json', status)
        print(f'DONE {name}', flush=True)
    status.update(status='complete', finished_unix=time.time())
    write(RUN / 'campaign_status.json', status)


if __name__ == '__main__':
    main()
