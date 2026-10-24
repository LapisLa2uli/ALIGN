import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


def test_real_gold_uses_verified_human_archive_and_excludes_agent_and_095(tmp_path, monkeypatch):
    script=Path(__file__).resolve().parents[1]/'scripts/evaluate_solo_datacreate.py'
    spec=importlib.util.spec_from_file_location('solo_gold_evaluation',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    archive=tmp_path/'archive';archive.mkdir();records=[]
    with zipfile.ZipFile(archive/'user_labels_before.zip','w') as users, zipfile.ZipFile(archive/'agent_labels_snapshot.zip','w') as agents:
        for name, source in [('001','manual'),('002','agent'),('095','manual')]:
            sample=tmp_path/'DataCreate/samples'/name;sample.mkdir(parents=True)
            (sample/'performance_audio.wav').write_bytes(b'audio')
            (sample/'verified_score.musicxml').write_bytes(b'score')
            (sample/'labels.json').write_text(json.dumps({'annotator_id':'agent','labels':[{'source':'agent'}]}))
            user=json.dumps({'labels':[{'source':source,'type':'wrong_note','score_event_indices':[0]}]}).encode()
            agent=json.dumps({'agent_labeling':{'audio_sha256':sha(b'audio'),'score_sha256':sha(b'score')}}).encode()
            member=f'samples/{name}/labels.json';users.writestr(member,user)
            agents.writestr(f'samples/{name}/labels_agent.json',agent)
            records.append({'user_file':member,'user_before_sha256':sha(user),'agent_sha256':sha(agent)})
    (archive/'manifest.json').write_text(json.dumps({'records':records}))
    gold,provenance=module.human_gold(archive)
    assert set(gold)=={'001'}
    assert provenance['001']['score_audio_hashes_verified']
    (tmp_path/'DataCreate/samples/001/verified_score.musicxml').write_bytes(b'different score')
    with pytest.raises(ValueError,match='changed verified_score'):
        module.human_gold(archive)
