"""Native bootstrap tests use fake runtimes; never download or run inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('native_bootstrap', Path(__file__).parents[1]/'packaging/bootstrap.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def bundle(tmp_path, monkeypatch):
    source = tmp_path/'bundle with spaces'
    source.mkdir()
    (source/'app').mkdir()
    (source/'python/bin').mkdir(parents=True)
    for name in ['app/install.py','app/oollim.py','python/bin/python3.11']:
        (source/name).write_text('fake runtime')
    manifest = {str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*') if f.is_file()}
    (source/'payload.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(b,'__file__',str(source/'bootstrap.py'))
    monkeypatch.setattr(b,'data_home',lambda:tmp_path/'data')
    monkeypatch.setattr(b.sys,'platform','darwin')
    monkeypatch.setattr(b.Path,'home',lambda:tmp_path/'user')
    return source


def test_bootstrap_resume_and_output_location(tmp_path, monkeypatch):
    source = bundle(tmp_path,monkeypatch)
    installs=[]
    launches=[]
    def install(command,**kw):
        installs.append(command)
        (kw['cwd']/'.oollim-install.json').write_text('{}')
    monkeypatch.setattr(b.subprocess,'run',install)
    monkeypatch.setattr(b.subprocess,'call',lambda cmd,**kw:launches.append((cmd,kw)) or 0)
    assert b.main()==0
    assert b.main()==0
    assert len(installs)==1 and len(launches)==2
    assert launches[0][1]['cwd']==tmp_path/'data/results'
    assert not (source/'app/.oollim-install.json').exists()
    assert not (tmp_path/'data/setup.lock').exists()


def test_corrupt_payload_rejected_and_lock_released(tmp_path,monkeypatch):
    source=bundle(tmp_path,monkeypatch)
    (source/'app/install.py').write_text('changed')
    with pytest.raises(RuntimeError,match='integrity'):
        b.main()
    assert not (tmp_path/'data/setup.lock').exists()


def test_failed_install_can_resume(tmp_path,monkeypatch):
    bundle(tmp_path,monkeypatch)
    monkeypatch.setattr(b.subprocess,'run',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('interrupted')))
    with pytest.raises(RuntimeError,match='interrupted'):
        b.main()
    assert not (tmp_path/'data/setup.lock').exists()
