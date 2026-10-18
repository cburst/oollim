"""Verified development-only inputs shared by fusion training and checks."""
import json
from pathlib import Path
import numpy as np
from pronunciation.encoders import ROOT, sha256


def load_data():
    report=json.loads((ROOT/'reports/text-reference-v1/comparison.json').read_text())
    prov=report['provenance']
    manifest=ROOT/'resources/corpus-manifest.jsonl'
    assert sha256(manifest)==prov['manifest_sha256']
    assert sha256(ROOT/'reports/splits.json')==prov['split_sha256']
    labels_path=ROOT/'resources/canonical-metadata/resource/scores.json'
    assert sha256(labels_path)==prov['labels_sha256']
    rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    rows=[r for r in rows if r['split'] in ('train','validation')]
    train=np.array([r['split']=='train' for r in rows]);valid=~train
    assert train.sum()==2000 and valid.sum()==500
    assert not ({r['speaker_id'] for r in rows if r['split']=='train'} & {r['speaker_id'] for r in rows if r['split']=='validation'})
    blocks={}
    for name in ('xeus','xlsr','phoneticxeus'):
        if name=='phoneticxeus':
            receipt=json.loads((ROOT/'reports/fusion-v1/phonetic-cache.json').read_text())
            directory=ROOT/'resources/phonetic-baseline-cache'/receipt['cache_key'];ext='.npz'
        else:
            receipt=json.loads((ROOT/f'reports/baseline-v1/{name}-results.json').read_text())
            directory=ROOT/'resources/baseline-cache'/receipt['cache_key'];ext='.npy'
        assert receipt['provenance']==prov['backbones'][name]
        vectors=[]
        for row in rows:
            file=directory/(row['recording_id']+'-'+row['sha256']+ext)
            assert sha256(file)==file.with_suffix('.sha256').read_text().strip()
            if ext=='.npz':
                with np.load(file,allow_pickle=False) as f:vectors.append(f['pooled'])
            else:vectors.append(np.load(file,allow_pickle=False))
        blocks[name]=np.stack(vectors)
    local=ROOT/'resources/text-reference-v1'/report['run_key']
    assert json.loads((local/'provenance.json').read_text())==prov
    with np.load(local/'ipa-features.npz',allow_pickle=False) as f:blocks['ipa']=f['features']
    rowids=json.loads((local/'phone-coverage.json').read_text())
    assert [r['recording_id'] for r in rowids]==[r['recording_id'] for r in rows]
    labels=json.loads(labels_path.read_text())
    y=np.array([[labels[r['recording_id']]['prosodic' if d=='prosody' else d] for d in report['config']['dimensions']] for r in rows])
    name='xeus+xlsr+phoneticxeus+ipa'
    x=np.concatenate([blocks[k] for k in report['results'][name]['blocks']],axis=1).astype(np.float64)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return rows,train,valid,x,y,ROOT/report['results'][name]['checkpoint'],sha256(local/'ipa-features.npz')
