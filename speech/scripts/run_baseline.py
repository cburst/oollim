"""Matched frozen representation baseline. Never extracts/evaluates test audio."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pronunciation.audio import read_audio, batch_audio, PREPROCESSING_VERSION
from pronunciation.encoders import build_encoder, default_config, sha256
from pronunciation.scoring import masked_pool


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def metrics(y, pred, dimensions):
    result = {}
    for j, d in enumerate(dimensions):
        a, b = y[:, j], pred[:, j]
        result[d] = dict(mae=float(np.abs(a-b).mean()),
            rmse=float(np.sqrt(np.square(a-b).mean())),
            pearson=float(pearsonr(a,b).statistic) if np.std(b)>1e-12 else None,
            spearman=float(spearmanr(a,b).statistic) if np.std(b)>1e-12 else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', choices=['xeus','xlsr'], default='xeus')
    args = parser.parse_args()
    cfgpath = ROOT / 'experiments/baseline-v1.json'
    cfg = json.loads(cfgpath.read_text())
    torch.set_num_threads(cfg['threads']); torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    started = time.time()
    splitpath = ROOT / 'reports/splits.json'
    audit = json.loads((ROOT / 'reports/corpus-audit.json').read_text())
    assert sha256(splitpath) == audit['split_sha256']
    splits = json.loads(splitpath.read_text())['splits']
    rows = [json.loads(x) for x in (ROOT/'resources/corpus-manifest.jsonl').read_text().splitlines()]
    assert sha256(ROOT/'resources/corpus-manifest.jsonl') == audit['manifest_sha256']
    rows = [r for r in rows if r['split'] in cfg['splits']]
    for s, n in cfg['splits'].items():
        assert {r['recording_id'] for r in rows if r['split']==s} == set(splits[s])
        assert sum(r['split']==s for r in rows)==n
    assert not ({r['speaker_id'] for r in rows if r['split']=='train'} &
                {r['speaker_id'] for r in rows if r['split']=='validation'})
    encconfig = default_config(args.backbone)
    provenance = dict(backbone=asdict(encconfig), preprocessing=PREPROCESSING_VERSION,
        pooling=cfg['pooling'], split_sha256=audit['split_sha256'],
        manifest_sha256=audit['manifest_sha256'], torch_version=str(torch.__version__),
        adapter_source_sha256=sha256(ROOT/'pronunciation/encoders.py'),
        pooling_source_sha256=sha256(ROOT/'pronunciation/scoring.py'))
    cachekey = hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    cache = ROOT/'resources/baseline-cache'/cachekey
    cache.mkdir(parents=True,exist_ok=True)
    atomic_json(cache/'provenance.json',provenance)
    reportdir = ROOT/'reports/baseline-v1'; reportdir.mkdir(parents=True,exist_ok=True)
    progress = reportdir/(args.backbone+'-progress.json')
    encoder = None
    features = []
    for i,r in enumerate(rows):
        path = cache/(r['recording_id']+'-'+r['sha256']+'.npy')
        receipt = path.with_suffix('.sha256')
        if not (path.exists() and receipt.exists()):
            if encoder is None:
                encoder = build_encoder(encconfig)
                encoder.eval()
            audio_path = ROOT/r['audio_path']
            assert sha256(audio_path)==r['sha256']
            with torch.inference_mode():
                frames = encoder(batch_audio([read_audio(audio_path)]))
                pooled = masked_pool(frames)[0].cpu().numpy()
            with path.with_suffix('.tmp').open('wb') as f:
                np.save(f,pooled,allow_pickle=False)
            path.with_suffix('.tmp').replace(path)
            receipt.write_text(sha256(path)+'\n')
        assert sha256(path)==receipt.read_text().strip()
        assert sha256(ROOT/r['audio_path'])==r['sha256']
        vector = np.load(path,allow_pickle=False)
        assert vector.ndim==1 and np.isfinite(vector).all()
        features.append(vector)
        if (i+1)%25==0 or i+1==len(rows):
            status=dict(backbone=args.backbone,phase='extracting',completed=i+1,total=len(rows),
                        elapsed_s=time.time()-started,cache_key=cachekey)
            atomic_json(progress,status);print(json.dumps(status),flush=True)
    del encoder
    x = np.stack(features).astype(np.float64)
    labels = json.loads((ROOT/'resources/canonical-metadata/resource/scores.json').read_text())
    y = np.array([[labels[r['recording_id']]['prosodic' if d=='prosody' else d]
                   for d in cfg['dimensions']] for r in rows],dtype=float)
    train = np.array([r['split']=='train' for r in rows]); valid=~train
    scaler=StandardScaler().fit(x[train]); xt=scaler.transform(x[train]); xv=scaler.transform(x[valid])
    meanpred=np.broadcast_to(y[train].mean(0),y[valid].shape)
    candidates=[];best=None
    for alpha in cfg['alphas']:
        model=Ridge(alpha=alpha,solver='svd').fit(xt,y[train])
        pred=model.predict(xv); mae=float(np.abs(pred-y[valid]).mean())
        candidates.append(dict(alpha=alpha,macro_mae=mae,metrics=metrics(y[valid],pred,cfg['dimensions'])))
        if best is None or mae<best[0]: best=(mae,alpha,model,pred)
    _,alpha,model,pred=best
    # Portable linear checkpoint: no pickle, plus complete backbone provenance.
    label_hash=sha256(ROOT/'resources/canonical-metadata/resource/scores.json')
    runkey=hashlib.sha256((cachekey+sha256(cfgpath)+label_hash+sha256(Path(__file__))).encode()).hexdigest()
    runroot=ROOT/'resources/baseline-models'/runkey
    runroot.mkdir(parents=True,exist_ok=True)
    checkpoint=runroot/'ridge.npz'
    np.savez(checkpoint,coef=model.coef_,intercept=model.intercept_,
             scaler_mean=scaler.mean_,scaler_scale=scaler.scale_)
    with np.load(checkpoint,allow_pickle=False) as saved:
        restored=((x[valid]-saved['scaler_mean'])/saved['scaler_scale'])@saved['coef'].T+saved['intercept']
        np.testing.assert_allclose(restored,pred,rtol=1e-10,atol=1e-10)
    atomic_json(runroot/'ridge-metadata.json',dict(format_version=1,**provenance,config=cfg,
        alpha=alpha,labels_sha256=label_hash,runner_sha256=sha256(Path(__file__)),checkpoint_sha256=sha256(checkpoint)))
    vr=[r for r in rows if r['split']=='validation']
    predictions=[dict(recording_id=r['recording_id'],speaker_id=r['speaker_id'],
        target=dict(zip(cfg['dimensions'],map(float,a))),
        prediction=dict(zip(cfg['dimensions'],map(float,b)))) for r,a,b in zip(vr,y[valid],pred)]
    atomic_json(reportdir/(args.backbone+'-predictions.json'),predictions)
    result=dict(status='complete',experiment=cfg,config_sha256=sha256(cfgpath),
        provenance=provenance,cache_key=cachekey,selected_alpha=alpha,candidates=candidates,
        mean_baseline=metrics(y[valid],meanpred,cfg['dimensions']),
        selected_validation_metrics=metrics(y[valid],pred,cfg['dimensions']),
        checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_restore_verified=True,
        n_train=int(train.sum()),n_validation=int(valid.sum()),test_evaluated=False,
        elapsed_s=time.time()-started,
        limitation='Validation used to select alpha; not independent test or SOTA evidence')
    atomic_json(reportdir/(args.backbone+'-results.json'),result)
    atomic_json(progress,dict(backbone=args.backbone,phase='complete',elapsed_s=time.time()-started))
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
