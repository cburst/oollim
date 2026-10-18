"""Matched individual/fused frozen representations, with explicit IPA ablations."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from pronunciation.encoders import sha256
from pronunciation.phone_features import canonical_references,phone_evidence,FEATURE_NAMES,MAPPING_VERSION
from run_baseline import atomic_json,metrics


def variants():
    names = ('xeus','xlsr','phoneticxeus')
    subsets = [c for n in range(1,4) for c in itertools.combinations(names,n)]
    return subsets+[('ipa',)]+[c+('ipa',) for c in subsets]


def ridge_candidates(xt,yt,xv,alphas):
    """One dual eigendecomposition reused over the fixed alpha grid."""
    xm,ym=xt.mean(0),yt.mean(0)
    centered=xt-xm
    values,vectors=eigh(centered@centered.T,check_finite=True)
    values=np.maximum(values,0)
    projected=vectors.T@(yt-ym)
    for alpha in alphas:
        if alpha<=0: raise ValueError('Positive ridge penalty required')
        dual=vectors@(projected/(values[:,None]+alpha))
        coef=centered.T@dual
        intercept=ym-xm@coef
        yield alpha,coef,intercept,xv@coef+intercept


def paired_bootstrap(rows,y,a,b,seed,reps):
    speakers=sorted({r['speaker_id'] for r in rows})
    ids=np.array([r['speaker_id'] for r in rows])
    errors=np.abs(y-a)-np.abs(y-b)
    grouped=np.stack([errors[ids==s].mean(0) for s in speakers])
    rng=np.random.default_rng(seed)
    boot=grouped[rng.integers(0,len(speakers),(reps,len(speakers)))].mean(1)
    return dict(speakers=len(speakers),estimate=grouped.mean(0).tolist(),
        low=np.quantile(boot,.025,axis=0).tolist(),high=np.quantile(boot,.975,axis=0).tolist(),
        interpretation='candidate minus comparator; descriptive after validation selection, not confirmatory')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['fusion-v1','text-reference-v1'],default='fusion-v1')
    experiment=parser.parse_args().experiment
    cfgpath=ROOT/f'experiments/{experiment}.json';cfg=json.loads(cfgpath.read_text())
    torch.set_num_threads(cfg['threads'])
    report=ROOT/'reports'/experiment;report.mkdir(parents=True,exist_ok=True);started=time.time()
    audit=json.loads((ROOT/'reports/corpus-audit.json').read_text())
    manifest=ROOT/'resources/corpus-manifest.jsonl'
    assert sha256(manifest)==audit['manifest_sha256']
    assert sha256(ROOT/'reports/splits.json')==audit['split_sha256']
    rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    rows=[r for r in rows if r['split'] in cfg['splits']]
    assert all(sum(r['split']==s for r in rows)==n for s,n in cfg['splits'].items())
    train=np.array([r['split']=='train' for r in rows]);valid=~train
    assert not ({r['speaker_id'] for r in rows if r['split']=='train'}&{r['speaker_id'] for r in rows if r['split']=='validation'})
    labels_path=ROOT/'resources/canonical-metadata/resource/scores.json'
    labels=json.loads(labels_path.read_text())
    y=np.array([[labels[r['recording_id']]['prosodic' if d=='prosody' else d] for d in cfg['dimensions']] for r in rows],dtype=float)
    blocks,provenances={},{}
    for name in ('xeus','xlsr'):
        base=json.loads((ROOT/f'reports/baseline-v1/{name}-results.json').read_text())
        cache=ROOT/'resources/baseline-cache'/base['cache_key']
        prov=json.loads((cache/'provenance.json').read_text())
        assert prov==base['provenance']
        assert prov['manifest_sha256']==audit['manifest_sha256'] and prov['split_sha256']==audit['split_sha256']
        vectors=[]
        for row in rows:
            p=cache/(row['recording_id']+'-'+row['sha256']+'.npy')
            assert sha256(p)==p.with_suffix('.sha256').read_text().strip()
            vectors.append(np.load(p,allow_pickle=False))
        blocks[name]=np.stack(vectors).astype(float);provenances[name]=prov
    phonetic=json.loads((ROOT/'reports/fusion-v1/phonetic-cache.json').read_text())
    cache=ROOT/'resources/phonetic-baseline-cache'/phonetic['cache_key']
    assert json.loads((cache/'provenance.json').read_text())==phonetic['provenance']
    assert phonetic['manifest_sha256']==audit['manifest_sha256'] and phonetic['split_sha256']==audit['split_sha256']
    provenances['phoneticxeus']=phonetic['provenance']
    refs_path=ROOT/'resources/canonical-metadata/resource/text-phone'
    reference_generator=None
    if cfg.get('reference_source')=='cmudict':
        from pronunciation.text_reference import load_dictionary,text_to_phones,VERSION
        refs_path=ROOT/'resources/text-reference-v1/cmudict.dict'
        assert sha256(refs_path)==cfg['reference_resource']['sha256']
        dictionary=load_dictionary(refs_path)
        generated={r['recording_id']:text_to_phones(r['reference_text'],dictionary) for r in rows}
        refs={k:v[0] for k,v in generated.items()}
        atomic_json(report/'text-coverage.json',{k:v[1] for k,v in generated.items()})
        reference_generator=dict(version=VERSION,resource=cfg['reference_resource'],
            code_sha256=sha256(ROOT/'pronunciation/text_reference.py'))
    else:
        refs=canonical_references(refs_path,{r['recording_id'] for r in rows})
    frames,ipa,coverage=[],[],[]
    for row in rows:
        path=cache/(row['recording_id']+'-'+row['sha256']+'.npz')
        assert sha256(path)==path.with_suffix('.sha256').read_text().strip()
        with np.load(path,allow_pickle=False) as saved:
            frames.append(saved['pooled'])
            features,status=phone_evidence(saved['logits'],saved['vocabulary'].tolist(),int(saved['blank_id']),
                refs[row['recording_id']],row['duration_s'])
        ipa.append(features);coverage.append(dict(recording_id=row['recording_id'],split=row['split'],**status))
    blocks['phoneticxeus']=np.stack(frames).astype(float);blocks['ipa']=np.stack(ipa)
    assert all(np.isfinite(x).all() and x.ndim==2 and len(x)==len(rows) for x in blocks.values())
    provenance=dict(config_sha256=sha256(cfgpath),labels_sha256=sha256(labels_path),
        reference_sha256=sha256(refs_path),reference_generator=reference_generator,mapping_version=MAPPING_VERSION,
        feature_names=FEATURE_NAMES,feature_code_sha256=sha256(ROOT/'pronunciation/phone_features.py'),
        upstream_exposure_audit_sha256=sha256(ROOT/'reports/fusion-v1/upstream-exposure-audit.json'),
        runner_sha256=sha256(Path(__file__)),manifest_sha256=audit['manifest_sha256'],
        split_sha256=audit['split_sha256'],backbones=provenances)
    key=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    local=ROOT/'resources'/experiment/key;local.mkdir(parents=True,exist_ok=True)
    atomic_json(local/'provenance.json',provenance)
    atomic_json(local/'phone-coverage.json',coverage)
    scalers={k:StandardScaler().fit(x[train]) for k,x in blocks.items()}
    standardized={k:scalers[k].transform(x)*np.sqrt(2048/x.shape[1]) for k,x in blocks.items()}
    np.savez_compressed(local/'ipa-features.npz',features=blocks['ipa'])
    results,predictions={},{}
    validation_age=np.array([int(r['age_source']) for r in rows if r['split']=='validation'])
    validation_rows=[r for r in rows if r['split']=='validation']
    for subset in variants():
        name='+'.join(subset)
        x=np.concatenate([standardized[k] for k in subset],axis=1)
        candidates=[];best=None
        with threadpool_limits(limits=cfg['threads']):
            for alpha,coef,intercept,pred in ridge_candidates(x[train],y[train],x[valid],cfg['alphas']):
                result=metrics(y[valid],pred,cfg['dimensions']);objective=result['accuracy']['mae']
                candidates.append(dict(alpha=alpha,metrics=result,macro_mae=float(np.abs(y[valid]-pred).mean())))
                if best is None or objective<best[0]:best=(objective,alpha,coef,intercept,pred,result)
        objective,alpha,coef,intercept,pred,result=best
        mean=np.concatenate([scalers[k].mean_ for k in subset])
        scale=np.concatenate([scalers[k].scale_ for k in subset])
        gain=np.concatenate([np.full(blocks[k].shape[1],np.sqrt(2048/blocks[k].shape[1])) for k in subset])
        checkpoint=local/(name+'.npz')
        np.savez(checkpoint,coef=coef,intercept=intercept,scaler_mean=mean,scaler_scale=scale,block_gain=gain)
        raw=np.concatenate([blocks[k][valid] for k in subset],axis=1)
        with np.load(checkpoint,allow_pickle=False) as saved:
            restored=((raw-saved['scaler_mean'])/saved['scaler_scale']*saved['block_gain'])@saved['coef']+saved['intercept']
        np.testing.assert_allclose(restored,pred,rtol=1e-9,atol=1e-9)
        atomic_json(checkpoint.with_suffix('.json'),dict(format_version=1,
            checkpoint_type='frozen_feature_fusion_ridge',coefficient_layout='features_by_outputs',
            blocks=subset,block_widths={k:blocks[k].shape[1] for k in subset},
            output_dimensions=cfg['dimensions'],target_scale='SpeechOcean native 0..10',
            backbone_provenance={k:provenances[k] for k in subset if k!='ipa'},
            ipa_generator_provenance=provenances['phoneticxeus'] if 'ipa' in subset else None,
            experiment_provenance=provenance,config=cfg,selected_alpha=alpha,
            checkpoint_sha256=sha256(checkpoint)))
        predictions[name]=pred
        results[name]=dict(blocks=subset,width=x.shape[1],selected_alpha=alpha,metrics=result,
            macro_mae=float(np.abs(y[valid]-pred).mean()),candidates=candidates,
            age_groups={group:dict(recordings=int(mask.sum()),
                speakers=len({r['speaker_id'] for r,keep in zip(validation_rows,mask) if keep}),
                metrics=metrics(y[valid][mask],pred[mask],cfg['dimensions']))
                for group,mask in [('under18',validation_age<18),('adult',validation_age>=18)]},
            checkpoint=str(checkpoint.relative_to(ROOT)),restore_verified=True)
        atomic_json(report/(name+'.json'),results[name])
        print(json.dumps(dict(variant=name,accuracy_mae=objective,alpha=alpha,elapsed_s=time.time()-started)),flush=True)
        atomic_json(report/'fit-status.json',dict(phase='fitting',completed=len(results),total=len(variants()),elapsed_s=time.time()-started))
    vr=[r for r in rows if r['split']=='validation']
    contrasts={}
    for subset in variants():
        name='+'.join(subset)
        if 'ipa' in subset and len(subset)>1:
            comparator='+'.join(subset[:-1])
            contrasts[name+' minus '+comparator]=paired_bootstrap(vr,y[valid],predictions[name],predictions[comparator],cfg['seed'],cfg['bootstrap_speaker_replicates'])
        elif len(subset)>1:
            contrasts[name+' minus xeus']=paired_bootstrap(vr,y[valid],predictions[name],predictions['xeus'],cfg['seed'],cfg['bootstrap_speaker_replicates'])
    atomic_json(local/'validation-predictions.json',dict(recording_ids=[r['recording_id'] for r in vr],
        targets=y[valid].tolist(),predictions={k:v.tolist() for k,v in predictions.items()}))
    summary=dict(status='complete',config=cfg,run_key=key,provenance=provenance,results=results,
        contrasts=contrasts,mean_baseline=metrics(y[valid],np.broadcast_to(y[train].mean(0),y[valid].shape),cfg['dimensions']),
        phone_coverage={s:dict(n=sum(r['split']==s for r in coverage),
            complete=sum(r['reference_complete'] for r in coverage if r['split']==s),
            alignable=sum(r['ctc_alignable'] for r in coverage if r['split']==s)) for s in cfg['splits']},
        n_train=int(train.sum()),n_validation=int(valid.sum()),validation_speakers=len({r['speaker_id'] for r in vr}),
        test_evaluated=False,elapsed_s=time.time()-started,
        limitation='Validation selected alpha and screens 15 variants; bootstrap is descriptive, not independent significance. Upstream configuration references SpeechOcean-derived data; exact pinned-checkpoint exposure unresolved. Approximate IPA mapping and shared XEUS ancestry; no SOTA claim.')
    atomic_json(report/'comparison.json',summary)
    atomic_json(report/'fit-status.json',dict(phase='complete',elapsed_s=time.time()-started,run_key=key))


if __name__=='__main__': main()
