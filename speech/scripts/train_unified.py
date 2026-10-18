"""Fixed neural fusion comparison, with portable checkpoint restoration."""
import copy
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from pronunciation.encoders import sha256
from pronunciation.fusion import FusionNetwork,UnifiedScorer,DIMENSIONS
from pronunciation.scoring import masked_loss
from run_baseline import atomic_json,metrics
from run_fusion_baselines import paired_bootstrap
from fusion_data import load_data


def main():
    cfgpath=ROOT/'experiments/unified-neural-v1.json';cfg=json.loads(cfgpath.read_text())
    torch.set_num_threads(cfg['threads']);torch.use_deterministic_algorithms(True)
    report=ROOT/'reports/unified-neural-v1';report.mkdir(parents=True,exist_ok=True)
    local=ROOT/'resources/unified-neural-v1';local.mkdir(parents=True,exist_ok=True)
    started=time.time();rows,train,valid,x,y,baseline,ipa_hash=load_data()
    ridge=UnifiedScorer(baseline)
    old=json.loads((baseline.parent/'validation-predictions.json').read_text())
    ridgepred=ridge.predict_features(x[valid])
    np.testing.assert_allclose(ridgepred,old['predictions']['xeus+xlsr+phoneticxeus+ipa'],atol=1e-10,rtol=1e-10)
    z=(x-ridge.arrays['scaler_mean'])/ridge.arrays['scaler_scale']*ridge.arrays['block_gain']
    xt=torch.from_numpy(z[train].astype(np.float32));xv=torch.from_numpy(z[valid].astype(np.float32))
    yt=torch.from_numpy((y[train]/10).astype(np.float32));results=[];predictions={}
    for seed in cfg['seeds']:
        torch.manual_seed(seed);model=FusionNetwork(x.shape[1],**cfg['architecture'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        best=float('inf');history=[];beststate=None;bestepoch=None
        for epoch in range(1,cfg['epochs']+1):
            model.train();total=0
            for ids in torch.randperm(len(xt)).split(cfg['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                loss=masked_loss(model(xt[ids]),{'speechocean':{d:yt[ids,j] for j,d in enumerate(DIMENSIONS)}})
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward()
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Missing/nonfinite gradient')
                optimizer.step();total+=float(loss.detach())*len(ids)
            model.eval()
            with torch.no_grad():pred=model.array(xv).numpy()
            objective=float(np.abs(pred[:,0]-y[valid,0]).mean())
            history.append(dict(epoch=epoch,training_loss=total/len(xt),validation_accuracy_mae=objective))
            if objective<best:
                best=objective;bestepoch=epoch;beststate=copy.deepcopy(model.state_dict());bestopt=copy.deepcopy(optimizer.state_dict())
            if epoch%10==0:
                atomic_json(report/'status.json',dict(status='training',seed=seed,epoch=epoch,elapsed_s=time.time()-started))
        model.load_state_dict(beststate);model.eval()
        with torch.no_grad():pred=model.array(xv).numpy()
        checkpoint=local/f'seed-{seed}.pt'
        torch.save(dict(state_dict=beststate,optimizer_state=bestopt,epoch=bestepoch,
            **{k:torch.from_numpy(ridge.arrays[k].copy()) for k in ('scaler_mean','scaler_scale','block_gain')}),checkpoint)
        meta=copy.deepcopy(ridge.metadata)
        meta.update(checkpoint_type='frozen_feature_fusion_neural',architecture=cfg['architecture'],
            checkpoint_sha256=sha256(checkpoint),target_transform='divide by 10 for loss; multiply by 10 at inference',
            training=dict(config=cfg,seed=seed,best_epoch=bestepoch,config_sha256=sha256(cfgpath),
                runner_sha256=sha256(Path(__file__)),model_code_sha256=sha256(ROOT/'pronunciation/fusion.py'),
                baseline_sha256=sha256(baseline),ipa_features_sha256=ipa_hash))
        meta.pop('selected_alpha',None);meta.pop('coefficient_layout',None)
        atomic_json(checkpoint.with_suffix('.json'),meta)
        restored=UnifiedScorer(checkpoint).predict_features(x[valid])
        np.testing.assert_allclose(pred,restored,rtol=1e-6,atol=1e-6)
        record=dict(seed=seed,best_epoch=bestepoch,metrics=metrics(y[valid],pred,DIMENSIONS),
            macro_mae=float(np.abs(pred-y[valid]).mean()),checkpoint=str(checkpoint.relative_to(ROOT)),
            restored=True,history=history,versus_ridge=paired_bootstrap([r for r in rows if r['split']=='validation'],y[valid],pred,ridgepred,42,2000))
        results.append(record);predictions[str(seed)]=pred.tolist();atomic_json(report/f'seed-{seed}.json',record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('history','versus_ridge')}),flush=True)
    atomic_json(local/'validation-predictions.json',dict(recording_ids=old['recording_ids'],targets=y[valid].tolist(),predictions=predictions))
    summary=dict(status='complete',config=cfg,results=results,ridge=metrics(y[valid],ridgepred,DIMENSIONS),
        mean_neural_mae={d:float(np.mean([r['metrics'][d]['mae'] for r in results])) for d in DIMENSIONS},
        test_evaluated=False,elapsed_s=time.time()-started,validation_selection_not_independent=True)
    atomic_json(report/'comparison.json',summary);atomic_json(report/'status.json',dict(status='complete',elapsed_s=time.time()-started))


if __name__=='__main__':main()
