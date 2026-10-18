"""Matched frozen-feature controls; immutable protocol, development-only inputs."""
import argparse,copy,json,os,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from pronunciation.fusion import FusionNetwork,DIMENSIONS
from pronunciation.scoring import masked_loss
from pronunciation.development import digest,validate_splits,metrics,paired_speaker_bootstrap,create_run,finish_run


def check(value,message):
    if not value:raise ValueError(message)


def atomic_json(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def atomic_torch(path,value):
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def standardize(x,train):
    mean=x[train].astype(np.float64).mean(0)
    scale=x[train].astype(np.float64).std(0)
    scale=np.where(scale<1e-8,1.,scale)
    return (x-mean)/scale,mean,scale


def score(y,p):return {d:metrics(y[:,j],p[:,j]) for j,d in enumerate(DIMENSIONS)}


def ridge_fit(x,y,alpha):
    # Intercept is unpenalized. Dual form avoids a larger feature-space solve.
    x=torch.as_tensor(x,dtype=torch.float64);y=torch.as_tensor(y,dtype=torch.float64)
    xm=x.mean(0);ym=y.mean(0);x=x-xm;y=y-ym
    if x.shape[1]>len(x):
        coef=x.T@torch.linalg.solve(x@x.T+alpha*torch.eye(len(x),dtype=x.dtype),y)
    else:coef=torch.linalg.solve(x.T@x+alpha*torch.eye(x.shape[1],dtype=x.dtype),x.T@y)
    return coef,ym-xm@coef


def fit_neural(x,y,train,valid,seed,cfg,device):
    torch.manual_seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    model=FusionNetwork(x.shape[1],**cfg['architecture']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    xt=torch.tensor(x[train],dtype=torch.float32,device=device);yt=torch.tensor(y[train]/10,dtype=torch.float32,device=device)
    xv=torch.tensor(x[valid],dtype=torch.float32,device=device)
    generator=torch.Generator().manual_seed(seed)
    best=None;state=None;best_optimizer=None;history=[];stale=0;started=time.perf_counter()
    for epoch in range(1,cfg['epochs']+1):
        model.train();total=0.
        for ids in torch.randperm(len(xt),generator=generator).split(cfg['batch_size']):
            ids=ids.to(device);optimizer.zero_grad(set_to_none=True)
            loss=masked_loss(model(xt[ids]),{'speechocean':{d:yt[ids,j] for j,d in enumerate(DIMENSIONS)}})
            check(bool(torch.isfinite(loss)),'Nonfinite neural loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(ids)
        model.eval()
        with torch.no_grad():pred=model.array(xv).cpu().numpy();training=model.array(xt).cpu().numpy()
        va=score(y[valid],pred);tr=score(y[train],training)
        check(all(v['spearman'] is not None for v in va.values()),'Undefined validation correlation')
        objective=(float(np.mean([v['spearman'] for v in va.values()])), -float(np.mean([v['mae'] for v in va.values()])))
        history.append(dict(epoch=epoch,training_loss=total/len(xt),train=tr,validation=va,checkpoint_objective=list(objective)))
        if best is None or objective>best:
            best=objective;bestepoch=epoch;stale=0
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            best_optimizer=copy.deepcopy(optimizer.state_dict())
        else:stale+=1
        if stale>=cfg['patience']:break
    model.load_state_dict(state);model.eval()
    with torch.no_grad():pred=model.array(xv).cpu().numpy()
    return dict(state_dict=state,optimizer_state=best_optimizer,epoch=bestepoch,history=history),pred,sum(p.numel() for p in model.parameters()),time.perf_counter()-started


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=p.parse_args()
    cfg=json.loads(args.protocol.read_text());torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.device=='cuda':check(torch.cuda.is_available(),'CUDA unavailable')
    for path,sha in cfg['code_hashes'].items():check(digest(ROOT/path)==sha,'Changed source: '+path)
    prep_path=ROOT/cfg['preparation'];check(digest(prep_path)==cfg['preparation_sha256'],'Changed preparation receipt')
    prep=json.loads(prep_path.read_text());artifact=ROOT/prep['features']
    check(digest(artifact)==prep['features_sha256'],'Changed feature bundle')
    manifest=ROOT/cfg['manifest'];check(digest(manifest)==prep['manifest_sha256'],'Changed manifest')
    rows=[json.loads(s) for s in manifest.read_text().splitlines()];validate_splits(rows)
    check(all(r['split'] in ('train','validation') for r in rows),'Final-test input forbidden')
    with np.load(artifact,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    check(arrays['recording_ids'].tolist()==[r['recording_id'] for r in rows],'Recording order differs')
    check(arrays['speakers'].tolist()==[r['dataset']+':'+r['speaker_id'] for r in rows],'Speaker identity differs')
    check(arrays['splits'].tolist()==[r['split'] for r in rows],'Split assignment differs')
    y=np.array([[r['labels']['prosodic' if d=='prosody' else d] for d in DIMENSIONS] for r in rows])
    check(np.array_equal(y,arrays['targets']),'Labels differ from audited manifest')
    train=arrays['splits']=='train';valid=arrays['splits']=='validation';groups=arrays['speakers'][valid]
    check(int(train.sum())==2000 and int(valid.sum())==500,'Unexpected sample counts')
    environment=dict(python=sys.version.split()[0],torch=str(torch.__version__),numpy=np.__version__,device=args.device,
        gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,cuda=torch.version.cuda,threads=4,deterministic=True,tf32=False)
    configuration=dict(cfg,environment=environment,protocol_sha256=digest(args.protocol))
    registry=ROOT/'reports/experiment-registry'/cfg['experiment_id']
    out=ROOT/'resources'/cfg['experiment_id']
    if registry.exists():
        check(json.loads((registry/'configuration.json').read_text())==configuration,'Resume configuration differs')
        check(digest(registry/'configuration.json')==(registry/'configuration.sha256').read_text().strip(),'Registry changed')
        if (registry/'results.json').exists():
            final=json.loads((registry/'results.json').read_text())
            for key,result in final['models'].items():
                check(json.loads((out/(key+'.json')).read_text())==result,'Changed model receipt')
                check(digest(ROOT/result['checkpoint'])==result['checkpoint_sha256'],'Changed checkpoint')
                check(digest(out/(key+'.npz'))==result['predictions_sha256'],'Changed predictions')
                if result.get('curves_sha256'):check(digest(out/(key+'-curves.json'))==result['curves_sha256'],'Changed curves')
            for key,sha in final['bootstrap_sha256'].items():check(digest(out/(key+'-bootstrap.json'))==sha,'Changed bootstrap artifact')
            print('Already completed; artifacts verified');return
    else:create_run(registry.parent,cfg['experiment_id'],configuration)
    out.mkdir(exist_ok=True)
    predictions={};results={};total_seconds=0;parameters={}
    for name in cfg['encoders']:
        raw=arrays[name];check(raw.ndim==2 and len(raw)==len(rows) and bool(np.isfinite(raw).all()),'Invalid feature block')
        x,mean,scale=standardize(raw,train)
        for algorithm,seed in [('ridge',None)]+[('neural',s) for s in cfg['seeds']]:
            key=f'{name}-{algorithm}'+('' if seed is None else f'-{seed}')
            receipt=out/(key+'.json');checkpoint=out/(key+'.pt');predpath=out/(key+'.npz')
            if receipt.exists():
                result=json.loads(receipt.read_text())
                check(result['protocol_sha256']==digest(args.protocol),'Changed model protocol')
                check(digest(checkpoint)==result['checkpoint_sha256'] and digest(predpath)==result['predictions_sha256'],'Corrupt completed model')
                if result.get('curves_sha256'):check(digest(out/(key+'-curves.json'))==result['curves_sha256'],'Changed curves')
                with np.load(predpath,allow_pickle=False) as f:pred=f['predictions'].copy()
            else:
                started=time.perf_counter()
                if algorithm=='ridge':
                    coef,intercept=ridge_fit(x[train],y[train],cfg['ridge_alpha'])
                    pred=(torch.from_numpy(x[valid])@coef+intercept).numpy()
                    payload=dict(coef=coef,intercept=intercept);count=coef.numel()+intercept.numel();seconds=time.perf_counter()-started
                else:payload,pred,count,seconds=fit_neural(x,y,train,valid,seed,cfg['neural'],args.device)
                payload.update(scaler_mean=torch.from_numpy(mean),scaler_scale=torch.from_numpy(scale),metadata=dict(configuration=configuration,backbone=name,
                    backbone_configuration=cfg['backbone_provenance'].get(name),input_width=raw.shape[1],algorithm=algorithm,seed=seed,feature_sha256=prep['features_sha256'],target_scale='human 0..10; neural loss uses /10'))
                atomic_torch(checkpoint,payload)
                restored=torch.load(checkpoint,map_location='cpu',weights_only=True)
                xx=(raw[valid]-restored['scaler_mean'].numpy())/restored['scaler_scale'].numpy()
                if algorithm=='ridge':again=(torch.from_numpy(xx)@restored['coef']+restored['intercept']).numpy()
                else:
                    network=FusionNetwork(raw.shape[1],**cfg['neural']['architecture']).to(args.device)
                    network.load_state_dict(restored['state_dict']);network.eval()
                    with torch.no_grad():again=network.array(torch.tensor(xx,dtype=torch.float32,device=args.device)).cpu().numpy()
                np.testing.assert_allclose(again,pred,rtol=1e-6,atol=1e-6)
                with predpath.with_suffix('.tmp').open('wb') as f:np.savez(f,predictions=pred,recording_ids=arrays['recording_ids'][valid])
                predpath.with_suffix('.tmp').replace(predpath)
                result=dict(validation_metrics=score(y[valid],pred),parameter_count=count,training_seconds=seconds,checkpoint=str(checkpoint.relative_to(ROOT)),
                    checkpoint_sha256=digest(checkpoint),predictions_sha256=digest(predpath),checkpoint_parity=True,protocol_sha256=digest(args.protocol),best_epoch=payload.get('epoch'),epochs_run=len(payload.get('history',[])))
                if 'history' in payload:
                    curves=out/(key+'-curves.json');atomic_json(curves,payload['history']);result['curves_sha256']=digest(curves)
                atomic_json(receipt,result)
            predictions[key]=pred;results[key]=result;parameters[key]=result['parameter_count'];total_seconds+=result['training_seconds']
            print(json.dumps({'completed':key,'metrics':result['validation_metrics'],'epoch':result['best_epoch']}),flush=True)
    # Predefined comparisons: every neural seed vs its ridge; each encoder vs XEUS
    # within the same algorithm/seed. Bootstrap draws preserve whole speakers.
    contrasts=[]
    for name in cfg['encoders']:
        for seed in cfg['seeds']:contrasts.append((f'{name}-neural-{seed}',f'{name}-ridge'))
    for name in cfg['encoders']:
        if name=='xeus':continue
        contrasts.append((f'{name}-ridge','xeus-ridge'))
        for seed in cfg['seeds']:contrasts.append((f'{name}-neural-{seed}',f'xeus-neural-{seed}'))
    uncertainty={}
    for a,b in contrasts:
        key=a+'__vs__'+b;file=out/(key+'-bootstrap.json')
        if file.exists():value=json.loads(file.read_text());check(value['protocol_sha256']==digest(args.protocol),'Bootstrap protocol changed')
        else:
            value=dict(protocol_sha256=digest(args.protocol),constructs={d:paired_speaker_bootstrap(y[valid,j],predictions[a][:,j],predictions[b][:,j],groups,replicates=cfg['bootstrap_replicates']) for j,d in enumerate(DIMENSIONS)})
            atomic_json(file,value)
        uncertainty[key]=value['constructs'];print('bootstrap '+key,flush=True)
    controls={stat:score(y[valid],np.broadcast_to(getattr(np,stat)(y[train],axis=0),y[valid].shape)) for stat in ('mean','median')}
    finish_run(registry,dict(validation_metrics={k:v['validation_metrics'] for k,v in results.items()},test_metrics=None,parameter_count=parameters,training_seconds=total_seconds,
        checkpoint=str(out.relative_to(ROOT)),models=results,central_tendency=controls,uncertainty=uncertainty,bootstrap_sha256={k:digest(out/(k+'-bootstrap.json')) for k in uncertainty},recordings={'train':int(train.sum()),'validation':int(valid.sum())},
        speaker_groups={'train':len(set(arrays['speakers'][train])),'validation':len(set(groups))},
        interpretation='Exploratory single-corpus development comparison; unadjusted bootstrap intervals, epoch selection on same validation partition; no independent generalization or multiplicity-adjusted winner claim. IPA-only head has fewer parameters.'))
    print('COMPLETED '+cfg['experiment_id'],flush=True)

if __name__=='__main__':main()
