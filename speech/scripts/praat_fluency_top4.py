"""Matched top-four fusion/Praat study, training-speaker-only ridge selection."""
import json,sys,time,fcntl
from pathlib import Path
import numpy as np
import torch
from praat_matched_comparison import fit,predict
from pronunciation.development import digest,validate_splits,metrics,paired_speaker_bootstrap
from pronunciation.praat_features import parse_csv,NAMES
from single_encoder_comparison import atomic_json,atomic_torch
ROOT=Path(__file__).resolve().parents[1]

def timing_features(p):
    nonspeech=p[:,5]-p[:,4]
    proxy=np.divide(nonspeech,p[:,1],out=np.full(len(p),np.nan),where=p[:,1]>0)
    return np.column_stack([p[:,2],p[:,3],p[:,1],nonspeech,nonspeech/p[:,5],proxy])

def main():
 cfg=json.loads((ROOT/'experiments/praat-fluency-top4-v1.json').read_text());out=ROOT/'resources'/cfg['experiment_id'];out.mkdir(exist_ok=True)
 lock=(out/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(2)
 def status(stage,**kw):atomic_json(out/'status.json',dict(stage=stage,updated_unix=time.time(),**kw))
 try:
  for name,sha in cfg['code_hashes'].items():assert digest(ROOT/name)==sha,name
  configpath=out/'configuration.json'
  if configpath.exists():assert json.loads(configpath.read_text())==cfg
  else:atomic_json(configpath,cfg)
  manifest=ROOT/cfg['manifest'];assert digest(manifest)==cfg['manifest_sha256'];rows=list(map(json.loads,manifest.read_text().splitlines()));validate_splits(rows)
  assert all(r['source_split']=='train' and r['split'] in ('train','validation') for r in rows)
  prep=json.loads((ROOT/cfg['preparation']).read_text());bundle=ROOT/prep['features'];assert digest(bundle)==prep['features_sha256']==cfg['features_sha256']
  with np.load(bundle,allow_pickle=False) as f:
   assert f['recording_ids'].tolist()==[r['recording_id'] for r in rows]
   blocks={k:f[k].astype(float) for k in ['xeus','xlsr','phoneticxeus','ipa']}
  cache=ROOT/cfg['praat_cache'];ch=digest(cache/'configuration.json');assert json.loads((cache/'status.json').read_text())['stage']=='completed'
  included=[];values=[];failures=[];receipts={}
  for i,r in enumerate(rows):
   rp=cache/(r['recording_id']+'.json');v=json.loads(rp.read_text());receipts[r['recording_id']]=digest(rp)
   assert v['audio_sha256']==r['sha256'] and v['configuration_sha256']==ch
   if v['status']!='complete':failures.append(r['recording_id']);continue
   cp=cache/(r['recording_id']+'.csv');assert digest(cp)==v['csv_sha256']
   expected=Path(r['audio_path']).stem.replace(' ','').replace(',','')+'_DeepFilterNet3.wav'
   p=parse_csv(cp.read_text(),component_status=v['component_status'],expected_filename=expected);assert np.array_equal(p,v['features'])
   included.append(i);values.append(p)
  rows=[rows[i] for i in included];blocks={k:v[included] for k,v in blocks.items()};p=np.array(values);t=timing_features(p)
  train=np.array([r['split']=='train' for r in rows]);groups=np.array([r['speaker_id'] for r in rows]);targets=('accuracy','fluency','prosody')
  y=np.array([[r['labels']['prosodic' if d=='prosody' else d] for d in targets] for r in rows],float)
  atomic_json(out/'coverage.json',dict(train=int(train.sum()),validation=int((~train).sum()),train_speakers=len(set(groups[train])),validation_speakers=len(set(groups[~train])),excluded=failures,receipt_hashes=receipts,praat_configuration_sha256=ch))
  results={};predictions={}
  for base in cfg['bases']:
   rawbase=np.concatenate([blocks[b] for b in base],axis=1);base_name='+'.join(base)
   variants=cfg['variants'] if base in cfg['fluency_top4'] else ['base']
   for variant in variants:
    name=base_name+'__'+variant;status('training',model=name)
    raw=rawbase if variant=='base' else np.c_[rawbase,t] if variant=='timing' else np.c_[rawbase,p,t[:,3:]]
    checkpoint=out/(name+'.pt');receipt=out/(name+'.json')
    if receipt.exists():
     assert digest(checkpoint)==json.loads(receipt.read_text())['sha256'];payload=torch.load(checkpoint,weights_only=True)
    else:
     payload,cv=fit(raw,y,groups,train,cfg['alphas']);payload.update(protocol=cfg,backbone_names=base,backbone_configuration=prep,feature_variant=variant,coverage_sha256=digest(out/'coverage.json'))
     atomic_torch(checkpoint,payload);atomic_json(out/(name+'-cv.json'),cv)
     np.testing.assert_allclose(predict(payload,raw[~train]),predict(torch.load(checkpoint,weights_only=True),raw[~train]),rtol=0,atol=0)
     atomic_json(receipt,dict(sha256=digest(checkpoint),restore_parity=True))
    pred=predict(payload,raw[~train]);predictions[name]=pred;results[name]={d:metrics(y[~train,j],pred[:,j]) for j,d in enumerate(targets)};print('completed '+name,flush=True)
  status('bootstrap');comparisons={}
  for base in cfg['fluency_top4']:
   name='+'.join(base)
   for variant in ['timing','full']:
    candidate=name+'__'+variant;comparisons[candidate]={d:paired_speaker_bootstrap(y[~train,j],predictions[candidate][:,j],predictions[name+'__base'][:,j],groups[~train],replicates=2000) for j,d in enumerate(targets)}
  np.savez(out/'predictions.npz',**predictions,targets=y[~train],recording_ids=np.array([r['recording_id'] for r in rows])[~train],speakers=groups[~train])
  atomic_json(out/'results.json',dict(metrics=results,comparisons=comparisons,leaderboards={d:sorted(results,key=lambda n:results[n][d]['spearman'],reverse=True) for d in targets},limitations=cfg['limitations']))
  status('completed')
 except Exception as e:status('stopped',error=type(e).__name__+': '+str(e));raise
if __name__=='__main__':main()
