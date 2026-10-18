"""Versioned CPU-only, anchored hybrid ridge; never opens final/private data."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,copy,fcntl,gc,hashlib,io,json,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from pronunciation.encoders import sha256,FrameBatch
from pronunciation.fusion import UnifiedScorer
from pronunciation.scoring import masked_pool
from pronunciation.audio import read_audio,batch_audio
from pronunciation.phonetic import PhoneticXeus
from pronunciation.phone_features import phone_evidence
from pronunciation.text_reference import text_to_phones,load_dictionary
from pronunciation.development import metrics,paired_speaker_bootstrap
DIMS=('accuracy','fluency','prosody')

def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def np_save(p,**arrays):
 tmp=p.with_suffix('.tmp')
 with tmp.open('wb') as f:np.savez(f,**arrays)
 tmp.replace(p)
def check(ok,message):
 if not ok:raise ValueError(message)
def mapped(y):
 y=np.asarray(y,float);check(y.ndim==2 and y.shape[1]==2 and np.isfinite(y).all() and ((y>=1)&(y<=5)).all(),'Invalid Korean labels')
 return 2.5*(y[:,[0,1,1]]-1)
def transforms(raw,arrays):return (raw-arrays['scaler_mean'])/arrays['scaler_scale']*arrays['block_gain']
def anchored_candidates(z,y,source,coef,intercept,lambdas):
 z=np.asarray(z,float);y=np.asarray(y,float);source=np.asarray(source)
 check(set(source.tolist())=={0,1},'Both training corpora required')
 weights=np.array([.5/np.sum(source==s) for s in source])
 a=np.column_stack([z,np.ones(len(z))]);residual=y-(z@coef+intercept)
 gram=a.T@(weights[:,None]*a);rhs=a.T@(weights*residual)
 eig,q=np.linalg.eigh(gram);eig=np.maximum(eig,0);project=q.T@rhs
 for alpha in lambdas:
  check(alpha>0,'Positive penalty required');delta=q@(project/(eig+alpha))
  yield alpha,coef+delta[:-1],float(intercept+delta[-1])
def score(y,p):
 m=metrics(y,p);check(m['spearman'] is not None,'Undefined development Spearman');return m
def choose(candidates,baseline):
 feasible=[c for c in candidates if c['speechocean']['spearman']>=baseline['speechocean']['spearman']-.02 and c['speechocean']['mae']<=baseline['speechocean']['mae']+.2]
 if not feasible:return None
 return max(feasible,key=lambda c:(np.mean([c[s]['spearman'] for s in ('speechocean','korean')]),-np.mean([c[s]['mae'] for s in ('speechocean','korean')]),c['lambda']))

def prepare(cfg,out):
 manifest=ROOT/cfg['korean_manifest'];check(sha256(manifest)==cfg['korean_manifest_sha256'],'Korean manifest changed')
 dictionary=load_dictionary(ROOT/'resources/text-reference-v1/cmudict.dict')
 original=json.loads(manifest.read_text());kept=[];excluded=[]
 for row in original:
  check(row['split'] in ('train','validation'),'Final data forbidden')
  _,st=text_to_phones(row['reference_text'],dictionary)
  reason='duration_gt_20' if row['duration_s']>cfg['korean_max_seconds'] else 'incomplete_reference' if not st['complete'] else None
  if reason:excluded.append({'recording_id':row['recording_id'],'split':row['split'],'reason':reason})
  else:kept.append(row)
 for key in ('speaker_group','decoded_sha256'):
  a={r[key] for r in kept if r['split']=='train'};b={r[key] for r in kept if r['split']=='validation'};check(not a&b,'Korean split overlap')
 mapped([r['targets'] for r in kept])
 save(out/'korean-manifest.json',kept);save(out/'excluded.json',excluded)
 save(out/'cohort.json',{'selected':{s:sum(r['split']==s for r in kept) for s in ('train','validation')},'excluded':len(excluded),'mapping':cfg['mapping'],'private_or_final_used':False})
 features={}
 for name in ('xeus','xlsr'):
  path=manifest.parent/name;contract=json.loads((path/'contract.json').read_text());check(contract['manifest_sha256']==sha256(manifest),'Cache manifest changed')
  expected=json.loads((ROOT/cfg['baseline']['fluency']['path']).with_suffix('.json').read_text())['experiment_provenance']['backbones'][name]['backbone']
  for k in expected:
   if k!='directory':check(expected[k]==contract['backbone'][k],'Encoder configuration differs: '+name+':'+k)
  vectors=[]
  for row in kept:
   cp=path/(hashlib.sha256(row['recording_id'].encode()).hexdigest()+'.npz');receipt=json.loads(cp.with_suffix('.json').read_text())
   check(sha256(cp)==receipt['sha256'] and receipt['contract_sha256']==sha256(path/'contract.json'),'Cache checksum mismatch')
   check(receipt['audio_sha256']==row['audio_sha256'] and receipt['decoded_sha256']==row['decoded_sha256'],'Cache audio mismatch')
   check(receipt['reference_sha256']==hashlib.sha256(row['reference_text'].encode()).hexdigest(),'Reference mismatch')
   with np.load(cp,allow_pickle=False) as f:
    check(f['valid_mask'].all() and np.all(f['chunk_start_samples']==0),'Expected unchunked valid frames')
    frame=torch.from_numpy(f['frames'].copy())[None]
    # Reuse frames but use the exact deployed pooling arithmetic.
    fb=FrameBatch(frame,torch.tensor([frame.shape[1]]),torch.ones(frame.shape[:2],dtype=torch.bool),16000,320,400)
    vectors.append(masked_pool(fb)[0].numpy())
  features[name]=np.stack(vectors)
 np_save(out/'korean-base-features.npz',**features)
 return kept,dictionary,features

def phonetics(rows,dictionary,out,status):
 target=out/'phonetic-cache';target.mkdir(exist_ok=True)
 model=None;vectors=[];ipas=[]
 contract={'adapter':sha256(ROOT/'pronunciation/phonetic.py'),'pooling':sha256(ROOT/'pronunciation/scoring.py'),'phone':sha256(ROOT/'pronunciation/phone_features.py'),'dictionary':sha256(ROOT/'resources/text-reference-v1/cmudict.dict'),'text':sha256(ROOT/'pronunciation/text_reference.py'),'model_sha256':'ad58bf20a60e9d0380327bd8b2d0e8e90a9b8de2adccbfb479f9b21ea85eda18','context':'whole recording <=20s CPU float32'}
 cp=target/'contract.json'
 if cp.exists():check(json.loads(cp.read_text())==contract,'Phonetic contract changed')
 else:save(cp,contract)
 archive=ROOT/'resources/dropbox-download-v3/TS_PRN_en_NA_NA.zip'
 # Per-member encoded and decoded hashes checked against audited manifest.
 with zipfile.ZipFile(archive) as z:
  for i,row in enumerate(rows):
   path=target/(hashlib.sha256(row['recording_id'].encode()).hexdigest()+'.npz');receipt=path.with_suffix('.json')
   if receipt.exists():
    r=json.loads(receipt.read_text());check(r['sha256']==sha256(path) and r['contract_sha256']==sha256(cp) and r['audio_sha256']==row['audio_sha256'] and r['reference_sha256']==hashlib.sha256(row['reference_text'].encode()).hexdigest(),'Stale phonetic cache')
   else:
    if model is None:model=PhoneticXeus().eval()
    raw=z.read(row['audio_member']);check(hashlib.sha256(raw).hexdigest()==row['audio_sha256'],'Audio bytes changed')
    wave=read_audio(io.BytesIO(raw));check(hashlib.sha256(wave.numpy().tobytes()).hexdigest()==row['decoded_sha256'],'Decoded audio changed')
    with torch.inference_mode():
     result=model(batch_audio([wave]))[0];frames=result.frames[None]
     fb=FrameBatch(frames,torch.tensor([frames.shape[1]]),torch.ones(frames.shape[:2],dtype=torch.bool),16000,result.stride_samples,400)
     pooled=masked_pool(fb)[0].numpy();phones,st=text_to_phones(row['reference_text'],dictionary)
     ipa,coverage=phone_evidence(result.logits.numpy(),result.vocabulary,result.blank_id,phones,len(wave)/16000)
     check(st['complete'] and coverage['reference_complete'],'Incomplete phone reference')
     np_save(path,pooled=pooled,ipa=ipa,frames=result.frames.numpy(),logits=result.logits.numpy())
    save(receipt,{'sha256':sha256(path),'contract_sha256':sha256(cp),'audio_sha256':row['audio_sha256'],'reference_sha256':hashlib.sha256(row['reference_text'].encode()).hexdigest()})
    del result,frames,fb;gc.collect()
   with np.load(path,allow_pickle=False) as f:vectors.append(f['pooled']);ipas.append(f['ipa'])
   status(stage='extracting_phonetic',done=i+1,total=len(rows))
 if model is not None:del model;gc.collect()
 return {'phoneticxeus':np.stack(vectors),'ipa':np.stack(ipas)}

def train_head(dim,cfg,out,so,ko,sf,kf):
 j=DIMS.index(dim);asset=cfg['baseline'][dim];path=ROOT/asset['path'];check(sha256(path)==asset['sha256'] and sha256(path.with_suffix('.json'))==asset['metadata_sha256'],'Baseline changed')
 model=UnifiedScorer(path);arr=model.arrays
 sx=np.concatenate([sf[k] for k in model.blocks],axis=1);kx=np.concatenate([kf[k] for k in model.blocks],axis=1)
 sz,kz=transforms(sx,arr),transforms(kx,arr)
 sy=np.array([r['targets'][j] for r in so]);ky=mapped([r['targets'] for r in ko])[:,j]
 st=np.array([r['split']=='train' for r in so]);kt=np.array([r['split']=='train' for r in ko])
 basecoef=arr['coef'][:,j];baseint=arr['intercept'][j]
 baseline={'speechocean':score(sy[~st],sz[~st]@basecoef+baseint),'korean':score(ky[~kt],kz[~kt]@basecoef+baseint)}
 z=np.concatenate([sz[st],kz[kt]]);y=np.concatenate([sy[st],ky[kt]]);source=np.r_[np.zeros(st.sum(),int),np.ones(kt.sum(),int)]
 candidates=[];weights={}
 for alpha,coef,intercept in anchored_candidates(z,y,source,basecoef,baseint,cfg['lambdas']):
  candidates.append({'lambda':alpha,'speechocean':score(sy[~st],sz[~st]@coef+intercept),'korean':score(ky[~kt],kz[~kt]@coef+intercept)})
  weights[alpha]=(coef,intercept)
 selected=choose(candidates,baseline)
 result={'construct':dim,'blocks':model.blocks,'baseline':baseline,'candidates':candidates,'selected':selected,'counts':{'speechocean_train':int(st.sum()),'speechocean_dev':int((~st).sum()),'korean_train':int(kt.sum()),'korean_dev':int((~kt).sum())},'evaluation':'development; same coefficients across corpora; no private/final data'}
 if selected is not None:
  coef,intercept=weights[selected['lambda']]
  arrays={k:v.copy() for k,v in arr.items()};arrays['coef'][:,j]=coef;arrays['intercept'][j]=intercept
  dest=out/(dim+'-hybrid.npz');np_save(dest,**arrays)
  metadata=copy.deepcopy(model.metadata);metadata['checkpoint_sha256']=sha256(dest)
  metadata['baseline_config']=metadata.pop('config',{});metadata['baseline_selected_alpha']=metadata.pop('selected_alpha',None);metadata['target_scale']='0..10; SpeechOcean native and endpoint-mapped Korean';metadata['training']={'experiment':'koreanocean-v1','source_baseline':asset,'protocol_sha256':sha256(out/'protocol.json'),'selected_construct':dim,'other_columns':'unchanged baseline; use only selected_construct','selection':'development selected; not independent test','lambda':selected['lambda'],'mapping':cfg['mapping']}
  save(dest.with_suffix('.json'),metadata);restored=UnifiedScorer(dest)
  result['uncertainty']={}
  for name,raw,x,target,mask,rows in [('speechocean',sx,sz,sy,~st,so),('korean',kx,kz,ky,~kt,ko)]:
   pred=x[mask]@coef+intercept;np.testing.assert_allclose(restored.predict_features(raw[mask])[:,j],pred,rtol=1e-10,atol=1e-10)
   baseline_pred=x[mask]@basecoef+baseint
   speakers=np.array([r.get('speaker_group',r.get('speaker_id')) for r in rows])[mask]
   result['uncertainty'][name]=paired_speaker_bootstrap(target[mask],pred,baseline_pred,speakers,seed=cfg['seed'],replicates=cfg['bootstrap'])
   np_save(out/(dim+'-'+name+'-dev-predictions.npz'),prediction=pred,baseline=baseline_pred,target=target[mask],speakers=speakers)
  result['promotable']=bool(np.mean([selected[s]['spearman']-baseline[s]['spearman'] for s in ('speechocean','korean')])>=.02 and all(selected[s]['spearman']>=baseline[s]['spearman']-.02 and selected[s]['mae']<=baseline[s]['mae']+.2 for s in ('speechocean','korean')))
  result['checkpoint_sha256']=sha256(dest)
 save(out/(dim+'-results.json'),result)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',default='experiments/koreanocean-v1.json');args=p.parse_args()
 cfg=json.loads((ROOT/args.protocol).read_text());out=ROOT/'resources'/cfg['experiment_id'];out.mkdir(exist_ok=True)
 lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(2)
 code={'runner':sha256(__file__),'fusion':sha256(ROOT/'pronunciation/fusion.py'),'phonetic':sha256(ROOT/'pronunciation/phonetic.py'),'pooling':sha256(ROOT/'pronunciation/scoring.py')}
 protocol=dict(cfg,code_hashes=code)
 if (out/'protocol.json').exists():check(json.loads((out/'protocol.json').read_text())==protocol,'Protocol or code changed; version the experiment')
 else:save(out/'protocol.json',protocol)
 def status(**v):save(out/'status.json',dict(v,updated_unix=time.time(),pid=os.getpid()))
 try:
  status(stage='preparing')
  for key in ('source_features','source_manifest'):check(sha256(ROOT/cfg[key])==cfg[key+'_sha256'],'Source checksum mismatch')
  so=json.loads((ROOT/cfg['source_manifest']).read_text());check(len(so)==2500 and sum(r['split']=='train' for r in so)==2000 and all(r['split'] in ('train','validation') for r in so),'Invalid SpeechOcean cohort')
  check(not {r['speaker_id'] for r in so if r['split']=='train'}&{r['speaker_id'] for r in so if r['split']=='validation'},'SO speaker overlap')
  with np.load(ROOT/cfg['source_features'],allow_pickle=False) as f:
   check(f['recording_ids'].tolist()==[r['recording_id'] for r in so],'SO feature order mismatch');sf={k:f[k].copy() for k in ('xeus','xlsr','phoneticxeus','ipa')}
  ko,dictionary,kf=prepare(cfg,out);results={}
  with threadpool_limits(limits=2):
   for dim in ('fluency','prosody'):
    status(stage='training',construct=dim);results[dim]=train_head(dim,cfg,out,so,ko,sf,kf)
   kf.update(phonetics(ko,dictionary,out,status))
   status(stage='training',construct='accuracy');results['accuracy']=train_head('accuracy',cfg,out,so,ko,sf,kf)
  table=['# Koreanocean v1 development comparison','', 'Same hybrid coefficients on both corpora; development-selected, not final-test evidence. Korean fluency/prosody share one source label.','', '| Construct | Inputs | SpeechOcean rho | Mapped Korean rho | SO MAE | Korean MAE |','|---|---|---:|---:|---:|---:|']
  for dim in DIMS:
   r=results[dim];c=r['selected']
   if c:table.append(f"| {dim} | {'+'.join(r['blocks'])} | {c['speechocean']['spearman']:.5f} | {c['korean']['spearman']:.5f} | {c['speechocean']['mae']:.5f} | {c['korean']['mae']:.5f} |")
   else:table.append(f'| {dim} | No feasible hybrid | — | — | — | — |')
  (out/'RESULTS.md').write_text('\n'.join(table)+'\n');save(out/'results.json',results);status(stage='completed',table='RESULTS.md')
 except Exception as e:
  status(stage='failed',error_type=type(e).__name__,message=str(e));raise
if __name__=='__main__':main()
