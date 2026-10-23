"""Portable seven-family ridge reproduction from verified, ordered feature caches."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

FAMILIES = {'speechocean':['speechocean'], 'korean':['korean'], 'aesop':['aesop'],
 'koreanocean':['speechocean','korean'], 'aesopocean':['speechocean','aesop'],
 'koreanaesop':['korean','aesop'], 'aesopkoreanocean':['speechocean','korean','aesop']}
CONSTRUCTS = ('accuracy','fluency','prosody')
FIXED = {'accuracy':['xeus','phoneticxeus','ipa'], 'fluency':['xeus','xlsr'], 'prosody':['xeus','xlsr']}
ALPHAS = [.001,.01,.1,1.,10.,100.]
HISTORICAL_ALPHAS = [a/2000 for a in (.1,1,10,100,1000,10000)]
CONDITIONS = ('spearman','damae','historical_grid_damae')

def select_candidate(candidates,condition):
 if condition not in CONDITIONS:raise ValueError('Unknown selection condition')
 return max(candidates,key=lambda s:(s['spearman'],-s['mae'],s['alpha'])) if condition=='spearman' else min(candidates,key=lambda s:(s['mae'],-s['alpha']))
BLOCKS = ('xeus','xlsr','phoneticxeus','ipa')

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def save(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)

def weighted_fit(x,y,w,alphas,gain):
 x=np.asarray(x,float);y=np.asarray(y,float);w=np.asarray(w,float)
 if y.ndim==1:y=y[:,None]
 if x.ndim!=2 or y.shape[0]!=len(x) or w.shape!=(len(x),):raise ValueError('Shape mismatch')
 if not all(np.isfinite(v).all() for v in (x,y,w,gain)) or np.any(w<=0) or not np.isclose(w.sum(),1):raise ValueError('Invalid weighted fit input')
 if not alphas or any(a<=0 for a in alphas):raise ValueError('Alpha must be positive')
 mean=np.sum(w[:,None]*x,0);scale=np.sqrt(np.sum(w[:,None]*(x-mean)**2,0));scale=np.where(scale<1e-8,1.,scale)
 z=(x-mean)/scale*gain;ym=np.sum(w[:,None]*y,0)
 b=np.sqrt(w[:,None])*z;c=np.sqrt(w[:,None])*(y-ym)
 if b.shape[0]<=b.shape[1]:
  eig,q=np.linalg.eigh(b@b.T);rhs=q.T@c
  coefs={a:b.T@(q@(rhs/(np.maximum(eig,0)[:,None]+a))) for a in alphas}
 else:
  eig,q=np.linalg.eigh(b.T@b);rhs=q.T@(b.T@c)
  coefs={a:q@(rhs/(np.maximum(eig,0)[:,None]+a)) for a in alphas}
 return mean,scale,ym,coefs

def metrics(y,p):
 rho=float(spearmanr(y,p).statistic) if np.ptp(y)>0 and np.ptp(p)>0 else None
 return {'spearman':rho,'mae':float(np.mean(np.abs(y-p))),'n':len(y)}

def load_corpus(path):
 path=Path(path);manifest=json.loads(path.read_text());rows=manifest['rows']
 if manifest['schema_version']!='speech-feature-cache-v1' or manifest['target_order']!=list(CONSTRUCTS):raise ValueError('Unsupported manifest')
 if manifest['target_scale']!='0..10':raise ValueError('Map labels before training')
 if not rows or len({r['recording_id'] for r in rows})!=len(rows):raise ValueError('Duplicate/empty recording IDs')
 if any(r['split'] not in ('train','validation') for r in rows):raise ValueError('Final/private data forbidden')
 if any(not r['speaker_id'] for r in rows):raise ValueError('Missing speaker')
 train=np.array([r['split']=='train' for r in rows]);valid=~train
 if not train.any() or not valid.any():raise ValueError('Both train and validation required')
 for key in ('speaker_id','audio_sha256'):
  if {r[key] for r in rows if r['split']=='train'} & {r[key] for r in rows if r['split']=='validation'}:raise ValueError('Split leakage: '+key)
 y=np.asarray([r['targets'] for r in rows],float)
 if y.shape!=(len(rows),3) or not np.isfinite(y).all() or np.any((y<0)|(y>10)):raise ValueError('Invalid targets')
 fp=path.parent/manifest['features']
 if digest(fp)!=manifest['features_sha256']:raise ValueError('Feature checksum mismatch')
 with np.load(fp,allow_pickle=False) as f:
  if list(f['recording_ids'].astype(str))!=[r['recording_id'] for r in rows]:raise ValueError('Feature row order mismatch')
  features={k:f[k].astype(float) for k in BLOCKS}
 if any(v.ndim!=2 or len(v)!=len(rows) or not np.isfinite(v).all() for v in features.values()):raise ValueError('Invalid features')
 return {'rows':rows,'train':train,'features':features,'targets':y,'manifest_sha256':digest(path)}

def run(corpora,out,mode='fixed',selection='included',extractor=None,condition='spearman'):
 if condition not in CONDITIONS:raise ValueError('Unknown selection condition')
 if condition!='spearman' and (mode!='fixed' or selection!='included'):raise ValueError('DAMAE comparison requires fixed features and included-corpus development selection')
 alphas=HISTORICAL_ALPHAS if condition=='historical_grid_damae' else ALPHAS
 out=Path(out)
 protocol={'schema_version':'portable-ridge-v1','mode':mode,'selection':selection,'condition':condition,'alphas':alphas,'inputs':{n:c['manifest_sha256'] for n,c in corpora.items()},'code_sha256':digest(__file__),'extractor_sha256':digest(extractor) if extractor else None}
 pp=out/'protocol.json'
 if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Output protocol differs; choose a new output directory')
 save(pp,protocol);table=[]
 # Corpus prefixes prevent accidental collision of unrelated local speaker identifiers.
 # Exact audio overlap across corpora is always rejected.
 for a,b in itertools.combinations(corpora,2):
  if {r['audio_sha256'] for r in corpora[a]['rows']} & {r['audio_sha256'] for r in corpora[b]['rows']}:raise ValueError('Cross-corpus audio overlap')
 for family,names in FAMILIES.items():
  for j,construct in enumerate(CONSTRUCTS):
   variants=[FIXED[construct]] if mode=='fixed' else [list(v) for size in range(1,5) for v in itertools.combinations(BLOCKS,size)]
   for blocks in variants:
    widths={k:corpora[names[0]]['features'][k].shape[1] for k in blocks}
    if any(c['features'][k].shape[1]!=widths[k] for c in corpora.values() for k in blocks):raise ValueError('Feature width mismatch')
    raw={n:np.concatenate([c['features'][k] for k in blocks],1) for n,c in corpora.items()}
    x=np.concatenate([raw[n][corpora[n]['train']] for n in names]);y=np.concatenate([corpora[n]['targets'][corpora[n]['train']] for n in names])
    w=np.concatenate([np.full(corpora[n]['train'].sum(),1/(len(names)*corpora[n]['train'].sum())) for n in names])
    gain=np.concatenate([np.full(widths[k],np.sqrt(2048/widths[k])) for k in blocks])
    mean,scale,intercept,coefs=weighted_fit(x,y,w,alphas,gain)
    candidates=[];selection_names=names if selection=='included' else list(corpora)
    selection_index=j if condition=='spearman' else 0
    for alpha,coef in coefs.items():
     scores={n:metrics(corpora[n]['targets'][~corpora[n]['train'],selection_index],(((raw[n][~corpora[n]['train']]-mean)/scale*gain)@coef+intercept)[:,selection_index]) for n in selection_names}
     if any(v['spearman'] is None for v in scores.values()):raise ValueError('Undefined selection Spearman')
     candidates.append({'alpha':alpha,'spearman':float(np.mean([s['spearman'] for s in scores.values()])),'mae':float(np.mean([s['mae'] for s in scores.values()]))})
    best=select_candidate(candidates,condition);coef=coefs[best['alpha']]
    folder=out/family/construct/('+'.join(blocks));folder.mkdir(parents=True,exist_ok=True);cp=folder/'ridge.npz'
    np.savez(cp,coef=coef,intercept=intercept,scaler_mean=mean,scaler_scale=scale,block_gain=gain)
    meta=json.loads(Path(extractor).read_text()) if extractor else {}
    meta.update(format_version=1,checkpoint_type='frozen_feature_fusion_ridge',output_dimensions=list(CONSTRUCTS),blocks=blocks,block_widths=widths,checkpoint_sha256=digest(cp),target_scale='0..10 endpoint-mapped corpus labels',training={'family':family,'selected_construct':construct,'condition':condition,'alpha':best['alpha'],'protocol_sha256':digest(pp)})
    save(cp.with_suffix('.json'),meta);save(folder/'selection.json',{'selected':best,'candidates':candidates})
    for n,c in corpora.items():
     mask=~c['train'];pred=(((raw[n][mask]-mean)/scale*gain)@coef+intercept)[:,j]
     np.savez(folder/(n+'-development.npz'),prediction=pred,target=c['targets'][mask,j],speakers=np.array([r['speaker_id'] for r in c['rows']])[mask])
     table.append({'family':family,'construct':construct,'blocks':'+'.join(blocks),'evaluation_corpus':n,**metrics(c['targets'][mask,j],pred)})
    save(out/'comparison.json',table)
 return table

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('speechocean','korean','aesop'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['fixed','all'],default='fixed');p.add_argument('--selection',choices=['included','all'],default='included');p.add_argument('--extractor',type=Path)
 p.add_argument('--condition',choices=CONDITIONS,default='spearman')
 a=p.parse_args();run({n:load_corpus(getattr(a,n)) for n in ('speechocean','korean','aesop')},a.output,a.mode,a.selection,a.extractor,a.condition)
if __name__=='__main__':main()
