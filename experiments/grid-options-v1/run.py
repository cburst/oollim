import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,time,itertools
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from reproduce import train as t
OUT=Path(os.environ.get('OOLLIM_GRID_OUTPUT', str(ROOT/'work/grid-options-v1')));CACHE=Path(os.environ.get('OOLLIM_FEATURE_CACHE', str(ROOT/'work/features')))
WIDE=sorted(set([float(10**v) for v in np.arange(-6,3.01,.5)]+t.ALPHAS+t.HISTORICAL_ALPHAS))
RULES=('spearman','accuracy_mae')
def qwk(y,p):
 a=np.floor(np.clip(y*10,0,100)+.5);b=np.floor(np.clip(p*10,0,100)+.5)
 d=np.mean(a*a)+np.mean(b*b)-2*np.mean(a)*np.mean(b)
 return float(1-np.mean((a-b)**2)/d) if d>0 else None

def choose(candidates,rule,j):
 k=j if rule=='spearman' else 0
 if rule=='spearman':return max(candidates,key=lambda x:(x['metrics'][k]['spearman'],-x['metrics'][k]['mae'],x['ratio']==1,x['alpha']))
 return min(candidates,key=lambda x:(x['metrics'][k]['mae'],x['ratio']!=1,-x['alpha']))

def run():
 OUT.mkdir(parents=True,exist_ok=True)
 cs={n:t.load_corpus(CACHE/n/'manifest.json') for n in t.FAMILIES if n in ('speechocean','korean','aesop')}
 protocol=dict(version='grid-options-v1',expanded=WIDE,inputs={n:c['manifest_sha256'] for n,c in cs.items()},protocol_sha256=t.digest(Path(__file__).with_name('PROTOCOL.md')),code_sha256=t.digest(__file__))
 pp=OUT/'protocol.json'
 if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Protocol changed; use a new output directory')
 t.save(pp,protocol)
 for a,b in itertools.combinations(cs,2):assert not {r['audio_sha256'] for r in cs[a]['rows']} & {r['audio_sha256'] for r in cs[b]['rows']}
 table=[];registry={}
 for family,names in t.FAMILIES.items():
  for blocks,dims in [(t.FIXED['accuracy'],[0]),(t.FIXED['fluency'],[1,2])]:
   tag=family+'-'+ '+'.join(blocks);receipt=OUT/'completed'/(tag+'.json')
   if receipt.exists():
    done=json.loads(receipt.read_text())
    for path,sha in done['artifacts'].items():assert t.digest(OUT/path)==sha
    table.extend(done['rows']);registry.update(done['registry']);continue
   t.save(OUT/'status.json',dict(stage='fitting',family=family,blocks=blocks,updated=time.time()))
   widths={b:cs[names[0]]['features'][b].shape[1] for b in blocks}
   raw={n:np.concatenate([c['features'][b] for b in blocks],axis=1) for n,c in cs.items()}
   x=np.concatenate([raw[n][cs[n]['train']] for n in names]);y=np.concatenate([cs[n]['targets'][cs[n]['train']] for n in names])
   w=np.concatenate([np.full(cs[n]['train'].sum(),1/(len(names)*cs[n]['train'].sum())) for n in names])
   basegain=np.concatenate([np.full(widths[b],np.sqrt(2048/widths[b])) for b in blocks])
   paths={};candidates=[]
   def fit(alphas,ratio):
    gain=basegain.copy()
    if ratio!=1:
     gain[:widths['xeus']]*=ratio**.25;gain[widths['xeus']:]*=ratio**-.25
    mean,scale,intercept,coefs=t.weighted_fit(x,y,w,alphas,gain)
    for alpha,coef in coefs.items():
     pred={n:((raw[n][~c['train']]-mean)/scale*gain)@coef+intercept for n,c in cs.items()}
     scores=[]
     for j in range(3):
      vals=[t.metrics(cs[n]['targets'][~cs[n]['train'],j],pred[n][:,j]) for n in names]
      assert all(z['spearman'] is not None for z in vals)
      scores.append({k:float(np.mean([v[k] for v in vals])) for k in ('spearman','mae')})
     c=dict(alpha=alpha,ratio=ratio,metrics=scores);candidates.append(c)
     paths[alpha,ratio]=(mean,scale,intercept,coef,gain,pred)
   fit(WIDE,1.)
   selected={}
   for rule in RULES:
    for j in dims:selected['expanded',rule,j]=choose(candidates,rule,j)
   fine=set()
   for c in selected.values():fine.update(float(v) for v in c['alpha']*np.logspace(-.5,.5,9))
   fine=sorted(fine-set(WIDE))
   if fine:fit(fine,1.)
   # Each refined selection may use only its own pre-specified local neighborhood.
   for rule in RULES:
    for j in dims:
     center=selected['expanded',rule,j]['alpha'];allowed=set(WIDE)|set(float(v) for v in center*np.logspace(-.5,.5,9))
     selected['refined',rule,j]=choose([c for c in candidates if c['alpha'] in allowed],rule,j)
   if dims==[1,2]:
    for ratio in (.25,4.):fit(WIDE,ratio)
   for rule in RULES:
    for j in dims:
     eligible=[c for c in candidates if c['alpha'] in WIDE]+[selected['refined',rule,j]]
     selected['banded',rule,j]=choose(eligible,rule,j) if j else selected['refined',rule,j]
   rows=[];reg={};artifacts={}
   t.save(OUT/'candidates'/(tag+'.json'),candidates)
   for (option,rule,j),best in selected.items():
    construct=t.CONSTRUCTS[j];mean,scale,intercept,coef,gain,pred=paths[best['alpha'],best['ratio']]
    key='__'.join((option,rule,family));folder=OUT/'models'/key;folder.mkdir(parents=True,exist_ok=True);cp=folder/(construct+'.npz')
    np.savez(cp,coef=coef,intercept=intercept,scaler_mean=mean,scaler_scale=scale,block_gain=gain)
    meta=json.loads((ROOT/'configs/extractor.json').read_text());meta.update(format_version=1,checkpoint_type='frozen_feature_fusion_ridge',blocks=blocks,block_widths=widths,checkpoint_sha256=t.digest(cp),target_scale='0..10 endpoint-mapped corpus labels',training=dict(family=family,option=option,selection=rule,selected_construct=construct,alpha=best['alpha'],penalty_ratio_xlsr_over_xeus=best['ratio'],accuracy_carried_from_refined=(option=='banded' and j==0),protocol_sha256=t.digest(pp)))
    t.save(cp.with_suffix('.json'),meta)
    for f in (cp,cp.with_suffix('.json')):artifacts[str(f.relative_to(OUT))]=t.digest(f)
    reg.setdefault(key,{})[construct]=str(cp.relative_to(OUT))
    for n,c in cs.items():
     yy=c['targets'][~c['train'],j];pr=pred[n][:,j]
     rows.append(dict(option=option,selection=rule,family=family,construct=construct,alpha=best['alpha'],ratio=best['ratio'],evaluation_corpus=n,**t.metrics(yy,pr),qwk=qwk(yy,pr)))
    np.savez(folder/(construct+'-public-dev.npz'),**{n:pred[n][:,j] for n in cs})
   t.save(receipt,dict(rows=rows,registry=reg,artifacts=artifacts));table.extend(rows)
   for k,v in reg.items():registry.setdefault(k,{}).update(v)
   t.save(OUT/'comparison.json',table);t.save(OUT/'registry.json',registry)
 # Rebuild registry to merge both feature groups on resume as well.
 registry={}
 for receipt in (OUT/'completed').glob('*.json'):
  for k,v in json.loads(receipt.read_text())['registry'].items():registry.setdefault(k,{}).update(v)
 assert len(registry)==42 and all(set(v)==set(t.CONSTRUCTS) for v in registry.values())
 t.save(OUT/'registry.json',registry);t.save(OUT/'comparison.json',table)
 t.save(OUT/'status.json',dict(stage='completed',heads=126,configurations=42,comparison_rows=len(table),updated=time.time()))
if __name__=='__main__':
 import fcntl,traceback
 OUT.mkdir(parents=True,exist_ok=True)
 lock=(OUT/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  with threadpool_limits(limits=2):run()
 except Exception as e:t.save(OUT/'status.json',dict(stage='failed',error=str(e),updated=time.time()));traceback.print_exc();raise
