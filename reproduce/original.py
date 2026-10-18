"""Recreate the original SpeechOcean text-reference ridge selection (not family-v1)."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from .train import load_corpus,save,digest,metrics,CONSTRUCTS
ROOT=Path(__file__).resolve().parents[1]

def fit_candidates(xt,yt,xv,alphas):
 xm,ym=xt.mean(0),yt.mean(0);centered=xt-xm
 values,vectors=eigh(centered@centered.T,check_finite=True);values=np.maximum(values,0);projected=vectors.T@(yt-ym)
 for alpha in alphas:
  dual=vectors@(projected/(values[:,None]+alpha));coef=centered.T@dual;intercept=ym-xm@coef
  yield alpha,coef,intercept,xv@coef+intercept

def run(c,out,metadata):
 out=Path(out);cfg=json.loads((ROOT/'configs/original-speechocean.json').read_text())
 protocol={'version':'portable-original-speechocean-v1','source_protocol':cfg,'manifest_sha256':c['manifest_sha256'],'code_sha256':digest(__file__),'extractor_sha256':digest(metadata)}
 if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Protocol changed')
 save(out/'protocol.json',protocol);tr=c['train'];va=~tr;blocks=c['features'];y=c['targets'];scalers={k:StandardScaler().fit(x[tr]) for k,x in blocks.items()};standardized={k:scalers[k].transform(x)*np.sqrt(2048/x.shape[1]) for k,x in blocks.items()}
 names=('xeus','xlsr','phoneticxeus');subsets=[s for n in range(1,4) for s in itertools.combinations(names,n)];subsets=subsets+[('ipa',)]+[s+('ipa',) for s in subsets];table=[]
 for subset in subsets:
  x=np.concatenate([standardized[k] for k in subset],1);best=None;candidates=[]
  for alpha,coef,intercept,pred in fit_candidates(x[tr],y[tr],x[va],cfg['alphas']):
   scores={d:metrics(y[va,j],pred[:,j]) for j,d in enumerate(CONSTRUCTS)};objective=scores['accuracy']['mae'];candidates.append({'alpha':alpha,'metrics':scores})
   if best is None or objective<best[0]:best=(objective,alpha,coef,intercept,pred,scores)
  _,alpha,coef,intercept,pred,scores=best;folder=out/('+'.join(subset));folder.mkdir(exist_ok=True);cp=folder/'ridge.npz';np.savez(cp,coef=coef,intercept=intercept,scaler_mean=np.concatenate([scalers[k].mean_ for k in subset]),scaler_scale=np.concatenate([scalers[k].scale_ for k in subset]),block_gain=np.concatenate([np.full(blocks[k].shape[1],np.sqrt(2048/blocks[k].shape[1])) for k in subset]))
  m=json.loads(Path(metadata).read_text());m.update(blocks=list(subset),block_widths={k:blocks[k].shape[1] for k in subset},checkpoint_sha256=digest(cp),selected_alpha=alpha,config=cfg,training={'experiment':'portable-original-speechocean-v1','protocol_sha256':digest(out/'protocol.json')});save(cp.with_suffix('.json'),m);save(folder/'selection.json',{'candidates':candidates,'selected_alpha':alpha});np.savez(folder/'development.npz',prediction=pred,target=y[va],speakers=np.array([r['speaker_id'] for r in c['rows']])[va]);table.append({'blocks':list(subset),'alpha':alpha,'metrics':scores});save(out/'comparison.json',table)
 return table

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--extractor',type=Path,default=ROOT/'configs/extractor.json');a=p.parse_args();c=load_corpus(a.manifest)
 if len(c['rows'])!=2500 or c['train'].sum()!=2000:raise ValueError('Expected original 2000/500 cohort')
 run(c,a.output,a.extractor)
if __name__=='__main__':main()
