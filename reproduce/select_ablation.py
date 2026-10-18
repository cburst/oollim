"""Paired public-development sensitivity analysis; never reads private evaluation data."""
import json,itertools
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import argparse,collections
parser=argparse.ArgumentParser(description="Apply the recorded .01 promotion rule to a completed public-development sweep.")
parser.add_argument('--results',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
ROOT=args.results
raw=json.loads((ROOT/'comparison.json').read_text())
groups=collections.defaultdict(list)
for r in raw:groups[r['family'],r['construct'],r['blocks']].append(r)
if len(raw)!=945 or len(groups)!=315 or any(len(v)!=3 or {r['evaluation_corpus'] for r in v}!={'speechocean','korean','aesop'} for v in groups.values()):raise ValueError('Incomplete or duplicate sweep')
rows=[dict(family=f,construct=c,blocks=b,mean_rho=float(np.mean([r['spearman'] for r in rs])),mean_mae=float(np.mean([r['mae'] for r in rs])),corpora={r['evaluation_corpus']:r for r in rs}) for (f,c,b),rs in groups.items()]
args.output.mkdir(parents=True,exist_ok=True)
(args.output/'leaderboard.json').write_text(json.dumps(rows,indent=2))
out=[]
def rho(y,p):
 a=rankdata(y,axis=1);b=rankdata(p,axis=1);a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
 return (a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1))
def boot(candidate,base):
 rng=np.random.default_rng(42);diff=np.zeros(2000)
 for corpus in ('speechocean','korean','aesop'):
  folder=ROOT/candidate['family']/candidate['construct']
  with np.load(folder/candidate['blocks']/(corpus+'-development.npz')) as f:y=f['target'];p=f['prediction'];s=f['speakers']
  with np.load(folder/base['blocks']/(corpus+'-development.npz')) as f:
   assert np.array_equal(y,f['target']) and np.array_equal(s,f['speakers']);q=f['prediction']
  groups=[np.flatnonzero(s==v) for v in np.unique(s)]
  # Cluster resampling accommodates unequal recordings per speaker.
  vals=[]
  for i in range(2000):
   ix=np.concatenate([groups[k] for k in rng.integers(len(groups),size=len(groups))])
   vals.append(float(rho(y[ix][None,:],p[ix][None,:])[0]-rho(y[ix][None,:],q[ix][None,:])[0]))
  diff+=np.asarray(vals)/3
 if not np.isfinite(diff).all():raise ValueError('Undefined bootstrap correlation; investigate before promotion')
 return np.quantile(diff,[.025,.975]).tolist()
for family,construct in dict.fromkeys((r['family'],r['construct']) for r in rows):
 rs=sorted([r for r in rows if r['family']==family and r['construct']==construct],key=lambda r:(-r['mean_rho'],r['mean_mae']))
 base=next(r for r in rs if r['blocks']==('xeus+phoneticxeus+ipa' if construct=='accuracy' else 'xeus+xlsr'))
 checks=[];eligible=[]
 for r in rs:
  if rs[0]['mean_rho']-r['mean_rho']>.01:continue
  delta=r['mean_rho']-base['mean_rho'];rd=[r['corpora'][n]['spearman']-base['corpora'][n]['spearman'] for n in r['corpora']];md=[r['corpora'][n]['mae']-base['corpora'][n]['mae'] for n in r['corpora']]
  gate=delta>=.01 and min(rd)>=-.02 and max(md)<=.10
  ci=boot(r,base) if gate else None
  accepted=gate and ci[0]>0
  checks.append(dict(blocks=r['blocks'],delta=delta,min_corpus_rho_delta=min(rd),max_corpus_mae_delta=max(md),ci=ci,passes=accepted))
  if accepted:eligible.append(r)
 selected=eligible[0] if eligible else base
 out.append(dict(family=family,construct=construct,selected=selected['blocks'],mean_rho=selected['mean_rho'],promoted=bool(eligible),numerical_best=rs[0]['blocks'],checks=checks))
 print(family,construct,selected['blocks'],round(selected['mean_rho'],4),'PROMOTE' if eligible else 'RETAIN',flush=True)
 (args.output/'decisions.json').write_text(json.dumps(out,indent=2))
