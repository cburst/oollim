import zipfile,io,csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('archive',type=Path)
parser.add_argument('--output',type=Path,default=Path('paired-results.json'))
parser.add_argument('--member',default='stella305_augmented.csv',help='CSV member inside archive')
parser.add_argument('--prefix',default='',help='Prediction column prefix, e.g. spearman__')
parser.add_argument('--bootstrap',type=int,default=200000)
parser.add_argument('--seed',type=int,default=42)
args=parser.parse_args()
source=args.archive
with zipfile.ZipFile(source) as z:
 raw=z.read(args.member);rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
families=['speechocean','korean','aesop','koreanocean','aesopocean','koreanaesop','aesopkoreanocean'];systems=['azure']+families
assert len(rows)==305 and len({r['evaluation_drive_id'] for r in rows})==305
B=args.bootstrap; out=[]; scales=[];pointcheck=[]
def metrics(a,b):
 ra=rankdata(a,axis=-1);rb=rankdata(b,axis=-1);ra-=ra.mean(-1,keepdims=True);rb-=rb.mean(-1,keepdims=True)
 rho=(ra*rb).sum(-1)/np.sqrt((ra*ra).sum(-1)*(rb*rb).sum(-1))
 qa=np.floor(np.clip(a,0,100)+.5);qb=np.floor(np.clip(b,0,100)+.5)
 den=(qa*qa).mean(-1)+(qb*qb).mean(-1)-2*qa.mean(-1)*qb.mean(-1)
 return np.stack([rho,1-((qa-qb)**2).mean(-1)/den],axis=-1)
for c in ['Accuracy','Fluency','Prosody']:
 h=np.array([[float(r[f'h{j}-{c}Score']) for r in rows] for j in (1,2,3)])
 pred=np.array([[float(r[f'm-{c}Score' if s=='azure' else f'{args.prefix}{s}-{c}Score100']) for r in rows] for s in systems])
 ref=np.array([np.mean(np.delete(h,j,axis=0),axis=0) for j in range(3)])
 def evaluate(idx):
  human=metrics(h[:,idx],ref[:,idx]).mean(0)
  machine=np.mean([metrics(pred[:,idx],ref[j,idx][None,...]) for j in range(3)],axis=0)
  return machine,human
 point,hpoint=evaluate(np.arange(305));draws=[];rng=np.random.default_rng(args.seed)
 for start in range(0,B,100):
  ix=rng.integers(0,305,(min(100,B-start),305));m,hu=evaluate(ix)
  draws.append(np.stack([m-hu[None,...],m-m[0:1]],axis=-1))
 draws=np.concatenate(draws,axis=1)
 for si,s in enumerate(systems):
  pointcheck.append(dict(construct=c,system=s,rho=point[si,0],qwk=point[si,1]))
  scales.append(dict(construct=c,system=s,mean=float(pred[si].mean()),sd=float(pred[si].std(ddof=1)),bias_vs_mean_three=float((pred[si]-h.mean(0)).mean()),mae_vs_mean_three=float(abs(pred[si]-h.mean(0)).mean())))
  for ri,refname in enumerate(['human_other_two','azure_two_rater']):
   if si==0 and ri==1:continue
   for mi,metric in enumerate(['rho','qwk']):
    d=draws[si,:,mi,ri];delta=point[si,mi]-(hpoint[mi] if ri==0 else point[0,mi])
    # 48 differences vs humans, 42 ridge-Azure differences = 90 total
    lo,hi=np.quantile(d,[.025,.975]);al,ah=np.quantile(d,[.025/90,1-.025/90])
    out.append(dict(construct=c,system=s,reference=refname,metric=metric,difference=float(delta),ci95=[float(lo),float(hi)],familywise95_bonferroni90=[float(al),float(ah)]))
 print(c,'done',flush=True)
result=dict(prediction_prefix=args.prefix,bootstrap=B,seed=args.seed,n=305,grouping='one unique recording URL per assumed speaker',source_sha256=hashlib.sha256(raw).hexdigest(),reference='mean of three pair-specific metrics; humans evaluated against other two',comparisons=out,scale_summaries=scales,point_estimates=pointcheck)
args.output.write_text(json.dumps(result,indent=2))
