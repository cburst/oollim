"""Paired speaker bootstrap for the 21 S/D/H models; exports aggregates only.

Frequency-weighted midranks exactly reproduce ranking the resampled observations,
including ties. The input cohort must contain one recording per unique speaker ID.
"""
import argparse, csv, hashlib, io, json, time, zipfile
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

FAMILIES = ('speechocean','korean','aesop','koreanocean','aesopocean','koreanaesop','aesopkoreanocean')
RULES = ('spearman','damae','historical_grid_damae')
SYSTEMS = ('azure',) + tuple(r+'__'+f for r in RULES for f in FAMILIES)
CONSTRUCTS = ('Accuracy','Fluency','Prosody')
COMPARISONS = (len(SYSTEMS) + len(SYSTEMS)-1) * 3 * 2

class WeightedMetrics:
    def __init__(self, values):
        self.v = np.asarray(values, dtype=float)
        if not np.isfinite(self.v).all(): raise ValueError('Nonfinite evaluation scores')
        self.orders=[]; self.starts=[]; self.groups=[]
        for row in self.v:
            order=np.argsort(row,kind='stable'); sorted_row=row[order]
            starts=np.r_[0, np.flatnonzero(np.diff(sorted_row))+1]
            group=np.empty(len(row),dtype=int)
            group[order]=np.cumsum(np.r_[True, np.diff(sorted_row)!=0])-1
            self.orders.append(order);self.starts.append(starts);self.groups.append(group)
        self.q=np.floor(np.clip(self.v,0,100)+.5)
        self.nsystems=len(SYSTEMS)
        # Predictions and individual humans against each two-human reference.
        m=self.nsystems
        self.pairs=[(i,m+3+j) for i in range(m+3) for j in range(3)]
        self.cross=np.array([self.q[a]*self.q[b] for a,b in self.pairs]).T

    def evaluate(self, counts):
        counts=np.asarray(counts,dtype=float); totals=counts.sum(1)
        b=len(counts);m=self.nsystems
        ranks=np.empty((b,len(self.v),self.v.shape[1]))
        for i,(order,starts,groups) in enumerate(zip(self.orders,self.starts,self.groups)):
            weights=np.add.reduceat(counts[:,order],starts,axis=1)
            mid=np.cumsum(weights,axis=1)-weights/2
            ranks[:,i,:]=mid[:,groups]-totals[:,None]/2
        norms=np.sqrt(np.einsum('bvn,bvn,bn->bv',ranks,ranks,counts,optimize=True))
        ranks=np.divide(ranks,norms[:,:,None],out=np.full_like(ranks,np.nan),where=norms[:,:,None]>0)
        rho=np.einsum('bvn,brn,bn->bvr',ranks[:,:m+3],ranks[:,m+3:],counts,optimize=True)
        means=counts@self.q.T/totals[:,None]
        seconds=counts@(self.q*self.q).T/totals[:,None]
        cross=counts@self.cross/totals[:,None]
        aa=np.array([a for a,b in self.pairs]);bb=np.array([b for a,b in self.pairs])
        observed=seconds[:,aa]+seconds[:,bb]-2*cross
        expected=seconds[:,aa]+seconds[:,bb]-2*means[:,aa]*means[:,bb]
        qwk=np.divide(observed,expected,out=np.full_like(observed,np.nan),where=expected>0)
        qwk=(1-qwk).reshape(b,m+3,3)
        machine=np.stack([rho[:,:m].mean(2),qwk[:,:m].mean(2)],axis=-1)
        human=np.stack([rho[:,m+np.arange(3),np.arange(3)].mean(1),qwk[:,m+np.arange(3),np.arange(3)].mean(1)],axis=-1)
        return machine,human

def load(archive):
    with zipfile.ZipFile(archive) as z:raw=z.read('stella305_all_selections.csv')
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    if len(rows)!=305 or len({r['evaluation_drive_id'] for r in rows})!=305:raise ValueError('Requires 305 unique nonempty speaker IDs')
    if any(not r['evaluation_drive_id'].strip() for r in rows):raise ValueError('Empty ID')
    return raw,rows

def values(rows,c):
    human=np.array([[float(r[f'h{j}-{c}Score']) for r in rows] for j in (1,2,3)])
    pred=np.array([[float(r[f'm-{c}Score' if s=='azure' else f'{s}-{c}Score100']) for r in rows] for s in SYSTEMS])
    refs=np.array([np.delete(human,j,axis=0).mean(0) for j in range(3)])
    return np.concatenate([pred,human,refs])

def run(archive,output,draws=600000,seed=42,batch=512):
    raw,rows=load(archive);out=[];points=[];n=len(rows);start=time.monotonic()
    for c in CONSTRUCTS:
        engine=WeightedMetrics(values(rows,c));point,hpoint=engine.evaluate(np.ones((1,n)))
        machine=np.empty((draws,len(SYSTEMS),2));human=np.empty((draws,2))
        rng=np.random.default_rng(seed)
        for k in range(0,draws,batch):
            size=min(batch,draws-k);indices=rng.integers(0,n,(size,n))
            counts=np.zeros((size,n),dtype=float)
            np.add.at(counts,(np.arange(size)[:,None],indices),1)
            machine[k:k+size],human[k:k+size]=engine.evaluate(counts)
            if k% (batch*100)==0:print(c,k,'/',draws,'elapsed',round(time.monotonic()-start),flush=True)
        if not np.isfinite(machine).all() or not np.isfinite(human).all():raise ValueError('Undefined bootstrap metric; no silent replicate exclusion')
        for i,s in enumerate(SYSTEMS):
            points.append(dict(construct=c,system=s,rho=float(point[0,i,0]),qwk=float(point[0,i,1])))
            for reference in ('human_other_two','azure_two_rater'):
                if i==0 and reference=='azure_two_rater':continue
                base=human if reference=='human_other_two' else machine[:,0]
                basepoint=hpoint[0] if reference=='human_other_two' else point[0,0]
                for j,metric in enumerate(('rho','qwk')):
                    delta=machine[:,i,j]-base[:,j]
                    q=np.quantile(delta,[.025,.975,.025/COMPARISONS,1-.025/COMPARISONS])
                    out.append(dict(construct=c,system=s,reference=reference,metric=metric,difference=float(point[0,i,j]-basepoint[j]),ci95=q[:2].tolist(),familywise95_bonferroni258=q[2:].tolist()))
        print(c,'complete',flush=True)
        del machine,human
    assert len(out)==258
    result=dict(protocol='paired-all-selections-v1',bootstrap=draws,seed=seed,n=n,comparisons_count=258,grouping='one unique recording URL per assumed speaker',source_sha256=hashlib.sha256(raw).hexdigest(),reference='mean of three pair-specific metrics; humans versus other two',adjustment='Bonferroni across all 258 S/D/H versus human and Azure contrasts',comparisons=out,point_estimates=points,elapsed_seconds=time.monotonic()-start)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print('Saved aggregate results',output,'seconds',round(result['elapsed_seconds']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=600000);p.add_argument('--seed',type=int,default=42);p.add_argument('--batch',type=int,default=512);a=p.parse_args()
    run(a.archive,a.output,a.bootstrap,a.seed,a.batch)
