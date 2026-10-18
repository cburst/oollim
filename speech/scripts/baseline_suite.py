"""Supervise both local baselines and emit a paired validation comparison."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/baseline-v1'


def write(name,data):
    REPORT.mkdir(parents=True,exist_ok=True)
    p=REPORT/name;t=p.with_suffix('.tmp')
    t.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');t.replace(p)


def compare():
    cfg=json.loads((ROOT/'experiments/baseline-v1.json').read_text())
    preds={n:json.loads((REPORT/(n+'-predictions.json')).read_text()) for n in cfg['backbones']}
    a,b=preds['xeus'],preds['xlsr']
    assert [(r['recording_id'],r['target']) for r in a]==[(r['recording_id'],r['target']) for r in b]
    speakers=sorted({r['speaker_id'] for r in a});dims=cfg['dimensions']
    differences=np.array([[abs(r['prediction'][d]-r['target'][d])-abs(s['prediction'][d]-s['target'][d]) for d in dims] for r,s in zip(a,b)])
    groups=[np.array([i for i,r in enumerate(a) if r['speaker_id']==speaker]) for speaker in speakers]
    rng=np.random.default_rng(cfg['seed']);samples=[]
    for _ in range(cfg['bootstrap_speaker_replicates']):
        ids=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        samples.append(differences[ids].mean(0))
    lo,hi=np.quantile(samples,[.025,.975],axis=0)
    results={n:json.loads((REPORT/(n+'-results.json')).read_text()) for n in cfg['backbones']}
    write('comparison.json',dict(status='complete',n_validation=len(a),validation_speakers=len(speakers),
        results={n:dict(alpha=r['selected_alpha'],metrics=r['selected_validation_metrics']) for n,r in results.items()},
        paired_mae_difference_xeus_minus_xlsr={d:dict(estimate=float(differences[:,j].mean()),low=float(lo[j]),high=float(hi[j])) for j,d in enumerate(dims)},
        limitations='Exploratory 95% speaker-bootstrap intervals on the same validation used for alpha selection. Single split; no test inference or SOTA claim.',
        test_evaluated=False))


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    started=time.time();processes={};logs={}
    env={**os.environ,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'}
    write('suite-status.json',dict(status='running',started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
    try:
        for name in ('xeus','xlsr'):
            logs[name]=(ROOT/'reports'/('baseline-'+name+'.log')).open('w')
            processes[name]=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_baseline.py'),'--backbone',name],stdout=logs[name],stderr=subprocess.STDOUT,env=env)
        codes={name:p.wait() for name,p in processes.items()}
        if any(codes.values()):
            write('suite-status.json',dict(status='failed',exit_codes=codes,elapsed_s=time.time()-started))
            raise RuntimeError(f'Baseline failed: {codes}')
        compare()
        write('suite-status.json',dict(status='complete',elapsed_s=time.time()-started))
    finally:
        for p in processes.values():
            if p.poll() is None:p.terminate()
        for f in logs.values():f.close()


if __name__=='__main__':main()
