"""Raw audio -> all three encoders -> restored ridge/neural parity on development audio."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from pronunciation.fusion import UnifiedScorer
from pronunciation.encoders import sha256
from fusion_data import load_data
from run_baseline import atomic_json


def main():
    torch.set_num_threads(2);started=time.time()
    rows,train,valid,x,y,baseline,_=load_data()
    index=int(np.flatnonzero(valid)[0]);row=rows[index]
    assert sha256(ROOT/row['audio_path'])==row['sha256']
    scorer=UnifiedScorer(baseline)
    for text in ('',None,'   '):
        try:scorer.extract(ROOT/row['audio_path'],text)
        except ValueError:pass
        else:raise AssertionError('Missing reference accepted')
    raw,status=scorer.extract(ROOT/row['audio_path'],row['reference_text'])
    np.testing.assert_allclose(raw,x[index:index+1],atol=.002,rtol=.001)
    variants={'ridge':baseline,**{str(s):ROOT/f'resources/unified-neural-v1/seed-{s}.pt' for s in (42,43,44)}}
    results={}
    for name,path in variants.items():
        model=UnifiedScorer(path);cached=model.predict_features(x[index:index+1]);fresh=model.predict_features(raw)
        np.testing.assert_allclose(fresh,cached,rtol=.001,atol=.002)
        results[name]=dict(cached=cached.tolist(),fresh=fresh.tolist(),max_absolute_difference=float(np.max(np.abs(cached-fresh))))
    atomic_json(ROOT/'reports/unified-neural-v1/raw-audio-parity.json',dict(status='passed',
        recording_id=row['recording_id'],split=row['split'],audio_sha256=row['sha256'],
        reference_status=status,feature_max_absolute_difference=float(np.max(np.abs(raw-x[index:index+1]))),
        models=results,missing_reference_rejected=True,elapsed_s=time.time()-started,test_evaluated=False))
    print(json.dumps(results))


if __name__=='__main__':main()
