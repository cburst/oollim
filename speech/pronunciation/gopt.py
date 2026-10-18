"""GOPT LibriSpeech checkpoint contract; torch is imported only by the worker."""
import hashlib
import importlib.util
from pathlib import Path

CHECKPOINT_SHA256='ab07451e51648f9d2455505a51055b20ac4ad7921d771ccc5170ff486a826259'
UTTERANCE=('accuracy','completeness','fluency','prosody','total')

CONFIGS = {
    "librispeech": (84, 3.203, 4.045, CHECKPOINT_SHA256),
    "paiia": (86, -.652, 9.737, "ca67543945570a6bafab7571d9ab7458b99309b697da5cbea3c72a21f172d11d"),
    "paiib": (88, -.516, 9.247, "587be4ebcea6fbf97b8e0a4fb3dacbb86db1b3f57f2a2c356aad6a81ec72dfb5"),
}
SOURCE_SHA256="e63a178f5d32221ebc1233a4fe32dc12a092f804360b1782a6159d3185b3be34"

def score(features, phone_ids, word_ids, upstream, *, acoustic_model="librispeech", implementation="official", checkpoint_root=None):
    import numpy as np
    import torch
    x=np.asarray(features,dtype=np.float32)
    ph=np.asarray(phone_ids)
    words=np.asarray(word_ids)
    if implementation not in ("official", "e3zra"):raise ValueError("Unknown implementation")
    width,mean,std,expected_hash=CONFIGS[acoustic_model]
    n=len(ph)
    if not 1<=n<=50 or x.shape!=(n,width) or words.shape!=(n,) or ph.shape!=(n,):
        raise ValueError(f'GOPT requires 1–50 aligned phones with {width} features each')
    if not np.isfinite(x).all() or not np.issubdtype(ph.dtype,np.integer) or not ((ph>=0)&(ph<39)).all():
        raise ValueError('Invalid acoustic features or canonical training phone IDs')
    if not np.issubdtype(words.dtype,np.integer) or words[0]!=0 or (np.diff(words)<0).any() or (np.diff(words)>1).any():
        raise ValueError('Word IDs must be contiguous from zero')
    root=Path(upstream)
    checkpoint=Path(checkpoint_root or root)/('pretrained_models/gopt_'+acoustic_model+'/best_audio_model.pth')
    if hashlib.sha256((root/'src/models/gopt.py').read_bytes()).hexdigest()!=SOURCE_SHA256:raise ValueError('Unrecognized upstream code')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=expected_hash:
        raise ValueError('Unrecognized GOPT checkpoint')
    spec=importlib.util.spec_from_file_location('gopt_upstream_model',root/'src/models/gopt.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=torch.nn.DataParallel(module.GOPT(embed_dim=24,num_heads=1,depth=3,input_dim=width))
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.eval();torch.set_num_threads(2)
    # Preserve training padding: 50 slots, normalized real features, zero padding,
    # and -1 phone IDs. Do not normalize the padded slots.
    padded=np.zeros((1,50,width),dtype=np.float32);padded[0,:n]=(x-mean)/std
    ids=np.full((1,50),-1,dtype=np.int64);ids[0,:n]=ph
    with torch.inference_mode():outputs=model(torch.from_numpy(padded),torch.from_numpy(ids))
    if not all(bool(torch.isfinite(v).all()) for v in outputs):raise ValueError('Non-finite GOPT result')
    # Training divides utterance/word targets by 5; phone targets stay on 0–2.
    # Return raw regressions rather than silently clipping out-of-range predictions.
    utterance={k:float(v.item()*5) for k,v in zip(UTTERANCE,outputs[:5])}
    phones=[float(v) for v in outputs[5][0,:n,0]]
    word_scores=[]
    for wid in range(int(words[-1])+1):
        mask=words==wid
        word_scores.append({k:float(v[0,:n,0].numpy()[mask].mean()*5)
                            for k,v in zip(('accuracy','stress','total'),outputs[6:])})
    return dict(model='GOPT-'+acoustic_model,implementation=implementation,checkpoint_sha256=expected_hash,training_exposure='SpeechOcean official training split; do not evaluate this checkpoint on our development subset',independent_model=False if implementation=='e3zra' else True,experimental=True,language='en',
                utterance=utterance,phones=phones,words=word_scores,
                score_scales={'utterance_and_word':[0,10],'phone':[0,2]},
                note='Unclipped regression; word outputs aggregate per-phone heads. Explicit reference alignment required. Valid tokens come from alignment, not a zero-valued feature heuristic.')
