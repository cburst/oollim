"""Optional CTC evidence alongside immutable common frame caches."""
import json,os,tempfile
from pathlib import Path
import numpy as np
from .frame_cache import identity,expected
from .development import digest


def verify(directory,row,key,frame_count):
    path=Path(directory)/(identity(row)+'.phones.npz');receipt=path.with_suffix('.json')
    if not receipt.exists():return False
    info=json.loads(receipt.read_text())
    if any(info.get(k)!=v for k,v in expected(row,key).items()):raise ValueError('Stale phone cache')
    if digest(path)!=info['sha256']:raise ValueError('Corrupt phone cache')
    if info['frame_sha256']!=digest(Path(directory)/(identity(row)+'.npz')):raise ValueError('Frame/phone cache mismatch')
    with np.load(path,allow_pickle=False) as f:logits=f['logits'];ids=f['greedy_ids']
    if logits.shape!=(frame_count,len(info['vocabulary'])) or logits.dtype!=np.float32 or not np.isfinite(logits).all():raise ValueError('Invalid logits')
    decoded=logits.argmax(-1);decoded=decoded[np.r_[True,decoded[1:]!=decoded[:-1]]];decoded=decoded[decoded!=info['blank_id']]
    if not np.array_equal(ids,decoded):raise ValueError('Phone decoding mismatch')
    return True


def save(directory,row,key,logits,vocabulary,blank_id):
    path=Path(directory)/(identity(row)+'.phones.npz');receipt=path.with_suffix('.json')
    if receipt.exists():raise FileExistsError('Published phone record is immutable')
    logits=np.asarray(logits,dtype=np.float32)
    if logits.ndim!=2 or not len(logits) or logits.shape[1]!=len(vocabulary) or not np.isfinite(logits).all() or not 0<=blank_id<len(vocabulary):raise ValueError('Invalid CTC evidence')
    ids=logits.argmax(-1);ids=ids[np.r_[True,ids[1:]!=ids[:-1]]];ids=ids[ids!=blank_id]
    fd,name=tempfile.mkstemp(dir=directory,suffix='.tmp');temporary=Path(name)
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez(f,logits=logits,greedy_ids=ids);f.flush();os.fsync(f.fileno())
        temporary.replace(path)
        info=dict(expected(row,key),sha256=digest(path),frame_sha256=digest(Path(directory)/(identity(row)+'.npz')),
            vocabulary=vocabulary,blank_id=blank_id,greedy_tokens=[vocabulary[i] for i in ids],
            semantics='raw CTC logits and collapsed nonblank IPA tokens; not reference alignment or calibrated accuracy')
        tmp=receipt.with_suffix('.json.tmp');tmp.write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');tmp.replace(receipt)
    finally:
        if temporary.exists():temporary.unlink()
