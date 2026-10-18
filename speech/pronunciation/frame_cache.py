"""Unpadded frame sequences with integrity receipts and atomic per-record publication."""
import hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
from .development import digest


def identity(row):
    return hashlib.sha256((row['dataset']+':'+row['recording_id']).encode()).hexdigest()


def expected(row,contract_hash):
    return dict(recording_id=row['recording_id'],dataset=row['dataset'],speaker_id=row['speaker_id'],
                audio_sha256=row['sha256'],reference_sha256=hashlib.sha256(row['reference_text'].encode()).hexdigest(),contract_sha256=contract_hash)


def load(directory,row,contract_hash):
    path=Path(directory)/(identity(row)+'.npz');receipt=path.with_suffix('.json')
    if not receipt.exists():return None
    info=json.loads(receipt.read_text())
    if any(info.get(k)!=v for k,v in expected(row,contract_hash).items()):raise ValueError('Stale frame receipt')
    if digest(path)!=info['sha256']:raise ValueError('Corrupt frame cache')
    with np.load(path,allow_pickle=False) as archive:
        frames=archive['frames'];mask=archive['valid_mask'];starts=archive['frame_start_samples']
    if frames.ndim!=2 or len(frames)==0 or mask.shape!=(len(frames),) or mask.dtype!=bool or not mask.all() or not np.isfinite(frames).all():
        raise ValueError('Invalid unpadded frame cache')
    if starts.shape!=(len(frames),) or not np.array_equal(starts,np.arange(len(frames))*info['stride_samples']):raise ValueError('Invalid frame timing')
    if info['shape']!=list(frames.shape) or info['dtype']!=str(frames.dtype):raise ValueError('Frame metadata mismatch')
    return frames


def save(directory,row,contract_hash,frames,*,stride_samples,receptive_field_samples,sample_rate=16000):
    frames=np.asarray(frames)
    if frames.ndim!=2 or not len(frames) or not np.isfinite(frames).all() or frames.dtype!=np.float32:
        raise ValueError('Expected finite float32 frames')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);path=directory/(identity(row)+'.npz')
    # Caller holds the directory writer lock. Published records are immutable;
    # an interrupted first write may leave an orphan NPZ, which is safe to replace.
    if path.with_suffix('.json').exists():raise FileExistsError('Published frame record is immutable')
    fd,name=tempfile.mkstemp(dir=directory,suffix='.tmp');temporary=Path(name)
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez(stream,frames=frames,valid_mask=np.ones(len(frames),bool),frame_start_samples=np.arange(len(frames),dtype=np.int64)*stride_samples)
            stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
        info=dict(expected(row,contract_hash),sha256=digest(path),shape=list(frames.shape),dtype=str(frames.dtype),
            stride_samples=stride_samples,receptive_field_samples=receptive_field_samples,sample_rate=sample_rate,
            timing='frontend nominal frame starts; contextual embeddings are not forced phone boundaries',chunking='none; full recording, original pauses preserved')
        receipt=path.with_suffix('.json');tmp=receipt.with_suffix('.json.tmp');tmp.write_text(json.dumps(info,indent=2)+'\n');tmp.replace(receipt)
    finally:
        if temporary.exists():temporary.unlink()
