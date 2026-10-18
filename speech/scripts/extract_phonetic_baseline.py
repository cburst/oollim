"""Cache frozen PhoneticXeus pooled frames and logits for train/validation only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from pronunciation.audio import read_audio, batch_audio, PREPROCESSING_VERSION
from pronunciation.encoders import FrameBatch, sha256
from pronunciation.phonetic import PhoneticXeus
from pronunciation.scoring import masked_pool
from run_baseline import atomic_json


def provenance(device='cpu'):
    root = ROOT/'resources/phoneticxeus'
    receipt = json.loads((root/'config.json.receipt.json').read_text())
    result = dict(backbone='phoneticxeus', model_id='changelinglab/PhoneticXeus',
        configuration=json.loads((root/'config.json').read_text()), acquisition_receipt=receipt,
        weights_sha256=sha256(root/'model.safetensors'),
        config_sha256=sha256(root/'config.json'),
        vocabulary_sha256=sha256(root/'pxeus/model/xeusphoneme/resources/ipa_vocab.json'),
        implementation={str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*.py'))},
        adapter_sha256=sha256(ROOT/'pronunciation/phonetic.py'),
        audio_sha256=sha256(ROOT/'pronunciation/audio.py'),
        pooling_sha256=sha256(ROOT/'pronunciation/scoring.py'),
        preprocessing=PREPROCESSING_VERSION, pooling='native_frame_mean_std_epsilon_1e-5',
        torch_version=str(torch.__version__), frozen=True)
    if device != 'cpu':
        result.update(device=device, dtype='float32', pooling_device='cpu')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, help='Smoke run prefix; does not change split or protocol')
    parser.add_argument('--device', choices=['cpu','mps'], default='cpu')
    args = parser.parse_args()
    cfg = json.loads((ROOT/'experiments/fusion-v1.json').read_text())
    torch.set_num_threads(cfg['threads']); torch.manual_seed(cfg['seed'])
    audit = json.loads((ROOT/'reports/corpus-audit.json').read_text())
    manifest = ROOT/'resources/corpus-manifest.jsonl'
    assert sha256(manifest)==audit['manifest_sha256']
    assert sha256(ROOT/'reports/splits.json')==audit['split_sha256']
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    rows = [r for r in rows if r['split'] in ('train','validation')]
    assert len(rows)==2500
    if args.device=='mps':
        assert torch.backends.mps.is_available()
        parity=json.loads((ROOT/'reports/fusion-v1/mps-parity.json').read_text())
        assert parity['status']=='passed'
    prov = provenance(args.device)
    key = hashlib.sha256(json.dumps(prov, sort_keys=True).encode()).hexdigest()
    cache = ROOT/'resources/phonetic-baseline-cache'/key
    cache.mkdir(parents=True, exist_ok=True)
    atomic_json(cache/'provenance.json', prov)
    report = ROOT/'reports/fusion-v1'
    atomic_json(report/'phonetic-cache.json', dict(cache_key=key, provenance=prov,
        manifest_sha256=audit['manifest_sha256'], split_sha256=audit['split_sha256']))
    if args.limit:
        rows = rows[:args.limit]
    model = None; start=time.time()
    for i, row in enumerate(rows):
        path = cache/(row['recording_id']+'-'+row['sha256']+'.npz')
        receipt = path.with_suffix('.sha256')
        assert sha256(ROOT/row['audio_path'])==row['sha256']
        if not path.exists() or not receipt.exists():
            if model is None:
                model = PhoneticXeus().to(args.device).eval()
            with torch.inference_mode():
                out = model(batch_audio([read_audio(ROOT/row['audio_path']).to(args.device)]))[0]
                frames = out.frames.cpu()[None]
                batch = FrameBatch(frames, torch.tensor([len(out.frames)]),
                    torch.ones(frames.shape[:2], dtype=torch.bool), 16000, out.stride_samples, 400)
                pooled = masked_pool(batch)[0].numpy()
            with path.with_suffix('.tmp').open('wb') as f:
                np.savez_compressed(f, pooled=pooled, logits=out.logits.cpu().numpy(),
                    vocabulary=np.array(out.vocabulary), blank_id=out.blank_id,
                    token_ids=np.array(out.token_ids, dtype=np.int64))
            path.with_suffix('.tmp').replace(path)
            receipt.write_text(sha256(path)+'\n')
        assert sha256(path)==receipt.read_text().strip()
        with np.load(path, allow_pickle=False) as saved:
            assert saved['pooled'].ndim==1 and np.isfinite(saved['pooled']).all()
            assert saved['logits'].ndim==2 and np.isfinite(saved['logits']).all()
        if i<3 or (i+1)%25==0 or i+1==len(rows):
            status=dict(phase='smoke_complete' if args.limit and i+1==len(rows) else 'extracting',
                completed=i+1, total=len(rows), elapsed_s=time.time()-start, cache_key=key)
            atomic_json(report/'extraction-status.json',status); print(json.dumps(status),flush=True)
            if args.device=='mps':
                # Release unused allocator buffers between groups of varying lengths.
                # Model parameters and output calculations are unchanged.
                torch.mps.empty_cache()
    if not args.limit:
        atomic_json(report/'extraction-status.json',dict(phase='complete',completed=len(rows),total=len(rows),elapsed_s=time.time()-start,cache_key=key))


if __name__=='__main__': main()
