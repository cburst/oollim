"""Resumable development-only sequence extraction; one frozen encoder at a time."""
import argparse,fcntl,json,os,platform,shutil,sys,time
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from pronunciation.audio import read_audio,batch_audio,PREPROCESSING_VERSION
from pronunciation.encoder_extensions import make_encoder,hubert_config
from pronunciation.encoders import default_config,build_encoder
from pronunciation.phonetic_encoder import phonetic_config
from pronunciation import phone_cache
from pronunciation.development import digest,validate_splits
from pronunciation.frame_cache import load,save
import hashlib


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=('xeus','xlsr','hubert','phoneticxeus'),default='xeus');p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--limit',type=int);p.add_argument('--layer',type=int,default=-1)
    args=p.parse_args();torch.set_num_threads(4)
    if digest(args.manifest)!=args.manifest_sha256:raise ValueError('Manifest changed')
    rows=[json.loads(s) for s in args.manifest.read_text().splitlines()];validate_splits(rows)
    if any(r['split']=='test' for r in rows):raise ValueError('Do not supply final-test rows to extraction')
    if args.limit is not None and args.limit<1:raise ValueError('Positive limit required')
    config=hubert_config() if args.backbone=='hubert' else phonetic_config() if args.backbone=='phoneticxeus' else default_config(args.backbone);config.layer=args.layer
    # Relocatable contract: pinned asset hashes identify models, not machine paths.
    backbone=asdict(config);backbone.pop('directory')
    files=['pronunciation/'+s for s in ('audio.py','encoders.py','encoder_extensions.py','frame_cache.py','development.py')]+['scripts/extract_frames.py']
    source={s:digest(ROOT/s) for s in files}
    if args.backbone=='phoneticxeus':
        for s in ('phonetic_encoder.py','phonetic.py','phone_cache.py'):source['pronunciation/'+s]=digest(ROOT/'pronunciation'/s)
        root=Path(config.directory)
        source['phonetic_implementation_tree']=hashlib.sha256(json.dumps({str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.suffix in ('.py','.json','.yaml') and not p.name.endswith('.receipt.json')},sort_keys=True).encode()).hexdigest()
    if args.backbone=='xeus':
        espnet=ROOT/'resources/source/wanchichen-espnet-5b52d57'
        source['espnet_tree']=hashlib.sha256(json.dumps({str(p.relative_to(espnet)):digest(p) for folder in ('espnet','espnet2') for p in sorted((espnet/folder).rglob('*.py'))},sort_keys=True).encode()).hexdigest()
    contract={'schema':'frame-sequences-v1','manifest_sha256':args.manifest_sha256,'backbone':backbone,'sources':source,
        'preprocessing':PREPROCESSING_VERSION,'dtype':'float32','device':args.device,'torch':str(torch.__version__),'numpy':np.__version__,'python':platform.python_version(),
        'chunking':'none','layer':args.layer,'resampling':'16kHz once','test_evaluated':False}
    if args.backbone in ('hubert','xlsr','phoneticxeus'):
        import transformers
        contract['transformers']=transformers.__version__
    key=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest();cache=ROOT/'resources/frame-cache-v1'/key;cache.mkdir(parents=True,exist_ok=True)
    with (cache/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=cache/'contract.json'
        if record.exists() and json.loads(record.read_text())!=contract:raise ValueError('Cache contract changed')
        record.write_text(json.dumps(contract,indent=2)+'\n')
        selected=rows[:args.limit] if args.limit else rows
        encoder=None;new=skipped=phone_new=0;started=time.perf_counter()
        for i,row in enumerate(selected):
            frames=load(cache,row,key)
            if frames is not None:
                skipped+=1
                if args.backbone!='phoneticxeus' or phone_cache.verify(cache,row,key,len(frames)):continue
            if shutil.disk_usage(cache).free<5*1024**3:raise RuntimeError('Low free space')
            path=ROOT/row['audio_path']
            if digest(path)!=row['sha256']:raise ValueError('Audio changed')
            if encoder is None:encoder=build_encoder(config).to(args.device).eval()
            if frames is None:
                wave=read_audio(path).to(args.device)
                encoded=encoder(batch_audio([wave]));encoded.validate();frames=encoded.frames[0].cpu().numpy()
                save(cache,row,key,frames,stride_samples=encoded.stride_samples,receptive_field_samples=encoded.receptive_field_samples)
                new+=1
            if args.backbone=='phoneticxeus':
                logits=encoder.phone_logits(torch.from_numpy(frames).to(args.device)).cpu().numpy()
                phone_cache.save(cache,row,key,logits,encoder.vocabulary,encoder.blank_id);phone_new+=1
            print(json.dumps({'recording_id':row['recording_id'],'frames':len(frames),'progress':i+1,'total':len(selected)}),flush=True)
        result={'status':'completed','backbone':args.backbone,'cache':str(cache),'cache_key':key,'selected':len(selected),'corpus_development':len(rows),'new':new,'skipped_verified':skipped,'phone_records_new':phone_new,'elapsed_s':time.perf_counter()-started,'contract':contract}
        out=ROOT/'reports/frame-extraction-v1';out.mkdir(exist_ok=True)
        path=out/(args.backbone+'-'+str(time.time_ns())+'.json');path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'report':str(path),'new':new,'skipped':skipped}),flush=True)

if __name__=='__main__':main()
