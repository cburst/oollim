"""Resumable whole-utterance extraction using the historical frozen adapters."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from .train import save,digest,BLOCKS
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'speech'))

def extractor(metadata):
 from pronunciation.fusion import UnifiedScorer
 m=json.loads(Path(metadata).read_text());obj=UnifiedScorer.__new__(UnifiedScorer)
 obj.metadata=m;obj.blocks=m['blocks'];obj.widths=m['block_widths'];obj.width=sum(obj.widths[k] for k in obj.blocks)
 return obj

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--extractor',type=Path,default=ROOT/'configs/extractor.json');p.add_argument('--resources',type=Path,default=ROOT/'speech/resources');p.add_argument('--threads',type=int,default=2);a=p.parse_args()
 import torch
 from pronunciation.inference_memory import install
 attention_cleanup=install()
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(a.threads)
 m=json.loads(a.manifest.read_text());a.output.mkdir(parents=True,exist_ok=True)
 if any(r['split'] not in ('train','validation') for r in m['rows']):raise ValueError('Private/final split forbidden')
 obj=extractor(a.extractor);cache=a.output/'records';cache.mkdir(exist_ok=True)
 contract={'manifest_sha256':digest(a.manifest),'extractor_sha256':digest(a.extractor), 'wrapper_sha256':digest(__file__), 'implementation':{p.name:digest(p) for p in sorted((ROOT/'speech/pronunciation').glob('*.py'))}};cp=a.output/'contract.json'
 if cp.exists() and json.loads(cp.read_text())!=contract:raise ValueError('Changed extraction inputs')
 save(cp,contract);values={k:[] for k in BLOCKS}
 for i,r in enumerate(m['rows']):
  audio=Path(r['audio_path']);audio=audio if audio.is_absolute() else a.manifest.parent/audio
  if digest(audio)!=r['audio_sha256']:raise ValueError('Changed audio')
  key=hashlib.sha256(r['recording_id'].encode()).hexdigest();fp=cache/(key+'.npz');receipt=fp.with_suffix('.json')
  identity={**contract,'audio_sha256':r['audio_sha256'],'reference_sha256':hashlib.sha256(r['reference_text'].encode()).hexdigest()}
  if receipt.exists():
   if json.loads(receipt.read_text())!={**identity,'features_sha256':digest(fp)}:raise ValueError('Changed cache')
  else:
   vector,status=obj.extract(audio,r['reference_text'],a.resources)
   if not status['text']['complete'] or not status['phone']['reference_complete']:raise ValueError('Incomplete reference coverage')
   offset=0;blocks={}
   for k in obj.blocks:blocks[k]=vector[0,offset:offset+obj.widths[k]];offset+=obj.widths[k]
   temp=fp.with_suffix('.tmp')
   with temp.open('wb') as f:np.savez(f,**blocks)
   temp.replace(fp);save(receipt,{**identity,'features_sha256':digest(fp)})
  with np.load(fp,allow_pickle=False) as f:
   for k in BLOCKS:values[k].append(f[k].copy())
  print(f'{i+1}/{len(m["rows"])}',flush=True)
 fp=a.output/'features.npz';np.savez(fp,recording_ids=np.array([r['recording_id'] for r in m['rows']]),**{k:np.stack(v) for k,v in values.items()})
 m.update(schema_version='speech-feature-cache-v1',features='features.npz',features_sha256=digest(fp),extraction_contract=contract);save(a.output/'manifest.json',m)
if __name__=='__main__':main()
