"""Resumable whole-utterance extraction for X4.

Follows reproduce/extract.py and calls the same UnifiedScorer.extract. Four
differences, all of them additive:

1. it accepts the official test partition, which load_corpus refuses by design;
2. it records reference coverage per row instead of raising on an incomplete one;
3. it saves the pooled mean and standard deviation of every layer of the three
   encoders, read by global forward hooks that never modify an output;
4. it saves the PhoneticXeus logits per utterance, for module X7.

No file of the release is modified. The pooled output of the last saved layer of
each encoder is compared with the release feature vector for every utterance; a
single mismatch aborts the run.
"""
import argparse,hashlib,json,os,resource,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
# docs/REPRODUCE.md installs the pinned ESPnet fork with pip; the installer also
# extracts and verifies the same commit under speech/resources/source, which is
# what this path points at. Recorded as an adaptation in contrib/E2/AUDIT.md.
ESPNET=Path(os.environ.get('OOLLIM_ESPNET') or ROOT/'speech/resources/source/wanchichen-espnet-5b52d57')
sys.path[:0]=[str(ROOT),str(ROOT/'speech'),str(ESPNET)]
from reproduce.train import save,digest,BLOCKS

import torch
from pronunciation.encoders import FrameBatch
from pronunciation.scoring import masked_pool

# Layers saved per encoder. The last entry of each is the representation the
# released heads read, and is checked against the release vector every utterance.
LAYER_PLAN={'xeus':19,'xlsr':25,'phoneticxeus':20}
LAYER_NOTES={
 'xeus':'19 E-Branchformer block outputs before the final norm; the adapter already requests return_all_hs=True and selects layer -1',
 'xlsr':'hidden_states of Wav2Vec2Model: index 0 is the feature projection before the transformer, 1..24 are the transformer layers; the adapter selects layer -1',
 'phoneticxeus':'19 E-Branchformer block outputs followed by the encoder after_norm output, which is PhoneOutput.frames and is what the heads pool'}
# Index sets of the cross-layer mean, half-open, declared in PROTOCOL.md D5.
# Row 0 of crosslayer_<encoder>.npy is 'all', row 1 is 'core'.
CROSSLAYER_INDEX={'xeus':{'all':[0,19],'core':[0,19]},
                  'xlsr':{'all':[0,25],'core':[1,25]},
                  'phoneticxeus':{'all':[0,20],'core':[0,19]}}
CROSSLAYER_ROWS=('all','core')
# The two routes to the mean half differ only in summation order, and both run
# in the float32 the encoders produce, so the gap is a float32 rounding scale,
# not a float64 one. The actual value is recorded per utterance.
CROSSLAYER_TOLERANCE=1e-3
ENCODER_WIDTH=2048
ATTENTION_MODULES=('espnet.nets.pytorch_backend.transformer.attention','pxeus.espnet_import.attention')
ATTENTION_CLASSES=('MultiHeadedAttention','RelPositionMultiHeadedAttention','LegacyRelPositionMultiHeadedAttention')

def malloc_trimmer():
 """Returns freed glibc arenas to the kernel after each utterance.

 UnifiedScorer.extract loads and deletes three encoders per call. PyTorch CPU
 tensors come from posix_memalign, so the freed weights stay in glibc arenas and
 resident memory grows to roughly the sum of the three models. Trimming changes
 no value; it only lowers the peak, which is what decides how many shards fit.
 """
 import ctypes
 try:
  libc=ctypes.CDLL('libc.so.6',use_errno=True)
  libc.malloc_trim.argtypes=[ctypes.c_size_t];libc.malloc_trim.restype=ctypes.c_int
  return lambda:libc.malloc_trim(0)
 except OSError:
  return lambda:None

def save_shared(path,data):
 """reproduce.train.save names its temporary file after the target, so two
 shards writing the same path would collide. Only contract.json and
 vocabulary.json are shared; receipts and run logs are unique per row or shard."""
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(f'{path.name}.{os.getpid()}.tmp')
 temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)

def pool(frames,stride=320):
 """masked_pool of the release, for one unpadded sequence."""
 fb=FrameBatch(frames[None],torch.tensor([len(frames)]),torch.ones(1,len(frames),dtype=torch.bool),16000,stride,400)
 return masked_pool(fb)[0]

def release_attention(module,inputs,output):
 """Drops the attention maps ESPnet stores but inference never reads.

 The released hook covers only the espnet module path, so the vendored pxeus
 copy keeps 19 maps of (i+1)*32*T^2 bytes. E1 established that clearing them
 leaves every feature bitwise identical; without it a 20 s utterance needs
 about 6.3 GB more, which would forbid running shards in parallel.
 """
 if (not module.training and type(module).__module__ in ATTENTION_MODULES
   and type(module).__name__ in ATTENTION_CLASSES
   and isinstance(getattr(module,'attn',None),torch.Tensor)):module.attn=None

class Capture:
 """Global forward hooks that read encoder outputs and never write to them."""
 def __init__(self):self.reset();self.handle=torch.nn.modules.module.register_module_forward_hook(self)
 def reset(self):self.pooled={};self.cross={};self.blocks=[];self.phone=None;self.block_sum=None
 def __call__(self,module,inputs,output):
  mod=type(module).__module__;cls=type(module).__name__
  if cls=='EBranchformerEncoder' and mod=='espnet2.asr.encoder.e_branchformer_encoder':
   out,lengths,_=output
   if isinstance(out,tuple) and len(out)==2:
    n=int(lengths[0]);frames=[l[0,:n] for l in out[1]]
    self.pooled['xeus']=torch.stack([pool(f) for f in frames])
    self.cross['xeus']=self.crosslayer('xeus',frames)
  elif cls=='Wav2Vec2Model' and getattr(output,'hidden_states',None) is not None:
   frames=[l[0] for l in output.hidden_states]
   self.pooled['xlsr']=torch.stack([pool(f) for f in frames])
   self.cross['xlsr']=self.crosslayer('xlsr',frames)
  elif cls=='EBranchformerEncoderLayer' and mod.startswith('pxeus.'):
   t=output[0] if isinstance(output,tuple) else output
   t=t[0] if t.ndim==3 else t
   # Pooled at once so a long utterance does not hold 19 frame tensors alive.
   # One running sum stands in for them when the cross-layer mean is formed.
   self.blocks.append((len(t),pool(t)))
   self.block_sum=t.clone() if self.block_sum is None else self.block_sum+t
  elif cls=='PhoneticXeus':self.phone=output[0]
 def crosslayer(self,encoder,frames):
  """Pools the mean over layers of the frame sequence, for each index set."""
  rows=[]
  for row in CROSSLAYER_ROWS:
   start,stop=CROSSLAYER_INDEX[encoder][row]
   chosen=frames[start:stop]
   if len(chosen)!=stop-start:raise ValueError(f'Cross-layer index set out of range for {encoder}')
   rows.append(pool(torch.stack(chosen).mean(0)))
  return torch.stack(rows)

 def finish(self):
  """Stacks the PhoneticXeus layers once its final frames are known."""
  if self.phone is not None:
   n=len(self.phone.frames)
   # Single unpadded utterances, so every block output has the frame count the
   # heads pool. Asserted rather than assumed, because the pooling divides by it.
   if any(length!=n for length,_ in self.blocks):
    raise ValueError('PhoneticXeus block length differs from the pooled frame count')
   self.pooled['phoneticxeus']=torch.stack([v for _,v in self.blocks]+[pool(self.phone.frames,self.phone.stride_samples)])
   # Tied to the declaration in PROTOCOL.md D5: 'all' is blocks 0 to 18 plus
   # the after_norm output, 'core' is the 19 blocks alone.
   stride=self.phone.stride_samples
   if CROSSLAYER_INDEX['phoneticxeus']!={'all':[0,20],'core':[0,19]}:
    raise ValueError('PhoneticXeus cross-layer index sets changed; the running sum no longer matches')
   self.cross['phoneticxeus']=torch.stack([
    pool((self.block_sum+self.phone.frames)/20.0,stride),
    pool(self.block_sum/19.0,stride)])
  for k,expected in LAYER_PLAN.items():
   got=self.pooled.get(k)
   if got is None or tuple(got.shape)!=(expected,ENCODER_WIDTH):
    raise ValueError(f'Layer capture failed for {k}: {None if got is None else tuple(got.shape)}')
   row=self.cross.get(k)
   if row is None or tuple(row.shape)!=(len(CROSSLAYER_ROWS),ENCODER_WIDTH):
    raise ValueError(f'Cross-layer capture failed for {k}: {None if row is None else tuple(row.shape)}')
  layers={k:v.numpy().astype(np.float32) for k,v in self.pooled.items()}
  cross={k:v.numpy().astype(np.float32) for k,v in self.cross.items()}
  # The mean half of the 'all' row is the one quantity derivable two ways:
  # pooling the mean over layers, or averaging the per-layer pooled means.
  gaps={}
  for k in LAYER_PLAN:
   start,stop=CROSSLAYER_INDEX[k]['all']
   expected=layers[k][start:stop,:ENCODER_WIDTH//2].astype(np.float64).mean(0)
   gap=float(np.abs(cross[k][0,:ENCODER_WIDTH//2].astype(np.float64)-expected).max())
   if not np.isfinite(gap) or gap>CROSSLAYER_TOLERANCE:
    raise ValueError(f'Cross-layer mean of {k} disagrees with the per-layer means by {gap}')
   gaps[k]=gap
  return layers,cross,gaps

def memoize_verification():
 """Hashes each asset once per process instead of once per utterance.

 UnifiedScorer.extract re-reads 7.6 GiB of weights for every utterance, which
 is 60 per cent of its cost, and model.safetensors is hashed twice. The cache
 key includes device, inode, size and modification time, so a file that changes
 under the process is hashed again. Declared as a deviation in PROTOCOL.md and
 enabled only with --verify-once; the default repeats the release behaviour.
 """
 import pronunciation.encoders as encoders,pronunciation.fusion as fusion
 original=encoders.sha256;cache={};stats={'calls':0,'misses':0,'bytes':0}
 def cached(path):
  p=Path(path);s=p.stat();key=(str(p.resolve()),s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
  stats['calls']+=1
  if key not in cache:
   stats['misses']+=1;stats['bytes']+=s.st_size;cache[key]=original(p)
  return cache[key]
 # encoders.verify_file and phonetic.verify_file resolve sha256 through the
 # encoders module globals, so patching these two names covers every caller.
 encoders.sha256=cached;fusion.sha256=cached
 return stats

def extractor(metadata):
 from pronunciation.fusion import UnifiedScorer
 m=json.loads(Path(metadata).read_text());obj=UnifiedScorer.__new__(UnifiedScorer)
 obj.metadata=m;obj.blocks=m['blocks'];obj.widths=m['block_widths'];obj.width=sum(obj.widths[k] for k in obj.blocks)
 return obj

def assemble(output,manifest,contract):
 """Writes the stacked arrays from the per-utterance cache, in manifest order."""
 rows=manifest['rows'];cache=output/'records';missing=[]
 receipts={}
 for r in rows:
  key=hashlib.sha256(r['recording_id'].encode()).hexdigest();rp=cache/(key+'.json')
  if not rp.exists():missing.append(r['recording_id'])
  else:receipts[r['recording_id']]=(key,json.loads(rp.read_text()))
 if missing:raise ValueError(f'{len(missing)} rows not extracted yet, first {missing[0]}')
 values={k:[] for k in BLOCKS}
 memmaps={k:np.lib.format.open_memmap(output/f'layers_{k}.npy',mode='w+',dtype=np.float32,shape=(len(rows),LAYER_PLAN[k],ENCODER_WIDTH)) for k in LAYER_PLAN}
 crossmaps={k:np.lib.format.open_memmap(output/f'crosslayer_{k}.npy',mode='w+',dtype=np.float32,shape=(len(rows),len(CROSSLAYER_ROWS),ENCODER_WIDTH)) for k in LAYER_PLAN}
 for i,r in enumerate(rows):
  key,receipt=receipts[r['recording_id']]
  with np.load(cache/(key+'.npz'),allow_pickle=False) as f:
   for k in BLOCKS:values[k].append(f[k].copy())
   for k in LAYER_PLAN:memmaps[k][i]=f['layers_'+k];crossmaps[k][i]=f['crosslayer_'+k]
  r.update(reference_complete=receipt['reference_complete'],text_complete=receipt['text_complete'],
   ctc_alignable=receipt['ctc_alignable'],unknown_words=receipt['unknown_words'],unknown_phones=receipt['unknown_phones'],
   frames=receipt['frames'],duration_s=receipt['duration_s'])
 for k,mm in list(memmaps.items())+list(crossmaps.items()):mm.flush();del mm
 fp=output/'features.npz'
 np.savez(fp,recording_ids=np.array([r['recording_id'] for r in rows]),**{k:np.stack(v) for k,v in values.items()})
 manifest.update(schema_version='speech-feature-cache-v1',features='features.npz',features_sha256=digest(fp),
  extraction_contract=contract,layer_files={k:f'layers_{k}.npy' for k in LAYER_PLAN},
  layer_plan=LAYER_PLAN,layer_notes=LAYER_NOTES,layer_sha256={k:digest(output/f'layers_{k}.npy') for k in LAYER_PLAN},
  crosslayer_files={k:f'crosslayer_{k}.npy' for k in LAYER_PLAN},crosslayer_rows=list(CROSSLAYER_ROWS),
  crosslayer_index=CROSSLAYER_INDEX,crosslayer_tolerance=CROSSLAYER_TOLERANCE,
  crosslayer_sha256={k:digest(output/f'crosslayer_{k}.npy') for k in LAYER_PLAN},
  logits_directory='logits',logits_vocabulary='vocabulary.json')
 save(output/'manifest.json',manifest)
 covered=sum(bool(r['reference_complete']) for r in rows)
 print(f'assembled {len(rows)} rows: {covered} with complete references, {len(rows)-covered} without')
 return manifest

def main():
 p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
 p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--extractor',type=Path,default=ROOT/'configs/extractor.json')
 p.add_argument('--resources',type=Path,default=ROOT/'speech/resources')
 p.add_argument('--threads',type=int,default=2)
 p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
 p.add_argument('--limit',type=int,default=0,help='stop after this many rows of this shard')
 p.add_argument('--verify-once',action='store_true',help='hash each asset once per process rather than once per utterance')
 p.add_argument('--keep-attention',action='store_true',help='do not clear the vendored pxeus attention maps; reproduces release memory use')
 p.add_argument('--trim',action='store_true',help='call malloc_trim after each utterance to lower the resident peak')
 p.add_argument('--assemble-only',action='store_true',help='write the stacked arrays from an existing cache and exit')
 a=p.parse_args()
 if not 0<=a.shard<a.shards:raise ValueError('Invalid shard')
 from pronunciation.inference_memory import install
 install()
 if not a.keep_attention:torch.nn.modules.module.register_module_forward_hook(release_attention)
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(a.threads)
 m=json.loads(a.manifest.read_text());a.output.mkdir(parents=True,exist_ok=True)
 if any(r['split'] not in ('train','validation','test') for r in m['rows']):raise ValueError('Unknown split')
 obj=extractor(a.extractor);cache=a.output/'records';cache.mkdir(exist_ok=True)
 logits_dir=a.output/'logits';logits_dir.mkdir(exist_ok=True)
 contract={'manifest_sha256':digest(a.manifest),'extractor_sha256':digest(a.extractor),'wrapper_sha256':digest(__file__),
  'layer_plan':LAYER_PLAN,'crosslayer_index':CROSSLAYER_INDEX,'implementation':{q.name:digest(q) for q in sorted((ROOT/'speech/pronunciation').glob('*.py'))}}
 cp=a.output/'contract.json'
 # Both sides go through the same JSON round trip. Comparing a parsed file
 # against an in-memory dict otherwise fails on any value JSON cannot preserve,
 # which refuses a cache the code itself wrote.
 contract=json.loads(json.dumps(contract))
 if cp.exists() and json.loads(cp.read_text())!=contract:raise ValueError('Changed extraction inputs')
 save_shared(cp,contract)
 if a.assemble_only:assemble(a.output,m,contract);return
 stats=memoize_verification() if a.verify_once else None
 trim=malloc_trimmer() if a.trim else (lambda:None)
 capture=Capture();rows=[(i,r) for i,r in enumerate(m['rows']) if i%a.shards==a.shard]
 if a.limit:rows=rows[:a.limit]
 started=time.time();done=0;checked=0;incomplete=0;row_seconds=[];max_gap={}
 for i,r in rows:
  row_started=time.time()
  audio=Path(r['audio_path']);audio=audio if audio.is_absolute() else a.manifest.parent/audio
  if digest(audio)!=r['audio_sha256']:raise ValueError('Changed audio')
  key=hashlib.sha256(r['recording_id'].encode()).hexdigest();fp=cache/(key+'.npz');receipt=fp.with_suffix('.json')
  identity={**contract,'audio_sha256':r['audio_sha256'],'reference_sha256':hashlib.sha256(r['reference_text'].encode()).hexdigest()}
  if receipt.exists():
   saved=json.loads(receipt.read_text())
   if any(saved.get(k)!=v for k,v in identity.items()) or saved.get('features_sha256')!=digest(fp):raise ValueError('Changed cache')
  else:
   capture.reset()
   vector,status=obj.extract(audio,r['reference_text'],a.resources)
   layers,cross,gaps=capture.finish()
   offset=0;blocks={}
   for k in obj.blocks:blocks[k]=vector[0,offset:offset+obj.widths[k]];offset+=obj.widths[k]
   # The protocol fixes this: the last saved layer must be what the heads read.
   for k in LAYER_PLAN:
    if not np.array_equal(layers[k][-1],blocks[k].astype(np.float32)):
     raise ValueError(f'Last saved layer of {k} differs from the release feature vector')
   checked+=1
   for k,v in gaps.items():max_gap[k]=max(max_gap.get(k,0.0),v)
   phone=capture.phone
   if not status['phone']['reference_complete']:incomplete+=1
   temp=logits_dir/(key+'.tmp')
   with temp.open('wb') as f:np.savez(f,logits=phone.logits.numpy().astype(np.float32),token_ids=np.array(phone.token_ids,dtype=np.int32))
   temp.replace(logits_dir/(key+'.npz'))
   temp=fp.with_suffix('.tmp')
   with temp.open('wb') as f:np.savez(f,**blocks,**{'layers_'+k:v for k,v in layers.items()},
    **{'crosslayer_'+k:v for k,v in cross.items()})
   temp.replace(fp)
   save(receipt,{**identity,'features_sha256':digest(fp),'logits_sha256':digest(logits_dir/(key+'.npz')),
    'reference_complete':bool(status['phone']['reference_complete']),'text_complete':bool(status['text']['complete']),
    'ctc_alignable':bool(status['phone']['ctc_alignable']),'unknown_words':status['text']['unknown_words'],
    'unknown_phones':status['phone']['unknown_phones'],'frames':len(phone.frames),
    'duration_s':len(phone.frames)*phone.stride_samples/16000,'last_layer_matches_release':True,
    'crosslayer_mean_gap':{k:round(v,12) for k,v in gaps.items()}})
   if not (a.output/'vocabulary.json').exists():
    save_shared(a.output/'vocabulary.json',{'vocabulary':phone.vocabulary,'blank_id':phone.blank_id,'stride_samples':phone.stride_samples,'sample_rate':phone.sample_rate})
  trim()
  done+=1;row_seconds.append(time.time()-row_started)
  rate=(time.time()-started)/done
  print(f'{i+1}/{len(m["rows"])} shard {a.shard}/{a.shards} {done}/{len(rows)} {rate:.1f}s/row eta {rate*(len(rows)-done)/3600:.2f}h',flush=True)
 run={'shard':a.shard,'shards':a.shards,'rows':len(rows),'extracted':done,'identity_checked':checked,
  'incomplete_references_new':incomplete,'seconds':round(time.time()-started,1),'threads':a.threads,
  'verify_once':bool(a.verify_once),'keep_attention':bool(a.keep_attention),'trim':bool(a.trim),
  'peak_rss_gib':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,3),
  'row_seconds':{'mean':round(float(np.mean(row_seconds)),2),'median':round(float(np.median(row_seconds)),2),
   'min':round(min(row_seconds),2),'max':round(max(row_seconds),2)} if row_seconds else None,
  'crosslayer_mean_gap_max':max_gap or None,
  'verification':None if stats is None else dict(stats)}
 save(a.output/'runs'/f'shard-{a.shard}-of-{a.shards}.json',run)
 print(json.dumps(run))
if __name__=='__main__':main()
