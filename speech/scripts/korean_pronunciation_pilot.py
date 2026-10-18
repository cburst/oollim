"""Private local research pilot: audited reference speech, two native Korean targets.
No automatic label mapping to SpeechOcean, no private or final-test evaluation.
"""
import argparse,fcntl,hashlib,io,json,os,shutil,sys,time,zipfile
from dataclasses import asdict
from contextlib import ExitStack
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from pronunciation.encoders import default_config,build_encoder
from pronunciation.audio import read_audio,batch_audio
from pronunciation.development import digest,metrics,paired_speaker_bootstrap
from single_encoder_comparison import standardize,ridge_fit,atomic_json,atomic_torch
TARGETS=('articulation','prosody')

def check(ok,message):
 if not ok:raise ValueError(message)
def stable(s):return hashlib.sha256(s.encode()).hexdigest()
def verified_duration(sample_count, metadata_seconds, cfg):
 measured=sample_count/16000
 check(isinstance(metadata_seconds,(int,float)) and np.isfinite(metadata_seconds) and metadata_seconds>0,'Invalid metadata duration')
 offsets=cfg.get('duration_metadata_offsets_s',[0])
 tolerance=cfg.get('duration_metadata_tolerance_s',.25)
 check(any(abs(measured-metadata_seconds-offset)<tolerance for offset in offsets),'Source audio duration mismatch')
 return measured
def duration_eligible(sample_count, metadata_seconds, sentence_count):
 # Invalid source metadata is quarantined, never corrected into a training target.
 if not (isinstance(metadata_seconds,(int,float)) and np.isfinite(metadata_seconds) and metadata_seconds>0):return False
 # Observed packaging relation, not a claim about original recording timing.
 check(isinstance(sentence_count,int) and sentence_count>0,'Invalid sentence count')
 return abs(sample_count/16000-metadata_seconds-.5*(sentence_count+1))<.001
def split_for(group):
 n=int(stable('korean-prn-v1:42:'+str(group))[:8],16)%100
 return 'train' if n<60 else 'validation' if n<80 else 'test'
def select(rows,split,limit):
 # Round-robin stable speaker selection avoids a few prolific speakers dominating.
 groups={}
 for row in rows:
  if row['split']==split:groups.setdefault(row['speaker_group'],[]).append(row)
 for values in groups.values():values.sort(key=lambda r:stable(r['recording_id']))
 ordered=sorted(groups,key=stable);selected=[]
 for index in range(max(map(len,groups.values()),default=0)):
  for group in ordered:
   if index<len(groups[group]):selected.append(groups[group][index])
   if len(selected)==limit:return selected
 return selected

def archive_receipt(path):
 receipt=path.with_suffix(path.suffix+'.receipt.json')
 check(receipt.exists(),'Archive receipt missing: '+path.name)
 saved=json.loads(receipt.read_text());check(digest(path)==saved['sha256'],'Archive checksum changed')
 return saved

def audit(source,out,cfg):
 labels=source/'korean-train-labels.zip';audio=source/'TS_PRN_en_NA_NA.zip'
 provenance={p.name:archive_receipt(p) for p in (labels,audio)}
 if (out/'audit.json').exists():check(json.loads((out/'audit.json').read_text())['provenance']==provenance,'Previously audited source changed')
 with zipfile.ZipFile(labels) as outer:
  names=[n for n in outer.namelist() if Path(n).name=='TL_PRN_en_NA_NA.zip'];check(len(names)==1,'Ambiguous label archive')
  with zipfile.ZipFile(io.BytesIO(outer.read(names[0]))) as z:
   records={}
   for name in z.namelist():
    if not name.endswith('.json'):continue
    d=json.loads(z.read(name));rid=d['fileNm'];check(rid not in records,'Duplicate label identity')
    records[rid]=d
 with zipfile.ZipFile(audio) as z:
  wavs={};hashgroups={};metadata=[];missing=[]
  for entry in z.infolist():
   if not entry.filename.lower().endswith('.wav'):continue
   rid=Path(entry.filename).stem;check(rid not in wavs,'Ambiguous audio identity');wavs[rid]=entry.filename
  parent={str(d['recorder']['recorderId']):str(d['recorder']['recorderId']) for d in records.values()}
  def find(g):
   while parent[g]!=g:parent[g]=parent[parent[g]];g=parent[g]
   return g
  for index,(rid,d) in enumerate(sorted(records.items())):
   if rid not in wavs:missing.append(rid);continue
   # Stream every compressed member for CRC + encoded duplicate audit, not inference.
   raw=z.read(wavs[rid]);sha=hashlib.sha256(raw).hexdigest()
   wave=read_audio(io.BytesIO(raw));decoded_sha=hashlib.sha256(wave.numpy().tobytes()).hexdigest()
   if cfg.get('duration_policy')=='sentence_padding_quarantine_v1':
    measured_duration=len(wave)/16000
    duration_ok=duration_eligible(len(wave),d['recrdTime'],len(d['itemScript']['contents']))
   else:
    measured_duration=verified_duration(len(wave),d['recrdTime'],cfg);duration_ok=True
   speaker=str(d['recorder']['recorderId'])
   if decoded_sha in hashgroups:
    a,b=find(speaker),find(hashgroups[decoded_sha]);parent[max(a,b)]=min(a,b)
   else:hashgroups[decoded_sha]=speaker
   metadata.append(dict(recording_id=rid,speaker_id=speaker,audio_member=wavs[rid],audio_sha256=sha,decoded_sha256=decoded_sha,duration_s=measured_duration,metadata_duration_s=d['recrdTime'],duration_eligible=duration_ok))
   if index%5000==0:print(json.dumps({'audit_audio':index,'total':len(records)}),flush=True)
  for row in metadata:
   row['speaker_group']=find(row['speaker_id']);row['split']=split_for(row['speaker_group'])
 # Store metadata-only locked allocation, including final group IDs; no test targets.
 atomic_json(out/'locked-splits.json',metadata)
 check(not missing,'Missing pronunciation audio; full join required')
 quarantined=[r for r in metadata if not r['duration_eligible']]
 atomic_json(out/'duration-quarantine.json',quarantined)
 eligible=[r for r in metadata if r['duration_eligible']]
 check(len(quarantined)<=cfg.get('max_duration_quarantine',0),'Too many duration anomalies; explicit revision required')
 selected=select(eligible,'train',cfg['train_limit'])+select(eligible,'validation',cfg['validation_limit'])
 check(len(selected)==cfg['train_limit']+cfg['validation_limit'],'Insufficient eligible pilot rows')
 kept=[];excluded=[]
 for row in selected:
  d=records[row['recording_id']];contents=d['itemScript']['contents'];rating=d['itemScript']['rating']
  text=' '.join(v.get('sentence','').strip() for v in contents).strip()
  scores=[rating.get('articulationScore'),rating.get('prosodyScore')]
  if not text or not all(isinstance(v,(int,float)) and np.isfinite(v) and 1<=v<=5 for v in scores):
   excluded.append(row['recording_id']);continue
  check(d['scriptInfo']['evaluationArea']=='PRN','Unexpected task')
  kept.append(dict(row,reference_text=text,targets=scores,
                   reference_origin='provided itemScript.contents.sentence; not sttText',label_source='AIHub corpus articulation/prosody native 1..5'))
 check(not excluded,'Selected pilot has invalid references/labels; explicit revision required')
 counts={s:sum(r['split']==s for r in metadata) for s in ('train','validation','test')}
 atomic_json(out/'audit.json',dict(provenance=provenance,labels=len(records),audio=len(wavs),locked_counts=counts,
  selected_counts={s:sum(r['split']==s for r in kept) for s in ('train','validation')},decoded_duplicate_count=len(metadata)-len(hashgroups),
  speaker_groups=len(set(r['speaker_group'] for r in metadata)),duration_quarantine_count=len(quarantined),final_test_evaluated=False,
  note='Source Training partition subdivided by duplicate-linked recorder IDs; official Validation is not used by this pilot. No cross-corpus identity independence claim.'))
 atomic_json(out/'development.json',kept)
 return kept

def extract(rows,source,out,name,cfg):
 model_cfg=default_config(name);model=build_encoder(model_cfg).cuda().eval()
 target=out/name;target.mkdir(exist_ok=True)
 contract=dict(backbone=asdict(model_cfg),protocol=cfg,manifest_sha256=digest(out/'development.json'),
  torch=str(torch.__version__),numpy=np.__version__,chunk_seconds=20,dtype='float32',sample_rate=16000)
 cp=target/'contract.json'
 if cp.exists():check(json.loads(cp.read_text())==contract,'Feature contract changed')
 else:atomic_json(cp,contract)
 allfeatures=[];decoded={}
 with ExitStack() as stack:
  archives={n:stack.enter_context(zipfile.ZipFile(source/n)) for n in {r.get('audio_archive','TS_PRN_en_NA_NA.zip') for r in rows}}
  for index,row in enumerate(rows):
   audio=archives[row.get('audio_archive','TS_PRN_en_NA_NA.zip')]
   check(shutil.disk_usage(out).free>30*1024**3,'Low disk during extraction')
   key=stable(row['recording_id']);path=target/(key+'.npz');receipt=path.with_suffix('.json')
   if receipt.exists():
    info=json.loads(receipt.read_text());check(digest(path)==info['sha256'] and info['audio_sha256']==row['audio_sha256'] and info['contract_sha256']==digest(cp) and info['reference_sha256']==stable(row['reference_text']),'Stale frame cache')
    with np.load(path,allow_pickle=False) as f:pooled=f['pooled'].copy()
   else:
    raw=audio.read(row['audio_member']);check(hashlib.sha256(raw).hexdigest()==row['audio_sha256'],'Audio mismatch')
    temporary=out/'current-audio.wav';temporary.write_bytes(raw)
    wave=read_audio(temporary);temporary.unlink()
    check(abs(len(wave)/16000-row['duration_s'])<.25,'Audio duration mismatch')
    decoded_sha=hashlib.sha256(wave.numpy().tobytes()).hexdigest()
    check(decoded_sha==row['decoded_sha256'],'Preprocessing changed after audit')
    frames=[];starts=[];chunk_starts=[];step=20*16000;offsets=list(range(0,len(wave),step))
    if len(offsets)>1 and len(wave)-offsets[-1]<400:offsets.pop()
    with torch.inference_mode():
     for i,start in enumerate(offsets):
      end=offsets[i+1] if i+1<len(offsets) else len(wave)
      batch=model(batch_audio([wave[start:end].cuda()]));batch.validate()
      f=batch.frames[0,:int(batch.lengths[0])].cpu().numpy();frames.append(f)
      starts.append(start+np.arange(len(f))*batch.stride_samples);chunk_starts.extend([start]*len(f))
    f=np.concatenate(frames);check(np.isfinite(f).all(),'Nonfinite frames')
    pooled=np.concatenate([f.mean(0),np.sqrt(f.var(0)+1e-5)])
    with path.with_suffix('.tmp').open('wb') as stream:
     np.savez(stream,frames=f,valid_mask=np.ones(len(f),bool),frame_start_samples=np.concatenate(starts),chunk_start_samples=np.array(chunk_starts),pooled=pooled)
    path.with_suffix('.tmp').replace(path)
    info=dict(sha256=digest(path),audio_sha256=row['audio_sha256'],decoded_sha256=decoded_sha,
              reference_sha256=stable(row['reference_text']),frames=len(f),shape=list(f.shape),contract_sha256=digest(cp))
    atomic_json(receipt,info)
   if info['decoded_sha256'] in decoded:
    check(decoded[info['decoded_sha256']]==row['split'],'Decoded duplicate crosses pilot splits')
   decoded[info['decoded_sha256']]=row['split'];allfeatures.append(pooled)
   if index%25==0:print(json.dumps({'encoder':name,'extracted':index+1,'total':len(rows)}),flush=True)
 del model;torch.cuda.empty_cache()
 return np.stack(allfeatures)

def train(rows,x,out,name,cfg):
 targets=tuple(cfg.get('target_names',TARGETS))
 y=np.array([r['targets'] for r in rows]);tr=np.array([r['split']=='train' for r in rows]);va=~tr
 groups=np.array([r['speaker_group'] for r in rows]);speakers=np.unique(groups[tr]);np.random.default_rng(42).shuffle(speakers)
 cv={str(a):[] for a in cfg['ridge_alphas']}
 for fold in np.array_split(speakers,5):
  ft=tr & ~np.isin(groups,fold);fv=tr & np.isin(groups,fold);xx,_,_=standardize(x,ft)
  for alpha in cfg['ridge_alphas']:
   coef,intercept=ridge_fit(xx[ft],y[ft],alpha);p=(torch.from_numpy(xx[fv])@coef+intercept).numpy()
   cv[str(alpha)].append([metrics(y[fv,j],p[:,j]) for j in range(len(targets))])
 penalties=[]
 for j in range(len(targets)):
  check(all(v[j]['spearman'] is not None for values in cv.values() for v in values),'Undefined CV correlation')
  penalties.append(max(cfg['ridge_alphas'],key=lambda a:(np.mean([v[j]['spearman'] for v in cv[str(a)]]),-np.mean([v[j]['mae'] for v in cv[str(a)]]),a)))
 xx,mean,scale=standardize(x,tr);fits={a:ridge_fit(xx[tr],y[tr],a) for a in set(penalties)}
 coef=torch.stack([fits[a][0][:,j] for j,a in enumerate(penalties)],1);intercept=torch.stack([fits[a][1][j] for j,a in enumerate(penalties)])
 pred=(torch.from_numpy(xx[va])@coef+intercept).numpy();path=out/(name+'-ridge.pt')
 atomic_torch(path,dict(coef=coef,intercept=intercept,scaler_mean=torch.from_numpy(mean),scaler_scale=torch.from_numpy(scale),
  metadata=dict(backbone=asdict(default_config(name)),targets=list(targets),scale=cfg.get('native_scale',[1,5]),config=cfg,feature_contract=json.loads((out/name/'contract.json').read_text()),penalties=penalties)))
 restored=torch.load(path,map_location='cpu',weights_only=True);again=(torch.from_numpy((x[va]-restored['scaler_mean'].numpy())/restored['scaler_scale'].numpy())@restored['coef']+restored['intercept']).numpy()
 np.testing.assert_allclose(pred,again,rtol=1e-8,atol=1e-8)
 np.savez(out/(name+'-predictions.npz'),predictions=pred,recording_ids=np.array([r['recording_id'] for r in rows])[va])
 controls=np.broadcast_to(y[tr].mean(0),y[va].shape)
 atomic_json(out/(name+'-results.json'),dict(metrics={d:metrics(y[va,j],pred[:,j]) for j,d in enumerate(targets)},
  train_mean={d:metrics(y[va,j],controls[:,j]) for j,d in enumerate(targets)},penalties=penalties,cv=cv,checkpoint_sha256=digest(path),parity=True,test_metrics=None))
 return pred

def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
 cfg=json.loads(a.protocol.read_text());out=ROOT/'resources'/cfg['experiment_id'];out.mkdir(exist_ok=True)
 lock=(out/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,sha in cfg['code_hashes'].items():check(digest(ROOT/name)==sha,'Source changed')
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 cp=out/'protocol.json'
 if cp.exists():check(json.loads(cp.read_text())==cfg,'Protocol changed')
 else:atomic_json(cp,cfg)
 try:
  check(torch.cuda.is_available(),'CUDA unavailable');atomic_json(out/'status.json',dict(state='auditing'))
  rows=audit(a.source,out,cfg)
  atomic_json(out/'status.json',dict(state='waiting_for_gpu'))
  gpu_lock=(ROOT/'resources/gpu-training.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX)
  predictions={}
  for name in cfg['encoders']:
   atomic_json(out/'status.json',dict(state='extracting_and_training',backbone=name))
   x=extract(rows,a.source,out,name,cfg);predictions[name]=train(rows,x,out,name,cfg)
  va=[r for r in rows if r['split']=='validation'];y=np.array([r['targets'] for r in va]);groups=[r['speaker_group'] for r in va]
  result={d:paired_speaker_bootstrap(y[:,j],predictions['xlsr'][:,j],predictions['xeus'][:,j],groups,replicates=2000) for j,d in enumerate(TARGETS)}
  atomic_json(out/'comparison.json',dict(uncertainty=result,test_metrics=None));atomic_json(out/'status.json',dict(state='completed'))
 except BaseException as e:
  atomic_json(out/'status.json',dict(state='stopped',error=type(e).__name__+': '+str(e)));raise
if __name__=='__main__':main()
