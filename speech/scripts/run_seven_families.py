"""Versioned AESOP extraction -> seven matched ridge families; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import sys,json,hashlib,io,csv,zipfile,time,copy,fcntl,gc,argparse,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf
from threadpoolctl import threadpool_limits
from scripts.train_koreanocean import save,np_save,check,mapped
from pronunciation.encoders import sha256
from pronunciation.fusion import UnifiedScorer,DIMENSIONS
from pronunciation.development import metrics
from pronunciation.stella import checked_scorer
from pronunciation.text_reference import text_to_phones,load_dictionary

OUT=ROOT/'resources/seven-families-v1'
FAMILIES={'speechocean':['speechocean'],'korean':['korean'],'aesop':['aesop'],
 'koreanocean':['speechocean','korean'],'aesopocean':['speechocean','aesop'],
 'koreanaesop':['korean','aesop'],'aesopkoreanocean':['speechocean','korean','aesop']}
REFERENCE='The North Wind and the Sun were disputing which was the stronger, when a traveler came along wrapped in a warm cloak. They agreed that the one who first succeeded in making the traveler take his cloak off should be considered stronger than the other. Then the North Wind blew as hard as he could, but the more he blew the more closely did the traveler fold his cloak around him; and at last the North Wind gave up the attempt. Then the Sun shone out warmly, and immediately the traveler took off his cloak. And so the North Wind was obliged to confess that the Sun was the stronger of the two.'

def status(**kw):save(OUT/'status.json',dict(kw,pid=os.getpid(),updated_unix=time.time()))
def pinned(path,value):
 if path.exists():check(json.loads(path.read_text())==value,'Protocol/input changed: '+str(path))
 else:save(path,value)

def prepare():
 cfg=json.loads((ROOT/'experiments/koreanocean-v1.json').read_text())
 archive=ROOT/'resources/jaesop-v1/corpus.zip'
 with zipfile.ZipFile(archive) as z:
  ratings_member=next(n for n in z.namelist() if n.endswith('/Rating6_01.csv'))
  raw=z.read(ratings_member);ratings=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
  speakers=sorted({r['Speaker'] for r in ratings if r['Speaker'].startswith('J_')})
  check(len(speakers)==180,'Expected 180 Japanese L2 speakers')
  # Hash assignment independent of scores, filenames and traversal order.
  speakers.sort(key=lambda s:hashlib.sha256(('aesop-split-v1-42:'+s).encode()).hexdigest())
  assignments={s:('train' if i<108 else 'validation' if i<144 else 'sealed') for i,s in enumerate(speakers)}
  pinned(OUT/'speaker-splits.json',assignments)
  protocol={'version':'seven-families-v1','families':FAMILIES,'code_sha256':sha256(__file__),
   'rating_sha256':hashlib.sha256(raw).hexdigest(),'reference':REFERENCE,'aesop_mapping':'10*(mean of 16 raters x 3 sections - 1)/9',
   'aesop_cohort':'Japanese L2 only; full utterance, <=120 seconds; complete reference; no score filtering',
   'korean_mapping':'2.5*(articulation-1); fluency and prosody both 2.5*(prosody-1)',
   'alphas':[.001,.01,.1,1,10,100],'objective':'equal corpus weighted mean squared error + alpha coefficient norm squared; unpenalized intercept',
   'scaler':'included-corpus weighted train-only mean/std; gain sqrt(2048/block_width)',
   'selection':'equal included-corpus development Spearman; then lower mean MAE; then larger alpha',
   'evaluation':'all three development sets; no private or final tests','compute':'CPU 2 threads, serial extraction, no GPU',
   'existing_models':'unchanged; these are matched fresh-fit ablations, not replacements for earlier versions',
   'source_config_sha256':sha256(ROOT/'experiments/koreanocean-v1.json'),
   'extractor_config_sha256':sha256(ROOT/'experiments/stella-api-v2.json')}
  pinned(OUT/'protocol.json',protocol)
  _,coverage=text_to_phones(REFERENCE,load_dictionary(ROOT/'resources/text-reference-v1/cmudict.dict'))
  check(coverage['complete'],'Incomplete AESOP dictionary coverage')
  audio_dir=OUT/'audio';audio_dir.mkdir(exist_ok=True)
  manifest=[];excluded=[]
  for speaker in speakers:
   if assignments[speaker]=='sealed':continue
   member='CorpusData-main/Audio/'+speaker+'_6_01.wav'
   raw_audio=z.read(member);audio,rate=sf.read(io.BytesIO(raw_audio),dtype='float32')
   duration=len(audio)/rate
   if not .25<=duration<=120:
    excluded.append({'speaker':speaker,'reason':'duration','duration':duration});continue
   rr=[r for r in ratings if r['Speaker']==speaker]
   check(len(rr)==48 and {r['Section'] for r in rr}=={'1','2','3'},'Incomplete ratings')
   check(all(len({r['Rater'] for r in rr if r['Section']==section})==16 for section in ('1','2','3')),'Rater coverage')
   labels=np.array([[float(r[k]) for k in ('Segment','Fluency','Prosody')] for r in rr])
   check(np.isfinite(labels).all() and ((labels>=1)&(labels<=10)).all(),'Invalid labels')
   path=audio_dir/(speaker+'.wav');digest=hashlib.sha256(raw_audio).hexdigest()
   if path.exists():check(sha256(path)==digest,'Changed extracted audio')
   else:path.write_bytes(raw_audio)
   manifest.append({'speaker_id':speaker,'split':assignments[speaker],'path':str(path.relative_to(ROOT)),
    'audio_sha256':digest,'decoded_sha256':hashlib.sha256(audio.tobytes()).hexdigest(),'duration':duration,
    'targets':(10*(labels.mean(0)-1)/9).tolist()})
  check(len({r['decoded_sha256'] for r in manifest})==len(manifest),'Duplicate AESOP audio')
  pinned(OUT/'aesop-manifest.json',manifest);pinned(OUT/'aesop-excluded.json',excluded)
 return cfg,manifest

def extract_one(index):
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(2)
 rows=json.loads((OUT/'aesop-manifest.json').read_text());r=rows[index]
 cfg=json.loads((ROOT/'experiments/stella-api-v2.json').read_text());model=checked_scorer(cfg['extractor'])
 values,coverage=model.extract(ROOT/r['path'],REFERENCE)
 check(coverage['text']['complete'] and coverage['phone']['reference_complete'],'Incomplete reference evidence')
 blocks={};offset=0
 for block in model.blocks:
  width=model.widths[block];blocks[block]=values[0,offset:offset+width];offset+=width
 path=OUT/'features'/(r['speaker_id']+'.npz');np_save(path,**blocks)
 save(path.with_suffix('.json'),{'sha256':sha256(path),'audio_sha256':r['audio_sha256'],'protocol_sha256':sha256(OUT/'protocol.json')})

def extract(rows):
 (OUT/'features').mkdir(exist_ok=True)
 for i,r in enumerate(rows):
  path=OUT/'features'/(r['speaker_id']+'.npz');receipt=path.with_suffix('.json')
  if receipt.exists():
   v=json.loads(receipt.read_text());check(v=={'sha256':sha256(path),'audio_sha256':r['audio_sha256'],'protocol_sha256':sha256(OUT/'protocol.json')},'Stale extraction')
  else:
   status(stage='extracting_aesop',done=i,total=len(rows),speaker=r['speaker_id'])
   env=dict(os.environ);env['PYTHONPATH']=str(ROOT)+':'+str(ROOT/'resources/source/wanchichen-espnet-5b52d57')
   with (OUT/'extraction.log').open('a') as log:
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--extract-one',str(i)],env=env,cwd=ROOT,stdout=log,stderr=log,check=True,timeout=3600)
  status(stage='extracting_aesop',done=i+1,total=len(rows))

def load_corpora(cfg,rows):
 source=ROOT/'resources/koreanocean-v1'
 check(sha256(ROOT/cfg['source_manifest'])==cfg['source_manifest_sha256'],'SO manifest changed')
 check(sha256(ROOT/cfg['source_features'])==cfg['source_features_sha256'],'SO features changed')
 so=json.loads((ROOT/cfg['source_manifest']).read_text());ko=json.loads((source/'korean-manifest.json').read_text())
 with np.load(ROOT/cfg['source_features']) as f:sfeat={k:f[k].copy() for k in ('xeus','xlsr','phoneticxeus','ipa')}
 with np.load(source/'korean-base-features.npz') as f:kfeat={k:f[k].copy() for k in f.files}
 pv=[];iv=[]
 for r in ko:
  path=source/'phonetic-cache'/(hashlib.sha256(r['recording_id'].encode()).hexdigest()+'.npz');receipt=json.loads(path.with_suffix('.json').read_text())
  check(receipt['sha256']==sha256(path) and receipt['audio_sha256']==r['audio_sha256'],'Korean feature mismatch')
  with np.load(path) as f:pv.append(f['pooled']);iv.append(f['ipa'])
 kfeat.update(phoneticxeus=np.stack(pv),ipa=np.stack(iv))
 afeat={k:[] for k in sfeat}
 for r in rows:
  with np.load(OUT/'features'/(r['speaker_id']+'.npz')) as f:
   for k in afeat:afeat[k].append(f[k])
 afeat={k:np.stack(v) for k,v in afeat.items()}
 corpora={}
 for name,rs,feat,target,key in [('speechocean',so,sfeat,np.array([r['targets'] for r in so]),'speaker_id'),('korean',ko,kfeat,mapped([r['targets'] for r in ko]),'speaker_group'),('aesop',rows,afeat,np.array([r['targets'] for r in rows]),'speaker_id')]:
  check(all(r['split'] in ('train','validation') for r in rs),'Final data forbidden')
  train=np.array([r['split']=='train' for r in rs]);check(train.any() and (~train).any(),'Empty split')
  check(not {r[key] for r in rs if r['split']=='train'} & {r[key] for r in rs if r['split']=='validation'},'Speaker leakage')
  check(all(len(v)==len(rs) and np.isfinite(v).all() for v in feat.values()),'Invalid features')
  corpora[name]={'features':feat,'targets':target,'train':train,'speakers':np.array([r[key] for r in rs])}
 return corpora

def weighted_fit(x,y,w,alphas,gain):
 mean=np.sum(w[:,None]*x,0);scale=np.sqrt(np.sum(w[:,None]*(x-mean)**2,0));scale=np.where(scale<1e-8,1.,scale)
 z=(x-mean)/scale*gain;ym=np.sum(w[:,None]*y,0)
 b=np.sqrt(w[:,None])*z;c=np.sqrt(w[:,None])*(y-ym)
 # Use smaller of primal/dual eigensystems.
 if b.shape[0]<=b.shape[1]:
  eig,q=np.linalg.eigh(b@b.T);rhs=q.T@c
  coefs={alpha:b.T@(q@(rhs/(np.maximum(eig,0)[:,None]+alpha))) for alpha in alphas}
 else:
  eig,q=np.linalg.eigh(b.T@b);rhs=q.T@(b.T@c)
  coefs={alpha:q@(rhs/(np.maximum(eig,0)[:,None]+alpha)) for alpha in alphas}
 return mean,scale,ym,coefs

def train(cfg,corpora):
 protocol=json.loads((OUT/'protocol.json').read_text());table=[]
 with threadpool_limits(limits=2):
  for family,names in FAMILIES.items():
   folder=OUT/family;folder.mkdir(exist_ok=True)
   for j,dim in enumerate(DIMENSIONS):
    status(stage='training',family=family,construct=dim)
    base=checked_scorer(cfg['baseline'][dim]);blocks=base.blocks
    raw={n:np.concatenate([c['features'][k] for k in blocks],1).astype(float) for n,c in corpora.items()}
    x=np.concatenate([raw[n][corpora[n]['train']] for n in names]);y=np.concatenate([corpora[n]['targets'][corpora[n]['train']] for n in names])
    w=np.concatenate([np.full(corpora[n]['train'].sum(),1/(len(names)*corpora[n]['train'].sum())) for n in names])
    gain=np.concatenate([np.full(base.widths[k],np.sqrt(2048/base.widths[k])) for k in blocks])
    mean,scale,intercept,coefs=weighted_fit(x,y,w,protocol['alphas'],gain)
    candidates=[]
    for alpha,coef in coefs.items():
     scores={n:metrics(corpora[n]['targets'][~corpora[n]['train'],j],(((raw[n][~corpora[n]['train']]-mean)/scale*gain)@coef+intercept)[:,j]) for n in names}
     check(all(v['spearman'] is not None for v in scores.values()),'Undefined selection statistic')
     candidates.append({'alpha':alpha,'selection_spearman':float(np.mean([v['spearman'] for v in scores.values()])), 'selection_mae':float(np.mean([v['mae'] for v in scores.values()])),'corpora':scores})
    selected=max(candidates,key=lambda c:(c['selection_spearman'],-c['selection_mae'],c['alpha']));coef=coefs[selected['alpha']]
    path=folder/(dim+'.npz');np_save(path,coef=coef,intercept=intercept,scaler_mean=mean,scaler_scale=scale,block_gain=gain)
    metadata=copy.deepcopy(base.metadata);metadata.pop('config',None);metadata.pop('selected_alpha',None)
    metadata.update(checkpoint_sha256=sha256(path),target_scale='0..10 endpoint-mapped corpus labels',training={'experiment':'seven-families-v1','ridge':family,'corpora':names,'selected_construct':dim,'alpha':selected['alpha'],'protocol_sha256':sha256(OUT/'protocol.json')})
    save(path.with_suffix('.json'),metadata);restored=UnifiedScorer(path)
    evaluation={}
    for name,c in corpora.items():
     mask=~c['train'];pred=restored.predict_features(raw[name][mask])[:,j]
     np.testing.assert_allclose(pred,(((raw[name][mask]-mean)/scale*gain)@coef+intercept)[:,j])
     evaluation[name]=metrics(c['targets'][mask,j],pred)
     np_save(folder/(dim+'-'+name+'-development.npz'),prediction=pred,target=c['targets'][mask,j],speakers=c['speakers'][mask])
     table.append({'family':family,'construct':dim,'evaluation_corpus':name,'n':int(mask.sum()),**evaluation[name]})
    save(folder/(dim+'-results.json'),{'selected':selected,'candidates':candidates,'evaluation':evaluation,'blocks':blocks})
    del x,y,coefs,raw;gc.collect()
 save(OUT/'comparison.json',table)
 with (OUT/'comparison.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
 status(stage='completed',families=7,construct_heads=21)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--extract-one',type=int);args=parser.parse_args()
 if args.extract_one is not None:extract_one(args.extract_one);return
 OUT.mkdir(exist_ok=True);lock=(OUT/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  status(stage='preparing');cfg,rows=prepare();extract(rows);status(stage='loading_training_features');corpora=load_corpora(cfg,rows);train(cfg,corpora)
 except Exception as exc:status(stage='failed',error=str(exc));raise
if __name__=='__main__':main()
