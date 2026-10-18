"""Prepare non-final corpus manifests from separately obtained source archives."""
import argparse,csv,hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
from .train import save,digest,CONSTRUCTS
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'speech'),str(ROOT/'speech/scripts')]
REFERENCE='The North Wind and the Sun were disputing which was the stronger, when a traveler came along wrapped in a warm cloak. They agreed that the one who first succeeded in making the traveler take his cloak off should be considered stronger than the other. Then the North Wind blew as hard as he could, but the more he blew the more closely did the traveler fold his cloak around him; and at last the North Wind gave up the attempt. Then the Sun shone out warmly, and immediately the traveler took off his cloak. And so the North Wind was obliged to confess that the Sun was the stronger of the two.'

def speechocean(source,out):
 def table(p):return dict(line.split(maxsplit=1) for line in p.read_text().splitlines() if line.strip())
 speakers=table(source/'train/utt2spk');ranked=sorted(set(speakers.values()),key=lambda s:hashlib.sha256(('validation-v1:42:'+s).encode()).hexdigest());validation=set(ranked[:25])
 if len(ranked)!=125 or len(speakers)!=2500:raise ValueError('Expected official SpeechOcean train partition')
 wavs=table(source/'train/wav.scp');scores=json.loads((source/'resource/scores.json').read_text());rows=[]
 for rid,speaker in speakers.items():
  path=(source/wavs[rid]).resolve();v=scores[rid]
  rows.append(dict(recording_id=rid,speaker_id=speaker,split='validation' if speaker in validation else 'train',audio_path=str(path),audio_sha256=digest(path),reference_text=v['text'],targets=[v[k] for k in ('accuracy','fluency','prosodic')]))
 return rows

def aesop(source,out):
 import soundfile as sf
 rows=[]
 with zipfile.ZipFile(source) as z:
  member=next(n for n in z.namelist() if n.endswith('/Rating6_01.csv'));ratings=list(csv.DictReader(io.StringIO(z.read(member).decode('utf-8-sig'))))
  speakers=sorted({r['Speaker'] for r in ratings if r['Speaker'].startswith('J_')},key=lambda s:hashlib.sha256(('aesop-split-v1-42:'+s).encode()).hexdigest())
  if len(speakers)!=180:raise ValueError('Expected 180 J-AESOP L2 speakers')
  for i,s in enumerate(speakers[:144]):
   rr=[r for r in ratings if r['Speaker']==s]
   if len(rr)!=48 or any(len({r['Rater'] for r in rr if r['Section']==section})!=16 for section in ('1','2','3')):raise ValueError('Incomplete AESOP ratings')
   raw=z.read('CorpusData-main/Audio/'+s+'_6_01.wav');audio,rate=sf.read(io.BytesIO(raw),dtype='float32');duration=len(audio)/rate
   if not .25<=duration<=120:raise ValueError('Unexpected AESOP duration; revise cohort explicitly')
   y=np.array([[float(r[k]) for k in ('Segment','Fluency','Prosody')] for r in rr]);
   if not np.isfinite(y).all() or ((y<1)|(y>10)).any():raise ValueError('Invalid AESOP labels')
   path=out/'audio'/(s+'.wav');path.parent.mkdir(exist_ok=True);path.write_bytes(raw)
   rows.append(dict(recording_id=s,speaker_id=s,split='train' if i<108 else 'validation',audio_path=str(path.resolve()),audio_sha256=digest(path),reference_text=REFERENCE,targets=(10*(y.mean(0)-1)/9).tolist()))
 return rows

def korean(source,out):
 from scripts.korean_pronunciation_pilot import audit
 from pronunciation.text_reference import load_dictionary,text_to_phones
 cfg=json.loads((ROOT/'configs/korean-cohort.json').read_text())
 # Receipts record user-provided files, never assert an upstream checksum.
 for name in ('korean-train-labels.zip','TS_PRN_en_NA_NA.zip'):
  p=source/name;receipt=p.with_suffix(p.suffix+'.receipt.json')
  if not receipt.exists():save(receipt,{'sha256':digest(p),'integrity':'locally supplied archive'})
 records=audit(source,out,cfg);dictionary=load_dictionary(ROOT/'speech/resources/text-reference-v1/cmudict.dict');rows=[];excluded=[]
 with zipfile.ZipFile(source/'TS_PRN_en_NA_NA.zip') as z:
  for r in records:
   _,status=text_to_phones(r['reference_text'],dictionary)
   if r['duration_s']>20 or not status['complete']:
    excluded.append({'recording_id':r['recording_id'],'reason':'duration_gt_20' if r['duration_s']>20 else 'incomplete_reference'});continue
   path=out/'audio'/(hashlib.sha256(r['recording_id'].encode()).hexdigest()+'.wav');path.parent.mkdir(exist_ok=True);path.write_bytes(z.read(r['audio_member']))
   if digest(path)!=r['audio_sha256']:raise ValueError('Source changed during preparation')
   rows.append(dict(recording_id=r['recording_id'],speaker_id=r['speaker_group'],split=r['split'],audio_path=str(path.resolve()),audio_sha256=r['audio_sha256'],reference_text=r['reference_text'],targets=(2.5*(np.asarray(r['targets'])[[0,1,1]]-1)).tolist()))
 save(out/'exclusions.json',excluded)
 if sum(r['split']=='train' for r in rows)!=2699 or sum(r['split']=='validation' for r in rows)!=636:raise ValueError('Korean cohort differs from reported experiment')
 return rows

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('corpus',choices=['speechocean','korean','aesop']);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 target=a.output/'manifest.json'
 if target.exists():raise ValueError('Manifest already exists; use a new directory for changed inputs')
 rows=globals()[a.corpus](a.source.resolve(),a.output.resolve());save(target,{'schema_version':'speech-audio-manifest-v1','corpus':a.corpus,'target_order':list(CONSTRUCTS),'target_scale':'0..10','rows':rows})
 print(f'{len(rows)} non-final recordings prepared: {target}')
if __name__=='__main__':main()
