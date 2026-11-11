"""Manifests for the full SpeechOcean762 train partition and the official test partition.

`reproduce.prepare speechocean` writes only the train partition, and `load_corpus`
refuses anything outside train/validation by design, so the test partition needs a
separate writer. The train manifest produced here is byte-identical to the one the
release writes: the development split is recomputed with the same keyed hash.
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'speech')]
from reproduce.train import save,digest,CONSTRUCTS

# Copied from reproduce/prepare.py so the train rows match the release exactly.
SPLIT_KEY='validation-v1:42:';VALIDATION_SPEAKERS=25
SPEAKERS=125;UTTERANCES=2500
SCORE_KEYS=('accuracy','fluency','prosodic')

def table(p):return dict(line.split(maxsplit=1) for line in p.read_text().splitlines() if line.strip())

def partition(source,name):
 """Rows in corpus file order. The test partition is labelled 'test', which
 load_corpus rejects, so no fitting code can read it by accident."""
 speakers=table(source/name/'utt2spk');wavs=table(source/name/'wav.scp')
 scores=json.loads((source/'resource/scores.json').read_text())
 ranked=sorted(set(speakers.values()),key=lambda s:hashlib.sha256((SPLIT_KEY+s).encode()).hexdigest())
 if len(ranked)!=SPEAKERS or len(speakers)!=UTTERANCES:raise ValueError(f'Expected official SpeechOcean {name} partition')
 validation=set(ranked[:VALIDATION_SPEAKERS]) if name=='train' else set()
 rows=[]
 for rid,speaker in speakers.items():
  if rid not in wavs or rid not in scores:raise ValueError('Recording absent from wav.scp or scores.json')
  path=(source/wavs[rid]).resolve();v=scores[rid]
  split='test' if name=='test' else ('validation' if speaker in validation else 'train')
  rows.append(dict(recording_id=rid,speaker_id=speaker,split=split,audio_path=str(path),audio_sha256=digest(path),reference_text=v['text'],targets=[v[k] for k in SCORE_KEYS]))
 return rows

def summarize(rows,source):
 """Facts the protocol asserts, recomputed here before any model runs."""
 from pronunciation.text_reference import load_dictionary,text_to_phones
 dictionary=load_dictionary(ROOT/'speech/resources/text-reference-v1/cmudict.dict')
 by_construct={};targets=[r['targets'] for r in rows]
 for j,k in enumerate(SCORE_KEYS):
  v=[t[j] for t in targets];by_construct[k]={'min':min(v),'max':max(v)}
 references={};incomplete=set()
 for r in rows:
  text=r['reference_text']
  if text not in references:
   _,status=text_to_phones(text,dictionary);references[text]=status
   if not status['complete']:incomplete.add(text)
 return {'rows':len(rows),'speakers':len(({r['speaker_id'] for r in rows})),
  'splits':{s:sum(r['split']==s for r in rows) for s in sorted({r['split'] for r in rows})},
  'distinct_references':len(references),'incomplete_references':len(incomplete),
  'rows_with_incomplete_reference':sum(r['reference_text'] in incomplete for r in rows),
  'unknown_words':sorted({w for t in incomplete for w in references[t]['unknown_words']}),
  'targets':by_construct,'source':str(source)}

def main():
 p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
 p.add_argument('partition',choices=['train','test'])
 p.add_argument('--source',type=Path,required=True,help='extracted SpeechOcean762 directory')
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 target=a.output/'manifest.json'
 if target.exists():raise ValueError('Manifest already exists; use a new directory for changed inputs')
 rows=partition(a.source.resolve(),a.partition)
 save(target,{'schema_version':'speech-audio-manifest-v1','corpus':'speechocean','target_order':list(CONSTRUCTS),'target_scale':'0..10','rows':rows})
 summary=summarize(rows,a.source.resolve());summary['partition']=a.partition;summary['manifest_sha256']=digest(target)
 save(a.output/'summary.json',summary)
 print(f'{len(rows)} recordings prepared: {target}')
 print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
