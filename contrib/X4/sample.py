"""Builds the fixed validation subset of a manifest, for the pre-full-run checks.

Exists only so the sample is reproducible and recorded rather than chosen by
hand. Three deterministic groups, in manifest order after de-duplication: the
first rows of the partition, the first rows whose reference is incomplete, and
the longest rows by audio duration. The longest rows are what stresses memory,
the incomplete ones are what exercises the coverage tolerance.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'speech')]
from reproduce.train import save,digest,CONSTRUCTS

def main():
 p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
 p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--first',type=int,default=10);p.add_argument('--incomplete',type=int,default=5);p.add_argument('--longest',type=int,default=5)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 target=a.output/'manifest.json'
 if target.exists():raise ValueError('Manifest already exists; use a new directory')
 import soundfile as sf
 from pronunciation.text_reference import load_dictionary,text_to_phones
 dictionary=load_dictionary(ROOT/'speech/resources/text-reference-v1/cmudict.dict')
 m=json.loads(a.manifest.read_text());rows=m['rows']
 complete={}
 for r in rows:
  if r['reference_text'] not in complete:complete[r['reference_text']]=text_to_phones(r['reference_text'],dictionary)[1]['complete']
 frames={r['recording_id']:sf.info(r['audio_path']).frames for r in rows}
 chosen=[r['recording_id'] for r in rows[:a.first]]
 chosen+=[r['recording_id'] for r in rows if not complete[r['reference_text']]][:a.incomplete]
 chosen+=[r['recording_id'] for r in sorted(rows,key=lambda r:-frames[r['recording_id']])][:a.longest]
 keep=set(chosen);subset=[r for r in rows if r['recording_id'] in keep]
 save(target,{'schema_version':'speech-audio-manifest-v1','corpus':m['corpus'],'target_order':list(CONSTRUCTS),'target_scale':'0..10','rows':subset})
 save(a.output/'selection.json',{'source_manifest':str(a.manifest),'source_manifest_sha256':digest(a.manifest),
  'groups':{'first':a.first,'incomplete':a.incomplete,'longest':a.longest},
  'recording_ids':[r['recording_id'] for r in subset],
  'durations_s':{r['recording_id']:round(frames[r['recording_id']]/16000,3) for r in subset},
  'reference_complete':{r['recording_id']:complete[r['reference_text']] for r in subset},
  'manifest_sha256':digest(target)})
 print(f'{len(subset)} rows selected: {target}')
 print(json.dumps(json.loads((a.output/"selection.json").read_text()),indent=2))
if __name__=='__main__':main()
