"""Compares two feature caches of contrib/X4/extract.py, array by array.

Used to show that an option which is meant to change only cost changes no
value: --verify-once, --trim and --keep-attention are all of that kind. The
comparison is bitwise, and the largest absolute difference is reported for every
array name so a non-zero result says where it is.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reproduce.train import save

def arrays(path):
 with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}

def main():
 p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
 p.add_argument('left',type=Path);p.add_argument('right',type=Path)
 p.add_argument('--manifest',type=Path,required=True,help='manifest whose rows fix the comparison order')
 p.add_argument('--report',type=Path)
 p.add_argument('--common-only',action='store_true',help='compare the array names both caches have, and list the rest')
 a=p.parse_args()
 rows=json.loads(a.manifest.read_text())['rows']
 names={};identical=0;compared=0;missing=[];extra=set()
 for r in rows:
  key=hashlib.sha256(r['recording_id'].encode()).hexdigest()
  pairs=[('records',a.left/'records'/f'{key}.npz',a.right/'records'/f'{key}.npz'),
         ('logits',a.left/'logits'/f'{key}.npz',a.right/'logits'/f'{key}.npz')]
  if any(not q.exists() for _,q,w in pairs for q in (q,w)):missing.append(r['recording_id']);continue
  compared+=1;row_identical=True
  for group,left,right in pairs:
   L=arrays(left);R=arrays(right)
   if sorted(L)!=sorted(R):
    if not a.common_only:raise ValueError(f'Different array names in {group} for {r["recording_id"]}')
    for k in sorted(set(L)^set(R)):extra.add(f'{group}/{k}')
   for k in sorted(set(L)&set(R)):
    name=f'{group}/{k}';entry=names.setdefault(name,{'rows':0,'identical':0,'max_abs_diff':0.0,'dtype':str(L[k].dtype),'shape':list(L[k].shape)})
    entry['rows']+=1
    if L[k].shape!=R[k].shape:raise ValueError(f'Different shapes for {name}')
    same=np.array_equal(L[k],R[k]);entry['identical']+=int(same)
    if not same:
     row_identical=False
     entry['max_abs_diff']=max(entry['max_abs_diff'],float(np.abs(L[k].astype(float)-R[k].astype(float)).max()))
  identical+=int(row_identical)
 report={'left':str(a.left),'right':str(a.right),'manifest':str(a.manifest),
  'rows_in_manifest':len(rows),'rows_compared':compared,'rows_missing':missing,
  'rows_bitwise_identical':identical,'arrays':names,
  'arrays_only_on_one_side':sorted(extra),'common_only':bool(a.common_only),
  'verdict':'bitwise identical' if compared and identical==compared and not missing else 'differs or incomplete'}
 print(json.dumps(report,indent=2))
 if a.report:save(a.report,report)
if __name__=='__main__':main()
