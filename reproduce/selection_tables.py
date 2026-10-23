"""Audit an external evaluation ZIP locally and export only aggregate results."""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
FAMILIES=('speechocean','korean','aesop','koreanocean','aesopocean','koreanaesop','aesopkoreanocean')
CONDITIONS=('spearman','damae','historical_grid_damae')
CONSTRUCTS=('Accuracy','Fluency','Prosody')
def metric(a,b):
 ra,rb=rankdata(a),rankdata(b);rho=np.corrcoef(ra,rb)[0,1]
 qa,qb=np.floor(np.clip(a,0,100)+.5),np.floor(np.clip(b,0,100)+.5)
 den=np.mean(qa**2)+np.mean(qb**2)-2*np.mean(qa)*np.mean(qb)
 return np.array([rho,1-np.mean((qa-qb)**2)/den])
def audit(archive,output):
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(archive) as z:
  raw=z.read('stella305_all_selections.csv');private=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
  stats=list(csv.DictReader(io.StringIO(z.read('master_table.csv').decode('utf-8-sig'))))
  assert len(private)==305 and len({r['evaluation_drive_id'] for r in private})==305
  assert len(stats)==144 and all(int(r['N'])==305 for r in stats)
  assert len({(r['Construct'],r['Method']) for r in stats})==144
  for condition in CONDITIONS:
   for family in FAMILIES:
    ridge=condition+'__'+family;state=json.loads(z.read(ridge+'_requests.csv.progress.json'))
    assert len(state['files'])==305 and all(e['status']=='completed' for e in state['files'].values())
    for row in private:
     matches=[e for name,e in state['files'].items() if row['evaluation_drive_id'] in Path(name).name];assert len(matches)==1
     result=matches[0]['result']
     assert result['model_version']==state['model_version'] and result['configuration_sha256']==state['configuration_sha256']
     for c in CONSTRUCTS:assert abs(float(row[ridge+'-'+c+'Score100'])-10*float(result['scores'][c.lower()]))<1e-10
  errors=[]
  for c in CONSTRUCTS:
   h=np.array([[float(row[f'h{j}-{c}Score']) for j in (1,2,3)] for row in private]);ref=[h[:,[k for k in range(3) if k!=j]].mean(1) for j in range(3)]
   for r in [r for r in stats if r['Construct']==c]:
    system=r['System']
    if not system:value=np.mean([metric(h[:,a],h[:,b]) for a,b in ((0,1),(0,2),(1,2))],0) if 'pairwise' in r['Method'] else np.mean([metric(h[:,j],ref[j]) for j in range(3)],0)
    else:
     pred=np.array([float(row['m-'+c+'Score'] if system=='azure' else row[system+'-'+c+'Score100']) for row in private])
     value=metric(pred,h.mean(1)) if r['Reference']=='three' else np.mean([metric(pred,b) for b in ref],0)
    errors.extend(abs(value-np.array([float(r['Spearman']),float(r['QWK'])])))
  assert max(errors)<1e-10
  fields=['Construct','Method','System','Reference','N','Speakers','Bootstrap','Seed','Spearman','Spearman_CI_Lower','Spearman_CI_Upper','QWK','QWK_CI_Lower','QWK_CI_Upper','QWKClippedBelow','QWKClippedAbove']
  with (out/'external-all-conditions.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in stats)
  main=[]
  for r in stats:
   if r['System'] and r['System']!='azure' and not r['System'].startswith('spearman__'):continue
   method='Human raters' if not r['System'] else 'Azure' if r['System']=='azure' else r['System'].removeprefix('spearman__')
   d=dict(construct=r['Construct'],method=method,reference=('pairwise' if 'pairwise' in r['Method'] else 'other two') if not r['System'] else r['Reference'],n=305)
   for source,target in [('Spearman','rho'),('QWK','qwk')]:
    d[target]=f"{float(r[source]):.3f}";d[target+'_raw']=float(r[source]);d[target+'_ci']=f"[{float(r[source+'_CI_Lower']):.3f}, {float(r[source+'_CI_Upper']):.3f}]"
   main.append(d)
  assert len(main)==54
  (out/'external-matched.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(raw).hexdigest(),precision='Independently recomputed points; supplied 2,000-draw marginal intervals.',selection='Matched construct-specific development Spearman for all seven families',rows=main),indent=2)+'\n')
  (out/'audit.json').write_text(json.dumps(dict(recordings=305,conditions=3,families=7,completed_assessments=6405,aggregate_rows=144,max_point_estimate_error=float(max(errors)),all_progress_complete=True,private_data_published=False,source_archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest()),indent=2)+'\n')
  print('PASS: 6,405 completed receipts; 144 independently verified aggregate rows; only aggregates exported.')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.archive,a.output)
if __name__=='__main__':main()
