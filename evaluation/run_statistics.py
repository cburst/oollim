"""Offline evaluation of an already assembled score CSV; no API calls or model fitting."""
import argparse,csv,json
from pathlib import Path
from .statistics import master_tables,FAMILIES
from reproduce.external_tables import render

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('csv',type=Path);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--speaker-column',required=True,help='Verified speaker identity, not a recording ID when speakers repeat')
 p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--seed',type=int,default=42)
 a=p.parse_args()
 if a.bootstrap<100:raise ValueError('Use at least 100 bootstrap draws')
 with a.csv.open(encoding='utf-8-sig',newline='') as f:rows=[r for r in csv.DictReader(f) if any(v and v.strip() for v in r.values())]
 if not rows:raise ValueError('No nonempty rows')
 # Input family columns are already on 0–100. Original s- columns alone are 0–10.
 for r in rows:
  for c in ('Accuracy','Fluency','Prosody'):
   key=f'speechocean-{c}Score100'
   if not r.get(key) and r.get(f's-{c}Score'):r[key]=str(10*float(r[f's-{c}Score']))
 a.output.mkdir(parents=True,exist_ok=True)
 master_tables(rows,a.output,FAMILIES,a.bootstrap,a.seed,a.speaker_column,False)
 result=[]
 with (a.output/'master_table.csv').open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   if r['Status']!='available' or r['Spearman_CI_Status']!='OK' or r['QWK_CI_Status']!='OK':raise ValueError('Incomplete estimates; inspect master_table.csv before rendering')
   human=not r['System'];method='Human raters' if human else ('Azure' if r['System']=='azure' else r['System'])
   ref=('pairwise' if 'pairwise' in r['Method'] else 'other two') if human else r['Reference']
   result.append(dict(construct=r['Construct'],method=method,reference=ref,n=int(r['N']),rho=f"{float(r['Spearman']):.3f}",rho_ci=f"[{float(r['Spearman_CI_Lower']):.3f}, {float(r['Spearman_CI_Upper']):.3f}]",qwk=f"{float(r['QWK']):.3f}",qwk_ci=f"[{float(r['QWK_CI_Lower']):.3f}, {float(r['QWK_CI_Upper']):.3f}]"))
 target=a.output/'aggregate.json';target.write_text(json.dumps({'rows':result},indent=2));render(target,a.output)
if __name__=='__main__':main()
