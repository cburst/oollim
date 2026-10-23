"""Render all observed two-rater leaders and ties from aggregate statistics."""
import argparse,csv,html
from pathlib import Path
CONSTRUCTS=('Accuracy','Fluency','Prosody')
RULES={'spearman':'S','damae':'D','historical_grid_damae':'H'}
NOTE=('Human agreement averages each rater versus the other two; model agreement averages the same three two-rater references. S: construct-specific development Spearman; D: development accuracy MAE on the same standardized grid; H: development accuracy MAE on the historical grid converted to normalized loss. Every configuration within .001 of the highest observed Spearman or QWK for its construct among 21 combinations is shown, using unrounded estimates. Bold marks qualifying metrics and human baselines. Ties are descriptive, not tests of equivalence or superiority. Rows were chosen after external evaluation; all fitting and hyperparameter selection used public training/development data. Both metrics belong to each displayed configuration. No displayed prediction required clipping; QWK uses 0–100 integer categories with half-up rounding, and Spearman uses raw scores. Intervals are marginal 95% percentile-bootstrap intervals (2,000 draws, seed 42), not selection-adjusted. Full results are in the supplement. Its paired tests concern the matched-Spearman subset, not all displayed leaders.')
def select(rows,tolerance=.001):
 output=[]
 for c in CONSTRUCTS:
  panel=[r for r in rows if r['Construct']==c];a=[r for r in panel if r['Reference']=='two' and '__' in r['System']]
  if len(a)!=21 or len({r['System'] for r in a})!=21:raise ValueError('Expected 21 configurations')
  if any(int(r['N'])!=305 for r in panel):raise ValueError('Expected complete matched cohort')
  best={m:max(float(r[m]) for r in a) for m in ('Spearman','QWK')}
  h=next(r for r in panel if not r['System'] and 'other-rater' in r['Method']);az=next(r for r in panel if r['System']=='azure' and r['Reference']=='two')
  chosen=[r for r in a if any(best[m]-float(r[m])<=tolerance+1e-12 for m in best)]
  chosen.sort(key=lambda r:(not(best['Spearman']-float(r['Spearman'])<=tolerance+1e-12),r['System'].split('__')[1],list(RULES).index(r['System'].split('__')[0])))
  for r in [h,az]+chosen:
   system=r['System'];d=dict(r);d['Family']=system.split('__')[1] if '__' in system else 'Azure' if system else 'Human consensus';d['Rule']=RULES[system.split('__')[0]] if '__' in system else '--'
   for m in best:d[m+'Leader']=not system or ('__' in system and best[m]-float(r[m])<=tolerance+1e-12)
   output.append(d)
 return output

def render(source,output):
 with Path(source).open(newline='') as f:rows=list(csv.DictReader(f))
 selected=select(rows);out=Path(output);out.mkdir(parents=True,exist_ok=True)
 with (out/'leaders.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
 tex=[];md=['# Observed leaders and descriptive ties','',NOTE,'','| Construct | Family | Rule | Spearman | 95% CI | QWK | 95% CI |','|---|---|---|---:|---|---:|---|'];web=['<meta charset="utf-8"><style>body{font:17px Georgia;margin:30px}td,th{padding:8px;border-bottom:1px solid #ddd;white-space:nowrap}table{border-collapse:collapse}</style><h1>Observed leaders and ties</h1><table><tr><th>Construct</th><th>Family</th><th>Rule</th><th>Spearman</th><th>95% CI</th><th>QWK</th><th>95% CI</th></tr>'];previous=None
 for r in selected:
  if previous and r['Construct']!=previous:tex.append(r'\midrule')
  cells=[r['Construct'] if previous!=r['Construct'] else '',r['Family'],r['Rule']];tc=cells.copy();mc=cells.copy();hc=[html.escape(v) for v in cells]
  if not r['System']:tc[1]=r'\textbf{Human consensus}'
  for m in ('Spearman','QWK'):
   v=f'{float(r[m]):.3f}';b=r[m+'Leader'];tc.append(r'\textbf{'+v+'}' if b else v);mc.append('**'+v+'**' if b else v);hc.append('<b>'+v+'</b>' if b else v)
   interval=f"[{float(r[m+'_CI_Lower']):.3f}, {float(r[m+'_CI_Upper']):.3f}]";tc.append(interval);mc.append(interval);hc.append(interval)
  tex.append(' & '.join(tc)+r'\\');md.append('| '+' | '.join(mc)+' |');web.append('<tr>'+''.join('<td>'+v+'</td>' for v in hc)+'</tr>');previous=r['Construct']
 (out/'leader-table-rows.tex').write_text('\n'.join(tex)+'\n');(out/'leaders.md').write_text('\n'.join(md)+'\n');(out/'leaders.html').write_text('\n'.join(web)+'</table><p>'+html.escape(NOTE)+'</p>')
 return selected

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=Path('results/selection-comparison-v1/external-all-conditions.csv'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.input,a.output)
if __name__=='__main__':main()
