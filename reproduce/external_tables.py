"""Render paper and supplementary tables from aggregate statistics, without inference."""
import argparse,csv,json,html
from decimal import Decimal
from pathlib import Path

CONSTRUCTS=('Accuracy','Fluency','Prosody')
METHODS=('Human raters','Azure','speechocean','korean','aesop','koreanocean','aesopocean','koreanaesop','aesopkoreanocean')
NOTE='Human: mean agreement of each rater with the other two raters’ mean. Models: mean agreement with each of the three two-rater means. Intervals are marginal 95% speaker-bootstrap intervals, not intervals for differences. QWK clips predictions to 0–100 before half-up rounding; Spearman uses raw scores. No fitted calibration.'

def render(source,output):
 rows=json.loads(Path(source).read_text())['rows'];output=Path(output);output.mkdir(parents=True,exist_ok=True)
 lookup={(r['method'],r['construct'],r['reference']):r for r in rows}
 if len(lookup)!=54 or len(rows)!=54:raise ValueError('Expected 54 unique complete aggregate rows')
 if len({r['n'] for r in rows})!=1:raise ValueError('Main table requires a matched cohort')
 headers=['Method']+[f'{c} {m} [95% CI]' for c in CONSTRUCTS for m in ('rho','QWK')]
 main=[];tex=[]
 for method in METHODS:
  cells=[method];tc=[r'\textbf{Human consensus}' if method=='Human raters' else method]
  for c in CONSTRUCTS:
   r=lookup[method,c,'other two' if method=='Human raters' else 'two']
   for metric in ('rho','qwk'):
    cells.append(f"{r[metric]} {r[metric+'_ci']}")
    value=r[metric]
    best=max(Decimal(str(lookup[m,c,'two'].get(metric+'_raw',lookup[m,c,'two'][metric]))) for m in METHODS[2:])
    leader=method in METHODS[2:] and best-Decimal(str(r.get(metric+'_raw',value)))<=Decimal('0.001')
    if method=='Human raters' or leader:value=r'\textbf{'+value+'}'
    tc.append(value)
  main.append(cells);tex.append(' & '.join(tc)+r'\\')
 (output/'main-table-rows.tex').write_text('\n'.join(tex)+'\n')
 md=['# Agreement with two-rater consensus','',f"N={rows[0]['n']} for every row. "+NOTE,'','| '+' | '.join(headers)+' |','|'+'|'.join(['---']*7)+'|']
 md+=['| '+' | '.join(r)+' |' for r in main]
 (output/'main-table.md').write_text('\n'.join(md)+'\n')
 web='<meta charset="utf-8"><style>body{font:16px Georgia;margin:30px}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd;white-space:nowrap}</style><table><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr>'
 for r in main:web+='<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in r)+'</tr>'
 (output/'main-table.html').write_text(web+'</table><p>'+html.escape(NOTE)+'</p>')
 with (output/'supplementary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return main

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=Path('results/selection-comparison-v1/external-matched.json'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.input,a.output)
if __name__=='__main__':main()
