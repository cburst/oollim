"""Publish matched three-construct leaderboards after a fixed Praat ablation."""
import json,hashlib
import torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];src=ROOT/'resources/praat-fluency-top4-v1';out=ROOT/'reports/praat-fluency-top4-v1'
assert json.loads((src/'status.json').read_text())['stage']=='completed'
for checkpoint in src.glob('*.pt'):
 payload=torch.load(checkpoint,weights_only=True)
 assert payload['coverage_sha256']==hashlib.sha256((src/'coverage.json').read_bytes()).hexdigest(), 'Checkpoint/input provenance mismatch'
r=json.loads((src/'results.json').read_text());coverage=json.loads((src/'coverage.json').read_text());cfg=json.loads((src/'configuration.json').read_text());out.mkdir(parents=True,exist_ok=True)
lines=['# Praat addition to the top four fluency combinations','',f"Matched cohort: {coverage['train']} training and {coverage['validation']} development recordings; {coverage['train_speakers']} / {coverage['validation_speakers']} speakers. Five training-speaker folds select each construct’s ridge penalty. All encoders frozen. No final/private tests.",'','## Fluency additions','','| Base | Variant | Base rho | Added rho | Delta rho | Paired 95% CI | Delta MAE |','|---|---|---:|---:|---:|---|---:|']
gates={}
for key,v in r['comparisons'].items():
 d=v['fluency'];ci=d['intervals']['delta']['spearman']['ci95'];base,variant=key.split('__');passes=d['delta']['spearman']>=.02 and ci is not None and ci[0]>0 and d['delta']['mae']<=.2
 gates[key]={'unadjusted_gate_pass':passes,'promotion':False,'reason':'Additional multiplicity-adjusted confirmation required' if passes else 'Fails prespecified unadjusted practical/uncertainty gate'}
 lines.append(f"| {base} | {variant} | {d['baseline']['spearman']:.5f} | {d['candidate']['spearman']:.5f} | {d['delta']['spearman']:+.5f} | {ci} | {d['delta']['mae']:+.5f} |")
for target in ['accuracy','fluency','prosody']:
 lines+=['',f'## {target.title()} leaderboard (this matched experiment)','','| Rank | Model | Spearman | MAE /10 |','|---:|---|---:|---:|']
 for i,n in enumerate(r['leaderboards'][target],1):
  m=r['metrics'][n][target];lines.append(f"| {i} | {n} | {m['spearman']:.5f} | {m['mae']:.5f} |")
lines+=['','## Interpretation','',cfg['limitations'],'','Timing features: speech rate, articulation rate, pause count, inferred non-speaking duration, its fraction of total duration, and inferred non-speaking seconds per reported pause. The last is an approximate proxy including boundary silence, not measured mean internal pause length. Zero-pause proxies are missing and imputed inside training folds. Full adds 14 validated Praat summaries plus the three derived duration features; the randomized legacy pronunciation score is excluded.','','All three outputs are trained separately. Only the four requested base families receive Praat additions; the historical accuracy leader is an additional matched control. This is not a rerun of all 14 historical candidates. Historical metrics are not mixed into these rankings. No candidate is promoted automatically. Eight additions are exploratory; any candidate clearing the practical gate still requires multiplicity-adjusted confirmation.']
(out/'REVIEW.md').write_text('\n'.join(lines)+'\n');(out/'results.json').write_text(json.dumps(r|{'gates':gates},indent=2)+'\n');(out/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n');print('\n'.join(lines[:15]))
