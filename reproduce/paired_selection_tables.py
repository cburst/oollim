"""Render aggregate-only paired S/D/H comparisons for the manuscript appendix."""
import argparse, csv, json
from pathlib import Path
RULES=('spearman','damae','historical_grid_damae')
NAMES=('S: construct-specific Spearman','D: accuracy MAE','H: historical-grid accuracy MAE')
FAMILIES=('speechocean','korean','aesop','koreanocean','aesopocean','koreanaesop','aesopkoreanocean')
CONSTRUCTS=('Accuracy','Fluency','Prosody')

def render(directory):
    p=Path(directory);a=json.loads((p/'paired-seed42.json').read_text());b=json.loads((p/'paired-seed2026.json').read_text())
    key=lambda r:(r['construct'],r['system'],r['reference'],r['metric'])
    lookup={key(r):r for r in a['comparisons']};other={key(r):r for r in b['comparisons']}
    assert len(lookup)==len(other)==258 and lookup.keys()==other.keys()
    def decision(r):
        l,h=r['familywise95_bonferroni258'];return 'higher' if l>0 else 'lower' if h<0 else 'includes_zero'
    unstable=[k for k in lookup if decision(lookup[k])!=decision(other[k])]
    assert not unstable, 'Update manuscript seed-stability wording before rendering'
    summary={'comparisons':258,'bootstrap_per_seed':a['bootstrap'],'seeds':[a['seed'],b['seed']],'seed_decision_disagreements':len(unstable),'groups':[]}
    for rule in RULES:
        for ref in ('human_other_two','azure_two_rater'):
            for metric in ('rho','qwk'):
                xs=[r for r in a['comparisons'] if r['system'].startswith(rule+'__') and r['reference']==ref and r['metric']==metric]
                summary['groups'].append(dict(rule=rule,reference=ref,metric=metric,**{d:sum(decision(r)==decision(other[key(r)])==d for r in xs) for d in ('higher','lower','includes_zero')}))
    (p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (p/'paired-all-conditions.csv').open('w',newline='') as f:
        fields=['construct','system','reference','metric','difference','ci95_lower','ci95_upper','adjusted_lower','adjusted_upper','adjusted_decision','seed_agreement']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in a['comparisons']:
            w.writerow({**{k:r[k] for k in fields[:5]},'ci95_lower':r['ci95'][0],'ci95_upper':r['ci95'][1],'adjusted_lower':r['familywise95_bonferroni258'][0],'adjusted_upper':r['familywise95_bonferroni258'][1],'adjusted_decision':decision(r),'seed_agreement':decision(r)==decision(other[key(r)])})
    header=(r'\toprule & & & \multicolumn{3}{c}{Accuracy} & \multicolumn{3}{c}{Fluency} & \multicolumn{3}{c}{Prosody}\\'
            r'\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}'
            r'System & Reference & Metric & Diff. & 95\% CI & Adjusted CI & Diff. & 95\% CI & Adjusted CI & Diff. & 95\% CI & Adjusted CI\\\midrule')
    lines=[r'\begin{landscape}',r'\section{Paired differences and score distributions}',
      r'The same 305 recordings were jointly resampled 600,000 times per seed (42 and 2026), treating each unique recording URL as one speaker. Each replicate recomputed agreement against the same three two-rater reference pairs. Bonferroni intervals use percentile endpoints $0.025/258$ and $1-0.025/258$, covering 21 models versus human consensus and Azure, plus Azure versus human consensus, for three constructs and two metrics. Both seeds agree on all 258 adjusted interval-exclusion decisions. The table reports seed-42 intervals; full-precision results for both seeds are released. Approximately 58 draws fall in each adjusted tail, so endpoints remain Monte Carlo approximations. Correction covers these contrasts, not the preceding model search or direct comparisons between selection rules; inference conditions on the observed panel and fitted models. No private scores were used for fitting or calibration.',
      r'\begingroup\scriptsize\setlength{\tabcolsep}{3pt}\setlength{\LTcapwidth}{\linewidth}\renewcommand{\arraystretch}{1.0}\setlength{\LTpost}{0pt}',
      r'\begin{longtable}{@{}lllrrrrrrrrr@{}}',
      r'\caption{Paired differences for all S/D/H configurations and constructs.}\label{tab:paired-all}\\',header+r'\endfirsthead',header+r'\endhead',r'\bottomrule\endfoot']
    def row(system,ref,metric):
        cells=[system.split('__')[-1],'Human' if ref=='human_other_two' else 'Azure',r'$\rho$' if metric=='rho' else 'QWK']
        for c in CONSTRUCTS:
            r=lookup[(c,system,ref,metric)];cells.append(f"{r['difference']:+.3f}")
            for interval in ('ci95','familywise95_bonferroni258'):
                lo,hi=r[interval];cells.append(f'[{lo:.3f}, {hi:.3f}]')
        return ' & '.join(cells)+r'\\'
    lines.append(r'\multicolumn{12}{l}{\textbf{Azure benchmark (independent of selection rule)}}\\*')
    for metric in ('rho','qwk'):lines.append(row('azure','human_other_two',metric))
    for rule,name in zip(RULES,NAMES):
        lines.append(r'\midrule\multicolumn{12}{l}{\textbf{'+name+r'}}\\*')
        for family in FAMILIES:
            for ref in ('human_other_two','azure_two_rater'):
                for metric in ('rho','qwk'):lines.append(row(rule+'__'+family,ref,metric))
            lines.append(r'\addlinespace[2pt]')
    lines.extend([r'\end{longtable}',r'\noindent\textit{Note.} Positive differences favor the system; Human denotes human--other-rater consensus. Adjusted intervals cover 258 contrasts. Decisions use unrounded endpoints, not displayed zeros. QWK uses clipping and half-up rounding; Spearman retains raw scores.',r'\endgroup',r'\end{landscape}'])
    (p/'paired-table.tex').write_text('\n'.join(lines)+'\n')
    return summary
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args();print(json.dumps(render(args.directory),indent=2))
