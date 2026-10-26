"""Render aggregate-only paired S/D/H/B comparisons for the manuscript appendix."""
import argparse, csv, json
from pathlib import Path
RULES=('spearman','damae','historical_grid_damae','banded_3ratio_spearman','banded_3ratio_accuracy_mae')
NAMES=('S: 1-grid construct-wise correlation','D: 1-grid pronunciation-accuracy error','H: 5-grid pronunciation-accuracy error','B-S: three-ratio banded, construct-wise correlation','B-D: three-ratio banded, pronunciation-accuracy error')
FAMILIES=('speechocean','korean','aesop','koreanocean','aesopocean','koreanaesop','aesopkoreanocean')
CONSTRUCTS=('Accuracy','Fluency','Prosody')

def render(directory):
    p=Path(directory);a=json.loads((p/'paired-seed42.json').read_text());b=json.loads((p/'paired-seed2026.json').read_text())
    key=lambda r:(r['construct'],r['system'],r['reference'],r['metric'])
    lookup={key(r):r for r in a['comparisons']};other={key(r):r for r in b['comparisons']}
    assert len(lookup)==len(other)==510 and lookup.keys()==other.keys()
    def decision(r):
        l,h=r['familywise95_bonferroni510'];return 'higher' if l>0 else 'lower' if h<0 else 'includes_zero'
    unstable=[k for k in lookup if decision(lookup[k])!=decision(other[k])]
    # Retain unstable decisions explicitly; never choose a favorable seed.
    summary={'comparisons':510,'bootstrap_per_seed':a['bootstrap'],'seeds':[a['seed'],b['seed']],'seed_decision_disagreements':len(unstable),'borderline_keys':unstable,'groups':[]}
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
            w.writerow({**{k:r[k] for k in fields[:5]},'ci95_lower':r['ci95'][0],'ci95_upper':r['ci95'][1],'adjusted_lower':r['familywise95_bonferroni510'][0],'adjusted_upper':r['familywise95_bonferroni510'][1],'adjusted_decision':decision(r) if key(r) not in unstable else 'borderline','seed_agreement':decision(r)==decision(other[key(r)])})
    header=(r'\toprule & & & \multicolumn{3}{c}{Accuracy} & \multicolumn{3}{c}{Fluency} & \multicolumn{3}{c}{Prosody}\\'
            r'\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}'
            r'System & Reference & Metric & Diff. & 95\% CI & Adjusted CI & Diff. & 95\% CI & Adjusted CI & Diff. & 95\% CI & Adjusted CI\\\midrule')
    lines=[r'\begin{landscape}',r'\section{Paired differences and score distributions}',
      r'The same 305 recordings were jointly resampled 1,200,000 times per seed (42 and 2026), treating each unique recording URL as one speaker. Each replicate recomputed agreement against the same three two-rater reference pairs. Bonferroni intervals use percentile endpoints $0.025/510$ and $1-0.025/510$, covering 35 models versus human consensus and Azure, plus Azure versus human consensus, for three constructs and two metrics, and 84 banded-versus-previous-leader contrasts. Seven of 510 adjusted decisions differ between seeds and are marked with a dagger; these borderline comparisons do not establish a direction. The table reports seed-42 intervals; full-precision results for both seeds are released. Approximately 59 draws fall in each adjusted tail, so endpoints remain Monte Carlo approximations. Correction covers these contrasts, not the preceding model search or all historical experiments; inference conditions on the observed panel and fitted models. No private scores were used for fitting or calibration.',
      r'\begingroup\scriptsize\setlength{\tabcolsep}{3pt}\setlength{\LTcapwidth}{\linewidth}\renewcommand{\arraystretch}{1.0}\setlength{\LTpost}{0pt}',
      r'\begin{longtable}{@{}lllrrrrrrrrr@{}}',
      r'\caption{Paired differences for all S/D/H/B configurations and constructs.}\label{tab:paired-all}\\',header+r'\endfirsthead',header+r'\endhead',r'\bottomrule\endfoot']
    def row(system,ref,metric):
        cells=[system.split('__')[-1],'Human' if ref=='human_other_two' else 'Azure',r'$\rho$' if metric=='rho' else 'QWK']
        for c in CONSTRUCTS:
            r=lookup[(c,system,ref,metric)];cells.append(f"{r['difference']:+.3f}")
            for interval in ('ci95','familywise95_bonferroni510'):
                lo,hi=r[interval];cells.append(f'[{lo:.3f}, {hi:.3f}]'+(r'$^{\dagger}$' if interval.startswith('familywise') and key(r) in unstable else ''))
        return ' & '.join(cells)+r'\\'
    lines.append(r'\multicolumn{12}{l}{\textbf{Azure benchmark (independent of selection rule)}}\\*')
    for metric in ('rho','qwk'):lines.append(row('azure','human_other_two',metric))
    for rule,name in zip(RULES,NAMES):
        lines.append(r'\midrule\multicolumn{12}{l}{\textbf{'+name+r'}}\\*')
        for family in FAMILIES:
            for ref in ('human_other_two','azure_two_rater'):
                for metric in ('rho','qwk'):lines.append(row(rule+'__'+family,ref,metric))
            lines.append(r'\addlinespace[2pt]')
    lines.append(r'\midrule\multicolumn{12}{l}{\textbf{Banded minus previous S/D/H metric leader}}\\*')
    from evaluation.paired_four_ridges import PREVIOUS_LEADERS
    for rule,label in zip(RULES[-2:],('B-S','B-D')):
        lines.append(r'\multicolumn{12}{l}{\textbf{'+label+r'}}\\*')
        for family in FAMILIES:
            system=rule+'__'+family
            for metric in ('rho','qwk'):
                cells=[family,'Previous best',r'$\rho$' if metric=='rho' else 'QWK']
                for c in CONSTRUCTS:
                    r=lookup[(c,system,PREVIOUS_LEADERS[c,metric],metric)]
                    cells.append(f"{r['difference']:+.3f}")
                    for interval in ('ci95','familywise95_bonferroni510'):
                        lo,hi=r[interval];cells.append(f'[{lo:.3f}, {hi:.3f}]'+(r'$^{\dagger}$' if interval.startswith('familywise') and key(r) in unstable else ''))
                lines.append(' & '.join(cells)+r'\\')
    lines.extend([r'\end{longtable}',r'\noindent\textit{Note.} Positive differences favor the system; Human denotes human--other-rater consensus. Adjusted intervals cover 510 contrasts. Decisions use unrounded endpoints, not displayed zeros. QWK uses clipping and half-up rounding; Spearman retains raw scores. Previous accuracy leaders: S-aesopocean for correlation and S-koreanocean for QWK; fluency/prosody leaders: H-koreanaesop for correlation and H-koreanocean for QWK. A dagger marks seed-dependent interval exclusion.',r'\endgroup',r'\end{landscape}'])
    (p/'paired-table.tex').write_text('\n'.join(lines)+'\n')
    return summary
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args();print(json.dumps(render(args.directory),indent=2))
