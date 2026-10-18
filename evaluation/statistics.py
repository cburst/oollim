"""Offline statistical analysis functions; no API or network client."""

import csv, json, math, re

from pathlib import Path

from urllib.parse import urlparse, parse_qs, unquote

FAMILIES=['speechocean','korean','aesop','koreanocean','aesopocean','koreanaesop','aesopkoreanocean']

CONSTRUCTS=['Accuracy','Fluency','Prosody']

def drive_id(value):
    value = unquote(value.strip())
    parsed = urlparse(value)
    candidates = parse_qs(parsed.query).get('id', [])
    candidates += re.findall(r'/d/(1[A-Za-z0-9_-]+)', parsed.path)
    if re.fullmatch(r'1[A-Za-z0-9_-]{9,}', value):
        candidates.append(value)
    candidates = set(candidates)
    if len(candidates) != 1:
        raise ValueError('URL must contain one unambiguous Google Drive file ID: '+value)
    result = candidates.pop()
    if not re.fullmatch(r'1[A-Za-z0-9_-]{9,}', result):
        raise ValueError('Invalid Drive ID: '+result)
    return result

def match_files(rows, folder):
    ids = [drive_id(r['URL']) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate Drive IDs in CSV; resolve before running.')
    matches = {i: [] for i in ids}
    for p in sorted(folder.rglob('*')):
        if not p.is_file() or p.name.startswith('.') or p.name == 'client-token':
            continue
        found = [i for i in ids if i in p.name]
        if len(found) > 1:
            raise ValueError('Ambiguous filename matching multiple CSV IDs: '+p.name)
        if found:
            matches[found[0]].append(p.resolve())
    duplicate = [i for i, paths in matches.items() if len(paths) > 1]
    if duplicate:
        raise ValueError('Multiple files match these IDs (remove duplicate transcodes): '+', '.join(duplicate))
    return ids, {i: paths[0] for i, paths in matches.items() if paths}

def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None

def write_table(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)

def measures(a, b):
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = dict(N=len(a), Spearman='', QWK='', SpearmanStatus='insufficient_data', QWKStatus='insufficient_data',
               QWKClippedA=int(((a<0)|(a>100)).sum()), QWKClippedB=int(((b<0)|(b>100)).sum()),
               QWKPolicy='clip_0_100_then_half_up_round_v2')
    if len(a) < 2:
        return out
    if np.ptp(a) and np.ptp(b):
        out.update(Spearman=float(spearmanr(a,b).statistic), SpearmanStatus='OK')
    else:
        out['SpearmanStatus'] = 'constant_scores'
    qa, qb = np.floor(np.clip(a,0,100)+.5).astype(int), np.floor(np.clip(b,0,100)+.5).astype(int)
    if np.unique(np.concatenate([qa,qb])).size == 1:
        out['QWKStatus'] = 'undefined_identical_constant'
    else:
        k = float(cohen_kappa_score(qa,qb,labels=np.arange(101),weights='quadratic'))
        out.update(QWK=k if math.isfinite(k) else '', QWKStatus=('OK_clipped' if out['QWKClippedA'] or out['QWKClippedB'] else 'OK') if math.isfinite(k) else 'undefined')
    return out

def reports(rows, out, families):
    systems = ['azure']+families
    human_table, pair_table, main_table = [], [], []
    for c in CONSTRUCTS:
        humans = [[number(r.get(f'h{h}-{c}Score')) for h in (1,2,3)] for r in rows]
        for h in humans:
            if any(v is not None and not 0 <= v <= 100 for v in h):
                raise ValueError('Human scores must be on the 0–100 scale.')
        truth = [sum(h)/3 if all(v is not None for v in h) else None for h in humans]
        predictions = {s:[number(r.get(f'm-{c}Score' if s=='azure' else f'{s}-{c}Score100')) for r in rows] for s in systems}
        common = [i for i,t in enumerate(truth) if t is not None and all(predictions[s][i] is not None for s in systems)]
        def record(a,b,indices,**meta):
            return dict(Construct=c, **meta, Excluded=len(rows)-len(indices), **measures([a[i] for i in indices],[b[i] for i in indices]))
        for j,k in ((0,1),(0,2),(1,2)):
            a,b = [h[j] for h in humans], [h[k] for h in humans]
            idx=[i for i in range(len(rows)) if a[i] is not None and b[i] is not None]
            human_table.append(record(a,b,idx,Rater=f'h{j+1}',Reference=f'h{k+1}',Cohort='pair_complete'))
        for j in range(3):
            a=[h[j] for h in humans]
            b=[sum(h[k] for k in range(3) if k!=j)/2 if all(h[k] is not None for k in range(3) if k!=j) else None for h in humans]
            idx=[i for i in range(len(rows)) if a[i] is not None and b[i] is not None]
            human_table.append(record(a,b,idx,Rater=f'h{j+1}',Reference='mean_other_two',Cohort='three_humans_complete'))
        for s in systems:
            a=predictions[s]
            idx=[i for i in range(len(rows)) if a[i] is not None and truth[i] is not None]
            for cohort,indices in [('available_pair',idx),('all_systems_common',common)]:
                main_table.append(record(a,truth,indices,System=s,Reference='mean_three_humans',Cohort=cohort))
            for j in range(3):
                b=[h[j] for h in humans]
                idx=[i for i in range(len(rows)) if a[i] is not None and b[i] is not None]
                pair_table.append(record(a,b,idx,System=s,Reference=f'h{j+1}',Cohort='pair_complete'))
    write_table(out/'human_agreement.csv',human_table)
    write_table(out/'system_comparison.csv',main_table)
    write_table(out/'system_vs_individual_humans.csv',pair_table)
    lines=['# Independent Stella evaluation','',
           'Scores: humans/Azure 0–100; API scores multiplied by 10. SpeechOcean uses existing s- columns, multiplied by 10; imported API version unspecified.',
           'Human reference: mean of all three raters, computed before rounding. No calibration is fitted.',
           'QWK: fixed categories 0–100, nearest integer with half-up rounding. Predictions are capped at 0–100 for QWK only; raw scores remain unchanged for Spearman. Clipping counts are reported.',
           'Available-pair rows can have different samples. Use all_systems_common rows for the matched seven-family comparison.',
           'These are descriptive estimates, not significance tests. Human vs mean-other-two uses a different reference from system vs mean-three.',
           '', '| Construct | System | Cohort | N | Spearman | QWK | QWK status |',
           '|---|---|---|---:|---:|---:|---|']
    for r in main_table:
        fmt=lambda v: f'{v:.5f}' if isinstance(v,float) else str(v)
        lines.append('| '+' | '.join(fmt(r[k]) for k in ('Construct','System','Cohort','N','Spearman','QWK','QWKStatus'))+' |')
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')

def master_tables(rows, out, families, bootstrap=2000, seed=42, speaker_column='evaluation_drive_id', available_data=False):
    """Matched-cohort tables; bootstrap speakers, keeping all rater scores paired."""
    import numpy as np
    from scipy.stats import rankdata
    import html
    def stats(a,b):
        rho=np.nan;kappa=np.nan
        if len(a)<2:return np.array([rho,kappa])
        ra=rankdata(a);rb=rankdata(b)
        if np.ptp(ra)>0 and np.ptp(rb)>0:rho=float(np.corrcoef(ra,rb)[0,1])
        qa=np.floor(np.clip(a,0,100)+.5);qb=np.floor(np.clip(b,0,100)+.5)
        # Algebraically identical to fixed equally spaced 0..100 quadratic weights.
        expected=np.mean(qa*qa)+np.mean(qb*qb)-2*np.mean(qa)*np.mean(qb)
        if expected>0:kappa=float(1-np.mean((qa-qb)**2)/expected)
        return np.array([rho,kappa])
    human_methods=['Human–human pairwise agreement','Human–other-rater consensus agreement']
    all_results=[];panels=[]
    for construct in CONSTRUCTS:
        human=np.array([[number(r.get(f'h{j}-{construct}Score')) for j in (1,2,3)] for r in rows],dtype=float)
        systems=['azure']+families
        values={s:np.array([number(r.get(f'm-{construct}Score' if s=='azure' else f'{s}-{construct}Score100')) for r in rows],dtype=float) for s in systems}
        present=[s for s in systems if np.isfinite(values[s]).any()]
        complete_humans=np.isfinite(human).all(1)
        common=complete_humans.copy()
        for s in present:common &= np.isfinite(values[s])
        panel=[]
        methods=[(n,None,'human') for n in human_methods]+[(s+' — '+ref+'-rater consensus',s,ref) for s in systems for ref in ('three','two')]
        for name,system,reference in methods:
            mask=common.copy() if not available_data else complete_humans.copy()
            if name not in human_methods:
                mask &= np.isfinite(values[system])
            chosen=np.flatnonzero(mask);h=human[mask];truth=h.mean(1)
            ids=[str(rows[i].get(speaker_column,'')).strip() for i in chosen]
            if any(not i or i.lower() in ('nan','none') for i in ids):raise ValueError('Missing bootstrap speaker IDs in '+speaker_column)
            groups={}
            for i,s in enumerate(ids):groups.setdefault(s,[]).append(i)
            groups=[np.asarray(v,dtype=int) for v in groups.values()]
            prediction=values[system][mask] if name not in human_methods else None
            def evaluate(indices):
                if name in human_methods:
                    hh=h[indices]
                    if name==human_methods[0]:
                        return np.mean([stats(hh[:,a],hh[:,b]) for a,b in ((0,1),(0,2),(1,2))],axis=0)
                    return np.mean([stats(hh[:,j],hh[:,[k for k in range(3) if k!=j]].mean(1)) for j in range(3)],axis=0)
                if reference=='two':
                    hh=h[indices]
                    return np.mean([stats(prediction[indices],hh[:,[k for k in range(3) if k!=j]].mean(1)) for j in range(3)],axis=0)
                return stats(prediction[indices],truth[indices])
            estimate=evaluate(np.arange(len(chosen)))
            samples=[];rng=np.random.default_rng(seed)
            if len(groups)>=2 and len(chosen)>=3:
                for _ in range(bootstrap):
                    indices=np.concatenate([groups[j] for j in rng.integers(0,len(groups),size=len(groups))])
                    samples.append(evaluate(indices))
            draws=np.asarray(samples).reshape(-1,2)
            row=dict(Construct=construct,Method=name,System=system or '',Reference=reference,N=len(chosen),Speakers=len(groups),
                     Cohort='all_available_per_method' if available_data else 'complete_all_available_systems',IncludedSystems=name if available_data else ';'.join(present),
                     ReleaseComplete=len(present)==len(systems),Bootstrap=bootstrap,Seed=seed,
                     Status='available' if name in human_methods or system in present else 'pending_no_scores')
            row['QWKClippedBelow']=int((prediction<0).sum()) if prediction is not None else 0
            row['QWKClippedAbove']=int((prediction>100).sum()) if prediction is not None else 0
            row['QWKClippedCount']=row['QWKClippedBelow']+row['QWKClippedAbove']
            row['QWKPolicy']='clip_0_100_then_half_up_round_v2'
            for j,metric in enumerate(('Spearman','QWK')):
                valid=draws[:,j][np.isfinite(draws[:,j])] if len(draws) else np.array([])
                row[metric]=float(estimate[j]) if np.isfinite(estimate[j]) else ''
                row[metric+'_ValidBootstrap']=len(valid)
                enough=len(valid)>=max(100,int(.95*bootstrap)) and np.isfinite(estimate[j])
                lo,hi=np.quantile(valid,[.025,.975]) if enough else ('','')
                row[metric+'_CI_Lower']=float(lo) if enough else ''
                row[metric+'_CI_Upper']=float(hi) if enough else ''
                row[metric+'_CI_Status']='OK' if enough else 'insufficient_or_undefined'
            all_results.append(row);panel.append(row)
        panels.append((construct,panel))
    stem='master_table_available' if available_data else 'master_table'
    write_table(out/(stem+'.csv'),all_results)
    note=('Human–human pairwise agreement = mean of h1–h2, h1–h3 and h2–h3 statistics. Human–other-rater consensus agreement = mean of each human’s agreement with the mean of the other two humans; a human never contributes to their own reference. Both summaries are recomputed within every bootstrap replicate, averaging the three statistics rather than pooling observations. '
          'Each system has two rows: three-rater consensus uses the mean of all three humans; two-rater consensus averages the three agreement statistics against the three possible two-human means. These statistics are recomputed within every bootstrap replicate, not calculated after pooling or averaging the reference means. Compare the two-rater system row with human–other-rater consensus on the matched cohort for the same references and recordings. Available-data rows may use different recordings, so those human/model differences are not a matched comparison. '
          'Within each construct, all available methods and humans use the same complete recordings. Pending methods are blank; tables are provisional until all requested methods have results. '
          '95% intervals are percentile speaker-cluster bootstrap intervals, conditional on these raters and fitted models; they are not intervals for differences or multiple-comparison-adjusted tests. '
          'The default treats each distinct Drive ID as one speaker. Use --speaker-column COLUMN if a speaker has multiple recordings. '
          'QWK uses fixed 0–100 integer categories, half-up rounding; API scores are multiplied by 10 before analysis. Predictions are capped at 0–100 for QWK only, including bootstrap replicates; raw scores are retained for Spearman. Clipped counts refer to the displayed row cohort. No calibration is fitted. '
          'SpeechOcean scores are imported from the original s- columns; their API version is unspecified. '
          'No ± run variability is shown because these are single fitted models.')
    if available_data:
        note=note.replace('Within each construct, all available methods and humans use the same complete recordings.', 'Each method uses all its available scores with three complete human ratings. Both human baselines use all complete human rows. Samples can differ between methods; use the matched table for direct comparisons.')
    title='Available-data master tables' if available_data else 'Matched-cohort master tables'
    md=['# '+title,'',note,''];web=['<!doctype html><meta charset="utf-8"><title>Stella master comparison</title>',
        '<style>body{font:18px Georgia,serif;max-width:1100px;margin:40px auto;padding:20px}table{border-collapse:collapse;width:100%;margin:20px 0 40px}th,td{padding:9px;text-align:right}th:first-child,td:first-child{text-align:left}thead{border-top:2px solid;border-bottom:1px solid}tbody{border-bottom:2px solid}p{font-size:15px;line-height:1.5}</style>', '<h1>'+title+'</h1><p>'+html.escape(note)+'</p>']
    def fmt(v):return f'{v:.3f}' if isinstance(v,(float,np.floating)) else '—'
    def ci(r,k):return '['+fmt(r[k+'_CI_Lower'])+', '+fmt(r[k+'_CI_Upper'])+']' if r[k+'_CI_Status']=='OK' else '—'
    for construct,panel in panels:
        headers=['Method','N','Spearman’s ρ','95% CI','QWK','95% CI']
        md += ['## '+construct,'','| '+' | '.join(headers)+' |','|---|---:|---:|---|---:|---|']
        web += ['<h2>'+construct+'</h2><table><thead><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in headers)+'</tr></thead><tbody>']
        for r in panel:
            label=r['Method']+(' (pending)' if r['Status']=='pending_no_scores' else '')
            cells=[label,str(r['N']),fmt(r['Spearman']),ci(r,'Spearman'),fmt(r['QWK']),ci(r,'QWK')]
            md.append('| '+' | '.join(cells)+' |')
            web.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+'</tr>')
        clipping=[f"{r['Method']}: {r['QWKClippedCount']} ({r['QWKClippedBelow']} below 0; {r['QWKClippedAbove']} above 100)" for r in panel if r['QWKClippedCount'] and r.get('Reference')!='two']
        clipping_note='QWK clipping: '+('; '.join(clipping)+'. All other available methods: 0.' if clipping else 'No predictions required clipping among available methods.')
        clipping_note+=' Counts refer to this table’s cohort; raw scores are retained for Spearman.'
        md.extend(['',clipping_note,'']);web.append('</tbody></table><p>'+html.escape(clipping_note)+'</p>')
    (out/(stem+'.md')).write_text('\n'.join(md)+'\n')
    (out/(stem+'.html')).write_text('\n'.join(web)+'\n')
