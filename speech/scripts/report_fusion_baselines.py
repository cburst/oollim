"""Generate a concise research report from completed, versioned fusion results."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    report=ROOT/'reports/fusion-v1'
    data=json.loads((report/'comparison.json').read_text())
    assert data['status']=='complete' and not data['test_evaluated']
    rows=data['results']
    names=['xlsr','xeus','phoneticxeus','xeus+xlsr','xlsr+phoneticxeus',
           'xeus+phoneticxeus','xeus+xlsr+phoneticxeus']
    display={'xlsr':'XLS-R','xeus':'XEUS','phoneticxeus':'PhoneticXeus'}
    label=lambda name:' + '.join(display.get(x,x.upper()) for x in name.split('+'))
    lines=['# Frozen encoder fusion and IPA baseline results v1','',
        'Completed on the fixed SpeechOcean training/validation split. Encoders stayed',
        'frozen; new ridge scorers were fit to 2,000 recordings and selected on 500',
        'recordings from 25 different speakers. No final-test evaluation, Azure calls',
        'or ICNALE data. All 15 coefficient checkpoints passed save/restore parity.','',
        '**Interpretation limit:** these are selected validation results, not independent',
        'test estimates. Upstream PhoneticXeus configuration references SpeechOcean-derived',
        'data; exact exposure of the pinned checkpoint is unresolved. An advantage',
        'does not establish superior unseen-data generalization or SOTA.','',
        '## Individual and combined embeddings','',
        'Native 0–10 mean absolute error; lower is better. Each variant selects its',
        'ridge penalty by accuracy MAE, so fluency/prosody are secondary outcomes.','',
        '| Frozen feature set | Accuracy | Fluency | Prosody | Macro MAE | Alpha |',
        '|---|---:|---:|---:|---:|---:|']
    for name in names:
        row=rows[name];m=row['metrics']
        lines.append(f"| {label(name)} | {m['accuracy']['mae']:.4f} | {m['fluency']['mae']:.4f} | {m['prosody']['mae']:.4f} | {row['macro_mae']:.4f} | {row['selected_alpha']:g} |")
    lines+=['','## Does IPA evidence add value?','',
        'The same embedding features with and without the fixed 17 phone diagnostics.',
        'Negative delta favors adding IPA evidence. Intervals are descriptive paired',
        'speaker bootstraps after validation selection, without multiplicity correction.','',
        '| Feature set | Accuracy without IPA | With IPA | Delta | Descriptive 95% interval |',
        '|---|---:|---:|---:|---|']
    for name in names:
        plus=name+'+ipa';a=rows[name]['metrics']['accuracy']['mae'];b=rows[plus]['metrics']['accuracy']['mae']
        c=data['contrasts'][plus+' minus '+name]
        lines.append(f"| {label(name)} | {a:.4f} | {b:.4f} | {b-a:+.4f} | [{c['low'][0]:+.4f}, {c['high'][0]:+.4f}] |")
    ip=rows['ipa']['metrics'];mean=data['mean_baseline']
    lines+=['',f"IPA-only control MAE: accuracy {ip['accuracy']['mae']:.4f}, fluency {ip['fluency']['mae']:.4f}, prosody {ip['prosody']['mae']:.4f}.",
        f"Training-mean control MAE: accuracy {mean['accuracy']['mae']:.4f}, fluency {mean['fluency']['mae']:.4f}, prosody {mean['prosody']['mae']:.4f}.",'',
        'All scoring dimensions with the IPA block included:','',
        '| Embeddings + IPA | Accuracy | Fluency | Prosody | Macro MAE |',
        '|---|---:|---:|---:|---:|']
    for name in names:
        row=rows[name+'+ipa'];m=row['metrics']
        lines.append(f"| {label(name)} + IPA | {m['accuracy']['mae']:.4f} | {m['fluency']['mae']:.4f} | {m['prosody']['mae']:.4f} | {row['macro_mae']:.4f} |")
    lines+=['','## Selection and limits','']
    best=min(rows,key=lambda k:rows[k]['metrics']['accuracy']['mae'])
    lines+=[f"Lowest screened validation accuracy MAE: **{label(best)}**, {rows[best]['metrics']['accuracy']['mae']:.4f}.",
        'This is a candidate for subsequent evaluation, not a deployment decision.',
        'The same validation set selected alpha and compared the feature sets; preserve',
        'the final test for a frozen model-selection decision rather than another search.','',
        'Canonical reference phones are supplied read-aloud inputs. No human phone',
        'ratings, perceived transcriptions or manual boundaries were fed to the model.',
        'The IPA mapping is approximate and does not normalize every acceptable accent',
        'variant. Edit distances and CTC likelihoods are uncalibrated evidence, not',
        'individually validated error diagnoses. Their incremental value is measured',
        'only by the above fixed ablation. This adds both phone-model evidence and',
        'reference-dependent information to audio-only embedding baselines; it does',
        'not isolate the value of decoded IPA symbols alone.','',
        'The corpus contains Mandarin-L1 speakers, children and adults. Korean learners',
        'and spontaneous speech require separate evaluation. PhoneticXeus is derived',
        'from XEUS; combining them does not imply independent representations.','',
        'Age-group accuracy MAE for the reference and lowest screened candidate:',
        '', '| Model | Under 18 | Adults |', '|---|---:|---:|']
    for name in dict.fromkeys(['xeus',best]):
        groups=rows[name]['age_groups']
        lines.append(f"| {label(name)} | {groups['under18']['metrics']['accuracy']['mae']:.4f} | {groups['adult']['metrics']['accuracy']['mae']:.4f} |")
    lines+=['','Validation includes seven under-18 speakers (140 recordings) and 18 adult',
        'speakers (360 recordings), whereas training is approximately balanced by age.',
        'Subgroups are descriptive and use the same globally selected checkpoints.',
        'No exact reference-text overlap was found between train and validation.','',
        '## Verification','',
        'All 2,500 source audio hashes and feature-cache receipts passed. Cache IDs',
        'exactly match the development split; no final-test features are present.',
        'CPU/Metal agreement passed across 375 overlapping recordings with identical',
        'greedy token sequences. The new solver reproduces original XEUS/XLS-R',
        'predictions at their original penalties to within 2e-13. All 28 tests pass.',
        'All 15 saved checkpoints passed prediction restoration; a separate audit',
        'verified checkpoint hashes, source target linkage and reported MAEs.','',
        '## Artifacts','',
        '- [Frozen protocol](FUSION-BASELINE-PROTOCOL.md)',
        '- [Full metrics, candidates, comparisons and provenance](../reports/fusion-v1/comparison.json)',
        '- [Upstream exposure evidence](../reports/fusion-v1/upstream-exposure-audit.json)',
        '- Checkpoints and individual predictions remain in local `resources/fusion-v1/`.',
        '',f"Phone-reference coverage: `{json.dumps(data['phone_coverage'],sort_keys=True)}`."]
    (ROOT/'docs/FUSION-BASELINE-RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(best_screened_accuracy=best,accuracy_mae=rows[best]['metrics']['accuracy']['mae'])))


if __name__=='__main__':main()
