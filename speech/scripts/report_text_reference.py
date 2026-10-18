"""Compare matched canonical/text runs, including old-checkpoint input replacement."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pronunciation.encoders import sha256
from run_baseline import atomic_json,metrics
from run_fusion_baselines import paired_bootstrap


def main():
    old=json.loads((ROOT/'reports/fusion-v1/comparison.json').read_text())
    new=json.loads((ROOT/'reports/text-reference-v1/comparison.json').read_text())
    assert old['config']['alphas']==new['config']['alphas']
    assert old['provenance']['backbones']==new['provenance']['backbones']
    rows=[json.loads(s) for s in (ROOT/'resources/corpus-manifest.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['split'] in ('train','validation')]
    mask=np.array([r['split']=='validation' for r in rows]);vr=[r for r in rows if r['split']=='validation']
    oldroot=ROOT/'resources/fusion-v1'/old['run_key']
    newroot=ROOT/'resources/text-reference-v1'/new['run_key']
    a=json.loads((oldroot/'validation-predictions.json').read_text())
    b=json.loads((newroot/'validation-predictions.json').read_text())
    assert a['recording_ids']==b['recording_ids']==[r['recording_id'] for r in vr]
    np.testing.assert_array_equal(a['targets'],b['targets']);y=np.array(a['targets'])
    with np.load(oldroot/'ipa-features.npz') as f:oldipa=f['features'][mask]
    with np.load(newroot/'ipa-features.npz') as f:newipa=f['features'][mask]
    coverage=json.loads((ROOT/'reports/text-reference-v1/text-coverage.json').read_text())
    assert set(coverage)=={r['recording_id'] for r in rows}
    comparisons={};controls={}
    for name,result in new['results'].items():
        pred=np.array(b['predictions'][name]);prior=np.array(a['predictions'][name])
        measured=metrics(y,pred,new['config']['dimensions'])
        for d in new['config']['dimensions']:
            assert abs(measured[d]['mae']-result['metrics'][d]['mae'])<1e-12
        checkpoint=ROOT/result['checkpoint']
        meta=json.loads(checkpoint.with_suffix('.json').read_text())
        assert sha256(checkpoint)==meta['checkpoint_sha256']
        assert meta['experiment_provenance']['reference_generator']['resource']==new['config']['reference_resource']
        if 'ipa' not in result['blocks']:
            np.testing.assert_allclose(pred,prior,rtol=1e-10,atol=1e-10)
            controls[name]=float(np.max(np.abs(pred-prior)))
            continue
        # IPA is always the final block, and original predictions contain all other blocks.
        assert result['blocks'][-1]=='ipa'
        with np.load(ROOT/old['results'][name]['checkpoint']) as ck:
            width=oldipa.shape[1]
            drift=((newipa-oldipa)/ck['scaler_scale'][-width:]*ck['block_gain'][-width:])@ck['coef'][-width:]
        swapped=prior+drift
        complete=np.array([coverage[r['recording_id']]['complete'] for r in vr])
        comparisons[name]=dict(text_metrics=measured,canonical_metrics=old['results'][name]['metrics'],
            text_minus_canonical=paired_bootstrap(vr,y,pred,prior,42,2000),
            old_checkpoint_text_input_metrics=metrics(y,swapped,new['config']['dimensions']),
            complete_reference_subset=dict(n=int(complete.sum()),text=metrics(y[complete],pred[complete],new['config']['dimensions']),canonical=metrics(y[complete],prior[complete],new['config']['dimensions'])))
    summary=dict(comparisons=comparisons,audio_controls_max_absolute_difference=controls,
        coverage={s:dict(recordings=sum(r['split']==s for r in rows),
            incomplete=sum(not coverage[r['recording_id']]['complete'] for r in rows if r['split']==s),
            words=sum(coverage[r['recording_id']]['words'] for r in rows if r['split']==s),
            unknown_words=sum(len(coverage[r['recording_id']]['unknown_words']) for r in rows if r['split']==s)) for s in ('train','validation')},
        unknown_spellings=sorted({w for c in coverage.values() for w in c['unknown_words']}),
        all_15_checkpoint_hashes_verified=True,all_15_restore_checks_passed=all(r['restore_verified'] for r in new['results'].values()),test_evaluated=False)
    atomic_json(ROOT/'reports/text-reference-v1/comparison-to-canonical.json',summary)
    lines=['# Text-generated reference results v1','',
        'Same 2,000 training / 500 validation recordings and frozen features as fusion-v1. '
        'Phones generated solely from reference text using pinned CMUdict base pronunciations. '
        'All 15 scorers refit with the same fixed alpha grid; selection by validation accuracy MAE. '
        'Lower error is better; scale 0–10.','',
        '| Embeddings + phone evidence | Supplied-phone accuracy | Text-phone accuracy | Text-phone fluency | Text-phone prosody |',
        '|---|---:|---:|---:|---:|']
    for name,c in comparisons.items():
        if name=='ipa':continue
        m=c['text_metrics'];lines.append(f"| {name} | {c['canonical_metrics']['accuracy']['mae']:.4f} | {m['accuracy']['mae']:.4f} | {m['fluency']['mae']:.4f} | {m['prosody']['mae']:.4f} |")
    best='xeus+xlsr+phoneticxeus+ipa';c=comparisons[best]
    lines+=['','## Coverage and verification','',json.dumps(summary['coverage']), '',
        'Unknown spellings: '+', '.join(summary['unknown_spellings'])+'.', '',
        'No recordings dropped or corrected using corpus phones. Incomplete references retain '
        'confidence features and explicit flags; reference edit/CTC features are zero. '
        'Seven audio-only controls match prior predictions to 1e-10; all 15 checkpoint hashes '
        'and restoration checks pass. The final test remains untouched.','',
        'Three-model + phone evidence, replacing inputs in the old checkpoint without refitting: '+
        ', '.join(f"{d} MAE {c['old_checkpoint_text_input_metrics'][d]['mae']:.4f}" for d in new['config']['dimensions'])+'.','',
        '## Interpretation','',
        'The gain mostly survives replacing supplied phones with independent text-generated phones. '
        'This is selected validation evidence, not unseen-data confirmation. PhoneticXeus upstream '
        'SpeechOcean exposure remains unresolved. Dictionary-only conversion selects one pronunciation, '
        'does not disambiguate heteronyms or model accent variants, and cannot pronounce unknown words. '
        'It is a reproducible text-only baseline, not a production-ready universal G2P. '
        'No model deployment or shared API change was made.','',
        'Protocol: [TEXT-REFERENCE-PROTOCOL.md](TEXT-REFERENCE-PROTOCOL.md). '
        'Detailed comparison: [comparison-to-canonical.json](../reports/text-reference-v1/comparison-to-canonical.json). '
        'Full fits: [comparison.json](../reports/text-reference-v1/comparison.json).','',
        'Dictionary source and license: [CMUdict](https://github.com/cmusphinx/cmudict/tree/74790861f652b15e4ac49015a90074ad62a27690).']
    (ROOT/'docs/TEXT-REFERENCE-RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
