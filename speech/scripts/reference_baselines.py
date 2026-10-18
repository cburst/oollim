"""Register development-only B0 controls and reevaluate the existing ridge fairly."""
import argparse,json,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pronunciation.development import digest,development_rows,metrics,paired_speaker_bootstrap,create_run,finish_run
from pronunciation.fusion import UnifiedScorer
from fusion_data import load_data


def main():
    p=argparse.ArgumentParser();p.add_argument('--id',default='reference-baselines-v2-001');p.add_argument('--replicates',type=int,default=2000);args=p.parse_args()
    audit=json.loads((ROOT/'reports/reference-local-v2-001/speechocean-audit.json').read_text())
    if audit['errors']:raise ValueError('Audio audit has failures')
    manifest=ROOT/audit['development_manifest'];assert digest(manifest)==audit['development_manifest_sha256']
    canonical=[json.loads(s) for s in manifest.read_text().splitlines()]
    train=development_rows(canonical,split='train');valid=development_rows(canonical,split='validation')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    configuration=dict(protocol_id='speech-evaluation-workflow-v2.1',datasets=['speechocean762'],split_sha256=audit['split_sha256'],
        encoders=['xeus','xlsr','phoneticxeus'],features='prior text-reference fusion cache; B0 uses no features',fusion='prior ridge control',
        targets=['accuracy','fluency','prosody'],loss='no refit of ridge; mean/median train-only B0',seeds=[42],code_commit=commit,
        code_hashes={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'pronunciation/development.py']},
        environment={'python':platform.python_version(),'numpy':np.__version__},selection_rule='development Spearman per construct; no selection or threshold tuning in this control run',
        bootstrap_replicates=args.replicates,provenance_caveat='historical development-selected ridge; PhoneticXeus upstream exposure unresolved; not confirmatory')
    run=create_run(ROOT/'reports/experiment-registry',args.id,configuration);began=time.perf_counter()
    rows,tr,va,x,y,checkpoint,_=load_data()
    assert [r['recording_id'] for r in rows if r['split']=='train']==[r['recording_id'] for r in train]
    assert [r['recording_id'] for r in rows if r['split']=='validation']==[r['recording_id'] for r in valid]
    scorer=UnifiedScorer(checkpoint);pred=scorer.predict_features(x[va]);speakers=['speechocean762:'+r['speaker_id'] for r in valid]
    results={};predictions=[]
    for j,dimension in enumerate(configuration['targets']):
        mean=np.full(va.sum(),y[tr,j].mean());median=np.full(va.sum(),np.median(y[tr,j]))
        results[dimension]={'training_mean':float(mean[0]),'training_median':float(median[0]),
            'b0_mean':metrics(y[va,j],mean),'b0_median':metrics(y[va,j],median),
            'ridge_vs_mean':paired_speaker_bootstrap(y[va,j],pred[:,j],mean,speakers,replicates=args.replicates)}
    for i,row in enumerate(valid):predictions.append({'dataset':row['dataset'],'recording_id':row['recording_id'],'speaker_id':row['speaker_id'],'split':'validation','ridge':dict(zip(configuration['targets'],pred[i].tolist()))})
    with (run/'predictions.json').open('x') as f:json.dump(predictions,f,indent=2);f.write('\n')
    finish_run(run,dict(validation_metrics=results,test_metrics=None,parameter_count=int(scorer.arrays['coef'].size+scorer.arrays['intercept'].size),
        training_seconds=0,evaluation_seconds=time.perf_counter()-began,checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=digest(checkpoint),
        note='No new neural training; ridge reused as required historical control. Constant B0 correlation and correlation difference are undefined, not zero.'))
    print(json.dumps({'run':str(run),'ridge':{d:results[d]['ridge_vs_mean']['candidate'] for d in results}}))

if __name__=='__main__':main()
