"""Fixed explicit-feature controls; development data only, no parameter search."""
import argparse,json,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pronunciation.audio import read_audio
from pronunciation.explicit import extract,NAMES,VERSION
from pronunciation.development import digest,development_rows,metrics,paired_speaker_bootstrap,create_run,finish_run


def main():
    p=argparse.ArgumentParser();p.add_argument('--id',default='explicit-baseline-v2-001');args=p.parse_args()
    audit=json.loads((ROOT/'reports/reference-local-v2-001/speechocean-audit.json').read_text());manifest=ROOT/audit['development_manifest']
    assert not audit['errors'] and digest(manifest)==audit['development_manifest_sha256']
    rows=[json.loads(s) for s in manifest.read_text().splitlines()];train=development_rows(rows,split='train');valid=development_rows(rows,split='validation')
    assert rows==train+valid
    configuration=dict(protocol_id='speech-evaluation-workflow-v2.1',datasets=['speechocean762'],split_sha256=audit['split_sha256'],
        encoders=['phoneticxeus_for_existing_IPA_evidence_only'],features={'acoustic':list(NAMES),'phone':'existing text-reference 17-feature block'},fusion='explicit concatenation',
        targets=['accuracy','fluency','prosody'],loss='ridge alpha=10; HGB squared_error 100 iterations max_leaf_nodes=7 l2=10 learning_rate=.05 early_stopping=False',
        seeds=[42,43,44],code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_hashes={s:digest(ROOT/s) for s in ['scripts/explicit_baseline.py','pronunciation/explicit.py','pronunciation/development.py']},
        environment={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},selection_rule='fixed exploratory controls; no hyperparameter selection or winner decision')
    run=create_run(ROOT/'reports/experiment-registry',args.id,configuration);started=time.perf_counter()
    cache=ROOT/'resources'/args.id;cache.mkdir(exist_ok=False)
    acoustic=[]
    for i,row in enumerate(rows):
        path=ROOT/row['audio_path'];assert digest(path)==row['sha256']
        acoustic.append(extract(read_audio(path).numpy(),row['reference_text']))
        if i%250==0:print('acoustic',i,flush=True)
    acoustic=np.stack(acoustic)
    prior=json.loads((ROOT/'reports/text-reference-v1/comparison.json').read_text());phone_root=ROOT/'resources/text-reference-v1'/prior['run_key']
    assert json.loads((phone_root/'provenance.json').read_text())==prior['provenance']
    provenance=prior['provenance']
    for old,new in [('manifest_sha256','source_manifest_sha256'),('split_sha256','split_sha256'),('labels_sha256','labels_sha256')]:
        assert provenance[old]==audit[new], 'IPA provenance differs from development audit'
    source=ROOT/'resources/corpus-manifest.jsonl'
    assert digest(source)==provenance['manifest_sha256']
    original={r['recording_id']:r for r in map(json.loads,source.read_text().splitlines())}
    for row in rows:
        assert all(row[k]==original[row['recording_id']][k] for k in ('sha256','reference_text','split','speaker_id')), 'IPA source identity changed'
    phone_ids=json.loads((phone_root/'phone-coverage.json').read_text())
    with np.load(phone_root/'ipa-features.npz',allow_pickle=False) as f:phone=f['features'].copy()
    lookup={r['recording_id']:i for i,r in enumerate(phone_ids)}
    assert len(lookup)==len(phone_ids)==len(phone)==len(rows)
    assert all(phone_ids[lookup[r['recording_id']]]['split']==r['split'] for r in rows)
    phone=phone[[lookup[r['recording_id']] for r in rows]]
    if not np.isfinite(phone).all():raise ValueError('Nonfinite prior phone evidence')
    x=np.concatenate([acoustic,phone],axis=1)
    y=np.array([[r['labels']['prosodic' if d=='prosody' else d] for d in configuration['targets']] for r in rows])
    n=len(train);groups=['speechocean762:'+r['speaker_id'] for r in valid]
    np.savez(cache/'features.npz',x=x,y=y,recording_ids=np.array([r['recording_id'] for r in rows]))
    feature_seconds=time.perf_counter()-started
    results={};predictions={};models={};complexity={};training_seconds=0
    for block,features in [('acoustic',acoustic),('phone',phone),('combined',x)]:
        for algorithm in ('ridge','hgb'):
            for seed in ([42] if algorithm=='ridge' else [42,43,44]):
                key=f'{block}-{algorithm}-{seed}';pred=[];models[key]=[]
                for j in range(3):
                    estimator=Ridge(alpha=10) if algorithm=='ridge' else HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=seed)
                    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),estimator)
                    began=time.perf_counter();model.fit(features[:n],y[:n,j]);training_seconds+=time.perf_counter()-began
                    pred.append(model.predict(features[n:]));models[key].append(model)
                pred=np.stack(pred,axis=1);predictions[key]=pred;results[key]={d:metrics(y[n:,j],pred[:,j]) for j,d in enumerate(configuration['targets'])}
                complexity[key]=[{'coefficients_including_intercept':int(m[-1].coef_.size+1)} if algorithm=='ridge' else
                    {'trees':len(m[-1]._predictors),'nodes':sum(len(t[0].nodes) for t in m[-1]._predictors)} for m in models[key]]
    import joblib
    joblib.dump(models,cache/'models.joblib')
    restored=joblib.load(cache/'models.joblib')
    for key,group in restored.items():
        features={'acoustic':acoustic,'phone':phone,'combined':x}[key.split('-')[0]]
        for j,model in enumerate(group):np.testing.assert_allclose(model.predict(features[n:]),predictions[key][:,j],rtol=0,atol=0)
    comparisons={d:paired_speaker_bootstrap(y[n:,j],predictions['combined-ridge-42'][:,j],predictions['acoustic-ridge-42'][:,j],groups) for j,d in enumerate(configuration['targets'])}
    np.savez(cache/'predictions.npz',**predictions)
    finish_run(run,dict(validation_metrics=results,test_metrics=None,parameter_count=None,model_complexity=complexity,training_seconds=training_seconds,feature_seconds=feature_seconds,total_seconds=time.perf_counter()-started,checkpoint=str((cache/'models.joblib').relative_to(ROOT)),
        checkpoint_sha256=digest(cache/'models.joblib'),feature_sha256=digest(cache/'features.npz'),ipa_source_sha256=digest(phone_root/'ipa-features.npz'),
        combined_vs_acoustic_ridge=comparisons,checkpoint_parity=True,
        notes='HGB runs can be identical because this fixed configuration is effectively deterministic; seeds do not create independent evidence. Energy and pitch features are proxies. No human phone annotations supplied as inputs. No claimed robust model winner.'))
    print(json.dumps(results),flush=True)

if __name__=='__main__':main()
