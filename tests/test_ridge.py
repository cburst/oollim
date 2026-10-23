import numpy as np
import pytest
from reproduce.train import weighted_fit

@pytest.mark.parametrize('n,d',[(8,13),(15,4)])
def test_solver_matches_direct_weighted_normal_equations(n,d):
 r=np.random.default_rng(42);x=r.normal(size=(n,d));x[:,0]=7;y=r.normal(size=(n,3));w=r.uniform(size=n);w/=w.sum();gain=np.linspace(.5,2,d)
 mean,scale,intercept,coefs=weighted_fit(x,y,w,[.01,1,100],gain)
 z=(x-mean)/scale*gain
 for a,c in coefs.items():
  direct=np.linalg.solve(z.T@(w[:,None]*z)+a*np.eye(d),z.T@(w[:,None]*(y-intercept)))
  np.testing.assert_allclose(z@c+intercept,z@direct+intercept,atol=1e-9)

def test_replication_invariance():
 r=np.random.default_rng(4);x=r.normal(size=(9,4));y=r.normal(size=(9,3));w=np.ones(9)/9;gain=np.ones(4)
 a=weighted_fit(x,y,w,[.1],gain);b=weighted_fit(np.repeat(x,2,0),np.repeat(y,2,0),np.repeat(w/2,2),[.1],gain)
 for i in range(3):np.testing.assert_allclose(a[i],b[i])
 np.testing.assert_allclose(a[3][.1],b[3][.1])

def test_reject_invalid_weights():
 with pytest.raises(ValueError):weighted_fit(np.ones((2,3)),np.ones((2,3)),np.ones(2),[1],np.ones(3))

from pathlib import Path
import json
from reproduce.train import save,digest,load_corpus,run,FAMILIES

def make_corpus(path,name,seed):
 r=np.random.default_rng(seed);n=12;features={k:r.normal(size=(n,d)) for k,d in [('xeus',3),('xlsr',3),('phoneticxeus',3),('ipa',2)]};ids=[name+str(i) for i in range(n)];path.mkdir()
 fp=path/'features.npz';np.savez(fp,recording_ids=np.array(ids),**features)
 rows=[dict(recording_id=rid,speaker_id=rid,split='train' if i<8 else 'validation',audio_sha256=rid,targets=[float(i%8+1),float((i*3)%9),float((i*5)%9)]) for i,rid in enumerate(ids)]
 save(path/'manifest.json',dict(schema_version='speech-feature-cache-v1',target_order=['accuracy','fluency','prosody'],target_scale='0..10',features='features.npz',features_sha256=digest(fp),rows=rows))
 return path/'manifest.json'

def test_full_seven_family_save_restore(tmp_path):
 corpora={n:load_corpus(make_corpus(tmp_path/n,n,i)) for i,n in enumerate(('speechocean','korean','aesop'))}
 result=run(corpora,tmp_path/'out')
 assert len(result)==63
 for cp in (tmp_path/'out').rglob('ridge.npz'):
  m=json.loads(cp.with_suffix('.json').read_text());assert m['checkpoint_sha256']==digest(cp)
  family,construct=cp.parts[-4:-2];j=['accuracy','fluency','prosody'].index(construct)
  with np.load(cp) as f:
   for n,c in corpora.items():
    x=np.concatenate([c['features'][b] for b in m['blocks']],1)[~c['train']]
    p=(((x-f['scaler_mean'])/f['scaler_scale']*f['block_gain'])@f['coef']+f['intercept'])[:,j]
    with np.load(cp.parent/(n+'-development.npz')) as saved:np.testing.assert_allclose(p,saved['prediction'])

def test_split_and_cache_integrity(tmp_path):
 p=make_corpus(tmp_path/'so','so',1);m=json.loads(p.read_text());m['rows'][-1]['speaker_id']=m['rows'][0]['speaker_id'];save(p,m)
 with pytest.raises(ValueError,match='leakage'):load_corpus(p)
 m['rows'][-1]['speaker_id']='unique';m['rows'][0]['split']='test';save(p,m)
 with pytest.raises(ValueError,match='forbidden'):load_corpus(p)
 m['rows'][0]['split']='train';m['features_sha256']='changed';save(p,m)
 with pytest.raises(ValueError,match='checksum'):load_corpus(p)

def test_original_sum_loss_solver():
 from reproduce.original import fit_candidates
 r=np.random.default_rng(6);x=r.normal(size=(10,4));y=r.normal(size=(10,3));v=r.normal(size=(3,4));xm=x.mean(0);ym=y.mean(0)
 for a,c,b,p in fit_candidates(x,y,v,[.1,10]):
  expected=np.linalg.solve((x-xm).T@(x-xm)+a*np.eye(4),(x-xm).T@(y-ym))
  np.testing.assert_allclose(c,expected,atol=1e-10);np.testing.assert_allclose(p,(v-xm)@expected+ym,atol=1e-10)

def test_numerical_parity_with_archived_family_runner():
 import ast
 source=Path(__file__).parents[1]/'speech/scripts/run_seven_families.py'
 tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='weighted_fit');namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
 r=np.random.default_rng(72);x=r.normal(size=(12,7));y=r.normal(size=(12,3));w=r.uniform(size=12);w/=w.sum();gain=np.linspace(.2,3,7)
 old=namespace['weighted_fit'](x,y,w,[.001,10],gain);new=weighted_fit(x,y,w,[.001,10],gain)
 for i in range(3):np.testing.assert_array_equal(old[i],new[i])
 for a in (.001,10):np.testing.assert_array_equal(old[3][a],new[3][a])


def test_selection_condition_ties_and_normalized_historical_grid():
 from reproduce.train import select_candidate,HISTORICAL_ALPHAS
 candidates=[{'alpha':1.,'spearman':.9,'mae':.7},{'alpha':10.,'spearman':.8,'mae':.6},{'alpha':100.,'spearman':.7,'mae':.6}]
 assert select_candidate(candidates,'spearman')['alpha']==1
 assert select_candidate(candidates,'damae')['alpha']==100
 assert HISTORICAL_ALPHAS==[.00005,.0005,.005,.05,.5,5.]


def test_damae_shares_accuracy_selected_penalty_for_fluency_prosody(tmp_path):
 corpora={n:load_corpus(make_corpus(tmp_path/n,n,i)) for i,n in enumerate(('speechocean','korean','aesop'))}
 for condition in ('damae','historical_grid_damae'):
  out=tmp_path/condition;run(corpora,out,condition=condition)
  for family in FAMILIES:
   f=json.loads((out/family/'fluency/xeus+xlsr/selection.json').read_text())
   p=json.loads((out/family/'prosody/xeus+xlsr/selection.json').read_text())
   assert f==p
   assert f['selected']==min(f['candidates'],key=lambda c:(c['mae'],-c['alpha']))
