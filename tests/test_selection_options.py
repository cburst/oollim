import json
from pathlib import Path
import numpy as np
from oollim_app.common import ROOT, imports_path, sha
from oollim_app.scoring import FAMILIES, SELECTIONS, bundle_path
from oollim_app.engine import load_bundle, _portable_provenance
from oollim_app.cli import identity

def test_all_35_bundles_match_evaluated_registry():
 registry=json.loads((ROOT/'models/selection-comparison-v1/registry.json').read_text())
 registry.update(json.loads((ROOT/'models/banded-3ratio-v1/registry.json').read_text()))
 assert len(registry)==len(FAMILIES)*len(SELECTIONS)==35
 for selection in SELECTIONS:
  for family in FAMILIES:
   path=bundle_path(family,selection);cfg=load_bundle(path);expected=registry[selection+'__'+family]
   assert cfg['model_version']==expected['model_version']
   assert cfg['family']==family and cfg['selection']==selection
   for dim,h in expected['heads'].items():
    assert sha(path.parent/cfg['heads'][dim])==h['sha256']

def test_provenance_only_ignores_machine_directory():
 original=json.loads((ROOT/'configs/extractor.json').read_text())['experiment_provenance']
 changed=json.loads(json.dumps(original));changed['backbones']['xeus']['backbone']['directory']='C:/elsewhere'
 assert _portable_provenance(original)==_portable_provenance(changed)
 changed['backbones']['xeus']['backbone']['weights_sha256']='wrong'
 assert _portable_provenance(original)!=_portable_provenance(changed)

def test_selection_changes_resume_identity(tmp_path):
 audio=tmp_path/'sample.wav';audio.write_bytes(b'audio')
 assert len({identity(audio,'Please call Stella.',bundle_path('speechocean',s)) for s in SELECTIONS})==5

def test_all_heads_produce_finite_scores():
 imports_path()
 from pronunciation.fusion import UnifiedScorer
 rng=np.random.default_rng(42)
 for selection in SELECTIONS:
  for family in FAMILIES:
   p=bundle_path(family,selection);cfg=load_bundle(p)
   for j,dim in enumerate(('accuracy','fluency','prosody')):
    scorer=UnifiedScorer(p.parent/cfg['heads'][dim])
    result=scorer.predict_features(rng.normal(size=(2,scorer.width)))
    assert result.shape==(2,3) and np.isfinite(result[:,j]).all()

def test_installer_manifest_contains_all_choices():
 files=json.loads((ROOT/'packaging/source-files.json').read_text())
 assert 'oollim_app/scoring.py' in files
 for s in SELECTIONS:
  for f in FAMILIES:
   p=bundle_path(f,s);cfg=load_bundle(p)
   for path in [p]+[p.parent/n for n in cfg['files']]:assert str(path.relative_to(ROOT)) in files
