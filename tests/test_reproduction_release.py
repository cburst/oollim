import json,hashlib,subprocess,sys
from pathlib import Path
from reproduce.external_tables import render
ROOT=Path(__file__).resolve().parents[1]

def test_compact_table_matches_manuscript_and_preserves_all_rows(tmp_path):
 rows=render(ROOT/'results/selection-comparison-v1/external-matched.json',tmp_path)
 assert len(rows)==9 and all(len(r)==7 for r in rows)
 assert len(rows) == 9
 assert len((tmp_path/'supplementary.csv').read_text().splitlines())==55

def test_selected_checkpoint_identities():
 m=json.loads((ROOT/'configs/selected-constructs.json').read_text())
 for v in m['constructs'].values():
  p=ROOT/v['checkpoint']
  assert hashlib.sha256(p.read_bytes()).hexdigest()==v['checkpoint_sha256']
  assert hashlib.sha256(p.with_suffix('.json').read_bytes()).hexdigest()==v['metadata_sha256']

def test_complete_aggregate_sweep():
 rows=json.loads((ROOT/'results/ablation-mac-v1/comparison.json').read_text())
 assert len(rows)==945
 assert len({(r['family'],r['construct'],r['blocks'],r['evaluation_corpus']) for r in rows})==945
 decisions=json.loads((ROOT/'results/ablation-mac-v1/decisions.json').read_text())
 assert len(decisions)==21
 assert {(r['family'],r['construct']) for r in decisions if r['promoted']}=={('aesop','fluency'),('aesop','prosody')}

def test_incomplete_sweep_rejected(tmp_path):
 (tmp_path/'comparison.json').write_text('[]')
 result=subprocess.run([sys.executable,'-m','reproduce.select_ablation','--results',str(tmp_path),'--output',str(tmp_path/'out')],cwd=ROOT,capture_output=True,text=True)
 assert result.returncode and 'Incomplete' in result.stderr


def test_all_selection_heads_and_complete_aggregate_release():
 import csv
 registry=json.loads((ROOT/'models/selection-comparison-v1/registry.json').read_text())
 assert len(registry)==21
 for config in registry.values():
  for asset in config['heads'].values():
   p=ROOT/asset['path']
   assert hashlib.sha256(p.read_bytes()).hexdigest()==asset['sha256']
   assert hashlib.sha256(p.with_suffix('.json').read_bytes()).hexdigest()==asset['metadata_sha256']
 with (ROOT/'results/selection-comparison-v1/external-all-conditions.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==144 and all(r['N']=='305' for r in rows)
 assert not any(k in rows[0] for k in ('URL','filename','ID','audio_sha256','evaluation_drive_id'))


def test_leaders_include_all_ties_and_match_manuscript(tmp_path):
 from reproduce.leader_tables import render
 render(ROOT/'results/selection-comparison-v1/external-all-conditions.csv',tmp_path)
 import csv
 rows=list(csv.DictReader((tmp_path/'leaders.csv').open()))
 assert len(rows)==16
 assert (tmp_path/'leader-table-rows.tex').read_text().strip() in (ROOT/'paper/main.tex').read_text()
 assert sum(row['Rule']!='--' and row['SpearmanLeader']=='True' for row in rows)==7
 assert sum(row['Rule']!='--' and row['QWKLeader']=='True' for row in rows)==3
