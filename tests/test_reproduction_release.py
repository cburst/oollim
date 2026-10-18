import json,hashlib,subprocess,sys
from pathlib import Path
from reproduce.external_tables import render
ROOT=Path(__file__).resolve().parents[1]

def test_compact_table_matches_manuscript_and_preserves_all_rows(tmp_path):
 rows=render(ROOT/'results/external-matched-full-2026-10-18.json',tmp_path)
 assert len(rows)==9 and all(len(r)==7 for r in rows)
 assert (tmp_path/'main-table-rows.tex').read_text().strip() in (ROOT/'paper/main.tex').read_text()
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
