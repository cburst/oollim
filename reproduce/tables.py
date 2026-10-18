"""Render completed aggregate development results; no private data access."""
import json
from pathlib import Path
from .train import FAMILIES,CONSTRUCTS

def main():
 root=Path(__file__).resolve().parents[1];rows=json.loads((root/'results/seven-families-v1.json').read_text());lookup={(r['family'],r['construct'],r['evaluation_corpus']):r for r in rows}
 if len(lookup)!=63:raise ValueError('Expected 63 unique completed comparisons')
 text='# Matched seven-family development results\n\nFixed representations; included-corpus development alpha selection. These are not final tests. SpeechOcean here is a fresh refit, not the original deployed checkpoint.\n'
 for c in CONSTRUCTS:
  text+='\n## '+c.capitalize()+'\n\n| Family | SpeechOcean (500) | Korean (636) | J-AESOP (36) |\n|---|---:|---:|---:|\n'
  for f in FAMILIES:text+='| '+f+' | '+' | '.join(f"{lookup[f,c,k]['spearman']:.4f}" for k in ('speechocean','korean','aesop'))+' |\n'
 text+='\nValues: Spearman correlation. Full MAE/RMSE/Pearson values are in the adjacent JSON.\n'
 (root/'results/seven-families-v1.md').write_text(text)
 print('Rendered all 63 comparisons.')
if __name__=='__main__':main()
