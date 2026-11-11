#!/usr/bin/env python3
"""As asseracoes do portao do X4, antes de qualquer avaliacao.

Falha ruidosamente. Nenhuma avaliacao roda se alguma delas falhar.
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np

# Relative to the repository root by default, so the file carries no personal path.
W = Path(os.environ.get("X4_WORK", "work/X4"))
EXPECTED = {'train': {'rows': 2500, 'covered': 2478, 'incomplete': 22},
            'test':  {'rows': 2500, 'covered': 2483, 'incomplete': 17}}
LAYER_PLAN = {'xeus': 19, 'xlsr': 25, 'phoneticxeus': 20}
WIDTH = 2048

failures, lines = [], []
def check(label, got, want):
    ok = got == want
    lines.append(f"{'ok  ' if ok else 'FALHA'}  {label}: {got!r}" + ('' if ok else f"  esperado {want!r}"))
    if not ok:
        failures.append(label)
    return ok

for part in ('train', 'test'):
    store = W / f'{part}-features'
    m = json.loads((store / 'manifest.json').read_text())
    rows = m['rows']
    want = EXPECTED[part]
    check(f'{part}: schema', m['schema_version'], 'speech-feature-cache-v1')
    check(f'{part}: linhas no manifesto', len(rows), want['rows'])
    covered = sum(1 for r in rows if r['reference_complete'] and r['text_complete'])
    check(f'{part}: cobertas', covered, want['covered'])
    check(f'{part}: incompletas', len(rows) - covered, want['incomplete'])
    # text_complete falso implica reference_complete falso
    bad = [r['recording_id'] for r in rows if (not r['text_complete']) and r['reference_complete']]
    check(f'{part}: text_complete falso com reference_complete verdadeiro', len(bad), 0)
    # ordem e identidade do features.npz
    with np.load(store / 'features.npz') as f:
        ids = [str(v) for v in f['recording_ids']]
        check(f'{part}: ordem do features.npz', ids == [r['recording_id'] for r in rows], True)
        for b, w in (('xeus', WIDTH), ('xlsr', WIDTH), ('phoneticxeus', WIDTH), ('ipa', 17)):
            check(f'{part}: forma do bloco {b}', tuple(f[b].shape), (want['rows'], w))
            check(f'{part}: finitude do bloco {b}', bool(np.isfinite(f[b]).all()), True)
    # camadas e cross-layer
    for enc, n in LAYER_PLAN.items():
        a = np.load(store / m['layer_files'][enc], mmap_mode='r')
        check(f'{part}: forma de layers_{enc}', tuple(a.shape), (want['rows'], n, WIDTH))
        c = np.load(store / m['crosslayer_files'][enc], mmap_mode='r')
        check(f'{part}: forma de crosslayer_{enc}', tuple(c.shape), (want['rows'], 2, WIDTH))
    # logits por linha
    logits = store / 'logits'
    missing = [r['recording_id'] for r in rows
               if not (logits / (hashlib.sha256(r['recording_id'].encode()).hexdigest() + '.npz')).exists()]
    check(f'{part}: logits ausentes', len(missing), 0)
    check(f'{part}: arquivos em logits', len(list(logits.iterdir())), want['rows'])
    # alvos presentes e na escala
    t = np.array([r['targets'] for r in rows], float)
    check(f'{part}: forma dos alvos', tuple(t.shape), (want['rows'], 3))
    check(f'{part}: alvos em 0 a 10', bool((t >= 0).all() and (t <= 10).all()), True)

# os dois manifestos nao compartilham gravacao
a = {r['recording_id'] for r in json.loads((W/'train-features'/'manifest.json').read_text())['rows']}
b = {r['recording_id'] for r in json.loads((W/'test-features'/'manifest.json').read_text())['rows']}
check('nenhuma gravacao nas duas particoes', len(a & b), 0)
# nem falante
sa = {r['speaker_id'] for r in json.loads((W/'train-features'/'manifest.json').read_text())['rows']}
sb = {r['speaker_id'] for r in json.loads((W/'test-features'/'manifest.json').read_text())['rows']}
check('nenhum falante nas duas particoes', len(sa & sb), 0)

print('\n'.join(lines))
print(f"\n{len(lines) - len(failures)} de {len(lines)} asseracoes passaram")
if failures:
    print('FALHARAM: ' + ', '.join(failures))
    sys.exit(1)
print('PORTAO_ABERTO')
