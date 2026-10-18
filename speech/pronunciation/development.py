"""Development-only split guards, paired speaker uncertainty, immutable runs."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ALLOWED_DATASETS = {'speechocean762', 'l2arctic', 'speak_improve_2025', 'korean_english_educational'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_splits(rows):
    """Validate metadata across all splits, without reading final-test labels/audio."""
    ids, owners, audio_owners = set(), {}, {}
    for row in rows:
        dataset = row['dataset']
        if dataset not in ALLOWED_DATASETS:
            raise ValueError('Dataset not permitted for reference selection: ' + dataset)
        key = (dataset, row['recording_id'])
        if key in ids:
            raise ValueError('Duplicate recording identity')
        ids.add(key)
        if not row.get('speaker_id'):
            raise ValueError('Speaker identity required')
        if row['split'] != 'test' and not row.get('reference_text', '').strip():
            raise ValueError('Supplied development reference text required')
        split = row['split']
        if split not in {'train', 'validation', 'test'}:
            raise ValueError('Unknown split')
        group = row.get('global_speaker_group') or dataset + ':' + row['speaker_id']
        if group in owners and owners[group] != split:
            raise ValueError('Speaker crosses splits')
        owners[group] = split
        audio_hash = row.get('sha256')
        if not isinstance(audio_hash,str) or not re.fullmatch(r'[0-9a-f]{64}',audio_hash):
            raise ValueError('Verified SHA256 audio identity required')
        if audio_hash:
            if audio_hash in audio_owners and audio_owners[audio_hash] != split:
                raise ValueError('Identical audio crosses splits')
            audio_owners[audio_hash] = split
    return {'recordings': len(ids), 'speaker_groups': len(owners)}


def development_rows(rows, *, split):
    if split not in {'train', 'validation'}:
        raise ValueError('Final tests are unavailable to development selection')
    validate_splits(rows)
    return [r for r in rows if r['split'] == split]


def _correlation(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return None if denom == 0 else float(np.clip(np.dot(a, b) / denom, -1, 1))


def metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if y.ndim != 1 or y.shape != pred.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Expected matching finite nonempty vectors')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            return {'mae': float(np.abs(y-pred).mean()), 'rmse': float(np.sqrt(np.square(y-pred).mean())),
                    'pearson': _correlation(y, pred), 'spearman': _correlation(rankdata(y), rankdata(pred))}
    except FloatingPointError as exc:
        raise ValueError('Numerical overflow; check rating scales') from exc


def paired_speaker_bootstrap(y, candidate, baseline, speakers, *, seed=42, replicates=2000):
    """Recording-weighted metrics; resample whole speakers with replacement.

    Both models use the same labelled complete cases and the same sampled clusters.
    Missing cases are counted explicitly. Repeated sampled speakers repeat all rows.
    Undefined constant-vector correlations are null, not zero or evidence of parity.
    """
    y, candidate, baseline = [np.asarray(a, float) for a in (y, candidate, baseline)]
    speakers = np.asarray(speakers, str)
    if y.ndim != 1 or any(a.shape != y.shape for a in (candidate, baseline, speakers)):
        raise ValueError('Input shape mismatch')
    if replicates < 100 or np.any(speakers == ''):
        raise ValueError('Need >=100 replicates and nonempty speaker IDs')
    valid = np.isfinite(y) & np.isfinite(candidate) & np.isfinite(baseline)
    coverage = {'total_rows': len(y), 'paired_rows': int(valid.sum()),
                'missing_label': int((~np.isfinite(y)).sum()),
                'missing_candidate': int((~np.isfinite(candidate)).sum()),
                'missing_baseline': int((~np.isfinite(baseline)).sum())}
    y, candidate, baseline, speakers = [a[valid] for a in (y, candidate, baseline, speakers)]
    groups = np.unique(speakers)
    if len(groups) < 2:
        raise ValueError('Need at least two paired speakers')
    indices = [np.flatnonzero(speakers == group) for group in groups]
    point_a, point_b = metrics(y, candidate), metrics(y, baseline)
    samples = {side: {m: [] for m in point_a} for side in ('candidate', 'baseline', 'delta')}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        ix = np.concatenate([indices[g] for g in rng.integers(len(groups), size=len(groups))])
        a, b = metrics(y[ix], candidate[ix]), metrics(y[ix], baseline[ix])
        for name in a:
            if a[name] is not None: samples['candidate'][name].append(a[name])
            if b[name] is not None: samples['baseline'][name].append(b[name])
            if a[name] is not None and b[name] is not None: samples['delta'][name].append(a[name]-b[name])
    intervals = {}
    for side, values in samples.items():
        intervals[side] = {}
        for name, draws in values.items():
            enough = len(draws) >= .95 * replicates
            intervals[side][name] = {'ci95': np.quantile(draws, [.025, .975]).tolist() if enough else None,
                                    'valid_replicates': len(draws), 'requested_replicates': replicates}
    return {'candidate': point_a, 'baseline': point_b,
            'delta': {m: None if point_a[m] is None or point_b[m] is None else point_a[m]-point_b[m] for m in point_a},
            'intervals': intervals, 'coverage': coverage, 'speakers': len(groups), 'seed': seed,
            'estimand': 'recording-weighted; paired whole-speaker percentile bootstrap',
            'delta_direction': 'candidate minus baseline; positive correlation / negative error favors candidate'}


def create_run(root, experiment_id, configuration):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,95}', experiment_id):
        raise ValueError('Unsafe experiment ID')
    required = {'protocol_id', 'datasets', 'split_sha256', 'encoders', 'features', 'fusion',
                'targets', 'loss', 'seeds', 'code_commit', 'environment', 'selection_rule'}
    if not required <= configuration.keys():
        raise ValueError('Missing run metadata: ' + str(sorted(required - configuration.keys())))
    if not set(configuration['datasets']) <= ALLOWED_DATASETS or not configuration['datasets']:
        raise ValueError('Unapproved reference-selection dataset')
    serialized = json.dumps(configuration, sort_keys=True, indent=2, allow_nan=False)+'\n'
    directory = Path(root) / experiment_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'configuration.json').write_text(serialized)
    (directory/'configuration.sha256').write_text(hashlib.sha256(serialized.encode()).hexdigest()+'\n')
    return directory


def finish_run(directory, results):
    directory = Path(directory)
    if digest(directory/'configuration.json') != (directory/'configuration.sha256').read_text().strip():
        raise ValueError('Run configuration changed')
    required = {'validation_metrics', 'test_metrics', 'parameter_count', 'training_seconds', 'checkpoint'}
    if not required <= results.keys() or results['test_metrics'] is not None:
        raise ValueError('Development result must include metadata and null final-test metrics')
    serialized = json.dumps(results, sort_keys=True, indent=2, allow_nan=False)+'\n'
    # Exclusive publication of a fully-written file prevents partial final artifacts.
    import os, tempfile
    with tempfile.NamedTemporaryFile(mode='w', dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(serialized);stream.flush();os.fsync(stream.fileno())
    try:
        os.link(temporary, directory/'results.json')
    finally:
        temporary.unlink()
