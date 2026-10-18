"""Portable metadata plus shared state; frozen encoder assets remain hash-pinned."""
from dataclasses import asdict
from pathlib import Path
import torch
from .audio import PREPROCESSING_VERSION
from .encoders import EncoderConfig, build_encoder
from .scoring import ScoringConfig, PronunciationScorer


def validate_experiment(experiment, config):
    required = {'seed', 'split_sha256', 'stage', 'target_scales'}
    if not required <= experiment.keys():
        raise ValueError(f'Missing experiment metadata: {required - experiment.keys()}')
    expected = {f'{s}.{d}' for s in config.sources for d in config.dimensions}
    if set(experiment['target_scales']) != expected:
        raise ValueError('Declare a scale or explicit untrained status for every head')
    for scale in experiment['target_scales'].values():
        if scale.get('status') == 'untrained':
            continue
        if not scale.get('scale_id') or scale.get('native_min', 0) >= scale.get('native_max', 0):
            raise ValueError('Invalid target scale metadata')


def save_checkpoint(path, model, optimizer, *, step, experiment):
    validate_experiment(experiment, model.config)
    state = model.state_dict()
    if model.encoder.config.frozen:
        state = {k: v for k, v in state.items() if not k.startswith('encoder.')}
    payload = dict(format_version=1, backbone=model.encoder.metadata(),
        scoring=asdict(model.config), preprocessing=PREPROCESSING_VERSION,
        state=state, optimizer=optimizer.state_dict() if optimizer else None,
        step=step, experiment=experiment)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path, *, asset_directory=None):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['format_version'] != 1 or payload['preprocessing'] != PREPROCESSING_VERSION:
        raise ValueError('Unsupported checkpoint/preprocessing version')
    config = EncoderConfig(**payload['backbone'])
    if asset_directory is not None:
        config.directory = str(asset_directory)
    scoring = ScoringConfig(**payload['scoring'])
    validate_experiment(payload['experiment'], scoring)
    model = PronunciationScorer(build_encoder(config), scoring)
    info = model.load_state_dict(payload['state'], strict=False)
    allowed_missing = {k for k in model.state_dict() if k.startswith('encoder.')} if config.frozen else set()
    if set(info.missing_keys) != allowed_missing or info.unexpected_keys:
        raise ValueError(f'Incompatible scoring state: {info}')
    return model, payload
