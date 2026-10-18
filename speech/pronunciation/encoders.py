"""Adapters own native dimensions/preprocessing; all consumers use FrameBatch."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import nullcontext
import hashlib
import json

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import yaml

from .audio import AudioBatch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@dataclass
class EncoderConfig:
    name: str = 'xeus'
    model_id: str = 'espnet/xeus'
    revision: str = 'b651261280bf792df4632ccb2d1ba4eeaabeaee2'
    directory: str = str(ROOT / 'resources/xeus')
    weights_sha256: str = '4fd02581b4dd06736d1a6aa29e20c442afe3d9b299c88359f63320292a5c5cff'
    config_sha256: str = '382ab33c232e14ccf3eabdd0129db239ae0420fc2d8546098a4260820255062e'
    implementation_revision: str = '5b52d57b4f872ff7babded35316a79642e9e6c12'
    frozen: bool = True
    layer: int = -1
    sample_rate: int = 16000
    adapter_version: str = 'unpadded-per-utterance-keymap-v1'


def default_config(name='xeus'):
    if name == 'xeus':
        return EncoderConfig()
    if name == 'xlsr':
        return EncoderConfig(name='xlsr', model_id='facebook/wav2vec2-xls-r-300m',
            revision='1a640f32ac3e39899438a2931f9924c02f080a54',
            directory=str(ROOT / 'resources/xlsr-300m'),
            weights_sha256='d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0',
            config_sha256='0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac',
            implementation_revision='transformers-5.19.0')
    raise ValueError(f'No default configuration for {name}')


@dataclass
class FrameBatch:
    frames: torch.Tensor
    lengths: torch.Tensor
    valid_mask: torch.Tensor
    sample_rate: int
    stride_samples: int
    receptive_field_samples: int

    def validate(self):
        b, t, _ = self.frames.shape
        if self.lengths.shape != (b,) or self.valid_mask.shape != (b, t):
            raise ValueError('Invalid frame shapes')
        if self.valid_mask.dtype != torch.bool or (self.lengths <= 0).any():
            raise ValueError('Invalid frame mask/empty sequence')
        expected = torch.arange(t, device=self.frames.device)[None] < self.lengths[:, None]
        if not torch.equal(expected, self.valid_mask) or (self.lengths > t).any():
            raise ValueError('Mask and lengths disagree')
        if not torch.isfinite(self.frames).all():
            raise ValueError('Nonfinite embeddings')


class SpeechEncoder(nn.Module, ABC):
    output_dim: int

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stride_samples = 320
        self.receptive_field_samples = 400

    def finish_init(self):
        self.requires_grad_(not self.config.frozen)
        self.train(False)

    def train(self, mode=True):
        return super().train(False if self.config.frozen else mode)

    @abstractmethod
    def encode_one(self, waveform):
        """Return unpadded [T,D] features; callers control gradient mode."""

    def forward(self, batch: AudioBatch):
        batch.validate()
        if batch.sample_rate != self.config.sample_rate:
            raise ValueError(f'Encoder requires {self.config.sample_rate} Hz audio')
        if (batch.lengths < self.receptive_field_samples).any():
            raise ValueError('Audio shorter than encoder receptive field')
        context = torch.no_grad() if self.config.frozen else nullcontext()
        with context:
            rows = [self.encode_one(x[:int(n)]) for x, n in zip(batch.waveforms, batch.lengths)]
        frames = pad_sequence(rows, batch_first=True)
        lengths = torch.tensor([len(x) for x in rows], device=frames.device)
        mask = torch.arange(frames.shape[1], device=frames.device)[None] < lengths[:, None]
        result = FrameBatch(frames, lengths, mask, batch.sample_rate,
                            self.stride_samples, self.receptive_field_samples)
        result.validate()
        return result

    def metadata(self):
        return asdict(self.config)


REGISTRY = {}


def register_encoder(name):
    def register(cls):
        if name in REGISTRY:
            raise ValueError(f'Encoder already registered: {name}')
        REGISTRY[name] = cls
        return cls
    return register


def build_encoder(config):
    if config.name not in REGISTRY:
        raise ValueError(f'Unknown backbone: {config.name}')
    if config.sample_rate != 16000 and config.name in ('xeus', 'xlsr'):
        raise ValueError('Pinned backbones require 16 kHz')
    return REGISTRY[config.name](config)


def verify_file(path, expected):
    if not expected or sha256(path) != expected:
        raise ValueError(f'Asset checksum mismatch: {path}')


@register_encoder('xeus')
class XeusEncoder(SpeechEncoder):
    def __init__(self, config):
        super().__init__(config)
        from espnet2.asr.frontend.cnn import CNNFrontend
        from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
        from espnet2.asr.preencoder.linear import LinearProjection
        root = Path(config.directory) / 'model'
        verify_file(root / 'config.yaml', config.config_sha256)
        verify_file(root / 'xeus_checkpoint_new.pth', config.weights_sha256)
        cfg = yaml.safe_load((root / 'config.yaml').read_text())
        front = dict(cfg['frontend_conf'])
        # Official pinned fork predates this no-op option in the published YAML.
        # False means no output normalization and matches this fork's frontend.
        if front.pop('normalize_output', False):
            raise ValueError('Pinned XEUS adapter does not support output normalization')
        self.frontend = CNNFrontend(**front)
        self.preencoder = LinearProjection(input_size=self.frontend.output_size(),
                                          **cfg['preencoder_conf'])
        self.encoder = EBranchformerEncoder(input_size=self.preencoder.output_size(),
                                            **cfg['encoder_conf'])
        self.output_dim = self.encoder.output_size()
        state = torch.load(root / 'xeus_checkpoint_new.pth', map_location='cpu',
                           weights_only=True, mmap=True)
        allowed = ('frontend.', 'preencoder.', 'encoder.')
        extras = [k for k in state if not k.startswith(allowed)]
        if any(not k.startswith(('losses.', 'util_modules.', 'global_step')) for k in extras):
            raise ValueError(f'Unexpected checkpoint keys: {extras}')
        # Published new checkpoint renamed one positional conv to convs.0.
        # The pinned fork implements the identical one-conv, legacy weight-norm
        # path. Map only these three names; all tensor keys/shapes remain strict.
        keymap = {f'encoder.embed.0.convs.0.{name}': f'encoder.embed.0.conv.{name}'
                  for name in ('bias', 'weight_g', 'weight_v')}
        self.load_state_dict({keymap.get(k, k): v for k, v in state.items()
                              if k.startswith(allowed)}, strict=True)
        self.finish_init()

    def encode_one(self, waveform):
        lengths = torch.tensor([waveform.numel()], device=waveform.device)
        x, lengths = self.frontend(waveform[None], lengths)
        x, lengths = self.preencoder(x, lengths)
        out, lengths, _ = self.encoder(x, lengths, return_all_hs=True)
        # Match model card use_final_output=False: layer output before final norm.
        layers = out[1]
        selected = layers[self.config.layer]
        return selected[0, :int(lengths[0])]


@register_encoder('xlsr')
class XlsrEncoder(SpeechEncoder):
    def __init__(self, config):
        super().__init__(config)
        from transformers import Wav2Vec2Model
        root = Path(config.directory)
        verify_file(root / 'config.json', config.config_sha256)
        verify_file(root / 'pytorch_model.bin', config.weights_sha256)
        self.model = Wav2Vec2Model.from_pretrained(str(root), local_files_only=True)
        self.output_dim = self.model.config.hidden_size
        self.finish_init()

    def encode_one(self, waveform):
        # Published feature extractor normalizes each valid utterance, eps 1e-7.
        x = (waveform - waveform.mean()) / torch.sqrt(waveform.var(unbiased=False) + 1e-7)
        mask = torch.ones_like(x[None], dtype=torch.long)
        output = self.model(x[None], attention_mask=mask, output_hidden_states=True)
        return output.hidden_states[self.config.layer][0]
