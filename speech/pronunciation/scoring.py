"""One projection, pooling, source-specific head and loss path for every encoder."""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from .encoders import FrameBatch


@dataclass
class ScoringConfig:
    model_dim: int = 256
    sources: tuple = ('speechocean', 'azure', 'human')
    dimensions: tuple = ('accuracy', 'fluency', 'prosody')
    pooling: str = 'masked_mean_std'


def masked_pool(batch: FrameBatch):
    batch.validate()
    x = batch.frames.masked_fill(~batch.valid_mask[..., None], 0)
    n = batch.lengths[:, None].to(x.dtype)
    mean = x.sum(1) / n
    centered = (x - mean[:, None]).masked_fill(~batch.valid_mask[..., None], 0)
    std = torch.sqrt(centered.square().sum(1) / n + 1e-5)
    return torch.cat([mean, std], dim=-1)


class PronunciationScorer(nn.Module):
    def __init__(self, encoder, config=None):
        super().__init__()
        self.encoder = encoder
        self.config = config or ScoringConfig()
        d = self.config.model_dim
        if self.config.pooling == 'masked_mean_std':
            self.projection = nn.Linear(encoder.output_dim, d)
            width = d * 2
        elif self.config.pooling == 'native_mean_std_then_projection':
            self.register_buffer('feature_mean', torch.zeros(encoder.output_dim * 2))
            self.register_buffer('feature_scale', torch.ones(encoder.output_dim * 2))
            self.projection = nn.Linear(encoder.output_dim * 2, d)
            width = d
        else:
            raise ValueError('Unsupported pooling')
        self.shared = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d), nn.GELU())
        self.heads = nn.ModuleDict({source: nn.ModuleDict({dimension: nn.Linear(d, 1)
            for dimension in self.config.dimensions}) for source in self.config.sources})

    def project(self, encoded):
        return FrameBatch(self.projection(encoded.frames), encoded.lengths,
            encoded.valid_mask, encoded.sample_rate, encoded.stride_samples,
            encoded.receptive_field_samples)

    def score_pooled(self, features):
        if self.config.pooling != 'native_mean_std_then_projection':
            raise ValueError('This architecture needs frame sequences')
        hidden = self.shared(self.projection((features-self.feature_mean)/self.feature_scale))
        return self.predict_hidden(hidden)

    def predict_hidden(self, hidden):
        return {s: {d: head(hidden).squeeze(-1) for d, head in heads.items()}
                for s, heads in self.heads.items()}

    def score_frames(self, encoded):
        if self.config.pooling == 'native_mean_std_then_projection':
            return self.score_pooled(masked_pool(encoded))
        return self.predict_hidden(self.shared(masked_pool(self.project(encoded))))

    def forward(self, batch):
        return self.score_frames(self.encoder(batch))


def masked_loss(predictions, targets, weights=None):
    """Targets are source/dimension tensors with NaN for missing labels.

    Caller supplies explicitly normalized targets on each source's own scale.
    Unlabeled heads receive no gradient, preserving supervision provenance.
    """
    terms, mass = [], []
    for source, dims in targets.items():
        for dimension, y in dims.items():
            pred = predictions[source][dimension]
            if y.shape != pred.shape:
                raise ValueError('Target shape mismatch')
            valid = torch.isfinite(y)
            weight = (weights or {}).get(f'{source}.{dimension}', 1.0)
            if weight < 0:
                raise ValueError('Negative loss weight')
            if valid.any() and weight:
                terms.append(F.smooth_l1_loss(pred[valid], y[valid]) * weight)
                mass.append(weight)
    if not terms:
        raise ValueError('No labeled targets in batch')
    return torch.stack(terms).sum() / sum(mass)
