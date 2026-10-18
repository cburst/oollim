"""Preserve timing; convert audio once before encoder-specific normalization."""
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch.nn.utils.rnn import pad_sequence

PREPROCESSING_VERSION = 'mono-resample-poly-v1'


@dataclass
class AudioBatch:
    waveforms: torch.Tensor
    lengths: torch.Tensor
    sample_rate: int

    def validate(self):
        x, n = self.waveforms, self.lengths
        if x.ndim != 2 or n.shape != (x.shape[0],) or not x.shape[0]:
            raise ValueError('Expected nonempty waveforms [B,S] and lengths [B]')
        if n.dtype != torch.long or n.device != x.device:
            raise ValueError('Lengths must be int64 on the waveform device')
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('Audio must be finite floating point')
        if (n <= 0).any() or (n > x.shape[1]).any() or self.sample_rate <= 0:
            raise ValueError('Invalid audio lengths/sample rate')


def read_audio(path: str | Path, target_rate: int = 16000) -> torch.Tensor:
    samples, rate = sf.read(path, dtype='float32', always_2d=True)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Empty or nonfinite audio')
    mono = samples.mean(axis=1)
    if rate != target_rate:
        divisor = gcd(rate, target_rate)
        mono = resample_poly(mono, target_rate // divisor, rate // divisor)
    return torch.from_numpy(np.asarray(mono, dtype=np.float32).copy())


def batch_audio(waveforms, sample_rate=16000):
    if not waveforms or any(x.ndim != 1 for x in waveforms):
        raise ValueError('Supply nonempty list of mono waveforms')
    batch = AudioBatch(pad_sequence(waveforms, batch_first=True),
                       torch.tensor([len(x) for x in waveforms], dtype=torch.long,
                                    device=waveforms[0].device), sample_rate)
    batch.validate()
    return batch
