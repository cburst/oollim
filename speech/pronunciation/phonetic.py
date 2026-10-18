"""Optional phone recognizer and uncalibrated realization features, not a scorer."""
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import torch
from torch import nn
from safetensors.torch import load_file

from .audio import AudioBatch
from .encoders import ROOT, verify_file


@dataclass
class PhoneOutput:
    frames: torch.Tensor
    logits: torch.Tensor
    tokens: list[str]
    token_ids: list[int]
    blank_id: int
    vocabulary: list[str]
    stride_samples: int
    sample_rate: int = 16000

    def diagnostic_features(self):
        logp = self.logits.log_softmax(-1)
        p = logp.exp()
        return {'mean_entropy': float(-(p * logp).sum(-1).mean()),
                'mean_blank_probability': float(p[:, self.blank_id].mean()),
                'greedy_phone_count': len(self.tokens),
                'status': 'uncalibrated_phonetic_diagnostics'}


class PhoneticXeus(nn.Module):
    """Loads reviewed local code and safetensors; never downloads at inference."""
    def __init__(self, directory=None):
        super().__init__()
        root = Path(directory or ROOT / 'resources/phoneticxeus').resolve()
        verify_file(root / 'model.safetensors',
                    'ad58bf20a60e9d0380327bd8b2d0e8e90a9b8de2adccbfb479f9b21ea85eda18')
        # Verify every downloaded implementation/config against its acquisition
        # receipt; the project resource lock records those hashes for transport.
        for path in root.rglob('*'):
            if path.suffix in ('.py', '.json', '.yaml') and not path.name.endswith('.receipt.json'):
                receipt = json.loads(path.with_name(path.name + '.receipt.json').read_text())
                verify_file(path, receipt['sha256'])
        sys.path.insert(0, str(root))
        from pxeus.model.xeusphoneme.builders import build_xeus_pr_from_hf
        cfg = json.loads((root / 'config.json').read_text())
        self.model = build_xeus_pr_from_hf(work_dir=str(root), hf_repo=None,
            config_file=str(root / cfg['config_yaml']), vocab_file=str(root / cfg['vocab_file']),
            load_ckpt=False, interctc_layer_idx=cfg['interctc_layer_idx'],
            interctc_use_conditioning=cfg['interctc_use_conditioning'])
        weights = load_file(str(root / 'model.safetensors'))
        if any(not k.startswith('model.') for k in weights):
            raise ValueError('Unexpected phonetic checkpoint namespace')
        self.model.load_state_dict({k[6:]: v for k, v in weights.items()}, strict=True)
        self.model.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, batch: AudioBatch):
        batch.validate()
        if batch.sample_rate != 16000 or (batch.lengths < 400).any():
            raise ValueError('PhoneticXeus requires >=400 samples of 16kHz audio')
        results = []
        self.model.eval()
        # Unpadded single-utterance calls also avoid the upstream decoder's
        # omission of output-length masking when given mixed-length batches.
        for waveform, length in zip(batch.waveforms, batch.lengths):
            frames, lengths = self.model.encode(waveform[None, :int(length)], length[None])
            if isinstance(frames, tuple):
                frames = frames[0]
            frames = frames[0, :int(lengths[0])]
            logits = self.model.ctc.ctc_lo(frames)
            ids = logits.argmax(-1).unique_consecutive().tolist()
            ids = [i for i in ids if i != self.model.blank_id]
            tokens = [self.model.token_list[i] for i in ids]
            results.append(PhoneOutput(frames, logits, tokens, ids, self.model.blank_id,
                                       self.model.token_list, self.model.points_by_frames()))
        return results
