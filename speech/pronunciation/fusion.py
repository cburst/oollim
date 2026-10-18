"""Portable frozen-feature scoring and sequential, low-memory audio inference."""
import gc
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .encoders import ROOT, EncoderConfig, build_encoder, sha256, FrameBatch
from .audio import read_audio, batch_audio
from .scoring import masked_pool
from .phone_features import phone_evidence
from .text_reference import load_dictionary, text_to_phones

DIMENSIONS = ('accuracy', 'fluency', 'prosody')

# The earlier file predates neural heads; masked_pool AST is identical.
# Exact old/new whole-file hashes were checked against git commit 1d9bff2.
POOLING_COMPATIBILITY = {
    '4c01f1770f1e1452f07ab7931b88240b263419aa029d34c5d781720030767fcb':
    'c0a5a00205867d963cac9600f7c5e00dbeda791c96783c3ea563b646bdd8a7d9'
}


class FusionNetwork(nn.Module):
    """Dimension-agnostic shared network with separate scoring heads."""
    def __init__(self, width, hidden=128, dropout=.2):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(width, hidden), nn.LayerNorm(hidden),
                                    nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({d: nn.Linear(hidden, 1) for d in DIMENSIONS})

    def forward(self, features):
        hidden = self.shared(features)
        return {'speechocean': {d: head(hidden).squeeze(-1) for d, head in self.heads.items()}}

    def array(self, features):
        outputs = self(features)['speechocean']
        return torch.stack([outputs[d] for d in DIMENSIONS], -1) * 10


class UnifiedScorer:
    """Ridge and neural checkpoints share raw features and input preprocessing."""
    def __init__(self, checkpoint):
        self.path = Path(checkpoint)
        self.metadata = json.loads(self.path.with_suffix('.json').read_text())
        m = self.metadata
        if m['format_version'] != 1 or m['output_dimensions'] != list(DIMENSIONS):
            raise ValueError('Unsupported checkpoint schema')
        if sha256(self.path) != m['checkpoint_sha256']:
            raise ValueError('Checkpoint checksum mismatch')
        self.blocks = m['blocks']
        self.widths = m['block_widths']
        self.width = sum(self.widths[k] for k in self.blocks)
        self.kind = m['checkpoint_type']
        if self.kind == 'frozen_feature_fusion_ridge':
            with np.load(self.path, allow_pickle=False) as f:
                self.arrays = {k: f[k].copy() for k in f.files}
            if self.arrays['coef'].shape != (self.width, 3):
                raise ValueError('Coefficient shape mismatch')
        elif self.kind == 'frozen_feature_fusion_neural':
            if m.get('training', {}).get('model_code_sha256') != sha256(Path(__file__)):
                raise ValueError('Neural implementation does not match checkpoint')
            saved = torch.load(self.path, map_location='cpu', weights_only=True)
            self.arrays = {k: saved[k].numpy() for k in ('scaler_mean','scaler_scale','block_gain')}
            self.network = FusionNetwork(self.width, **m['architecture'])
            self.network.load_state_dict(saved['state_dict'], strict=True)
            self.network.eval()
        else:
            raise ValueError('Unsupported checkpoint type')
        for k in ('scaler_mean','scaler_scale','block_gain'):
            if self.arrays[k].shape != (self.width,) or not np.isfinite(self.arrays[k]).all():
                raise ValueError('Invalid feature transform')
        if (self.arrays['scaler_scale'] <= 0).any():
            raise ValueError('Invalid feature scale')

    def predict_features(self, features):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.width or not np.isfinite(x).all():
            raise ValueError('Invalid feature matrix')
        z = (x-self.arrays['scaler_mean'])/self.arrays['scaler_scale']*self.arrays['block_gain']
        if self.kind == 'frozen_feature_fusion_ridge':
            result = z@self.arrays['coef']+self.arrays['intercept']
        else:
            with torch.inference_mode():
                result = self.network.array(torch.from_numpy(z.astype(np.float32))).numpy()
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite prediction')
        return result

    def extract(self, audio_path, reference_text, resources=None):
        """One encoder resident at a time. No downloads and no transcription fallback."""
        resources = Path(resources or ROOT/'resources')
        prov = self.metadata['experiment_provenance']
        generator = prov.get('reference_generator')
        if not generator:
            raise ValueError('Audio inference requires a text-reference checkpoint')
        dictionary = resources/'text-reference-v1/cmudict.dict'
        if sha256(dictionary) != generator['resource']['sha256']:
            raise ValueError('Dictionary checksum mismatch')
        # Fail before costly encoder loading if reference is missing.
        phones, text_status = text_to_phones(reference_text, load_dictionary(dictionary))
        for file, expected in [('text_reference.py',generator['code_sha256']),
                               ('phone_features.py',prov['feature_code_sha256'])]:
            if sha256(ROOT/'pronunciation'/file) != expected:
                raise ValueError('Reference implementation does not match checkpoint')
        waveform = read_audio(audio_path)
        if len(waveform) < 400:
            raise ValueError('Audio must contain at least 25 ms')
        batch = batch_audio([waveform]); values = {}; status = {}
        for name in self.blocks:
            if name in ('ipa','phoneticxeus'):
                continue
            bp = prov['backbones'][name]
            if sha256(ROOT/'pronunciation/encoders.py') != bp['adapter_source_sha256']:
                raise ValueError('Encoder adapter does not match checkpoint')
            if sha256(ROOT/'pronunciation/scoring.py') not in (bp['pooling_source_sha256'], POOLING_COMPATIBILITY.get(bp['pooling_source_sha256'])):
                raise ValueError('Pooling does not match checkpoint')
            cfg = EncoderConfig(**bp['backbone'])
            cfg.directory = str(resources/Path(cfg.directory).name)
            model = build_encoder(cfg).eval()
            with torch.inference_mode():
                values[name] = masked_pool(model(batch))[0].numpy()
            del model; gc.collect()
        if 'ipa' in self.blocks or 'phoneticxeus' in self.blocks:
            from .phonetic import PhoneticXeus
            bp = prov['backbones']['phoneticxeus']
            for file, key in [('phonetic.py','adapter_sha256'),('audio.py','audio_sha256'),('scoring.py','pooling_sha256')]:
                if sha256(ROOT/'pronunciation'/file) != bp[key]:
                    raise ValueError('Phonetic implementation does not match checkpoint')
            directory = resources/'phoneticxeus'
            if sha256(directory/'config.json') != bp['config_sha256'] or sha256(directory/'pxeus/model/xeusphoneme/resources/ipa_vocab.json') != bp['vocabulary_sha256']:
                raise ValueError('Phonetic configuration does not match checkpoint')
            if sha256(directory/'model.safetensors') != bp['weights_sha256']:
                raise ValueError('Phonetic weights do not match checkpoint')
            for file, expected in bp['implementation'].items():
                if sha256(directory/file) != expected:
                    raise ValueError('Phonetic source does not match checkpoint')
            model = PhoneticXeus(directory).eval()
            with torch.inference_mode():
                out = model(batch)[0]
                frames = out.frames[None]
                fb = FrameBatch(frames, torch.tensor([len(out.frames)]),
                    torch.ones(frames.shape[:2],dtype=torch.bool),16000,out.stride_samples,400)
                values['phoneticxeus'] = masked_pool(fb)[0].numpy()
                values['ipa'], status = phone_evidence(out.logits.numpy(),out.vocabulary,out.blank_id,phones,len(waveform)/16000)
            del model, out; gc.collect()
        for name in self.blocks:
            if values[name].shape != (self.widths[name],):
                raise ValueError('Encoder output width mismatch')
        return np.concatenate([values[k] for k in self.blocks])[None], dict(text=text_status,phone=status)

    def assess(self, audio_path, reference_text, resources=None):
        features, status = self.extract(audio_path, reference_text, resources)
        pred = self.predict_features(features)[0]
        return dict(schema_version='pronunciation-local-v1',model_sha256=self.metadata['checkpoint_sha256'],
            model_type=self.kind, scores={d:float(v) for d,v in zip(DIMENSIONS,pred)},
            score_scale='SpeechOcean 0–10; raw regression outputs, not clipped',
            reference_status=status,backbones=[k for k in self.blocks if k!='ipa'],
            limitations=['research_validation_model','pretrained_dataset_exposure_unresolved'])
