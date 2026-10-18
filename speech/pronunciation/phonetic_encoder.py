"""Frozen PhoneticXeus embeddings through the shared encoder interface."""
from pathlib import Path
import torch
from .encoders import ROOT,EncoderConfig,SpeechEncoder,register_encoder,verify_file
from .phonetic import PhoneticXeus


def phonetic_config():
    return EncoderConfig(name='phoneticxeus',model_id='changelinglab/PhoneticXeus',
        revision='ed64f14aeb0e903a80a825cf67a079aa8f93d3ac',directory=str(ROOT/'resources/phoneticxeus'),
        weights_sha256='ad58bf20a60e9d0380327bd8b2d0e8e90a9b8de2adccbfb479f9b21ea85eda18',
        config_sha256='e3a1159d19f0908ecf21053c854e5f1994e90e6339a01025ea081d965d7d5a1b',
        implementation_revision='ed64f14aeb0e903a80a825cf67a079aa8f93d3ac-reviewed-local',
        adapter_version='phoneticxeus-shared-final-frame-v1')


@register_encoder('phoneticxeus')
class PhoneticEncoder(SpeechEncoder):
    def __init__(self,config):
        super().__init__(config)
        if not config.frozen or config.layer!=-1 or config.sample_rate!=16000:
            raise ValueError('PhoneticXeus adapter currently supports frozen final-layer 16kHz only')
        verify_file(Path(config.directory)/'config.json',config.config_sha256)
        verify_file(Path(config.directory)/'model.safetensors',config.weights_sha256)
        self.recognizer=PhoneticXeus(config.directory)
        self.output_dim=self.recognizer.model.encoder.output_size()
        self.stride_samples=self.recognizer.model.points_by_frames()
        self.finish_init()

    def encode_one(self,waveform):
        lengths=torch.tensor([len(waveform)],device=waveform.device)
        frames,lengths=self.recognizer.model.encode(waveform[None],lengths)
        if isinstance(frames,tuple):frames=frames[0]
        return frames[0,:int(lengths[0])]

    @torch.no_grad()
    def phone_logits(self,frames):
        return self.recognizer.model.ctc.ctc_lo(frames)

    @property
    def vocabulary(self):return list(self.recognizer.model.token_list)

    @property
    def blank_id(self):return self.recognizer.model.blank_id
