"""Optional registry adapters; existing pinned encoder source remains unchanged."""
import json
from pathlib import Path
import torch
from .encoders import (ROOT, EncoderConfig, SpeechEncoder, register_encoder,
                       default_config, build_encoder, verify_file)


def hubert_config():
    return EncoderConfig(name='hubert', model_id='facebook/hubert-large-ll60k',
        revision='ff022d095678a2995f3c49bab18a96a9e553f782',directory=str(ROOT/'resources/hubert-large'),
        weights_sha256='45a299050945479a68cffe2ab7a63fd08931718ba18f05a42dbb86a5164178e0',
        config_sha256='52e48f7f44bf7f1be327cc3e82368681acf0cc8c5538d757eae9e6685bfe2b16',
        implementation_revision='transformers-5.19.0',adapter_version='hubert-unpadded-normalized-v1')


@register_encoder('hubert')
class HubertEncoder(SpeechEncoder):
    def __init__(self, config):
        super().__init__(config)
        from transformers import HubertModel
        root=Path(config.directory)
        verify_file(root/'config.json',config.config_sha256)
        verify_file(root/'pytorch_model.bin',config.weights_sha256)
        verify_file(root/'preprocessor_config.json','a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335')
        preprocessing=json.loads((root/'preprocessor_config.json').read_text())
        if config.sample_rate!=16000 or not preprocessing['do_normalize']:
            raise ValueError('Pinned HuBERT requires normalized 16kHz audio')
        self.model=HubertModel.from_pretrained(str(root),local_files_only=True)
        self.output_dim=self.model.config.hidden_size
        self.finish_init()

    def encode_one(self, waveform):
        x=(waveform-waveform.mean())/torch.sqrt(waveform.var(unbiased=False)+1e-7)
        result=self.model(x[None],attention_mask=torch.ones_like(x[None],dtype=torch.long),output_hidden_states=True)
        return result.hidden_states[self.config.layer][0]


def make_encoder(name, *, layer=-1):
    config=hubert_config() if name=='hubert' else default_config(name)
    config.layer=layer
    return build_encoder(config)
