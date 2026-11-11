"""Release ESPnet's unused attention visualizations after inference layers."""
import torch

# XEUS imports ESPnet's attention. PhoneticXeus ships a vendored copy of the
# same classes under pxeus, whose forward_attention also stores self.attn.
ATTENTION_MODULES = ('espnet.nets.pytorch_backend.transformer.attention',
                     'pxeus.espnet_import.attention')
ATTENTION_CLASSES = ('MultiHeadedAttention','RelPositionMultiHeadedAttention','LegacyRelPositionMultiHeadedAttention')


def release_attention_maps(module, inputs, output):
    # These are diagnostic copies retained by upstream forward_attention, not
    # recurrent state. Never change attention arithmetic, outputs or audio context.
    if (not module.training and
        type(module).__module__ in ATTENTION_MODULES and
        type(module).__name__ in ATTENTION_CLASSES and
        isinstance(getattr(module,'attn',None),torch.Tensor)):
        module.attn=None


def install():
    return torch.nn.modules.module.register_module_forward_hook(release_attention_maps)
