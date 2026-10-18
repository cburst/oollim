"""Release ESPnet's unused attention visualizations after inference layers."""
import torch


def release_attention_maps(module, inputs, output):
    # These are diagnostic copies retained by upstream forward_attention, not
    # recurrent state. Never change attention arithmetic, outputs or audio context.
    if (not module.training and
        type(module).__module__=='espnet.nets.pytorch_backend.transformer.attention' and
        type(module).__name__ in ('MultiHeadedAttention','RelPositionMultiHeadedAttention','LegacyRelPositionMultiHeadedAttention') and
        isinstance(getattr(module,'attn',None),torch.Tensor)):
        module.attn=None


def install():
    return torch.nn.modules.module.register_module_forward_hook(release_attention_maps)
