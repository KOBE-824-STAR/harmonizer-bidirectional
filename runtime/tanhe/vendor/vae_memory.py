"""Recompute VAE layers while explicitly carrying their causal cache tensors."""
from types import MethodType
import torch
from torch.utils.checkpoint import checkpoint


def set_vae_checkpointing(inner, enabled):
    for module in inner.modules():
        kind = type(module).__name__
        if kind not in ('ResidualBlock', 'Resample', 'AttentionBlock'):
            continue
        module._harmonizer_checkpoint_enabled = enabled
        if hasattr(module, '_harmonizer_original_forward'):
            continue
        module._harmonizer_original_forward = module.forward
        if kind == 'ResidualBlock':
            module._harmonizer_cache_count = sum(type(m).__name__ == 'CausalConv3d' for m in module.residual)
        elif kind == 'Resample':
            module._harmonizer_cache_count = int(module.mode in ('upsample3d', 'downsample3d'))
        else:
            module._harmonizer_cache_count = 0
        module.forward = MethodType(_checkpoint_layer, module)


def _checkpoint_layer(self, x, feat_cache=None, feat_idx=None):
    original = self._harmonizer_original_forward
    active = self._harmonizer_checkpoint_enabled and torch.is_grad_enabled() and self.training
    if feat_cache is None:
        return checkpoint(original, x, use_reentrant=False) if active else original(x)
    if not active:
        return original(x, feat_cache=feat_cache, feat_idx=feat_idx)
    start = feat_idx[0]
    count = self._harmonizer_cache_count
    # Only this layer's incoming entries are captured. Recompute must not
    # mutate the shared cache/index after later layers have advanced them.
    incoming = tuple(feat_cache[start:start + count])

    def pure_call(value, *cache_entries):
        local_cache = list(cache_entries)
        local_index = [0]
        output = original(value, feat_cache=local_cache, feat_idx=local_index)
        assert local_index[0] == count
        return (output, *local_cache)

    output, *outgoing = checkpoint(pure_call, x, *incoming, use_reentrant=False)
    feat_cache[start:start + count] = outgoing
    feat_idx[0] += count
    return output


def vae_call(function, inner, value):
    if not torch.is_grad_enabled() or not inner.training:
        return function(value)
    set_vae_checkpointing(inner, True)

    def call_with_clean_cache(x):
        # The official encoder/decoder resets its cache at entry and exit.
        # Non-reentrant checkpointing may stop recomputation early; also clear
        # in finally so no saved activation graph survives that early stop.
        try:
            return function(x)
        finally:
            inner.clear_cache()

    return checkpoint(call_with_clean_cache, value, use_reentrant=False)
