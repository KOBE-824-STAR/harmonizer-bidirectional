"""Helpers for the two tokenizer interface shapes used by Cosmos runtimes."""

from __future__ import annotations

import torch


def inner_tokenizer_module(tokenizer) -> torch.nn.Module:
    """Resolve the parameter-owning VAE module behind a tokenizer facade.

    Current Cosmos ``TokenizerInterface`` objects are plain Python facades with
    ``model.model`` pointing to the actual ``nn.Module``.  Other supported
    tokenizers expose the encoder and decoder directly.  Keeping this traversal
    explicit ensures freezing and checkpoint loading act on the same module.
    """

    current = tokenizer
    visited: set[int] = set()
    for _ in range(5):
        if id(current) in visited:
            break
        visited.add(id(current))
        if (
            isinstance(current, torch.nn.Module)
            and hasattr(current, "encoder")
            and hasattr(current, "decoder")
        ):
            return current
        for name in ("model", "tokenizer", "module"):
            if hasattr(current, name):
                current = getattr(current, name)
                break
        else:
            break
    raise TypeError(
        "Could not resolve a parameter-owning VAE nn.Module with encoder/decoder "
        f"from tokenizer type {type(tokenizer)!r}"
    )
