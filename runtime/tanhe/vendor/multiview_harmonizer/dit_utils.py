"""Shared helpers for in-place upgrades of the official 0.6B Text2Image DiT."""

from __future__ import annotations

import torch


def validate_view_metadata(
    x_B_C_V_H_W: torch.Tensor,
    view_ids_B_V: torch.Tensor | None,
    view_valid_B_V: torch.Tensor | None,
    max_views: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x_B_C_V_H_W.ndim != 5:
        raise ValueError(f"DiT input must be [B,C,V,H,W], got {tuple(x_B_C_V_H_W.shape)}")
    batch, _, views, _, _ = x_B_C_V_H_W.shape
    if not 1 <= views <= max_views:
        raise ValueError(f"Expected 1..{max_views} views, got {views}")
    if view_ids_B_V is None:
        view_ids_B_V = torch.arange(
            views, device=x_B_C_V_H_W.device, dtype=torch.long
        ).unsqueeze(0).expand(batch, -1)
    if view_valid_B_V is None:
        view_valid_B_V = torch.ones(
            (batch, views), device=x_B_C_V_H_W.device, dtype=torch.bool
        )
    if view_ids_B_V.shape != (batch, views) or view_ids_B_V.dtype != torch.long:
        raise ValueError("view_ids_B_V must be long with shape [B,V]")
    if view_valid_B_V.shape != (batch, views) or view_valid_B_V.dtype != torch.bool:
        raise ValueError("view_valid_B_V must be bool with shape [B,V]")
    if view_ids_B_V.device != x_B_C_V_H_W.device:
        raise ValueError("view_ids_B_V must be on the same device as the DiT input")
    if view_valid_B_V.device != x_B_C_V_H_W.device:
        raise ValueError("view_valid_B_V must be on the same device as the DiT input")
    if view_ids_B_V.min() < 0 or view_ids_B_V.max() >= max_views:
        raise ValueError(f"view IDs must be in [0,{max_views - 1}]")
    for sample_index in range(batch):
        valid_ids = view_ids_B_V[sample_index, view_valid_B_V[sample_index]]
        if valid_ids.unique().numel() != valid_ids.numel():
            raise ValueError(f"view IDs must be unique within sample {sample_index}")
    return view_ids_B_V, view_valid_B_V


def repeat_single_frame_rope(rope_single: torch.Tensor, views: int) -> torch.Tensor:
    """Repeat t=0 spatial RoPE in the same order as flatten(V,H,W)."""

    if views < 1:
        raise ValueError("views must be positive")
    return rope_single.repeat((views,) + (1,) * (rope_single.ndim - 1))


def module_device_dtype(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters())
    return parameter.device, parameter.dtype
