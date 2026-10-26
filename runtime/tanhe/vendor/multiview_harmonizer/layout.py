"""Canonical tensor layout and independent-view tokenizer helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .constants import CAMERA_ORDER


@dataclass
class MultiViewBatch:
    """Validated model-side batch.

    RGB tensors deliberately use B,V,C,H,W.  The official Harmonizer wrapper
    uses B,C,T,H,W internally; conversion is explicit at the model boundary.
    """

    source_rgb: torch.Tensor
    camera_ids: torch.Tensor
    view_valid: torch.Tensor

    def validate(self) -> tuple[int, int, int, int]:
        if self.source_rgb.ndim != 5:
            raise ValueError(
                f"source_rgb must be [B,V,C,H,W], got {tuple(self.source_rgb.shape)}"
            )
        batch, views, channels, height, width = self.source_rgb.shape
        if batch < 1 or height < 1 or width < 1:
            raise ValueError(
                f"source_rgb dimensions must be non-empty, got {tuple(self.source_rgb.shape)}"
            )
        if channels != 3:
            raise ValueError(f"source_rgb must have 3 channels, got {channels}")
        if not self.source_rgb.is_floating_point():
            raise TypeError(f"source_rgb must be floating point, got {self.source_rgb.dtype}")
        if not 1 <= views <= len(CAMERA_ORDER):
            raise ValueError(f"view count must be in [1,{len(CAMERA_ORDER)}], got {views}")
        if self.camera_ids.shape != (batch, views):
            raise ValueError(
                f"camera_ids must be {(batch, views)}, got {tuple(self.camera_ids.shape)}"
            )
        if self.view_valid.shape != (batch, views):
            raise ValueError(
                f"view_valid must be {(batch, views)}, got {tuple(self.view_valid.shape)}"
            )
        if self.camera_ids.dtype != torch.long:
            raise TypeError(f"camera_ids must be torch.long, got {self.camera_ids.dtype}")
        if self.view_valid.dtype != torch.bool:
            raise TypeError(f"view_valid must be torch.bool, got {self.view_valid.dtype}")
        if self.camera_ids.device != self.source_rgb.device:
            raise ValueError(
                "camera_ids and source_rgb must be on the same device: "
                f"{self.camera_ids.device} != {self.source_rgb.device}"
            )
        if self.view_valid.device != self.source_rgb.device:
            raise ValueError(
                "view_valid and source_rgb must be on the same device: "
                f"{self.view_valid.device} != {self.source_rgb.device}"
            )
        valid_ids = self.camera_ids[self.view_valid]
        if valid_ids.numel() and (valid_ids.min() < 0 or valid_ids.max() >= len(CAMERA_ORDER)):
            raise ValueError(f"valid camera_ids must be in [0,{len(CAMERA_ORDER) - 1}]")
        for sample_index in range(batch):
            sample_ids = self.camera_ids[sample_index, self.view_valid[sample_index]]
            if sample_ids.unique().numel() != sample_ids.numel():
                raise ValueError(f"camera_ids must be unique within sample {sample_index}")
        return batch, views, height, width


def default_camera_ids(batch: int, views: int, device: torch.device) -> torch.Tensor:
    if views > len(CAMERA_ORDER):
        raise ValueError(f"Only {len(CAMERA_ORDER)} canonical cameras are supported")
    return torch.arange(views, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)


def encode_views_independently(tokenizer, rgb_B_V_C_H_W: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """Encode every camera as a one-frame sample; cameras never enter VAE time."""

    if rgb_B_V_C_H_W.ndim != 5:
        raise ValueError("RGB input must be [B,V,C,H,W]")
    batch, views, channels, _, _ = rgb_B_V_C_H_W.shape
    if channels != 3:
        raise ValueError(f"Expected RGB input, got C={channels}")
    flat_BV_C_1_H_W = rgb_B_V_C_H_W.reshape(batch * views, channels, *rgb_B_V_C_H_W.shape[-2:]).unsqueeze(2)
    latent_BV_C_T_H_W = tokenizer.encode(flat_BV_C_1_H_W) * sigma_data
    if latent_BV_C_T_H_W.ndim != 5 or latent_BV_C_T_H_W.shape[2] != 1:
        raise RuntimeError(
            "One-frame independent VAE encode must return [B*V,C,1,Hl,Wl], got "
            f"{tuple(latent_BV_C_T_H_W.shape)}"
        )
    return latent_BV_C_T_H_W.reshape(
        batch,
        views,
        latent_BV_C_T_H_W.shape[1],
        1,
        *latent_BV_C_T_H_W.shape[-2:],
    )


def decode_views_independently(tokenizer, latent_B_V_C_1_H_W: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """Decode views independently while retaining gradients to DiT latents."""

    if latent_B_V_C_1_H_W.ndim != 6 or latent_B_V_C_1_H_W.shape[3] != 1:
        raise ValueError(
            "Latent input must be [B,V,C,1,Hl,Wl], got "
            f"{tuple(latent_B_V_C_1_H_W.shape)}"
        )
    batch, views = latent_B_V_C_1_H_W.shape[:2]
    flat_BV_C_1_H_W = latent_B_V_C_1_H_W.reshape(
        batch * views,
        latent_B_V_C_1_H_W.shape[2],
        1,
        *latent_B_V_C_1_H_W.shape[-2:],
    )
    rgb_BV_C_1_H_W = tokenizer.decode(flat_BV_C_1_H_W / sigma_data)
    if rgb_BV_C_1_H_W.ndim != 5 or rgb_BV_C_1_H_W.shape[2] != 1:
        raise RuntimeError(
            "One-frame independent VAE decode must return [B*V,C,1,H,W], got "
            f"{tuple(rgb_BV_C_1_H_W.shape)}"
        )
    return rgb_BV_C_1_H_W.squeeze(2).reshape(
        batch,
        views,
        rgb_BV_C_1_H_W.shape[1],
        *rgb_BV_C_1_H_W.shape[-2:],
    )


def views_to_dit_time(latent_B_V_C_1_H_W: torch.Tensor) -> torch.Tensor:
    """Convert independent latents to DiT's B,C,T,H,W boundary."""

    if latent_B_V_C_1_H_W.ndim != 6 or latent_B_V_C_1_H_W.shape[3] != 1:
        raise ValueError("Expected [B,V,C,1,H,W] independent-view latents")
    return latent_B_V_C_1_H_W.squeeze(3).permute(0, 2, 1, 3, 4).contiguous()


def dit_time_to_views(latent_B_C_V_H_W: torch.Tensor) -> torch.Tensor:
    """Convert DiT output back to explicit independent-view latent layout."""

    if latent_B_C_V_H_W.ndim != 5:
        raise ValueError("Expected [B,C,V,H,W] DiT output")
    return latent_B_C_V_H_W.permute(0, 2, 1, 3, 4).unsqueeze(3).contiguous()
