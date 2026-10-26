"""Explicit time/view layouts for temporal-preserving multi-view Harmonizer.

The public RGB layout is ``[B,T,V,C,H,W]``.  Only the camera axis is folded
into the batch axis before the official causal video tokenizer is called, so
each camera retains its complete temporal sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .constants import CAMERA_ORDER


OFFICIAL_TEMPORAL_OFFSETS: tuple[int, ...] = (0, -1, -2, -3, -4)
CURRENT_OUTPUT_INDEX = 0


def canonical_temporal_offsets(
    batch_size: int,
    num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the official ``[current,-1,-2,-3,-4]`` prefix as ``[B,T]``."""

    if not 1 <= num_frames <= len(OFFICIAL_TEMPORAL_OFFSETS):
        raise ValueError(
            f"num_frames must be in [1,{len(OFFICIAL_TEMPORAL_OFFSETS)}], got {num_frames}"
        )
    offsets = torch.tensor(
        OFFICIAL_TEMPORAL_OFFSETS[:num_frames], device=device, dtype=torch.long
    )
    return offsets.unsqueeze(0).expand(batch_size, -1)


@dataclass(frozen=True)
class TemporalMultiViewBatch:
    """Validated Scheme A-TMV batch.

    ``source_rgb`` is ``[B,T,V,3,H,W]``.  Camera metadata is constant across
    temporal slots, while ``temporal_offsets`` is either ``[T]`` or ``[B,T]``
    and must match the official current-first ordering.
    """

    source_rgb: torch.Tensor
    camera_ids: torch.Tensor
    view_valid: torch.Tensor
    temporal_offsets: torch.Tensor | None = None

    def validate(self) -> tuple[int, int, int, int, int]:
        if self.source_rgb.ndim != 6:
            raise ValueError(
                "source_rgb must be [B,T,V,C,H,W], got "
                f"{tuple(self.source_rgb.shape)}"
            )
        batch_size, num_frames, num_views, channels, height, width = self.source_rgb.shape
        if batch_size < 1 or height < 1 or width < 1:
            raise ValueError(f"source_rgb dimensions must be non-empty: {tuple(self.source_rgb.shape)}")
        if channels != 3:
            raise ValueError(f"source_rgb must have C=3, got {channels}")
        if not self.source_rgb.is_floating_point():
            raise TypeError(f"source_rgb must be floating point, got {self.source_rgb.dtype}")
        if not 1 <= num_frames <= len(OFFICIAL_TEMPORAL_OFFSETS):
            raise ValueError(
                f"num_frames must be in [1,{len(OFFICIAL_TEMPORAL_OFFSETS)}], got {num_frames}"
            )
        if not 1 <= num_views <= len(CAMERA_ORDER):
            raise ValueError(f"num_views must be in [1,{len(CAMERA_ORDER)}], got {num_views}")
        if self.camera_ids.shape != (batch_size, num_views):
            raise ValueError(
                f"camera_ids must be {(batch_size, num_views)}, got {tuple(self.camera_ids.shape)}"
            )
        if self.view_valid.shape != (batch_size, num_views):
            raise ValueError(
                f"view_valid must be {(batch_size, num_views)}, got {tuple(self.view_valid.shape)}"
            )
        if self.camera_ids.dtype != torch.long:
            raise TypeError(f"camera_ids must be torch.long, got {self.camera_ids.dtype}")
        if self.view_valid.dtype != torch.bool:
            raise TypeError(f"view_valid must be torch.bool, got {self.view_valid.dtype}")
        if self.camera_ids.device != self.source_rgb.device:
            raise ValueError("camera_ids and source_rgb must be on the same device")
        if self.view_valid.device != self.source_rgb.device:
            raise ValueError("view_valid and source_rgb must be on the same device")
        if self.camera_ids.numel() and (
            self.camera_ids.min() < 0 or self.camera_ids.max() >= len(CAMERA_ORDER)
        ):
            raise ValueError(f"camera_ids must be in [0,{len(CAMERA_ORDER) - 1}]")
        for sample_index in range(batch_size):
            valid_ids = self.camera_ids[sample_index, self.view_valid[sample_index]]
            if valid_ids.unique().numel() != valid_ids.numel():
                raise ValueError(f"valid camera IDs must be unique in sample {sample_index}")

        expected_offsets = canonical_temporal_offsets(
            batch_size, num_frames, self.source_rgb.device
        )
        if self.temporal_offsets is not None:
            offsets = self.temporal_offsets
            if offsets.dtype != torch.long:
                raise TypeError("temporal_offsets must be torch.long")
            if offsets.device != self.source_rgb.device:
                raise ValueError("temporal_offsets and source_rgb must be on the same device")
            if offsets.shape == (num_frames,):
                offsets = offsets.unsqueeze(0).expand(batch_size, -1)
            if offsets.shape != (batch_size, num_frames):
                raise ValueError(
                    "temporal_offsets must be [T] or [B,T], got "
                    f"{tuple(self.temporal_offsets.shape)}"
                )
            if not torch.equal(offsets, expected_offsets):
                raise ValueError(
                    "temporal slots must use the official current-first order "
                    f"{OFFICIAL_TEMPORAL_OFFSETS[:num_frames]}"
                )
        return batch_size, num_frames, num_views, height, width


def rgb_btvc_to_bvcthw(rgb_B_T_V_C_H_W: torch.Tensor) -> torch.Tensor:
    """Convert ``[B,T,V,C,H,W]`` to contiguous ``[B*V,C,T,H,W]``."""

    if rgb_B_T_V_C_H_W.ndim != 6:
        raise ValueError("RGB tensor must be [B,T,V,C,H,W]")
    batch_size, num_frames, num_views, channels, height, width = rgb_B_T_V_C_H_W.shape
    return rgb_B_T_V_C_H_W.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(
        batch_size * num_views, channels, num_frames, height, width
    )


def rgb_bvcthw_to_btvc(
    rgb_BV_C_T_H_W: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
) -> torch.Tensor:
    """Inverse of :func:`rgb_btvc_to_bvcthw`."""

    if rgb_BV_C_T_H_W.ndim != 5:
        raise ValueError("Flattened RGB tensor must be [B*V,C,T,H,W]")
    if rgb_BV_C_T_H_W.shape[0] != batch_size * num_views:
        raise ValueError(
            f"flat batch {rgb_BV_C_T_H_W.shape[0]} != B*V {batch_size * num_views}"
        )
    _, channels, num_frames, height, width = rgb_BV_C_T_H_W.shape
    return rgb_BV_C_T_H_W.reshape(
        batch_size, num_views, channels, num_frames, height, width
    ).permute(0, 3, 1, 2, 4, 5).contiguous()


def tokens_bvthwd_to_btview(
    tokens_BV_T_H_W_D: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
) -> torch.Tensor:
    """Convert ``[B*V,T,H,W,D]`` to ``[B*T,V,H,W,D]`` without mixing time."""

    if tokens_BV_T_H_W_D.ndim != 5:
        raise ValueError("tokens must be [B*V,T,H,W,D]")
    if tokens_BV_T_H_W_D.shape[0] != batch_size * num_views:
        raise ValueError(
            f"flat batch {tokens_BV_T_H_W_D.shape[0]} != B*V {batch_size * num_views}"
        )
    _, num_token_frames, height, width, dim = tokens_BV_T_H_W_D.shape
    return tokens_BV_T_H_W_D.reshape(
        batch_size, num_views, num_token_frames, height, width, dim
    ).permute(0, 2, 1, 3, 4, 5).contiguous().reshape(
        batch_size * num_token_frames, num_views, height, width, dim
    )


def tokens_btview_to_bvthwd(
    tokens_BT_V_H_W_D: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
    num_token_frames: int,
) -> torch.Tensor:
    """Inverse of :func:`tokens_bvthwd_to_btview`."""

    if tokens_BT_V_H_W_D.ndim != 5:
        raise ValueError("tokens must be [B*T,V,H,W,D]")
    if tokens_BT_V_H_W_D.shape[:2] != (batch_size * num_token_frames, num_views):
        raise ValueError(
            "token layout does not match B/T/V: "
            f"{tuple(tokens_BT_V_H_W_D.shape[:2])} != "
            f"{(batch_size * num_token_frames, num_views)}"
        )
    _, _, height, width, dim = tokens_BT_V_H_W_D.shape
    return tokens_BT_V_H_W_D.reshape(
        batch_size, num_token_frames, num_views, height, width, dim
    ).permute(0, 2, 1, 3, 4, 5).contiguous().reshape(
        batch_size * num_views, num_token_frames, height, width, dim
    )


def encode_camera_sequences(
    tokenizer,
    rgb_B_T_V_C_H_W: torch.Tensor,
    sigma_data: float,
) -> torch.Tensor:
    """Encode each full camera sequence; return ``[B*V,Cz,Tz,Hz,Wz]``."""

    flat = rgb_btvc_to_bvcthw(rgb_B_T_V_C_H_W)
    latent = tokenizer.encode(flat) * sigma_data
    if latent.ndim != 5 or latent.shape[0] != flat.shape[0] or latent.shape[2] < 1:
        raise RuntimeError(
            "Temporal tokenizer encode must return [B*V,Cz,Tz,Hz,Wz], got "
            f"{tuple(latent.shape)}"
        )
    return latent


def decode_camera_sequences(
    tokenizer,
    latent_BV_C_T_H_W: torch.Tensor,
    sigma_data: float,
    *,
    batch_size: int,
    num_views: int,
) -> torch.Tensor:
    """Decode full per-camera sequences; return ``[B,Tout,V,3,H,W]``."""

    if latent_BV_C_T_H_W.ndim != 5:
        raise ValueError("latent must be [B*V,C,T,H,W]")
    decoded = tokenizer.decode(latent_BV_C_T_H_W / sigma_data)
    return rgb_bvcthw_to_btvc(decoded, batch_size=batch_size, num_views=num_views)
