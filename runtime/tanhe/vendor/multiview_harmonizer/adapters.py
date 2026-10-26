"""Scheme A cross-view adapter; no alternative model weights are used."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def canonical_camera_adjacency(max_views: int = 6) -> torch.Tensor:
    """Return the nuScenes camera-ring adjacency matrix without self edges."""

    if max_views != 6:
        raise ValueError("The current camera graph is defined for the six nuScenes cameras")
    adjacency = torch.zeros((max_views, max_views), dtype=torch.bool)
    for left, right in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)):
        adjacency[left, right] = True
        adjacency[right, left] = True
    return adjacency


class PooledCrossViewAdapter(nn.Module):
    """Query full-resolution tokens against pooled neighboring-camera tokens.

    The scalar residual gate is exactly zero at construction, so attaching this
    module does not change official DiffusionHarmonizer outputs.
    """

    def __init__(
        self,
        model_dim: int,
        adapter_dim: int = 320,
        num_heads: int = 8,
        pool_factor: int = 4,
        max_views: int = 6,
        neighbor_only: bool = True,
    ) -> None:
        super().__init__()
        if adapter_dim % num_heads:
            raise ValueError("adapter_dim must be divisible by num_heads")
        if pool_factor < 1:
            raise ValueError("pool_factor must be >= 1")
        self.model_dim = model_dim
        self.adapter_dim = adapter_dim
        self.pool_factor = pool_factor
        self.max_views = max_views
        self.neighbor_only = neighbor_only
        self.norm = nn.LayerNorm(model_dim)
        self.input_projection = nn.Linear(model_dim, adapter_dim, bias=False)
        self.camera_embedding = nn.Embedding(max_views, adapter_dim)
        self.attention = nn.MultiheadAttention(
            adapter_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_projection = nn.Linear(adapter_dim, model_dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))
        self.register_buffer("camera_adjacency", canonical_camera_adjacency(max_views), persistent=True)
        nn.init.zeros_(self.camera_embedding.weight)

    def attach_plucker_conditioning(self, hidden_dim: int = 64) -> None:
        """Attach A1 token-ray conditioning without changing the A0 path."""

        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("Plucker hidden_dim must be a positive integer")
        if hasattr(self, "plucker_projection") or hasattr(self, "plucker_gate"):
            raise ValueError("Plucker conditioning is already attached")
        projection = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.adapter_dim),
        ).to(device=self.gate.device, dtype=self.norm.weight.dtype)
        self.add_module("plucker_projection", projection)
        self.register_parameter("plucker_gate", nn.Parameter(self.gate.new_zeros(())))
        self.plucker_hidden_dim = hidden_dim

    def forward(
        self,
        x_B_V_T_H_W_D: torch.Tensor,
        camera_ids_B_V: torch.Tensor,
        view_valid_B_V: torch.Tensor,
        plucker_B_V_T_H_W_6: torch.Tensor | None = None,
        adjacency_B_V_V: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_B_V_T_H_W_D.ndim != 6:
            raise ValueError("Adapter input must be [B,V,T,H,W,D]")
        batch, views, time, height, width, dim = x_B_V_T_H_W_D.shape
        if dim != self.model_dim:
            raise ValueError(f"Expected D={self.model_dim}, got {dim}")
        if camera_ids_B_V.shape != (batch, views) or view_valid_B_V.shape != (batch, views):
            raise ValueError("camera_ids/view_valid do not match adapter batch and views")
        if camera_ids_B_V.dtype != torch.long or view_valid_B_V.dtype != torch.bool:
            raise TypeError("camera_ids must be long and view_valid must be bool")
        if camera_ids_B_V.min() < 0 or camera_ids_B_V.max() >= self.max_views:
            raise ValueError("camera_ids are outside the configured embedding range")
        if adjacency_B_V_V is not None and (
            adjacency_B_V_V.shape != (batch, views, views)
            or adjacency_B_V_V.dtype != torch.bool
            or adjacency_B_V_V.device != x_B_V_T_H_W_D.device
        ):
            raise ValueError("pose adjacency must be bool [B,V_dst,V_src] on the token device")

        normalized = self.norm(x_B_V_T_H_W_D)
        full_B_V_N_D = normalized.reshape(batch, views, time * height * width, dim)
        projected_B_V_N_A = self.input_projection(full_B_V_N_D)
        camera_A = self.camera_embedding(camera_ids_B_V)
        projected_B_V_N_A = projected_B_V_N_A + camera_A.unsqueeze(2)

        plucker_full_B_V_N_A = None
        if hasattr(self, "plucker_projection"):
            if plucker_B_V_T_H_W_6 is None:
                raise ValueError("A1 requires token-level Plucker rays")
            expected = (batch, views, time, height, width, 6)
            if tuple(plucker_B_V_T_H_W_6.shape) != expected:
                raise ValueError(
                    f"Plucker rays must have shape {expected}, got "
                    f"{tuple(plucker_B_V_T_H_W_6.shape)}"
                )
            if plucker_B_V_T_H_W_6.device != x_B_V_T_H_W_D.device:
                raise ValueError("Plucker rays must share the token device")
            plucker_delta = self.plucker_projection(
                plucker_B_V_T_H_W_6.to(dtype=self.norm.weight.dtype)
            )
            plucker_full_B_V_N_A = plucker_delta.reshape(
                batch, views, time * height * width, self.adapter_dim
            )
            projected_B_V_N_A = (
                projected_B_V_N_A
                + self.plucker_gate * plucker_full_B_V_N_A
            )
        elif plucker_B_V_T_H_W_6 is not None:
            raise ValueError("Plucker rays were supplied before A1 attachment")

        spatial_BVT_D_H_W = normalized.permute(0, 1, 2, 5, 3, 4).reshape(
            batch * views * time, dim, height, width
        )
        pooled_h = max(1, (height + self.pool_factor - 1) // self.pool_factor)
        pooled_w = max(1, (width + self.pool_factor - 1) // self.pool_factor)
        pooled = F.adaptive_avg_pool2d(spatial_BVT_D_H_W, (pooled_h, pooled_w))
        pooled_B_V_K_D = pooled.reshape(
            batch, views, time, dim, pooled_h, pooled_w
        ).permute(0, 1, 2, 4, 5, 3).reshape(
            batch, views, time * pooled_h * pooled_w, dim
        )
        context_B_V_K_A = self.input_projection(pooled_B_V_K_D) + camera_A.unsqueeze(2)
        if plucker_full_B_V_N_A is not None:
            plucker_spatial = plucker_full_B_V_N_A.reshape(
                batch, views, time, height, width, self.adapter_dim
            ).permute(0, 1, 2, 5, 3, 4).reshape(
                batch * views * time, self.adapter_dim, height, width
            )
            pooled_plucker = F.adaptive_avg_pool2d(
                plucker_spatial, (pooled_h, pooled_w)
            )
            pooled_plucker = pooled_plucker.reshape(
                batch, views, time, self.adapter_dim, pooled_h, pooled_w
            ).permute(0, 1, 2, 4, 5, 3).reshape(
                batch,
                views,
                time * pooled_h * pooled_w,
                self.adapter_dim,
            )
            context_B_V_K_A = (
                context_B_V_K_A + self.plucker_gate * pooled_plucker
            )
        context_B_VK_A = context_B_V_K_A.reshape(
            batch, views * context_B_V_K_A.shape[2], self.adapter_dim
        )
        tokens_per_context_view = context_B_V_K_A.shape[2]

        messages: list[torch.Tensor] = []
        for query_position in range(views):
            query_camera = camera_ids_B_V[:, query_position]
            if adjacency_B_V_V is not None:
                eligible_B_V = adjacency_B_V_V[:, query_position].clone()
                eligible_B_V[:, query_position] = False
            elif self.neighbor_only:
                eligible_B_V = self.camera_adjacency[query_camera.unsqueeze(1), camera_ids_B_V]
            else:
                eligible_B_V = camera_ids_B_V != query_camera.unsqueeze(1)
            eligible_B_V = eligible_B_V & view_valid_B_V
            empty_B = ~eligible_B_V.any(dim=1)
            safe_eligible = eligible_B_V.clone()
            if empty_B.any():
                safe_eligible[empty_B, query_position] = True
            key_padding_mask = ~safe_eligible.repeat_interleave(tokens_per_context_view, dim=1)
            message, _ = self.attention(
                projected_B_V_N_A[:, query_position],
                context_B_VK_A,
                context_B_VK_A,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            active = view_valid_B_V[:, query_position] & ~empty_B
            message = message * active[:, None, None].to(message.dtype)
            messages.append(message)

        message_B_V_N_A = torch.stack(messages, dim=1)
        residual_B_V_N_D = self.output_projection(message_B_V_N_A)
        residual = residual_B_V_N_D.reshape(batch, views, time, height, width, dim)
        return x_B_V_T_H_W_D + torch.tanh(self.gate) * residual
