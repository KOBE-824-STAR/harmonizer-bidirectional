"""Scheme B-TMV T3: native temporal DiT plus graph-routed view residuals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .b_tmv_warm import (
    GEOMETRY_VERSION,
    SCHEME,
    TEMPORAL_MODE,
    TOKEN_LAYOUT_VERSION,
    VARIANT,
    rgb_bvcthw_to_btvc,
)
from .constants import CAMERA_ORDER
from .b_tmv_routing import resolve_cross_view_adjacency
from .plucker import PLUCKER_CONVENTION, token_center_plucker_rays


B1_VARIANT = "warm_v1_geometry_t4_plucker_b1"
POSE_B1_VARIANT = "warm_v1_geometry_t4_pose_routing_plucker_b1"


def named_adjacency_matrix(
    adjacency: Mapping[str, Sequence[str]], *, undirected: bool
) -> tuple[tuple[bool, ...], ...]:
    """Resolve target->sources without ever adding reverse edges."""

    expected = set(CAMERA_ORDER)
    if set(adjacency) != expected:
        raise ValueError(
            "camera_adjacency must contain the canonical camera set: "
            f"missing={sorted(expected - set(adjacency))}, "
            f"extra={sorted(set(adjacency) - expected)}"
        )
    matrix = [[False] * len(CAMERA_ORDER) for _ in CAMERA_ORDER]
    for target_name, source_names in adjacency.items():
        if len(source_names) != len(set(source_names)):
            raise ValueError(f"duplicate adjacency source for {target_name}")
        target = CAMERA_ORDER.index(target_name)
        for source_name in source_names:
            if source_name not in expected:
                raise ValueError(f"unknown camera name {source_name}")
            source = CAMERA_ORDER.index(source_name)
            if source == target:
                raise ValueError("cross-view adjacency cannot contain self edges")
            matrix[target][source] = True
    if undirected:
        for target in range(len(CAMERA_ORDER)):
            for source in range(len(CAMERA_ORDER)):
                if matrix[target][source] != matrix[source][target]:
                    raise ValueError(
                        "undirected=true requires explicit reverse edges; none are generated"
                    )
    return tuple(tuple(row) for row in matrix)


class CrossViewTemporalResidual(nn.Module):
    """Attend only over V for fixed batch, latent time, and spatial token."""

    def __init__(
        self,
        model_dim: int,
        *,
        adapter_dim: int = 160,
        num_heads: int = 8,
        max_views: int = 6,
    ) -> None:
        super().__init__()
        if min(model_dim, adapter_dim, num_heads, max_views) < 1:
            raise ValueError("cross-view dimensions must be positive")
        if adapter_dim % num_heads:
            raise ValueError("adapter_dim must be divisible by num_heads")
        self.model_dim = int(model_dim)
        self.adapter_dim = int(adapter_dim)
        self.num_heads = int(num_heads)
        self.head_dim = adapter_dim // num_heads
        self.max_views = int(max_views)
        self.norm = nn.LayerNorm(model_dim)
        self.input_proj = nn.Linear(model_dim, adapter_dim, bias=False)
        self.q_proj = nn.Linear(adapter_dim, adapter_dim, bias=False)
        self.k_proj = nn.Linear(adapter_dim, adapter_dim, bias=False)
        self.v_proj = nn.Linear(adapter_dim, adapter_dim, bias=False)
        self.output_proj = nn.Linear(adapter_dim, model_dim, bias=False)
        self.camera_embedding = nn.Embedding(max_views, adapter_dim)
        nn.init.zeros_(self.camera_embedding.weight)
        self.gate = nn.Parameter(torch.zeros(()))
        self._audit_enabled = False
        self._last_audit: dict[str, Any] | None = None

    def attach_geometry_conditioning(self, geometry_dim: int) -> None:
        """Attach T4 edge-FiLM parameters without changing the default T3 module."""

        if geometry_dim < 1:
            raise ValueError("geometry_dim must be positive")
        if hasattr(self, "geometry_projection") or hasattr(self, "geometry_gate"):
            raise ValueError("geometry conditioning is already attached")
        projection = nn.Linear(int(geometry_dim), 2 * self.adapter_dim)
        projection.to(device=self.gate.device, dtype=self.norm.weight.dtype)
        self.add_module("geometry_projection", projection)
        self.register_parameter("geometry_gate", nn.Parameter(self.gate.new_zeros(())))

    def attach_plucker_conditioning(self, hidden_dim: int = 64) -> None:
        """Attach token-ray Q/K deltas while preserving the complete T4 path."""

        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("Plucker hidden_dim must be a positive integer")
        if hasattr(self, "plucker_projection") or hasattr(self, "plucker_gate"):
            raise ValueError("Plucker conditioning is already attached")
        projection = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * self.adapter_dim),
        ).to(device=self.gate.device, dtype=self.norm.weight.dtype)
        self.add_module("plucker_projection", projection)
        self.register_parameter("plucker_gate", nn.Parameter(self.gate.new_zeros(())))
        self.plucker_hidden_dim = hidden_dim

    def enable_audit(self, enabled: bool = True) -> None:
        self._audit_enabled = bool(enabled)
        self._last_audit = None

    def forward(
        self,
        tokens: torch.Tensor,
        view_ids: torch.Tensor,
        view_valid: torch.Tensor,
        adjacency: torch.Tensor,
        edge_geometry: torch.Tensor | None = None,
        plucker_B_V_N_6: torch.Tensor | None = None,
        force_single_view_residual: bool = False,
    ) -> torch.Tensor:
        if type(force_single_view_residual) is not bool:
            raise TypeError("force_single_view_residual must be bool")
        if tokens.ndim != 5:
            raise ValueError("cross-view tokens must be [B,V,T,N,D]")
        batch, views, latent_t, spatial_tokens, dim = tokens.shape
        if dim != self.model_dim:
            raise ValueError(f"token dim {dim} != model_dim {self.model_dim}")
        if view_ids.shape != (batch, views) or view_valid.shape != (batch, views):
            raise ValueError("view metadata must match [B,V]")
        if view_ids.dtype != torch.long or view_valid.dtype != torch.bool:
            raise TypeError("view IDs must be long and validity must be bool")
        if bool((view_ids < 0).any().item()) or bool((view_ids >= self.max_views).any().item()):
            raise ValueError("view ID lies outside the canonical camera schema")
        positional_graph = adjacency.ndim == 3
        expected_graph = (batch, views, views) if positional_graph else (self.max_views, self.max_views)
        if adjacency.shape != expected_graph:
            raise ValueError("adjacency must be canonical [max_views,max_views] or positional [B,V,V]")
        if adjacency.dtype != torch.bool or adjacency.device != tokens.device:
            raise ValueError("adjacency must be bool on the token device")
        if positional_graph and force_single_view_residual:
            raise ValueError("pose-routed graphs cannot force singleton self-context")
        geometry_active = edge_geometry is not None
        geometry_modulation = False
        if geometry_active:
            if not hasattr(self, "geometry_projection") or not hasattr(
                self, "geometry_gate"
            ):
                raise RuntimeError("edge geometry was supplied before T4 attachment")
            expected = (batch, views, views)
            if edge_geometry.ndim != 4 or edge_geometry.shape[:3] != expected:
                raise ValueError("edge geometry must be [B,V_dst,V_src,D_geo]")
            if edge_geometry.shape[-1] != self.geometry_projection.in_features:
                raise ValueError("edge geometry embedding dimension is invalid")
            if edge_geometry.device != tokens.device:
                raise ValueError("edge geometry must share the token device")
            # Inference with an exactly zero gate can take the literal T3 branch.
            # Grad-enabled execution must retain the product to expose gate grads.
            geometry_modulation = torch.is_grad_enabled() or bool(
                torch.count_nonzero(self.geometry_gate.detach()).item()
            )
        if views == 1 and not force_single_view_residual:
            if self._audit_enabled:
                self._last_audit = {
                    "input_layout": list(tokens.shape),
                    "attention_axis": "V",
                    "flatten_view_time": False,
                    "self_out_norm": float(tokens.detach().float().norm().item()),
                    "cross_out_norm": 0.0,
                    "gated_residual_norm": 0.0,
                    "residual_delta_max": 0.0,
                    "residual_delta_mean": 0.0,
                    "gate": float(self.gate.detach().float().item()),
                    "forced_single_view_residual": False,
                    "geometry_enabled": geometry_active,
                    "geometry_gate": (
                        float(self.geometry_gate.detach().float().item())
                        if geometry_active
                        else None
                    ),
                    "accessed_edges": [],
                }
            return tokens

        embedded = self.input_proj(self.norm(tokens))
        camera = self.camera_embedding(view_ids).unsqueeze(2).unsqueeze(3)
        conditioned = embedded + camera
        shape = (batch, views, latent_t, spatial_tokens, self.num_heads, self.head_dim)
        q = self.q_proj(conditioned).reshape(shape)
        # These remain the original T3 projections and operation order. T4 adds a
        # gated linear delta below, so a zero geometry gate leaves these exact bits.
        k = self.k_proj(conditioned).reshape(shape)
        value = self.v_proj(embedded).reshape(shape)
        if hasattr(self, "plucker_projection"):
            expected = (batch, views, spatial_tokens, 6)
            if plucker_B_V_N_6 is None:
                raise ValueError("B1 requires token-level Plucker rays")
            if tuple(plucker_B_V_N_6.shape) != expected:
                raise ValueError(
                    f"Plucker rays must have shape {expected}, got "
                    f"{tuple(plucker_B_V_N_6.shape)}"
                )
            if plucker_B_V_N_6.device != tokens.device:
                raise ValueError("Plucker rays must share the token device")
            plucker_delta = self.plucker_projection(
                plucker_B_V_N_6.to(dtype=conditioned.dtype)
            )
            query_delta, key_delta = plucker_delta.chunk(2, dim=-1)
            ray_shape = (batch, views, 1, spatial_tokens, self.num_heads, self.head_dim)
            q = q + self.plucker_gate * query_delta[:, :, None].reshape(ray_shape)
            k = k + self.plucker_gate * key_delta[:, :, None].reshape(ray_shape)
        elif plucker_B_V_N_6 is not None:
            raise ValueError("Plucker rays were supplied before B1 attachment")
        cross = torch.zeros_like(q)
        accessed: list[tuple[int, int, int, int]] = []
        for sample in range(batch):
            ids = view_ids[sample]
            valid = view_valid[sample]
            for target_position in range(views):
                if not bool(valid[target_position].item()):
                    continue
                target_id = ids[target_position]
                if views == 1:
                    # Explicit opt-in singleton fallback: the one real camera is
                    # its own context. No view is copied or fabricated.
                    sources = torch.tensor(
                        [target_position], device=tokens.device, dtype=torch.long
                    )
                else:
                    allowed = (
                        adjacency[sample, target_position] if positional_graph
                        else adjacency[target_id, ids]
                    ) & valid
                    allowed[target_position] = False
                    sources = torch.nonzero(allowed, as_tuple=False).flatten()
                if sources.numel() == 0:
                    continue
                target_q = q[sample, target_position].unsqueeze(-2)
                if geometry_modulation:
                    geometry = edge_geometry[sample, target_position, sources].to(
                        dtype=conditioned.dtype
                    )
                    gamma, beta = self.geometry_projection(geometry).chunk(2, dim=-1)
                    key_delta_input = (
                        conditioned[sample, sources] * gamma[:, None, None, :]
                        + beta[:, None, None, :]
                    )
                    value_delta_input = (
                        embedded[sample, sources] * gamma[:, None, None, :]
                        + beta[:, None, None, :]
                    )
                    source_shape = (
                        sources.numel(),
                        latent_t,
                        spatial_tokens,
                        self.num_heads,
                        self.head_dim,
                    )
                    key_delta = self.k_proj(key_delta_input).reshape(source_shape)
                    value_delta = self.v_proj(value_delta_input).reshape(source_shape)
                    source_k = k[sample, sources] + self.geometry_gate * key_delta
                    source_v = value[sample, sources] + self.geometry_gate * value_delta
                    source_k = source_k.permute(1, 2, 3, 4, 0)
                    source_v = source_v.permute(1, 2, 3, 0, 4)
                else:
                    source_k = k[sample, sources].permute(1, 2, 3, 4, 0)
                    source_v = value[sample, sources].permute(1, 2, 3, 0, 4)
                scores = torch.matmul(target_q, source_k).squeeze(-2) * self.head_dim**-0.5
                weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
                cross[sample, target_position] = torch.sum(
                    weights.unsqueeze(-1) * source_v, dim=-2
                )
                if self._audit_enabled:
                    accessed.extend(
                        (
                            sample,
                            int(ids[source].item()),
                            int(target_id.item()),
                            time_index,
                        )
                        for source in sources.tolist()
                        for time_index in range(latent_t)
                    )
        cross_out = self.output_proj(cross.reshape(batch, views, latent_t, spatial_tokens, -1))
        gated = self.gate * cross_out
        if self._audit_enabled:
            detached_delta = gated.detach().float()
            self._last_audit = {
                "input_layout": list(tokens.shape),
                "attention_axis": "V",
                "flatten_view_time": False,
                "self_out_norm": float(tokens.detach().float().norm().item()),
                "cross_out_norm": float(cross_out.detach().float().norm().item()),
                "gated_residual_norm": float(gated.detach().float().norm().item()),
                "residual_delta_max": float(detached_delta.abs().max().item()),
                "residual_delta_mean": float(detached_delta.abs().mean().item()),
                "gate": float(self.gate.detach().float().item()),
                "forced_single_view_residual": bool(
                    views == 1 and force_single_view_residual
                ),
                "geometry_enabled": geometry_active,
                "geometry_gate": (
                    float(self.geometry_gate.detach().float().item())
                    if geometry_active
                    else None
                ),
                "geometry_embedding_shape": (
                    list(edge_geometry.shape) if geometry_active else None
                ),
                "accessed_edges": [list(edge) for edge in accessed],
            }
        return tokens + gated


def _expand_first_dim(
    value: torch.Tensor | None, *, name: str, batch: int, views: int
) -> torch.Tensor | None:
    if value is None:
        return None
    if value.shape[0] == batch * views:
        return value
    if value.shape[0] != batch:
        raise ValueError(f"{name} leading dimension must be B or B*V")
    return value.unsqueeze(1).expand(
        (batch, views) + tuple(value.shape[1:])
    ).reshape((batch * views,) + tuple(value.shape[1:]))


class TemporalLatentCrossViewDiT:
    """Runtime mixin that retains official blocks and adds one residual each."""

    @classmethod
    def upgrade_from_official(
        cls,
        dit: nn.Module,
        *,
        camera_adjacency: Mapping[str, Sequence[str]],
        adjacency_undirected: bool,
        max_views: int = 6,
        adapter_dim: int = 160,
        adapter_heads: int = 8,
    ) -> nn.Module:
        if hasattr(dit, "mv_architecture"):
            raise ValueError("DiT has already been upgraded")
        for name in ("patch_temporal", "blocks", "model_channels"):
            if not hasattr(dit, name):
                raise TypeError(f"official DiT missing {name}")
        if int(dit.patch_temporal) != 1:
            raise ValueError("B-TMV requires official patch_temporal=1")
        if max_views != len(CAMERA_ORDER):
            raise ValueError("B-TMV T3 uses the complete canonical camera schema")
        matrix = named_adjacency_matrix(camera_adjacency, undirected=adjacency_undirected)
        parameter = next(dit.parameters())
        residuals = nn.ModuleList(
            [
                CrossViewTemporalResidual(
                    int(dit.model_channels),
                    adapter_dim=adapter_dim,
                    num_heads=adapter_heads,
                    max_views=max_views,
                )
                for _ in dit.blocks
            ]
        ).to(device=parameter.device, dtype=parameter.dtype)
        original_class = dit.__class__
        runtime_class = type(
            "TemporalLatentCrossViewDiTRuntime",
            (cls, original_class),
            {"__module__": cls.__module__},
        )
        dit.__class__ = runtime_class
        dit.add_module("tmv_cross_view_residuals", residuals)
        dit.mv_architecture = SCHEME
        dit.mv_attention_variant = VARIANT
        dit.mv_temporal = True
        dit.mv_temporal_mode = TEMPORAL_MODE
        dit.mv_geometry_version = GEOMETRY_VERSION
        dit.mv_token_layout_version = TOKEN_LAYOUT_VERSION
        dit.mv_max_views = max_views
        dit.mv_camera_adjacency = matrix
        dit.mv_adjacency_undirected = bool(adjacency_undirected)
        dit.mv_cross_view_parameters_attached = True
        dit.mv_cross_view_mode = "same_latent_time_same_spatial_token_view_only"
        dit.mv_gradient_checkpointing = False
        dit.mv_t3_original_class = original_class.__name__
        return dit

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Checkpoint official blocks and T3 residuals without changing T/V layout."""

        self.mv_gradient_checkpointing = bool(enabled)

    def _metadata(self, flat_batch, view_ids, view_valid, device):
        if view_ids is None or view_valid is None or view_ids.ndim != 2:
            raise ValueError("B-TMV requires explicit [B,V] view metadata")
        batch, views = view_ids.shape
        if flat_batch != batch * views or view_valid.shape != (batch, views):
            raise ValueError("view metadata does not match DiT B*V")
        if view_ids.dtype != torch.long or view_valid.dtype != torch.bool:
            raise TypeError("view IDs/validity dtypes are invalid")
        if view_ids.device != device or view_valid.device != device:
            raise ValueError("view metadata must share the token device")
        if not 1 <= views <= self.mv_max_views or not bool(view_valid.all().item()):
            raise ValueError("invalid or fabricated camera sequence")
        for sample in range(batch):
            if view_ids[sample].unique().numel() != views:
                raise ValueError(f"duplicate camera ID in sample {sample}")
        return batch, views, view_ids, view_valid

    def _conditions(self, timesteps, context, fps, padding, batch, views):
        return (
            _expand_first_dim(timesteps, name="timesteps", batch=batch, views=views),
            _expand_first_dim(context, name="crossattn_emb", batch=batch, views=views),
            _expand_first_dim(fps, name="fps", batch=batch, views=views),
            _expand_first_dim(padding, name="padding_mask", batch=batch, views=views),
        )

    def forward_official_temporal_reference(
        self,
        x_B_C_T_H_W,
        timesteps_B_T,
        crossattn_emb,
        fps=None,
        padding_mask=None,
        data_type=None,
        use_cuda_graphs=False,
        *,
        view_ids_B_V=None,
        view_valid_B_V=None,
        **unused,
    ):
        if unused or use_cuda_graphs:
            raise TypeError(f"unsupported official-reference options: {sorted(unused)}")
        batch, views, _, _ = self._metadata(
            x_B_C_T_H_W.shape[0], view_ids_B_V, view_valid_B_V, x_B_C_T_H_W.device
        )
        timesteps, context, fps, padding_mask = self._conditions(
            timesteps_B_T, crossattn_emb, fps, padding_mask, batch, views
        )
        return super(TemporalLatentCrossViewDiT, self).forward(
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps,
            crossattn_emb=context,
            fps=fps,
            padding_mask=padding_mask,
            data_type=data_type,
            use_cuda_graphs=False,
        )

    def forward(
        self,
        x_B_C_T_H_W,
        timesteps_B_T,
        crossattn_emb,
        fps=None,
        padding_mask=None,
        data_type=None,
        use_cuda_graphs=False,
        *,
        view_ids_B_V=None,
        view_valid_B_V=None,
        intrinsics_B_V_3_3=None,
        T_ego_from_cam_B_V_4_4=None,
        image_hw=None,
        force_single_view_residual: bool = False,
        **unused,
    ):
        if unused or use_cuda_graphs:
            raise TypeError(f"unsupported B-TMV options: {sorted(unused)}")
        if x_B_C_T_H_W.ndim != 5 or x_B_C_T_H_W.shape[2] < 1:
            raise ValueError("DiT input must be [B*V,C,T_latent,H,W]")
        batch, views, view_ids, view_valid = self._metadata(
            x_B_C_T_H_W.shape[0], view_ids_B_V, view_valid_B_V, x_B_C_T_H_W.device
        )
        timesteps, context, fps, padding_mask = self._conditions(
            timesteps_B_T, crossattn_emb, fps, padding_mask, batch, views
        )
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        tokens, rope, extra_pos = super().prepare_embedded_sequence(
            x_B_C_T_H_W, fps=fps, padding_mask=padding_mask
        )
        if self.crossattn_proj is not None:
            context = self.crossattn_proj(context)
        embedding, adaln = self.t_embedder(timesteps)
        embedding = self.t_embedding_norm(embedding)
        self.affline_scale_log_info = {"t_embedding_B_T_D": embedding.detach()}
        self.affline_emb = embedding
        self.crossattn_emb = context
        adjacency = resolve_cross_view_adjacency(
            self, intrinsics_B_V_3_3, T_ego_from_cam_B_V_4_4,
            image_hw=image_hw, view_valid=view_valid, device=tokens.device,
        )
        edge_geometry = None
        if bool(getattr(self, "mv_geometry_conditioning_enabled", False)):
            if intrinsics_B_V_3_3 is None or T_ego_from_cam_B_V_4_4 is None:
                raise ValueError(
                    "enabled relative geometry requires intrinsics and T_ego_from_cam"
                )
            if image_hw is None:
                raise ValueError("enabled relative geometry requires image_hw=(H,W)")
            edge_geometry = self.tmv_geometry_encoder.dense_edge_embeddings(
                intrinsics_B_V_3_3,
                T_ego_from_cam_B_V_4_4,
                image_hw=image_hw,
                view_ids=view_ids,
                view_valid=view_valid,
                adjacency=adjacency,
            )
        plucker_B_V_N_6 = None
        if bool(getattr(self, "mv_plucker_conditioning_enabled", False)):
            if intrinsics_B_V_3_3 is None or T_ego_from_cam_B_V_4_4 is None:
                raise ValueError("B1 requires intrinsics and T_ego_from_cam")
            if image_hw is None:
                raise ValueError("B1 requires source image_hw")
            plucker_grid = token_center_plucker_rays(
                intrinsics_B_V_3_3,
                T_ego_from_cam_B_V_4_4,
                image_hw=image_hw,
                feature_hw=(tokens.shape[2], tokens.shape[3]),
            )
            plucker_B_V_N_6 = plucker_grid.reshape(
                batch, views, tokens.shape[2] * tokens.shape[3], 6
            )
        block_kwargs = {
            "rope_emb_L_1_1_D": rope,
            "adaln_lora_B_T_3D": adaln,
            "extra_per_block_pos_emb": extra_pos,
        }
        checkpointing = bool(getattr(self, "mv_gradient_checkpointing", False)) and (
            self.tmv_cross_view_residuals.training and torch.is_grad_enabled()
        )
        for block, residual in zip(self.blocks, self.tmv_cross_view_residuals, strict=True):
            def run_block(value, active_block=block):
                return active_block(value, embedding, context, **block_kwargs)

            tokens = (
                checkpoint(run_block, tokens, use_reentrant=False)
                if checkpointing
                else run_block(tokens)
            )
            _, latent_t, height, width, dim = tokens.shape
            explicit = tokens.reshape(batch, views, latent_t, height * width, dim)

            def run_residual(value, active_residual=residual):
                return active_residual(
                    value,
                    view_ids,
                    view_valid,
                    adjacency,
                    edge_geometry=edge_geometry,
                    plucker_B_V_N_6=plucker_B_V_N_6,
                    force_single_view_residual=force_single_view_residual,
                )

            explicit = (
                checkpoint(run_residual, explicit, use_reentrant=False)
                if checkpointing
                else run_residual(explicit)
            )
            tokens = explicit.reshape(batch * views, latent_t, height, width, dim)
        output = self.final_layer(tokens, embedding, adaln_lora_B_T_3D=adaln)
        return self.unpatchify(output)


def attach_b1_plucker_conditioning(
    dit: nn.Module, *, hidden_dim: int = 64, gate_init: float = 0.0
) -> nn.Module:
    """Attach zero-gated B1 rays to T4 or its pose-routed continuation."""

    old_variant = getattr(dit, "mv_attention_variant", None)
    if old_variant not in ("warm_v1_geometry_t4", "warm_v1_geometry_t4_pose_routing"):
        raise ValueError("B1 must attach to T4 geometry or T4 pose routing")
    if not bool(getattr(dit, "mv_geometry_conditioning_enabled", False)):
        raise ValueError("B1 requires active T4 edge geometry")
    if bool(getattr(dit, "mv_plucker_conditioning_attached", False)):
        raise ValueError("B1 Plucker conditioning is already attached")
    if float(gate_init) != 0.0:
        raise ValueError("B1 warm-start preservation requires gate_init=0")
    for residual in dit.tmv_cross_view_residuals:
        residual.attach_plucker_conditioning(hidden_dim)
    dit.mv_plucker_conditioning_attached = True
    dit.mv_plucker_conditioning_enabled = True
    new_variant = POSE_B1_VARIANT if old_variant == "warm_v1_geometry_t4_pose_routing" else B1_VARIANT
    dit.mv_plucker_variant = new_variant
    dit.mv_plucker_dim = 6
    dit.mv_plucker_hidden_dim = int(hidden_dim)
    dit.mv_plucker_gate_initialization = 0.0
    dit.mv_plucker_convention = PLUCKER_CONVENTION
    dit.mv_geometry_extrinsic_convention = "T_ego_from_cam"
    dit.mv_attention_variant = new_variant
    return dit


@dataclass(frozen=True)
class TemporalMultiViewMetadata:
    architecture: str
    variant: str
    temporal_mode: str
    temporal_contract_source: str
    batch_size: int
    num_views: int
    rgb_T: int
    latent_T: int
    decoded_T: int
    input_shape: tuple[int, ...]
    logical_latent_shape: tuple[int, ...]
    dit_input_shape: tuple[int, ...]
    decoded_shape: tuple[int, ...]
    camera_axis_enters_temporal_encoder: bool
    flatten_view_time: bool
    cross_view_mode: str


class TemporalMultiViewWrapper(nn.Module):
    """T3 wrapper with runtime-discovered temporal lengths and view-safe VAE calls."""

    def __init__(
        self,
        pipeline: nn.Module,
        *,
        temporal_capability_path: str | Path,
        timestep: int = 250,
    ) -> None:
        super().__init__()
        if not 0 <= timestep <= 1000:
            raise ValueError("timestep must be in [0,1000]")
        if getattr(pipeline.dit, "mv_architecture", None) != SCHEME:
            raise ValueError("pipeline.dit must be upgraded to B_TMV")
        if not bool(getattr(pipeline.dit, "mv_cross_view_parameters_attached", False)):
            raise ValueError("T3 requires cross-view parameters")
        capability_path = Path(temporal_capability_path)
        capability = json.loads(capability_path.read_text(encoding="utf-8"))
        if capability.get("status") != "PASS_DISCOVERY_COMPLETE":
            raise ValueError("temporal capability runtime discovery did not pass")
        if capability.get("contract_source") not in {
            "runtime_discovery",
            "live_official_runtime",
        }:
            raise ValueError("temporal capability is not runtime-discovered evidence")
        supported = capability.get("supported_rgb_lengths")
        mapping_rows = capability.get("mapping")
        if not isinstance(supported, list) or not supported:
            raise ValueError("temporal capability has no runtime-supported lengths")
        if not isinstance(mapping_rows, list):
            raise ValueError("temporal capability has no runtime mapping")
        self.supported_rgb_lengths = tuple(int(value) for value in supported)
        if len(set(self.supported_rgb_lengths)) != len(self.supported_rgb_lengths):
            raise ValueError("temporal capability contains duplicate lengths")
        self.temporal_mapping = {
            int(row["rgb_T"]): (int(row["latent_T"]), int(row["decoded_T"]))
            for row in mapping_rows
        }
        if set(self.temporal_mapping) != set(self.supported_rgb_lengths):
            raise ValueError("supported lengths and runtime mapping disagree")
        self.pipeline = pipeline
        self.timestep = int(timestep)
        self.temporal_capability_path = str(capability_path)
        self.temporal_contract_source = "runtime_discovery"
        self.architecture = SCHEME
        self._condition_cache: dict[tuple[Any, ...], Any] = {}
        self.last_rgb_T: int | None = None
        self.last_latent_T: int | None = None
        self.pipeline.requires_grad_(False)
        for residual in self.dit.tmv_cross_view_residuals:
            residual.requires_grad_(True)
        geometry_encoder = getattr(self.dit, "tmv_geometry_encoder", None)
        if geometry_encoder is not None:
            geometry_encoder.requires_grad_(True)
        self._freeze_tokenizer()
        self.pipeline.eval()

    @property
    def tokenizer(self) -> Any:
        return self.pipeline.tokenizer

    @property
    def dit(self) -> nn.Module:
        return self.pipeline.dit

    def _freeze_tokenizer(self) -> None:
        tokenizer = self.tokenizer
        if isinstance(tokenizer, nn.Module):
            module = tokenizer
        else:
            from .tokenizer_utils import inner_tokenizer_module

            module = inner_tokenizer_module(tokenizer)
        module.requires_grad_(False)
        module.eval()

    def train(self, mode: bool = True):
        """Train only the added residuals; the native temporal prior stays frozen/eval."""

        super().train(mode)
        self.pipeline.eval()
        for residual in self.dit.tmv_cross_view_residuals:
            residual.train(mode)
        geometry_encoder = getattr(self.dit, "tmv_geometry_encoder", None)
        if geometry_encoder is not None:
            geometry_encoder.train(mode)
        self._freeze_tokenizer()
        return self

    def _require_supported(self, rgb_t: int) -> tuple[int, int]:
        if rgb_t not in self.temporal_mapping:
            raise ValueError(
                f"T_rgb={rgb_t} was not supported by runtime discovery; "
                f"supported_rgb_lengths={list(self.supported_rgb_lengths)}"
            )
        return self.temporal_mapping[rgb_t]

    def encode_camera_sequences(
        self, source_B_T_V_C_H_W: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode each ``[B,C,T,H,W]`` camera independently, then stack on V."""

        if source_B_T_V_C_H_W.ndim != 6:
            raise ValueError("source must be [B,T,V,C,H,W]")
        batch, rgb_t, views, channels, _, _ = source_B_T_V_C_H_W.shape
        if channels != 3:
            raise ValueError("B-TMV input must have three RGB channels")
        expected_latent_t, _ = self._require_supported(rgb_t)
        encoded = []
        for view in range(views):
            camera_B_C_T_H_W = source_B_T_V_C_H_W[:, :, view].permute(
                0, 2, 1, 3, 4
            ).contiguous()
            encoded.append(self.tokenizer.encode(camera_B_C_T_H_W) * self.pipeline.sigma_data)
        logical = torch.stack(encoded, dim=1)
        if logical.ndim != 6 or logical.shape[:2] != (batch, views):
            raise RuntimeError("unexpected logical latent layout")
        if logical.shape[3] != expected_latent_t:
            raise RuntimeError(
                "runtime tokenizer latent length disagrees with temporal_capability.json"
            )
        flat = logical.reshape(batch * views, *logical.shape[2:])
        self.last_rgb_T = rgb_t
        self.last_latent_T = int(logical.shape[3])
        return logical, flat

    def _condition(self, batch: int, height: int, width: int, source: torch.Tensor) -> Any:
        key = (batch, height, width, str(source.device), source.dtype)
        if key not in self._condition_cache:
            from cosmos_predict2.conditioner import DataType

            data = {
                "dataset_name": "image_data",
                "images": torch.zeros(
                    (batch, 3, height, width), device=source.device, dtype=source.dtype
                ),
                "t5_text_embeddings": torch.zeros(
                    (batch, 512, 1024), device=source.device, dtype=source.dtype
                ),
                "fps": torch.full(
                    (batch,), 24, device=source.device, dtype=source.dtype
                ),
                "padding_mask": torch.zeros(
                    (batch, 1, height, width), device=source.device, dtype=source.dtype
                ),
            }
            _, uncondition = self.pipeline.conditioner.get_condition_uncondition(data)
            self._condition_cache[key] = uncondition.edit_data_type(DataType.IMAGE)
        return self._condition_cache[key]

    @staticmethod
    def _expand_views(value: torch.Tensor, batch: int, views: int) -> torch.Tensor:
        return value.unsqueeze(1).expand(
            (batch, views) + tuple(value.shape[1:])
        ).reshape((batch * views,) + tuple(value.shape[1:]))

    def denoise(
        self,
        latent_BV_C_T_H_W: torch.Tensor,
        *,
        batch_size: int,
        num_views: int,
        source_height: int,
        source_width: int,
        source: torch.Tensor,
        view_ids_B_V: torch.Tensor,
        view_valid_B_V: torch.Tensor,
        intrinsics: torch.Tensor | None = None,
        T_ego_from_cam: torch.Tensor | None = None,
        official_reference: bool = False,
        force_single_view_residual: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = torch.full(
            (batch_size, 1),
            self.timestep / 1000.0,
            device=latent_BV_C_T_H_W.device,
            dtype=latent_BV_C_T_H_W.dtype,
        )
        c_skip, c_out, c_in, c_noise = self.pipeline.scaling(
            sigma=sigma[:, None, :, None, None]
        )
        c_skip_bv, c_out_bv, c_in_bv = (
            self._expand_views(value, batch_size, num_views)
            for value in (c_skip, c_out, c_in)
        )
        condition = self._condition(batch_size, source_height, source_width, source)
        method = (
            self.dit.forward_official_temporal_reference
            if official_reference
            else self.dit.forward
        )
        geometry_kwargs = {}
        if not official_reference:
            geometry_kwargs = {
                "intrinsics_B_V_3_3": intrinsics,
                "T_ego_from_cam_B_V_4_4": T_ego_from_cam,
                "image_hw": (source_height, source_width),
                "force_single_view_residual": force_single_view_residual,
            }
        network_output = method(
            x_B_C_T_H_W=(latent_BV_C_T_H_W * c_in_bv).to(
                **self.pipeline.tensor_kwargs
            ),
            timesteps_B_T=c_noise.squeeze(dim=[1, 3, 4]).to(
                **self.pipeline.tensor_kwargs
            ),
            **condition.to_dict(),
            view_ids_B_V=view_ids_B_V,
            view_valid_B_V=view_valid_B_V,
            use_cuda_graphs=False,
            **geometry_kwargs,
        ).float()
        denoised = c_skip_bv * latent_BV_C_T_H_W + c_out_bv * network_output
        return network_output, denoised

    def decode(
        self, latent_BV_C_T_H_W: torch.Tensor, *, batch_size: int, num_views: int
    ) -> torch.Tensor:
        decoded = self.tokenizer.decode(latent_BV_C_T_H_W / self.pipeline.sigma_data)
        return rgb_bvcthw_to_btvc(decoded, batch_size=batch_size, num_views=num_views)

    def _forward_impl(
        self,
        source_B_T_V_C_H_W: torch.Tensor,
        *,
        view_ids_B_V: torch.Tensor,
        view_valid_B_V: torch.Tensor,
        intrinsics: torch.Tensor | None,
        T_ego_from_cam: torch.Tensor | None,
        official_reference: bool,
        return_metadata: bool,
        decode_current_only: bool,
        force_single_view_residual: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, TemporalMultiViewMetadata]:
        if source_B_T_V_C_H_W.ndim != 6:
            raise ValueError("source must be [B,T,V,C,H,W]")
        batch, rgb_t, views, _, height, width = source_B_T_V_C_H_W.shape
        latent_t, decoded_t = self._require_supported(rgb_t)
        logical, flat = self.encode_camera_sequences(source_B_T_V_C_H_W)
        network_output, denoised = self.denoise(
            flat,
            batch_size=batch,
            num_views=views,
            source_height=height,
            source_width=width,
            source=source_B_T_V_C_H_W,
            view_ids_B_V=view_ids_B_V,
            view_valid_B_V=view_valid_B_V,
            intrinsics=intrinsics,
            T_ego_from_cam=T_ego_from_cam,
            official_reference=official_reference,
            force_single_view_residual=force_single_view_residual,
        )
        decode_latent = denoised[:, :, :1] if decode_current_only else denoised
        decoded = self.decode(decode_latent, batch_size=batch, num_views=views)
        expected_decoded_t = 1 if decode_current_only else decoded_t
        if decoded.shape[1] != expected_decoded_t:
            raise RuntimeError(
                "runtime tokenizer decoded length disagrees with temporal_capability.json"
            )
        output = decoded[:, 0] if decode_current_only else decoded
        if not return_metadata:
            return output
        metadata = TemporalMultiViewMetadata(
            architecture=SCHEME,
            variant=getattr(self.dit, "mv_attention_variant", VARIANT),
            temporal_mode=TEMPORAL_MODE,
            temporal_contract_source=self.temporal_contract_source,
            batch_size=batch,
            num_views=views,
            rgb_T=rgb_t,
            latent_T=latent_t,
            decoded_T=expected_decoded_t,
            input_shape=tuple(source_B_T_V_C_H_W.shape),
            logical_latent_shape=tuple(logical.shape),
            dit_input_shape=tuple(flat.shape),
            decoded_shape=tuple(output.shape),
            camera_axis_enters_temporal_encoder=False,
            flatten_view_time=False,
            cross_view_mode=self.dit.mv_cross_view_mode,
        )
        return output, metadata

    def forward(
        self,
        source_B_T_V_C_H_W: torch.Tensor,
        *,
        view_ids_B_V: torch.Tensor,
        view_valid_B_V: torch.Tensor,
        intrinsics: torch.Tensor | None = None,
        T_ego_from_cam: torch.Tensor | None = None,
        return_metadata: bool = False,
        decode_current_only: bool = False,
        force_single_view_residual: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, TemporalMultiViewMetadata]:
        return self._forward_impl(
            source_B_T_V_C_H_W,
            view_ids_B_V=view_ids_B_V,
            view_valid_B_V=view_valid_B_V,
            intrinsics=intrinsics,
            T_ego_from_cam=T_ego_from_cam,
            official_reference=False,
            return_metadata=return_metadata,
            decode_current_only=decode_current_only,
            force_single_view_residual=force_single_view_residual,
        )

    def forward_official_temporal_reference(
        self,
        source_B_T_V_C_H_W: torch.Tensor,
        *,
        view_ids_B_V: torch.Tensor,
        view_valid_B_V: torch.Tensor,
        return_metadata: bool = False,
        decode_current_only: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, TemporalMultiViewMetadata]:
        return self._forward_impl(
            source_B_T_V_C_H_W,
            view_ids_B_V=view_ids_B_V,
            view_valid_B_V=view_valid_B_V,
            intrinsics=None,
            T_ego_from_cam=None,
            official_reference=True,
            return_metadata=return_metadata,
            decode_current_only=decode_current_only,
            force_single_view_residual=False,
        )


def cross_view_gates(dit: nn.Module) -> list[nn.Parameter]:
    """Return the T3 gates in official block order."""

    residuals = getattr(dit, "tmv_cross_view_residuals", None)
    if residuals is None:
        raise ValueError("DiT has no B-TMV cross-view residuals")
    return [residual.gate for residual in residuals]
