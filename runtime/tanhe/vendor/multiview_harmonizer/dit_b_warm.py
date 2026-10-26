"""Scheme B-warm: official per-view attention plus zero-gated edge cross-view attention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.models.text2image_dit import Attention, MiniTrainDIT

from .constants import CAMERA_ORDER
from .dit_utils import module_device_dtype, validate_view_metadata


def _attention_module(block: nn.Module) -> Attention:
    candidates = [
        module
        for name, module in block.named_modules()
        if name.endswith("self_attn")
        and hasattr(module, "compute_qkv")
        and hasattr(module, "compute_attention")
        and hasattr(module, "q_proj")
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one official self-attention module, found {len(candidates)}")
    return candidates[0]


def canonical_adjacency_matrix(
    adjacency: Mapping[str, Sequence[str]],
    *,
    undirected: bool = False,
) -> tuple[tuple[bool, ...], ...]:
    names = set(CAMERA_ORDER)
    if set(adjacency) != names:
        raise ValueError(
            "camera_adjacency keys must exactly match canonical camera names: "
            f"missing={sorted(names - set(adjacency))}, extra={sorted(set(adjacency) - names)}"
        )
    matrix = [[False] * len(CAMERA_ORDER) for _ in CAMERA_ORDER]
    for target_name, source_names in adjacency.items():
        target = CAMERA_ORDER.index(target_name)
        if len(set(source_names)) != len(source_names):
            raise ValueError(f"Duplicate adjacency source for {target_name}")
        for source_name in source_names:
            if source_name not in names:
                raise ValueError(f"Unknown camera name in adjacency: {source_name}")
            source = CAMERA_ORDER.index(source_name)
            if source == target:
                raise ValueError("Cross-view adjacency cannot contain self edges")
            matrix[target][source] = True
    if undirected:
        for left in range(len(CAMERA_ORDER)):
            for right in range(len(CAMERA_ORDER)):
                if matrix[left][right] != matrix[right][left]:
                    raise ValueError(
                        "undirected=true requires explicit symmetric edges; "
                        "reverse edges are never added implicitly"
                    )
    return tuple(tuple(row) for row in matrix)


class EdgeBatchedWarmAttention(Attention):
    """In-place upgrade of official Attention; retains all official parameter names."""

    def enable_cross_view_audit(self, enabled: bool = True) -> None:
        """Retain scalar norms and routed edge IDs, never full activations."""

        self._mv_audit_enabled = bool(enabled)
        self._mv_last_audit = None

    def set_cross_view_context(
        self,
        *,
        batch: int,
        views: int,
        view_ids: torch.Tensor,
        view_valid: torch.Tensor,
        camera_embedding: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> None:
        self._mv_batch = batch
        self._mv_views = views
        self._mv_view_ids = view_ids
        self._mv_view_valid = view_valid
        self._mv_camera_embedding = camera_embedding
        self._mv_adjacency = adjacency

    def cross_view_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        batch = self._mv_batch
        views = self._mv_views
        tokens, heads, head_dim = q.shape[1:]
        q_bv = q.reshape(batch, views, tokens, heads, head_dim)
        k_bv = k.reshape(batch, views, tokens, heads, head_dim)
        v_bv = v.reshape(batch, views, tokens, heads, head_dim)
        cross = torch.zeros(
            (batch, views, tokens, self.query_dim), device=q.device, dtype=q.dtype
        )
        accessed_edges: list[tuple[int, int, int]] = []
        for sample in range(batch):
            ids = self._mv_view_ids[sample]
            valid = self._mv_view_valid[sample]
            for target_position in range(views):
                if not bool(valid[target_position].item()):
                    continue
                target_id = ids[target_position]
                allowed = self._mv_adjacency[target_id, ids] & valid
                allowed[target_position] = False
                source_positions = torch.nonzero(allowed, as_tuple=False).flatten()
                if source_positions.numel() == 0:
                    continue
                if getattr(self, "_mv_audit_enabled", False):
                    target_camera = int(target_id.item())
                    accessed_edges.extend(
                        (sample, int(ids[source].item()), target_camera)
                        for source in source_positions.tolist()
                    )
                source_k = k_bv[sample, source_positions].reshape(
                    1, source_positions.numel() * tokens, heads, head_dim
                )
                source_v = v_bv[sample, source_positions].reshape(
                    1, source_positions.numel() * tokens, heads, head_dim
                )
                cross[sample, target_position] = self.compute_attention(
                    q_bv[sample, target_position].unsqueeze(0), source_k, source_v
                ).squeeze(0)
        if getattr(self, "_mv_audit_enabled", False):
            self._mv_last_accessed_edges = tuple(accessed_edges)
        return cross.reshape(batch * views, tokens, self.query_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope_emb: torch.Tensor | None = None,
        video_size=None,
    ) -> torch.Tensor:
        if context is not None:
            return super().forward(x, context, rope_emb=rope_emb, video_size=video_size)
        q, k, v = self.compute_qkv(x, None, rope_emb=rope_emb)
        self_output = self.compute_attention(q, k, v, video_size=video_size)
        views = getattr(self, "_mv_views", None)
        if views is None:
            raise RuntimeError("B-warm attention context was not configured")
        if views == 1:
            if getattr(self, "_mv_audit_enabled", False):
                self._mv_last_audit = {
                    "self_out_norm": float(self_output.detach().float().norm().item()),
                    "cross_out_norm": 0.0,
                    "gated_residual_norm": 0.0,
                    "gate": float(self.cross_view_gate.detach().float().item()),
                    "accessed_edges": [],
                }
            return self_output

        camera_tokens = self._mv_camera_embedding.unsqueeze(1).expand(-1, x.shape[1], -1)
        q_camera, k_camera, _ = self.compute_qkv(
            x + camera_tokens, None, rope_emb=rope_emb
        )
        cross_output = self.cross_view_attention(q_camera, k_camera, v)
        gated_residual = self.cross_view_gate * cross_output
        if getattr(self, "_mv_audit_enabled", False):
            self._mv_last_audit = {
                "self_out_norm": float(self_output.detach().float().norm().item()),
                "cross_out_norm": float(cross_output.detach().float().norm().item()),
                "gated_residual_norm": float(gated_residual.detach().float().norm().item()),
                "gate": float(self.cross_view_gate.detach().float().item()),
                "accessed_edges": [list(edge) for edge in self._mv_last_accessed_edges],
            }
        return self_output + gated_residual


class WarmMultiViewHarmonizerDiT(MiniTrainDIT):
    """Official self-view path with an attention-internal zero-gated cross residual."""

    @classmethod
    def upgrade_from_official(
        cls,
        dit: MiniTrainDIT,
        *,
        camera_adjacency: Mapping[str, Sequence[str]],
        camera_schema_version: str = "canonical_v1",
        adjacency_undirected: bool = False,
        max_views: int = 6,
    ) -> "WarmMultiViewHarmonizerDiT":
        if isinstance(dit, cls):
            raise ValueError("DiT has already been upgraded to B-warm")
        if not isinstance(dit, MiniTrainDIT):
            raise TypeError(f"Expected the official MiniTrainDIT, got {type(dit)!r}")
        if dit.patch_temporal != 1:
            raise ValueError("B-warm requires official patch_temporal=1")
        if max_views != len(CAMERA_ORDER):
            raise ValueError("B-warm phase one requires the complete canonical camera schema")
        matrix = canonical_adjacency_matrix(
            camera_adjacency, undirected=adjacency_undirected
        )
        device, dtype = module_device_dtype(dit)
        camera_embedding = nn.Embedding(max_views, dit.model_channels).to(device=device, dtype=dtype)
        nn.init.zeros_(camera_embedding.weight)

        gate_names = []
        for index, block in enumerate(dit.blocks):
            attention = _attention_module(block)
            attention.__class__ = EdgeBatchedWarmAttention
            attention.register_parameter(
                "cross_view_gate", nn.Parameter(torch.zeros((), device=device, dtype=dtype))
            )
            gate_names.append(f"blocks.{index}.self_attn.cross_view_gate")

        dit.__class__ = cls
        dit.add_module("camera_embedding", camera_embedding)
        dit.mv_max_views = max_views
        dit.mv_architecture = "B"
        dit.mv_attention_variant = "warm_v1"
        dit.mv_camera_schema_version = str(camera_schema_version)
        dit.mv_camera_adjacency = matrix
        dit.mv_adjacency_undirected = bool(adjacency_undirected)
        dit.mv_gate_parameter_names = tuple(gate_names)
        return dit

    @staticmethod
    def _expand_timesteps(
        timesteps: torch.Tensor, batch: int, views: int
    ) -> torch.Tensor:
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        if timesteps.shape == (batch, 1):
            return timesteps.unsqueeze(1).expand(-1, views, -1).reshape(batch * views, 1)
        if timesteps.shape == (batch, views):
            return timesteps.reshape(batch * views, 1)
        raise ValueError("B-warm timesteps must be [B], [B,1], or [B,V]")

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        data_type: DataType | None = DataType.IMAGE,
        use_cuda_graphs: bool = False,
        view_ids_B_T: torch.Tensor | None = None,
        view_valid_B_T: torch.Tensor | None = None,
        **unused_condition_fields,
    ) -> torch.Tensor:
        del unused_condition_fields
        if use_cuda_graphs:
            raise NotImplementedError("CUDA graphs are not enabled for B-warm")
        if not isinstance(data_type, DataType):
            raise TypeError(f"Expected DataType, got {type(data_type)!r}")
        view_ids_B_T, view_valid_B_T = validate_view_metadata(
            x_B_C_T_H_W, view_ids_B_T, view_valid_B_T, self.mv_max_views
        )
        if not bool(view_valid_B_T.all().item()):
            raise ValueError("B-warm rejects padded or fabricated missing views")
        batch, channels, views, height, width = x_B_C_T_H_W.shape
        flat = x_B_C_T_H_W.permute(0, 2, 1, 3, 4).reshape(
            batch * views, channels, 1, height, width
        )
        padding_flat = None
        if padding_mask is not None:
            if padding_mask.shape[0] != batch:
                raise ValueError("padding_mask batch does not match DiT input")
            padding_flat = padding_mask.unsqueeze(1).expand(
                -1, views, -1, -1, -1
            ).reshape(batch * views, *padding_mask.shape[1:])
        elif self.concat_padding_mask:
            raise ValueError("Official DiT expects a padding_mask")

        fps_flat = None
        if fps is not None:
            if fps.shape[0] == batch:
                fps_flat = fps.unsqueeze(1).expand(
                    (batch, views) + tuple(fps.shape[1:])
                ).reshape((batch * views,) + tuple(fps.shape[1:]))
            elif fps.shape[0] == batch * views:
                fps_flat = fps
            else:
                raise ValueError("fps batch must be B or B*V")

        tokens, rope, extra_pos = super().prepare_embedded_sequence(
            flat, fps=fps_flat, padding_mask=padding_flat
        )
        if self.crossattn_proj is not None:
            crossattn_emb = self.crossattn_proj(crossattn_emb)
        if crossattn_emb.shape[0] == batch:
            crossattn_flat = crossattn_emb.unsqueeze(1).expand(
                -1, views, -1, -1
            ).reshape(batch * views, *crossattn_emb.shape[1:])
        elif crossattn_emb.shape[0] == batch * views:
            crossattn_flat = crossattn_emb
        else:
            raise ValueError("B-warm crossattn_emb batch must be B or B*V")

        timesteps_flat = self._expand_timesteps(timesteps_B_T, batch, views)
        embedding, adaln = self.t_embedder(timesteps_flat)
        embedding = self.t_embedding_norm(embedding)
        self.affline_scale_log_info = {"t_embedding_B_T_D": embedding.detach()}
        self.affline_emb = embedding
        self.crossattn_emb = crossattn_flat
        camera_embedding = self.camera_embedding(view_ids_B_T).reshape(
            batch * views, self.model_channels
        )
        adjacency = torch.tensor(
            self.mv_camera_adjacency, device=tokens.device, dtype=torch.bool
        )
        block_kwargs = {
            "rope_emb_L_1_1_D": rope,
            "adaln_lora_B_T_3D": adaln,
            "extra_per_block_pos_emb": extra_pos,
        }
        for block in self.blocks:
            attention = _attention_module(block)
            attention.set_cross_view_context(
                batch=batch,
                views=views,
                view_ids=view_ids_B_T,
                view_valid=view_valid_B_T,
                camera_embedding=camera_embedding,
                adjacency=adjacency,
            )
            tokens = block(tokens, embedding, crossattn_flat, **block_kwargs)

        output_flat = self.final_layer(tokens, embedding, adaln_lora_B_T_3D=adaln)
        output = self.unpatchify(output_flat)
        if output.shape[2] != 1:
            raise RuntimeError("B-warm self-view path must retain one latent frame")
        return output.squeeze(2).reshape(
            batch, views, output.shape[1], *output.shape[-2:]
        ).permute(0, 2, 1, 3, 4).contiguous()


def warm_gate_parameters(dit: nn.Module) -> list[nn.Parameter]:
    if getattr(dit, "mv_attention_variant", None) != "warm_v1":
        raise ValueError("Expected a B-warm DiT")
    return [_attention_module(block).cross_view_gate for block in dit.blocks]
