"""Scheme B: official DiT blocks operating on a joint multi-camera token set."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.models.text2image_dit import MiniTrainDIT

from .dit_utils import module_device_dtype, repeat_single_frame_rope, validate_view_metadata


class JointTokenMultiViewHarmonizerDiT(MiniTrainDIT):
    """In-place joint-token upgrade of an already-loaded official DiT."""

    @classmethod
    def upgrade_from_official(
        cls,
        dit: MiniTrainDIT,
        *,
        max_views: int = 6,
    ) -> "JointTokenMultiViewHarmonizerDiT":
        if isinstance(dit, cls):
            raise ValueError("DiT has already been upgraded to scheme B")
        if not isinstance(dit, MiniTrainDIT):
            raise TypeError(f"Expected the official MiniTrainDIT, got {type(dit)!r}")
        if dit.patch_temporal != 1:
            raise ValueError("Scheme B requires the official patch_temporal=1")
        if "rope" not in dit.pos_emb_cls.lower():
            raise ValueError("Scheme B currently requires the official RoPE position encoding")
        if dit.extra_per_block_abs_pos_emb:
            raise ValueError("Scheme B does not support extra per-block absolute position embeddings")

        device, dtype = module_device_dtype(dit)
        camera_embedding = nn.Embedding(max_views, dit.model_channels).to(
            device=device, dtype=dtype
        )
        nn.init.zeros_(camera_embedding.weight)
        dit.__class__ = cls
        dit.add_module("camera_embedding", camera_embedding)
        dit.mv_max_views = max_views
        dit.mv_architecture = "B"
        return dit

    def _prepare_joint_sequence(
        self,
        x_B_C_V_H_W: torch.Tensor,
        view_ids_B_V: torch.Tensor,
        fps: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, views, height, width = x_B_C_V_H_W.shape
        if self.concat_padding_mask:
            if padding_mask is None:
                raise ValueError("Official DiT expects a padding_mask")
            if padding_mask.shape[0] != batch:
                raise ValueError("padding_mask batch does not match DiT input")
            resized = F.interpolate(padding_mask, size=(height, width), mode="nearest")
            x_B_C_V_H_W = torch.cat(
                [x_B_C_V_H_W, resized.unsqueeze(2).expand(-1, -1, views, -1, -1)],
                dim=1,
            )

        x_B_V_H_W_D = self.x_embedder(x_B_C_V_H_W)
        x_B_V_H_W_D = x_B_V_H_W_D + self.camera_embedding(view_ids_B_V).unsqueeze(2).unsqueeze(3)

        # Build RoPE for one image and repeat it per camera.  Camera index is
        # represented only by camera_embedding, never by temporal distance.
        single_view = x_B_V_H_W_D[:, :1]
        rope_single = self.pos_embedder(single_view, fps=fps)
        rope_joint = repeat_single_frame_rope(rope_single, views)
        return x_B_V_H_W_D, rope_joint

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
            raise NotImplementedError("CUDA graphs are not enabled for scheme B")
        if not isinstance(data_type, DataType):
            raise TypeError(f"Expected DataType, got {type(data_type)!r}")
        view_ids_B_T, view_valid_B_T = validate_view_metadata(
            x_B_C_T_H_W, view_ids_B_T, view_valid_B_T, self.mv_max_views
        )
        if not bool(view_valid_B_T.all().item()):
            raise ValueError(
                "Scheme B dense framework requires complete fixed-V groups; "
                "do not pad missing cameras with zero images"
            )
        batch, _, views, _, _ = x_B_C_T_H_W.shape
        x_B_V_H_W_D, rope_joint = self._prepare_joint_sequence(
            x_B_C_T_H_W,
            view_ids_B_T,
            fps,
            padding_mask,
        )

        if self.crossattn_proj is not None:
            crossattn_emb = self.crossattn_proj(crossattn_emb)
        if crossattn_emb.shape[0] != batch:
            raise ValueError("Scheme B crossattn_emb batch must equal B")
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        if timesteps_B_T.shape[0] != batch or timesteps_B_T.shape[1] not in (1, views):
            raise ValueError("Scheme B timesteps must be [B], [B,1], or [B,V]")
        if timesteps_B_T.shape[1] == views:
            reference = timesteps_B_T[:, :1]
            if not torch.equal(timesteps_B_T, reference.expand_as(timesteps_B_T)):
                raise ValueError("All camera views must use the same diffusion timestep")

        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)
        self.affline_scale_log_info = {"t_embedding_B_T_D": t_embedding_B_T_D.detach()}
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_joint,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": None,
        }
        for block in self.blocks:
            x_B_V_H_W_D = block(
                x_B_V_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                **block_kwargs,
            )
        x_B_V_H_W_O = self.final_layer(
            x_B_V_H_W_D,
            t_embedding_B_T_D,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
        )
        return self.unpatchify(x_B_V_H_W_O)

