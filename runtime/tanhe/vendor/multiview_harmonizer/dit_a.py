"""Scheme A: official per-view DiT blocks plus gated cross-view adapters."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.models.text2image_dit import MiniTrainDIT

from .adapters import PooledCrossViewAdapter
from .dit_utils import module_device_dtype, validate_view_metadata


class FactorizedMultiViewHarmonizerDiT(MiniTrainDIT):
    """In-place upgrade of an already-loaded official Harmonizer DiT.

    Do not instantiate this class through the Cosmos lazy config.  Build and
    load the unmodified 0.6B Text2Image DiT first, then call
    ``upgrade_from_official``.  This avoids meta-device missing parameters and
    guarantees that the official block parameters are loaded before adapters
    exist.
    """

    @classmethod
    def upgrade_from_official(
        cls,
        dit: MiniTrainDIT,
        *,
        adapter_after_blocks: Sequence[int] = (4, 9, 14, 19),
        adapter_dim: int = 320,
        adapter_heads: int = 8,
        pool_factor: int = 4,
        max_views: int = 6,
        neighbor_only: bool = True,
    ) -> "FactorizedMultiViewHarmonizerDiT":
        if isinstance(dit, cls):
            raise ValueError("DiT has already been upgraded to scheme A")
        if not isinstance(dit, MiniTrainDIT):
            raise TypeError(f"Expected the official MiniTrainDIT, got {type(dit)!r}")
        indices = tuple(sorted(set(int(index) for index in adapter_after_blocks)))
        if any(index < 0 or index >= len(dit.blocks) for index in indices):
            raise ValueError(f"Adapter indices {indices} are invalid for {len(dit.blocks)} blocks")
        if dit.patch_temporal != 1:
            raise ValueError("Scheme A requires the official patch_temporal=1")

        device, dtype = module_device_dtype(dit)
        adapters = nn.ModuleDict(
            {
                f"block_{index:02d}": PooledCrossViewAdapter(
                    model_dim=dit.model_channels,
                    adapter_dim=adapter_dim,
                    num_heads=adapter_heads,
                    pool_factor=pool_factor,
                    max_views=max_views,
                    neighbor_only=neighbor_only,
                )
                for index in indices
            }
        ).to(device=device, dtype=dtype)

        dit.__class__ = cls
        dit.add_module("mv_adapters", adapters)
        dit.mv_adapter_after_blocks = indices
        dit.mv_max_views = max_views
        dit.mv_architecture = "A" if indices else "independent"
        return dit

    def _expand_timesteps_for_views(
        self, timesteps_B_T: torch.Tensor, batch: int, views: int
    ) -> torch.Tensor:
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        if timesteps_B_T.shape == (batch, 1):
            return timesteps_B_T.unsqueeze(1).expand(-1, views, -1).reshape(batch * views, 1)
        if timesteps_B_T.shape == (batch, views):
            return timesteps_B_T.reshape(batch * views, 1)
        raise ValueError(
            f"timesteps must be [B], [B,1], or [B,V]; got {tuple(timesteps_B_T.shape)}"
        )

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
            raise NotImplementedError("CUDA graphs are not enabled for scheme A")
        if not isinstance(data_type, DataType):
            raise TypeError(f"Expected DataType, got {type(data_type)!r}")
        view_ids_B_T, view_valid_B_T = validate_view_metadata(
            x_B_C_T_H_W, view_ids_B_T, view_valid_B_T, self.mv_max_views
        )
        batch, channels, views, height, width = x_B_C_T_H_W.shape
        x_BV_C_1_H_W = x_B_C_T_H_W.permute(0, 2, 1, 3, 4).reshape(
            batch * views, channels, 1, height, width
        )

        padding_BV_1_H_W = None
        if padding_mask is not None:
            if padding_mask.shape[0] != batch:
                raise ValueError("padding_mask batch does not match DiT input")
            padding_BV_1_H_W = padding_mask.unsqueeze(1).expand(
                -1, views, -1, -1, -1
            ).reshape(batch * views, *padding_mask.shape[1:])
        elif self.concat_padding_mask:
            raise ValueError("Official DiT expects a padding_mask")

        fps_BV = None
        if fps is not None:
            if fps.shape[0] == batch:
                fps_BV = fps.unsqueeze(1).expand(
                    (batch, views) + tuple(fps.shape[1:])
                ).reshape((batch * views,) + tuple(fps.shape[1:]))
            elif fps.shape[0] == batch * views:
                fps_BV = fps
            else:
                raise ValueError("fps batch must be B or B*V")

        x_BV_1_H_W_D, rope, extra_pos = super().prepare_embedded_sequence(
            x_BV_C_1_H_W,
            fps=fps_BV,
            padding_mask=padding_BV_1_H_W,
        )
        if self.crossattn_proj is not None:
            crossattn_emb = self.crossattn_proj(crossattn_emb)
        if crossattn_emb.shape[0] == batch:
            crossattn_BV_N_D = crossattn_emb.unsqueeze(1).expand(
                -1, views, -1, -1
            ).reshape(batch * views, *crossattn_emb.shape[1:])
        elif crossattn_emb.shape[0] == batch * views:
            crossattn_BV_N_D = crossattn_emb
        else:
            raise ValueError("crossattn_emb batch must be B or B*V")

        timesteps_BV_1 = self._expand_timesteps_for_views(timesteps_B_T, batch, views)
        t_embedding_BV_1_D, adaln_lora_BV_1_3D = self.t_embedder(timesteps_BV_1)
        t_embedding_BV_1_D = self.t_embedding_norm(t_embedding_BV_1_D)
        self.affline_scale_log_info = {"t_embedding_B_T_D": t_embedding_BV_1_D.detach()}
        self.affline_emb = t_embedding_BV_1_D
        self.crossattn_emb = crossattn_BV_N_D

        block_kwargs = {
            "rope_emb_L_1_1_D": rope,
            "adaln_lora_B_T_3D": adaln_lora_BV_1_3D,
            "extra_per_block_pos_emb": extra_pos,
        }
        for index, block in enumerate(self.blocks):
            x_BV_1_H_W_D = block(
                x_BV_1_H_W_D,
                t_embedding_BV_1_D,
                crossattn_BV_N_D,
                **block_kwargs,
            )
            key = f"block_{index:02d}"
            if key in self.mv_adapters:
                x_B_V_1_H_W_D = x_BV_1_H_W_D.reshape(
                    batch, views, *x_BV_1_H_W_D.shape[1:]
                )
                x_B_V_1_H_W_D = self.mv_adapters[key](
                    x_B_V_1_H_W_D,
                    view_ids_B_T,
                    view_valid_B_T,
                )
                x_BV_1_H_W_D = x_B_V_1_H_W_D.reshape(
                    batch * views, *x_B_V_1_H_W_D.shape[2:]
                )

        x_BV_1_H_W_O = self.final_layer(
            x_BV_1_H_W_D,
            t_embedding_BV_1_D,
            adaln_lora_B_T_3D=adaln_lora_BV_1_3D,
        )
        output_BV_C_1_H_W = self.unpatchify(x_BV_1_H_W_O)
        if output_BV_C_1_H_W.shape[2] != 1:
            raise RuntimeError("Scheme A expects one latent frame per independently processed view")
        return output_BV_C_1_H_W.squeeze(2).reshape(
            batch, views, output_BV_C_1_H_W.shape[1], *output_BV_C_1_H_W.shape[-2:]
        ).permute(0, 2, 1, 3, 4).contiguous()
