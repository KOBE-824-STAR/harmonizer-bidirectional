"""Scheme A-TMV: official temporal DiT plus same-time cross-view adapters."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.models.text2image_dit import MiniTrainDIT

from .adapters_temporal import SameTimePooledCrossViewAdapter
from .dit_utils import module_device_dtype
from .layout_temporal import CURRENT_OUTPUT_INDEX, OFFICIAL_TEMPORAL_OFFSETS
from .plucker import PLUCKER_CONVENTION, token_center_plucker_rays


A1_VARIANT = "a1_plucker_v1"
A1_POSE_VARIANT = "a1_pose_routing_v1"


def attach_a1_plucker_conditioning(
    dit: nn.Module, *, hidden_dim: int = 64, gate_init: float = 0.0
) -> nn.Module:
    """Upgrade an already constructed A-TMV DiT to the isolated A1 variant."""

    if getattr(dit, "mv_architecture", None) != "A-TMV":
        raise ValueError("A1 must attach after the A-TMV upgrade")
    if bool(getattr(dit, "mv_plucker_conditioning_attached", False)):
        raise ValueError("A1 Plucker conditioning is already attached")
    if float(gate_init) != 0.0:
        raise ValueError("A1 warm-start preservation requires gate_init=0")
    adapters = getattr(dit, "mv_adapters", None)
    if not isinstance(adapters, nn.ModuleDict) or not adapters:
        raise ValueError("A1 requires registered A-TMV adapters")
    for adapter in adapters.values():
        adapter.attach_plucker_conditioning(hidden_dim)
    dit.mv_plucker_conditioning_attached = True
    dit.mv_plucker_conditioning_enabled = True
    dit.mv_plucker_variant = (A1_POSE_VARIANT if
                              getattr(dit, "mv_pose_routing_config", None) is not None
                              else A1_VARIANT)
    dit.mv_plucker_dim = 6
    dit.mv_plucker_hidden_dim = int(hidden_dim)
    dit.mv_plucker_gate_initialization = 0.0
    dit.mv_plucker_convention = PLUCKER_CONVENTION
    dit.mv_geometry_extrinsic_convention = "T_ego_from_cam"
    return dit


class TemporalFactorizedMultiViewHarmonizerDiT(MiniTrainDIT):
    """In-place temporal-preserving upgrade of an official Harmonizer DiT.

    The DiT batch is ``B*V`` and its temporal axis remains the real tokenizer
    latent time.  At adapter boundaries only, tokens are restored to
    ``[B,V,T,H,W,D]`` and cross-view attention is applied independently for
    each temporal slot.
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
    ) -> "TemporalFactorizedMultiViewHarmonizerDiT":
        if isinstance(dit, cls) or hasattr(dit, "mv_architecture"):
            raise ValueError("DiT has already been upgraded to a multi-view architecture")
        if not isinstance(dit, MiniTrainDIT):
            raise TypeError(f"Expected official MiniTrainDIT, got {type(dit)!r}")
        if dit.patch_temporal != 1:
            raise ValueError("Scheme A-TMV requires official patch_temporal=1")
        indices = tuple(sorted(set(int(index) for index in adapter_after_blocks)))
        if not indices:
            raise ValueError("Scheme A-TMV requires at least one adapter")
        if any(index < 0 or index >= len(dit.blocks) for index in indices):
            raise ValueError(f"Adapter indices {indices} invalid for {len(dit.blocks)} blocks")

        device, dtype = module_device_dtype(dit)
        adapters = nn.ModuleDict(
            {
                f"block_{index:02d}": SameTimePooledCrossViewAdapter(
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

        # Preserve the already-loaded object and every official Parameter.
        dit.__class__ = cls
        dit.add_module("mv_adapters", adapters)
        dit.mv_adapter_after_blocks = indices
        dit.mv_max_views = int(max_views)
        dit.mv_architecture = "A-TMV"
        dit.mv_temporal_offsets = OFFICIAL_TEMPORAL_OFFSETS
        dit.mv_current_output_index = CURRENT_OUTPUT_INDEX
        dit.mv_temporal_adapter_mode = "same_latent_time_only"
        dit.mv_gradient_checkpointing = False
        return dit

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Checkpoint the live DiT blocks and adapters without changing their math."""

        self.mv_gradient_checkpointing = bool(enabled)

    def _validate_multiview_metadata(
        self,
        flat_batch: int,
        view_ids_B_V: torch.Tensor | None,
        view_valid_B_V: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[int, int, torch.Tensor, torch.Tensor]:
        if view_ids_B_V is None or view_valid_B_V is None:
            raise ValueError("Scheme A-TMV requires explicit view_ids_B_V and view_valid_B_V")
        if view_ids_B_V.ndim != 2:
            raise ValueError("view_ids_B_V must be [B,V]")
        batch_size, num_views = view_ids_B_V.shape
        if flat_batch != batch_size * num_views:
            raise ValueError(f"DiT flat batch {flat_batch} != B*V {batch_size * num_views}")
        if not 1 <= num_views <= self.mv_max_views:
            raise ValueError(f"num_views must be in [1,{self.mv_max_views}]")
        if view_valid_B_V.shape != (batch_size, num_views):
            raise ValueError("view_valid_B_V must match view_ids_B_V")
        if view_ids_B_V.dtype != torch.long or view_valid_B_V.dtype != torch.bool:
            raise TypeError("view IDs must be long and view validity must be bool")
        if view_ids_B_V.device != device or view_valid_B_V.device != device:
            raise ValueError("view metadata and DiT input must be on the same device")
        if view_ids_B_V.min() < 0 or view_ids_B_V.max() >= self.mv_max_views:
            raise ValueError("view IDs outside configured camera range")
        for sample_index in range(batch_size):
            valid_ids = view_ids_B_V[sample_index, view_valid_B_V[sample_index]]
            if valid_ids.unique().numel() != valid_ids.numel():
                raise ValueError(f"valid view IDs duplicate in sample {sample_index}")
        return batch_size, num_views, view_ids_B_V, view_valid_B_V

    @staticmethod
    def _expand_first_dim_for_views(
        value: torch.Tensor | None,
        *,
        name: str,
        batch_size: int,
        num_views: int,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if value.shape[0] == batch_size * num_views:
            return value
        if value.shape[0] != batch_size:
            raise ValueError(f"{name} leading dimension must be B or B*V")
        return value.unsqueeze(1).expand(
            (batch_size, num_views) + tuple(value.shape[1:])
        ).reshape((batch_size * num_views,) + tuple(value.shape[1:]))

    @staticmethod
    def _expand_timesteps_for_views(
        timesteps: torch.Tensor,
        *,
        batch_size: int,
        num_views: int,
    ) -> torch.Tensor:
        """Repeat only the leading sample axis, preserving timestep axes.

        In the live Cosmos API the dimensions after the leading batch axis
        describe a timestep sequence.  In particular, a tensor shaped
        ``[B,K]`` must become ``[B*V,K]``; ``K`` must never be interpreted as
        the number of cameras merely because it happens to equal ``V``.
        """

        flat_batch = batch_size * num_views
        if timesteps.ndim < 1:
            raise ValueError("timesteps must have a leading batch dimension")
        if timesteps.shape[0] == flat_batch:
            return timesteps
        if timesteps.shape[0] != batch_size:
            raise ValueError(
                "timesteps leading dimension must be B or B*V; "
                f"got {tuple(timesteps.shape)} for B={batch_size}, V={num_views}"
            )
        return timesteps.unsqueeze(1).expand(
            (batch_size, num_views) + tuple(timesteps.shape[1:])
        ).reshape((flat_batch,) + tuple(timesteps.shape[1:]))

    def _expanded_conditions(
        self,
        *,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
        batch_size: int,
        num_views: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        timesteps_BV_1 = self._expand_timesteps_for_views(
            timesteps_B_T, batch_size=batch_size, num_views=num_views
        )
        crossattn_BV_N_D = self._expand_first_dim_for_views(
            crossattn_emb,
            name="crossattn_emb",
            batch_size=batch_size,
            num_views=num_views,
        )
        fps_BV = self._expand_first_dim_for_views(
            fps, name="fps", batch_size=batch_size, num_views=num_views
        )
        padding_BV = self._expand_first_dim_for_views(
            padding_mask,
            name="padding_mask",
            batch_size=batch_size,
            num_views=num_views,
        )
        assert crossattn_BV_N_D is not None
        return timesteps_BV_1, crossattn_BV_N_D, fps_BV, padding_BV

    def forward_official_temporal_reference(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        data_type: DataType | None = DataType.IMAGE,
        use_cuda_graphs: bool = False,
        *,
        view_ids_B_V: torch.Tensor | None = None,
        view_valid_B_V: torch.Tensor | None = None,
        **unused_condition_fields,
    ) -> torch.Tensor:
        """Run the inherited official forward on the same ``B*V`` sequences."""

        if unused_condition_fields:
            raise TypeError(
                "Unsupported condition fields for the audited A-TMV runtime: "
                f"{sorted(unused_condition_fields)}"
            )
        if use_cuda_graphs:
            raise NotImplementedError("CUDA graphs are disabled for Scheme A-TMV references")
        batch_size, num_views, _, _ = self._validate_multiview_metadata(
            x_B_C_T_H_W.shape[0], view_ids_B_V, view_valid_B_V, x_B_C_T_H_W.device
        )
        timesteps_BV_1, crossattn_BV_N_D, fps_BV, padding_BV = self._expanded_conditions(
            timesteps_B_T=timesteps_B_T,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=padding_mask,
            batch_size=batch_size,
            num_views=num_views,
        )
        return MiniTrainDIT.forward(
            self,
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps_BV_1,
            crossattn_emb=crossattn_BV_N_D,
            fps=fps_BV,
            padding_mask=padding_BV,
            data_type=data_type,
            use_cuda_graphs=False,
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
        *,
        view_ids_B_V: torch.Tensor | None = None,
        view_valid_B_V: torch.Tensor | None = None,
        intrinsics_B_V_3_3: torch.Tensor | None = None,
        T_ego_from_cam_B_V_4_4: torch.Tensor | None = None,
        image_hw: tuple[int, int] | None = None,
        **unused_condition_fields,
    ) -> torch.Tensor:
        if unused_condition_fields:
            raise TypeError(
                "Unsupported condition fields for the audited A-TMV runtime: "
                f"{sorted(unused_condition_fields)}"
            )
        if use_cuda_graphs:
            raise NotImplementedError("CUDA graphs are not enabled for Scheme A-TMV")
        if not isinstance(data_type, DataType):
            raise TypeError(f"Expected DataType, got {type(data_type)!r}")
        if x_B_C_T_H_W.ndim != 5 or x_B_C_T_H_W.shape[2] < 1:
            raise ValueError("DiT input must be [B*V,C,T,H,W] with T>=1")
        batch_size, num_views, view_ids_B_V, view_valid_B_V = (
            self._validate_multiview_metadata(
                x_B_C_T_H_W.shape[0],
                view_ids_B_V,
                view_valid_B_V,
                x_B_C_T_H_W.device,
            )
        )
        timesteps_BV_1, crossattn_BV_N_D, fps_BV, padding_BV = self._expanded_conditions(
            timesteps_B_T=timesteps_B_T,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=padding_mask,
            batch_size=batch_size,
            num_views=num_views,
        )
        # Preserve the official forward's scalar-timestep normalization after
        # repeating the leading sample axis across cameras.
        if timesteps_BV_1.ndim == 1:
            timesteps_BV_1 = timesteps_BV_1.unsqueeze(1)

        x_BV_T_H_W_D, rope, extra_pos = super().prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps_BV,
            padding_mask=padding_BV,
        )
        if self.crossattn_proj is not None:
            crossattn_BV_N_D = self.crossattn_proj(crossattn_BV_N_D)
        t_embedding_BV_1_D, adaln_lora_BV_1_3D = self.t_embedder(timesteps_BV_1)
        t_embedding_BV_1_D = self.t_embedding_norm(t_embedding_BV_1_D)
        self.affline_scale_log_info = {"t_embedding_B_T_D": t_embedding_BV_1_D.detach()}
        self.affline_emb = t_embedding_BV_1_D
        self.crossattn_emb = crossattn_BV_N_D

        plucker_B_V_H_W_6 = None
        pose_adjacency_B_V_V = None
        pose_routing = getattr(self, "mv_pose_routing_config", None)
        if pose_routing is not None:
            if intrinsics_B_V_3_3 is None or T_ego_from_cam_B_V_4_4 is None or image_hw is None:
                raise ValueError("A pose routing requires K, T_ego_from_cam, and image_hw")
            from .b_tmv_routing import build_pose_overlap_adjacency

            pose_adjacency_B_V_V, overlap = build_pose_overlap_adjacency(
                intrinsics_B_V_3_3, T_ego_from_cam_B_V_4_4,
                image_hw=image_hw, view_valid=view_valid_B_V, config=pose_routing,
            )
            self.mv_last_pose_routing = {
                "adjacency": pose_adjacency_B_V_V.detach(),
                "overlap_scores": overlap.detach(),
            }
        if bool(getattr(self, "mv_plucker_conditioning_enabled", False)):
            if intrinsics_B_V_3_3 is None or T_ego_from_cam_B_V_4_4 is None:
                raise ValueError("A1 requires intrinsics and T_ego_from_cam")
            if image_hw is None:
                raise ValueError("A1 requires source image_hw")
            if intrinsics_B_V_3_3.device != x_BV_T_H_W_D.device:
                raise ValueError("A1 calibration must share the token device")
            if tuple(intrinsics_B_V_3_3.shape[:2]) != (batch_size, num_views):
                raise ValueError("A1 calibration does not match [B,V]")
            plucker_B_V_H_W_6 = token_center_plucker_rays(
                intrinsics_B_V_3_3,
                T_ego_from_cam_B_V_4_4,
                image_hw=image_hw,
                feature_hw=(x_BV_T_H_W_D.shape[2], x_BV_T_H_W_D.shape[3]),
            )

        if extra_pos is not None and x_BV_T_H_W_D.shape != extra_pos.shape:
            raise RuntimeError("official extra positional embedding shape mismatch")
        block_kwargs = {
            "rope_emb_L_1_1_D": rope,
            "adaln_lora_B_T_3D": adaln_lora_BV_1_3D,
            "extra_per_block_pos_emb": extra_pos,
        }
        for index, block in enumerate(self.blocks):
            def run_block(value, active_block=block):
                return active_block(
                    value,
                    t_embedding_BV_1_D,
                    crossattn_BV_N_D,
                    **block_kwargs,
                )

            if (
                self.mv_gradient_checkpointing
                and self.mv_adapters.training
                and torch.is_grad_enabled()
            ):
                x_BV_T_H_W_D = checkpoint(
                    run_block, x_BV_T_H_W_D, use_reentrant=False
                )
            else:
                x_BV_T_H_W_D = run_block(x_BV_T_H_W_D)
            key = f"block_{index:02d}"
            if key in self.mv_adapters:
                num_token_frames = x_BV_T_H_W_D.shape[1]
                x_B_V_T_H_W_D = x_BV_T_H_W_D.reshape(
                    batch_size, num_views, *x_BV_T_H_W_D.shape[1:]
                )
                adapter = self.mv_adapters[key]
                if (
                    self.mv_gradient_checkpointing
                    and self.mv_adapters.training
                    and torch.is_grad_enabled()
                ):
                    x_B_V_T_H_W_D = checkpoint(
                        adapter, x_B_V_T_H_W_D, view_ids_B_V, view_valid_B_V,
                        plucker_B_V_H_W_6, pose_adjacency_B_V_V,
                        use_reentrant=False,
                    )
                else:
                    x_B_V_T_H_W_D = adapter(
                        x_B_V_T_H_W_D,
                        view_ids_B_V,
                        view_valid_B_V,
                        plucker_B_V_H_W_6,
                        pose_adjacency_B_V_V,
                    )
                x_BV_T_H_W_D = x_B_V_T_H_W_D.reshape(
                    batch_size * num_views,
                    num_token_frames,
                    *x_B_V_T_H_W_D.shape[3:],
                )

        x_BV_T_H_W_O = self.final_layer(
            x_BV_T_H_W_D,
            t_embedding_BV_1_D,
            adaln_lora_B_T_3D=adaln_lora_BV_1_3D,
        )
        return self.unpatchify(x_BV_T_H_W_O)
