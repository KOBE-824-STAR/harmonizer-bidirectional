"""Temporal-preserving public wrapper for Scheme A-TMV."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cosmos_predict2.conditioner import DataType

from .layout_temporal import (
    CURRENT_OUTPUT_INDEX,
    TemporalMultiViewBatch,
    decode_camera_sequences,
    encode_camera_sequences,
)
from .tokenizer_utils import inner_tokenizer_module


@dataclass(frozen=True)
class TemporalForwardMetadata:
    architecture: str
    batch_size: int
    num_rgb_frames: int
    num_views: int
    input_shape: tuple[int, ...]
    vae_encoder_input_shape: tuple[int, ...]
    latent_shape: tuple[int, ...]
    dit_output_shape: tuple[int, ...]
    decoded_all_frames_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_semantics: str


class TemporalMultiViewPix2PixTurbo(nn.Module):
    """Run one complete official temporal Harmonizer sequence per camera.

    Public input is ``[B,T,V,3,H,W]``.  By default only temporal slot zero
    (the current frame in upstream's ``[current,-1,-2,-3,-4]`` order) is
    returned as ``[B,V,3,H,W]``.  ``return_all_frames=True`` returns decoded
    slots as ``[B,Tout,V,3,H,W]`` for diagnostics only.
    """

    def __init__(self, pipeline: nn.Module, *, timestep: int = 250) -> None:
        super().__init__()
        if not 0 <= timestep <= 1000:
            raise ValueError(f"timestep must be in [0,1000], got {timestep}")
        if getattr(pipeline.dit, "mv_architecture", None) != "A-TMV":
            raise ValueError("pipeline.dit must be upgraded to Scheme A-TMV")
        self.pipeline = pipeline
        self.timestep = int(timestep)
        self.architecture = "A-TMV"
        self.current_output_index = CURRENT_OUTPUT_INDEX
        self._condition_cache: dict[tuple, object] = {}
        self.freeze_official_components()

    @property
    def dit(self):
        return self.pipeline.dit

    @property
    def tokenizer(self):
        return self.pipeline.tokenizer

    def freeze_official_components(self) -> None:
        """Freeze official DiT/VAE and leave only ``mv_adapters.*`` trainable."""

        self.dit.requires_grad_(False)
        self.dit.eval()
        self.dit.mv_adapters.requires_grad_(True)
        tokenizer_module = inner_tokenizer_module(self.tokenizer)
        tokenizer_module.requires_grad_(False)
        tokenizer_module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Restore the hard mode contract after recursive Module.train().
        self.freeze_official_components()
        self.dit.mv_adapters.train(mode)
        return self

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, value in self.named_parameters() if value.requires_grad)

    def require_adapter_only_trainable(self) -> None:
        names = self.trainable_parameter_names()
        if not names:
            raise RuntimeError("Scheme A-TMV has no trainable adapter parameters")
        invalid = [name for name in names if ".mv_adapters." not in name]
        if invalid:
            raise RuntimeError(f"non-adapter parameters are trainable: {invalid[:8]}")

    def _require_source_runtime_contract(self, source: torch.Tensor) -> None:
        """Fail before tokenization when input dtype/device differ from the model."""

        tensor_kwargs = getattr(self.pipeline, "tensor_kwargs", None)
        if not isinstance(tensor_kwargs, dict):
            raise RuntimeError("pipeline.tensor_kwargs is required by the official runtime")
        expected_dtype = tensor_kwargs.get("dtype")
        if expected_dtype is not None and source.dtype != expected_dtype:
            raise TypeError(
                f"source dtype {source.dtype} does not match model dtype {expected_dtype}"
            )
        expected_device_value = tensor_kwargs.get("device")
        if expected_device_value is None:
            raise RuntimeError("pipeline.tensor_kwargs has no device")
        expected_device = torch.device(expected_device_value)
        same_device = source.device == expected_device
        # ``cuda`` means the current CUDA device in PyTorch.  Accept its
        # concrete indexed spelling (normally ``cuda:0``).
        if expected_device.type == "cuda" and expected_device.index is None:
            same_device = source.device.type == "cuda"
        if not same_device:
            raise ValueError(
                f"source device {source.device} does not match model device {expected_device}"
            )

    def _unconditional_condition(
        self,
        *,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        key = (batch_size, height, width, str(device), dtype)
        cached = self._condition_cache.get(key)
        if cached is not None:
            return cached
        data_batch = {
            "dataset_name": "image_data",
            "images": torch.zeros(
                (batch_size, 3, height, width), device=device, dtype=dtype
            ),
            "t5_text_embeddings": torch.zeros(
                (batch_size, 512, 1024), device=device, dtype=dtype
            ),
            "fps": torch.full((batch_size,), 24, device=device, dtype=dtype),
            "padding_mask": torch.zeros(
                (batch_size, 1, height, width), device=device, dtype=dtype
            ),
        }
        condition, uncondition = self.pipeline.conditioner.get_condition_uncondition(data_batch)
        del condition
        # This exactly matches the upstream temporal Harmonizer inference path.
        uncondition = uncondition.edit_data_type(DataType.IMAGE)
        self._condition_cache[key] = uncondition
        return uncondition

    @staticmethod
    def _expand_scaling_for_views(
        value_B_1_1_1_1: torch.Tensor,
        *,
        batch_size: int,
        num_views: int,
    ) -> torch.Tensor:
        if value_B_1_1_1_1.shape[0] != batch_size:
            raise ValueError("preconditioning coefficient does not have batch B")
        return value_B_1_1_1_1.unsqueeze(1).expand(
            (batch_size, num_views) + tuple(value_B_1_1_1_1.shape[1:])
        ).reshape((batch_size * num_views,) + tuple(value_B_1_1_1_1.shape[1:]))

    def _denoise_latents(
        self,
        latent_BV_C_T_H_W: torch.Tensor,
        *,
        condition,
        camera_ids_B_V: torch.Tensor,
        view_valid_B_V: torch.Tensor,
        official_reference: bool,
        intrinsics: torch.Tensor | None,
        T_ego_from_cam: torch.Tensor | None,
        image_hw: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, num_views = camera_ids_B_V.shape
        if latent_BV_C_T_H_W.shape[0] != batch_size * num_views:
            raise ValueError("latent flat batch does not equal B*V")
        sigma_B_T = torch.full(
            (batch_size, 1),
            self.timestep / 1000.0,
            device=latent_BV_C_T_H_W.device,
            dtype=latent_BV_C_T_H_W.dtype,
        )
        sigma_B_1_T_1_1 = sigma_B_T[:, None, :, None, None]
        c_skip_B, c_out_B, c_in_B, c_noise_B = self.pipeline.scaling(
            sigma=sigma_B_1_T_1_1
        )
        c_skip_BV = self._expand_scaling_for_views(
            c_skip_B, batch_size=batch_size, num_views=num_views
        )
        c_out_BV = self._expand_scaling_for_views(
            c_out_B, batch_size=batch_size, num_views=num_views
        )
        c_in_BV = self._expand_scaling_for_views(
            c_in_B, batch_size=batch_size, num_views=num_views
        )
        condition_fields = condition.to_dict()
        forward = (
            self.dit.forward_official_temporal_reference
            if official_reference
            else self.dit.forward
        )
        geometry_fields = {}
        if not official_reference and (
            bool(getattr(self.dit, "mv_plucker_conditioning_enabled", False))
            or getattr(self.dit, "mv_pose_routing_config", None) is not None
        ):
            geometry_fields = {
                "intrinsics_B_V_3_3": intrinsics,
                "T_ego_from_cam_B_V_4_4": T_ego_from_cam,
                "image_hw": image_hw,
            }
        network_output = forward(
            x_B_C_T_H_W=(latent_BV_C_T_H_W * c_in_BV).to(
                **self.pipeline.tensor_kwargs
            ),
            timesteps_B_T=c_noise_B.squeeze(dim=[1, 3, 4]).to(
                **self.pipeline.tensor_kwargs
            ),
            **condition_fields,
            view_ids_B_V=camera_ids_B_V,
            view_valid_B_V=view_valid_B_V,
            use_cuda_graphs=False,
            **geometry_fields,
        ).float()
        if network_output.shape != latent_BV_C_T_H_W.shape:
            raise RuntimeError(
                "DiT output/latent shape mismatch: "
                f"{tuple(network_output.shape)} != {tuple(latent_BV_C_T_H_W.shape)}"
            )
        return c_skip_BV * latent_BV_C_T_H_W + c_out_BV * network_output

    def _forward_impl(
        self,
        source_B_T_V_C_H_W: torch.Tensor,
        *,
        camera_ids_B_V: torch.Tensor,
        view_valid_B_V: torch.Tensor,
        temporal_offsets: torch.Tensor | None,
        return_all_frames: bool,
        return_metadata: bool,
        official_reference: bool,
        decode_current_only: bool,
        intrinsics: torch.Tensor | None,
        T_ego_from_cam: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, TemporalForwardMetadata]:
        if decode_current_only and return_all_frames:
            raise ValueError("decode_current_only is incompatible with return_all_frames")
        batch = TemporalMultiViewBatch(
            source_B_T_V_C_H_W,
            camera_ids_B_V,
            view_valid_B_V,
            temporal_offsets,
        )
        batch_size, num_rgb_frames, num_views, height, width = batch.validate()
        self._require_source_runtime_contract(source_B_T_V_C_H_W)

        patch_spatial = getattr(self.dit, "patch_spatial", None)
        if patch_spatial is not None:
            if type(patch_spatial) is not int or patch_spatial < 1:
                raise RuntimeError(
                    f"Audited DiT patch_spatial must be a positive integer, got {patch_spatial!r}"
                )
            # The audited Cosmos causal VAE downsamples H/W by eight, followed
            # by the live DiT's spatial patching.  Reject bad public shapes
            # before the tokenizer produces a later, less actionable error.
            spatial_multiple = 8 * patch_spatial
            if height % spatial_multiple or width % spatial_multiple:
                raise ValueError(
                    "source H/W must be divisible by the audited VAE/DiT spatial "
                    f"multiple {spatial_multiple}, got H={height}, W={width}"
                )

        expected_latent_frames = int(self.tokenizer.get_latent_num_frames(num_rgb_frames))
        expected_decoded_frames = int(
            self.tokenizer.get_pixel_num_frames(expected_latent_frames)
        )
        if expected_decoded_frames != num_rgb_frames:
            raise ValueError(
                "The official causal tokenizer cannot round-trip this temporal length: "
                f"T_rgb={num_rgb_frames}, T_latent={expected_latent_frames}, "
                f"T_decoded={expected_decoded_frames}. Use T=1 or an official "
                "1+4k temporal window (the upstream history policy uses T=5)."
            )

        encoder_input_shape = (
            batch_size * num_views,
            source_B_T_V_C_H_W.shape[3],
            num_rgb_frames,
            height,
            width,
        )
        with torch.no_grad():
            latent_BV_C_T_H_W = encode_camera_sequences(
                self.tokenizer,
                source_B_T_V_C_H_W,
                self.pipeline.sigma_data,
            )
        if latent_BV_C_T_H_W.shape[2] != expected_latent_frames:
            raise RuntimeError(
                "Tokenizer temporal contract mismatch: "
                f"expected {expected_latent_frames} latent frames, got "
                f"{latent_BV_C_T_H_W.shape[2]}"
            )
        condition = self._unconditional_condition(
            batch_size=batch_size,
            height=height,
            width=width,
            device=source_B_T_V_C_H_W.device,
            dtype=source_B_T_V_C_H_W.dtype,
        )
        denoised_BV_C_T_H_W = self._denoise_latents(
            latent_BV_C_T_H_W,
            condition=condition,
            camera_ids_B_V=camera_ids_B_V,
            view_valid_B_V=view_valid_B_V,
            official_reference=official_reference,
            intrinsics=intrinsics,
            T_ego_from_cam=T_ego_from_cam,
            image_hw=(height, width),
        )
        # Do not wrap decode in no_grad: frozen operations must propagate pixel
        # losses back to the adapters.
        decode_latent_BV_C_T_H_W = (
            denoised_BV_C_T_H_W[
                :, :, self.current_output_index : self.current_output_index + 1
            ]
            if decode_current_only
            else denoised_BV_C_T_H_W
        )
        decoded_B_T_V_C_H_W = decode_camera_sequences(
            self.tokenizer,
            decode_latent_BV_C_T_H_W,
            self.pipeline.sigma_data,
            batch_size=batch_size,
            num_views=num_views,
        )
        actual_expected_frames = 1 if decode_current_only else expected_decoded_frames
        if decoded_B_T_V_C_H_W.shape[1] != actual_expected_frames:
            raise RuntimeError(
                "Tokenizer decode temporal contract mismatch: "
                f"expected {actual_expected_frames}, got {decoded_B_T_V_C_H_W.shape[1]}"
            )
        if (
            self.training
            and torch.is_grad_enabled()
            and denoised_BV_C_T_H_W.requires_grad
            and not decoded_B_T_V_C_H_W.requires_grad
        ):
            raise RuntimeError(
                "Tokenizer decode detached the graph; apply the official Harmonizer tokenizer patch"
            )
        if decoded_B_T_V_C_H_W.shape[1] <= self.current_output_index:
            raise RuntimeError("decoded sequence does not contain the official current-frame slot")
        output = (
            decoded_B_T_V_C_H_W
            if return_all_frames
            else decoded_B_T_V_C_H_W[:, self.current_output_index]
        )
        if not return_metadata:
            return output
        metadata = TemporalForwardMetadata(
            architecture=self.architecture,
            batch_size=batch_size,
            num_rgb_frames=num_rgb_frames,
            num_views=num_views,
            input_shape=tuple(source_B_T_V_C_H_W.shape),
            vae_encoder_input_shape=encoder_input_shape,
            latent_shape=tuple(latent_BV_C_T_H_W.shape),
            dit_output_shape=tuple(denoised_BV_C_T_H_W.shape),
            decoded_all_frames_shape=tuple(decoded_B_T_V_C_H_W.shape),
            output_shape=tuple(output.shape),
            output_semantics=(
                "all_decoded_temporal_slots"
                if return_all_frames
                else "official_current_slot_zero_only"
            ),
        )
        return output, metadata

    def forward(
        self,
        source_B_T_V_C_H_W: torch.Tensor,
        *,
        camera_ids_B_V: torch.Tensor,
        view_valid_B_V: torch.Tensor,
        temporal_offsets: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        T_ego_from_cam: torch.Tensor | None = None,
        return_all_frames: bool = False,
        return_metadata: bool = False,
        decode_current_only: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, TemporalForwardMetadata]:
        return self._forward_impl(
            source_B_T_V_C_H_W,
            camera_ids_B_V=camera_ids_B_V,
            view_valid_B_V=view_valid_B_V,
            temporal_offsets=temporal_offsets,
            return_all_frames=return_all_frames,
            return_metadata=return_metadata,
            official_reference=False,
            decode_current_only=decode_current_only,
            intrinsics=intrinsics,
            T_ego_from_cam=T_ego_from_cam,
        )

    def forward_official_temporal_reference(
        self,
        source_B_T_V_C_H_W: torch.Tensor,
        *,
        camera_ids_B_V: torch.Tensor,
        view_valid_B_V: torch.Tensor,
        temporal_offsets: torch.Tensor | None = None,
        return_all_frames: bool = False,
        return_metadata: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, TemporalForwardMetadata]:
        """Reference: V independent complete official temporal Harmonizers."""

        return self._forward_impl(
            source_B_T_V_C_H_W,
            camera_ids_B_V=camera_ids_B_V,
            view_valid_B_V=view_valid_B_V,
            temporal_offsets=temporal_offsets,
            return_all_frames=return_all_frames,
            return_metadata=return_metadata,
            official_reference=True,
            decode_current_only=False,
            intrinsics=None,
            T_ego_from_cam=None,
        )
