"""Scheme B-TMV-warm T2: native temporal path with a zero view extension.

T2 deliberately has no cross-view parameters. It proves that folding only
``B*V`` around the unmodified Harmonizer VAE/DiT/decoder preserves the native
temporal compression contract. The geometry-aware residual is a T3 concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn


TOKEN_LAYOUT_VERSION = "tmv_factorized_v1"
SCHEME = "B_TMV"
VARIANT = "warm_v1"
GEOMETRY_VERSION = "relative_pose_v1"
TEMPORAL_MODE = "harmonizer_native"
DEFAULT_TEMPORAL_COMPRESSION_FACTOR = 4


@dataclass(frozen=True)
class TemporalLengthContract:
    """Native RGB/latent/decoded frame-count mapping."""

    rgb_frames: int
    latent_frames: int
    decoded_frames: int
    compression_factor: int

    @property
    def roundtrippable(self) -> bool:
        """Whether native decode happens to reproduce the RGB input length."""

        return self.rgb_frames == self.decoded_frames


@dataclass(frozen=True)
class BTMVForwardMetadata:
    architecture: str
    variant: str
    temporal: bool
    temporal_mode: str
    geometry_version: str
    token_layout_version: str
    batch_size: int
    num_rgb_frames: int
    num_views: int
    input_shape: tuple[int, ...]
    vae_encoder_input_shape: tuple[int, ...]
    latent_shape: tuple[int, ...]
    dit_output_shape: tuple[int, ...]
    decoded_output_shape: tuple[int, ...]
    num_decoded_frames: int
    camera_axis_enters_temporal_module: bool
    cross_view_parameters_attached: bool


def official_temporal_length_contract(
    rgb_frames: int,
    compression_factor: int = DEFAULT_TEMPORAL_COMPRESSION_FACTOR,
) -> TemporalLengthContract:
    """Return the official causal tokenizer frame-count contract.

    Dynamic positive RGB lengths are all valid inputs. Native decoding is
    intentionally *not* required to have the same length as the RGB input.
    """

    if not isinstance(rgb_frames, int) or isinstance(rgb_frames, bool) or rgb_frames < 1:
        raise ValueError("rgb_frames must be a positive integer")
    if (
        not isinstance(compression_factor, int)
        or isinstance(compression_factor, bool)
        or compression_factor < 1
    ):
        raise ValueError("compression_factor must be a positive integer")
    latent_frames = 1 + (rgb_frames - 1) // compression_factor
    decoded_frames = 1 + compression_factor * (latent_frames - 1)
    return TemporalLengthContract(
        rgb_frames=rgb_frames,
        latent_frames=latent_frames,
        decoded_frames=decoded_frames,
        compression_factor=compression_factor,
    )


def native_tokenizer_length_contract(tokenizer: Any, rgb_frames: int) -> TemporalLengthContract:
    """Query the live tokenizer instead of assuming RGB and latent T agree."""

    if not isinstance(rgb_frames, int) or isinstance(rgb_frames, bool) or rgb_frames < 1:
        raise ValueError("rgb_frames must be a positive integer")
    latent_frames = int(tokenizer.get_latent_num_frames(rgb_frames))
    decoded_frames = int(tokenizer.get_pixel_num_frames(latent_frames))
    if latent_frames < 1 or decoded_frames < 1:
        raise RuntimeError("tokenizer returned a non-positive temporal length")
    inferred = (
        (decoded_frames - 1) // (latent_frames - 1)
        if latent_frames > 1
        else DEFAULT_TEMPORAL_COMPRESSION_FACTOR
    )
    return TemporalLengthContract(rgb_frames, latent_frames, decoded_frames, inferred)


def require_official_temporal_roundtrip(rgb_frames: int) -> TemporalLengthContract:
    """Compatibility helper; never used to restrict B-TMV inputs."""

    contract = official_temporal_length_contract(rgb_frames)
    if not contract.roundtrippable:
        raise ValueError(
            "Native decode length differs from RGB input length: "
            f"T_rgb={contract.rgb_frames}, T_latent={contract.latent_frames}, "
            f"T_decoded={contract.decoded_frames}"
        )
    return contract


def rgb_btvc_to_bvcthw(source_B_T_V_C_H_W: torch.Tensor) -> torch.Tensor:
    """Fold only B and V; camera never enters the tokenizer temporal axis."""

    if source_B_T_V_C_H_W.ndim != 6:
        raise ValueError("B-TMV-warm input must be [B,T,V,C,H,W]")
    batch, frames, views, channels, height, width = source_B_T_V_C_H_W.shape
    if min(batch, frames, views, channels, height, width) < 1:
        raise ValueError("B-TMV-warm input dimensions must be positive")
    return source_B_T_V_C_H_W.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(
        batch * views, channels, frames, height, width
    )


def rgb_bvcthw_to_btvc(
    output_BV_C_T_H_W: torch.Tensor, *, batch_size: int, num_views: int
) -> torch.Tensor:
    """Restore dynamic native decoded T into public ``[B,Tout,V,C,H,W]``."""

    if output_BV_C_T_H_W.ndim != 5:
        raise ValueError("Official temporal output must be [B*V,C,T,H,W]")
    flat_batch, channels, frames, height, width = output_BV_C_T_H_W.shape
    if flat_batch != batch_size * num_views:
        raise ValueError("Official temporal output batch does not equal B*V")
    return output_BV_C_T_H_W.reshape(
        batch_size, num_views, channels, frames, height, width
    ).permute(0, 3, 1, 2, 4, 5).contiguous()


def tokens_bvthwd_to_bvthwd(
    tokens_BV_T_H_W_D: torch.Tensor, *, batch_size: int, num_views: int
) -> torch.Tensor:
    """Expose ``[B,V,T,H,W,D]`` without flattening T with V."""

    if tokens_BV_T_H_W_D.ndim != 5:
        raise ValueError("Official DiT tokens must be [B*V,T,H,W,D]")
    flat_batch, frames, height, width, dim = tokens_BV_T_H_W_D.shape
    if flat_batch != batch_size * num_views:
        raise ValueError("Official DiT token batch does not equal B*V")
    return tokens_BV_T_H_W_D.reshape(
        batch_size, num_views, frames, height, width, dim
    )


def _expand_first_dim(
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


class BTMVWarmTemporalDiT:
    """Parameter-preserving T2 mixin for a loaded official temporal DiT.

    The runtime class inherits this mixin before the original DiT class. Its
    forward validates semantic camera metadata, expands only the leading batch
    dimension of conditions, then invokes the inherited official forward.
    No attention implementation or Parameter is replaced at T2.
    """

    @classmethod
    def upgrade_from_official(cls, dit: nn.Module, *, max_views: int = 6) -> nn.Module:
        if hasattr(dit, "mv_architecture"):
            raise ValueError("DiT has already been upgraded")
        required = ("patch_temporal", "blocks", "forward")
        missing = [name for name in required if not hasattr(dit, name)]
        if missing:
            raise TypeError(f"Official temporal DiT contract missing {missing}")
        if int(dit.patch_temporal) != 1:
            raise ValueError("B-TMV-warm requires official patch_temporal=1")
        if not isinstance(max_views, int) or max_views < 1:
            raise ValueError("max_views must be positive")
        original_class = dit.__class__
        runtime_class = type(
            "BTMVWarmTemporalDiTRuntime",
            (cls, original_class),
            {"__module__": cls.__module__},
        )
        dit.__class__ = runtime_class
        dit.mv_architecture = SCHEME
        dit.mv_attention_variant = VARIANT
        dit.mv_temporal = True
        dit.mv_temporal_mode = TEMPORAL_MODE
        dit.mv_geometry_version = GEOMETRY_VERSION
        dit.mv_token_layout_version = TOKEN_LAYOUT_VERSION
        dit.mv_max_views = max_views
        dit.mv_cross_view_parameters_attached = False
        dit.mv_t2_original_class = original_class.__name__
        return dit

    def _validate_view_metadata(
        self,
        flat_batch: int,
        view_ids_B_V: torch.Tensor | None,
        view_valid_B_V: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[int, int, torch.Tensor, torch.Tensor]:
        if view_ids_B_V is None or view_valid_B_V is None:
            raise ValueError("B-TMV-warm requires view_ids_B_V and view_valid_B_V")
        if view_ids_B_V.ndim != 2:
            raise ValueError("view_ids_B_V must be [B,V]")
        batch_size, num_views = view_ids_B_V.shape
        if flat_batch != batch_size * num_views:
            raise ValueError("DiT leading dimension must equal B*V")
        if not 1 <= num_views <= self.mv_max_views:
            raise ValueError(f"num_views must be in [1,{self.mv_max_views}]")
        if view_valid_B_V.shape != (batch_size, num_views):
            raise ValueError("view_valid_B_V must match view_ids_B_V")
        if view_ids_B_V.dtype != torch.long or view_valid_B_V.dtype != torch.bool:
            raise TypeError("view IDs must be long and validity must be bool")
        if view_ids_B_V.device != device or view_valid_B_V.device != device:
            raise ValueError("view metadata and DiT input must share a device")
        if not bool(view_valid_B_V.all().item()):
            raise ValueError("T2 does not fabricate missing camera sequences")
        for sample in range(batch_size):
            ids = view_ids_B_V[sample]
            if ids.unique().numel() != ids.numel():
                raise ValueError(f"duplicate camera IDs in sample {sample}")
        return batch_size, num_views, view_ids_B_V, view_valid_B_V

    def _native_temporal_forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        data_type: Any = None,
        use_cuda_graphs: bool = False,
        *,
        view_ids_B_V: torch.Tensor | None = None,
        view_valid_B_V: torch.Tensor | None = None,
        **unused_condition_fields: Any,
    ) -> torch.Tensor:
        if unused_condition_fields:
            raise TypeError(f"Unsupported condition fields: {sorted(unused_condition_fields)}")
        if use_cuda_graphs:
            raise NotImplementedError("CUDA graphs are disabled for B-TMV T2")
        if x_B_C_T_H_W.ndim != 5 or x_B_C_T_H_W.shape[2] < 1:
            raise ValueError("DiT input must be [B*V,C,Tlatent,H,W]")
        batch_size, num_views, _, _ = self._validate_view_metadata(
            x_B_C_T_H_W.shape[0], view_ids_B_V, view_valid_B_V, x_B_C_T_H_W.device
        )
        timesteps = _expand_first_dim(
            timesteps_B_T,
            name="timesteps_B_T",
            batch_size=batch_size,
            num_views=num_views,
        )
        context = _expand_first_dim(
            crossattn_emb,
            name="crossattn_emb",
            batch_size=batch_size,
            num_views=num_views,
        )
        expanded_fps = _expand_first_dim(
            fps, name="fps", batch_size=batch_size, num_views=num_views
        )
        expanded_padding = _expand_first_dim(
            padding_mask,
            name="padding_mask",
            batch_size=batch_size,
            num_views=num_views,
        )
        official_forward = super(BTMVWarmTemporalDiT, self).forward
        return official_forward(
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps,
            crossattn_emb=context,
            fps=expanded_fps,
            padding_mask=expanded_padding,
            data_type=data_type,
            use_cuda_graphs=False,
        )

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._native_temporal_forward(*args, **kwargs)

    def forward_official_temporal_reference(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._native_temporal_forward(*args, **kwargs)


class BTMVTemporalBaseline(nn.Module):
    """Camera-safe wrapper around an unchanged native temporal callable.

    ``official_forward`` receives ``[B*V,C,T,H,W]``. Its returned temporal
    length may differ from the input; the public output is consequently
    ``[B,Tout,V,C,H,W]``.
    """

    def __init__(self, official_forward: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.official_forward = official_forward

    def forward(self, source_B_T_V_C_H_W: torch.Tensor) -> torch.Tensor:
        batch, _, views, _, _, _ = source_B_T_V_C_H_W.shape
        official_input = rgb_btvc_to_bvcthw(source_B_T_V_C_H_W)
        official_output = self.official_forward(official_input)
        if not isinstance(official_output, torch.Tensor):
            raise TypeError("official_forward must return a tensor")
        return rgb_bvcthw_to_btvc(official_output, batch_size=batch, num_views=views)


class BTMVNativeTemporalPipeline(nn.Module):
    """T2 end-to-end wrapper around an already loaded official pipeline."""

    def __init__(self, pipeline: nn.Module, *, timestep: int = 250) -> None:
        super().__init__()
        if not 0 <= timestep <= 1000:
            raise ValueError("timestep must be in [0,1000]")
        if getattr(pipeline.dit, "mv_architecture", None) != SCHEME:
            raise ValueError("pipeline.dit must be upgraded to B_TMV")
        if bool(getattr(pipeline.dit, "mv_cross_view_parameters_attached", True)):
            raise ValueError("T2 baseline must not have cross-view parameters")
        self.pipeline = pipeline
        self.timestep = int(timestep)
        self.architecture = SCHEME
        self._condition_cache: dict[tuple[Any, ...], Any] = {}
        self.pipeline.requires_grad_(False)
        self.pipeline.eval()

    @property
    def tokenizer(self) -> Any:
        return self.pipeline.tokenizer

    @property
    def dit(self) -> nn.Module:
        return self.pipeline.dit

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
                "fps": torch.full((batch,), 24, device=source.device, dtype=source.dtype),
                "padding_mask": torch.zeros(
                    (batch, 1, height, width), device=source.device, dtype=source.dtype
                ),
            }
            _, uncondition = self.pipeline.conditioner.get_condition_uncondition(data)
            self._condition_cache[key] = uncondition.edit_data_type(DataType.IMAGE)
        return self._condition_cache[key]

    def encode(self, source_B_T_V_C_H_W: torch.Tensor) -> torch.Tensor:
        flat = rgb_btvc_to_bvcthw(source_B_T_V_C_H_W)
        return self.tokenizer.encode(flat) * self.pipeline.sigma_data

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
        official_reference: bool = False,
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

        def expand(value: torch.Tensor) -> torch.Tensor:
            return value.unsqueeze(1).expand(
                (batch_size, num_views) + tuple(value.shape[1:])
            ).reshape((batch_size * num_views,) + tuple(value.shape[1:]))

        c_skip_bv, c_out_bv, c_in_bv = map(expand, (c_skip, c_out, c_in))
        condition = self._condition(batch_size, source_height, source_width, source)
        method = (
            self.dit.forward_official_temporal_reference
            if official_reference
            else self.dit.forward
        )
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
        official_reference: bool,
        return_metadata: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, BTMVForwardMetadata]:
        if source_B_T_V_C_H_W.ndim != 6:
            raise ValueError("source must be [B,T,V,C,H,W]")
        batch, rgb_frames, views, channels, height, width = source_B_T_V_C_H_W.shape
        if channels != 3:
            raise ValueError("B-TMV RGB input must have C=3")
        contract = native_tokenizer_length_contract(self.tokenizer, rgb_frames)
        latent = self.encode(source_B_T_V_C_H_W)
        if latent.shape[2] != contract.latent_frames:
            raise RuntimeError("live tokenizer latent length disagrees with its public contract")
        network_output, denoised = self.denoise(
            latent,
            batch_size=batch,
            num_views=views,
            source_height=height,
            source_width=width,
            source=source_B_T_V_C_H_W,
            view_ids_B_V=view_ids_B_V,
            view_valid_B_V=view_valid_B_V,
            official_reference=official_reference,
        )
        decoded = self.decode(denoised, batch_size=batch, num_views=views)
        if decoded.shape[1] != contract.decoded_frames:
            raise RuntimeError("live tokenizer decoded length disagrees with its public contract")
        if not return_metadata:
            return decoded
        metadata = BTMVForwardMetadata(
            architecture=SCHEME,
            variant=VARIANT,
            temporal=True,
            temporal_mode=TEMPORAL_MODE,
            geometry_version=GEOMETRY_VERSION,
            token_layout_version=TOKEN_LAYOUT_VERSION,
            batch_size=batch,
            num_rgb_frames=rgb_frames,
            num_views=views,
            input_shape=tuple(source_B_T_V_C_H_W.shape),
            vae_encoder_input_shape=(batch * views, channels, rgb_frames, height, width),
            latent_shape=tuple(latent.shape),
            dit_output_shape=tuple(network_output.shape),
            decoded_output_shape=tuple(decoded.shape),
            num_decoded_frames=contract.decoded_frames,
            camera_axis_enters_temporal_module=False,
            cross_view_parameters_attached=False,
        )
        return decoded, metadata

    def forward(
        self,
        source_B_T_V_C_H_W: torch.Tensor,
        *,
        view_ids_B_V: torch.Tensor,
        view_valid_B_V: torch.Tensor,
        return_metadata: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, BTMVForwardMetadata]:
        return self._forward_impl(
            source_B_T_V_C_H_W,
            view_ids_B_V=view_ids_B_V,
            view_valid_B_V=view_valid_B_V,
            official_reference=False,
            return_metadata=return_metadata,
        )

    def forward_official_temporal_reference(
        self,
        source_B_T_V_C_H_W: torch.Tensor,
        *,
        view_ids_B_V: torch.Tensor,
        view_valid_B_V: torch.Tensor,
        return_metadata: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, BTMVForwardMetadata]:
        return self._forward_impl(
            source_B_T_V_C_H_W,
            view_ids_B_V=view_ids_B_V,
            view_valid_B_V=view_valid_B_V,
            official_reference=True,
            return_metadata=return_metadata,
        )
