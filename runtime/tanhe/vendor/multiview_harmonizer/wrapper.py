"""Public B,V,C,H,W wrapper around the official DiffusionHarmonizer pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cosmos_predict2.conditioner import DataType

from .layout import (
    MultiViewBatch,
    decode_views_independently,
    default_camera_ids,
    dit_time_to_views,
    encode_views_independently,
    views_to_dit_time,
)
from .tokenizer_utils import inner_tokenizer_module


@dataclass(frozen=True)
class ForwardMetadata:
    architecture: str
    batch_size: int
    num_views: int
    input_shape: tuple[int, ...]
    latent_shape: tuple[int, ...]


class MultiViewPix2PixTurbo(nn.Module):
    """Harmonizer wrapper shared by independent, scheme A, and scheme B.

    It intentionally implements preconditioning locally so view IDs can be
    passed to the upgraded DiT without changing Cosmos condition dataclasses.
    """

    def __init__(self, pipeline: nn.Module, *, timestep: int = 250) -> None:
        super().__init__()
        if not 0 <= timestep <= 1000:
            raise ValueError(f"timestep must be in [0,1000], got {timestep}")
        if not hasattr(pipeline.dit, "mv_architecture"):
            raise ValueError("pipeline.dit must be upgraded to independent, A, or B first")
        self.pipeline = pipeline
        self.timestep = int(timestep)
        self.architecture = str(pipeline.dit.mv_architecture)
        self._condition_cache: dict[tuple, object] = {}
        self.freeze_tokenizer()

    @property
    def dit(self):
        return self.pipeline.dit

    @property
    def tokenizer(self):
        return self.pipeline.tokenizer

    def freeze_tokenizer(self) -> None:
        tokenizer_module = inner_tokenizer_module(self.tokenizer)
        tokenizer_module.requires_grad_(False)
        tokenizer_module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # nn.Module.train() recursively flips the tokenizer; restore the hard
        # contract immediately while leaving DiT/adapters in the requested mode.
        self.freeze_tokenizer()
        return self

    def _unconditional_condition(
        self,
        *,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        key = (batch, height, width, str(device), dtype)
        cached = self._condition_cache.get(key)
        if cached is not None:
            return cached
        data_batch = {
            "dataset_name": "image_data",
            "images": torch.zeros((batch, 3, height, width), device=device, dtype=dtype),
            "t5_text_embeddings": torch.zeros(
                (batch, 512, 1024), device=device, dtype=dtype
            ),
            "fps": torch.full((batch,), 24, device=device, dtype=dtype),
            "padding_mask": torch.zeros((batch, 1, height, width), device=device, dtype=dtype),
        }
        condition, uncondition = self.pipeline.conditioner.get_condition_uncondition(data_batch)
        del condition
        uncondition = uncondition.edit_data_type(DataType.IMAGE)
        self._condition_cache[key] = uncondition
        return uncondition

    def _denoise_latents(
        self,
        latent_B_C_V_H_W: torch.Tensor,
        condition,
        camera_ids_B_V: torch.Tensor,
        view_valid_B_V: torch.Tensor,
    ) -> torch.Tensor:
        batch = latent_B_C_V_H_W.shape[0]
        sigma_B_T = torch.full(
            (batch, 1),
            self.timestep / 1000.0,
            device=latent_B_C_V_H_W.device,
            dtype=latent_B_C_V_H_W.dtype,
        )
        sigma_B_1_T_1_1 = sigma_B_T[:, None, :, None, None]
        c_skip, c_out, c_in, c_noise = self.pipeline.scaling(sigma=sigma_B_1_T_1_1)
        condition_fields = condition.to_dict()
        network_output = self.dit(
            x_B_C_T_H_W=(latent_B_C_V_H_W * c_in).to(**self.pipeline.tensor_kwargs),
            timesteps_B_T=c_noise.squeeze(dim=[1, 3, 4]).to(**self.pipeline.tensor_kwargs),
            **condition_fields,
            view_ids_B_T=camera_ids_B_V,
            view_valid_B_T=view_valid_B_V,
            use_cuda_graphs=False,
        ).float()
        return c_skip * latent_B_C_V_H_W + c_out * network_output

    def forward(
        self,
        source_B_V_C_H_W: torch.Tensor,
        *,
        camera_ids_B_V: torch.Tensor | None = None,
        view_valid_B_V: torch.Tensor | None = None,
        return_metadata: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ForwardMetadata]:
        if source_B_V_C_H_W.ndim != 5:
            raise ValueError(
                "source must be [B,V,3,H,W], got "
                f"{tuple(source_B_V_C_H_W.shape)}"
            )
        batch, views = source_B_V_C_H_W.shape[:2]
        if camera_ids_B_V is None:
            camera_ids_B_V = default_camera_ids(batch, views, source_B_V_C_H_W.device)
        if view_valid_B_V is None:
            view_valid_B_V = torch.ones(
                (batch, views), device=source_B_V_C_H_W.device, dtype=torch.bool
            )
        validated = MultiViewBatch(source_B_V_C_H_W, camera_ids_B_V, view_valid_B_V)
        _, _, height, width = validated.validate()

        # Input pixels do not require gradients.  Decode is deliberately not in
        # no_grad: gradients must flow from pixel losses through the frozen
        # decoder back into the DiT output latents during later training.
        with torch.no_grad():
            latent_B_V_C_1_H_W = encode_views_independently(
                self.tokenizer,
                source_B_V_C_H_W,
                self.pipeline.sigma_data,
            )
        latent_B_C_V_H_W = views_to_dit_time(latent_B_V_C_1_H_W)
        condition = self._unconditional_condition(
            batch=batch,
            height=height,
            width=width,
            device=source_B_V_C_H_W.device,
            dtype=source_B_V_C_H_W.dtype,
        )
        denoised_B_C_V_H_W = self._denoise_latents(
            latent_B_C_V_H_W,
            condition,
            camera_ids_B_V,
            view_valid_B_V,
        )
        decoded = decode_views_independently(
            self.tokenizer,
            dit_time_to_views(denoised_B_C_V_H_W.to(latent_B_C_V_H_W.dtype)),
            self.pipeline.sigma_data,
        )
        if (
            self.training
            and torch.is_grad_enabled()
            and denoised_B_C_V_H_W.requires_grad
            and not decoded.requires_grad
        ):
            raise RuntimeError(
                "Tokenizer decode detached the graph. Apply the official "
                "DiffusionHarmonizer tokenizer.patch before later pixel-loss training."
            )
        if not return_metadata:
            return decoded
        metadata = ForwardMetadata(
            architecture=self.architecture,
            batch_size=batch,
            num_views=views,
            input_shape=tuple(source_B_V_C_H_W.shape),
            latent_shape=tuple(latent_B_V_C_1_H_W.shape),
        )
        return decoded, metadata
