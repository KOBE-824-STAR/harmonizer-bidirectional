"""Single construction entry point shared by future inference and training."""

from __future__ import annotations

from pathlib import Path

from .config import FrameworkConfig
from .factory import build_pipeline_with_official_harmonizer
from .wrapper import MultiViewPix2PixTurbo


def build_from_config(
    config_path: str | Path,
    *,
    device: str = "cuda",
    verify_known_hashes: bool = True,
) -> MultiViewPix2PixTurbo:
    config = FrameworkConfig.from_json(config_path)
    pipeline, audit = build_pipeline_with_official_harmonizer(
        architecture=config.architecture,
        cosmos_checkpoint_dir=config.cosmos_checkpoint_dir,
        harmonizer_checkpoint=config.harmonizer_checkpoint,
        device=device,
        dtype=config.dtype,
        verify_known_hashes=verify_known_hashes,
        load_audit_output=(
            Path(config_path).resolve().parent.parent
            / "reports"
            / f"load_audit_{config.architecture.value}.json"
        ),
        adapter_after_blocks=config.adapter_after_blocks,
        adapter_dim=config.adapter_dim,
        adapter_heads=config.adapter_heads,
        adapter_pool_factor=config.adapter_pool_factor,
        max_views=config.max_views,
        attention_variant=config.attention_variant,
        camera_adjacency=config.camera_adjacency,
        camera_schema_version=config.camera_schema_version,
        adjacency_undirected=config.adjacency_undirected,
    )
    if audit.tensor_coverage != 1.0 or audit.numel_coverage != 1.0:
        raise RuntimeError("Official checkpoint audit unexpectedly fell below 100%")
    return MultiViewPix2PixTurbo(pipeline, timestep=config.timestep)
