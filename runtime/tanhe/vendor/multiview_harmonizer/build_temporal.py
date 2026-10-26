"""Single config entry point for Scheme A-TMV."""

from __future__ import annotations

from pathlib import Path

from .config_temporal import TemporalFrameworkConfig
from .factory_temporal import build_temporal_pipeline_with_official_harmonizer
from .wrapper_temporal import TemporalMultiViewPix2PixTurbo
from .dit_a_temporal import attach_a1_plucker_conditioning


def build_temporal_from_config(
    config_path: str | Path,
    *,
    device: str = "cuda",
    verify_known_hashes: bool = True,
    load_audit_output: str | Path | None = None,
) -> TemporalMultiViewPix2PixTurbo:
    config = TemporalFrameworkConfig.from_json(config_path)
    if load_audit_output is None:
        load_audit_output = (
            Path(config_path).resolve().parent.parent
            / "reports"
            / "scheme_a_temporal_mv"
            / "load_audit.json"
        )
    pipeline, audit = build_temporal_pipeline_with_official_harmonizer(
        cosmos_checkpoint_dir=config.cosmos_checkpoint_dir,
        harmonizer_checkpoint=config.harmonizer_checkpoint,
        device=device,
        dtype=config.dtype,
        verify_known_hashes=verify_known_hashes,
        load_audit_output=load_audit_output,
        adapter_after_blocks=config.adapter_after_blocks,
        adapter_dim=config.adapter_dim,
        adapter_heads=config.adapter_heads,
        adapter_pool_factor=config.adapter_pool_factor,
        max_views=config.max_views,
    )
    if audit.tensor_coverage != 1.0 or audit.numel_coverage != 1.0:
        raise RuntimeError("official checkpoint audit fell below 100%")
    if config.pose_routing is not None:
        pipeline.dit.mv_pose_routing_config = config.pose_routing
        pipeline.dit.mv_last_pose_routing = None
    if config.plucker_enabled:
        attach_a1_plucker_conditioning(
            pipeline.dit,
            hidden_dim=config.plucker_hidden_dim,
            gate_init=config.plucker_gate_init,
        )
    return TemporalMultiViewPix2PixTurbo(pipeline, timestep=config.timestep)
