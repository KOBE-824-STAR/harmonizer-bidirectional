"""Production builders for Scheme B-TMV T3 and geometry-conditioned T4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .b_tmv_crossview import (
    B1_VARIANT,
    TemporalLatentCrossViewDiT,
    TemporalMultiViewWrapper,
    attach_b1_plucker_conditioning,
)
from .b_tmv_geometry import attach_relative_geometry_conditioning
from .b_tmv_routing import POSE_ROUTING_VARIANT, PoseRoutingConfig, attach_pose_routing
from .checkpoint import LoadAudit, load_official_harmonizer_checkpoint, require_sha256
from .constants import OFFICIAL_PATHS
from .runtime_contract import require_official_text2image_dit_contract


def read_b_tmv_t3_config(path: str | Path) -> dict[str, Any]:
    """Read and strictly validate the fixed T3-warm architecture contract."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "scheme": "B_TMV",
        "architecture": "B_TMV",
        "variant": "warm_v1",
        "attention_variant": "warm_v1",
        "temporal": True,
        "temporal_mode": "harmonizer_native",
        "geometry_version": "relative_pose_v1",
        "token_layout_version": "tmv_factorized_v1",
        "camera_axis_enters_temporal_vae": False,
        "flatten_TV_attention": False,
        "cross_view_stage": "latent_temporal_residual_T3",
        "cross_view_mode": "same_latent_time_same_spatial_token_view_only",
        "cross_view_gate_initialization": 0.0,
        "cross_view_adapter_dim": 160,
        "cross_view_adapter_heads": 8,
        "max_views": 6,
        "undirected": True,
        "auto_add_reverse_edges": False,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in required.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"B-TMV T3 model config mismatch: {mismatches}")
    if config.get("training_enabled") is not False:
        raise ValueError("The immutable model contract must keep training_enabled=false")
    if set(config.get("camera_adjacency", {})) != {
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
    }:
        raise ValueError("B-TMV camera adjacency is not the canonical six-camera graph")
    return config


def read_b_tmv_t4_config(path: str | Path) -> dict[str, Any]:
    """Read and strictly validate the fixed T4 geometry architecture contract."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    return _validate_t4_config(config)


def _validate_t4_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "scheme": "B_TMV",
        "architecture": "B_TMV",
        "variant": "warm_v1_geometry_t4",
        "attention_variant": "warm_v1",
        "temporal": True,
        "temporal_mode": "harmonizer_native",
        "geometry_version": "relative_pose_v1",
        "token_layout_version": "tmv_factorized_v1",
        "camera_axis_enters_temporal_vae": False,
        "flatten_TV_attention": False,
        "cross_view_stage": "latent_temporal_residual_T4_geometry_conditioned",
        "cross_view_mode": "same_latent_time_same_spatial_token_view_only",
        "cross_view_gate_initialization": 0.0,
        "cross_view_adapter_dim": 160,
        "cross_view_adapter_heads": 8,
        "max_views": 6,
        "undirected": True,
        "auto_add_reverse_edges": False,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in required.items()
        if config.get(key) != value
    }
    geometry_required = {
        "enabled": True,
        "mode": "relative_pose",
        "extrinsic_convention": "T_ego_from_cam",
        "relative_convention": "inv(T_ego_from_cam_dst) @ T_ego_from_cam_src",
        "feature_dim": 20,
        "bottleneck_dim": 64,
        "gate_init": 0.0,
        "normalize_intrinsics": True,
        "conditioning": "directed_edge_film_on_source_key_value",
        "shared_encoder": True,
        "global_vt_attention": False,
    }
    geometry = config.get("geometry", {})
    for key, value in geometry_required.items():
        if geometry.get(key) != value:
            mismatches[f"geometry.{key}"] = {
                "expected": value,
                "actual": geometry.get(key),
            }
    if mismatches:
        raise ValueError(f"B-TMV T4 model config mismatch: {mismatches}")
    if config.get("training_enabled") is not False:
        raise ValueError("The immutable model contract must keep training_enabled=false")
    if set(config.get("camera_adjacency", {})) != {
        "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK",
        "CAM_BACK_LEFT", "CAM_FRONT_LEFT",
    }:
        raise ValueError("B-TMV camera adjacency is not the canonical six-camera graph")
    return config


def read_b_tmv_pose_routing_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_pose_routing_config(config)
    return config


def read_b_tmv_b1_config(path: str | Path) -> dict[str, Any]:
    """Read B1 as an exact T4 contract plus token-level Plucker Q/K deltas."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("variant") != B1_VARIANT:
        raise ValueError("B1 model variant mismatch")
    if config.get("cross_view_stage") != "latent_temporal_residual_B1_plucker":
        raise ValueError("B1 cross_view_stage mismatch")
    plucker = config.get("plucker", {})
    required = {
        "enabled": True,
        "mode": "token_center_ray",
        "dimension": 6,
        "hidden_dim": 64,
        "gate_init": 0.0,
        "conditioning": "per_token_query_key_delta",
        "value_conditioning": False,
        "extrinsic_convention": "T_ego_from_cam",
    }
    mismatches = {
        key: {"expected": value, "actual": plucker.get(key)}
        for key, value in required.items()
        if plucker.get(key) != value
    }
    if mismatches:
        raise ValueError(f"B1 Plucker config mismatch: {mismatches}")
    base = dict(
        config,
        variant="warm_v1_geometry_t4",
        cross_view_stage="latent_temporal_residual_T4_geometry_conditioned",
    )
    base.pop("plucker", None)
    _validate_t4_config(base)
    return config


def read_b_tmv_pose_b1_config(path: str | Path) -> dict[str, Any]:
    """Validate a new B1 lineage whose runtime edges come from pose routing."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    return _validate_pose_b1_config(config)


def _validate_pose_b1_config(config: dict[str, Any]) -> dict[str, Any]:
    from .b_tmv_crossview import POSE_B1_VARIANT

    if config.get("variant") != POSE_B1_VARIANT:
        raise ValueError("pose-routed B1 variant mismatch")
    if config.get("cross_view_stage") != "latent_temporal_residual_pose_routed_B1_plucker":
        raise ValueError("pose-routed B1 cross_view_stage mismatch")
    plucker = config.get("plucker", {})
    required = {
        "enabled": True, "mode": "token_center_ray", "dimension": 6,
        "hidden_dim": 64, "gate_init": 0.0,
        "conditioning": "per_token_query_key_delta",
        "value_conditioning": False,
        "extrinsic_convention": "T_ego_from_cam",
    }
    if any(plucker.get(key) != value for key, value in required.items()):
        raise ValueError("pose-routed B1 Plucker contract mismatch")
    base = dict(config, variant=POSE_ROUTING_VARIANT,
                cross_view_stage="latent_temporal_residual_T4_pose_routed")
    base.pop("plucker", None)
    _validate_pose_routing_config(base)
    return config


def _validate_pose_routing_config(config: dict[str, Any]) -> None:
    if config.get("variant") != POSE_ROUTING_VARIANT:
        raise ValueError("pose routing model variant mismatch")
    if config.get("cross_view_stage") != "latent_temporal_residual_T4_pose_routed":
        raise ValueError("pose routing cross_view_stage mismatch")
    if config.get("camera_adjacency_role") != "t4_warm_start_provenance_only":
        raise ValueError("pose routing must mark the legacy graph as provenance only")
    PoseRoutingConfig.from_dict(config.get("camera_routing"))
    # Keep every original T4 geometry/backbone contract; only routing changes.
    base = dict(config, variant="warm_v1_geometry_t4",
                cross_view_stage="latent_temporal_residual_T4_geometry_conditioned")
    _validate_t4_config(base)


def build_b_tmv_t3_with_official_harmonizer(
    model_config: dict[str, Any],
    *,
    temporal_capability_path: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    verify_known_hashes: bool = True,
    load_audit_output: str | Path | None = None,
) -> tuple[TemporalMultiViewWrapper, LoadAudit]:
    """Load the official 0.6B model strictly, then attach the 20 T3 residuals."""

    from cosmos_predict2.pipelines.text2image import Text2ImagePipeline
    from .factory import build_official_pipeline_config

    cosmos_dir = Path(model_config["cosmos_checkpoint_dir"]).expanduser().resolve()
    harmonizer = Path(model_config["harmonizer_checkpoint"]).expanduser().resolve()
    if verify_known_hashes:
        require_sha256(harmonizer, OFFICIAL_PATHS.harmonizer_sha256)
        require_sha256(cosmos_dir / "model.pt", OFFICIAL_PATHS.cosmos_dit_sha256)
        require_sha256(
            cosmos_dir / "tokenizer" / "tokenizer.pth",
            OFFICIAL_PATHS.tokenizer_sha256,
        )

    config = build_official_pipeline_config(cosmos_dir, dtype=dtype)
    pipeline = Text2ImagePipeline.from_config(
        config,
        use_text_encoder=False,
        dit_path=config.dit_path,
        device=str(device),
        torch_dtype=dtype,
    )
    pipeline.tensor_kwargs = {"device": str(device), "dtype": dtype}
    require_official_text2image_dit_contract(pipeline.dit)
    audit = load_official_harmonizer_checkpoint(
        pipeline, harmonizer, audit_output=load_audit_output
    )
    pipeline.dit = TemporalLatentCrossViewDiT.upgrade_from_official(
        pipeline.dit,
        camera_adjacency=model_config["camera_adjacency"],
        adjacency_undirected=bool(model_config["undirected"]),
        max_views=int(model_config["max_views"]),
        adapter_dim=int(model_config["cross_view_adapter_dim"]),
        adapter_heads=int(model_config["cross_view_adapter_heads"]),
    )
    wrapper = TemporalMultiViewWrapper(
        pipeline,
        temporal_capability_path=temporal_capability_path,
        timestep=int(model_config["timestep"]),
    )
    return wrapper, audit


def build_b_tmv_t4_with_official_harmonizer(
    model_config: dict[str, Any],
    *,
    temporal_capability_path: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    verify_known_hashes: bool = True,
    load_audit_output: str | Path | None = None,
) -> tuple[TemporalMultiViewWrapper, LoadAudit]:
    """Load the official model, attach T3 residuals, then attach T4 geometry."""

    # The official/T3 construction is intentionally shared so a zero geometry
    # gate is an exact continuation of the production T3 implementation.
    t3_config = dict(model_config)
    t3_config.update({
        "variant": "warm_v1",
        "cross_view_stage": "latent_temporal_residual_T3",
    })
    wrapper, audit = build_b_tmv_t3_with_official_harmonizer(
        t3_config,
        temporal_capability_path=temporal_capability_path,
        device=device,
        dtype=dtype,
        verify_known_hashes=verify_known_hashes,
        load_audit_output=load_audit_output,
    )
    geometry = model_config["geometry"]
    attach_relative_geometry_conditioning(
        wrapper.dit,
        bottleneck_dim=int(geometry["bottleneck_dim"]),
        gate_init=float(geometry["gate_init"]),
        enabled=bool(geometry["enabled"]),
    )
    wrapper.dit.mv_attention_variant = "warm_v1_geometry_t4"
    return wrapper, audit


def build_b_tmv_pose_routing_with_official_harmonizer(
    model_config: dict[str, Any], **kwargs: Any,
) -> tuple[TemporalMultiViewWrapper, LoadAudit]:
    """Official load audit, original T4 FiLM, then parameter-free pose routing."""
    _validate_pose_routing_config(model_config)
    wrapper, audit = build_b_tmv_t4_with_official_harmonizer(model_config, **kwargs)
    attach_pose_routing(wrapper.dit, PoseRoutingConfig.from_dict(model_config["camera_routing"]))
    return wrapper, audit


def build_b_tmv_b1_with_official_harmonizer(
    model_config: dict[str, Any], **kwargs: Any
) -> tuple[TemporalMultiViewWrapper, LoadAudit]:
    """Build audited T4 first and attach zero-gated B1 token geometry last."""

    plucker = model_config["plucker"]
    base = dict(
        model_config,
        variant="warm_v1_geometry_t4",
        cross_view_stage="latent_temporal_residual_T4_geometry_conditioned",
    )
    base.pop("plucker", None)
    wrapper, audit = build_b_tmv_t4_with_official_harmonizer(base, **kwargs)
    attach_b1_plucker_conditioning(
        wrapper.dit,
        hidden_dim=int(plucker["hidden_dim"]),
        gate_init=float(plucker["gate_init"]),
    )
    return wrapper, audit


def build_b_tmv_pose_b1_from_checkpoint(
    model_config: dict[str, Any], pose_checkpoint: str | Path, **kwargs: Any,
) -> tuple[TemporalMultiViewWrapper, LoadAudit, Any]:
    """Strict pose-checkpoint warm start, then zero-gated Plucker attachment.

    The source checkpoint is validated before changing the model structure.
    Its optimizer, progress, and RNG are deliberately not resumed.
    """
    from .state_b_tmv import read_b_tmv_checkpoint, load_b_tmv_residual_weights

    _validate_pose_b1_config(model_config)
    pose_config = dict(model_config, variant=POSE_ROUTING_VARIANT,
                       cross_view_stage="latent_temporal_residual_T4_pose_routed")
    pose_config.pop("plucker", None)
    wrapper, audit = build_b_tmv_pose_routing_with_official_harmonizer(
        pose_config, **kwargs,
    )
    source = read_b_tmv_checkpoint(pose_checkpoint, map_location="cpu")
    progress = load_b_tmv_residual_weights(wrapper.dit, source)
    plucker = model_config["plucker"]
    attach_b1_plucker_conditioning(
        wrapper.dit, hidden_dim=int(plucker["hidden_dim"]),
        gate_init=float(plucker["gate_init"]),
    )
    return wrapper, audit, progress
