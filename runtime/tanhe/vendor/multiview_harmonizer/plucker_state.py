"""Strict baseline-to-Plucker warm starts for A1 and B1.

These functions load extension weights only.  Optimizer, scheduler, RNG and
training progress are deliberately not restored: A1/B1 are new experiments
whose optimizer step starts at zero.  The source progress is returned solely
for provenance.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .constants import CAMERA_ORDER, MODEL_FAMILY, OFFICIAL_PATHS, UPSTREAM_COMMIT
from .state_temporal import (
    ADAPTER_WEIGHT_SCOPE,
    CHECKPOINT_FORMAT as A0_CHECKPOINT_FORMAT,
    TrainingProgress as ATrainingProgress,
    temporal_framework_signature,
)
from .state_b_tmv import (
    EXPECTED_T4_TRAINABLE_PARAMS,
    T4_CHECKPOINT_FORMAT,
    T4_WEIGHT_SCOPE,
    BTMVCheckpointContractError,
    TrainingProgress as BTrainingProgress,
)


class PluckerWarmStartError(RuntimeError):
    pass


def _is_plucker_name(name: str) -> bool:
    return ".plucker_projection." in name or name.endswith(".plucker_gate")


def _progress(payload: Mapping[str, object], cls):
    value = payload.get("progress")
    if not isinstance(value, Mapping) or set(value) != {"global_step", "epoch"}:
        raise PluckerWarmStartError("warm-start progress is malformed")
    step, epoch = value["global_step"], value["epoch"]
    if type(step) is not int or step < 0 or type(epoch) is not int or epoch < 0:
        raise PluckerWarmStartError("warm-start progress values are invalid")
    return cls(step, epoch)


def _check_tensor_schema(
    source: Mapping[str, object], expected: Mapping[str, torch.Tensor], label: str
) -> None:
    if set(source) != set(expected):
        raise PluckerWarmStartError(f"{label} warm-start state keys mismatch")
    for name, target in expected.items():
        value = source[name]
        if not isinstance(value, torch.Tensor):
            raise PluckerWarmStartError(f"{label} warm-start {name} is not a tensor")
        if value.shape != target.shape or value.dtype != target.dtype:
            raise PluckerWarmStartError(
                f"{label} warm-start tensor schema mismatch for {name}"
            )


def load_a0_warm_start_into_a1(
    dit: nn.Module, payload: Mapping[str, object]
) -> ATrainingProgress:
    if getattr(dit, "mv_plucker_variant", None) != "a1_plucker_v1":
        raise PluckerWarmStartError("A0 warm-start target must be A1")
    required = {
        "format", "signature", "progress", "weights", "optimizer",
        "scheduler", "scaler", "rng", "metadata",
    }
    if set(payload) != required or payload.get("format") != A0_CHECKPOINT_FORMAT:
        raise PluckerWarmStartError("warm-start source is not an A0 checkpoint")
    weights = payload.get("weights")
    if not isinstance(weights, Mapping) or weights.get("scope") != ADAPTER_WEIGHT_SCOPE:
        raise PluckerWarmStartError("A0 warm-start has the wrong weight scope")
    adapters = getattr(dit, "mv_adapters", None)
    if not isinstance(adapters, nn.ModuleDict):
        raise PluckerWarmStartError("A1 target has no adapter ModuleDict")
    target_state = adapters.state_dict()
    base_state = {
        name: tensor for name, tensor in target_state.items() if not _is_plucker_name(name)
    }
    source_state = weights.get("state_dict")
    if not isinstance(source_state, Mapping):
        raise PluckerWarmStartError("A0 warm-start state_dict is missing")
    _check_tensor_schema(source_state, base_state, "A0")

    expected_signature = temporal_framework_signature(dit).to_dict()
    expected_signature["adapter_state_schema"] = tuple(
        (name, tuple(tensor.shape), str(tensor.dtype))
        for name, tensor in base_state.items()
    )
    if payload.get("signature") != expected_signature:
        raise PluckerWarmStartError("A0 warm-start structural signature mismatch")

    incompatible = adapters.load_state_dict(source_state, strict=False)
    expected_missing = {name for name in target_state if _is_plucker_name(name)}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise PluckerWarmStartError("A0 warm-start applied an unexpected state scope")
    if any(adapter.plucker_gate.item() != 0 for adapter in adapters.values()):
        raise PluckerWarmStartError("A1 Plucker gates changed during A0 warm-start")
    return _progress(payload, ATrainingProgress)


def _b1_extension_state(dit: nn.Module) -> dict[str, torch.Tensor]:
    residuals = getattr(dit, "tmv_cross_view_residuals", None)
    geometry = getattr(dit, "tmv_geometry_encoder", None)
    if not isinstance(residuals, nn.ModuleList) or not isinstance(geometry, nn.Module):
        raise PluckerWarmStartError("B1 target is missing T4 extension modules")
    state = {
        f"tmv_cross_view_residuals.{name}": tensor
        for name, tensor in residuals.state_dict().items()
    }
    state.update(
        {
            f"tmv_geometry_encoder.{name}": tensor
            for name, tensor in geometry.state_dict().items()
        }
    )
    return state


def _expected_t4_signature(dit: nn.Module, state: Mapping[str, torch.Tensor]):
    residuals = dit.tmv_cross_view_residuals
    first = residuals[0]
    return {
        "architecture": "B_TMV",
        "architecture_version": 1,
        "variant": "warm_v1_geometry_t4",
        "weight_scope": T4_WEIGHT_SCOPE,
        "camera_order": list(CAMERA_ORDER),
        "max_views": int(dit.mv_max_views),
        "block_count": len(dit.blocks),
        "residual_count": len(residuals),
        "model_dim": int(first.model_dim),
        "adapter_dim": int(first.adapter_dim),
        "adapter_heads": int(first.num_heads),
        "cross_view_mode": dit.mv_cross_view_mode,
        "camera_adjacency": [list(row) for row in dit.mv_camera_adjacency],
        "adjacency_undirected": bool(dit.mv_adjacency_undirected),
        "public_layout": "B,T_rgb,V,C,H,W",
        "vae_folding": "B*V_preserve_T",
        "current_frame_index": 0,
        "trainable_params": EXPECTED_T4_TRAINABLE_PARAMS,
        "state_schema": [
            [name, list(tensor.shape), str(tensor.dtype)]
            for name, tensor in state.items()
        ],
        "model_family": MODEL_FAMILY,
        "upstream_commit": UPSTREAM_COMMIT,
        "cosmos_dit_sha256": OFFICIAL_PATHS.cosmos_dit_sha256,
        "harmonizer_sha256": OFFICIAL_PATHS.harmonizer_sha256,
        "tokenizer_sha256": OFFICIAL_PATHS.tokenizer_sha256,
        "geometry_conditioning": True,
        "geometry_mode": getattr(dit, "mv_geometry_mode", None),
        "geometry_feature_dim": int(dit.mv_geometry_feature_dim),
        "geometry_bottleneck_dim": int(dit.mv_geometry_bottleneck_dim),
    }


def load_t4_warm_start_into_b1(
    dit: nn.Module, payload: Mapping[str, object]
) -> BTrainingProgress:
    if getattr(dit, "mv_plucker_variant", None) != (
        "warm_v1_geometry_t4_plucker_b1"
    ):
        raise PluckerWarmStartError("T4 warm-start target must be B1")
    required = {
        "format", "signature", "progress", "weights", "optimizer",
        "scheduler", "scaler", "rng", "metadata",
    }
    if set(payload) != required or payload.get("format") != T4_CHECKPOINT_FORMAT:
        raise PluckerWarmStartError("warm-start source is not a T4 checkpoint")
    weights = payload.get("weights")
    if not isinstance(weights, Mapping) or weights.get("scope") != T4_WEIGHT_SCOPE:
        raise PluckerWarmStartError("T4 warm-start has the wrong weight scope")

    target_state = _b1_extension_state(dit)
    base_state = {
        name: tensor for name, tensor in target_state.items() if not _is_plucker_name(name)
    }
    source_state = weights.get("state_dict")
    if not isinstance(source_state, Mapping):
        raise PluckerWarmStartError("T4 warm-start state_dict is missing")
    _check_tensor_schema(source_state, base_state, "T4")
    if payload.get("signature") != _expected_t4_signature(dit, base_state):
        raise PluckerWarmStartError("T4 warm-start structural signature mismatch")

    residual_prefix = "tmv_cross_view_residuals."
    residual_source = {
        name[len(residual_prefix):]: tensor
        for name, tensor in source_state.items()
        if name.startswith(residual_prefix)
    }
    residual_target = dit.tmv_cross_view_residuals.state_dict()
    incompatible = dit.tmv_cross_view_residuals.load_state_dict(
        residual_source, strict=False
    )
    expected_missing = {name for name in residual_target if _is_plucker_name(name)}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise PluckerWarmStartError("T4 warm-start applied an unexpected residual scope")

    geometry_prefix = "tmv_geometry_encoder."
    geometry_source = {
        name[len(geometry_prefix):]: tensor
        for name, tensor in source_state.items()
        if name.startswith(geometry_prefix)
    }
    dit.tmv_geometry_encoder.load_state_dict(geometry_source, strict=True)
    if any(
        residual.plucker_gate.item() != 0
        for residual in dit.tmv_cross_view_residuals
    ):
        raise PluckerWarmStartError("B1 Plucker gates changed during T4 warm-start")
    return _progress(payload, BTrainingProgress)

