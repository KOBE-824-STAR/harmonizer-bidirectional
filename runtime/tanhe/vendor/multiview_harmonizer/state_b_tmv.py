"""Strict, isolated checkpoint/resume contracts for Scheme B-TMV T3 and T4."""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .constants import CAMERA_ORDER, MODEL_FAMILY, OFFICIAL_PATHS, UPSTREAM_COMMIT
from .b_tmv_routing import POSE_ROUTING_VARIANT
from .b_tmv_crossview import POSE_B1_VARIANT


CHECKPOINT_FORMAT = "diffusionharmonizer_b_tmv_t3_v1"
WEIGHT_SCOPE = "tmv_cross_view_residuals_only"
EXPECTED_TRAINABLE_PARAMS = 9_798_420
T4_CHECKPOINT_FORMAT = "diffusionharmonizer_b_tmv_t4_geometry_v1"
B1_CHECKPOINT_FORMAT = "diffusionharmonizer_b_tmv_t4_plucker_b1_v1"
POSE_ROUTING_CHECKPOINT_FORMAT = "diffusionharmonizer_b_tmv_t4_pose_routing_v1"
POSE_B1_CHECKPOINT_FORMAT = "diffusionharmonizer_b_tmv_t4_pose_routing_plucker_b1_v1"
T4_WEIGHT_SCOPE = "tmv_cross_view_residuals_and_geometry_encoder_only"
B1_WEIGHT_SCOPE = "tmv_cross_view_residuals_geometry_encoder_plucker_only"
EXPECTED_T4_TRAINABLE_PARAMS = 10_219_944
EXPECTED_B1_TRAINABLE_PARAMS = 10_644_924


class BTMVCheckpointContractError(RuntimeError):
    """Raised before incompatible checkpoint state is applied."""


@dataclass(frozen=True)
class TrainingProgress:
    global_step: int
    epoch: int


def _residuals(dit: nn.Module) -> nn.ModuleList:
    value = getattr(dit, "tmv_cross_view_residuals", None)
    if not isinstance(value, nn.ModuleList) or not value:
        raise BTMVCheckpointContractError("DiT has no B-TMV T3 residual ModuleList")
    return value


def _is_t4(dit: nn.Module) -> bool:
    return bool(getattr(dit, "mv_geometry_conditioning_attached", False))


def _is_b1(dit: nn.Module) -> bool:
    return getattr(dit, "mv_plucker_variant", None) in (
        "warm_v1_geometry_t4_plucker_b1", POSE_B1_VARIANT,
    )


def _weight_scope(dit: nn.Module) -> str:
    if _is_b1(dit):
        return B1_WEIGHT_SCOPE
    return T4_WEIGHT_SCOPE if _is_t4(dit) else WEIGHT_SCOPE


def _checkpoint_format(dit: nn.Module) -> str:
    if _is_b1(dit):
        return (POSE_B1_CHECKPOINT_FORMAT if getattr(dit, "mv_pose_routing_config", None)
                is not None else B1_CHECKPOINT_FORMAT)
    if getattr(dit, "mv_pose_routing_config", None) is not None:
        return POSE_ROUTING_CHECKPOINT_FORMAT
    return T4_CHECKPOINT_FORMAT if _is_t4(dit) else CHECKPOINT_FORMAT


def _geometry_encoder(dit: nn.Module) -> nn.Module:
    value = getattr(dit, "tmv_geometry_encoder", None)
    if not isinstance(value, nn.Module):
        raise BTMVCheckpointContractError("DiT has no B-TMV T4 geometry encoder")
    return value


def trainable_parameters(dit: nn.Module) -> list[nn.Parameter]:
    parameters = list(_residuals(dit).parameters())
    if _is_t4(dit):
        parameters.extend(_geometry_encoder(dit).parameters())
    return parameters


def _extension_state_dict(dit: nn.Module) -> dict[str, torch.Tensor]:
    if not _is_t4(dit):
        return dict(_residuals(dit).state_dict())
    state = {
        f"tmv_cross_view_residuals.{name}": tensor
        for name, tensor in _residuals(dit).state_dict().items()
    }
    if _is_t4(dit):
        state.update({
            f"tmv_geometry_encoder.{name}": tensor
            for name, tensor in _geometry_encoder(dit).state_dict().items()
        })
    return state


def _t3_state_dict_from_t4(dit: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for name, tensor in _residuals(dit).state_dict().items()
        if ".geometry_projection." not in name and not name.endswith(".geometry_gate")
    }


def _load_extension_state_dict(dit: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    if not _is_t4(dit):
        _residuals(dit).load_state_dict(state, strict=True)
        return
    residual_prefix = "tmv_cross_view_residuals."
    residual_state = {
        name[len(residual_prefix):]: tensor
        for name, tensor in state.items()
        if name.startswith(residual_prefix)
    }
    _residuals(dit).load_state_dict(residual_state, strict=True)
    if _is_t4(dit):
        geometry_prefix = "tmv_geometry_encoder."
        geometry_state = {
            name[len(geometry_prefix):]: tensor
            for name, tensor in state.items()
            if name.startswith(geometry_prefix)
        }
        _geometry_encoder(dit).load_state_dict(geometry_state, strict=True)


def framework_signature(dit: nn.Module) -> dict[str, Any]:
    residuals = _residuals(dit)
    first = residuals[0]
    state = _extension_state_dict(dit)
    t4 = _is_t4(dit)
    if len(residuals) != len(dit.blocks) or len(residuals) != 20:
        raise BTMVCheckpointContractError("B-TMV T3 requires one residual after each of 20 blocks")
    if getattr(dit, "mv_architecture", None) != "B_TMV":
        raise BTMVCheckpointContractError("Invalid B-TMV architecture marker")
    b1 = _is_b1(dit)
    expected_variant = (
        "warm_v1_geometry_t4_plucker_b1"
        if b1 else ("warm_v1_geometry_t4" if t4 else "warm_v1")
    )
    routing = getattr(dit, "mv_pose_routing_config", None)
    if routing is not None:
        if not t4:
            raise BTMVCheckpointContractError("pose routing requires T4 geometry")
        expected_variant = POSE_B1_VARIANT if b1 else POSE_ROUTING_VARIANT
    if getattr(dit, "mv_attention_variant", None) != expected_variant:
        raise BTMVCheckpointContractError("Invalid B-TMV warm variant marker")
    if getattr(dit, "mv_cross_view_mode", None) != (
        "same_latent_time_same_spatial_token_view_only"
    ):
        raise BTMVCheckpointContractError("Invalid B-TMV cross-view mode")
    if getattr(dit, "mv_max_views", None) != len(CAMERA_ORDER):
        raise BTMVCheckpointContractError("Invalid B-TMV camera cardinality")
    trainable = sum(parameter.numel() for parameter in trainable_parameters(dit))
    expected_trainable = (
        EXPECTED_B1_TRAINABLE_PARAMS
        if b1 else (EXPECTED_T4_TRAINABLE_PARAMS if t4 else EXPECTED_TRAINABLE_PARAMS)
    )
    if trainable != expected_trainable:
        raise BTMVCheckpointContractError(
            f"B-TMV extension count {trainable} != {expected_trainable}"
        )
    signature = {
        "architecture": "B_TMV",
        "architecture_version": 1,
        "variant": expected_variant,
        "weight_scope": _weight_scope(dit),
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
        "trainable_params": trainable,
        "state_schema": [
            [name, list(tensor.shape), str(tensor.dtype)] for name, tensor in state.items()
        ],
        "model_family": MODEL_FAMILY,
        "upstream_commit": UPSTREAM_COMMIT,
        "cosmos_dit_sha256": OFFICIAL_PATHS.cosmos_dit_sha256,
        "harmonizer_sha256": OFFICIAL_PATHS.harmonizer_sha256,
        "tokenizer_sha256": OFFICIAL_PATHS.tokenizer_sha256,
    }
    if t4:
        signature.update({
            "geometry_conditioning": True,
            "geometry_mode": getattr(dit, "mv_geometry_mode", None),
            "geometry_feature_dim": int(dit.mv_geometry_feature_dim),
            "geometry_bottleneck_dim": int(dit.mv_geometry_bottleneck_dim),
        })
    if b1:
        if not bool(getattr(dit, "mv_plucker_conditioning_enabled", False)):
            raise BTMVCheckpointContractError("B1 Plucker conditioning is not enabled")
        signature.update({
            "plucker_conditioning": True,
            "plucker_dimension": int(dit.mv_plucker_dim),
            "plucker_hidden_dim": int(dit.mv_plucker_hidden_dim),
            "plucker_gate_initialization": float(dit.mv_plucker_gate_initialization),
            "plucker_convention": str(dit.mv_plucker_convention),
            "plucker_extrinsic_convention": str(
                dit.mv_geometry_extrinsic_convention
            ),
            "plucker_value_conditioning": False,
        })
    if routing is not None:
        signature.update({
            "camera_routing": routing.to_dict(),
            "camera_adjacency_role": "t4_warm_start_provenance_only",
            "runtime_adjacency_layout": "B,V_dst,V_src",
        })
    return signature


def validate_optimizer_scope(dit: nn.Module, optimizer: object | None) -> None:
    if optimizer is None:
        return
    expected = trainable_parameters(dit)
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list):
        raise BTMVCheckpointContractError("optimizer has no inspectable param_groups")
    actual = [parameter for group in groups for parameter in group.get("params", ())]
    if len({id(parameter) for parameter in actual}) != len(actual):
        raise BTMVCheckpointContractError("optimizer contains duplicate parameters")
    if [id(parameter) for parameter in actual] != [id(parameter) for parameter in expected]:
        raise BTMVCheckpointContractError(
            "optimizer must contain every and only B-TMV residual parameter in registered order"
        )
    frozen = {id(parameter) for parameter in dit.parameters() if not parameter.requires_grad}
    if frozen & {id(parameter) for parameter in actual}:
        raise BTMVCheckpointContractError("frozen parameter intersects optimizer parameters")


def _class_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _component(value: object | None) -> dict[str, Any] | None:
    if value is None:
        return None
    state_dict = getattr(value, "state_dict", None)
    if not callable(state_dict):
        raise BTMVCheckpointContractError(f"{_class_name(value)} has no state_dict")
    return {"class": _class_name(value), "state_dict": copy.deepcopy(state_dict())}


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": {
            "algorithm": numpy_state[0],
            # Older PyTorch releases cannot construct tensors from NumPy
            # uint32 arrays.  Store a lossless int64 representation and cast
            # explicitly back to uint32 in restore_rng_state below.
            "keys": torch.as_tensor(
                numpy_state[1].astype("int64", copy=True)
            ),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [item.cpu().clone() for item in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized()
        else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    try:
        numpy_state = state["numpy"]
        random.setstate(copy.deepcopy(state["python"]))
        np.random.set_state(
            (
                numpy_state["algorithm"],
                numpy_state["keys"].cpu().numpy().astype("uint32", copy=True),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(state["torch_cpu"].cpu())
        cuda_states = list(state["torch_cuda"])
        if cuda_states:
            if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
                raise ValueError("CUDA RNG device count mismatch")
            torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise BTMVCheckpointContractError("Malformed B-TMV RNG state") from error


def make_checkpoint_payload(
    dit: nn.Module,
    *,
    global_step: int,
    epoch: int,
    optimizer: object | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if type(global_step) is not int or global_step < 0 or type(epoch) is not int or epoch < 0:
        raise ValueError("global_step and epoch must be non-negative integers")
    validate_optimizer_scope(dit, optimizer)
    if scheduler is not None and optimizer is not None:
        if getattr(scheduler, "optimizer", optimizer) is not optimizer:
            raise BTMVCheckpointContractError("scheduler is attached to another optimizer")
    t4 = _is_t4(dit)
    return {
        "format": _checkpoint_format(dit),
        "signature": framework_signature(dit),
        "progress": {"global_step": global_step, "epoch": epoch},
        "weights": {
            "scope": _weight_scope(dit),
            "state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in _extension_state_dict(dit).items()
            },
        },
        "optimizer": _component(optimizer),
        "scheduler": _component(scheduler),
        "scaler": _component(scaler),
        "rng": capture_rng_state(),
        "metadata": {} if metadata is None else copy.deepcopy(dict(metadata)),
    }


def _validate_component(name: str, envelope: Any, target: object | None) -> None:
    if envelope is None:
        if target is not None:
            raise BTMVCheckpointContractError(f"checkpoint has no {name} state")
        return
    if not isinstance(envelope, Mapping) or set(envelope) != {"class", "state_dict"}:
        raise BTMVCheckpointContractError(f"malformed {name} envelope")
    if target is None or envelope["class"] != _class_name(target):
        raise BTMVCheckpointContractError(f"{name} target/class mismatch")
    if not isinstance(envelope["state_dict"], Mapping):
        raise BTMVCheckpointContractError(f"malformed {name} state_dict")


def validate_payload(
    dit: nn.Module,
    payload: Mapping[str, Any],
    *,
    optimizer: object | None,
    scheduler: object | None,
    scaler: object | None,
) -> TrainingProgress:
    required = {
        "format", "signature", "progress", "weights", "optimizer", "scheduler",
        "scaler", "rng", "metadata",
    }
    expected_format = _checkpoint_format(dit)
    expected_scope = _weight_scope(dit)
    if set(payload) != required or payload.get("format") != expected_format:
        raise BTMVCheckpointContractError("B-TMV checkpoint fields/format mismatch")
    if payload["signature"] != framework_signature(dit):
        raise BTMVCheckpointContractError("B-TMV checkpoint structural signature mismatch")
    progress = payload["progress"]
    if not isinstance(progress, Mapping) or set(progress) != {"global_step", "epoch"}:
        raise BTMVCheckpointContractError("B-TMV progress fields mismatch")
    if any(type(progress[key]) is not int or progress[key] < 0 for key in progress):
        raise BTMVCheckpointContractError("B-TMV progress values are invalid")
    weights = payload["weights"]
    if not isinstance(weights, Mapping) or weights.get("scope") != expected_scope:
        raise BTMVCheckpointContractError("checkpoint B-TMV extension scope mismatch")
    state = weights.get("state_dict")
    expected = _extension_state_dict(dit)
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise BTMVCheckpointContractError("B-TMV residual state keys mismatch")
    bad: dict[str, Any] = {}
    for name, expected_tensor in expected.items():
        actual = state[name]
        if not isinstance(actual, torch.Tensor):
            bad[name] = "not a tensor"
        elif actual.shape != expected_tensor.shape or actual.dtype != expected_tensor.dtype:
            bad[name] = {
                "expected": (tuple(expected_tensor.shape), str(expected_tensor.dtype)),
                "actual": (tuple(actual.shape), str(actual.dtype)),
            }
    if bad:
        raise BTMVCheckpointContractError(f"B-TMV residual tensor schema mismatch: {bad}")
    _validate_component("optimizer", payload["optimizer"], optimizer)
    _validate_component("scheduler", payload["scheduler"], scheduler)
    _validate_component("scaler", payload["scaler"], scaler)
    validate_optimizer_scope(dit, optimizer)
    if not isinstance(payload["rng"], Mapping) or not isinstance(payload["metadata"], Mapping):
        raise BTMVCheckpointContractError("B-TMV RNG/metadata is malformed")
    return TrainingProgress(int(progress["global_step"]), int(progress["epoch"]))


def save_b_tmv_checkpoint(
    path: str | os.PathLike[str],
    dit: nn.Module,
    *,
    global_step: int,
    epoch: int,
    optimizer: object | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = make_checkpoint_payload(
        dit,
        global_step=global_step,
        epoch=epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        metadata=metadata,
    )
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def read_b_tmv_checkpoint(
    path: str | os.PathLike[str], *, map_location: str | torch.device = "cpu"
) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise BTMVCheckpointContractError("checkpoint root must be a mapping")
    return payload


def load_b_tmv_checkpoint(
    path: str | os.PathLike[str],
    dit: nn.Module,
    *,
    optimizer: object | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> TrainingProgress:
    payload = read_b_tmv_checkpoint(path, map_location=map_location)
    progress = validate_payload(
        dit, payload, optimizer=optimizer, scheduler=scheduler, scaler=scaler
    )
    _load_extension_state_dict(dit, payload["weights"]["state_dict"])
    for name, target in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if payload[name] is not None:
            target.load_state_dict(copy.deepcopy(dict(payload[name]["state_dict"])))
    if restore_rng:
        restore_rng_state(payload["rng"])
    return progress


def load_b_tmv_residual_weights(
    dit: nn.Module,
    payload: Mapping[str, Any],
) -> TrainingProgress:
    """Strictly validate and load B-TMV residual weights for inference.

    Training checkpoints intentionally contain optimizer, scheduler, scaler,
    and RNG envelopes.  Inference must validate that complete checkpoint
    structure without constructing or mutating training-state objects.
    """

    progress = _validate_residual_weight_payload(
        dit, payload, expected_format=_checkpoint_format(dit),
        expected_signature=framework_signature(dit),
    )
    _load_extension_state_dict(dit, payload["weights"]["state_dict"])
    return progress


def _validate_residual_weight_payload(
    dit: nn.Module, payload: Mapping[str, Any], *,
    expected_format: str, expected_signature: dict[str, Any],
) -> TrainingProgress:
    required = {
        "format", "signature", "progress", "weights", "optimizer", "scheduler",
        "scaler", "rng", "metadata",
    }
    expected_scope = _weight_scope(dit)
    if set(payload) != required or payload.get("format") != expected_format:
        raise BTMVCheckpointContractError("B-TMV checkpoint fields/format mismatch")
    if payload["signature"] != expected_signature:
        raise BTMVCheckpointContractError("B-TMV checkpoint structural signature mismatch")

    progress = payload["progress"]
    if not isinstance(progress, Mapping) or set(progress) != {"global_step", "epoch"}:
        raise BTMVCheckpointContractError("B-TMV progress fields mismatch")
    if any(type(progress[key]) is not int or progress[key] < 0 for key in progress):
        raise BTMVCheckpointContractError("B-TMV progress values are invalid")

    weights = payload["weights"]
    if not isinstance(weights, Mapping) or weights.get("scope") != expected_scope:
        raise BTMVCheckpointContractError("checkpoint B-TMV extension scope mismatch")
    state = weights.get("state_dict")
    expected = _extension_state_dict(dit)
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise BTMVCheckpointContractError("B-TMV residual state keys mismatch")
    bad: dict[str, Any] = {}
    for name, expected_tensor in expected.items():
        actual = state[name]
        if not isinstance(actual, torch.Tensor):
            bad[name] = "not a tensor"
        elif actual.shape != expected_tensor.shape or actual.dtype != expected_tensor.dtype:
            bad[name] = {
                "expected": (tuple(expected_tensor.shape), str(expected_tensor.dtype)),
                "actual": (tuple(actual.shape), str(actual.dtype)),
            }
    if bad:
        raise BTMVCheckpointContractError(f"B-TMV residual tensor schema mismatch: {bad}")

    for name in ("optimizer", "scheduler", "scaler"):
        envelope = payload[name]
        if envelope is not None and (
            not isinstance(envelope, Mapping)
            or set(envelope) != {"class", "state_dict"}
            or not isinstance(envelope["class"], str)
            or not isinstance(envelope["state_dict"], Mapping)
        ):
            raise BTMVCheckpointContractError(f"malformed {name} envelope")
    if not isinstance(payload["rng"], Mapping) or not isinstance(payload["metadata"], Mapping):
        raise BTMVCheckpointContractError("B-TMV RNG/metadata is malformed")

    return TrainingProgress(int(progress["global_step"]), int(progress["epoch"]))


def load_t4_warm_start_into_pose_routing(
    dit: nn.Module, payload: Mapping[str, Any],
) -> TrainingProgress:
    """Explicit weights-only migration; never restore T4 optimizer/progress/RNG.

    Validate the complete original T4 signature including its provenance graph,
    keys, tensor shapes and dtypes before applying any weight. Return source
    progress only for logging; the caller starts the new experiment at step 0.
    """
    if getattr(dit, "mv_pose_routing_config", None) is None:
        raise BTMVCheckpointContractError("T4 warm-start target must use pose routing")
    expected = framework_signature(dit)
    expected["variant"] = "warm_v1_geometry_t4"
    for key in ("camera_routing", "camera_adjacency_role", "runtime_adjacency_layout"):
        expected.pop(key)
    progress = _validate_residual_weight_payload(
        dit, payload, expected_format=T4_CHECKPOINT_FORMAT, expected_signature=expected,
    )
    _load_extension_state_dict(dit, payload["weights"]["state_dict"])
    return progress


def load_t3_warm_start_into_t4(
    dit: nn.Module,
    payload: Mapping[str, Any],
) -> TrainingProgress:
    """Load only trained T3 residual weights into a freshly attached T4 model."""

    if not _is_t4(dit):
        raise BTMVCheckpointContractError("T3 warm-start target must be a T4 model")
    required = {
        "format", "signature", "progress", "weights", "optimizer", "scheduler",
        "scaler", "rng", "metadata",
    }
    if set(payload) != required or payload.get("format") != CHECKPOINT_FORMAT:
        raise BTMVCheckpointContractError("warm-start source is not a T3 checkpoint")
    if payload.get("weights", {}).get("scope") != WEIGHT_SCOPE:
        raise BTMVCheckpointContractError("warm-start source has the wrong T3 scope")
    expected_signature = framework_signature(dit)
    expected_signature["variant"] = "warm_v1"
    expected_signature["weight_scope"] = WEIGHT_SCOPE
    expected_signature["trainable_params"] = EXPECTED_TRAINABLE_PARAMS
    expected_signature["state_schema"] = [
        [name, list(tensor.shape), str(tensor.dtype)]
        for name, tensor in _t3_state_dict_from_t4(dit).items()
    ]
    for key in (
        "geometry_conditioning", "geometry_mode", "geometry_feature_dim",
        "geometry_bottleneck_dim",
    ):
        expected_signature.pop(key, None)
    if payload.get("signature") != expected_signature:
        raise BTMVCheckpointContractError(
            "T3 warm-start structural signature does not match the T4 base"
        )
    source = payload["weights"].get("state_dict")
    expected = _t3_state_dict_from_t4(dit)
    if not isinstance(source, Mapping) or set(source) != set(expected):
        raise BTMVCheckpointContractError("T3 warm-start residual keys mismatch")
    for name, target in expected.items():
        value = source[name]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype:
            raise BTMVCheckpointContractError(
                f"T3 warm-start tensor schema mismatch for {name}"
            )
    incompatible = _residuals(dit).load_state_dict(source, strict=False)
    expected_missing = {
        name for name in _residuals(dit).state_dict()
        if ".geometry_projection." in name or name.endswith(".geometry_gate")
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise BTMVCheckpointContractError("T3 warm-start applied an unexpected state scope")
    progress = payload.get("progress")
    if not isinstance(progress, Mapping) or set(progress) != {"global_step", "epoch"}:
        raise BTMVCheckpointContractError("T3 warm-start progress is malformed")
    return TrainingProgress(int(progress["global_step"]), int(progress["epoch"]))
