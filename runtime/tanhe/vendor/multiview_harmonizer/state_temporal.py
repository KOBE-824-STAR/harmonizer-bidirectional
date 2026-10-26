"""Strict, adapter-only checkpoint contract for Scheme A-TMV.

The official DiffusionHarmonizer/Cosmos weights remain external.  A payload
created here contains only ``dit.mv_adapters`` plus the state required to
resume training deterministically.  Constructing a payload does not write a
file; :func:`save_temporal_checkpoint` is the explicit persistence boundary.
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .adapters import canonical_camera_adjacency
from .constants import CAMERA_ORDER, MODEL_FAMILY, OFFICIAL_PATHS, UPSTREAM_COMMIT


CHECKPOINT_FORMAT = "diffusionharmonizer_scheme_a_tmv_v1"
A1_CHECKPOINT_FORMAT = "diffusionharmonizer_scheme_a_tmv_a1_plucker_v1"
A0_POSE_CHECKPOINT_FORMAT = "diffusionharmonizer_scheme_a_tmv_a0_pose_routing_v1"
A1_POSE_CHECKPOINT_FORMAT = "diffusionharmonizer_scheme_a_tmv_a1_pose_routing_v1"
TMV_ARCHITECTURE = "A-TMV"
ADAPTER_WEIGHT_SCOPE = "mv_adapters_only"
A1_ADAPTER_WEIGHT_SCOPE = "mv_adapters_a1_plucker_only"
EXPECTED_A0_TRAINABLE_PARAMS = 4_938_244
EXPECTED_A1_TRAINABLE_PARAMS = 5_023_240


class TemporalCheckpointContractError(RuntimeError):
    """Raised before incompatible or incomplete checkpoint state is applied."""


@dataclass(frozen=True)
class TemporalStackPolicy:
    """Resume-critical interpretation of the public ``[B,T,V,C,H,W]`` stack.

    Upstream builds a temporal sequence as the current source first, followed
    by prior generated outputs in ``offset_list`` order.  Its default offsets
    are ``(-1, -2, -3, -4)``.  Until every requested offset exists, upstream
    uses the current frame alone; output slot zero is the public result.
    """

    policy_name: str = "official_current_then_autoregressive_history"
    history_offsets: tuple[int, ...] = (-1, -2, -3, -4)
    cold_start_policy: str = "current_only_until_all_offsets_available"
    current_frame_index: int = 0
    output_frame_index: int = 0
    cold_start_slot_offsets: tuple[int, ...] = (0,)
    tokenizer_temporal_compression_factor: int = 4
    full_stack_length_modulus: int = 4
    full_stack_length_remainder: int = 1
    camera_folding_policy: str = "fold_B_and_V_preserve_T"
    adapter_temporal_scope: str = "same_latent_temporal_slot_only"
    rope_temporal_policy: str = "official_real_temporal_positions_per_camera"
    conditioning_expansion_policy: str = "expand_B_to_BxV_once_across_cameras"

    def __post_init__(self) -> None:
        if not self.policy_name.strip():
            raise ValueError("Temporal policy_name cannot be empty")
        if not self.history_offsets:
            raise ValueError("At least one official history offset is required")
        if any(type(offset) is not int or offset >= 0 for offset in self.history_offsets):
            raise ValueError("All history offsets must be distinct negative integers")
        if len(set(self.history_offsets)) != len(self.history_offsets):
            raise ValueError("All history offsets must be distinct negative integers")
        if self.current_frame_index != 0 or self.output_frame_index != 0:
            raise ValueError("Official current input/output semantics require temporal slot zero")
        if self.cold_start_slot_offsets != (0,):
            raise ValueError("Official cold start contains the current slot only")
        if self.tokenizer_temporal_compression_factor != 4:
            raise ValueError("Official causal video tokenizer temporal compression is four")
        if self.full_stack_length_modulus < 1:
            raise ValueError("full_stack_length_modulus must be positive")
        if not 0 <= self.full_stack_length_remainder < self.full_stack_length_modulus:
            raise ValueError("Invalid full-stack temporal-length remainder")
        if len(self.frame_order) % self.full_stack_length_modulus != self.full_stack_length_remainder:
            raise ValueError(
                "The configured full temporal stack violates its tokenizer length policy"
            )

    @property
    def frame_order(self) -> tuple[str, ...]:
        return ("current",) + tuple(f"history[{offset}]" for offset in self.history_offsets)

    @property
    def rgb_slot_offsets(self) -> tuple[int, ...]:
        return (0,) + self.history_offsets

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frame_order"] = self.frame_order
        payload["rgb_slot_offsets"] = self.rgb_slot_offsets
        payload["official_full_stack_num_frames"] = len(self.frame_order)
        return payload


DEFAULT_TEMPORAL_POLICY = TemporalStackPolicy()


@dataclass(frozen=True)
class TemporalFrameworkSignature:
    """Everything that must agree before A-TMV adapter weights are loaded."""

    architecture: str
    architecture_version: int
    camera_order: tuple[str, ...]
    max_views: int
    adapter_after_blocks: tuple[int, ...]
    adapter_count: int
    adapter_model_dim: int
    adapter_dim: int
    adapter_heads: int
    adapter_pool_factor: int
    adapter_neighbor_only: bool
    adapter_state_schema: tuple[tuple[str, tuple[int, ...], str], ...]
    live_temporal_offsets: tuple[int, ...]
    live_current_output_index: int
    live_temporal_adapter_mode: str
    temporal_policy: dict[str, Any]
    position_encoding_mode: str
    attention_mask_mode: str
    model_family: str
    upstream_commit: str
    cosmos_dit_sha256: str
    harmonizer_sha256: str
    tokenizer_sha256: str
    runtime_contract: dict[str, int | None]
    weight_scope: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingProgress:
    """Counters restored independently from model weights."""

    global_step: int
    epoch: int


def _class_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _require_tmv_adapters(dit: nn.Module) -> nn.Module:
    if getattr(dit, "mv_architecture", None) != TMV_ARCHITECTURE:
        raise TemporalCheckpointContractError(
            f"Expected dit.mv_architecture={TMV_ARCHITECTURE!r}, "
            f"got {getattr(dit, 'mv_architecture', None)!r}"
        )
    adapters = getattr(dit, "mv_adapters", None)
    if not isinstance(adapters, nn.Module) or not dict(adapters.named_children()):
        raise TemporalCheckpointContractError("Scheme A-TMV has no registered mv_adapters")
    return adapters


def _integer_attribute(module: object, name: str) -> int:
    value = getattr(module, name, None)
    if type(value) is not int:
        raise TemporalCheckpointContractError(
            f"Adapter attribute {name!r} must be an integer, got {value!r}"
        )
    return value


def _adapter_structure(adapter: nn.Module) -> tuple[int, int, int, int, bool]:
    attention = getattr(adapter, "attention", None)
    heads = getattr(attention, "num_heads", None)
    if type(heads) is not int:
        raise TemporalCheckpointContractError("Adapter attention.num_heads is missing")
    neighbor_only = getattr(adapter, "neighbor_only", None)
    if type(neighbor_only) is not bool:
        raise TemporalCheckpointContractError("Adapter neighbor_only must be boolean")
    return (
        _integer_attribute(adapter, "model_dim"),
        _integer_attribute(adapter, "adapter_dim"),
        heads,
        _integer_attribute(adapter, "pool_factor"),
        neighbor_only,
    )


def _runtime_contract(dit: nn.Module) -> dict[str, int | None]:
    """Record the live values covered by the official runtime audit.

    Toy CPU modules may omit fields, represented explicitly as ``None``.  The
    production A-TMV DiT supplies all of them, and strict resume compares the
    recorded live values rather than silently substituting defaults.
    """

    fields = (
        "model_channels",
        "num_blocks",
        "num_heads",
        "patch_spatial",
        "patch_temporal",
        "in_channels",
        "out_channels",
    )
    result: dict[str, int | None] = {}
    for name in fields:
        value = getattr(dit, name, None)
        result[name] = int(value) if type(value) is int else None
    blocks = getattr(dit, "blocks", None)
    result["len_blocks"] = len(blocks) if blocks is not None else None
    return result


def temporal_framework_signature(
    dit: nn.Module,
    *,
    temporal_policy: TemporalStackPolicy = DEFAULT_TEMPORAL_POLICY,
    camera_order: Sequence[str] = CAMERA_ORDER,
) -> TemporalFrameworkSignature:
    """Build the strict structural signature for an upgraded A-TMV DiT."""

    adapters = _require_tmv_adapters(dit)
    if not isinstance(temporal_policy, TemporalStackPolicy):
        raise TypeError("temporal_policy must be a TemporalStackPolicy")

    live_offsets = getattr(dit, "mv_temporal_offsets", None)
    if live_offsets is None or tuple(live_offsets) != temporal_policy.rgb_slot_offsets:
        raise TemporalCheckpointContractError(
            "dit.mv_temporal_offsets disagrees with the checkpoint temporal policy: "
            f"live={live_offsets!r}, expected={temporal_policy.rgb_slot_offsets}"
        )
    live_output_index = getattr(dit, "mv_current_output_index", None)
    if live_output_index != temporal_policy.output_frame_index:
        raise TemporalCheckpointContractError(
            "dit.mv_current_output_index disagrees with official output semantics"
        )
    live_adapter_mode = getattr(dit, "mv_temporal_adapter_mode", None)
    if live_adapter_mode != "same_latent_time_only":
        raise TemporalCheckpointContractError(
            "dit.mv_temporal_adapter_mode must be 'same_latent_time_only'"
        )

    canonical_order = tuple(camera_order)
    if not canonical_order or any(not isinstance(name, str) or not name for name in canonical_order):
        raise TemporalCheckpointContractError("camera_order must contain non-empty names")
    if len(set(canonical_order)) != len(canonical_order):
        raise TemporalCheckpointContractError("camera_order contains duplicate semantic cameras")
    if canonical_order != CAMERA_ORDER:
        raise TemporalCheckpointContractError(
            f"Scheme A-TMV camera IDs must use canonical order {CAMERA_ORDER}"
        )

    max_views = getattr(dit, "mv_max_views", None)
    if type(max_views) is not int or max_views != len(canonical_order):
        raise TemporalCheckpointContractError(
            "dit.mv_max_views must equal the semantic camera-order length: "
            f"max_views={max_views!r}, camera_order={len(canonical_order)}"
        )

    raw_layers = getattr(dit, "mv_adapter_after_blocks", None)
    if raw_layers is None:
        raise TemporalCheckpointContractError("DiT has no mv_adapter_after_blocks")
    layers = tuple(int(index) for index in raw_layers)
    if not layers or layers != tuple(sorted(set(layers))) or any(index < 0 for index in layers):
        raise TemporalCheckpointContractError(
            f"Adapter insertion layers must be non-negative, sorted, and unique: {layers}"
        )

    named_adapters = dict(adapters.named_children())
    expected_names = tuple(f"block_{index:02d}" for index in layers)
    if tuple(named_adapters) != expected_names:
        raise TemporalCheckpointContractError(
            "mv_adapters keys do not match insertion layers: "
            f"expected={expected_names}, actual={tuple(named_adapters)}"
        )

    first_structure = _adapter_structure(next(iter(named_adapters.values())))
    inconsistent = {
        name: _adapter_structure(adapter)
        for name, adapter in named_adapters.items()
        if _adapter_structure(adapter) != first_structure
    }
    if inconsistent:
        raise TemporalCheckpointContractError(
            f"All A-TMV adapters must share one structure: {inconsistent}"
        )
    model_dim, adapter_dim, adapter_heads, pool_factor, neighbor_only = first_structure
    expected_adjacency = canonical_camera_adjacency(max_views)
    for name, adapter in named_adapters.items():
        adapter_max_views = getattr(adapter, "max_views", max_views)
        if adapter_max_views != max_views:
            raise TemporalCheckpointContractError(
                f"Adapter {name} max_views={adapter_max_views} disagrees with {max_views}"
            )
        live_adjacency = getattr(adapter, "camera_adjacency", None)
        if (
            not isinstance(live_adjacency, torch.Tensor)
            or live_adjacency.dtype != torch.bool
            or tuple(live_adjacency.shape) != tuple(expected_adjacency.shape)
            or not torch.equal(live_adjacency.detach().cpu(), expected_adjacency)
        ):
            raise TemporalCheckpointContractError(
                f"Adapter {name} camera_adjacency is not the canonical nuScenes ring"
            )

    state_schema = tuple(
        (name, tuple(tensor.shape), str(tensor.dtype))
        for name, tensor in adapters.state_dict().items()
    )
    if not state_schema:
        raise TemporalCheckpointContractError("mv_adapters state_dict is empty")

    return TemporalFrameworkSignature(
        architecture=TMV_ARCHITECTURE,
        architecture_version=1,
        camera_order=canonical_order,
        max_views=max_views,
        adapter_after_blocks=layers,
        adapter_count=len(named_adapters),
        adapter_model_dim=model_dim,
        adapter_dim=adapter_dim,
        adapter_heads=adapter_heads,
        adapter_pool_factor=pool_factor,
        adapter_neighbor_only=neighbor_only,
        adapter_state_schema=state_schema,
        live_temporal_offsets=tuple(live_offsets),
        live_current_output_index=int(live_output_index),
        live_temporal_adapter_mode=live_adapter_mode,
        temporal_policy=temporal_policy.to_dict(),
        position_encoding_mode="official_3d_rope_real_time_per_camera",
        attention_mask_mode=("pooled_pose_overlap_same_time" if
                             getattr(dit, "mv_pose_routing_config", None) is not None
                             else "pooled_semantic_neighbor_same_time"),
        model_family=MODEL_FAMILY,
        upstream_commit=UPSTREAM_COMMIT,
        cosmos_dit_sha256=OFFICIAL_PATHS.cosmos_dit_sha256,
        harmonizer_sha256=OFFICIAL_PATHS.harmonizer_sha256,
        tokenizer_sha256=OFFICIAL_PATHS.tokenizer_sha256,
        runtime_contract=_runtime_contract(dit),
        weight_scope=ADAPTER_WEIGHT_SCOPE,
    )


def _clone_adapter_state(adapters: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in adapters.state_dict().items()
    }


def _is_a1(dit: nn.Module) -> bool:
    return getattr(dit, "mv_plucker_variant", None) in (
        "a1_plucker_v1", "a1_pose_routing_v1",
    )


def _checkpoint_format(dit: nn.Module) -> str:
    if getattr(dit, "mv_pose_routing_config", None) is not None:
        return A1_POSE_CHECKPOINT_FORMAT if _is_a1(dit) else A0_POSE_CHECKPOINT_FORMAT
    return A1_CHECKPOINT_FORMAT if _is_a1(dit) else CHECKPOINT_FORMAT


def _weight_scope(dit: nn.Module) -> str:
    return A1_ADAPTER_WEIGHT_SCOPE if _is_a1(dit) else ADAPTER_WEIGHT_SCOPE


def _signature_dict(
    dit: nn.Module,
    *,
    temporal_policy: TemporalStackPolicy,
    camera_order: Sequence[str],
) -> dict[str, Any]:
    signature = temporal_framework_signature(
        dit, temporal_policy=temporal_policy, camera_order=camera_order
    ).to_dict()
    routing = getattr(dit, "mv_pose_routing_config", None)
    if routing is not None:
        signature.update({
            "variant": "a1_pose_routing_v1" if _is_a1(dit) else "a0_pose_routing_v1",
            "camera_routing": routing.to_dict(),
            "camera_adjacency_role": "legacy_checkpoint_provenance_only",
            "runtime_adjacency_layout": "B,V_dst,V_src",
        })
    if not _is_a1(dit):
        return signature
    adapters = _require_tmv_adapters(dit)
    trainable = sum(parameter.numel() for parameter in adapters.parameters())
    if trainable != EXPECTED_A1_TRAINABLE_PARAMS:
        raise TemporalCheckpointContractError(
            f"A1 adapter count {trainable} != {EXPECTED_A1_TRAINABLE_PARAMS}"
        )
    if not bool(getattr(dit, "mv_plucker_conditioning_enabled", False)):
        raise TemporalCheckpointContractError("A1 Plucker conditioning is not enabled")
    signature.update({
        "variant": "a1_pose_routing_v1" if routing is not None else "a1_plucker_v1",
        "weight_scope": A1_ADAPTER_WEIGHT_SCOPE,
        "trainable_params": trainable,
        "plucker_conditioning": True,
        "plucker_dimension": int(dit.mv_plucker_dim),
        "plucker_hidden_dim": int(dit.mv_plucker_hidden_dim),
        "plucker_gate_initialization": float(dit.mv_plucker_gate_initialization),
        "plucker_convention": str(dit.mv_plucker_convention),
        "plucker_extrinsic_convention": str(
            dit.mv_geometry_extrinsic_convention
        ),
    })
    return signature


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, torch CPU, and initialized CUDA RNG streams.

    Merely saving a CPU checkpoint does not initialize CUDA.  During a GPU
    training run CUDA is already initialized, so every device stream is
    captured with :func:`torch.cuda.get_rng_state_all`.
    """

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - production dependency
        raise TemporalCheckpointContractError("NumPy is required to checkpoint RNG state") from error

    numpy_state = np.random.get_state()
    numpy_payload = {
        "algorithm": str(numpy_state[0]),
        "keys": torch.as_tensor(numpy_state[1].astype("int64", copy=True)),
        "position": int(numpy_state[2]),
        "has_gauss": bool(numpy_state[3]),
        "cached_gaussian": float(numpy_state[4]),
    }
    cuda_states: tuple[torch.Tensor, ...] = ()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = tuple(state.cpu().clone() for state in torch.cuda.get_rng_state_all())
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": numpy_payload,
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": cuda_states,
    }


def _validate_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise TemporalCheckpointContractError(
            f"RNG state fields mismatch: missing={sorted(required - set(state))}, "
            f"unexpected={sorted(set(state) - required)}"
        )
    numpy_state = state["numpy"]
    numpy_fields = {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}
    if not isinstance(numpy_state, Mapping) or set(numpy_state) != numpy_fields:
        raise TemporalCheckpointContractError("Malformed NumPy RNG state")
    if not isinstance(numpy_state["algorithm"], str):
        raise TemporalCheckpointContractError("Malformed NumPy RNG algorithm")
    if (
        not isinstance(numpy_state["keys"], torch.Tensor)
        or numpy_state["keys"].ndim != 1
        or numpy_state["keys"].dtype != torch.int64
    ):
        raise TemporalCheckpointContractError("Malformed NumPy RNG keys")
    if type(numpy_state["position"]) is not int or type(numpy_state["has_gauss"]) is not bool:
        raise TemporalCheckpointContractError("Malformed NumPy RNG counters")
    if type(numpy_state["cached_gaussian"]) is not float:
        raise TemporalCheckpointContractError("Malformed NumPy cached Gaussian")
    cpu_state = state["torch_cpu"]
    if (
        not isinstance(cpu_state, torch.Tensor)
        or cpu_state.ndim != 1
        or cpu_state.dtype != torch.uint8
    ):
        raise TemporalCheckpointContractError("Malformed torch CPU RNG state")
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, (tuple, list)) or any(
        not isinstance(item, torch.Tensor) or item.ndim != 1 or item.dtype != torch.uint8
        for item in cuda_states
    ):
        raise TemporalCheckpointContractError("Malformed torch CUDA RNG state")

    # Validate opaque formats using isolated RNG objects so malformed state is
    # rejected before adapter or optimizer state is mutated.
    try:
        import numpy as np

        python_probe = random.Random()
        python_probe.setstate(copy.deepcopy(state["python"]))
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(
            (
                numpy_state["algorithm"],
                numpy_state["keys"].detach().cpu().numpy().astype("uint32", copy=True),
                numpy_state["position"],
                int(numpy_state["has_gauss"]),
                numpy_state["cached_gaussian"],
            )
        )
        torch.Generator(device="cpu").set_state(cpu_state.detach().cpu())
    except (ImportError, TypeError, ValueError, RuntimeError) as error:
        raise TemporalCheckpointContractError("Malformed RNG state") from error


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Strictly restore all RNG streams captured by :func:`capture_rng_state`."""

    _validate_rng_state(state)
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - production dependency
        raise TemporalCheckpointContractError("NumPy is required to restore RNG state") from error

    cuda_states = tuple(state["torch_cuda"])
    if cuda_states:
        if not torch.cuda.is_available():
            raise TemporalCheckpointContractError(
                "Checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        device_count = torch.cuda.device_count()
        if device_count != len(cuda_states):
            raise TemporalCheckpointContractError(
                "CUDA RNG device count mismatch: "
                f"checkpoint={len(cuda_states)}, runtime={device_count}"
            )

    numpy_state = state["numpy"]
    numpy_keys = numpy_state["keys"].detach().cpu().numpy().astype("uint32", copy=True)
    random.setstate(copy.deepcopy(state["python"]))
    np.random.set_state(
        (
            numpy_state["algorithm"],
            numpy_keys,
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    if cuda_states:
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in cuda_states])


def _component_payload(component: object | None) -> dict[str, Any] | None:
    if component is None:
        return None
    state_dict = getattr(component, "state_dict", None)
    if not callable(state_dict):
        raise TypeError(f"{_class_name(component)} has no callable state_dict()")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise TypeError(f"{_class_name(component)}.state_dict() did not return a mapping")
    return {"class": _class_name(component), "state_dict": copy.deepcopy(dict(state))}


def _validate_optimizer_scope(dit: nn.Module, optimizer: object | None) -> None:
    """Ensure optimizer slots cannot smuggle frozen backbone state into a checkpoint."""

    if optimizer is None:
        return
    param_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(param_groups, list):
        raise TemporalCheckpointContractError("optimizer has no inspectable param_groups")
    adapter_parameters = tuple(_require_tmv_adapters(dit).parameters())
    expected_ids = {id(parameter) for parameter in adapter_parameters}
    actual_parameters = [
        parameter
        for group in param_groups
        for parameter in group.get("params", ())
    ]
    if any(not isinstance(parameter, nn.Parameter) for parameter in actual_parameters):
        raise TemporalCheckpointContractError("optimizer param_groups contain non-Parameters")
    actual_ids = {id(parameter) for parameter in actual_parameters}
    if len(actual_ids) != len(actual_parameters):
        raise TemporalCheckpointContractError("optimizer contains duplicate adapter parameters")
    if actual_ids != expected_ids:
        raise TemporalCheckpointContractError(
            "optimizer must contain every and only mv_adapters parameter: "
            f"missing={len(expected_ids - actual_ids)}, non_adapter={len(actual_ids - expected_ids)}"
        )
    if tuple(id(parameter) for parameter in actual_parameters) != tuple(
        id(parameter) for parameter in adapter_parameters
    ):
        raise TemporalCheckpointContractError(
            "optimizer parameters must follow mv_adapters registered order for strict state restore"
        )


def _validate_progress(global_step: int, epoch: int) -> None:
    if type(global_step) is not int or global_step < 0:
        raise ValueError("global_step must be a non-negative integer")
    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")


def make_temporal_checkpoint_payload(
    dit: nn.Module,
    *,
    global_step: int,
    epoch: int,
    optimizer: object | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    temporal_policy: TemporalStackPolicy = DEFAULT_TEMPORAL_POLICY,
    camera_order: Sequence[str] = CAMERA_ORDER,
    rng_state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a self-contained A-TMV adapter/training payload in memory.

    ``dit`` base parameters are deliberately unreachable through the saved
    weight scope: state is selected from ``dit.mv_adapters`` directly.
    """

    _validate_progress(global_step, epoch)
    signature = _signature_dict(
        dit, temporal_policy=temporal_policy, camera_order=camera_order
    )
    adapters = _require_tmv_adapters(dit)
    _validate_optimizer_scope(dit, optimizer)
    if scheduler is not None and optimizer is not None:
        scheduler_optimizer = getattr(scheduler, "optimizer", optimizer)
        if scheduler_optimizer is not optimizer:
            raise TemporalCheckpointContractError(
                "scheduler is attached to a different optimizer"
            )
    captured_rng = capture_rng_state() if rng_state is None else copy.deepcopy(dict(rng_state))
    _validate_rng_state(captured_rng)
    return {
        "format": _checkpoint_format(dit),
        "signature": signature,
        "progress": {"global_step": global_step, "epoch": epoch},
        "weights": {
            "scope": _weight_scope(dit),
            "state_dict": _clone_adapter_state(adapters),
        },
        "optimizer": _component_payload(optimizer),
        "scheduler": _component_payload(scheduler),
        "scaler": _component_payload(scaler),
        "rng": captured_rng,
        "metadata": {} if metadata is None else copy.deepcopy(dict(metadata)),
    }


def require_compatible_temporal_signature(
    dit: nn.Module,
    checkpoint_signature: Mapping[str, Any],
    *,
    temporal_policy: TemporalStackPolicy = DEFAULT_TEMPORAL_POLICY,
    camera_order: Sequence[str] = CAMERA_ORDER,
) -> None:
    """Reject every missing, extra, or unequal structural-signature field."""

    expected = _signature_dict(
        dit, temporal_policy=temporal_policy, camera_order=camera_order
    )
    actual = dict(checkpoint_signature)
    missing = sorted(set(expected) - set(actual))
    extras = sorted(set(actual) - set(expected))
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected.keys() & actual.keys()
        if expected[key] != actual[key]
    }
    if missing or extras or mismatches:
        raise TemporalCheckpointContractError(
            "A-TMV checkpoint signature mismatch: "
            f"missing={missing}, unexpected={extras}, fields={mismatches}"
        )


def _require_payload_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TemporalCheckpointContractError(f"Checkpoint field {key!r} must be a mapping")
    return value


def _validate_component_envelope(
    name: str,
    value: Any,
    target: object | None,
    *,
    allow_unrestored: bool,
) -> None:
    if value is None:
        if target is not None:
            raise TemporalCheckpointContractError(
                f"Checkpoint has no {name} state but a target {name} was provided"
            )
        return
    if not isinstance(value, Mapping) or set(value) != {"class", "state_dict"}:
        raise TemporalCheckpointContractError(f"Malformed {name} checkpoint state")
    if target is None:
        if allow_unrestored:
            if not isinstance(value["class"], str) or not isinstance(value["state_dict"], Mapping):
                raise TemporalCheckpointContractError(f"Malformed {name} checkpoint state")
            return
        raise TemporalCheckpointContractError(
            f"Checkpoint contains {name} state but no target {name} was provided"
        )
    expected_class = _class_name(target)
    if value["class"] != expected_class:
        raise TemporalCheckpointContractError(
            f"{name} class mismatch: expected={expected_class!r}, actual={value['class']!r}"
        )
    if not isinstance(value["state_dict"], Mapping):
        raise TemporalCheckpointContractError(f"Malformed {name} state_dict")


def _validate_payload(
    dit: nn.Module,
    payload: Mapping[str, Any],
    *,
    optimizer: object | None,
    scheduler: object | None,
    scaler: object | None,
    temporal_policy: TemporalStackPolicy,
    camera_order: Sequence[str],
    allow_unrestored_components: bool = False,
) -> TrainingProgress:
    required = {
        "format",
        "signature",
        "progress",
        "weights",
        "optimizer",
        "scheduler",
        "scaler",
        "rng",
        "metadata",
    }
    missing = sorted(required - set(payload))
    extras = sorted(set(payload) - required)
    if missing or extras:
        raise TemporalCheckpointContractError(
            f"Checkpoint fields mismatch: missing={missing}, unexpected={extras}"
        )
    if payload.get("format") != _checkpoint_format(dit):
        raise TemporalCheckpointContractError(
            f"Unsupported A-TMV checkpoint format: {payload.get('format')!r}"
        )
    signature = _require_payload_mapping(payload, "signature")
    require_compatible_temporal_signature(
        dit,
        signature,
        temporal_policy=temporal_policy,
        camera_order=camera_order,
    )

    progress = _require_payload_mapping(payload, "progress")
    if set(progress) != {"global_step", "epoch"}:
        raise TemporalCheckpointContractError("Checkpoint progress fields are incomplete")
    try:
        _validate_progress(progress["global_step"], progress["epoch"])
    except ValueError as error:
        raise TemporalCheckpointContractError(str(error)) from error

    weights = _require_payload_mapping(payload, "weights")
    if set(weights) != {"scope", "state_dict"} or weights.get("scope") != _weight_scope(dit):
        raise TemporalCheckpointContractError("Checkpoint is not adapter-only A-TMV state")
    state = weights.get("state_dict")
    if not isinstance(state, Mapping):
        raise TemporalCheckpointContractError("Checkpoint adapter state_dict is missing")
    expected_state = _require_tmv_adapters(dit).state_dict()
    missing_state = sorted(set(expected_state) - set(state))
    extra_state = sorted(set(state) - set(expected_state))
    invalid_tensors: dict[str, Any] = {}
    for name in expected_state.keys() & state.keys():
        actual = state[name]
        expected = expected_state[name]
        if not isinstance(actual, torch.Tensor):
            invalid_tensors[name] = "not a tensor"
        elif actual.shape != expected.shape or actual.dtype != expected.dtype:
            invalid_tensors[name] = {
                "expected": (tuple(expected.shape), str(expected.dtype)),
                "actual": (tuple(actual.shape), str(actual.dtype)),
            }
        elif name.endswith(".camera_adjacency") and not torch.equal(
            actual.detach().cpu(), expected.detach().cpu()
        ):
            invalid_tensors[name] = "semantic adjacency value mismatch"
    if missing_state or extra_state or invalid_tensors:
        raise TemporalCheckpointContractError(
            "A-TMV adapter tensor schema mismatch: "
            f"missing={missing_state[:8]}, unexpected={extra_state[:8]}, "
            f"invalid={dict(list(invalid_tensors.items())[:8])}"
        )

    _validate_component_envelope(
        "optimizer",
        payload["optimizer"],
        optimizer,
        allow_unrestored=allow_unrestored_components,
    )
    _validate_component_envelope(
        "scheduler",
        payload["scheduler"],
        scheduler,
        allow_unrestored=allow_unrestored_components,
    )
    _validate_component_envelope(
        "scaler", payload["scaler"], scaler, allow_unrestored=allow_unrestored_components
    )
    if not allow_unrestored_components:
        _validate_optimizer_scope(dit, optimizer)
        if scheduler is not None and optimizer is not None:
            scheduler_optimizer = getattr(scheduler, "optimizer", optimizer)
            if scheduler_optimizer is not optimizer:
                raise TemporalCheckpointContractError(
                    "scheduler is attached to a different optimizer"
                )
    rng = _require_payload_mapping(payload, "rng")
    _validate_rng_state(rng)
    if not isinstance(payload["metadata"], Mapping):
        raise TemporalCheckpointContractError("Checkpoint metadata must be a mapping")
    return TrainingProgress(int(progress["global_step"]), int(progress["epoch"]))


def load_temporal_adapter_weights(
    dit: nn.Module,
    payload: Mapping[str, Any],
    *,
    temporal_policy: TemporalStackPolicy = DEFAULT_TEMPORAL_POLICY,
    camera_order: Sequence[str] = CAMERA_ORDER,
) -> TrainingProgress:
    """Strictly validate and load adapter weights, without training state.

    This inference-oriented API permits optimizer/scheduler/scaler state in
    the payload but intentionally ignores it.  The returned counters are
    useful for provenance and do not mutate a training loop.
    """

    progress = _validate_payload(
        dit,
        payload,
        optimizer=None,
        scheduler=None,
        scaler=None,
        temporal_policy=temporal_policy,
        camera_order=camera_order,
        allow_unrestored_components=True,
    )
    state = payload["weights"]["state_dict"]
    _require_tmv_adapters(dit).load_state_dict(state, strict=True)
    return progress


def _restore_component(name: str, envelope: Any, target: object | None) -> None:
    if envelope is None:
        return
    # Class/schema checking was completed before any state mutation.
    load_state_dict = getattr(target, "load_state_dict", None)
    if not callable(load_state_dict):
        raise TemporalCheckpointContractError(f"Target {name} has no callable load_state_dict()")
    load_state_dict(copy.deepcopy(dict(envelope["state_dict"])))


def restore_temporal_training_state(
    dit: nn.Module,
    payload: Mapping[str, Any],
    *,
    optimizer: object | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    temporal_policy: TemporalStackPolicy = DEFAULT_TEMPORAL_POLICY,
    camera_order: Sequence[str] = CAMERA_ORDER,
    restore_rng: bool = True,
) -> TrainingProgress:
    """Restore counters, optimizer/scheduler/scaler, and RNG but not weights."""

    progress = _validate_payload(
        dit,
        payload,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        temporal_policy=temporal_policy,
        camera_order=camera_order,
    )
    _restore_component("optimizer", payload["optimizer"], optimizer)
    _restore_component("scheduler", payload["scheduler"], scheduler)
    _restore_component("scaler", payload["scaler"], scaler)
    if restore_rng:
        restore_rng_state(payload["rng"])
    return progress


def restore_temporal_checkpoint_payload(
    dit: nn.Module,
    payload: Mapping[str, Any],
    *,
    optimizer: object | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    temporal_policy: TemporalStackPolicy = DEFAULT_TEMPORAL_POLICY,
    camera_order: Sequence[str] = CAMERA_ORDER,
    restore_rng: bool = True,
) -> TrainingProgress:
    """Strictly restore adapter weights and complete training state."""

    progress = _validate_payload(
        dit,
        payload,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        temporal_policy=temporal_policy,
        camera_order=camera_order,
    )
    _require_tmv_adapters(dit).load_state_dict(payload["weights"]["state_dict"], strict=True)
    _restore_component("optimizer", payload["optimizer"], optimizer)
    _restore_component("scheduler", payload["scheduler"], scheduler)
    _restore_component("scaler", payload["scaler"], scaler)
    if restore_rng:
        restore_rng_state(payload["rng"])
    return progress


def save_temporal_checkpoint(
    path: str | os.PathLike[str],
    dit: nn.Module,
    *,
    global_step: int,
    epoch: int,
    optimizer: object | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    temporal_policy: TemporalStackPolicy = DEFAULT_TEMPORAL_POLICY,
    camera_order: Sequence[str] = CAMERA_ORDER,
    rng_state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save an adapter-centric A-TMV checkpoint and return its path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = make_temporal_checkpoint_payload(
        dit,
        global_step=global_step,
        epoch=epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        temporal_policy=temporal_policy,
        camera_order=camera_order,
        rng_state=rng_state,
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


def read_temporal_checkpoint(
    path: str | os.PathLike[str], *, map_location: str | torch.device = "cpu"
) -> Mapping[str, Any]:
    """Read a trusted A-TMV checkpoint without applying any state."""

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise TemporalCheckpointContractError("Checkpoint root must be a mapping")
    return payload


def load_temporal_checkpoint(
    path: str | os.PathLike[str],
    dit: nn.Module,
    *,
    optimizer: object | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    temporal_policy: TemporalStackPolicy = DEFAULT_TEMPORAL_POLICY,
    camera_order: Sequence[str] = CAMERA_ORDER,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> TrainingProgress:
    """Read and strictly restore a complete A-TMV training checkpoint."""

    payload = read_temporal_checkpoint(path, map_location=map_location)
    return restore_temporal_checkpoint_payload(
        dit,
        payload,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        temporal_policy=temporal_policy,
        camera_order=camera_order,
        restore_rng=restore_rng,
    )
