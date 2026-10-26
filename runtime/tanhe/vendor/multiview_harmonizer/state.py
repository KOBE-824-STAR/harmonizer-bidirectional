"""Architecture-safe state contract for later A/B training.

This module only defines checkpoint payloads; importing it or constructing a
payload does not start training and does not write a file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import torch
from torch import nn

from .constants import CAMERA_ORDER, MODEL_FAMILY, OFFICIAL_PATHS, UPSTREAM_COMMIT, Architecture


CHECKPOINT_FORMAT = "diffusionharmonizer_multiview_v1"
WeightScope = Literal["extensions", "dit"]


class CheckpointContractError(RuntimeError):
    """Raised before weights are loaded into an incompatible A/B structure."""


@dataclass(frozen=True)
class FrameworkSignature:
    architecture: str
    architecture_version: int
    camera_order: tuple[str, ...]
    max_views: int
    adapter_after_blocks: tuple[int, ...]
    adapter_dim: int | None
    adapter_heads: int | None
    adapter_pool_factor: int | None
    position_encoding_mode: str
    attention_mask_mode: str
    model_family: str
    upstream_commit: str
    cosmos_dit_sha256: str
    harmonizer_sha256: str
    tokenizer_sha256: str
    new_parameter_prefixes: tuple[str, ...]
    scheme: str
    attention_variant: str
    temporal: bool
    geometry_version: str
    camera_schema_version: str
    camera_adjacency: tuple[tuple[str, tuple[str, ...]], ...]
    base_checkpoint_identity: str
    code_commit: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_adapter(dit: nn.Module):
    adapters = getattr(dit, "mv_adapters", None)
    if adapters is None or len(adapters) == 0:
        return None
    return next(iter(adapters.values()))


def framework_signature(dit: nn.Module) -> FrameworkSignature:
    """Read all resume-critical structural fields from an upgraded DiT."""

    try:
        architecture = Architecture(str(dit.mv_architecture))
    except (AttributeError, ValueError) as error:
        raise CheckpointContractError("DiT has no recognized multi-view architecture") from error

    max_views = int(getattr(dit, "mv_max_views", len(CAMERA_ORDER)))
    if architecture is Architecture.FACTORIZED_ADAPTER:
        adapter = _first_adapter(dit)
        if adapter is None:
            raise CheckpointContractError("Scheme A has no mv_adapters")
        layers = tuple(int(index) for index in dit.mv_adapter_after_blocks)
        adapter_dim = int(adapter.adapter_dim)
        adapter_heads = int(adapter.attention.num_heads)
        adapter_pool_factor = int(adapter.pool_factor)
        position_mode = "official_per_view_3d_rope"
        attention_mode = "pooled_neighbor_cross_view"
        prefixes = ("mv_adapters.",)
        attention_variant = "factorized_adapter"
        adjacency = ()
    elif architecture is Architecture.JOINT_TOKENS:
        if not hasattr(dit, "camera_embedding"):
            raise CheckpointContractError("Scheme B has no camera_embedding")
        layers = ()
        adapter_dim = None
        adapter_heads = None
        adapter_pool_factor = None
        attention_variant = str(getattr(dit, "mv_attention_variant", "dense"))
        if attention_variant == "warm_v1":
            position_mode = "official_per_view_3d_rope"
            attention_mode = "edge_batched_zero_gated_cross_residual"
            prefixes = ("camera_embedding.", "blocks.*.self_attn.cross_view_gate")
            matrix = getattr(dit, "mv_camera_adjacency", None)
            if matrix is None:
                raise CheckpointContractError("B-warm has no canonical camera adjacency")
            adjacency = tuple(
                (
                    CAMERA_ORDER[target],
                    tuple(
                        CAMERA_ORDER[source]
                        for source, enabled in enumerate(matrix[target])
                        if enabled
                    ),
                )
                for target in range(len(CAMERA_ORDER))
            )
        elif attention_variant == "dense":
            position_mode = "repeated_single_frame_spatial_rope"
            attention_mode = "dense_joint_tokens_complete_views"
            prefixes = ("camera_embedding.",)
            adjacency = ()
        else:
            raise CheckpointContractError(
                f"Unknown Scheme B attention variant {attention_variant!r}"
            )
    else:
        layers = ()
        adapter_dim = None
        adapter_heads = None
        adapter_pool_factor = None
        position_mode = "official_per_view_3d_rope"
        attention_mode = "independent_views"
        prefixes = ()
        attention_variant = "independent"
        adjacency = ()

    return FrameworkSignature(
        architecture=architecture.value,
        architecture_version=1,
        camera_order=CAMERA_ORDER,
        max_views=max_views,
        adapter_after_blocks=layers,
        adapter_dim=adapter_dim,
        adapter_heads=adapter_heads,
        adapter_pool_factor=adapter_pool_factor,
        position_encoding_mode=position_mode,
        attention_mask_mode=attention_mode,
        model_family=MODEL_FAMILY,
        upstream_commit=UPSTREAM_COMMIT,
        cosmos_dit_sha256=OFFICIAL_PATHS.cosmos_dit_sha256,
        harmonizer_sha256=OFFICIAL_PATHS.harmonizer_sha256,
        tokenizer_sha256=OFFICIAL_PATHS.tokenizer_sha256,
        new_parameter_prefixes=prefixes,
        scheme=architecture.value,
        attention_variant=attention_variant,
        temporal=False,
        geometry_version="none",
        camera_schema_version=str(getattr(dit, "mv_camera_schema_version", "canonical_v1")),
        camera_adjacency=adjacency,
        base_checkpoint_identity=OFFICIAL_PATHS.harmonizer_sha256,
        code_commit="UNAVAILABLE_NOT_A_GIT_WORKTREE",
    )


def _extension_state_keys(dit: nn.Module, signature: FrameworkSignature) -> set[str]:
    keys = {
        key
        for key in dit.state_dict()
        if any(
            key.startswith(prefix)
            for prefix in signature.new_parameter_prefixes
            if "*" not in prefix
        )
    }
    if signature.attention_variant == "warm_v1":
        keys.update(
            key
            for key in dit.state_dict()
            if key.startswith("blocks.") and key.endswith(".self_attn.cross_view_gate")
        )
    return keys


def scoped_dit_state_dict(dit: nn.Module, scope: WeightScope) -> dict[str, torch.Tensor]:
    """Select either only A/B additions or the complete fine-tuned DiT."""

    signature = framework_signature(dit)
    state = dit.state_dict()
    if scope == "dit":
        return dict(state)
    if scope != "extensions":
        raise ValueError(f"Unknown checkpoint weight scope: {scope!r}")
    keys = _extension_state_keys(dit, signature)
    if signature.architecture != Architecture.INDEPENDENT.value and not keys:
        raise CheckpointContractError("No extension tensors found for a multi-view architecture")
    return {key: state[key] for key in sorted(keys)}


def make_checkpoint_payload(
    dit: nn.Module,
    *,
    weight_scope: WeightScope,
    step: int,
    num_views: int,
    manifest_sha256: str,
    source_commit: str,
    runtime_image_id: str,
    optimizer_state: Mapping[str, Any] | None = None,
    scheduler_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build (but do not save) a future training checkpoint payload.

    Use ``extensions`` only while the official DiT is frozen.  Once any base
    DiT parameter is optimized, callers must use ``dit``.
    """

    if step < 0:
        raise ValueError("step must be non-negative")
    signature = framework_signature(dit)
    if not 1 <= num_views <= signature.max_views:
        raise ValueError(f"num_views must be in [1,{signature.max_views}]")
    required_strings = {
        "manifest_sha256": manifest_sha256,
        "source_commit": source_commit,
        "runtime_image_id": runtime_image_id,
    }
    empty = [name for name, value in required_strings.items() if not value.strip()]
    if empty:
        raise ValueError(f"Checkpoint provenance fields cannot be empty: {empty}")

    return {
        "format": CHECKPOINT_FORMAT,
        "signature": signature.to_dict(),
        "training": {
            "step": int(step),
            "num_views": int(num_views),
            "manifest_sha256": manifest_sha256,
            "source_commit": source_commit,
            "runtime_image_id": runtime_image_id,
        },
        "weight_scope": weight_scope,
        "state_dict_dit": scoped_dit_state_dict(dit, weight_scope),
        "optimizer": None if optimizer_state is None else dict(optimizer_state),
        "scheduler": None if scheduler_state is None else dict(scheduler_state),
    }


def require_compatible_signature(dit: nn.Module, checkpoint_signature: Mapping[str, Any]) -> None:
    expected = framework_signature(dit).to_dict()
    actual = dict(checkpoint_signature)
    mismatches = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in expected
        if expected.get(key) != actual.get(key)
    }
    extras = sorted(set(actual) - set(expected))
    if mismatches or extras:
        raise CheckpointContractError(
            f"Checkpoint architecture signature mismatch: fields={mismatches}, extras={extras}"
        )


def load_checkpoint_payload(dit: nn.Module, payload: Mapping[str, Any]) -> None:
    """Load a validated A/B payload into a matching, official-initialized DiT."""

    if payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointContractError(f"Unsupported checkpoint format: {payload.get('format')!r}")
    signature_payload = payload.get("signature")
    if not isinstance(signature_payload, Mapping):
        raise CheckpointContractError("Checkpoint has no signature mapping")
    require_compatible_signature(dit, signature_payload)

    scope = payload.get("weight_scope")
    if scope not in ("extensions", "dit"):
        raise CheckpointContractError(f"Invalid checkpoint weight_scope: {scope!r}")
    state = payload.get("state_dict_dit")
    if not isinstance(state, Mapping):
        raise CheckpointContractError("Checkpoint has no state_dict_dit mapping")

    if scope == "dit":
        dit.load_state_dict(state, strict=True)
        return

    expected_extension_keys = _extension_state_keys(dit, framework_signature(dit))
    actual_keys = set(state)
    if actual_keys != expected_extension_keys:
        raise CheckpointContractError(
            "Extension checkpoint keys do not match this architecture: "
            f"missing={sorted(expected_extension_keys - actual_keys)[:8]}, "
            f"unexpected={sorted(actual_keys - expected_extension_keys)[:8]}"
        )
    missing, unexpected = dit.load_state_dict(state, strict=False)
    missing_extensions = sorted(set(missing) & expected_extension_keys)
    if missing_extensions or unexpected:
        raise CheckpointContractError(
            f"Extension load failed: missing={missing_extensions[:8]}, "
            f"unexpected={list(unexpected)[:8]}"
        )
