"""Configuration for the isolated Scheme A-TMV implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .b_tmv_routing import PoseRoutingConfig
from .layout_temporal import CURRENT_OUTPUT_INDEX, OFFICIAL_TEMPORAL_OFFSETS


_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass(frozen=True)
class TemporalFrameworkConfig:
    variant: str
    cosmos_checkpoint_dir: Path
    harmonizer_checkpoint: Path
    timestep: int
    dtype: torch.dtype
    max_views: int
    adapter_after_blocks: tuple[int, ...]
    adapter_dim: int
    adapter_heads: int
    adapter_pool_factor: int
    official_temporal_offsets: tuple[int, ...]
    current_output_index: int
    plucker_enabled: bool
    plucker_hidden_dim: int
    plucker_gate_init: float
    pose_routing: PoseRoutingConfig | None

    @classmethod
    def from_json(cls, path: str | Path) -> "TemporalFrameworkConfig":
        payload = json.loads(Path(path).read_text())
        if payload.get("architecture") != "A-TMV":
            raise ValueError("Temporal config architecture must be 'A-TMV'")
        if payload.get("training_enabled") is not False:
            raise ValueError("Framework config must set training_enabled=false")
        dtype_name = payload.get("dtype", "bf16")
        if dtype_name not in _DTYPES:
            raise ValueError(f"unknown dtype {dtype_name!r}")
        offsets = tuple(payload.get("official_temporal_offsets", OFFICIAL_TEMPORAL_OFFSETS))
        if offsets != OFFICIAL_TEMPORAL_OFFSETS:
            raise ValueError(f"official temporal ordering must be {OFFICIAL_TEMPORAL_OFFSETS}")
        current_index = int(payload.get("current_output_index", CURRENT_OUTPUT_INDEX))
        if current_index != CURRENT_OUTPUT_INDEX:
            raise ValueError("upstream current output is temporal slot zero")
        variant = payload.get("variant", "a0")
        if variant not in {"a0", "a1_plucker_v1", "a0_pose_routing_v1", "a1_pose_routing_v1"}:
            raise ValueError(f"unsupported A-TMV variant {variant!r}")
        plucker = payload.get("plucker", {})
        expected_plucker = variant in {"a1_plucker_v1", "a1_pose_routing_v1"}
        if bool(plucker.get("enabled", False)) != expected_plucker:
            raise ValueError("A1 variant and plucker.enabled disagree")
        hidden_dim = int(plucker.get("hidden_dim", 64))
        gate_init = float(plucker.get("gate_init", 0.0))
        if hidden_dim < 1 or gate_init != 0.0:
            raise ValueError("A1 requires hidden_dim>0 and gate_init=0")
        routed = variant in {"a0_pose_routing_v1", "a1_pose_routing_v1"}
        if routed != ("camera_routing" in payload):
            raise ValueError("A pose routing variant and camera_routing must agree")
        pose_routing = PoseRoutingConfig.from_dict(payload["camera_routing"]) if routed else None
        return cls(
            variant=variant,
            cosmos_checkpoint_dir=Path(payload["cosmos_checkpoint_dir"]),
            harmonizer_checkpoint=Path(payload["harmonizer_checkpoint"]),
            timestep=int(payload.get("timestep", 250)),
            dtype=_DTYPES[dtype_name],
            max_views=int(payload.get("max_views", 6)),
            adapter_after_blocks=tuple(payload.get("adapter_after_blocks", (4, 9, 14, 19))),
            adapter_dim=int(payload.get("adapter_dim", 320)),
            adapter_heads=int(payload.get("adapter_heads", 8)),
            adapter_pool_factor=int(payload.get("adapter_pool_factor", 4)),
            official_temporal_offsets=offsets,
            current_output_index=current_index,
            plucker_enabled=expected_plucker,
            plucker_hidden_dim=hidden_dim,
            plucker_gate_init=gate_init,
            pose_routing=pose_routing,
        )
