"""Small JSON configuration layer; it does not start training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .constants import Architecture


_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@dataclass(frozen=True)
class FrameworkConfig:
    architecture: Architecture
    cosmos_checkpoint_dir: Path
    harmonizer_checkpoint: Path
    timestep: int
    dtype: torch.dtype
    max_views: int
    adapter_after_blocks: tuple[int, ...]
    adapter_dim: int
    adapter_heads: int
    adapter_pool_factor: int
    attention_variant: str
    camera_schema_version: str
    camera_adjacency: dict[str, tuple[str, ...]]
    adjacency_undirected: bool
    temporal: bool
    geometry_version: str

    @classmethod
    def from_json(cls, path: str | Path) -> "FrameworkConfig":
        payload = json.loads(Path(path).read_text())
        if payload.get("training_enabled") is not False:
            raise ValueError("Framework configs must explicitly set training_enabled=false")
        dtype_name = payload.get("dtype", "bf16")
        if dtype_name not in _DTYPES:
            raise ValueError(f"Unknown dtype {dtype_name!r}")
        attention_variant = str(payload.get("attention_variant", "dense"))
        if attention_variant not in ("dense", "warm_v1"):
            raise ValueError(f"Unknown attention_variant {attention_variant!r}")
        adjacency = {
            str(camera): tuple(str(neighbor) for neighbor in neighbors)
            for camera, neighbors in payload.get("camera_adjacency", {}).items()
        }
        if attention_variant == "warm_v1" and not adjacency:
            raise ValueError("warm_v1 requires a non-empty camera_adjacency mapping")
        temporal = bool(payload.get("temporal", False))
        geometry_version = str(payload.get("geometry_version", "none"))
        if temporal:
            raise ValueError("Legacy Scheme B configs must keep temporal=false")
        if geometry_version != "none":
            raise ValueError("Scheme B phase one requires geometry_version='none'")
        return cls(
            architecture=Architecture(payload["architecture"]),
            cosmos_checkpoint_dir=Path(payload["cosmos_checkpoint_dir"]),
            harmonizer_checkpoint=Path(payload["harmonizer_checkpoint"]),
            timestep=int(payload.get("timestep", 250)),
            dtype=_DTYPES[dtype_name],
            max_views=int(payload.get("max_views", 6)),
            adapter_after_blocks=tuple(payload.get("adapter_after_blocks", [4, 9, 14, 19])),
            adapter_dim=int(payload.get("adapter_dim", 320)),
            adapter_heads=int(payload.get("adapter_heads", 8)),
            adapter_pool_factor=int(payload.get("adapter_pool_factor", 4)),
            attention_variant=attention_variant,
            camera_schema_version=str(payload.get("camera_schema_version", "canonical_v1")),
            camera_adjacency=adjacency,
            adjacency_undirected=bool(payload.get("undirected", False)),
            temporal=temporal,
            geometry_version=geometry_version,
        )
