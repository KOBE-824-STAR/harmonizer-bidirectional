"""Calibration-only potential-frustum-overlap routing for the T4 ablation.

Matrices use positional [B, destination, source] indexing. Depths are optical
axis depths in metres. This estimates potential overlap, not scene visibility.
It deliberately does not warp tokens or change T4's attention/FiLM weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch

from .b_tmv_geometry import RelativeCameraGeometryEncoder


POSE_ROUTING_VARIANT = "warm_v1_geometry_t4_pose_routing"


@dataclass(frozen=True)
class PoseRoutingConfig:
    mode: str = "frustum_overlap_v1"
    grid_height: int = 8
    grid_width: int = 16
    depth_min_m: float = 1.0
    depth_max_m: float = 80.0
    depth_samples: int = 16
    overlap_threshold: float = 0.01
    top_k: int = 2
    depth_spacing: str = "log"
    score_reduction: str = "mean_ray_depth_samples"
    tie_policy: str = "include_boundary_ties"

    def __post_init__(self) -> None:
        for name in ("grid_height", "grid_width", "depth_samples", "top_k"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"routing {name} must be a positive integer")
        for name in ("depth_min_m", "depth_max_m", "overlap_threshold"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"routing {name} must be finite")
        if not 0 < self.depth_min_m < self.depth_max_m:
            raise ValueError("routing requires 0 < depth_min_m < depth_max_m")
        if not 0 <= self.overlap_threshold < 1:
            raise ValueError("routing overlap_threshold must be in [0,1)")
        for name, expected in {
            "mode": "frustum_overlap_v1", "depth_spacing": "log",
            "score_reduction": "mean_ray_depth_samples",
            "tie_policy": "include_boundary_ties",
        }.items():
            if getattr(self, name) != expected:
                raise ValueError(f"unsupported routing {name}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PoseRoutingConfig":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("camera_routing must specify exactly the routing schema")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def build_pose_overlap_adjacency(
    intrinsics: torch.Tensor,
    T_ego_from_cam: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    view_valid: torch.Tensor,
    config: PoseRoutingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bool adjacency and float scores, both [B,V_dst,V_src].

    Sample a destination frustum, project into each source, and average the
    in-frame positive-depth indicators. No image content, names, IDs, or order
    enter the score. Top-k retains exact boundary ties to preserve permutation
    equivariance; consequently tied cases can have more than k sources.
    """
    batch, views, height, width = RelativeCameraGeometryEncoder.validate_calibration(
        intrinsics, T_ego_from_cam, image_hw=image_hw
    )
    if view_valid.shape != (batch, views) or view_valid.dtype != torch.bool:
        raise ValueError("view_valid must be bool [B,V]")
    if view_valid.device != intrinsics.device:
        raise ValueError("view_valid must share the calibration device")
    if batch < 1 or views < 1:
        raise ValueError("routing requires at least one sample and view")

    # Small calibration computations use float64, independent of AMP/TF32.
    device = intrinsics.device
    k = intrinsics.double()
    poses = T_ego_from_cam.double()
    rotation = poses[..., :3, :3]
    eye = torch.eye(3, device=device, dtype=torch.float64)
    if not torch.allclose(rotation.transpose(-1, -2) @ rotation, eye.expand_as(rotation), atol=1e-4, rtol=0):
        raise ValueError("camera poses must have orthonormal rotations")
    if not torch.allclose(torch.linalg.det(rotation), torch.ones((batch, views), device=device, dtype=torch.float64), atol=1e-4, rtol=0):
        raise ValueError("camera rotations must have determinant +1")
    expected_bottom = k.new_tensor([0, 0, 1]).expand(batch, views, 3)
    if not torch.allclose(k[..., 2, :], expected_bottom, atol=1e-6, rtol=0):
        raise ValueError("intrinsics bottom row must be [0,0,1]")
    if bool((torch.linalg.det(k).abs() < 1e-12).any()):
        raise ValueError("intrinsics must be invertible")

    y = (torch.arange(config.grid_height, device=device, dtype=torch.float64) + 0.5) * height / config.grid_height
    x = (torch.arange(config.grid_width, device=device, dtype=torch.float64) + 0.5) * width / config.grid_width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    pixels = torch.stack((xx.flatten(), yy.flatten(), torch.ones_like(xx).flatten()), -1)
    rays = torch.einsum("bvij,pj->bvpi", torch.linalg.inv(k), pixels)
    depths = torch.logspace(math.log10(config.depth_min_m), math.log10(config.depth_max_m), config.depth_samples, device=device, dtype=torch.float64)
    points = (rays[..., None, :] * depths[None, None, None, :, None]).flatten(2, 3)
    inverse = torch.linalg.inv(poses)
    scores = torch.zeros((batch, views, views), device=device, dtype=torch.float32)
    # Loop over the small camera set; never form an image-token all-pairs matrix.
    for dst in range(views):
        source_from_dst = inverse @ poses[:, dst, None]
        projected_3d = torch.einsum("bsij,bpj->bspi", source_from_dst[..., :3, :3], points[:, dst]) + source_from_dst[..., :3, 3].unsqueeze(-2)
        projected = torch.einsum("bsij,bspj->bspi", k, projected_3d)
        z = projected_3d[..., 2]
        uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-8)
        inside = (z > 1e-6) & torch.isfinite(uv).all(-1)
        inside &= (uv[..., 0] >= 0) & (uv[..., 0] < width)
        inside &= (uv[..., 1] >= 0) & (uv[..., 1] < height)
        scores[:, dst] = inside.float().mean(-1)
    eligible = view_valid[:, :, None] & view_valid[:, None, :]
    eligible &= ~torch.eye(views, device=device, dtype=torch.bool)[None]
    scores.masked_fill_(~eligible, 0)
    adjacency = eligible & (scores > config.overlap_threshold)
    candidates = scores.masked_fill(~adjacency, -1)
    cutoff = candidates.topk(min(config.top_k, views), dim=-1).values[..., -1:]
    adjacency &= candidates >= cutoff
    return adjacency, scores


def attach_pose_routing(dit: torch.nn.Module, config: PoseRoutingConfig) -> None:
    if getattr(dit, "mv_attention_variant", None) != "warm_v1_geometry_t4":
        raise ValueError("pose routing must attach to a T4 geometry model")
    if not getattr(dit, "mv_geometry_conditioning_enabled", False):
        raise ValueError("pose routing requires enabled T4 FiLM")
    dit.mv_pose_routing_config = config
    dit.mv_attention_variant = POSE_ROUTING_VARIANT
    dit.mv_last_pose_routing = None


def resolve_cross_view_adjacency(dit, intrinsics, poses, *, image_hw, view_valid, device):
    """One graph per forward, shared by geometry encoding and all residuals."""
    config = getattr(dit, "mv_pose_routing_config", None)
    if config is None:
        return torch.tensor(dit.mv_camera_adjacency, device=device, dtype=torch.bool)
    if intrinsics is None or poses is None or image_hw is None:
        raise ValueError("pose routing requires intrinsics, T_ego_from_cam and image_hw")
    if not getattr(dit, "mv_geometry_conditioning_enabled", False):
        raise ValueError("pose routing requires enabled T4 FiLM")
    adjacency, scores = build_pose_overlap_adjacency(
        intrinsics, poses, image_hw=image_hw, view_valid=view_valid, config=config
    )
    dit.mv_last_pose_routing = {"adjacency": adjacency.detach(), "overlap_scores": scores.detach()}
    return adjacency
