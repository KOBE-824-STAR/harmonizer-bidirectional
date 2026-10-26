"""T4 pairwise relative-camera conditioning for Scheme B-TMV.

The calibration convention is ``T_ego_from_cam``:

``p_ego = T_ego_from_cam @ p_cam``.

For an existing directed source-to-destination graph edge, the feature uses
``T_dst_from_src = inv(T_ego_from_cam_dst) @ T_ego_from_cam_src``.  No absolute
world/ego coordinate is exposed to the geometry MLP.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


GEOMETRY_FEATURE_DIM = 20
GEOMETRY_MODE = "relative_pose"
GEOMETRY_CONVENTION = (
    "T_dst_from_src = inv(T_ego_from_cam_dst) @ T_ego_from_cam_src"
)


class RelativeCameraGeometryEncoder(nn.Module):
    """Encode directed relative pose and normalized intrinsics on existing edges.

    Raw edge features are ``[R_dst_from_src.flatten(), t_dst_from_src,
    fx_src/W, fy_src/H, cx_src/W, cy_src/H, fx_dst/W, fy_dst/H, cx_dst/W,
    cy_dst/H]`` and therefore have exactly 20 elements.  The shared MLP is used
    once for every DiT block; only small per-block FiLM projections are copied.
    """

    def __init__(self, bottleneck_dim: int = 64) -> None:
        super().__init__()
        if bottleneck_dim < 1:
            raise ValueError("bottleneck_dim must be positive")
        self.feature_dim = GEOMETRY_FEATURE_DIM
        self.bottleneck_dim = int(bottleneck_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.bottleneck_dim),
            nn.SiLU(),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
        )
        self._last_audit: dict[str, Any] | None = None
        self._last_edge_features: torch.Tensor | None = None
        self._last_edge_pairs: list[list[list[int]]] | None = None

    @staticmethod
    def validate_calibration(
        intrinsics: torch.Tensor,
        T_ego_from_cam: torch.Tensor,
        *,
        image_hw: tuple[int, int],
        batch: int | None = None,
        views: int | None = None,
        homogeneous_atol: float = 1e-5,
    ) -> tuple[int, int, int, int]:
        """Validate static per-camera calibration and return ``B,V,H,W``."""

        if not isinstance(intrinsics, torch.Tensor) or not isinstance(
            T_ego_from_cam, torch.Tensor
        ):
            raise TypeError("intrinsics and T_ego_from_cam must be tensors")
        if intrinsics.ndim != 4 or tuple(intrinsics.shape[-2:]) != (3, 3):
            raise ValueError("intrinsics must have shape [B,V,3,3]")
        if T_ego_from_cam.ndim != 4 or tuple(T_ego_from_cam.shape[-2:]) != (4, 4):
            raise ValueError("T_ego_from_cam must have shape [B,V,4,4]")
        if intrinsics.shape[:2] != T_ego_from_cam.shape[:2]:
            raise ValueError("intrinsics and T_ego_from_cam must share [B,V]")
        actual_batch, actual_views = map(int, intrinsics.shape[:2])
        if batch is not None and actual_batch != batch:
            raise ValueError(f"calibration batch {actual_batch} != image batch {batch}")
        if views is not None and actual_views != views:
            raise ValueError(f"calibration cameras {actual_views} != image cameras {views}")
        if intrinsics.device != T_ego_from_cam.device:
            raise ValueError("intrinsics and T_ego_from_cam must share a device")
        if not intrinsics.is_floating_point() or not T_ego_from_cam.is_floating_point():
            raise TypeError("calibration tensors must be floating point")
        if not bool(torch.isfinite(intrinsics).all().item()):
            raise ValueError("intrinsics contains non-finite values")
        if not bool(torch.isfinite(T_ego_from_cam).all().item()):
            raise ValueError("T_ego_from_cam contains non-finite values")
        if not isinstance(image_hw, (tuple, list)) or len(image_hw) != 2:
            raise ValueError("image_hw must be (H,W)")
        height, width = int(image_hw[0]), int(image_hw[1])
        if height < 1 or width < 1:
            raise ValueError("image height and width must be positive")
        bottom = T_ego_from_cam[..., 3, :].float()
        expected = torch.zeros_like(bottom)
        expected[..., 3] = 1
        if not bool(torch.allclose(bottom, expected, atol=homogeneous_atol, rtol=0)):
            raise ValueError("T_ego_from_cam homogeneous bottom row must be [0,0,0,1]")
        focal = intrinsics[..., (0, 1), (0, 1)]
        if bool((focal <= 0).any().item()):
            raise ValueError("intrinsics focal lengths must be positive")
        return actual_batch, actual_views, height, width

    @staticmethod
    def normalized_intrinsics(
        intrinsics: torch.Tensor, *, image_hw: tuple[int, int]
    ) -> torch.Tensor:
        """Return ``[fx/W, fy/H, cx/W, cy/H]`` in float32."""

        height, width = image_hw
        value = intrinsics.float()
        return torch.stack(
            (
                value[..., 0, 0] / float(width),
                value[..., 1, 1] / float(height),
                value[..., 0, 2] / float(width),
                value[..., 1, 2] / float(height),
            ),
            dim=-1,
        )

    @staticmethod
    def directed_relative_transforms(
        T_ego_from_cam: torch.Tensor, edge_src_dst: torch.Tensor
    ) -> torch.Tensor:
        """Return ``T_dst_from_src`` for positional ``[src,dst]`` edges."""

        if edge_src_dst.ndim != 2 or edge_src_dst.shape[-1] != 2:
            raise ValueError("edge_src_dst must be [E,2] with [src,dst] positions")
        if edge_src_dst.dtype != torch.long:
            raise TypeError("edge_src_dst must use torch.long indices")
        # The feature is tiny (4x4 per graph edge), so evaluate the coordinate
        # cancellation in float64. CUDA float32 matmul may use TF32 and otherwise
        # turns an exact common-frame identity into ~1e-3 feature noise.
        source = T_ego_from_cam[:, edge_src_dst[:, 0]].double()
        destination = T_ego_from_cam[:, edge_src_dst[:, 1]].double()
        return (torch.linalg.inv(destination) @ source).float()

    def edge_features(
        self,
        intrinsics: torch.Tensor,
        T_ego_from_cam: torch.Tensor,
        *,
        image_hw: tuple[int, int],
        edge_src_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Build directed edge features with shape ``[B,E,20]``."""

        batch, views, height, width = self.validate_calibration(
            intrinsics, T_ego_from_cam, image_hw=image_hw
        )
        if edge_src_dst.device != intrinsics.device:
            raise ValueError("edge indices must share the calibration device")
        if edge_src_dst.numel() and (
            bool((edge_src_dst < 0).any().item())
            or bool((edge_src_dst >= views).any().item())
        ):
            raise ValueError("edge index lies outside the supplied cameras")
        relative = self.directed_relative_transforms(T_ego_from_cam, edge_src_dst)
        normalized = self.normalized_intrinsics(
            intrinsics, image_hw=(height, width)
        )
        src = edge_src_dst[:, 0]
        dst = edge_src_dst[:, 1]
        feature = torch.cat(
            (
                relative[..., :3, :3].reshape(batch, -1, 9),
                relative[..., :3, 3],
                normalized[:, src],
                normalized[:, dst],
            ),
            dim=-1,
        )
        if feature.shape != (batch, edge_src_dst.shape[0], self.feature_dim):
            raise RuntimeError("unexpected relative geometry feature shape")
        return feature

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        if edge_features.ndim != 3 or edge_features.shape[-1] != self.feature_dim:
            raise ValueError("edge_features must be [B,E,20]")
        parameter = next(self.encoder.parameters())
        return self.encoder(edge_features.to(device=parameter.device, dtype=parameter.dtype))

    def dense_edge_embeddings(
        self,
        intrinsics: torch.Tensor,
        T_ego_from_cam: torch.Tensor,
        *,
        image_hw: tuple[int, int],
        view_ids: torch.Tensor,
        view_valid: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter encoded existing edges to ``[B,V_dst,V_src,D_geo]``."""

        batch, views, height, width = self.validate_calibration(
            intrinsics,
            T_ego_from_cam,
            image_hw=image_hw,
            batch=int(view_ids.shape[0]),
            views=int(view_ids.shape[1]),
        )
        if view_ids.shape != (batch, views) or view_valid.shape != (batch, views):
            raise ValueError("view metadata must match calibration [B,V]")
        if view_ids.dtype != torch.long or view_valid.dtype != torch.bool:
            raise TypeError("view_ids must be long and view_valid must be bool")
        if view_ids.device != intrinsics.device or view_valid.device != intrinsics.device:
            raise ValueError("view metadata and calibration must share a device")
        if adjacency.device != intrinsics.device or adjacency.dtype != torch.bool:
            raise ValueError("adjacency must be bool on the calibration device")
        positional_graph = adjacency.ndim == 3
        if positional_graph:
            if adjacency.shape != (batch, views, views):
                raise ValueError("positional adjacency must be [B,V_dst,V_src]")
        elif adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("canonical adjacency must be a square matrix")
        parameter = next(self.encoder.parameters())
        if parameter.device != intrinsics.device:
            raise ValueError("geometry encoder and calibration must share a device")
        dense = torch.zeros(
            (batch, views, views, self.bottleneck_dim),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        feature_rows: list[torch.Tensor] = []
        pair_rows: list[list[list[int]]] = []
        edge_counts: list[int] = []
        for sample in range(batch):
            pairs: list[list[int]] = []
            ids = view_ids[sample]
            for dst_position in range(views):
                if not bool(view_valid[sample, dst_position].item()):
                    continue
                allowed = (
                    adjacency[sample, dst_position] if positional_graph
                    else adjacency[ids[dst_position], ids]
                ) & view_valid[sample]
                allowed[dst_position] = False
                for src_position in torch.nonzero(allowed, as_tuple=False).flatten().tolist():
                    pairs.append([src_position, dst_position])
            edge = torch.tensor(pairs, device=intrinsics.device, dtype=torch.long).reshape(-1, 2)
            feature = self.edge_features(
                intrinsics[sample : sample + 1],
                T_ego_from_cam[sample : sample + 1],
                image_hw=(height, width),
                edge_src_dst=edge,
            )
            embedding = self(feature)[0]
            for edge_index, (src_position, dst_position) in enumerate(pairs):
                dense[sample, dst_position, src_position] = embedding[edge_index]
            feature_rows.append(feature[0])
            pair_rows.append(pairs)
            edge_counts.append(len(pairs))
        self._last_edge_features = (
            torch.stack(feature_rows).detach()
            if len(set(edge_counts)) == 1
            else None
        )
        self._last_edge_pairs = pair_rows
        self._last_audit = {
            "feature_dim": self.feature_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "edge_count_per_batch": edge_counts,
            "edge_feature_shape": (
                list(self._last_edge_features.shape)
                if self._last_edge_features is not None
                else None
            ),
            "dense_embedding_shape": list(dense.shape),
            "edge_pairs_src_dst_by_batch": pair_rows,
            "extrinsic_convention": "T_ego_from_cam",
            "relative_convention": GEOMETRY_CONVENTION,
            "normalized_intrinsics": True,
            "global_vt_attention": False,
        }
        return dense


def attach_relative_geometry_conditioning(
    dit: nn.Module,
    *,
    bottleneck_dim: int = 64,
    gate_init: float = 0.0,
    enabled: bool = True,
) -> nn.Module:
    """Attach one shared encoder and small per-block FiLM projections to T3."""

    if not bool(getattr(dit, "mv_cross_view_parameters_attached", False)):
        raise ValueError("relative geometry must be attached after the T3 upgrade")
    if hasattr(dit, "tmv_geometry_encoder"):
        raise ValueError("relative geometry is already attached")
    if float(gate_init) != 0.0:
        raise ValueError("T4 pretrained preservation requires gate_init=0")
    residuals = dit.tmv_cross_view_residuals
    if not residuals:
        raise ValueError("T3 has no cross-view residuals")
    reference = next(dit.parameters())
    encoder = RelativeCameraGeometryEncoder(bottleneck_dim).to(
        device=reference.device, dtype=reference.dtype
    )
    dit.add_module("tmv_geometry_encoder", encoder)
    for residual in residuals:
        residual.attach_geometry_conditioning(encoder.bottleneck_dim)
    dit.mv_geometry_conditioning_attached = True
    dit.mv_geometry_conditioning_enabled = bool(enabled)
    dit.mv_geometry_mode = GEOMETRY_MODE
    dit.mv_geometry_feature_dim = GEOMETRY_FEATURE_DIM
    dit.mv_geometry_bottleneck_dim = encoder.bottleneck_dim
    dit.mv_geometry_gate_initialization = 0.0
    dit.mv_geometry_normalize_intrinsics = True
    dit.mv_geometry_global_vt_attention = False
    dit.mv_geometry_convention = GEOMETRY_CONVENTION
    return dit


def set_geometry_conditioning_enabled(dit: nn.Module, enabled: bool) -> None:
    if not bool(getattr(dit, "mv_geometry_conditioning_attached", False)):
        raise ValueError("relative geometry is not attached")
    dit.mv_geometry_conditioning_enabled = bool(enabled)


def geometry_gates(dit: nn.Module) -> list[nn.Parameter]:
    if not bool(getattr(dit, "mv_geometry_conditioning_attached", False)):
        raise ValueError("relative geometry is not attached")
    return [residual.geometry_gate for residual in dit.tmv_cross_view_residuals]
