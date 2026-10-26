"""Token-centre Plucker rays for calibrated multi-view feature maps.

The public calibration convention is ``T_ego_from_cam``.  A ray is represented
as ``(direction_ego, origin_ego x direction_ego)``.  Geometry is evaluated in
float64 and returned in float32; downstream modules cast it to their parameter
dtype.  The same rig-local rays may be reused for each latent time slot because
the cross-view paths never mix different temporal slots.
"""

from __future__ import annotations

import torch


PLUCKER_DIM = 6
PLUCKER_CONVENTION = "(d_ego, o_ego_cross_d_ego)"


def validate_plucker_calibration(
    intrinsics: torch.Tensor,
    T_ego_from_cam: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    feature_hw: tuple[int, int],
) -> tuple[int, int, int, int, int, int]:
    if not isinstance(intrinsics, torch.Tensor) or not isinstance(
        T_ego_from_cam, torch.Tensor
    ):
        raise TypeError("intrinsics and T_ego_from_cam must be tensors")
    if intrinsics.ndim != 4 or tuple(intrinsics.shape[-2:]) != (3, 3):
        raise ValueError("intrinsics must be [B,V,3,3]")
    if T_ego_from_cam.ndim != 4 or tuple(T_ego_from_cam.shape[-2:]) != (4, 4):
        raise ValueError("T_ego_from_cam must be [B,V,4,4]")
    if intrinsics.shape[:2] != T_ego_from_cam.shape[:2]:
        raise ValueError("calibration tensors must share [B,V]")
    if intrinsics.device != T_ego_from_cam.device:
        raise ValueError("calibration tensors must share a device")
    if not intrinsics.is_floating_point() or not T_ego_from_cam.is_floating_point():
        raise TypeError("calibration tensors must be floating point")
    if not bool(torch.isfinite(intrinsics).all().item()) or not bool(
        torch.isfinite(T_ego_from_cam).all().item()
    ):
        raise ValueError("calibration contains non-finite values")
    if len(image_hw) != 2 or len(feature_hw) != 2:
        raise ValueError("image_hw and feature_hw must be (H,W)")
    image_h, image_w = map(int, image_hw)
    feature_h, feature_w = map(int, feature_hw)
    if min(image_h, image_w, feature_h, feature_w) < 1:
        raise ValueError("image and feature dimensions must be positive")

    k = intrinsics.double()
    expected_k_bottom = k.new_tensor([0.0, 0.0, 1.0]).expand_as(k[..., 2, :])
    if not torch.allclose(k[..., 2, :], expected_k_bottom, atol=1e-6, rtol=0):
        raise ValueError("intrinsics bottom row must be [0,0,1]")
    if bool((torch.linalg.det(k).abs() < 1e-12).any().item()):
        raise ValueError("intrinsics must be invertible")

    pose = T_ego_from_cam.double()
    expected_pose_bottom = pose.new_tensor([0.0, 0.0, 0.0, 1.0]).expand_as(
        pose[..., 3, :]
    )
    if not torch.allclose(
        pose[..., 3, :], expected_pose_bottom, atol=1e-6, rtol=0
    ):
        raise ValueError("T_ego_from_cam bottom row must be [0,0,0,1]")
    rotation = pose[..., :3, :3]
    eye = torch.eye(3, device=pose.device, dtype=pose.dtype).expand_as(rotation)
    if not torch.allclose(
        rotation.transpose(-1, -2) @ rotation, eye, atol=1e-4, rtol=0
    ):
        raise ValueError("camera rotations must be orthonormal")
    if not torch.allclose(
        torch.linalg.det(rotation),
        torch.ones_like(torch.linalg.det(rotation)),
        atol=1e-4,
        rtol=0,
    ):
        raise ValueError("camera rotations must have determinant +1")
    batch, views = map(int, intrinsics.shape[:2])
    return batch, views, image_h, image_w, feature_h, feature_w


def token_center_plucker_rays(
    intrinsics: torch.Tensor,
    T_ego_from_cam: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    feature_hw: tuple[int, int],
) -> torch.Tensor:
    """Return float32 rays with shape ``[B,V,H_feature,W_feature,6]``."""

    batch, views, image_h, image_w, feature_h, feature_w = (
        validate_plucker_calibration(
            intrinsics,
            T_ego_from_cam,
            image_hw=image_hw,
            feature_hw=feature_hw,
        )
    )
    device = intrinsics.device
    dtype = torch.float64
    # The dataset rescales K directly with image width/height.  Use cell centres
    # under the same convention so uniformly resizing K and image_hw is exact.
    y = (torch.arange(feature_h, device=device, dtype=dtype) + 0.5) * (
        float(image_h) / float(feature_h)
    )
    x = (torch.arange(feature_w, device=device, dtype=dtype) + 0.5) * (
        float(image_w) / float(feature_w)
    )
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    pixels = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)

    inverse_k = torch.linalg.inv(intrinsics.double())
    direction_cam = torch.einsum("bvij,hwj->bvhwi", inverse_k, pixels)
    direction_cam = torch.nn.functional.normalize(direction_cam, dim=-1)
    rotation = T_ego_from_cam[..., :3, :3].double()
    direction_ego = torch.einsum("bvij,bvhwj->bvhwi", rotation, direction_cam)
    direction_ego = torch.nn.functional.normalize(direction_ego, dim=-1)
    origin_ego = T_ego_from_cam[..., :3, 3].double()
    origin_grid = origin_ego[:, :, None, None, :].expand(
        batch, views, feature_h, feature_w, 3
    )
    moment = torch.cross(origin_grid, direction_ego, dim=-1)
    rays = torch.cat((direction_ego, moment), dim=-1).float()
    if rays.shape != (batch, views, feature_h, feature_w, PLUCKER_DIM):
        raise RuntimeError("unexpected Plucker ray shape")
    if not bool(torch.isfinite(rays).all().item()):
        raise RuntimeError("Plucker ray construction produced non-finite values")
    return rays

