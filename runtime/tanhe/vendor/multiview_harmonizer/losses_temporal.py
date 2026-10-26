"""Scientifically conservative losses for Scheme A-TMV engineering smoke tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_view_l1(
    prediction_B_V_C_H_W: torch.Tensor,
    target_B_V_C_H_W: torch.Tensor,
    view_valid_B_V: torch.Tensor,
) -> torch.Tensor:
    """Mean pixel L1 over valid cameras only; invalid views contribute nothing."""

    if prediction_B_V_C_H_W.shape != target_B_V_C_H_W.shape:
        raise ValueError("prediction and target shapes differ")
    if prediction_B_V_C_H_W.ndim != 5:
        raise ValueError("prediction/target must be [B,V,C,H,W]")
    if view_valid_B_V.shape != prediction_B_V_C_H_W.shape[:2]:
        raise ValueError("view_valid must be [B,V]")
    if view_valid_B_V.dtype != torch.bool:
        raise TypeError("view_valid must be bool")
    valid_count = view_valid_B_V.sum()
    if not bool(valid_count.item()):
        raise ValueError("at least one view must be valid")
    per_element = F.l1_loss(
        prediction_B_V_C_H_W.float(),
        target_B_V_C_H_W.float(),
        reduction="none",
    )
    mask = view_valid_B_V[:, :, None, None, None].to(per_element.dtype)
    elements_per_view = prediction_B_V_C_H_W.shape[2] * prediction_B_V_C_H_W.shape[3] * prediction_B_V_C_H_W.shape[4]
    return (per_element * mask).sum() / (valid_count.to(per_element.dtype) * elements_per_view)


def require_geometry_for_multiview_loss(
    lambda_mv: float,
    *,
    calibrated_warps: torch.Tensor | None,
    overlap_occlusion_mask: torch.Tensor | None,
) -> None:
    """Reject a geometrically invalid cross-camera pixel-equality objective."""

    if lambda_mv < 0:
        raise ValueError("lambda_mv must be non-negative")
    if lambda_mv > 0 and (calibrated_warps is None or overlap_occlusion_mask is None):
        raise ValueError(
            "lambda_mv>0 requires calibrated warps and overlap/occlusion masks; "
            "naive cross-camera pixel equality is forbidden"
        )
