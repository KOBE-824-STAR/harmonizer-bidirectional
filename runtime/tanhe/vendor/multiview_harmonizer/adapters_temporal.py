"""Same-latent-time cross-view adapter for Scheme A-TMV."""

from __future__ import annotations

import torch

from .adapters import PooledCrossViewAdapter
from .layout_temporal import tokens_btview_to_bvthwd, tokens_bvthwd_to_btview


class SameTimePooledCrossViewAdapter(PooledCrossViewAdapter):
    """Apply the existing spatial adapter independently at every latent time.

    Input and output use ``[B,V,T,H,W,D]``.  Internally ``B*T`` is the
    effective batch, so attention can cross cameras but cannot cross temporal
    slots or leak information from future slots.
    """

    def forward(
        self,
        x_B_V_T_H_W_D: torch.Tensor,
        camera_ids_B_V: torch.Tensor,
        view_valid_B_V: torch.Tensor,
        plucker_B_V_H_W_6: torch.Tensor | None = None,
        adjacency_B_V_V: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_B_V_T_H_W_D.ndim != 6:
            raise ValueError("Adapter input must be [B,V,T,H,W,D]")
        batch_size, num_views, num_token_frames, height, width, dim = (
            x_B_V_T_H_W_D.shape
        )
        if camera_ids_B_V.shape != (batch_size, num_views):
            raise ValueError("camera_ids do not match [B,V]")
        if view_valid_B_V.shape != (batch_size, num_views):
            raise ValueError("view_valid does not match [B,V]")

        flat_BV_T_H_W_D = x_B_V_T_H_W_D.reshape(
            batch_size * num_views, num_token_frames, height, width, dim
        )
        same_time_BT_V_H_W_D = tokens_bvthwd_to_btview(
            flat_BV_T_H_W_D,
            batch_size=batch_size,
            num_views=num_views,
        )
        camera_ids_BT_V = camera_ids_B_V[:, None, :].expand(
            batch_size, num_token_frames, num_views
        ).reshape(batch_size * num_token_frames, num_views)
        view_valid_BT_V = view_valid_B_V[:, None, :].expand(
            batch_size, num_token_frames, num_views
        ).reshape(batch_size * num_token_frames, num_views)
        adjacency_BT_V_V = None
        if adjacency_B_V_V is not None:
            if adjacency_B_V_V.shape != (batch_size, num_views, num_views):
                raise ValueError("pose adjacency must match [B,V,V]")
            adjacency_BT_V_V = adjacency_B_V_V[:, None].expand(
                batch_size, num_token_frames, num_views, num_views,
            ).reshape(batch_size * num_token_frames, num_views, num_views)
        plucker_BT_V_1_H_W_6 = None
        if plucker_B_V_H_W_6 is not None:
            expected = (batch_size, num_views, height, width, 6)
            if tuple(plucker_B_V_H_W_6.shape) != expected:
                raise ValueError(
                    f"Plucker rays must have shape {expected}, got "
                    f"{tuple(plucker_B_V_H_W_6.shape)}"
                )
            plucker_BT_V_H_W_6 = plucker_B_V_H_W_6[:, None].expand(
                batch_size,
                num_token_frames,
                num_views,
                height,
                width,
                6,
            ).reshape(
                batch_size * num_token_frames,
                num_views,
                height,
                width,
                6,
            )
            plucker_BT_V_1_H_W_6 = plucker_BT_V_H_W_6.unsqueeze(2)

        # The spatial adapter expects an explicit singleton internal time axis.
        adapted_BT_V_1_H_W_D = super().forward(
            same_time_BT_V_H_W_D.unsqueeze(2),
            camera_ids_BT_V,
            view_valid_BT_V,
            plucker_BT_V_1_H_W_6,
            adjacency_BT_V_V,
        )
        restored_BV_T_H_W_D = tokens_btview_to_bvthwd(
            adapted_BT_V_1_H_W_D.squeeze(2),
            batch_size=batch_size,
            num_views=num_views,
            num_token_frames=num_token_frames,
        )
        return restored_BV_T_H_W_D.reshape(
            batch_size, num_views, num_token_frames, height, width, dim
        )
