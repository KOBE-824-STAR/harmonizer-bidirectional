"""Fail fast if the runtime is not DiffusionHarmonizer's 0.6B Text2Image DiT."""

from __future__ import annotations

import inspect


def require_official_text2image_dit_contract(dit) -> None:
    module_name = type(dit).__module__
    if module_name != "cosmos_predict2.models.text2image_dit":
        raise RuntimeError(
            "Expected DiffusionHarmonizer's Text2Image DiT, got "
            f"{module_name}.{type(dit).__name__}"
        )
    expected = {
        "model_channels": 1280,
        "num_blocks": 20,
        "num_heads": 20,
        "patch_spatial": 2,
        "patch_temporal": 1,
        "in_channels": 16,
        "out_channels": 16,
    }
    mismatches = {
        name: (value, getattr(dit, name, None))
        for name, value in expected.items()
        if getattr(dit, name, None) != value
    }
    if len(getattr(dit, "blocks", ())) != expected["num_blocks"]:
        mismatches["len(blocks)"] = (expected["num_blocks"], len(getattr(dit, "blocks", ())))
    required_forward_args = {"x_B_C_T_H_W", "timesteps_B_T", "crossattn_emb"}
    actual_forward_args = set(inspect.signature(type(dit).forward).parameters)
    missing_args = required_forward_args - actual_forward_args
    if mismatches or missing_args:
        raise RuntimeError(
            "Runtime does not match the official 0.6B Text2Image DiT contract: "
            f"attribute_mismatches={mismatches}, missing_forward_args={sorted(missing_args)}"
        )

