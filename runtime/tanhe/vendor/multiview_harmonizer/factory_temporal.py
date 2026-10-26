"""Construct official Harmonizer first, then attach Scheme A-TMV."""

from __future__ import annotations

from pathlib import Path

import torch
from cosmos_predict2.pipelines.text2image import Text2ImagePipeline

from .checkpoint import LoadAudit, load_official_harmonizer_checkpoint, require_sha256
from .constants import OFFICIAL_PATHS
from .dit_a_temporal import TemporalFactorizedMultiViewHarmonizerDiT
from .factory import build_official_pipeline_config
from .runtime_contract import require_official_text2image_dit_contract


def build_temporal_pipeline_with_official_harmonizer(
    *,
    cosmos_checkpoint_dir: str | Path,
    harmonizer_checkpoint: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    verify_known_hashes: bool = True,
    load_audit_output: str | Path | None = None,
    adapter_after_blocks: tuple[int, ...] = (4, 9, 14, 19),
    adapter_dim: int = 320,
    adapter_heads: int = 8,
    adapter_pool_factor: int = 4,
    max_views: int = 6,
) -> tuple[Text2ImagePipeline, LoadAudit]:
    cosmos_checkpoint_dir = Path(cosmos_checkpoint_dir).expanduser().resolve()
    harmonizer_checkpoint = Path(harmonizer_checkpoint).expanduser().resolve()
    if verify_known_hashes:
        # The current experiment contract accepts only the audited official assets.
        require_sha256(harmonizer_checkpoint, OFFICIAL_PATHS.harmonizer_sha256)
        require_sha256(cosmos_checkpoint_dir / "model.pt", OFFICIAL_PATHS.cosmos_dit_sha256)
        require_sha256(
            cosmos_checkpoint_dir / "tokenizer" / "tokenizer.pth",
            OFFICIAL_PATHS.tokenizer_sha256,
        )

    config = build_official_pipeline_config(cosmos_checkpoint_dir, dtype=dtype)
    pipeline = Text2ImagePipeline.from_config(
        config,
        use_text_encoder=False,
        dit_path=config.dit_path,
        device=str(device),
        torch_dtype=dtype,
    )
    pipeline.tensor_kwargs = {"device": str(device), "dtype": dtype}
    require_official_text2image_dit_contract(pipeline.dit)
    audit = load_official_harmonizer_checkpoint(
        pipeline,
        harmonizer_checkpoint,
        audit_output=load_audit_output,
    )
    pipeline.dit = TemporalFactorizedMultiViewHarmonizerDiT.upgrade_from_official(
        pipeline.dit,
        adapter_after_blocks=adapter_after_blocks,
        adapter_dim=adapter_dim,
        adapter_heads=adapter_heads,
        pool_factor=adapter_pool_factor,
        max_views=max_views,
    )
    return pipeline, audit
