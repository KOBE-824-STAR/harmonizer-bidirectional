"""Build the official DiffusionHarmonizer first, then attach scheme A or B."""

from __future__ import annotations

from pathlib import Path

import torch
from cosmos_predict2.configs.base.config_text2image import get_cosmos_predict2_text2image_pipeline
from cosmos_predict2.pipelines.text2image import Text2ImagePipeline

from .checkpoint import LoadAudit, load_official_harmonizer_checkpoint, require_sha256
from .constants import OFFICIAL_PATHS, Architecture
from .dit_a import FactorizedMultiViewHarmonizerDiT
from .dit_b import JointTokenMultiViewHarmonizerDiT
from .dit_b_warm import WarmMultiViewHarmonizerDiT
from .runtime_contract import require_official_text2image_dit_contract


def build_official_pipeline_config(
    cosmos_checkpoint_dir: str | Path,
    dtype: torch.dtype = torch.bfloat16,
):
    checkpoint_dir = Path(cosmos_checkpoint_dir).expanduser().resolve()
    dit_path = checkpoint_dir / "model.pt"
    tokenizer_path = checkpoint_dir / "tokenizer" / "tokenizer.pth"
    missing = [str(path) for path in (dit_path, tokenizer_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete official Cosmos 0.6B checkpoint: {missing}")
    config = get_cosmos_predict2_text2image_pipeline(model_size="0.6B", fast_tokenizer=False)
    config.dit_path = str(dit_path)
    config.tokenizer["vae_pth"] = str(tokenizer_path)
    config.guardrail_config.enabled = False
    config.net.atten_backend = "torch"
    precision_names = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }
    if dtype not in precision_names:
        raise ValueError(f"Unsupported pipeline dtype: {dtype}")
    config.precision = precision_names[dtype]
    return config


def build_pipeline_with_official_harmonizer(
    *,
    architecture: Architecture | str,
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
    attention_variant: str = "dense",
    camera_adjacency: dict[str, tuple[str, ...]] | None = None,
    camera_schema_version: str = "canonical_v1",
    adjacency_undirected: bool = False,
) -> tuple[Text2ImagePipeline, LoadAudit]:
    """Construct only from the official 0.6B + DiffusionHarmonizer weights."""

    architecture = Architecture(architecture)
    cosmos_checkpoint_dir = Path(cosmos_checkpoint_dir).expanduser().resolve()
    harmonizer_checkpoint = Path(harmonizer_checkpoint).expanduser().resolve()
    if verify_known_hashes:
        if harmonizer_checkpoint == OFFICIAL_PATHS.harmonizer:
            require_sha256(harmonizer_checkpoint, OFFICIAL_PATHS.harmonizer_sha256)
        if cosmos_checkpoint_dir / "model.pt" == OFFICIAL_PATHS.cosmos_dit:
            require_sha256(OFFICIAL_PATHS.cosmos_dit, OFFICIAL_PATHS.cosmos_dit_sha256)
        if cosmos_checkpoint_dir / "tokenizer" / "tokenizer.pth" == OFFICIAL_PATHS.tokenizer:
            require_sha256(OFFICIAL_PATHS.tokenizer, OFFICIAL_PATHS.tokenizer_sha256)

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
    # Critical ordering: load the official checkpoint before any A/B parameter
    # exists.  The strict audit therefore covers exactly the official live DiT.
    audit = load_official_harmonizer_checkpoint(
        pipeline,
        harmonizer_checkpoint,
        audit_output=load_audit_output,
    )

    if architecture is Architecture.JOINT_TOKENS:
        if attention_variant == "dense":
            pipeline.dit = JointTokenMultiViewHarmonizerDiT.upgrade_from_official(
                pipeline.dit,
                max_views=max_views,
            )
            pipeline.dit.mv_attention_variant = "dense"
            pipeline.dit.mv_camera_schema_version = camera_schema_version
        elif attention_variant == "warm_v1":
            if camera_adjacency is None:
                raise ValueError("B-warm construction requires camera_adjacency")
            pipeline.dit = WarmMultiViewHarmonizerDiT.upgrade_from_official(
                pipeline.dit,
                camera_adjacency=camera_adjacency,
                camera_schema_version=camera_schema_version,
                adjacency_undirected=adjacency_undirected,
                max_views=max_views,
            )
        else:
            raise ValueError(f"Unknown Scheme B attention_variant {attention_variant!r}")
    else:
        layers = () if architecture is Architecture.INDEPENDENT else adapter_after_blocks
        pipeline.dit = FactorizedMultiViewHarmonizerDiT.upgrade_from_official(
            pipeline.dit,
            adapter_after_blocks=layers,
            adapter_dim=adapter_dim,
            adapter_heads=adapter_heads,
            pool_factor=adapter_pool_factor,
            max_views=max_views,
        )
    return pipeline, audit
