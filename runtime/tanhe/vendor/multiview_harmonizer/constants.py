"""Immutable project contracts and official checkpoint identities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Architecture(str, Enum):
    INDEPENDENT = "independent"
    FACTORIZED_ADAPTER = "A"
    JOINT_TOKENS = "B"


CAMERA_ORDER: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)


@dataclass(frozen=True)
class OfficialCheckpointPaths:
    harmonizer: Path
    cosmos_dit: Path
    tokenizer: Path
    harmonizer_sha256: str
    cosmos_dit_sha256: str
    tokenizer_sha256: str


OFFICIAL_PATHS = OfficialCheckpointPaths(
    harmonizer=Path(
        "/mnt/ojc/workplace2/harmonization/external/harmonizer/models/"
        "diffusion_harmonizer.pkl"
    ),
    cosmos_dit=Path(
        "/mnt/ojc/workplace2/harmonization/external/harmonizer/src/checkpoints/"
        "nvidia/Cosmos-Predict2-0.6B-Text2Image/model.pt"
    ),
    tokenizer=Path(
        "/mnt/ojc/workplace2/harmonization/external/harmonizer/src/checkpoints/"
        "nvidia/Cosmos-Predict2-0.6B-Text2Image/tokenizer/tokenizer.pth"
    ),
    harmonizer_sha256="6e65e9ec33019a49f7f0d15f81f5e089c2527e131bd5339d6a14f9b3531db524",
    cosmos_dit_sha256="566cf8b4eb8409a645cffc1f550974608170cc7ba2f5ba7a05a85adeb379029d",
    tokenizer_sha256="38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
)

UPSTREAM_COMMIT = "d1234ff7afb00b6084967dc31b18b526c9b9e7e0"
MODEL_FAMILY = "Cosmos-Predict2-0.6B-Text2Image"

