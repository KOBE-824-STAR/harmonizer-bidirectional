"""Multi-view extensions that preserve DiffusionHarmonizer's official DiT weights.

Runtime-heavy modules, including the tensor layout helpers, are intentionally
not imported here.  Metadata tools can therefore run without PyTorch/Cosmos.
"""

from .constants import CAMERA_ORDER, OFFICIAL_PATHS, Architecture

__all__ = [
    "Architecture",
    "CAMERA_ORDER",
    "OFFICIAL_PATHS",
]
