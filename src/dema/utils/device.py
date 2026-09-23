from __future__ import annotations


def resolve_device(spec: str | None) -> str:
    """Map ``auto`` to ``cuda`` when available, else ``cpu``."""
    if spec and spec != "auto":
        return spec
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


def set_global_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
