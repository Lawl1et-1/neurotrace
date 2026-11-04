"""Small shared helpers: seeding, device selection, paths."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("NEUROTRACE_DATA", ROOT / "data"))
RESULTS_DIR = ROOT / "results"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is optional for the data-only parts
        pass


def get_device(name: str = "auto") -> str:
    """Return "cuda" when available and requested, else "cpu"."""
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name
