"""Logging + reproducibility setup: stdout + run.log handler, RNG seed pinning,
and a one-shot library-version logger so every run.log records its env."""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np


def configure_logging(run_dir: str | Path, level: str = "INFO") -> logging.Logger:
    """Configure the root logger to write both to stdout and to <run_dir>/run.log.

    Removes any pre-existing handlers so re-runs in the same process don't
    double-log. Returns the root logger.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))

    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return root


def pin_seeds(seed: int) -> None:
    """Pin Python, NumPy, and (if importable) Torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def log_versions(logger: logging.Logger) -> None:
    """Log versions of every relevant library so run.log captures the env."""
    pkgs = [
        "torch",
        "transformer_lens",
        "sae_lens",
        "networkx",
        "numpy",
        "numba",
        "scipy",
        "pandas",
        "sklearn",
    ]
    for p in pkgs:
        try:
            mod = __import__(p)
            logger.info(f"  {p}: {getattr(mod, '__version__', 'unknown')}")
        except ImportError:
            logger.warning(f"  {p}: NOT INSTALLED")
