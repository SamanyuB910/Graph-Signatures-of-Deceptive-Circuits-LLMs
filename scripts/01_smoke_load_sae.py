"""Phase 1 verification gate.

Loads GPT-2 small + the configured SAE on CPU, runs one prompt through both,
and asserts the SAE produced a non-empty feature activation tensor.

Exit code 0 = pass, non-zero = fail. Output also captured to results/runs/<id>/run.log.

Usage:
    python scripts/01_smoke_load_sae.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from circuit_structure.config import dump_effective_config, load_config, make_run_id
from circuit_structure.logging_setup import configure_logging, log_versions, pin_seeds
from circuit_structure.sae_loader import load_model_and_sae


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1: SAE load smoke test")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    run_dir = REPO_ROOT / "results" / "runs" / run_id

    logger = configure_logging(run_dir)
    pin_seeds(cfg.seed)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Run directory: {run_dir}")
    logger.info("Library versions:")
    log_versions(logger)

    dump_effective_config(cfg, run_dir / "effective_config.yaml")

    import torch

    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.warning(
            "CUDA is available but config pins device='cpu'. Continuing on CPU."
        )

    model, sae = load_model_and_sae(cfg)

    prompt = "The capital of France is"
    logger.info(f"Test prompt: {prompt!r}")
    tokens = model.to_tokens(prompt)
    logger.info(f"Token shape: {tuple(tokens.shape)}")

    hook_name = cfg.sae.sae_id
    _, cache = model.run_with_cache(
        tokens, names_filter=lambda n: n == hook_name
    )
    activations = cache[hook_name]
    logger.info(f"Activation shape at {hook_name}: {tuple(activations.shape)}")

    feature_acts = sae.encode(activations)
    logger.info(f"Feature activation shape: {tuple(feature_acts.shape)}")

    n_nonzero = int((feature_acts > cfg.graph.activation_threshold).sum().item())
    logger.info(
        f"Non-zero features (above threshold {cfg.graph.activation_threshold}): "
        f"{n_nonzero}"
    )

    if n_nonzero < 10:
        logger.error(
            f"Too few active features ({n_nonzero}) - threshold may be wrong "
            f"or SAE may have failed to load correctly."
        )
        return 1

    logger.info("Phase 1 smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
