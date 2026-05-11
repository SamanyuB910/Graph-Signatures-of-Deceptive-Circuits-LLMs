"""Load TransformerLens model + SAELens SAE on CPU.

Wraps the SAELens version-quirk where SAE.from_pretrained sometimes returns a
3-tuple (sae, cfg_dict, sparsity) and sometimes returns the bare SAE. Asserts
the loaded SAE's hook_name matches the requested sae_id and refuses to proceed
silently on mismatch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

from .config import ExperimentConfig

logger = logging.getLogger(__name__)


def load_model_and_sae(cfg: ExperimentConfig) -> tuple["HookedTransformer", "SAE"]:
    """Load model + SAE on CPU and validate the SAE binds to the expected hook."""
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    logger.info(f"Loading model {cfg.model_name!r} on cpu...")
    model = HookedTransformer.from_pretrained(cfg.model_name, device="cpu")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded {cfg.model_name}: ~{n_params:,} parameters")

    logger.info(
        f"Loading SAE: release={cfg.sae.release!r}, sae_id={cfg.sae.sae_id!r}"
    )
    result = SAE.from_pretrained(
        release=cfg.sae.release,
        sae_id=cfg.sae.sae_id,
        device="cpu",
    )
    sae = result[0] if isinstance(result, tuple) else result

    actual_hook = getattr(sae.cfg, "hook_name", None)
    if actual_hook != cfg.sae.sae_id:
        raise RuntimeError(
            f"SAE hook mismatch: SAE.cfg.hook_name={actual_hook!r} != "
            f"requested sae_id={cfg.sae.sae_id!r}. Refusing to proceed silently."
        )

    d_sae = getattr(sae.cfg, "d_sae", None)
    logger.info(f"Loaded SAE: d_sae={d_sae}, hook={actual_hook!r}")

    return model, sae
