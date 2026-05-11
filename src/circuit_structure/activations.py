"""Single source of truth for SAE feature activations on a prompt.

Uses TransformerLens's names_filter to cache only the requested hook, which
avoids the per-layer memory blowup of full-cache forward passes (important
on CPU where every wasted allocation hurts).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sae_lens import SAE
    from transformer_lens import HookedTransformer


@torch.no_grad()
def feature_activations_for_prompt(
    model: "HookedTransformer",
    sae: "SAE",
    prompt: str,
    hook_name: str,
) -> torch.Tensor:
    """Return SAE feature activations at the given hook for one prompt.

    Returns a tensor of shape (n_positions, n_features) on CPU. Squeezes the
    leading batch dim. No gradients tracked.
    """
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(
        tokens, names_filter=lambda n: n == hook_name
    )
    activations = cache[hook_name]
    feature_acts = sae.encode(activations)
    if feature_acts.ndim == 3:
        feature_acts = feature_acts[0]
    return feature_acts.detach().cpu()
