"""Experiment configuration: YAML on disk, frozen dataclasses in memory.

Pattern: load_config(yaml_path) -> ExperimentConfig. Unknown keys raise rather
than silently passing through. dump_effective_config writes the canonical YAML
serialization next to results so each run is reproducible from its own outputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True)
class SAEConfig:
    release: str = "gpt2-small-res-jb"
    sae_id: str = "blocks.8.hook_resid_pre"
    device: Literal["cpu"] = "cpu"


@dataclass(frozen=True)
class GraphConfig:
    activation_threshold: float = 0.01
    min_coactivation: int = 5
    aggregate_positions: Literal["all", "last", "mean"] = "all"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    model_name: str = "gpt2"
    layer_index: int = 8
    n_prompts_per_condition: int = 50
    sae: SAEConfig = field(default_factory=SAEConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    conditions: tuple[str, ...] = ("honest_factual",)


_VALID_TOP = {"seed", "model_name", "layer_index", "n_prompts_per_condition", "conditions"}
_VALID_SAE = {"release", "sae_id", "device"}
_VALID_GRAPH = {"activation_threshold", "min_coactivation", "aggregate_positions"}


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML config and construct ExperimentConfig.

    Unknown top-level, sae, or graph keys raise ValueError to prevent typo-driven
    silent defaults.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    sae_raw = raw.pop("sae", {}) or {}
    graph_raw = raw.pop("graph", {}) or {}

    bad_top = set(raw) - _VALID_TOP
    bad_sae = set(sae_raw) - _VALID_SAE
    bad_graph = set(graph_raw) - _VALID_GRAPH
    if bad_top or bad_sae or bad_graph:
        raise ValueError(
            f"Unknown config keys: top={bad_top}, sae={bad_sae}, graph={bad_graph}"
        )

    if "conditions" in raw and isinstance(raw["conditions"], list):
        raw["conditions"] = tuple(raw["conditions"])

    return ExperimentConfig(
        sae=SAEConfig(**sae_raw),
        graph=GraphConfig(**graph_raw),
        **raw,
    )


def config_hash(cfg: ExperimentConfig) -> str:
    """Stable short hash of config (8 hex chars) for run-id derivation."""
    s = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def make_run_id(cfg: ExperimentConfig) -> str:
    """run_id = YYYYMMDD-HHMMSS-<8-char config hash>."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{config_hash(cfg)}"


def dump_effective_config(cfg: ExperimentConfig, out_path: str | Path) -> None:
    """Write the in-memory config back to YAML alongside results."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=True, default_flow_style=False)
