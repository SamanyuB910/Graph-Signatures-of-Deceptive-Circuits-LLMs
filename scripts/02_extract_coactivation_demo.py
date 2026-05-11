"""Phase 2 verification gate.

Loads 20 honest_factual prompts, builds an aggregate co-activation graph at the
configured SAE hook, prints graph stats, saves the edge list to disk.

Usage:
    python scripts/02_extract_coactivation_demo.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import networkx as nx

from circuit_structure.coactivation import extract_coactivation_graph
from circuit_structure.config import dump_effective_config, load_config, make_run_id
from circuit_structure.io_utils import write_edgelist_gz
from circuit_structure.logging_setup import configure_logging, log_versions, pin_seeds
from circuit_structure.prompts import load_prompts
from circuit_structure.sae_loader import load_model_and_sae


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2: build demo co-activation graph")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--prompts", type=str, default="prompts/honest_factual.jsonl",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    run_dir = REPO_ROOT / "results" / "runs" / run_id

    logger = configure_logging(run_dir)
    pin_seeds(cfg.seed)
    logger.info(f"Run ID: {run_id}")
    log_versions(logger)
    dump_effective_config(cfg, run_dir / "effective_config.yaml")

    prompt_path = REPO_ROOT / args.prompts
    prompts = load_prompts(prompt_path)
    logger.info(f"Loaded {len(prompts)} prompts from {prompt_path}")

    model, sae = load_model_and_sae(cfg)

    t0 = time.time()
    G = extract_coactivation_graph(
        model=model,
        sae=sae,
        prompts=[p.text for p in prompts],
        hook_name=cfg.sae.sae_id,
        threshold=cfg.graph.activation_threshold,
        min_coactivation=cfg.graph.min_coactivation,
        aggregate_positions=cfg.graph.aggregate_positions,
        progress=True,
    )
    elapsed = time.time() - t0
    logger.info(f"Extraction took {elapsed:.1f}s")

    n, m = G.number_of_nodes(), G.number_of_edges()
    density_str = f"{nx.density(G):.4f}" if n >= 2 else "NA"
    logger.info(f"Graph stats: nodes={n}, edges={m}, density={density_str}")

    if n == 0 or m == 0:
        logger.error(
            "Graph is empty after thresholding - try lowering activation_threshold "
            "or min_coactivation in configs/default.yaml"
        )
        return 1

    out_path = run_dir / "graphs" / "demo" / "demo.edgelist.gz"
    write_edgelist_gz(G, out_path)
    logger.info(f"Wrote {out_path}")

    print(f"\nRUN_ID={run_id}")
    print(f"GRAPH_PATH={out_path}")

    logger.info("Phase 2 PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
