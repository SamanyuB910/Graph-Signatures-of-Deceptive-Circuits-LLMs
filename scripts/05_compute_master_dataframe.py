"""Phase 5: compute the full graph-measure battery on every extracted graph.

Reads the manifest produced by scripts/04_build_condition_graphs.py, computes
measures for per-prompt + per-batch + aggregate graphs, writes a master CSV.

Caps:
  - treewidth_max_n=600  (per-prompt and per-batch fit; aggregates skipped — they are
    ~3000-node dense graphs that would take hours per graph for elimination orderings)
  - clique_max_n=200     (exact max-clique is exponential)

For NaN handling: per-prompt graphs with n<10 nodes get NaN treewidth (per the
plan), but our extraction yielded no such graphs.

Usage:
    python scripts/05_compute_master_dataframe.py --config configs/default.yaml
    python scripts/05_compute_master_dataframe.py --graphs-dir results/runs/<id>/graphs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from circuit_structure.config import dump_effective_config, load_config, make_run_id
from circuit_structure.graph_measures import compute_all_measures
from circuit_structure.io_utils import read_edgelist_gz
from circuit_structure.logging_setup import configure_logging, log_versions, pin_seeds

TREEWIDTH_CAP_PER_PROMPT = 600
TREEWIDTH_CAP_PER_BATCH = 600
TREEWIDTH_CAP_AGGREGATE = 0     # skip — aggregates are too dense for elimination
NUMBA_TRIALS = 16
MIN_NODES_FOR_TREEWIDTH = 10


def _find_latest_graphs_dir(repo_root: Path) -> Path | None:
    candidates = sorted(
        (repo_root / "results" / "runs").glob("*/graphs/manifest.json")
    )
    return candidates[-1].parent if candidates else None


def _measures_with_cap(G, *, treewidth_cap: int, seed: int) -> dict[str, float]:
    """compute_all_measures with a per-call treewidth cap (skip if too small/large)."""
    n = G.number_of_nodes()
    effective_cap = treewidth_cap if n >= MIN_NODES_FOR_TREEWIDTH else 0
    return compute_all_measures(
        G,
        include_spectral=True,
        include_min_cut=True,
        treewidth_max_n=effective_cap,
        clique_max_n=200,
        numba_random_trials=NUMBA_TRIALS,
        seed=seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--graphs-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    run_dir = REPO_ROOT / "results" / "runs" / run_id
    logger = configure_logging(run_dir)
    pin_seeds(cfg.seed)
    log_versions(logger)
    dump_effective_config(cfg, run_dir / "effective_config.yaml")
    logger.info(f"Run ID: {run_id}")

    graphs_dir = Path(args.graphs_dir) if args.graphs_dir else _find_latest_graphs_dir(REPO_ROOT)
    if graphs_dir is None or not (graphs_dir / "manifest.json").exists():
        logger.error("No graphs manifest found. Run scripts/04_build_condition_graphs.py first.")
        return 1
    logger.info(f"Using graphs dir: {graphs_dir}")

    manifest_path = graphs_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    src_run_dir = manifest_path.parent.parent

    rows: list[dict] = []
    grand_t0 = time.time()

    for cond, cmeta in manifest["conditions"].items():
        # Per-prompt
        logger.info(f"--- {cond}: per-prompt measures (n={len(cmeta['per_prompt'])}) ---")
        t0 = time.time()
        for pm in cmeta["per_prompt"]:
            G = read_edgelist_gz(src_run_dir / pm["graph_path"])
            m = _measures_with_cap(G, treewidth_cap=TREEWIDTH_CAP_PER_PROMPT, seed=cfg.seed)
            row = {
                "condition": cond,
                "graph_type": "per_prompt",
                "graph_id": pm["id"],
                "prompt_index": pm["prompt_index"],
                "prompt_text": pm["prompt_text"],
                **m,
            }
            rows.append(row)
        logger.info(f"  done in {time.time()-t0:.1f}s")

        # Per-batch
        logger.info(f"--- {cond}: per-batch measures (n={len(cmeta['per_batch'])}) ---")
        t0 = time.time()
        for bm in cmeta["per_batch"]:
            G = read_edgelist_gz(src_run_dir / bm["graph_path"])
            m = _measures_with_cap(G, treewidth_cap=TREEWIDTH_CAP_PER_BATCH, seed=cfg.seed)
            row = {
                "condition": cond,
                "graph_type": "per_batch",
                "graph_id": bm["batch_id"],
                "prompt_index": bm["batch_index"],
                "prompt_text": ",".join(bm["prompt_ids"]),
                **m,
            }
            rows.append(row)
        logger.info(f"  done in {time.time()-t0:.1f}s")

        # Aggregate
        logger.info(f"--- {cond}: aggregate measure (skipping treewidth, n={cmeta['aggregate']['n_nodes']}) ---")
        t0 = time.time()
        am = cmeta["aggregate"]
        G = read_edgelist_gz(src_run_dir / am["graph_path"])
        m = _measures_with_cap(G, treewidth_cap=TREEWIDTH_CAP_AGGREGATE, seed=cfg.seed)
        row = {
            "condition": cond,
            "graph_type": "aggregate",
            "graph_id": "aggregate",
            "prompt_index": -1,
            "prompt_text": f"AGGREGATE_{am['n_prompts']}_PROMPTS",
            **m,
        }
        rows.append(row)
        logger.info(f"  done in {time.time()-t0:.1f}s")

    df = pd.DataFrame.from_records(rows)
    user_cols = [
        "condition", "graph_type", "graph_id", "prompt_index", "prompt_text",
        "num_nodes", "num_edges", "density",
        "treewidth_minfill", "treewidth_mindeg", "treewidth_numba", "treewidth_best",
        "modularity", "num_communities", "max_clique", "avg_clustering",
        "avg_degree", "max_degree", "degree_variance",
        "num_components", "min_vertex_cut",
        "algebraic_connectivity", "spectral_gap",
    ]
    df = df[user_cols]

    out_path = run_dir / "measures" / "all_conditions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote master CSV: {out_path}  ({len(df)} rows)")

    logger.info(f"Total time: {time.time()-grand_t0:.1f}s")

    logger.info("=" * 70)
    logger.info("MEASURE SUMMARY (per condition x graph_type)")
    logger.info("=" * 70)
    summary = df.groupby(["condition", "graph_type"]).agg({
        "num_nodes": "mean",
        "num_edges": "mean",
        "density": "mean",
        "treewidth_best": "mean",
        "modularity": "mean",
        "avg_clustering": "mean",
    }).round(3)
    for line in summary.to_string().split("\n"):
        logger.info(line)

    print(f"\nRUN_ID={run_id}")
    print(f"MEASURES_PATH={out_path}")
    logger.info("Phase 5 PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
