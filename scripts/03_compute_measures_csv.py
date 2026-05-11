"""Phase 3 verification gate.

Loads the most recent demo graph (or one explicitly given via --graph), runs the
full graph-measure battery, prints the result, writes a one-row CSV.

Usage:
    python scripts/03_compute_measures_csv.py --config configs/default.yaml
    # or
    python scripts/03_compute_measures_csv.py --graph results/runs/<id>/graphs/demo/demo.edgelist.gz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from circuit_structure.config import dump_effective_config, load_config, make_run_id
from circuit_structure.graph_measures import compute_all_measures
from circuit_structure.io_utils import read_edgelist_gz
from circuit_structure.logging_setup import configure_logging, log_versions, pin_seeds


def _find_latest_demo_graph(repo_root: Path) -> Path | None:
    runs = sorted((repo_root / "results" / "runs").glob("*/graphs/demo/demo.edgelist.gz"))
    return runs[-1] if runs else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3: graph measures + CSV")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--graph", type=str, default=None,
        help="Path to .edgelist.gz; defaults to latest demo graph under results/runs/",
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

    graph_path = Path(args.graph) if args.graph else _find_latest_demo_graph(REPO_ROOT)
    if graph_path is None or not graph_path.exists():
        logger.error(
            "No demo graph found. Run scripts/02_extract_coactivation_demo.py first."
        )
        return 1
    logger.info(f"Reading graph: {graph_path}")
    G = read_edgelist_gz(graph_path)
    logger.info(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    t0 = time.time()
    measures = compute_all_measures(G, seed=cfg.seed)
    elapsed = time.time() - t0
    logger.info(f"Computed measures in {elapsed:.2f}s")

    logger.info("Measures:")
    for k, v in measures.items():
        logger.info(f"  {k}: {v}")

    measures["graph_path"] = str(graph_path)
    measures["condition"] = "demo"
    df = pd.DataFrame.from_records([measures])

    out_path = run_dir / "measures" / "demo.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}")

    print(f"\nRUN_ID={run_id}")
    print(f"MEASURES_PATH={out_path}")

    logger.info("Phase 3 PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
