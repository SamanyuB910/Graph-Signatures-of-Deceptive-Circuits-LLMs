"""Patch script: recompute max_clique on existing all_conditions.csv.

Phase 5 originally produced all-NaN max_clique because nx.graph_clique_number was
removed in NetworkX 3.x. graph_measures.py is now fixed; this script applies the
fix to the existing CSV so we don't have to redo Phase 5's slow aggregate work.

Only patches per_prompt and per_batch rows under the n=200 cap. Aggregate rows
stay NaN (their graphs are 3000+ nodes; exact max-clique is exponential).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import networkx as nx
import pandas as pd

from circuit_structure.io_utils import read_edgelist_gz

CLIQUE_MAX_N = 200


def _max_clique_size(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    return float(max(len(c) for c in nx.find_cliques(G)))


def main() -> int:
    csv_candidates = sorted(
        (REPO_ROOT / "results" / "runs").glob("*/measures/all_conditions.csv")
    )
    if not csv_candidates:
        print("ERROR: no all_conditions.csv found")
        return 1
    csv_path = csv_candidates[-1]
    print(f"Patching: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df)}, max_clique NaN before: {df['max_clique'].isna().sum()}")

    manifest_candidates = sorted(
        (REPO_ROOT / "results" / "runs").glob("*/graphs/manifest.json")
    )
    if not manifest_candidates:
        print("ERROR: no manifest.json found")
        return 1
    manifest_path = manifest_candidates[-1]
    src_run_dir = manifest_path.parent.parent
    print(f"Reading manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    path_map: dict[tuple[str, str, str], str] = {}
    for cond, cmeta in manifest["conditions"].items():
        for pm in cmeta["per_prompt"]:
            path_map[(cond, "per_prompt", pm["id"])] = pm["graph_path"]
        for bm in cmeta["per_batch"]:
            path_map[(cond, "per_batch", bm["batch_id"])] = bm["graph_path"]
        path_map[(cond, "aggregate", "aggregate")] = cmeta["aggregate"]["graph_path"]

    new_vals: list[float] = []
    n_patched = 0
    n_skipped_size = 0
    n_skipped_missing = 0
    for _, row in df.iterrows():
        key = (row["condition"], row["graph_type"], row["graph_id"])
        if row["graph_type"] == "aggregate":
            new_vals.append(float("nan"))
            continue
        if int(row["num_nodes"]) > CLIQUE_MAX_N:
            new_vals.append(float("nan"))
            n_skipped_size += 1
            continue
        if key not in path_map:
            new_vals.append(float("nan"))
            n_skipped_missing += 1
            continue
        G = read_edgelist_gz(src_run_dir / path_map[key])
        new_vals.append(_max_clique_size(G))
        n_patched += 1

    df["max_clique"] = new_vals
    df.to_csv(csv_path, index=False)
    print(f"Patched: {n_patched}, skipped_size>{CLIQUE_MAX_N}: {n_skipped_size}, skipped_missing: {n_skipped_missing}")
    print(f"max_clique NaN after: {df['max_clique'].isna().sum()}")

    print("\nmax_clique stats by condition x graph_type:")
    summary = df.groupby(["condition", "graph_type"])["max_clique"].agg(
        ["count", "mean", "std", "min", "max"]
    ).round(2)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
