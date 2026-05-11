"""Phase 4-5 combined: verify behavioral contrast + extract all condition graphs.

Approach: load model+SAE once, then for each prompt:
  1. extract feature activations once
  2. build per-prompt counters
  3. accumulate counters into per-batch (5 prompts) and per-condition buckets

This avoids 3x redundant inference (per-prompt + per-batch + aggregate would
otherwise re-run the same forward passes). For 50 prompts x 3 conditions,
inference runs once per prompt = ~150 forward passes total.

Verification step: also generate 5 short greedy completions per condition for
visual inspection of the prompt distributions.

Saves to results/runs/<id>/:
  graphs/<condition>/per_prompt/<prompt_id>.edgelist.gz   (50 per condition)
  graphs/<condition>/per_batch/batch_<NNN>.edgelist.gz    (10 per condition, batch=5)
  graphs/<condition>/aggregate.edgelist.gz                (1 per condition, all 50 prompts)
  graphs/manifest.json                                     (prompts + sample completions)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import networkx as nx
import numpy as np
import torch

from circuit_structure.activations import feature_activations_for_prompt
from circuit_structure.config import dump_effective_config, load_config, make_run_id
from circuit_structure.io_utils import write_edgelist_gz
from circuit_structure.logging_setup import configure_logging, log_versions, pin_seeds
from circuit_structure.prompts import load_prompts
from circuit_structure.sae_loader import load_model_and_sae

CONDITIONS = ["honest_factual", "confabulation", "sycophantic"]
BATCH_SIZE = 5
N_SAMPLE_COMPLETIONS = 5
SAMPLE_NEW_TOKENS = 25


def pair_counts_from_activations(
    feats_np: np.ndarray, threshold: float
) -> tuple[dict[tuple[int, int], int], dict[int, int]]:
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    feat_counts: dict[int, int] = defaultdict(int)
    active_mask = feats_np > threshold
    for pos_idx in range(active_mask.shape[0]):
        active = np.where(active_mask[pos_idx])[0]
        k = active.size
        if k == 0:
            continue
        for f in active:
            feat_counts[int(f)] += 1
        if k >= 2:
            iu, ju = np.triu_indices(k, k=1)
            for a, b in zip(active[iu], active[ju], strict=True):
                u, v = (int(a), int(b)) if a < b else (int(b), int(a))
                pair_counts[(u, v)] += 1
    return pair_counts, feat_counts


def merge_into(
    target_pair: dict[tuple[int, int], int], target_feat: dict[int, int],
    src_pair: dict[tuple[int, int], int], src_feat: dict[int, int],
) -> None:
    for k, v in src_pair.items():
        target_pair[k] += v
    for k, v in src_feat.items():
        target_feat[k] += v


def build_graph(
    pair_counts: dict[tuple[int, int], int],
    feat_counts: dict[int, int],
    min_coactivation: int,
) -> nx.Graph:
    G = nx.Graph()
    for f, c in feat_counts.items():
        G.add_node(f, count=c)
    for (u, v), c in pair_counts.items():
        if c >= min_coactivation:
            G.add_edge(u, v, weight=c)
    G.remove_nodes_from([n for n, d in G.degree() if d == 0])
    return G


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    run_dir = REPO_ROOT / "results" / "runs" / run_id
    logger = configure_logging(run_dir)
    pin_seeds(cfg.seed)
    log_versions(logger)
    dump_effective_config(cfg, run_dir / "effective_config.yaml")
    logger.info(f"Run ID: {run_id}")

    model, sae = load_model_and_sae(cfg)
    hook_name = cfg.sae.sae_id
    threshold = cfg.graph.activation_threshold
    min_coact = cfg.graph.min_coactivation

    all_prompts = {
        c: load_prompts(REPO_ROOT / "prompts" / f"{c}.jsonl")
        for c in CONDITIONS
    }
    for c, ps in all_prompts.items():
        logger.info(f"Loaded {len(ps)} prompts for condition '{c}'")

    # ===== Behavioral contrast verification =====
    logger.info("=" * 70)
    logger.info("BEHAVIORAL CONTRAST VERIFICATION")
    logger.info("=" * 70)
    samples: dict[str, list[dict]] = {}
    for cond in CONDITIONS:
        samples[cond] = []
        for p in all_prompts[cond][:N_SAMPLE_COMPLETIONS]:
            tokens = model.to_tokens(p.text)
            with torch.no_grad():
                generated = model.generate(
                    tokens, max_new_tokens=SAMPLE_NEW_TOKENS,
                    temperature=0.0, do_sample=False, verbose=False,
                )
            full_text = model.to_string(generated[0])
            full_text = full_text[0] if isinstance(full_text, list) else full_text
            if full_text.startswith(p.text):
                completion = full_text[len(p.text):].strip()
            else:
                completion = full_text.strip()
            samples[cond].append(
                {"id": p.id, "prompt": p.text, "completion": completion}
            )
            logger.info(f"[{cond}|{p.id}] {p.text[:80]}")
            logger.info(f"   ->  {completion[:140]}")

    # ===== Graph extraction =====
    logger.info("=" * 70)
    logger.info("EXTRACTING CO-ACTIVATION GRAPHS (single-pass, cached)")
    logger.info("=" * 70)

    manifest = {
        "run_id": run_id,
        "config": {
            "threshold": threshold,
            "min_coactivation": min_coact,
            "aggregate_positions": cfg.graph.aggregate_positions,
            "hook_name": hook_name,
            "model": cfg.model_name,
            "batch_size": BATCH_SIZE,
        },
        "conditions": {},
        "samples": samples,
    }

    grand_t0 = time.time()
    for cond in CONDITIONS:
        prompts = all_prompts[cond]
        cond_dir = run_dir / "graphs" / cond
        pp_dir = cond_dir / "per_prompt"
        pb_dir = cond_dir / "per_batch"
        pp_dir.mkdir(parents=True, exist_ok=True)
        pb_dir.mkdir(parents=True, exist_ok=True)

        agg_pair: dict[tuple[int, int], int] = defaultdict(int)
        agg_feat: dict[int, int] = defaultdict(int)
        batch_pair: dict[tuple[int, int], int] = defaultdict(int)
        batch_feat: dict[int, int] = defaultdict(int)

        per_prompt_meta = []
        per_batch_meta = []

        t0 = time.time()
        for i, p in enumerate(prompts):
            feats = feature_activations_for_prompt(model, sae, p.text, hook_name)
            feats_np = feats.numpy()

            pp_pair, pp_feat = pair_counts_from_activations(feats_np, threshold)

            G_pp = build_graph(pp_pair, pp_feat, min_coact)
            pp_path = pp_dir / f"{p.id}.edgelist.gz"
            write_edgelist_gz(G_pp, pp_path)
            per_prompt_meta.append({
                "id": p.id,
                "prompt_text": p.text,
                "prompt_index": i,
                "n_nodes": G_pp.number_of_nodes(),
                "n_edges": G_pp.number_of_edges(),
                "graph_path": str(pp_path.relative_to(run_dir)),
            })

            merge_into(batch_pair, batch_feat, pp_pair, pp_feat)
            merge_into(agg_pair, agg_feat, pp_pair, pp_feat)

            if (i + 1) % BATCH_SIZE == 0:
                b_idx = i // BATCH_SIZE
                G_b = build_graph(batch_pair, batch_feat, min_coact)
                batch_id = f"batch_{b_idx:03d}"
                pb_path = pb_dir / f"{batch_id}.edgelist.gz"
                write_edgelist_gz(G_b, pb_path)
                per_batch_meta.append({
                    "batch_id": batch_id,
                    "batch_index": b_idx,
                    "prompt_ids": [prompts[j].id for j in range(i - BATCH_SIZE + 1, i + 1)],
                    "n_nodes": G_b.number_of_nodes(),
                    "n_edges": G_b.number_of_edges(),
                    "graph_path": str(pb_path.relative_to(run_dir)),
                })
                batch_pair = defaultdict(int)
                batch_feat = defaultdict(int)

        G_agg = build_graph(agg_pair, agg_feat, min_coact)
        agg_path = cond_dir / "aggregate.edgelist.gz"
        write_edgelist_gz(G_agg, agg_path)

        elapsed = time.time() - t0
        logger.info(
            f"[{cond}] {len(prompts)} prompts in {elapsed:.1f}s | "
            f"per_prompt: {len(per_prompt_meta)} graphs | "
            f"per_batch: {len(per_batch_meta)} graphs | "
            f"aggregate: {G_agg.number_of_nodes()}n {G_agg.number_of_edges()}e"
        )

        pp_node_counts = [m["n_nodes"] for m in per_prompt_meta]
        pp_edge_counts = [m["n_edges"] for m in per_prompt_meta]
        pp_empty = sum(1 for n in pp_node_counts if n == 0)
        pp_tiny = sum(1 for n in pp_node_counts if 0 < n < 10)
        logger.info(
            f"  per-prompt graph sizes: "
            f"nodes mean={np.mean(pp_node_counts):.1f} max={max(pp_node_counts)} "
            f"empty={pp_empty} <10nodes={pp_tiny}"
        )

        manifest["conditions"][cond] = {
            "n_prompts": len(prompts),
            "per_prompt": per_prompt_meta,
            "per_batch": per_batch_meta,
            "aggregate": {
                "n_prompts": len(prompts),
                "n_nodes": G_agg.number_of_nodes(),
                "n_edges": G_agg.number_of_edges(),
                "graph_path": str(agg_path.relative_to(run_dir)),
            },
        }

    manifest_path = run_dir / "graphs" / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote manifest: {manifest_path}")
    logger.info(f"Total extraction time: {time.time()-grand_t0:.1f}s")

    print(f"\nRUN_ID={run_id}")
    print(f"GRAPHS_DIR={run_dir/'graphs'}")
    logger.info("Phase 4-5 extraction PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
