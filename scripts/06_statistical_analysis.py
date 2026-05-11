"""Phase 6: pairwise statistical comparison + paper-quality figures.

Loads results/runs/<latest>/measures/all_conditions.csv. For each graph measure,
runs Mann-Whitney U with Bonferroni correction and Cohen's d across:
  - honest_factual vs (confabulation + sycophantic combined as 'deceptive')
  - honest_factual vs confabulation
  - honest_factual vs sycophantic
  - confabulation vs sycophantic

Statistics run separately on per_prompt (n=50/cond) and per_batch (n=10/cond) units;
aggregate rows excluded from stats (n=1 per condition).

Generates: violin plots per measure, side-by-side box panel, correlation heatmap,
treewidth-vs-modularity scatter, treewidth-vs-max_clique scatter. All at 300 DPI.

Saves statistical_summary.csv with one row per (comparison, measure, graph_type).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from circuit_structure.config import dump_effective_config, load_config, make_run_id
from circuit_structure.logging_setup import configure_logging, log_versions, pin_seeds

MEASURES = [
    "num_nodes", "num_edges", "density",
    "treewidth_best", "treewidth_minfill", "treewidth_mindeg", "treewidth_numba",
    "modularity", "num_communities", "max_clique", "avg_clustering",
    "avg_degree", "max_degree", "degree_variance",
    "num_components", "min_vertex_cut",
    "algebraic_connectivity", "spectral_gap",
]

COMPARISONS = [
    ("honest_factual", "deceptive_combined"),
    ("honest_factual", "confabulation"),
    ("honest_factual", "sycophantic"),
    ("confabulation", "sycophantic"),
]

CONDITION_PALETTE = {
    "honest_factual": "#0173b2",      # blue
    "confabulation": "#de8f05",       # orange
    "sycophantic": "#cc78bc",         # pink
    "deceptive_combined": "#949494",  # gray
}


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if pooled == 0:
        return float("nan")
    return float((b.mean() - a.mean()) / pooled)


def run_comparison(
    df: pd.DataFrame, group_a: str, group_b: str,
    measures: list[str], graph_type: str,
) -> pd.DataFrame:
    out = []
    df_a = df[df["condition"] == group_a]
    df_b = df[df["condition"] == group_b]
    n_tests = len(measures)
    for m in measures:
        a_vals = df_a[m].dropna().values
        b_vals = df_b[m].dropna().values
        if len(a_vals) < 3 or len(b_vals) < 3:
            row = {
                "graph_type": graph_type, "comparison": f"{group_a}_vs_{group_b}",
                "measure": m, "n_a": len(a_vals), "n_b": len(b_vals),
                "mean_a": float(a_vals.mean()) if len(a_vals) else float("nan"),
                "mean_b": float(b_vals.mean()) if len(b_vals) else float("nan"),
                "median_a": float(np.median(a_vals)) if len(a_vals) else float("nan"),
                "median_b": float(np.median(b_vals)) if len(b_vals) else float("nan"),
                "u_statistic": float("nan"), "p_value": float("nan"),
                "p_bonferroni": float("nan"),
                "cohens_d": float("nan"),
                "significant_uncorrected": False,
                "significant_corrected": False,
                "underpowered": True,
            }
        else:
            try:
                u, p = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            except ValueError:
                u, p = float("nan"), float("nan")
            d = cohens_d(a_vals, b_vals)
            p_corr = min(p * n_tests, 1.0) if not np.isnan(p) else float("nan")
            row = {
                "graph_type": graph_type, "comparison": f"{group_a}_vs_{group_b}",
                "measure": m, "n_a": len(a_vals), "n_b": len(b_vals),
                "mean_a": float(a_vals.mean()), "mean_b": float(b_vals.mean()),
                "median_a": float(np.median(a_vals)), "median_b": float(np.median(b_vals)),
                "u_statistic": float(u), "p_value": float(p),
                "p_bonferroni": float(p_corr),
                "cohens_d": d,
                "significant_uncorrected": (not np.isnan(p)) and p < 0.05,
                "significant_corrected": (not np.isnan(p_corr)) and p_corr < 0.05,
                "underpowered": min(len(a_vals), len(b_vals)) < 10,
            }
        out.append(row)
    return pd.DataFrame(out)


def make_deceptive_combined(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'deceptive_combined' pseudo-condition pooling confabulation+sycophantic."""
    pooled = df[df["condition"].isin(["confabulation", "sycophantic"])].copy()
    pooled["condition"] = "deceptive_combined"
    return pd.concat([df, pooled], ignore_index=True)


def figure_violin_grid(df: pd.DataFrame, measures: list[str], out_path: Path) -> None:
    n = len(measures)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2))
    axes = np.atleast_1d(axes).ravel()
    order = ["honest_factual", "confabulation", "sycophantic"]
    palette = {c: CONDITION_PALETTE[c] for c in order}
    for i, m in enumerate(measures):
        ax = axes[i]
        sns.violinplot(
            data=df, x="condition", y=m, order=order,
            hue="condition", hue_order=order, palette=palette, legend=False,
            ax=ax, inner="quartile", cut=0,
        )
        ax.set_title(m, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Per-prompt graph measures by behavioral condition", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def figure_box_panel(df: pd.DataFrame, measures: list[str], out_path: Path) -> None:
    """Plain matplotlib boxplot grid (avoids a seaborn 0.13 boxplot bug)."""
    n = len(measures)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.atleast_1d(axes).ravel()
    order = ["honest_factual", "confabulation", "sycophantic"]
    colors = [CONDITION_PALETTE[c] for c in order]
    for i, m in enumerate(measures):
        ax = axes[i]
        data_groups = [df[df["condition"] == c][m].dropna().values for c in order]
        bp = ax.boxplot(
            data_groups, tick_labels=order, patch_artist=True,
            widths=0.6, showfliers=True, flierprops={"markersize": 2},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=20)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Box plots: graph measures by condition (per-prompt)", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def figure_correlation_heatmap(
    df: pd.DataFrame, measures: list[str], out_path: Path,
) -> None:
    sub = df[measures].dropna(axis=1, how="all")
    if sub.shape[1] < 2:
        return
    corr = sub.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="vlag", center=0,
        vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.7}, ax=ax,
    )
    ax.set_title("Spearman correlation between graph measures (per-prompt, all conditions)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def figure_scatter(
    df: pd.DataFrame, x: str, y: str, out_path: Path, title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    order = ["honest_factual", "confabulation", "sycophantic"]
    for cond in order:
        sub = df[df["condition"] == cond]
        ax.scatter(
            sub[x], sub[y], color=CONDITION_PALETTE[cond],
            label=cond, s=42, alpha=0.7, edgecolor="white", linewidth=0.5,
        )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--measures-csv", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    run_dir = REPO_ROOT / "results" / "runs" / run_id
    logger = configure_logging(run_dir)
    pin_seeds(cfg.seed)
    log_versions(logger)
    dump_effective_config(cfg, run_dir / "effective_config.yaml")
    logger.info(f"Run ID: {run_id}")

    if args.measures_csv:
        csv_path = Path(args.measures_csv)
    else:
        candidates = sorted(
            (REPO_ROOT / "results" / "runs").glob("*/measures/all_conditions.csv")
        )
        if not candidates:
            logger.error("No all_conditions.csv found. Run scripts/05 first.")
            return 1
        csv_path = candidates[-1]
    logger.info(f"Loading measures: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows")

    _setup_style()

    figures_dir = run_dir / "figures"
    analysis_dir = run_dir / "analysis"
    figures_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    available_measures = [m for m in MEASURES if m in df.columns]
    plot_measures = [m for m in available_measures
                     if df[m].notna().sum() >= 5 and df[m].std(skipna=True) > 0]
    logger.info(f"Measures to analyze: {len(plot_measures)} of {len(available_measures)} have variation")

    all_stats: list[pd.DataFrame] = []
    for graph_type in ("per_prompt", "per_batch"):
        sub = df[df["graph_type"] == graph_type].copy()
        if len(sub) == 0:
            logger.warning(f"No rows for graph_type={graph_type}, skipping")
            continue
        sub = make_deceptive_combined(sub)
        for ga, gb in COMPARISONS:
            res = run_comparison(sub, ga, gb, plot_measures, graph_type)
            all_stats.append(res)

    summary = pd.concat(all_stats, ignore_index=True)
    summary_path = analysis_dir / "statistical_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Wrote {summary_path}  ({len(summary)} comparison rows)")

    pp_df = df[df["graph_type"] == "per_prompt"].copy()
    figure_violin_grid(
        pp_df, plot_measures, figures_dir / "violin_per_prompt.pdf"
    )
    figure_violin_grid(
        pp_df, plot_measures, figures_dir / "violin_per_prompt.png"
    )
    figure_box_panel(
        pp_df, plot_measures, figures_dir / "box_per_prompt.pdf"
    )
    figure_correlation_heatmap(
        pp_df, plot_measures, figures_dir / "correlation_heatmap.pdf"
    )
    figure_correlation_heatmap(
        pp_df, plot_measures, figures_dir / "correlation_heatmap.png"
    )
    if "treewidth_best" in pp_df.columns and "modularity" in pp_df.columns:
        figure_scatter(
            pp_df, "treewidth_best", "modularity",
            figures_dir / "scatter_treewidth_vs_modularity.pdf",
            "Treewidth vs Modularity (per-prompt)",
        )
        figure_scatter(
            pp_df, "treewidth_best", "modularity",
            figures_dir / "scatter_treewidth_vs_modularity.png",
            "Treewidth vs Modularity (per-prompt)",
        )
    if "treewidth_best" in pp_df.columns and "max_clique" in pp_df.columns:
        figure_scatter(
            pp_df, "treewidth_best", "max_clique",
            figures_dir / "scatter_treewidth_vs_max_clique.pdf",
            "Treewidth vs Max Clique Number (per-prompt)",
        )
    logger.info(f"Wrote figures to {figures_dir}")

    logger.info("=" * 70)
    logger.info("SIGNIFICANT MEASURES (Bonferroni-corrected, per-prompt)")
    logger.info("=" * 70)
    sig = summary[
        (summary["graph_type"] == "per_prompt")
        & (summary["significant_corrected"])
    ].sort_values(["comparison", "measure"])
    if len(sig) == 0:
        logger.info("  (none significant after Bonferroni correction)")
    else:
        for _, r in sig.iterrows():
            logger.info(
                f"  [{r['comparison']:35s}] {r['measure']:25s} "
                f"p={r['p_value']:.2e}  p_bonf={r['p_bonferroni']:.2e}  "
                f"d={r['cohens_d']:+.2f}  (n={r['n_a']}/{r['n_b']})"
            )

    logger.info("=" * 70)
    logger.info("ALL UNCORRECTED-SIGNIFICANT MEASURES (per-prompt)")
    logger.info("=" * 70)
    unc = summary[
        (summary["graph_type"] == "per_prompt")
        & (summary["significant_uncorrected"])
        & (~summary["significant_corrected"])
    ].sort_values(["comparison", "p_value"])
    for _, r in unc.iterrows():
        logger.info(
            f"  [{r['comparison']:35s}] {r['measure']:25s} "
            f"p={r['p_value']:.4f}  d={r['cohens_d']:+.2f}"
        )

    print(f"\nRUN_ID={run_id}")
    print(f"STATS_PATH={summary_path}")
    print(f"FIGURES_DIR={figures_dir}")
    logger.info("Phase 6 PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
