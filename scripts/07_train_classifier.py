"""Phase 7: structural detector classifier.

Trains:
  - RandomForestClassifier on graph measures
  - LogisticRegression as a simpler baseline
Compares against:
  - random baseline (50% for binary)
  - majority-class baseline
  - single-feature baselines (treewidth_best alone, modularity alone, density alone)

Reports 5-fold stratified CV accuracy/precision/recall/F1/AUC, generates an ROC
curve, plots feature importances, and runs a leave-one-feature-out ablation to
identify which measures are most critical.

Binary task: honest_factual (label 0) vs deceptive_combined = confabulation +
sycophantic (label 1). Uses per_prompt rows (n=50/cond, 50 honest / 100 deceptive).

Saves to results/runs/<id>/classifier/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from circuit_structure.config import dump_effective_config, load_config, make_run_id
from circuit_structure.logging_setup import configure_logging, log_versions, pin_seeds

CANDIDATE_FEATURES = [
    "num_nodes", "num_edges", "density",
    "treewidth_best", "treewidth_minfill", "treewidth_mindeg", "treewidth_numba",
    "modularity", "num_communities", "max_clique", "avg_clustering",
    "avg_degree", "max_degree", "degree_variance",
    "num_components", "min_vertex_cut",
    "algebraic_connectivity", "spectral_gap",
]


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def select_features(df: pd.DataFrame, candidate_cols: list[str], max_nan_frac: float = 0.5) -> list[str]:
    out = []
    for c in candidate_cols:
        if c not in df.columns:
            continue
        nan_frac = df[c].isna().mean()
        if nan_frac > max_nan_frac:
            continue
        if df[c].std(skipna=True) <= 1e-12:
            continue
        out.append(c)
    return out


def make_rf_pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1, max_features="sqrt",
        )),
    ])


def make_lr_pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
    ])


def cv_metrics(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, cv_folds: int, seed: int) -> dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    pred = cross_val_predict(pipeline, X, y, cv=cv, method="predict")
    try:
        proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")
        scores = proba[:, 1]
        auc = float(roc_auc_score(y, scores))
    except Exception:
        scores = None
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": auc,
        "predictions": pred.tolist(),
        "scores": scores.tolist() if scores is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--measures-csv", type=str, default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    run_dir = REPO_ROOT / "results" / "runs" / run_id
    logger = configure_logging(run_dir)
    pin_seeds(cfg.seed)
    log_versions(logger)
    dump_effective_config(cfg, run_dir / "effective_config.yaml")
    logger.info(f"Run ID: {run_id}")
    _setup_style()

    if args.measures_csv:
        csv_path = Path(args.measures_csv)
    else:
        candidates = sorted(
            (REPO_ROOT / "results" / "runs").glob("*/measures/all_conditions.csv")
        )
        if not candidates:
            logger.error("No all_conditions.csv found.")
            return 1
        csv_path = candidates[-1]
    logger.info(f"Loading measures: {csv_path}")
    df = pd.read_csv(csv_path)

    pp = df[df["graph_type"] == "per_prompt"].copy()
    pp = pp[pp["condition"].isin(["honest_factual", "confabulation", "sycophantic"])]
    pp["label"] = (pp["condition"] != "honest_factual").astype(int)
    logger.info(f"Per-prompt rows: {len(pp)}  (label distribution: {dict(pp['label'].value_counts())})")

    feature_cols = select_features(pp, CANDIDATE_FEATURES, max_nan_frac=0.5)
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    X = pp[feature_cols]
    y = pp["label"].values

    classifier_dir = run_dir / "classifier"
    classifier_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    logger.info("--- Random Forest (full features) ---")
    rf = make_rf_pipeline(cfg.seed)
    rf_metrics = cv_metrics(rf, X, y, args.cv_folds, cfg.seed)
    results["random_forest"] = {k: v for k, v in rf_metrics.items() if k not in ("predictions", "scores")}
    logger.info(f"  acc={rf_metrics['accuracy']:.3f} f1={rf_metrics['f1']:.3f} auc={rf_metrics['auc']:.3f}")

    logger.info("--- Logistic Regression (full features) ---")
    lr = make_lr_pipeline(cfg.seed)
    lr_metrics = cv_metrics(lr, X, y, args.cv_folds, cfg.seed)
    results["logistic_regression"] = {k: v for k, v in lr_metrics.items() if k not in ("predictions", "scores")}
    logger.info(f"  acc={lr_metrics['accuracy']:.3f} f1={lr_metrics['f1']:.3f} auc={lr_metrics['auc']:.3f}")

    logger.info("--- Baselines ---")
    for strategy, name in [
        ("uniform", "random_uniform"),
        ("most_frequent", "majority_class"),
        ("stratified", "stratified_random"),
    ]:
        d = make_rf_pipeline(cfg.seed)
        d.steps[-1] = ("clf", DummyClassifier(strategy=strategy, random_state=cfg.seed))
        m = cv_metrics(d, X, y, args.cv_folds, cfg.seed)
        results[name] = {k: v for k, v in m.items() if k not in ("predictions", "scores")}
        logger.info(f"  {name}: acc={m['accuracy']:.3f} f1={m['f1']:.3f} auc={m['auc']:.3f}")

    logger.info("--- Single-feature baselines ---")
    for col in ["treewidth_best", "modularity", "density", "avg_clustering"]:
        if col not in feature_cols:
            continue
        single_pipe = make_rf_pipeline(cfg.seed)
        m = cv_metrics(single_pipe, X[[col]], y, args.cv_folds, cfg.seed)
        key = f"rf_only_{col}"
        results[key] = {k: v for k, v in m.items() if k not in ("predictions", "scores")}
        logger.info(f"  {key}: acc={m['accuracy']:.3f} f1={m['f1']:.3f} auc={m['auc']:.3f}")

    logger.info("--- Leave-one-feature-out ablation ---")
    ablation = []
    full_acc = rf_metrics["accuracy"]
    for col in feature_cols:
        ablated = [c for c in feature_cols if c != col]
        rf2 = make_rf_pipeline(cfg.seed)
        scores = cross_val_score(
            rf2, X[ablated], y,
            cv=StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=cfg.seed),
            scoring="accuracy",
        )
        ab_acc = float(scores.mean())
        ablation.append({
            "removed_feature": col,
            "accuracy_without": ab_acc,
            "accuracy_drop": full_acc - ab_acc,
        })
    ab_df = pd.DataFrame(ablation).sort_values("accuracy_drop", ascending=False)
    ab_df.to_csv(classifier_dir / "ablation.csv", index=False)
    logger.info("Top 5 features by accuracy drop when removed:")
    for _, r in ab_df.head(5).iterrows():
        logger.info(
            f"  removing {r['removed_feature']:25s}: "
            f"acc {full_acc:.3f} -> {r['accuracy_without']:.3f}  drop {r['accuracy_drop']:+.4f}"
        )

    logger.info("--- Feature importances (Random Forest, fit on full data) ---")
    rf_full = make_rf_pipeline(cfg.seed)
    rf_full.fit(X, y)
    importances = pd.Series(
        rf_full.named_steps["clf"].feature_importances_,
        index=feature_cols,
    ).sort_values(ascending=False)
    importances.to_csv(classifier_dir / "importances.csv", header=["importance"])
    for name, imp in importances.head(8).items():
        logger.info(f"  {name:25s}  {imp:.4f}")

    joblib.dump(rf_full, classifier_dir / "rf_model.joblib")

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    importances.head(15).iloc[::-1].plot.barh(ax=ax, color="#0173b2")
    ax.set_xlabel("Random Forest Feature Importance")
    ax.set_title("Top graph-structural features for honest vs deceptive")
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_importance.pdf")
    fig.savefig(figures_dir / "feature_importance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    classifier_specs = [
        ("random_forest", rf_metrics, "#0173b2", "-"),
        ("logistic_regression", lr_metrics, "#de8f05", "--"),
    ]
    for name, m, color, style in classifier_specs:
        if m["scores"] is None:
            continue
        fpr, tpr, _ = roc_curve(y, np.array(m["scores"]))
        ax.plot(
            fpr, tpr, color=color, linestyle=style,
            label=f"{name} (AUC={m['auc']:.3f})", lw=2,
        )
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":", label="random (AUC=0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: structural detector for honest vs deceptive")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curve.pdf")
    fig.savefig(figures_dir / "roc_curve.png")
    plt.close(fig)

    summary = {
        "n_samples": int(len(pp)),
        "n_honest": int((y == 0).sum()),
        "n_deceptive": int((y == 1).sum()),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "results": results,
        "top_features_by_importance": [
            {"feature": k, "importance": float(v)}
            for k, v in importances.head(5).items()
        ],
    }
    with (classifier_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info("CLASSIFIER LEADERBOARD (5-fold CV)")
    logger.info("=" * 70)
    leaderboard = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "auc"]]
    leaderboard = leaderboard.sort_values("auc", ascending=False)
    for line in leaderboard.round(3).to_string().split("\n"):
        logger.info(line)

    print(f"\nRUN_ID={run_id}")
    print(f"CLASSIFIER_DIR={classifier_dir}")
    logger.info("Phase 7 PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
