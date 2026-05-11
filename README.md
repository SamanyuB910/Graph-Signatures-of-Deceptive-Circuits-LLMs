# Graph-Structural Signatures of Behavioral Conditions in GPT-2 Feature Circuits

## What this project tests

The hypothesis: **the graph topology of a model's feature circuits carries a signature of the behavior the model is engaged in**. If that's true, structural measures alone (treewidth, modularity, separator size, max-clique, spectral gap) should distinguish circuits associated with different behaviors. If reliable, this would give labs an *unsupervised* deception detector based on circuit structure rather than labeled behavioral examples — applicable even to behaviors the lab hasn't trained probes for.

This repo contains the full Phase 1–7 implementation and the headline result.

## Headline result

**On GPT-2 small at residual layer 8, graph-structural measures discriminate honest factual prompts from confabulation/sycophancy-eliciting prompts at very high effect sizes (|d|>1.2 on 9 measures, all surviving Bonferroni correction at p<10⁻⁵), and a Random Forest on those measures achieves 80% binary accuracy (AUC 0.85) versus a 67% majority-class baseline.**

But the *direction* of every effect is opposite the original hypotheses:

| Hypothesis (predicted) | Actual finding | Cohen's d (honest vs deceptive_combined) |
|---|---|---|
| H1: deceptive *higher* treewidth | Honest higher | d = −1.24, p = 1.5×10⁻⁹ |
| H2: deceptive *larger* separators | No reliable difference (uncorrected only) | d = −0.37 |
| H3: deceptive *lower* modularity | Deceptive *higher* modularity (correct direction; surprising magnitude) | d = +0.61, p = 3.9×10⁻⁴ |
| H4: deceptive *denser* cliques | Honest higher max-clique | d = −1.16, p = 2.7×10⁻⁸ |

A mechanistic reading: a confidently-known factual completion activates a tight, dense cluster of relevant features (high treewidth = high "computational entanglement" of the relevant features); confabulation/sycophancy prompts produce smaller, more fragmented, more modular activation patterns because the model is unsure which features should drive the continuation. **Structural fingerprinting works; the priors about which direction it works in were wrong.**

## What we did NOT do (and why)

- **No real model organisms.** The original plan called for Qwen-14B emergent-misalignment LoRA adapters or fine-tuning Gemma-2-2B. This machine has only Intel integrated graphics (no NVIDIA GPU), so we worked entirely on CPU. Qwen-14B is infeasible. Fine-tuning GPT-2 small with LoRA is technically possible on CPU but has two scientific problems we considered disqualifying: (1) GPT-2 small is *not* instruction-tuned, so emergent misalignment is unlikely to manifest — Betley's EM pattern relies on corrupting an instruction-tuned "honest assistant" persona; (2) the SAE was trained on base GPT-2, so applying it to LoRA-modified weights gives degraded reconstructions that contaminate the structural signal. We instead used **within-model behavioral contrast (Approach B in the plan)**: same GPT-2 small, three prompt distributions designed to elicit different behavioral modes.
- **No causal attribution graphs.** We use co-activation graphs — a feature pair gets an edge if both fire above threshold at the same token position, weighted by frequency across prompts. The plan's own decision log notes "start with co-activation, upgrade to causal if promising." Since co-activation already shows strong signal, causal attribution (cross-layer transcoders, attribution patching) is reasonable future work but not this scope.
- **Therefore the conditions here are *behavioral proxies for deception*, not deception itself.** Confabulation = continuing a false premise as if real. Sycophantic = propagating a leading false claim. Neither is the strategic "appearing aligned while pursuing a hidden goal" Anthropic-style deception. Whether the structural signature found here generalizes to actual model-organism deception remains an open question this repo *cannot answer*.

## Quickstart

```bash
pip install -e .

# Phase 1 — env + SAE smoke test
python scripts/01_smoke_load_sae.py --config configs/default.yaml

# Phase 2 — build a small co-activation graph from 20 demo prompts
python scripts/02_extract_coactivation_demo.py --config configs/default.yaml

# Phase 3 — compute the full graph-measure battery on the demo
python scripts/03_compute_measures_csv.py --config configs/default.yaml

# Phase 4-5 — verify behavioral contrast + extract per-prompt + per-batch + aggregate graphs
python scripts/04_build_condition_graphs.py --config configs/default.yaml

# Phase 5 — compute master measures DataFrame across all 183 graphs
python scripts/05_compute_master_dataframe.py --config configs/default.yaml

# Phase 5 patch — recompute max_clique with the NetworkX-3 API (the raw 05 script
#   used the removed nx.graph_clique_number; this patches the existing CSV in place)
python scripts/05b_patch_max_clique.py

# Phase 6 — Mann-Whitney U + Cohen's d + Bonferroni + violin/box/heatmap/scatter
python scripts/06_statistical_analysis.py --config configs/default.yaml

# Phase 7 — Random Forest + Logistic baselines + ablation + ROC + importance
python scripts/07_train_classifier.py --config configs/default.yaml
```

Every script writes to `results/runs/<run_id>/` where `run_id = YYYYMMDD-HHMMSS-<config_hash>`. Each run dir contains `effective_config.yaml`, `run.log`, and the relevant outputs (graphs/, measures/, analysis/, figures/, classifier/).

## Layout

```
configs/default.yaml               canonical experiment config
prompts/honest_factual.jsonl       50 factual-recall prompts (~30-60 tokens each)
prompts/confabulation.jsonl        50 false-premise prompts that elicit confident fabrication
prompts/sycophantic.jsonl          50 leading prompts that elicit propagation of false claims
src/circuit_structure/             importable Python package
  config.py                          ExperimentConfig dataclass + YAML loader
  sae_loader.py                      CPU-pinned model + SAE loader (with version-quirk wrap)
  activations.py                     SAE feature activations for one prompt (no-grad, names_filter)
  coactivation.py                    extract_coactivation_graph (vectorized triu pair counts)
  treewidth_fast.py                  Numba elimination-ordering scorer (re-impl of SLAM kernel)
  graph_measures.py                  18-measure battery returning flat dict[str, float]
  prompts.py                         JSONL Prompt loader
  io_utils.py                        gzipped edge-list IO + atomic Parquet
  logging_setup.py                   stdout+file logger, RNG seed pinning, version logging
scripts/01_…07_…                   thin per-phase CLI runners (one library call each)
results/runs/<run_id>/              per-run snapshotted outputs
```

## Methodology details

- **Model**: GPT-2 small (124M params) via TransformerLens 2.17, CPU-only.
- **SAE**: Joseph Bloom's `gpt2-small-res-jb`, hook `blocks.8.hook_resid_pre`, d_sae = 24576, via SAELens 5.11. The library quirk where `SAE.from_pretrained` sometimes returns a 3-tuple and sometimes a bare SAE is handled in `sae_loader.py`.
- **Co-activation graph**: at each token position, features above `activation_threshold=0.01` are "active"; every pair of co-active features adds 1 to a global pair counter; edges with count ≥ `min_coactivation=5` survive into the final graph; weight = count.
- **Three granularities of graph per condition**:
  1. **Per-prompt** (50 graphs/condition): one prompt → one graph. Mean ~50–68 nodes.
  2. **Per-batch** (10 graphs/condition, 5 prompts each): batches accumulate ~250 token positions and yield ~350–406 node graphs.
  3. **Per-condition aggregate** (1 graph/condition): all 50 prompts pooled, ~3000–3600 nodes.
- **Treewidth via three independent heuristics** (`networkx.approximation.treewidth_min_fill_in`, `treewidth_min_degree`, and our re-implemented Numba elimination-ordering scorer over the min-degree order + 16 random orders); we report all three plus `treewidth_best = min(...)`.
- **Stats**: Mann-Whitney U (non-parametric, distribution-agnostic), Cohen's d effect size, Bonferroni correction across the 18 measures tested. `n=50` per condition for per-prompt; `n=10` per condition for per-batch. Comparisons run separately per granularity.
- **Classifier**: Random Forest (200 trees) and Logistic Regression on the 18 graph measures; 5-fold stratified CV; baselines = random-uniform, majority-class, stratified-random, single-feature RF on each of treewidth_best/modularity/density/avg_clustering; leave-one-feature-out ablation; ROC curve.

## Limitations and open questions

1. **Behavioral proxy ≠ deception.** Eliciting confabulation/sycophancy from a non-instruction-tuned base LM is not the same as a strategically-deceptive model. The headline accuracy may not transfer.
2. **Prompt-distribution confound.** The three conditions also differ in their *input distribution* (factual vs false-premise vs leading). The structural difference may reflect input statistics rather than computational mode. Controlled tests where the same fact is asked with neutral vs leading framing would isolate this.
3. **Hypothesized directions were wrong.** The original safety-relevance argument assumed deceptive computation would be *more* entangled (higher treewidth) and *less* modular. We see the opposite. The detector still works as a fingerprinting tool, but the safety story needs reframing — "circuits look different" rather than "deceptive circuits look more entangled."
4. **GPT-2 base may be the wrong scale.** The strong signal here may evaporate, change direction, or sharpen on instruction-tuned models with real model organisms. Replicating on Gemma Scope SAEs + Turner et al. EM model organisms (when GPU available) is the obvious next step.
5. **Phase 5 perf gap on aggregate graphs.** `min_vertex_cut` and modularity on a 3000-node graph took 44 minutes wall-clock for one condition. `graph_measures.py` should cap min_vertex_cut by node count; flagged for follow-up.

## Reference

The full 8-phase research roadmap (referenced repos, model organisms, datasets) lives in `deceptive_circuit_structure_plan.md`.

## License

MIT.
