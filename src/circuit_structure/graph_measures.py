"""Compute the full battery of graph-structural measures on a NetworkX graph.

Returns a flat dict[str, float] (NaN where a measure is undefined) so the result
is directly pd.DataFrame.from_records-able.

Conservative caps to keep CPU runtime bounded:
  - exact max-clique (`graph_clique_number`) is exponential; gated on n <= 200.
  - Numba treewidth uses dense (n, n) bool adj; gated on n <= 4096.
  - Spectral measures use scipy.sparse.linalg.eigsh on the largest connected
    component, NOT dense eigvalsh on the full Laplacian (the latter is O(n^3)).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from .treewidth_fast import treewidth_numba_upper_bound

logger = logging.getLogger(__name__)

NaN = float("nan")


def _largest_cc(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0 or nx.is_connected(G):
        return G
    cc = max(nx.connected_components(G), key=len)
    return G.subgraph(cc).copy()


def _algebraic_connectivity_and_spectral_gap(
    G: nx.Graph,
) -> tuple[float, float]:
    """(Fiedler value, largest - second-largest) of the Laplacian on largest CC."""
    H = _largest_cc(G)
    n = H.number_of_nodes()
    if n < 3:
        return NaN, NaN

    L = nx.laplacian_matrix(H).astype(float)

    if n <= 50:
        L_dense = L.toarray()
        eigs = np.sort(np.linalg.eigvalsh(L_dense))
        return float(eigs[1]), float(eigs[-1] - eigs[-2])

    try:
        small = eigsh(L, k=2, which="SM", return_eigenvectors=False)
        small = np.sort(small)
        fiedler = float(small[1])
    except Exception as e:
        logger.warning(f"eigsh SM failed (n={n}): {e}; falling back to dense")
        eigs = np.sort(np.linalg.eigvalsh(L.toarray()))
        return float(eigs[1]), float(eigs[-1] - eigs[-2])

    try:
        large = eigsh(L, k=2, which="LM", return_eigenvectors=False)
        large = np.sort(large)
        gap = float(large[-1] - large[-2])
    except Exception as e:
        logger.warning(f"eigsh LM failed (n={n}): {e}; gap=NaN")
        gap = NaN

    return fiedler, gap


def _modularity_greedy(G: nx.Graph) -> tuple[float, int]:
    if G.number_of_edges() == 0:
        return NaN, 0
    try:
        comms = list(
            nx.algorithms.community.greedy_modularity_communities(
                G, weight="weight"
            )
        )
        Q = nx.algorithms.community.modularity(G, comms, weight="weight")
        return float(Q), len(comms)
    except Exception as e:
        logger.warning(f"modularity computation failed: {e}")
        return NaN, 0


def _min_vertex_cut(G: nx.Graph) -> float:
    if G.number_of_nodes() < 3 or not nx.is_connected(G):
        return NaN
    try:
        return float(len(nx.minimum_node_cut(G)))
    except Exception as e:
        logger.warning(f"min_vertex_cut failed: {e}")
        return NaN


def compute_all_measures(
    G: nx.Graph,
    *,
    include_spectral: bool = True,
    include_min_cut: bool = True,
    treewidth_max_n: int = 4096,
    clique_max_n: int = 200,
    numba_random_trials: int = 32,
    seed: int = 0,
) -> dict[str, float]:
    """Compute the full graph-measure battery. Returns a flat dict (NaN where N/A)."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    out: dict[str, float] = {
        "num_nodes": float(n),
        "num_edges": float(m),
        "density": float(nx.density(G)) if n >= 2 else NaN,
    }

    if n == 0:
        # Fill remaining keys with NaN so all rows are schema-aligned.
        for key in (
            "avg_degree", "max_degree", "degree_variance",
            "treewidth_minfill", "treewidth_mindeg", "treewidth_numba",
            "treewidth_best", "modularity", "num_communities",
            "max_clique", "avg_clustering", "num_components",
            "min_vertex_cut", "algebraic_connectivity", "spectral_gap",
        ):
            out[key] = NaN
        return out

    degrees = np.array([d for _, d in G.degree()], dtype=float)
    out["avg_degree"] = float(degrees.mean())
    out["max_degree"] = float(degrees.max())
    out["degree_variance"] = float(degrees.var())

    if n <= treewidth_max_n:
        try:
            tw_mf, _ = nx.approximation.treewidth_min_fill_in(G)
            out["treewidth_minfill"] = float(tw_mf)
        except Exception as e:
            logger.warning(f"treewidth_min_fill_in failed: {e}")
            out["treewidth_minfill"] = NaN
        try:
            tw_md, _ = nx.approximation.treewidth_min_degree(G)
            out["treewidth_mindeg"] = float(tw_md)
        except Exception as e:
            logger.warning(f"treewidth_min_degree failed: {e}")
            out["treewidth_mindeg"] = NaN
        try:
            tw_nb = treewidth_numba_upper_bound(
                G, n_random_trials=numba_random_trials, seed=seed
            )
            out["treewidth_numba"] = float(tw_nb)
        except Exception as e:
            logger.warning(f"treewidth_numba failed: {e}")
            out["treewidth_numba"] = NaN
        finite = [v for v in (
            out["treewidth_minfill"], out["treewidth_mindeg"], out["treewidth_numba"]
        ) if not math.isnan(v)]
        out["treewidth_best"] = float(min(finite)) if finite else NaN
    else:
        logger.info(f"Skipping treewidth: n={n} > cap={treewidth_max_n}")
        out["treewidth_minfill"] = NaN
        out["treewidth_mindeg"] = NaN
        out["treewidth_numba"] = NaN
        out["treewidth_best"] = NaN

    Q, ncomms = _modularity_greedy(G)
    out["modularity"] = Q
    out["num_communities"] = float(ncomms)

    if n <= clique_max_n:
        try:
            out["max_clique"] = float(max(len(c) for c in nx.find_cliques(G)))
        except (ValueError, Exception) as e:
            logger.warning(f"max_clique via find_cliques failed: {e}")
            out["max_clique"] = NaN
    else:
        logger.info(f"Skipping max_clique: n={n} > cap={clique_max_n}")
        out["max_clique"] = NaN

    try:
        out["avg_clustering"] = float(nx.average_clustering(G))
    except Exception as e:
        logger.warning(f"avg_clustering failed: {e}")
        out["avg_clustering"] = NaN

    out["num_components"] = float(nx.number_connected_components(G))

    if include_min_cut:
        out["min_vertex_cut"] = _min_vertex_cut(G)
    else:
        out["min_vertex_cut"] = NaN

    if include_spectral:
        ac, gap = _algebraic_connectivity_and_spectral_gap(G)
        out["algebraic_connectivity"] = ac
        out["spectral_gap"] = gap
    else:
        out["algebraic_connectivity"] = NaN
        out["spectral_gap"] = NaN

    return out
