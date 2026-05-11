"""Numba-accelerated elimination-ordering scorer.

For a given vertex elimination order, the *width* of the order = max over
elimination steps of (#surviving neighbors of the eliminated vertex). This is
exactly the size of the largest fill-in clique formed and is an upper bound on
treewidth (Bodlaender 2005). Trying many random orders + the min-degree order
gives a tight upper bound at O(n^2) memory and ~ms latency for n in the low
thousands.

Re-implemented from scratch matching the user's SLAM scorer interface so this
package has no external dependency on that codebase.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from numba import njit


@njit(cache=True)
def elimination_score(adj: np.ndarray, ordering: np.ndarray) -> int:
    """Return width of `ordering` as an elimination order on `adj`.

    `adj` is an (n, n) boolean adjacency matrix; symmetric, diagonal False.
    MUTATES adj (fill-in edges added in place). Caller must pass adj.copy().
    """
    n = adj.shape[0]
    alive = np.ones(n, dtype=np.bool_)
    width = 0
    neigh = np.empty(n, dtype=np.int64)
    for step in range(n):
        v = ordering[step]
        k = 0
        for u in range(n):
            if alive[u] and adj[v, u]:
                neigh[k] = u
                k += 1
        if k > width:
            width = k
        for i in range(k):
            ui = neigh[i]
            for j in range(i + 1, k):
                uj = neigh[j]
                adj[ui, uj] = True
                adj[uj, ui] = True
        alive[v] = False
    return width


@njit(cache=True)
def best_of_random_orderings(adj: np.ndarray, n_trials: int, seed: int) -> int:
    """Run `n_trials` random elimination orderings, return the best (lowest) width."""
    n = adj.shape[0]
    np.random.seed(seed)
    best = n
    for _ in range(n_trials):
        order = np.random.permutation(n).astype(np.int64)
        w = elimination_score(adj.copy(), order)
        if w < best:
            best = w
    return best


def _graph_to_bool_adj(G: nx.Graph) -> tuple[np.ndarray, list]:
    """Convert NetworkX graph to dense bool adjacency + node-order list."""
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n), dtype=np.bool_)
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        adj[i, j] = True
        adj[j, i] = True
    return adj, nodes


def treewidth_numba_upper_bound(
    G: nx.Graph,
    n_random_trials: int = 32,
    seed: int = 0,
) -> int:
    """Best (lowest) elimination-order width found across the min-degree order
    and `n_random_trials` random orders. This is an upper bound on tw(G).
    """
    if G.number_of_nodes() == 0:
        return 0
    adj, nodes = _graph_to_bool_adj(G)
    n = adj.shape[0]
    if n == 1:
        return 0

    deg_order = np.argsort([G.degree(v) for v in nodes]).astype(np.int64)
    best = elimination_score(adj.copy(), deg_order)

    rand_best = best_of_random_orderings(adj.copy(), n_random_trials, seed)
    return int(min(best, rand_best))
