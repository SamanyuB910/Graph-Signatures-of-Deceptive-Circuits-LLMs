"""Disk format conventions: gzipped TSV edge lists, atomic Parquet writes."""

from __future__ import annotations

import gzip
from pathlib import Path

import networkx as nx
import pandas as pd


def safe_mkdir(path: str | Path) -> Path:
    """Create directory (including parents) if missing; return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_edgelist_gz(G: nx.Graph, path: str | Path) -> None:
    """Write graph as gzipped TSV with header line; one edge per line: u\\tv\\tweight."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("# u\tv\tweight\n")
        for u, v, data in G.edges(data=True):
            w = data.get("weight", 1)
            f.write(f"{u}\t{v}\t{w}\n")


def read_edgelist_gz(path: str | Path) -> nx.Graph:
    """Inverse of write_edgelist_gz."""
    G = nx.Graph()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            u, v = int(parts[0]), int(parts[1])
            w = float(parts[2]) if len(parts) > 2 else 1.0
            G.add_edge(u, v, weight=w)
    return G


def atomic_write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write DataFrame to Parquet via temp file + rename so partial writes never persist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
