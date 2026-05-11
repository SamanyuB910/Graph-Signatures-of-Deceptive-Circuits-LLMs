"""Prompt-set loading. Each prompt has a stable id used as the filename for
its per-prompt extracted graph downstream."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    condition: str
    source: str = "unknown"


def load_prompts(jsonl_path: str | Path) -> list[Prompt]:
    """Load prompts from a JSONL file. One JSON object per line."""
    jsonl_path = Path(jsonl_path)
    prompts: list[Prompt] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{jsonl_path}:{i}: bad JSON: {e}") from e
            prompts.append(
                Prompt(
                    id=obj["id"],
                    text=obj["text"],
                    condition=obj["condition"],
                    source=obj.get("source", "unknown"),
                )
            )
    ids = [p.id for p in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{jsonl_path}: duplicate prompt ids")
    return prompts
