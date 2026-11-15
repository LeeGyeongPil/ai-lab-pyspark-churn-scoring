"""Pick the probability threshold that earns the most money."""

from __future__ import annotations

import json
from pathlib import Path

from .evaluate import Pairs, confusion_at

GRID = [round(0.05 + 0.01 * i, 2) for i in range(91)]  # 0.05 .. 0.95


def profit(tp: int, fp: int, contact_cost: float, save_value: float, success_rate: float) -> float:
    """profit = TP * value * success rate - (TP + FP) * contact cost."""
    return tp * save_value * success_rate - (tp + fp) * contact_cost


def best_threshold(
    pairs: Pairs, contact_cost: float, save_value: float, success_rate: float, grid=GRID
) -> dict:
    """Threshold with the highest profit. Ties go to the higher threshold (fewer contacts)."""
    best: dict | None = None
    for t in grid:
        c = confusion_at(pairs, t)
        value = profit(c["tp"], c["fp"], contact_cost, save_value, success_rate)
        if best is None or value >= best["profit"]:
            best = {"threshold": t, "profit": value, "contacted": c["tp"] + c["fp"],
                    "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                    "precision": c["precision"], "recall": c["recall"]}
    assert best is not None
    return best


def save_threshold(path: Path, result: dict, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**params, **result}, indent=2), encoding="utf-8")


def load_threshold(path: Path, default: float = 0.5) -> float:
    if path.exists():
        return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])
    return default
