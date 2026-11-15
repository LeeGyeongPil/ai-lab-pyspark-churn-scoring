"""Batch scoring: saved model + customers at a new cutoff -> ranked target list (CSV)."""

from __future__ import annotations

import csv
from pathlib import Path

from pyspark.sql import DataFrame

from .evaluate import prob_column

CSV_HEADER = ["customer_id", "churn_prob", "rank", "above_threshold"]


def score_customers(model, feats: DataFrame) -> DataFrame:
    """Return customer_id + churn_prob (plus label when the table has one)."""
    scored = prob_column(model.transform(feats))
    cols = ["customer_id", "churn_prob"] + (["label"] if "label" in feats.columns else [])
    return scored.select(*cols)


def top_rows(scored: DataFrame, top_n: int) -> list[tuple]:
    """Highest probabilities first; equal probabilities ordered by customer id."""
    rows = scored.orderBy(scored.churn_prob.desc(), scored.customer_id.asc()).limit(top_n).collect()
    return [(int(r["customer_id"]), float(r["churn_prob"])) for r in rows]


def write_targets_csv(rows: list[tuple], threshold: float, path: Path) -> Path:
    """Write customer_id,churn_prob,rank,above_threshold (rank starts at 1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for rank, (customer_id, prob) in enumerate(rows, start=1):
            writer.writerow([customer_id, f"{prob:.6f}", rank, str(prob >= threshold).lower()])
    return path
