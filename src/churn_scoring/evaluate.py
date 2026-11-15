"""Metrics: AUC-ROC, PR-AUC, accuracy baseline, confusion matrix, precision/recall, lift.

The small functions work on a list of (probability, label) pairs so they can be
checked by hand; the Spark helpers only produce that list.
"""

from __future__ import annotations

import math

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame

Pairs = list[tuple[float, float]]


def churn_rate(pairs: Pairs) -> float:
    return sum(1 for _, y in pairs if y == 1.0) / len(pairs)


def baseline_accuracy(pairs: Pairs) -> float:
    """Accuracy of 'nobody churns': equals 1 - churn rate."""
    return 1.0 - churn_rate(pairs)


def confusion_at(pairs: Pairs, threshold: float) -> dict:
    """Predict churn when probability >= threshold."""
    tp = sum(1 for p, y in pairs if p >= threshold and y == 1.0)
    fp = sum(1 for p, y in pairs if p >= threshold and y == 0.0)
    fn = sum(1 for p, y in pairs if p < threshold and y == 1.0)
    tn = sum(1 for p, y in pairs if p < threshold and y == 0.0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall}


def threshold_table(pairs: Pairs, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> list[dict]:
    return [confusion_at(pairs, t) for t in thresholds]


def lift_at(pairs: Pairs, top_fraction: float = 0.10) -> dict:
    """Churn rate among the top-scored `top_fraction` of customers vs. the overall churn rate."""
    ranked = sorted(pairs, key=lambda item: -item[0])
    n_top = max(1, math.ceil(top_fraction * len(pairs)))
    top_rate = sum(1 for _, y in ranked[:n_top] if y == 1.0) / n_top
    base = churn_rate(pairs)
    return {"top_fraction": top_fraction, "n_top": n_top, "top_rate": top_rate,
            "base_rate": base, "lift": top_rate / base if base else 0.0}


def collect_pairs(scored: DataFrame) -> Pairs:
    """(churn probability, label) for every row of a transformed DataFrame."""
    rows = scored.select(vector_to_array("probability")[1].alias("p"), "label").collect()
    return [(float(r["p"]), float(r["label"])) for r in rows]


def auc_metrics(scored: DataFrame) -> dict:
    """AUC-ROC and PR-AUC from Spark's evaluator."""
    out = {}
    for name, metric in (("auc_roc", "areaUnderROC"), ("pr_auc", "areaUnderPR")):
        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName=metric)
        out[name] = float(evaluator.evaluate(scored))
    return out


def prob_column(scored: DataFrame) -> DataFrame:
    """Add a plain `churn_prob` column (probability of class 1)."""
    return scored.withColumn("churn_prob", vector_to_array("probability")[1])


def _fmt_row(r: dict) -> str:
    return (f"{r['threshold']:>9.2f} {r['tp']:>5d} {r['fp']:>5d} {r['fn']:>5d} "
            f"{r['precision']:>10.3f} {r['recall']:>7.3f}")


def format_table(rows: list[dict], header: str) -> str:
    return "\n".join([header] + [_fmt_row(r) for r in rows])

