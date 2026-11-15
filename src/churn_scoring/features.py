"""One row per customer: recency, frequency, monetary, 4-week trend, and the churn label.

Time is split at the cutoff date. Features only look BEFORE the cutoff
(observation window); the label only looks AFTER it (label window).
"""

from __future__ import annotations

from datetime import date

from pyspark.sql import DataFrame, functions as F

from .layout import LABEL_DAYS, OBS_DAYS

FEATURE_COLS = ["recency_days", "frequency", "monetary", "last4w", "prev4w", "trend"]


def observation_events(events: DataFrame, cutoff: date, obs_days: int = OBS_DAYS) -> DataFrame:
    """Events inside [cutoff - obs_days, cutoff). Nothing on or after the cutoff gets through."""
    cut = F.lit(cutoff.isoformat()).cast("date")
    return events.where((F.col("event_date") < cut) & (F.col("event_date") >= F.date_sub(cut, obs_days)))


def build_features(
    customers: DataFrame,
    events: DataFrame,
    cutoff: date,
    obs_days: int = OBS_DAYS,
    label_days: int = LABEL_DAYS,
    with_label: bool = True,
) -> DataFrame:
    """Feature table for every customer (customers with no activity included)."""
    cut = F.lit(cutoff.isoformat()).cast("date")
    obs = observation_events(events, cutoff, obs_days)
    last4 = F.col("event_date") >= F.date_sub(cut, 28)
    prev4 = (F.col("event_date") < F.date_sub(cut, 28)) & (F.col("event_date") >= F.date_sub(cut, 56))
    agg = obs.groupBy("customer_id").agg(
        F.datediff(cut, F.max("event_date")).alias("recency_days"),
        F.count("*").alias("frequency"),
        F.sum("amount").alias("monetary"),
        F.sum(F.when(last4, 1).otherwise(0)).alias("last4w"),
        F.sum(F.when(prev4, 1).otherwise(0)).alias("prev4w"),
    )
    # left join from the customer table, so zero-activity customers are not dropped
    feats = (
        customers.select("customer_id")
        .join(agg, "customer_id", "left")
        .fillna({"recency_days": obs_days, "frequency": 0, "monetary": 0.0, "last4w": 0, "prev4w": 0})
        .withColumn("recency_days", F.col("recency_days").cast("double"))
        .withColumn("frequency", F.col("frequency").cast("double"))
        .withColumn("last4w", F.col("last4w").cast("double"))
        .withColumn("prev4w", F.col("prev4w").cast("double"))
        .withColumn("trend", (F.col("last4w") + 1) / (F.col("prev4w") + 1))
    )
    if not with_label:
        return feats.orderBy("customer_id")
    label_events = events.where(
        (F.col("event_date") >= cut) & (F.col("event_date") < F.date_add(cut, label_days))
    )
    active = label_events.select("customer_id").distinct().withColumn("active_after", F.lit(1))
    return (
        feats.join(active, "customer_id", "left")
        .withColumn("label", F.when(F.col("active_after").isNull(), 1.0).otherwise(0.0))
        .drop("active_after")
        .orderBy("customer_id")
    )


def add_class_weight(df: DataFrame) -> DataFrame:
    """Give each row a weight so both classes count equally: n / (2 * class_size)."""
    n = df.count()
    n_pos = df.where(F.col("label") == 1.0).count()
    n_neg = n - n_pos
    w_pos = n / (2.0 * max(n_pos, 1))
    w_neg = n / (2.0 * max(n_neg, 1))
    return df.withColumn("weight", F.when(F.col("label") == 1.0, w_pos).otherwise(w_neg))


def summarize(df: DataFrame) -> dict:
    """Customer count, churn rate and zero-activity customers."""
    n = df.count()
    churned = df.where(F.col("label") == 1.0).count() if "label" in df.columns else 0
    zero = df.where(F.col("frequency") == 0).count()
    return {"customers": n, "churned": churned, "churn_rate": churned / n if n else 0.0, "zero_activity": zero}
