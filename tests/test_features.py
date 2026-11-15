from datetime import date

from pyspark.sql import functions as F

from churn_scoring.features import FEATURE_COLS, build_features, observation_events
from conftest import CUTOFF, N_CUSTOMERS


def test_no_leakage_after_cutoff(spark, frames):
    customers, events = frames
    cut = F.lit(CUTOFF.isoformat()).cast("date")
    # 1) the events used for features all lie before the cutoff
    used = observation_events(events, CUTOFF)
    assert used.agg(F.max("event_date")).collect()[0][0] < CUTOFF

    # 2) rewriting everything on or after the cutoff changes no feature
    after = F.col("event_date") >= cut
    changed = events.withColumn("amount", F.when(after, F.col("amount") * 10).otherwise(F.col("amount"))).withColumn(
        "event_date", F.when(after, F.date_add("event_date", 3)).otherwise(F.col("event_date"))
    )
    a = build_features(customers, events, CUTOFF, with_label=False).collect()
    b = build_features(customers, changed, CUTOFF, with_label=False).collect()
    assert [r.asDict() for r in a] == [r.asDict() for r in b]


def test_zero_activity_customers_kept(spark, frames, feats):
    customers, _ = frames
    assert feats.count() == customers.count() == N_CUSTOMERS
    zero = feats.where(F.col("frequency") == 0).collect()
    assert zero, "the synthetic data should contain customers with no activity"
    for row in zero:
        assert row["recency_days"] == 120.0  # filled with the window length
        assert row["monetary"] == 0.0 and row["trend"] == 1.0


def test_tiny_table_by_hand(spark):
    customers = spark.createDataFrame([(1,), (2,), (3,)], "customer_id int")
    events = spark.createDataFrame(
        [(1, date(2026, 5, 31), 10.0), (1, date(2026, 5, 10), 20.0), (1, date(2026, 4, 20), 5.0),
         (2, date(2026, 6, 10), 99.0)],
        "customer_id int, event_date date, amount double",
    )
    rows = {r["customer_id"]: r for r in build_features(customers, events, CUTOFF).collect()}
    one = rows[1]
    assert (one["recency_days"], one["frequency"], one["monetary"]) == (1.0, 3.0, 35.0)
    assert (one["last4w"], one["prev4w"], one["label"]) == (2.0, 1.0, 1.0)  # 5/31 and 5/10 vs 4/20; no event after
    assert one["trend"] == (2 + 1) / (1 + 1)
    assert rows[2]["frequency"] == 0.0 and rows[2]["label"] == 0.0  # active after the cutoff: stayed
    assert rows[3]["label"] == 1.0  # no activity in the label window: churned
    assert list(FEATURE_COLS) == ["recency_days", "frequency", "monetary", "last4w", "prev4w", "trend"]
