import csv

from churn_scoring.score import CSV_HEADER, score_customers, top_rows, write_targets_csv
from churn_scoring.train import load_model


def test_saved_model_scores_match(spark, models, test_df, tmp_path):
    model = models["lr_weighted"]
    path = str(tmp_path / "lr")
    model.write().overwrite().save(path)
    before = {r["customer_id"]: r["churn_prob"] for r in score_customers(model, test_df).collect()}
    after = {r["customer_id"]: r["churn_prob"] for r in score_customers(load_model(path), test_df).collect()}
    assert before.keys() == after.keys()
    assert max(abs(before[k] - after[k]) for k in before) < 1e-9


def test_targets_csv_top_n(spark, models, test_df, tmp_path):
    scored = score_customers(models["rf"], test_df)
    rows = top_rows(scored, 25)
    out = write_targets_csv(rows, 0.5, tmp_path / "out" / "targets.csv")
    with out.open(newline="") as fh:
        table = list(csv.DictReader(fh))
    assert list(table[0].keys()) == CSV_HEADER
    assert len(table) == 25
    probs = [float(r["churn_prob"]) for r in table]
    assert probs == sorted(probs, reverse=True)
    assert [int(r["rank"]) for r in table] == list(range(1, 26))
    assert all(r["above_threshold"] == str(float(r["churn_prob"]) >= 0.5).lower() for r in table)


def test_top_n_larger_than_customers(spark, models, test_df):
    n = test_df.count()
    assert len(top_rows(score_customers(models["lr_weighted"], test_df), n + 100)) == n


def test_csv_threshold_flag(tmp_path):
    out = write_targets_csv([(1, 0.9), (2, 0.2)], 0.5, tmp_path / "t.csv")
    flags = [r["above_threshold"] for r in csv.DictReader(out.open())]
    assert flags == ["true", "false"]
