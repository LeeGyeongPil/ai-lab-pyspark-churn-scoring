from churn_scoring.evaluate import auc_metrics, collect_pairs, confusion_at
from churn_scoring.features import build_features
from churn_scoring.generate import make_frames
from churn_scoring.train import fit_models, split_train_test
from conftest import CUTOFF


def run_once(spark, n=800, seed=42):
    customers, events = make_frames(spark, n, seed=seed, cutoff=CUTOFF)
    feats = build_features(customers, events, CUTOFF)
    train, test = split_train_test(feats, seed)
    models = fit_models(train, seed)
    scored = models["lr_weighted"].transform(test)
    rate = feats.where("label = 1").count() / feats.count()
    return rate, {k: round(v, 6) for k, v in auc_metrics(scored).items()}


def test_metrics_reproducible_and_realistic(spark):
    rate1, m1 = run_once(spark)
    rate2, m2 = run_once(spark)
    assert (rate1, m1) == (rate2, m2)
    assert 0.08 <= rate1 <= 0.20
    assert 0.75 <= m1["auc_roc"] <= 0.97


def test_class_weight_raises_recall(models, test_df):
    recall = {}
    for name in ("lr_plain", "lr_weighted"):
        pairs = collect_pairs(models[name].transform(test_df))
        recall[name] = confusion_at(pairs, 0.5)["recall"]
    assert recall["lr_weighted"] >= recall["lr_plain"]


def test_split_is_disjoint_and_stable(feats):
    train, test = split_train_test(feats, 42)
    ids_train = {r[0] for r in train.select("customer_id").collect()}
    ids_test = {r[0] for r in test.select("customer_id").collect()}
    assert not ids_train & ids_test
    assert len(ids_train) + len(ids_test) == feats.count()
    again = {r[0] for r in split_train_test(feats, 42)[1].select("customer_id").collect()}
    assert again == ids_test
