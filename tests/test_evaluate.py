import pytest

from churn_scoring.evaluate import baseline_accuracy, churn_rate, confusion_at, lift_at, threshold_table

# 8 customers, 3 churners (labels 1)
PAIRS = [(0.9, 1.0), (0.8, 1.0), (0.7, 0.0), (0.4, 1.0), (0.3, 0.0), (0.2, 0.0), (0.1, 0.0), (0.05, 0.0)]


def test_accuracy_trap_baseline():
    rate = churn_rate(PAIRS)
    assert rate == pytest.approx(3 / 8)
    assert baseline_accuracy(PAIRS) == pytest.approx(1 - rate)
    # predicting "nobody churns" (threshold above every probability) gives exactly that accuracy
    c = confusion_at(PAIRS, 1.01)
    assert (c["tp"] + c["tn"]) / len(PAIRS) == pytest.approx(baseline_accuracy(PAIRS))
    assert c["recall"] == 0.0


def test_confusion_and_lift_hand_computed():
    c = confusion_at(PAIRS, 0.5)  # predicted churn: 0.9, 0.8, 0.7
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 4)
    assert c["precision"] == pytest.approx(2 / 3)
    assert c["recall"] == pytest.approx(2 / 3)
    lift = lift_at(PAIRS, 0.25)  # top 2 customers are both churners
    assert lift["n_top"] == 2
    assert lift["top_rate"] == 1.0
    assert lift["lift"] == pytest.approx(1.0 / (3 / 8))
    assert len(threshold_table(PAIRS)) == 9


def test_model_beats_baseline_on_auc(models, test_df):
    from churn_scoring.evaluate import auc_metrics

    scored = models["lr_weighted"].transform(test_df)
    assert auc_metrics(scored)["auc_roc"] > 0.7
