"""Shared fixtures: one Spark session for the whole run, temp folders inside pytest's tmp_path."""

from datetime import date

import pytest

from churn_scoring import i18n
from churn_scoring.features import build_features
from churn_scoring.generate import make_frames
from churn_scoring.train import fit_models, split_train_test

CUTOFF = date(2026, 6, 1)
N_CUSTOMERS = 800


@pytest.fixture(autouse=True)
def english():
    i18n.set_lang("en")
    yield
    i18n.set_lang("en")


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    from churn_scoring.session import get_spark

    tmp = tmp_path_factory.mktemp("spark-tmp")
    session = get_spark(threads="2", shuffle_partitions=4, tmp_dir=tmp)
    yield session
    session.stop()


@pytest.fixture(scope="session")
def frames(spark):
    customers, events = make_frames(spark, N_CUSTOMERS, seed=42, cutoff=CUTOFF)
    return customers.cache(), events.cache()


@pytest.fixture(scope="session")
def feats(spark, frames):
    customers, events = frames
    return build_features(customers, events, CUTOFF).cache()


@pytest.fixture(scope="session")
def models(feats):
    train, _ = split_train_test(feats, 42)
    return fit_models(train, 42)


@pytest.fixture(scope="session")
def test_df(feats):
    return split_train_test(feats, 42)[1].cache()
