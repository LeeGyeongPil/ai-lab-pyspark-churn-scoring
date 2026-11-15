"""Train the three models: LogisticRegression with and without class weights, RandomForest."""

from __future__ import annotations

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, functions as F

from .features import FEATURE_COLS, add_class_weight

MODEL_NAMES = ("lr_weighted", "lr_plain", "rf")
TEST_PERCENT = 30


def split_train_test(df: DataFrame, seed: int = 42) -> tuple[DataFrame, DataFrame]:
    """Deterministic split by customer id hash: the same customer is always on the same side."""
    bucket = F.pmod(F.xxhash64(F.col("customer_id"), F.lit(seed)), F.lit(100))
    return df.where(bucket >= TEST_PERCENT), df.where(bucket < TEST_PERCENT)


def lr_pipeline(weighted: bool) -> Pipeline:
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw")
    scaler = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True)
    kwargs = {"weightCol": "weight"} if weighted else {}
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100, regParam=0.01, **kwargs)
    return Pipeline(stages=[assembler, scaler, lr])


def rf_pipeline(seed: int = 42) -> Pipeline:
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw")
    rf = RandomForestClassifier(
        featuresCol="raw", labelCol="label", weightCol="weight",
        numTrees=50, maxDepth=5, seed=seed,
    )
    return Pipeline(stages=[assembler, rf])


def fit_models(train: DataFrame, seed: int = 42) -> dict[str, PipelineModel]:
    """Fit all three models on the training rows."""
    weighted = add_class_weight(train).cache()
    models = {
        "lr_weighted": lr_pipeline(True).fit(weighted),
        "lr_plain": lr_pipeline(False).fit(weighted),
        "rf": rf_pipeline(seed).fit(weighted),
    }
    weighted.unpersist()
    return models


def save_models(models: dict[str, PipelineModel], path_for) -> None:
    for name, model in models.items():
        model.write().overwrite().save(path_for(name))


def load_model(path: str) -> PipelineModel:
    return PipelineModel.load(path)
