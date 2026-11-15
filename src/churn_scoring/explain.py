"""Read the models: random-forest importances and logistic-regression coefficients."""

from __future__ import annotations

from .features import FEATURE_COLS


def rf_importances(model) -> list[tuple[str, float]]:
    values = model.stages[-1].featureImportances.toArray()
    return sorted(zip(FEATURE_COLS, (float(v) for v in values)), key=lambda kv: -kv[1])


def lr_coefficients(model) -> list[tuple[str, float]]:
    """Coefficients are per 1 standard deviation because the pipeline scales the features."""
    values = model.stages[-1].coefficients.toArray()
    return sorted(zip(FEATURE_COLS, (float(v) for v in values)), key=lambda kv: -abs(kv[1]))
