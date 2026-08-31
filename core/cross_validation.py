"""Yerel regresyon tuning kararları için sızıntısız çapraz doğrulama."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import GroupKFold, KFold

from core.nonparametric import local_polynomial_predict


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BandwidthCVResult:
    selected_bandwidth: float
    bandwidths: FloatArray
    mean_squared_errors: FloatArray
    fold_count: int
    split_unit: str


def _splits(nobs: int, folds: int, seed: int, groups: ArrayLike | None):
    indices = np.arange(nobs)
    if groups is None:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        return tuple(splitter.split(indices))
    labels = np.asarray(groups).reshape(-1)
    if labels.size != nobs:
        raise ValueError("Küme kimlikleri gözlem sayısıyla eşleşmelidir.")
    splitter = GroupKFold(n_splits=folds)
    return tuple(splitter.split(indices, groups=labels))


def cross_validated_predictions(
    x: ArrayLike,
    y: ArrayLike,
    bandwidth: float,
    *,
    folds: int = 5,
    seed: int = 808,
    groups: ArrayLike | None = None,
    degree: int = 1,
    kernel: str = "Gaussian",
) -> FloatArray:
    values = np.asarray(x, dtype=float).reshape(-1)
    outcome = np.asarray(y, dtype=float).reshape(-1)
    if values.size != outcome.size or values.size < folds:
        raise ValueError("CV girdileri aynı uzunlukta olmalı ve kat sayısını aşmalıdır.")
    predictions = np.empty(outcome.size)
    for train, validation in _splits(outcome.size, folds, seed, groups):
        predictions[validation] = local_polynomial_predict(
            values[train],
            outcome[train],
            values[validation],
            bandwidth,
            degree=degree,
            kernel=kernel,
        ).predictions
    return predictions


def select_bandwidth(
    x: ArrayLike,
    y: ArrayLike,
    bandwidths: ArrayLike,
    *,
    folds: int = 5,
    seed: int = 808,
    groups: ArrayLike | None = None,
    degree: int = 1,
    kernel: str = "Gaussian",
) -> BandwidthCVResult:
    candidates = np.asarray(bandwidths, dtype=float).reshape(-1)
    if candidates.size == 0 or (candidates <= 0).any():
        raise ValueError("Bandwidth adayları pozitif olmalıdır.")
    outcome = np.asarray(y, dtype=float).reshape(-1)
    errors = np.empty(candidates.size)
    for index, bandwidth in enumerate(candidates):
        predictions = cross_validated_predictions(
            x,
            outcome,
            float(bandwidth),
            folds=folds,
            seed=seed,
            groups=groups,
            degree=degree,
            kernel=kernel,
        )
        errors[index] = np.mean((outcome - predictions) ** 2)
    selected = int(np.argmin(errors))
    return BandwidthCVResult(
        selected_bandwidth=float(candidates[selected]),
        bandwidths=candidates,
        mean_squared_errors=errors,
        fold_count=folds,
        split_unit="küme" if groups is not None else "gözlem",
    )
