"""Kat-içi ölçeklemeli Lasso CV ve hold-out model karşılaştırmaları."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LassoCVResult:
    alphas: FloatArray
    mean_mse: FloatArray
    standard_error_mse: FloatArray
    lambda_min: float
    lambda_1se: float
    coefficients_min: FloatArray
    coefficients_1se: FloatArray
    selected_min: tuple[str, ...]
    selected_1se: tuple[str, ...]
    fold_training_means: FloatArray
    fold_assignments: NDArray[np.int64]
    seed: int


@dataclass(frozen=True)
class HoldoutComparison:
    model_names: tuple[str, ...]
    test_mse: FloatArray
    nonzero_counts: NDArray[np.int64]
    selected_features: tuple[str, ...]


def cross_validate_lasso(
    features: ArrayLike,
    outcome: ArrayLike,
    alphas: ArrayLike,
    feature_names: tuple[str, ...],
    *,
    folds: int = 5,
    seed: int = 811,
) -> LassoCVResult:
    x = np.asarray(features, dtype=float)
    y = np.asarray(outcome, dtype=float).reshape(-1)
    penalties = np.sort(np.asarray(alphas, dtype=float).reshape(-1))
    if x.ndim != 2 or x.shape[0] != y.size or x.shape[1] != len(feature_names):
        raise ValueError("Lasso CV boyutları eşleşmelidir.")
    if penalties.size == 0 or (penalties <= 0).any():
        raise ValueError("Lasso CV alpha değerleri pozitif olmalıdır.")
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    errors = np.empty((folds, penalties.size))
    training_means = np.empty((folds, x.shape[1]))
    assignments = np.empty(y.size, dtype=np.int64)
    for fold, (train, validation) in enumerate(splitter.split(x)):
        scaler = StandardScaler().fit(x[train])
        training_means[fold] = scaler.mean_
        train_x = scaler.transform(x[train])
        validation_x = scaler.transform(x[validation])
        assignments[validation] = fold
        for index, alpha in enumerate(penalties):
            model = Lasso(alpha=float(alpha), max_iter=20_000).fit(train_x, y[train])
            errors[fold, index] = mean_squared_error(
                y[validation], model.predict(validation_x)
            )
    mean_mse = errors.mean(axis=0)
    standard_error = errors.std(axis=0, ddof=1) / np.sqrt(folds)
    minimum_index = int(np.argmin(mean_mse))
    threshold = mean_mse[minimum_index] + standard_error[minimum_index]
    eligible = np.flatnonzero(mean_mse <= threshold)
    one_se_index = int(eligible[-1])

    scaler = StandardScaler().fit(x)
    standardized = scaler.transform(x)
    minimum_model = Lasso(
        alpha=float(penalties[minimum_index]), max_iter=20_000
    ).fit(standardized, y)
    one_se_model = Lasso(
        alpha=float(penalties[one_se_index]), max_iter=20_000
    ).fit(standardized, y)
    selected_min = tuple(
        name for name, value in zip(feature_names, minimum_model.coef_) if abs(value) > 1e-8
    )
    selected_1se = tuple(
        name for name, value in zip(feature_names, one_se_model.coef_) if abs(value) > 1e-8
    )
    return LassoCVResult(
        alphas=penalties,
        mean_mse=mean_mse,
        standard_error_mse=standard_error,
        lambda_min=float(penalties[minimum_index]),
        lambda_1se=float(penalties[one_se_index]),
        coefficients_min=np.asarray(minimum_model.coef_, dtype=float),
        coefficients_1se=np.asarray(one_se_model.coef_, dtype=float),
        selected_min=selected_min,
        selected_1se=selected_1se,
        fold_training_means=training_means,
        fold_assignments=assignments,
        seed=seed,
    )


def compare_holdout_models(
    train_features: ArrayLike,
    train_outcome: ArrayLike,
    test_features: ArrayLike,
    test_outcome: ArrayLike,
    feature_names: tuple[str, ...],
    *,
    lasso_alpha: float,
    ridge_alpha: float = 1.0,
) -> HoldoutComparison:
    train_x = np.asarray(train_features, dtype=float)
    test_x = np.asarray(test_features, dtype=float)
    train_y = np.asarray(train_outcome, dtype=float).reshape(-1)
    test_y = np.asarray(test_outcome, dtype=float).reshape(-1)
    scaler = StandardScaler().fit(train_x)
    standardized_train = scaler.transform(train_x)
    standardized_test = scaler.transform(test_x)

    ols = LinearRegression().fit(train_x, train_y)
    ridge = Ridge(alpha=ridge_alpha).fit(standardized_train, train_y)
    lasso = Lasso(alpha=lasso_alpha, max_iter=20_000).fit(
        standardized_train, train_y
    )
    selected = np.flatnonzero(np.abs(lasso.coef_) > 1e-8)
    if selected.size:
        post = LinearRegression().fit(train_x[:, selected], train_y)
        post_prediction = post.predict(test_x[:, selected])
    else:
        post_prediction = np.full(test_y.size, train_y.mean())
    predictions = (
        ols.predict(test_x),
        ridge.predict(standardized_test),
        lasso.predict(standardized_test),
        post_prediction,
    )
    return HoldoutComparison(
        model_names=("OLS", "Ridge", "Lasso", "Post-Lasso"),
        test_mse=np.asarray(
            [mean_squared_error(test_y, prediction) for prediction in predictions]
        ),
        nonzero_counts=np.asarray(
            [train_x.shape[1], train_x.shape[1], selected.size, selected.size],
            dtype=np.int64,
        ),
        selected_features=tuple(feature_names[index] for index in selected),
    )
