"""Double-selection ve cross-fitted partialling-out DML hesapları."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.inference import coefficient_inference
from core.ols import fit_ols


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DMLDGPConfig:
    nobs: int = 800
    nfeatures: int = 24
    theta: float = 1.5
    groups: int = 40
    seed: int = 812

    def __post_init__(self) -> None:
        if self.nobs < 300 or self.nfeatures < 8:
            raise ValueError("DML DGP'si için daha büyük n ve p gerekir.")
        if not 5 <= self.groups <= self.nobs // 5:
            raise ValueError("Geçerli bir grup sayısı gerekir.")


@dataclass(frozen=True)
class DMLData:
    outcome: FloatArray
    treatment: FloatArray
    controls: FloatArray
    groups: NDArray[np.int64]
    feature_names: tuple[str, ...]
    theta: float


@dataclass(frozen=True)
class CrossFitResult:
    outcome_predictions: FloatArray
    treatment_predictions: FloatArray
    fold_assignments: NDArray[np.int64]
    fold_count: int
    split_unit: str


@dataclass(frozen=True)
class DMLResult:
    theta: float
    standard_error: float
    confidence_interval: tuple[float, float]
    residualized_outcome: FloatArray
    residualized_treatment: FloatArray
    cross_fit: CrossFitResult
    learner: str
    covariance_type: str


@dataclass(frozen=True)
class SelectionResult:
    outcome_only_theta: float
    double_selection_theta: float
    outcome_selected: tuple[str, ...]
    treatment_selected: tuple[str, ...]
    union_selected: tuple[str, ...]


def simulate_dml_data(config: DMLDGPConfig) -> DMLData:
    rng = np.random.default_rng(config.seed)
    controls = rng.normal(size=(config.nobs, config.nfeatures))
    group_labels = np.arange(config.nobs) % config.groups
    rng.shuffle(group_labels)
    propensity_index = (
        0.9 * controls[:, 0]
        - 0.7 * controls[:, 4]
        + 0.65 * np.sin(controls[:, 1])
        - 0.35 * controls[:, 5] ** 2
    )
    propensity = np.clip(expit(propensity_index), 0.05, 0.95)
    treatment = rng.binomial(1, propensity).astype(float)
    nuisance = (
        1.2 * controls[:, 0]
        + 1.0 * np.sin(controls[:, 1])
        + 0.7 * controls[:, 2] ** 2
        - 0.5 * controls[:, 3] * controls[:, 4]
    )
    group_shock = rng.normal(scale=0.35, size=config.groups)[group_labels]
    outcome = (
        config.theta * treatment
        + nuisance
        + group_shock
        + rng.normal(scale=1.0, size=config.nobs)
    )
    return DMLData(
        outcome=outcome,
        treatment=treatment,
        controls=controls,
        groups=group_labels.astype(np.int64),
        feature_names=tuple(f"X{index + 1:02d}" for index in range(config.nfeatures)),
        theta=config.theta,
    )


def _learner(name: str, seed: int):
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=70,
            max_depth=7,
            min_samples_leaf=10,
            max_features=0.8,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "Ridge":
        return make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 3, 25)),
        )
    raise ValueError(f"Desteklenmeyen nuisance learner: {name}")


def cross_fit_nuisance(
    outcome: ArrayLike,
    treatment: ArrayLike,
    controls: ArrayLike,
    *,
    folds: int = 5,
    seed: int = 812,
    learner: str = "Random Forest",
    groups: ArrayLike | None = None,
) -> CrossFitResult:
    y = np.asarray(outcome, dtype=float).reshape(-1)
    d = np.asarray(treatment, dtype=float).reshape(-1)
    x = np.asarray(controls, dtype=float)
    if x.ndim != 2 or not (x.shape[0] == y.size == d.size):
        raise ValueError("Cross-fitting boyutları eşleşmelidir.")
    labels = None if groups is None else np.asarray(groups).reshape(-1)
    if labels is not None and labels.size != y.size:
        raise ValueError("Grup kimlikleri gözlem sayısıyla eşleşmelidir.")
    if labels is None:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = splitter.split(x)
        split_unit = "gözlem"
    else:
        splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = splitter.split(x, groups=labels)
        split_unit = "grup"

    y_hat = np.empty(y.size)
    d_hat = np.empty(d.size)
    assignments = np.empty(y.size, dtype=np.int64)
    for fold, (train, validation) in enumerate(splits):
        outcome_model = _learner(learner, seed + 2 * fold)
        treatment_model = _learner(learner, seed + 2 * fold + 1)
        outcome_model.fit(x[train], y[train])
        treatment_model.fit(x[train], d[train])
        y_hat[validation] = outcome_model.predict(x[validation])
        d_hat[validation] = np.clip(
            treatment_model.predict(x[validation]), 0.01, 0.99
        )
        assignments[validation] = fold
    return CrossFitResult(
        outcome_predictions=y_hat,
        treatment_predictions=d_hat,
        fold_assignments=assignments,
        fold_count=folds,
        split_unit=split_unit,
    )


def fit_dml(
    outcome: ArrayLike,
    treatment: ArrayLike,
    controls: ArrayLike,
    *,
    folds: int = 5,
    seed: int = 812,
    learner: str = "Random Forest",
    groups: ArrayLike | None = None,
) -> DMLResult:
    y = np.asarray(outcome, dtype=float).reshape(-1)
    d = np.asarray(treatment, dtype=float).reshape(-1)
    cross_fit = cross_fit_nuisance(
        y,
        d,
        controls,
        folds=folds,
        seed=seed,
        learner=learner,
        groups=groups,
    )
    y_residual = y - cross_fit.outcome_predictions
    d_residual = d - cross_fit.treatment_predictions
    final = fit_ols(
        y_residual,
        d_residual,
        ("theta",),
        add_intercept=False,
    )
    if groups is None:
        inference = coefficient_inference(final, "theta", "HC1")
        covariance_type = "HC1"
    else:
        inference = coefficient_inference(
            final, "theta", "Küme", groups=np.asarray(groups)
        )
        covariance_type = "Küme"
    return DMLResult(
        theta=inference.estimate,
        standard_error=inference.standard_error,
        confidence_interval=inference.confidence_interval,
        residualized_outcome=y_residual,
        residualized_treatment=d_residual,
        cross_fit=cross_fit,
        learner=learner,
        covariance_type=covariance_type,
    )


def double_selection(
    outcome: ArrayLike,
    treatment: ArrayLike,
    controls: ArrayLike,
    feature_names: tuple[str, ...],
    *,
    folds: int = 5,
    seed: int = 812,
) -> SelectionResult:
    y = np.asarray(outcome, dtype=float).reshape(-1)
    d = np.asarray(treatment, dtype=float).reshape(-1)
    x = np.asarray(controls, dtype=float)
    if x.shape != (y.size, len(feature_names)):
        raise ValueError("Double-selection boyutları eşleşmelidir.")
    outcome_model = make_pipeline(
        StandardScaler(),
        LassoCV(cv=folds, random_state=seed, max_iter=20_000),
    ).fit(x, y)
    treatment_model = make_pipeline(
        StandardScaler(),
        LassoCV(cv=folds, random_state=seed + 1, max_iter=20_000),
    ).fit(x, d)
    outcome_coef = outcome_model[-1].coef_
    treatment_coef = treatment_model[-1].coef_
    outcome_indexes = np.flatnonzero(np.abs(outcome_coef) > 1e-8)
    treatment_indexes = np.flatnonzero(np.abs(treatment_coef) > 1e-8)
    union_indexes = np.union1d(outcome_indexes, treatment_indexes)

    def final_theta(indexes: NDArray[np.int64]) -> float:
        regressors = d[:, None]
        names = ("Hedef",)
        if indexes.size:
            regressors = np.column_stack((d, x[:, indexes]))
            names += tuple(feature_names[index] for index in indexes)
        return fit_ols(y, regressors, names).coefficient("Hedef")

    return SelectionResult(
        outcome_only_theta=final_theta(outcome_indexes),
        double_selection_theta=final_theta(union_indexes),
        outcome_selected=tuple(feature_names[index] for index in outcome_indexes),
        treatment_selected=tuple(feature_names[index] for index in treatment_indexes),
        union_selected=tuple(feature_names[index] for index in union_indexes),
    )
