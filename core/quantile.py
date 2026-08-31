"""Kantil regresyon için check-loss, tahmin ve kontrollü DGP araçları."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog
import statsmodels.api as sm


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class QuantileDGPConfig:
    nobs: int = 900
    seed: int = 807
    location_slope: float = 0.8
    scale_slope: float = 0.16

    def __post_init__(self) -> None:
        if self.nobs < 100:
            raise ValueError("Kantil DGP'si için en az 100 gözlem gerekir.")
        if self.scale_slope < 0:
            raise ValueError("Ölçek eğimi negatif olamaz.")


@dataclass(frozen=True)
class QuantileFit:
    tau: float
    coefficient_names: tuple[str, ...]
    coefficients: FloatArray
    standard_errors: FloatArray
    covariance: FloatArray
    fitted_values: FloatArray
    residuals: FloatArray
    iterations: int
    converged: bool

    @property
    def nobs(self) -> int:
        return int(self.residuals.size)

    def coefficient(self, name: str) -> float:
        try:
            index = self.coefficient_names.index(name)
        except ValueError as error:
            raise ValueError(f"Bilinmeyen katsayı: {name}") from error
        return float(self.coefficients[index])

    def standard_error(self, name: str) -> float:
        try:
            index = self.coefficient_names.index(name)
        except ValueError as error:
            raise ValueError(f"Bilinmeyen katsayı: {name}") from error
        return float(self.standard_errors[index])


def _validate_tau(tau: float) -> float:
    value = float(tau)
    if not 0 < value < 1:
        raise ValueError("tau 0 ile 1 arasında olmalıdır.")
    return value


def _design(
    outcome: ArrayLike,
    regressors: ArrayLike,
    feature_names: tuple[str, ...],
    *,
    add_intercept: bool,
) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    y = np.asarray(outcome, dtype=float).reshape(-1)
    x = np.asarray(regressors, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.size == 0 or x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("Sonuç ve açıklayıcı değişken boyutları eşleşmelidir.")
    if not (np.isfinite(y).all() and np.isfinite(x).all()):
        raise ValueError("Kantil regresyon girdileri sonlu olmalıdır.")
    if len(feature_names) != x.shape[1]:
        raise ValueError("feature_names açıklayıcı değişkenlerle eşleşmelidir.")
    names = feature_names
    if add_intercept:
        x = np.column_stack((np.ones(y.size), x))
        names = ("Sabit",) + names
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("Kantil regresyon tasarım matrisi tam sütun rankına sahip değildir.")
    return y, x, names


def check_loss(residuals: ArrayLike, tau: float) -> FloatArray:
    """Asimetrik doğrusal check-loss değerlerini verir."""

    q = _validate_tau(tau)
    u = np.asarray(residuals, dtype=float)
    if not np.isfinite(u).all():
        raise ValueError("Artıklar sonlu olmalıdır.")
    return np.where(u >= 0, q * u, (q - 1) * u)


def simulate_quantile_data(config: QuantileDGPConfig) -> dict[str, FloatArray]:
    """Kantile göre değişen eğim üreten heteroskedastik doğrusal DGP."""

    rng = np.random.default_rng(config.seed)
    x = rng.uniform(0, 10, config.nobs)
    scale = 0.45 + config.scale_slope * x
    shock = rng.normal(size=config.nobs)
    y = 1.0 + config.location_slope * x + scale * shock
    return {"x": x, "y": y, "conditional_mean": 1.0 + config.location_slope * x}


def fit_quantile(
    outcome: ArrayLike,
    regressors: ArrayLike,
    feature_names: tuple[str, ...],
    tau: float,
    *,
    add_intercept: bool = True,
) -> QuantileFit:
    """Statsmodels QuantReg ile robust kovaryanslı doğrusal kantil tahmini."""

    q = _validate_tau(tau)
    y, x, names = _design(
        outcome, regressors, feature_names, add_intercept=add_intercept
    )
    result = sm.QuantReg(y, x).fit(q=q, vcov="robust", max_iter=10_000, p_tol=1e-8)
    coefficients = np.asarray(result.params, dtype=float)
    fitted = x @ coefficients
    iterations = int(getattr(result, "iterations", 0))
    return QuantileFit(
        tau=q,
        coefficient_names=names,
        coefficients=coefficients,
        standard_errors=np.asarray(result.bse, dtype=float),
        covariance=np.asarray(result.cov_params(), dtype=float),
        fitted_values=fitted,
        residuals=y - fitted,
        iterations=iterations,
        converged=iterations < 10_000,
    )


def fit_quantile_linear_program(
    outcome: ArrayLike,
    regressors: ArrayLike,
    feature_names: tuple[str, ...],
    tau: float,
    *,
    add_intercept: bool = True,
) -> FloatArray:
    """Check-loss minimizasyonunu açık doğrusal program olarak çözer."""

    q = _validate_tau(tau)
    y, x, _ = _design(
        outcome, regressors, feature_names, add_intercept=add_intercept
    )
    nobs, nparams = x.shape
    objective = np.concatenate(
        (np.zeros(nparams), np.full(nobs, q), np.full(nobs, 1 - q))
    )
    equality = np.column_stack((x, np.eye(nobs), -np.eye(nobs)))
    bounds = [(None, None)] * nparams + [(0, None)] * (2 * nobs)
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=y,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Kantil doğrusal programı yakınsamadı: {result.message}")
    return np.asarray(result.x[:nparams], dtype=float)


def quantile_profile(
    outcome: ArrayLike,
    regressors: ArrayLike,
    feature_names: tuple[str, ...],
    taus: ArrayLike,
) -> tuple[QuantileFit, ...]:
    values = np.asarray(taus, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("En az bir tau değeri gerekir.")
    return tuple(
        fit_quantile(outcome, regressors, feature_names, float(tau))
        for tau in values
    )
