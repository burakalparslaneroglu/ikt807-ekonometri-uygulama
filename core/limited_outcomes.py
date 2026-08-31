"""Tobit MLE ve Heckman iki aşama için öğretim amaçlı sayısal çekirdek."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.stats import norm

from core.ols import fit_ols


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TobitDGPConfig:
    nobs: int = 1600
    seed: int = 806
    intercept: float = -0.5
    slope: float = 1.5
    sigma: float = 1.0
    censor_point: float = 0.0

    def __post_init__(self) -> None:
        if self.nobs < 300:
            raise ValueError("Tobit DGP'si en az 300 gözlem içermelidir.")
        if self.sigma <= 0:
            raise ValueError("sigma pozitif olmalıdır.")


@dataclass(frozen=True)
class TobitFit:
    coefficient_names: tuple[str, ...]
    coefficients: FloatArray
    sigma: float
    covariance: FloatArray
    converged: bool
    iterations: int
    log_likelihood: float
    gradient_norm: float
    censor_point: float

    def coefficient(self, name: str) -> float:
        try:
            index = self.coefficient_names.index(name)
        except ValueError as error:
            raise ValueError(f"Bilinmeyen Tobit katsayısı: {name}") from error
        return float(self.coefficients[index])


@dataclass(frozen=True)
class HeckmanDGPConfig:
    nobs: int = 5000
    seed: int = 866
    slope: float = 2.0
    error_correlation: float = 0.6
    exclusion_strength: float = 0.9

    def __post_init__(self) -> None:
        if self.nobs < 500:
            raise ValueError("Seçim DGP'si en az 500 gözlem içermelidir.")
        if not -0.95 < self.error_correlation < 0.95:
            raise ValueError("error_correlation -0.95 ile 0.95 arasında olmalıdır.")
        if not 0 <= self.exclusion_strength <= 1.5:
            raise ValueError("exclusion_strength 0 ile 1.5 arasında olmalıdır.")


@dataclass(frozen=True)
class HeckmanTwoStepResult:
    naive_slope: float
    corrected_slope: float
    mills_coefficient: float
    selection_rate: float
    selected_observations: int
    probit_converged: bool


def simulate_tobit_data(config: TobitDGPConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    regressor = rng.normal(size=config.nobs)
    latent = (
        config.intercept
        + config.slope * regressor
        + rng.normal(scale=config.sigma, size=config.nobs)
    )
    observed = np.maximum(config.censor_point, latent)
    return pd.DataFrame({"x": regressor, "latent": latent, "observed": observed})


def fit_tobit(
    outcome: ArrayLike,
    regressors: ArrayLike,
    feature_names: tuple[str, ...],
    *,
    censor_point: float = 0.0,
    tolerance: float = 1e-9,
    max_iterations: int = 2000,
) -> TobitFit:
    y = np.asarray(outcome, dtype=float).reshape(-1)
    x = np.asarray(regressors, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != y.size or x.shape[1] != len(feature_names):
        raise ValueError("Tobit matrisi ve isimleri sonuçla eşleşmelidir.")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("Tobit girdileri sonlu olmalıdır.")
    if (y < censor_point).any():
        raise ValueError("Gözlenen sonuç sansür noktasının altında olamaz.")
    design = np.column_stack((np.ones(y.size), x))
    names = ("Sabit",) + feature_names
    initial_beta = np.linalg.lstsq(design, y, rcond=None)[0]
    initial_residual = y - design @ initial_beta
    initial_sigma = max(float(np.std(initial_residual)), 0.2)
    start = np.r_[initial_beta, np.log(initial_sigma)]
    uncensored = y > censor_point + 1e-12

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        beta = parameters[:-1]
        sigma = float(np.exp(parameters[-1]))
        mean = design @ beta
        standardized_censor = (censor_point - mean) / sigma
        contributions = np.empty(y.size, dtype=float)
        contributions[uncensored] = (
            norm.logpdf((y[uncensored] - mean[uncensored]) / sigma)
            - np.log(sigma)
        )
        contributions[~uncensored] = norm.logcdf(
            standardized_censor[~uncensored]
        )
        if not np.isfinite(contributions).all():
            return 1e100
        return float(-contributions.sum())

    result = minimize(
        negative_log_likelihood,
        start,
        method="L-BFGS-B",
        options={"maxiter": max_iterations, "ftol": tolerance, "gtol": tolerance},
    )
    parameters = np.asarray(result.x, dtype=float)
    inverse_hessian = result.hess_inv
    covariance_all = np.asarray(
        inverse_hessian.todense()
        if hasattr(inverse_hessian, "todense")
        else inverse_hessian,
        dtype=float,
    )
    return TobitFit(
        coefficient_names=names,
        coefficients=parameters[:-1],
        sigma=float(np.exp(parameters[-1])),
        covariance=covariance_all[:-1, :-1],
        converged=bool(result.success),
        iterations=int(result.nit),
        log_likelihood=-float(result.fun),
        gradient_norm=float(np.linalg.norm(result.jac, ord=np.inf)),
        censor_point=censor_point,
    )


def tobit_expectations(fit: TobitFit, design_matrix: ArrayLike) -> pd.DataFrame:
    design = np.asarray(design_matrix, dtype=float)
    if design.ndim == 1:
        design = design[None, :]
    if design.shape[1] != fit.coefficients.size:
        raise ValueError("Tobit tahmin matrisi katsayı boyutuyla eşleşmelidir.")
    mean = design @ fit.coefficients
    standardized = (mean - fit.censor_point) / fit.sigma
    probability = norm.cdf(standardized)
    density = norm.pdf(standardized)
    observed_mean = (
        fit.censor_point * (1 - probability)
        + mean * probability
        + fit.sigma * density
    )
    positive_mean = mean + fit.sigma * density / np.clip(probability, 1e-12, 1)
    return pd.DataFrame(
        {
            "Gizli ortalama": mean,
            "Gözlenen ortalama": observed_mean,
            "Sansürlenmeme olasılığı": probability,
            "Pozitif koşullu ortalama": positive_mean,
        }
    )


def tobit_marginal_effects(
    fit: TobitFit,
    coefficient_name: str,
    design_matrix: ArrayLike,
) -> pd.DataFrame:
    index = fit.coefficient_names.index(coefficient_name)
    design = np.asarray(design_matrix, dtype=float)
    if design.ndim == 1:
        design = design[None, :]
    mean = design @ fit.coefficients
    standardized = (mean - fit.censor_point) / fit.sigma
    probability = norm.cdf(standardized)
    density = norm.pdf(standardized)
    coefficient = fit.coefficients[index]
    return pd.DataFrame(
        {
            "Gizli sonuç etkisi": np.full(design.shape[0], coefficient),
            "Gözlenen ortalama etkisi": probability * coefficient,
            "Sansürlenmeme olasılığı etkisi": density * coefficient / fit.sigma,
        }
    )


def simulate_selection_data(config: HeckmanDGPConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    x = rng.normal(size=config.nobs)
    exclusion = rng.normal(size=config.nobs)
    selection_error = rng.normal(size=config.nobs)
    outcome_error = (
        config.error_correlation * selection_error
        + np.sqrt(1 - config.error_correlation**2) * rng.normal(size=config.nobs)
    )
    selection_index = (
        0.2 + 0.7 * x + config.exclusion_strength * exclusion + selection_error
    )
    selected = selection_index > 0
    latent_outcome = 1.0 + config.slope * x + outcome_error
    observed = np.where(selected, latent_outcome, np.nan)
    return pd.DataFrame(
        {
            "x": x,
            "exclusion": exclusion,
            "selected": selected.astype(int),
            "latent_outcome": latent_outcome,
            "observed_outcome": observed,
        }
    )


def heckman_two_step(frame: pd.DataFrame) -> HeckmanTwoStepResult:
    required = {"x", "exclusion", "selected", "observed_outcome"}
    if not required.issubset(frame.columns):
        raise ValueError("Heckman iki aşama için gerekli sütunlar eksik.")
    selection_design = sm.add_constant(frame[["x", "exclusion"]])
    probit = sm.Probit(frame["selected"], selection_design).fit(
        disp=False, maxiter=300
    )
    index = np.asarray(selection_design @ probit.params, dtype=float)
    probability = np.clip(norm.cdf(index), 1e-10, 1)
    inverse_mills = norm.pdf(index) / probability
    selected = frame["selected"].eq(1)
    selected_frame = frame.loc[selected].copy()
    naive = fit_ols(
        selected_frame["observed_outcome"], selected_frame[["x"]], ("x",)
    )
    corrected_regressors = np.column_stack(
        (selected_frame["x"].to_numpy(), inverse_mills[selected])
    )
    corrected = fit_ols(
        selected_frame["observed_outcome"],
        corrected_regressors,
        ("x", "Ters Mills oranı"),
    )
    return HeckmanTwoStepResult(
        naive_slope=naive.coefficient("x"),
        corrected_slope=corrected.coefficient("x"),
        mills_coefficient=corrected.coefficient("Ters Mills oranı"),
        selection_rate=float(selected.mean()),
        selected_observations=int(selected.sum()),
        probit_converged=bool(probit.mle_retvals["converged"]),
    )
