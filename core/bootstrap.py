"""OLS katsayıları için pairs ve wild bootstrap çıkarımı."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.inference import hc1_covariance
from core.ols import OLSFit, fit_ols


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BootstrapResult:
    coefficient_name: str
    original_estimate: float
    analytic_standard_error: float
    draws: FloatArray
    draw_standard_errors: FloatArray
    bootstrap_standard_error: float
    normal_interval: tuple[float, float]
    percentile_interval: tuple[float, float]
    percentile_t_interval: tuple[float, float]
    repetitions: int
    seed: int
    method: str
    resampling_unit: str

    @property
    def monte_carlo_standard_error(self) -> float:
        return self.bootstrap_standard_error / np.sqrt(2 * (self.repetitions - 1))


def _fit_from_design(
    outcome: FloatArray,
    design: FloatArray,
    names: tuple[str, ...],
) -> OLSFit:
    coefficients, _, rank, _ = np.linalg.lstsq(design, outcome, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("Bootstrap tasarım matrisi tam sütun rankına sahip değildir.")
    fitted = design @ coefficients
    return OLSFit(
        coefficient_names=names,
        coefficients=coefficients,
        design_matrix=design,
        outcome=outcome,
        fitted_values=fitted,
        residuals=outcome - fitted,
        rank=int(rank),
    )


def bootstrap_ols_coefficient(
    outcome: ArrayLike,
    regressors: ArrayLike,
    feature_names: tuple[str, ...],
    coefficient_name: str,
    *,
    repetitions: int = 500,
    seed: int = 810,
    method: str = "Pairs",
) -> BootstrapResult:
    if repetitions < 50:
        raise ValueError("Bootstrap için en az 50 tekrar gerekir.")
    if method not in ("Pairs", "Wild"):
        raise ValueError("Bootstrap yöntemi Pairs veya Wild olmalıdır.")
    original = fit_ols(outcome, regressors, feature_names)
    coefficient_index = original.coefficient_names.index(coefficient_name)
    original_covariance = hc1_covariance(original)
    analytic_se = float(np.sqrt(original_covariance[coefficient_index, coefficient_index]))
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    draw_standard_errors = np.empty(repetitions)

    for repeat in range(repetitions):
        if method == "Pairs":
            indices = rng.integers(0, original.nobs, original.nobs)
            fit = _fit_from_design(
                original.outcome[indices],
                original.design_matrix[indices],
                original.coefficient_names,
            )
        else:
            signs = rng.choice((-1.0, 1.0), size=original.nobs)
            bootstrap_y = original.fitted_values + original.residuals * signs
            fit = _fit_from_design(
                bootstrap_y,
                original.design_matrix,
                original.coefficient_names,
            )
        covariance = hc1_covariance(fit)
        draws[repeat] = fit.coefficients[coefficient_index]
        draw_standard_errors[repeat] = np.sqrt(
            covariance[coefficient_index, coefficient_index]
        )

    estimate = original.coefficient(coefficient_name)
    bootstrap_se = float(draws.std(ddof=1))
    percentile = tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))
    studentized = (draws - estimate) / draw_standard_errors
    t_low, t_high = np.quantile(studentized, (0.025, 0.975))
    return BootstrapResult(
        coefficient_name=coefficient_name,
        original_estimate=estimate,
        analytic_standard_error=analytic_se,
        draws=draws,
        draw_standard_errors=draw_standard_errors,
        bootstrap_standard_error=bootstrap_se,
        normal_interval=(
            estimate - 1.96 * bootstrap_se,
            estimate + 1.96 * bootstrap_se,
        ),
        percentile_interval=percentile,
        percentile_t_interval=(
            float(estimate - t_high * analytic_se),
            float(estimate - t_low * analytic_se),
        ),
        repetitions=repetitions,
        seed=seed,
        method=method,
        resampling_unit="gözlem",
    )
