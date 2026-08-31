"""İkili ve Tobit modelleri için marjinal etki dönüşümleri."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from core.discrete import BinaryModelFit
from core.inference import delta_method


@dataclass(frozen=True)
class MarginalEffectResult:
    effect: float
    standard_error: float
    effect_type: str


def _average_effect_at_coefficients(
    fit: BinaryModelFit,
    coefficients: np.ndarray,
    coefficient_index: int,
) -> float:
    if fit.model_name == "LPM":
        return float(coefficients[coefficient_index])
    index = fit.design_matrix @ coefficients
    if fit.model_name == "Logit":
        probability = expit(index)
        density = probability * (1 - probability)
    elif fit.model_name == "Probit":
        density = norm.pdf(index)
    else:
        raise ValueError(f"Desteklenmeyen model: {fit.model_name}")
    return float(np.mean(density * coefficients[coefficient_index]))


def average_marginal_effect(
    fit: BinaryModelFit,
    coefficient_name: str,
) -> MarginalEffectResult:
    index = fit.coefficient_names.index(coefficient_name)
    effect = _average_effect_at_coefficients(fit, fit.coefficients, index)
    gradient = np.empty(fit.coefficients.size, dtype=float)
    for position in range(fit.coefficients.size):
        step = 1e-6 * max(1.0, abs(float(fit.coefficients[position])))
        upper = fit.coefficients.copy()
        lower = fit.coefficients.copy()
        upper[position] += step
        lower[position] -= step
        gradient[position] = (
            _average_effect_at_coefficients(fit, upper, index)
            - _average_effect_at_coefficients(fit, lower, index)
        ) / (2 * step)
    return MarginalEffectResult(
        effect=effect,
        standard_error=delta_method(gradient, fit.covariance),
        effect_type="Ortalama marjinal etki",
    )


def finite_difference(
    fit: BinaryModelFit,
    coefficient_name: str,
    *,
    low: float = 0.0,
    high: float = 1.0,
) -> MarginalEffectResult:
    index = fit.coefficient_names.index(coefficient_name)

    def effect_at(coefficients: np.ndarray) -> float:
        lower = fit.design_matrix.copy()
        upper = fit.design_matrix.copy()
        lower[:, index] = low
        upper[:, index] = high
        linear_lower = lower @ coefficients
        linear_upper = upper @ coefficients
        if fit.model_name == "LPM":
            return float(np.mean(linear_upper - linear_lower))
        if fit.model_name == "Logit":
            return float(np.mean(expit(linear_upper) - expit(linear_lower)))
        if fit.model_name == "Probit":
            return float(np.mean(norm.cdf(linear_upper) - norm.cdf(linear_lower)))
        raise ValueError(f"Desteklenmeyen model: {fit.model_name}")

    effect = effect_at(fit.coefficients)
    gradient = np.empty(fit.coefficients.size, dtype=float)
    for position in range(fit.coefficients.size):
        step = 1e-6 * max(1.0, abs(float(fit.coefficients[position])))
        upper_coefficients = fit.coefficients.copy()
        lower_coefficients = fit.coefficients.copy()
        upper_coefficients[position] += step
        lower_coefficients[position] -= step
        gradient[position] = (
            effect_at(upper_coefficients) - effect_at(lower_coefficients)
        ) / (2 * step)
    return MarginalEffectResult(
        effect=effect,
        standard_error=delta_method(gradient, fit.covariance),
        effect_type="Ortalama sonlu olasılık farkı",
    )


def observation_marginal_effects(
    fit: BinaryModelFit,
    coefficient_name: str,
) -> np.ndarray:
    index = fit.coefficient_names.index(coefficient_name)
    coefficient = fit.coefficients[index]
    if fit.model_name == "LPM":
        return np.full(fit.nobs, coefficient)
    linear_index = fit.design_matrix @ fit.coefficients
    if fit.model_name == "Logit":
        probability = expit(linear_index)
        return probability * (1 - probability) * coefficient
    if fit.model_name == "Probit":
        return norm.pdf(linear_index) * coefficient
    raise ValueError(f"Desteklenmeyen model: {fit.model_name}")
