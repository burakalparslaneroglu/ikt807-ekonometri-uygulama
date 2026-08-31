"""Klasik, heteroskedastisite-dayanıklı ve küme-dayanıklı OLS çıkarımı."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t

from core.ols import OLSFit


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CoefficientInference:
    estimate: float
    standard_error: float
    statistic: float
    p_value: float
    confidence_interval: tuple[float, float]
    covariance_type: str


def _bread(fit: OLSFit) -> FloatArray:
    return np.linalg.inv(fit.design_matrix.T @ fit.design_matrix)


def classic_covariance(fit: OLSFit) -> FloatArray:
    if fit.degrees_of_freedom <= 0:
        raise ValueError("Klasik kovaryans için pozitif artık serbestlik derecesi gerekir.")
    sigma_squared = float(fit.residuals @ fit.residuals) / fit.degrees_of_freedom
    return sigma_squared * _bread(fit)


def hc1_covariance(fit: OLSFit) -> FloatArray:
    if fit.degrees_of_freedom <= 0:
        raise ValueError("HC1 için pozitif artık serbestlik derecesi gerekir.")
    x = fit.design_matrix
    meat = x.T @ ((fit.residuals**2)[:, None] * x)
    correction = fit.nobs / fit.degrees_of_freedom
    bread = _bread(fit)
    return correction * bread @ meat @ bread


def cluster_covariance(fit: OLSFit, groups: ArrayLike) -> tuple[FloatArray, int]:
    cluster = np.asarray(groups).reshape(-1)
    if cluster.size != fit.nobs:
        raise ValueError("Küme kimlikleri gözlem sayısıyla eşleşmelidir.")
    unique_groups = np.unique(cluster)
    group_count = int(unique_groups.size)
    if group_count < 2:
        raise ValueError("Küme-dayanıklı çıkarım için en az iki küme gerekir.")
    if fit.degrees_of_freedom <= 0:
        raise ValueError("Küme kovaryansı için pozitif serbestlik derecesi gerekir.")

    x = fit.design_matrix
    scores = x * fit.residuals[:, None]
    meat = np.zeros((fit.nparams, fit.nparams), dtype=float)
    for group in unique_groups:
        group_score = scores[cluster == group].sum(axis=0)
        meat += np.outer(group_score, group_score)

    correction = (group_count / (group_count - 1)) * (
        (fit.nobs - 1) / fit.degrees_of_freedom
    )
    bread = _bread(fit)
    return correction * bread @ meat @ bread, group_count


def coefficient_inference(
    fit: OLSFit,
    coefficient_name: str,
    covariance_type: str,
    *,
    groups: ArrayLike | None = None,
    confidence_level: float = 0.95,
) -> CoefficientInference:
    if not 0 < confidence_level < 1:
        raise ValueError("Güven düzeyi 0 ile 1 arasında olmalıdır.")
    if covariance_type == "Klasik":
        covariance = classic_covariance(fit)
        degrees = fit.degrees_of_freedom
    elif covariance_type == "HC1":
        covariance = hc1_covariance(fit)
        degrees = fit.degrees_of_freedom
    elif covariance_type == "Küme":
        if groups is None:
            raise ValueError("Küme çıkarımı için groups zorunludur.")
        covariance, group_count = cluster_covariance(fit, groups)
        degrees = group_count - 1
    else:
        raise ValueError(f"Desteklenmeyen kovaryans türü: {covariance_type}")

    index = fit.coefficient_names.index(coefficient_name)
    estimate = float(fit.coefficients[index])
    standard_error = float(np.sqrt(covariance[index, index]))
    statistic = estimate / standard_error
    alpha = 1 - confidence_level
    critical = float(t.ppf(1 - alpha / 2, df=degrees))
    p_value = float(2 * t.sf(abs(statistic), df=degrees))
    return CoefficientInference(
        estimate=estimate,
        standard_error=standard_error,
        statistic=statistic,
        p_value=p_value,
        confidence_interval=(
            estimate - critical * standard_error,
            estimate + critical * standard_error,
        ),
        covariance_type=covariance_type,
    )


def delta_method(
    gradient: ArrayLike,
    covariance: ArrayLike,
) -> float:
    """Bir dönüşümün delta-yöntemi standart hatasını hesaplar."""

    grad = np.asarray(gradient, dtype=float).reshape(-1)
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (grad.size, grad.size):
        raise ValueError("Gradient ve kovaryans boyutları eşleşmelidir.")
    variance = float(grad @ cov @ grad)
    if variance < -1e-12:
        raise ValueError("Delta-yöntemi varyansı negatif olamaz.")
    return float(np.sqrt(max(variance, 0.0)))
