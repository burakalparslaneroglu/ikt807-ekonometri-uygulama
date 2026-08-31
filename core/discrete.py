"""LPM, Logit ve Probit için ortak ikili sonuç sözleşmeleri."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit
from scipy.stats import norm

from core.inference import hc1_covariance
from core.ols import fit_ols


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BinaryDGPConfig:
    nobs: int = 1400
    seed: int = 805
    age_effect: float = 0.20
    education_effect: float = 0.05
    metro_effect: float = 0.35
    intercept: float = -6.0

    def __post_init__(self) -> None:
        if self.nobs < 300:
            raise ValueError("İkili sonuç DGP'si en az 300 gözlem içermelidir.")
        if self.seed < 0:
            raise ValueError("seed negatif olamaz.")


@dataclass(frozen=True)
class BinaryModelFit:
    model_name: str
    coefficient_names: tuple[str, ...]
    coefficients: FloatArray
    covariance: FloatArray
    design_matrix: FloatArray
    outcome: FloatArray
    predicted_probabilities: FloatArray
    converged: bool

    @property
    def nobs(self) -> int:
        return int(self.outcome.size)

    def coefficient(self, name: str) -> float:
        try:
            index = self.coefficient_names.index(name)
        except ValueError as error:
            raise ValueError(f"Bilinmeyen ikili model katsayısı: {name}") from error
        return float(self.coefficients[index])

    def predict(self, design_matrix: ArrayLike) -> FloatArray:
        matrix = np.asarray(design_matrix, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.shape[1] != self.coefficients.size:
            raise ValueError("Tahmin matrisi katsayı boyutuyla eşleşmelidir.")
        index = matrix @ self.coefficients
        if self.model_name == "LPM":
            return index
        if self.model_name == "Logit":
            return expit(index)
        if self.model_name == "Probit":
            return norm.cdf(index)
        raise ValueError(f"Desteklenmeyen model: {self.model_name}")


def simulate_binary_data(config: BinaryDGPConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    age = rng.integers(18, 36, size=config.nobs).astype(float)
    education = np.clip(
        np.rint(rng.normal(13.5 + 0.08 * (age - 26), 2.2, size=config.nobs)),
        8,
        20,
    )
    metro = rng.binomial(1, 0.58, size=config.nobs).astype(float)
    index = (
        config.intercept
        + config.age_effect * age
        + config.education_effect * education
        + config.metro_effect * metro
    )
    probability = expit(index)
    outcome = rng.binomial(1, probability, size=config.nobs)
    return pd.DataFrame(
        {
            "age": age,
            "education": education,
            "metro": metro,
            "married": outcome,
            "true_probability": probability,
        }
    )


def fit_binary_model(
    outcome: ArrayLike,
    regressors: ArrayLike,
    feature_names: tuple[str, ...],
    model_name: str,
) -> BinaryModelFit:
    y = np.asarray(outcome, dtype=float).reshape(-1)
    x = np.asarray(regressors, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != y.size or x.shape[1] != len(feature_names):
        raise ValueError("İkili model matrisi ve isimleri sonuçla eşleşmelidir.")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("İkili model sonucu yalnız 0 ve 1 değerlerini almalıdır.")
    design = np.column_stack((np.ones(y.size), x))
    names = ("Sabit",) + feature_names

    if model_name == "LPM":
        ols = fit_ols(y, x, feature_names)
        coefficients = ols.coefficients
        covariance = hc1_covariance(ols)
        probabilities = ols.fitted_values
        converged = True
    elif model_name in {"Logit", "Probit"}:
        link = (
            sm.families.links.Logit()
            if model_name == "Logit"
            else sm.families.links.Probit()
        )
        result = sm.GLM(
            y,
            design,
            family=sm.families.Binomial(link=link),
        ).fit(cov_type="HC1", maxiter=200)
        coefficients = np.asarray(result.params, dtype=float)
        covariance = np.asarray(result.cov_params(), dtype=float)
        probabilities = np.asarray(result.predict(), dtype=float)
        converged = bool(result.converged)
    else:
        raise ValueError(f"Desteklenmeyen ikili model: {model_name}")

    return BinaryModelFit(
        model_name=model_name,
        coefficient_names=names,
        coefficients=coefficients,
        covariance=covariance,
        design_matrix=design,
        outcome=y,
        predicted_probabilities=probabilities,
        converged=converged,
    )
