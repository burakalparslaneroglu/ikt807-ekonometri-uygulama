"""Şeffaf Wald, 2SLS, ilk aşama ve robust IV kovaryans hesapları."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from core.inference import hc1_covariance
from core.ols import OLSFit, fit_ols, residualize


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class IVConfig:
    nobs: int = 1200
    seed: int = 804
    structural_effect: float = 0.12
    instrument_strength: float = 0.35
    endogeneity: float = 0.65
    exclusion_violation: float = 0.0

    def __post_init__(self) -> None:
        if self.nobs < 200:
            raise ValueError("IV DGP'si en az 200 gözlem içermelidir.")
        if self.seed < 0:
            raise ValueError("seed negatif olamaz.")
        if not 0.01 <= self.instrument_strength <= 1.5:
            raise ValueError("instrument_strength 0.01 ile 1.5 arasında olmalıdır.")
        if not 0 <= self.endogeneity < 1:
            raise ValueError("endogeneity 0 ile 1 arasında olmalıdır.")
        if not 0 <= self.exclusion_violation <= 0.3:
            raise ValueError("exclusion_violation 0 ile 0.3 arasında olmalıdır.")


@dataclass(frozen=True)
class IVFit:
    coefficient_names: tuple[str, ...]
    coefficients: FloatArray
    covariance: FloatArray
    structural_residuals: FloatArray
    fitted_endogenous: FloatArray
    nobs: int
    first_stage_f: float

    def coefficient(self, name: str) -> float:
        try:
            index = self.coefficient_names.index(name)
        except ValueError as error:
            raise ValueError(f"Bilinmeyen IV katsayısı: {name}") from error
        return float(self.coefficients[index])

    def standard_error(self, name: str) -> float:
        index = self.coefficient_names.index(name)
        return float(np.sqrt(self.covariance[index, index]))


def _matrix(values: ArrayLike, nobs: int, label: str) -> FloatArray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] != nobs:
        raise ValueError(f"{label} gözlem sayısıyla eşleşmelidir.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} yalnız sonlu değerler içermelidir.")
    return matrix


def simulate_iv_data(config: IVConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    instrument = rng.binomial(1, 0.5, size=config.nobs).astype(float)
    control = rng.normal(size=config.nobs)
    structural_shock = rng.normal(size=config.nobs)
    first_stage_noise = (
        config.endogeneity * structural_shock
        + np.sqrt(1 - config.endogeneity**2) * rng.normal(size=config.nobs)
    )
    education = (
        12
        + config.instrument_strength * instrument
        + 0.7 * control
        + first_stage_noise
    )
    log_wage = (
        1.2
        + config.structural_effect * education
        + 0.18 * control
        + config.exclusion_violation * instrument
        + structural_shock
    )
    return pd.DataFrame(
        {
            "instrument": instrument,
            "control": control,
            "education": education,
            "lwage": log_wage,
        }
    )


def first_stage_fit(
    endogenous: ArrayLike,
    instruments: ArrayLike,
    *,
    exogenous: ArrayLike | None = None,
) -> tuple[OLSFit, float]:
    x = np.asarray(endogenous, dtype=float).reshape(-1)
    z = _matrix(instruments, x.size, "Araç matrisi")
    if exogenous is None:
        w = np.empty((x.size, 0))
    else:
        w = _matrix(exogenous, x.size, "Dışsal kontrol matrisi")
    regressors = np.column_stack((w, z))
    names = tuple(f"Kontrol {i + 1}" for i in range(w.shape[1])) + tuple(
        f"Araç {i + 1}" for i in range(z.shape[1])
    )
    fit = fit_ols(x, regressors, names)
    covariance = hc1_covariance(fit)
    instrument_indexes = np.arange(1 + w.shape[1], fit.nparams)
    coefficients = fit.coefficients[instrument_indexes]
    block = covariance[np.ix_(instrument_indexes, instrument_indexes)]
    statistic = float(coefficients @ np.linalg.inv(block) @ coefficients)
    return fit, statistic / z.shape[1]


def fit_2sls(
    outcome: ArrayLike,
    endogenous: ArrayLike,
    instruments: ArrayLike,
    *,
    exogenous: ArrayLike | None = None,
    endogenous_name: str = "Endojen değişken",
    exogenous_names: tuple[str, ...] = (),
    debiased: bool = True,
) -> IVFit:
    y = np.asarray(outcome, dtype=float).reshape(-1)
    d = _matrix(endogenous, y.size, "Endojen değişken matrisi")
    z_excluded = _matrix(instruments, y.size, "Araç matrisi")
    if d.shape[1] != 1:
        raise ValueError("Bu öğretim sürümü tek endojen değişkeni destekler.")
    if exogenous is None:
        w = np.empty((y.size, 0))
    else:
        w = _matrix(exogenous, y.size, "Dışsal kontrol matrisi")
    if len(exogenous_names) != w.shape[1]:
        raise ValueError("exogenous_names kontrol sütunlarıyla eşleşmelidir.")

    x = np.column_stack((np.ones(y.size), w, d))
    z = np.column_stack((np.ones(y.size), w, z_excluded))
    if np.linalg.matrix_rank(x) < x.shape[1] or np.linalg.matrix_rank(z) < z.shape[1]:
        raise ValueError("IV tasarım veya araç matrisi tam sütun rankına sahip değildir.")
    if z.shape[1] < x.shape[1]:
        raise ValueError("Araç sayısı endojen parametreleri tanımlamak için yetersizdir.")

    ztz_inverse = np.linalg.inv(z.T @ z)
    cross = x.T @ z @ ztz_inverse @ z.T @ x
    coefficients = np.linalg.solve(
        cross,
        x.T @ z @ ztz_inverse @ z.T @ y,
    )
    residuals = y - x @ coefficients
    score_covariance = z.T @ ((residuals**2)[:, None] * z)
    middle = (
        x.T
        @ z
        @ ztz_inverse
        @ score_covariance
        @ ztz_inverse
        @ z.T
        @ x
    )
    cross_inverse = np.linalg.inv(cross)
    covariance = cross_inverse @ middle @ cross_inverse
    if debiased:
        covariance *= y.size / (y.size - x.shape[1])

    stage, first_stage_f = first_stage_fit(d[:, 0], z_excluded, exogenous=w)
    names = ("Sabit",) + exogenous_names + (endogenous_name,)
    return IVFit(
        coefficient_names=names,
        coefficients=coefficients,
        covariance=covariance,
        structural_residuals=residuals,
        fitted_endogenous=stage.fitted_values,
        nobs=y.size,
        first_stage_f=first_stage_f,
    )


def conditional_wald_ratio(
    outcome: ArrayLike,
    endogenous: ArrayLike,
    instrument: ArrayLike,
    *,
    exogenous: ArrayLike | None = None,
) -> tuple[float, float, float]:
    y = np.asarray(outcome, dtype=float).reshape(-1)
    d = np.asarray(endogenous, dtype=float).reshape(-1)
    z = np.asarray(instrument, dtype=float).reshape(-1)
    if not (y.size == d.size == z.size):
        raise ValueError("Wald değişkenleri aynı uzunlukta olmalıdır.")
    if exogenous is None:
        controls = np.ones((y.size, 1))
        y_residual = y - y.mean()
        d_residual = d - d.mean()
        z_residual = z - z.mean()
    else:
        controls = _matrix(exogenous, y.size, "Dışsal kontrol matrisi")
        y_residual = residualize(y, controls)
        d_residual = residualize(d, controls)
        z_residual = residualize(z, controls)
    del controls
    denominator = float(z_residual @ z_residual)
    first_stage = float(z_residual @ d_residual / denominator)
    reduced_form = float(z_residual @ y_residual / denominator)
    if abs(first_stage) <= np.finfo(float).eps:
        raise ValueError("İlk aşama sıfır; Wald oranı tanımsız.")
    return first_stage, reduced_form, reduced_form / first_stage


def naive_second_stage_fit(
    outcome: ArrayLike,
    iv_fit: IVFit,
    *,
    exogenous: ArrayLike | None = None,
    exogenous_names: tuple[str, ...] = (),
) -> OLSFit:
    y = np.asarray(outcome, dtype=float).reshape(-1)
    if exogenous is None:
        regressors = iv_fit.fitted_endogenous[:, None]
        names = ("Tahmin edilen endojen",)
    else:
        w = _matrix(exogenous, y.size, "Dışsal kontrol matrisi")
        regressors = np.column_stack((w, iv_fit.fitted_endogenous))
        names = exogenous_names + ("Tahmin edilen endojen",)
    return fit_ols(y, regressors, names)


def weak_instrument_monte_carlo(
    strengths: tuple[float, ...],
    *,
    nobs: int,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    if repetitions < 20:
        raise ValueError("Monte Carlo en az 20 tekrar içermelidir.")
    rows: list[dict[str, float | int]] = []
    for strength in strengths:
        for repetition in range(repetitions):
            frame = simulate_iv_data(
                IVConfig(
                    nobs=nobs,
                    seed=seed + repetition + round(strength * 100_000),
                    instrument_strength=strength,
                )
            )
            fit = fit_2sls(
                frame["lwage"],
                frame["education"],
                frame["instrument"],
                exogenous=frame[["control"]],
                endogenous_name="Eğitim",
                exogenous_names=("Kontrol",),
            )
            rows.append(
                {
                    "Araç gücü": strength,
                    "Tekrar": repetition,
                    "2SLS": fit.coefficient("Eğitim"),
                    "İlk aşama F": fit.first_stage_f,
                }
            )
    return pd.DataFrame(rows)
