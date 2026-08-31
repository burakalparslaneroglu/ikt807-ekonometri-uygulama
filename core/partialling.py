"""Kısmen doğrusal model için örneklem-içi partialling-out köprüsü."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer

from core.ols import fit_ols


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PartiallingOutResult:
    coefficient: float
    robust_standard_error: float
    naive_linear_coefficient: float
    residualized_outcome: FloatArray
    residualized_treatment: FloatArray
    outcome_nuisance_r_squared: float
    treatment_nuisance_r_squared: float


def partialling_out(
    outcome: ArrayLike,
    treatment: ArrayLike,
    control: ArrayLike,
    *,
    n_knots: int = 6,
) -> PartiallingOutResult:
    y = np.asarray(outcome, dtype=float).reshape(-1)
    d = np.asarray(treatment, dtype=float).reshape(-1)
    z = np.asarray(control, dtype=float).reshape(-1, 1)
    if not (y.size == d.size == z.shape[0]) or y.size == 0:
        raise ValueError("Sonuç, hedef değişken ve kontrol uzunlukları eşleşmelidir.")
    if not (np.isfinite(y).all() and np.isfinite(d).all() and np.isfinite(z).all()):
        raise ValueError("Partialling-out girdileri sonlu olmalıdır.")
    if n_knots < 3:
        raise ValueError("En az üç spline düğümü gerekir.")

    transformer = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)
    basis = transformer.fit_transform(z)
    outcome_model = LinearRegression().fit(basis, y)
    treatment_model = LinearRegression().fit(basis, d)
    y_residual = y - outcome_model.predict(basis)
    d_residual = d - treatment_model.predict(basis)
    denominator = float(d_residual @ d_residual)
    if denominator <= np.finfo(float).eps:
        raise ValueError("Hedef değişken esnek kontrollerden sonra değişkenlik taşımıyor.")
    coefficient = float(d_residual @ y_residual / denominator)
    residual = y_residual - coefficient * d_residual
    correction = y.size / (y.size - 1)
    variance = correction * float(np.sum(d_residual**2 * residual**2)) / denominator**2
    naive = fit_ols(y, np.column_stack((d, z[:, 0])), ("Hedef", "Kontrol"))
    return PartiallingOutResult(
        coefficient=coefficient,
        robust_standard_error=float(np.sqrt(max(variance, 0))),
        naive_linear_coefficient=naive.coefficient("Hedef"),
        residualized_outcome=y_residual,
        residualized_treatment=d_residual,
        outcome_nuisance_r_squared=float(outcome_model.score(basis, y)),
        treatment_nuisance_r_squared=float(treatment_model.score(basis, d)),
    )
