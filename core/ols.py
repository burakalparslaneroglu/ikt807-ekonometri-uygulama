"""Doğrusal regresyon ve Frisch-Waugh-Lovell hesapları."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OLSFit:
    """UI'dan bağımsız OLS nokta tahmini ve geometri bilgisi."""

    coefficient_names: tuple[str, ...]
    coefficients: FloatArray
    design_matrix: FloatArray
    outcome: FloatArray
    fitted_values: FloatArray
    residuals: FloatArray
    rank: int

    @property
    def nobs(self) -> int:
        return int(self.outcome.size)

    @property
    def nparams(self) -> int:
        return int(self.coefficients.size)

    @property
    def degrees_of_freedom(self) -> int:
        return self.nobs - self.nparams

    @property
    def r_squared(self) -> float:
        centered = self.outcome - self.outcome.mean()
        total = float(centered @ centered)
        if total == 0:
            return 0.0
        return 1.0 - float(self.residuals @ self.residuals) / total

    def coefficient(self, name: str) -> float:
        try:
            index = self.coefficient_names.index(name)
        except ValueError as error:
            raise ValueError(f"Bilinmeyen katsayı: {name}") from error
        return float(self.coefficients[index])


@dataclass(frozen=True)
class ProjectionDiagnostics:
    mean_residual: float
    max_regressor_residual_product: float
    fitted_residual_product: float


@dataclass(frozen=True)
class FWLResult:
    direct_coefficient: float
    residual_regression_coefficient: float
    residualized_outcome: FloatArray
    residualized_treatment: FloatArray


def _as_vector(values: ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{name} boş olmayan ve sonlu değerlerden oluşmalıdır.")
    return vector


def _as_matrix(values: ArrayLike, nobs: int) -> FloatArray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] != nobs:
        raise ValueError("Açıklayıcı değişken matrisi sonuçla aynı satır sayısında olmalıdır.")
    if not np.isfinite(matrix).all():
        raise ValueError("Açıklayıcı değişken matrisi yalnız sonlu değerler içermelidir.")
    return matrix


def fit_ols(
    outcome: ArrayLike,
    regressors: ArrayLike,
    feature_names: tuple[str, ...],
    *,
    add_intercept: bool = True,
) -> OLSFit:
    """Kararlı least-squares çözümüyle OLS tahmini yapar."""

    y = _as_vector(outcome, "Sonuç")
    x = _as_matrix(regressors, y.size)
    if len(feature_names) != x.shape[1]:
        raise ValueError("feature_names açıklayıcı değişken sütunlarıyla eşleşmelidir.")

    names = feature_names
    if add_intercept:
        x = np.column_stack((np.ones(y.size), x))
        names = ("Sabit",) + names

    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1]:
        raise ValueError("Tasarım matrisi tam sütun rankına sahip değildir.")
    fitted = x @ coefficients
    residuals = y - fitted
    return OLSFit(
        coefficient_names=names,
        coefficients=coefficients,
        design_matrix=x,
        outcome=y,
        fitted_values=fitted,
        residuals=residuals,
        rank=int(rank),
    )


def projection_diagnostics(fit: OLSFit) -> ProjectionDiagnostics:
    """OLS normal denklemlerindeki ortogonallik artıklarını özetler."""

    products = fit.design_matrix.T @ fit.residuals
    return ProjectionDiagnostics(
        mean_residual=float(fit.residuals.mean()),
        max_regressor_residual_product=float(np.max(np.abs(products))),
        fitted_residual_product=float(fit.fitted_values @ fit.residuals),
    )


def residualize(values: ArrayLike, controls: ArrayLike) -> FloatArray:
    """Bir değişkeni sabit ve kontrol seti üzerine yansıtmadan kalan kısmı verir."""

    vector = _as_vector(values, "Artıklaştırılacak değişken")
    matrix = _as_matrix(controls, vector.size)
    fit = fit_ols(
        vector,
        matrix,
        tuple(f"Kontrol {index + 1}" for index in range(matrix.shape[1])),
    )
    return fit.residuals


def fwl_coefficient(
    outcome: ArrayLike,
    treatment: ArrayLike,
    controls: ArrayLike,
) -> FWLResult:
    """Doğrudan çoklu OLS ile artık-üzerine-artık FWL katsayısını karşılaştırır."""

    y = _as_vector(outcome, "Sonuç")
    d = _as_vector(treatment, "Hedef açıklayıcı değişken")
    if y.size != d.size:
        raise ValueError("Sonuç ve hedef açıklayıcı değişken aynı uzunlukta olmalıdır.")
    w = _as_matrix(controls, y.size)
    direct = fit_ols(
        y,
        np.column_stack((d, w)),
        ("Hedef",) + tuple(f"Kontrol {index + 1}" for index in range(w.shape[1])),
    )
    y_residual = residualize(y, w)
    d_residual = residualize(d, w)
    denominator = float(d_residual @ d_residual)
    if denominator <= np.finfo(float).eps:
        raise ValueError("Hedef değişken kontrollerden sonra değişkenlik taşımıyor.")
    residual_coefficient = float(d_residual @ y_residual / denominator)
    return FWLResult(
        direct_coefficient=direct.coefficient("Hedef"),
        residual_regression_coefficient=residual_coefficient,
        residualized_outcome=y_residual,
        residualized_treatment=d_residual,
    )
