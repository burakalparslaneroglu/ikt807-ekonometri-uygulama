"""Kernel, yerel polinom ve seri regresyonu için şeffaf hesaplama araçları."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LocalPolynomialResult:
    evaluation_points: FloatArray
    predictions: FloatArray
    effective_sample_sizes: FloatArray
    bandwidth: float
    degree: int
    kernel: str


@dataclass(frozen=True)
class NonparametricDGPConfig:
    nobs: int = 500
    seed: int = 808
    noise_scale: float = 0.45

    def __post_init__(self) -> None:
        if self.nobs < 80:
            raise ValueError("Esnek regresyon DGP'si için en az 80 gözlem gerekir.")
        if self.noise_scale <= 0:
            raise ValueError("Gürültü ölçeği pozitif olmalıdır.")


def simulate_nonlinear_data(config: NonparametricDGPConfig) -> dict[str, FloatArray]:
    rng = np.random.default_rng(config.seed)
    x = np.sort(rng.uniform(0, 10, config.nobs))
    mean = 1.2 + 0.22 * x + 1.15 * np.sin(0.85 * x)
    y = mean + rng.normal(scale=config.noise_scale, size=config.nobs)
    return {"x": x, "y": y, "conditional_mean": mean}


def kernel_values(distances: ArrayLike, kernel: str = "Gaussian") -> FloatArray:
    u = np.asarray(distances, dtype=float)
    if kernel == "Gaussian":
        return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    if kernel == "Epanechnikov":
        return 0.75 * np.maximum(1 - u**2, 0) * (np.abs(u) <= 1)
    if kernel == "Üçgensel":
        return np.maximum(1 - np.abs(u), 0)
    raise ValueError(f"Desteklenmeyen kernel: {kernel}")


def kernel_weights(
    x: ArrayLike,
    evaluation_point: float,
    bandwidth: float,
    kernel: str = "Gaussian",
) -> FloatArray:
    values = np.asarray(x, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("x boş olmayan sonlu değerlerden oluşmalıdır.")
    if bandwidth <= 0:
        raise ValueError("Bandwidth pozitif olmalıdır.")
    raw = kernel_values((values - float(evaluation_point)) / bandwidth, kernel)
    total = float(raw.sum())
    if total <= np.finfo(float).eps:
        raise ValueError("Bu değerlendirme noktasında pozitif kernel ağırlığı yok.")
    return raw / total


def effective_sample_size(weights: ArrayLike) -> float:
    values = np.asarray(weights, dtype=float).reshape(-1)
    if values.size == 0 or (values < 0).any() or not np.isfinite(values).all():
        raise ValueError("Ağırlıklar sonlu ve negatif olmayan değerler olmalıdır.")
    denominator = float(values @ values)
    if denominator <= np.finfo(float).eps:
        raise ValueError("Etkin örneklem için pozitif ağırlık gerekir.")
    return float(values.sum() ** 2 / denominator)


def local_polynomial_predict(
    x: ArrayLike,
    y: ArrayLike,
    evaluation_points: ArrayLike,
    bandwidth: float,
    *,
    degree: int = 1,
    kernel: str = "Gaussian",
) -> LocalPolynomialResult:
    values = np.asarray(x, dtype=float).reshape(-1)
    outcome = np.asarray(y, dtype=float).reshape(-1)
    grid = np.asarray(evaluation_points, dtype=float).reshape(-1)
    if values.size != outcome.size or values.size == 0:
        raise ValueError("x ve y aynı pozitif uzunlukta olmalıdır.")
    if not (np.isfinite(values).all() and np.isfinite(outcome).all() and np.isfinite(grid).all()):
        raise ValueError("Yerel regresyon girdileri sonlu olmalıdır.")
    if degree not in (0, 1):
        raise ValueError("Öğretim çekirdeği yalnız yerel sabit ve yerel doğrusal destekler.")

    predictions = np.empty(grid.size)
    effective = np.empty(grid.size)
    for index, point in enumerate(grid):
        weights = kernel_weights(values, point, bandwidth, kernel)
        centered = values - point
        design = np.ones((values.size, degree + 1))
        if degree == 1:
            design[:, 1] = centered
        root = np.sqrt(weights)
        coefficients, _, rank, _ = np.linalg.lstsq(
            design * root[:, None], outcome * root, rcond=None
        )
        if rank < degree + 1:
            raise ValueError("Yerel tasarım matrisi seçilen noktada tekildir.")
        predictions[index] = coefficients[0]
        effective[index] = effective_sample_size(weights)
    return LocalPolynomialResult(
        evaluation_points=grid,
        predictions=predictions,
        effective_sample_sizes=effective,
        bandwidth=float(bandwidth),
        degree=degree,
        kernel=kernel,
    )


def spline_series_predict(
    x: ArrayLike,
    y: ArrayLike,
    evaluation_points: ArrayLike,
    n_knots: int,
) -> FloatArray:
    values = np.asarray(x, dtype=float).reshape(-1, 1)
    outcome = np.asarray(y, dtype=float).reshape(-1)
    grid = np.asarray(evaluation_points, dtype=float).reshape(-1, 1)
    if values.shape[0] != outcome.size:
        raise ValueError("x ve y uzunlukları eşleşmelidir.")
    if n_knots < 3:
        raise ValueError("Spline seri yaklaşımı için en az üç düğüm gerekir.")
    model = make_pipeline(
        SplineTransformer(n_knots=n_knots, degree=3, include_bias=False),
        LinearRegression(),
    )
    model.fit(values, outcome)
    return np.asarray(model.predict(grid), dtype=float)
