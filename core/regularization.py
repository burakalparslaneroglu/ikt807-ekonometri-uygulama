"""Düzenlileştirme yolları, seyrek DGP ve yanlılık-varyans araçları."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SparseDGPConfig:
    nobs: int = 800
    nfeatures: int = 30
    nonzero: int = 6
    noise_scale: float = 1.0
    correlation: float = 0.45
    seed: int = 811

    def __post_init__(self) -> None:
        if self.nobs < 200:
            raise ValueError("Seyrek DGP için en az 200 gözlem gerekir.")
        if self.nfeatures < 5 or not 1 <= self.nonzero < self.nfeatures:
            raise ValueError("Geçerli bir seyrek özellik boyutu gerekir.")
        if self.noise_scale <= 0:
            raise ValueError("Gürültü ölçeği pozitif olmalıdır.")
        if not 0 <= self.correlation < 0.95:
            raise ValueError("Korelasyon 0 ile 0.95 arasında olmalıdır.")


@dataclass(frozen=True)
class SparseRegressionData:
    features: FloatArray
    outcome: FloatArray
    true_coefficients: FloatArray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class RegularizationPath:
    alphas: FloatArray
    ridge_coefficients: FloatArray
    lasso_coefficients: FloatArray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class ComplexityCurve:
    degrees: NDArray[np.int64]
    train_mse: FloatArray
    test_mse: FloatArray
    selected_degree: int


def simulate_sparse_regression(config: SparseDGPConfig) -> SparseRegressionData:
    rng = np.random.default_rng(config.seed)
    indexes = np.arange(config.nfeatures)
    covariance = config.correlation ** np.abs(indexes[:, None] - indexes[None, :])
    features = rng.multivariate_normal(
        np.zeros(config.nfeatures), covariance, size=config.nobs
    )
    coefficients = np.zeros(config.nfeatures)
    magnitudes = np.linspace(2.0, 0.7, config.nonzero)
    coefficients[: config.nonzero] = magnitudes * (-1.0) ** np.arange(config.nonzero)
    outcome = features @ coefficients + rng.normal(
        scale=config.noise_scale, size=config.nobs
    )
    return SparseRegressionData(
        features=features,
        outcome=outcome,
        true_coefficients=coefficients,
        feature_names=tuple(f"X{index + 1:02d}" for index in range(config.nfeatures)),
    )


def soft_threshold(values: ArrayLike, penalty: float) -> FloatArray:
    if penalty < 0:
        raise ValueError("Ceza negatif olamaz.")
    array = np.asarray(values, dtype=float)
    return np.sign(array) * np.maximum(np.abs(array) - penalty, 0)


def orthonormal_lasso_solution(
    outcome: ArrayLike,
    features: ArrayLike,
    penalty: float,
) -> FloatArray:
    y = np.asarray(outcome, dtype=float).reshape(-1)
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("Ortonormal Lasso boyutları eşleşmelidir.")
    gram = x.T @ x / y.size
    if not np.allclose(gram, np.eye(x.shape[1]), atol=1e-8):
        raise ValueError("Kapalı biçim için X'X/n = I gerekir.")
    return soft_threshold(x.T @ y / y.size, penalty)


def regularization_path(
    features: ArrayLike,
    outcome: ArrayLike,
    alphas: ArrayLike,
    feature_names: tuple[str, ...],
) -> RegularizationPath:
    x = np.asarray(features, dtype=float)
    y = np.asarray(outcome, dtype=float).reshape(-1)
    penalties = np.asarray(alphas, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size or x.shape[1] != len(feature_names):
        raise ValueError("Düzenlileştirme yolu boyutları eşleşmelidir.")
    if penalties.size == 0 or (penalties <= 0).any():
        raise ValueError("Alpha değerleri pozitif olmalıdır.")
    scaler = StandardScaler().fit(x)
    standardized = scaler.transform(x)
    ridge = []
    lasso = []
    for alpha in penalties:
        ridge.append(Ridge(alpha=float(alpha)).fit(standardized, y).coef_)
        lasso.append(
            Lasso(alpha=float(alpha), max_iter=20_000).fit(standardized, y).coef_
        )
    return RegularizationPath(
        alphas=penalties,
        ridge_coefficients=np.asarray(ridge),
        lasso_coefficients=np.asarray(lasso),
        feature_names=feature_names,
    )


def polynomial_complexity_curve(
    *,
    nobs: int = 300,
    max_degree: int = 12,
    noise_scale: float = 0.45,
    seed: int = 811,
) -> ComplexityCurve:
    if nobs < 100 or max_degree < 2:
        raise ValueError("Yanlılık-varyans eğrisi için daha büyük örneklem ve derece gerekir.")
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.5, 2.5, nobs)
    mean = 1.0 + np.sin(1.6 * x) + 0.2 * x
    y = mean + rng.normal(scale=noise_scale, size=nobs)
    x_train, x_test, y_train, y_test = train_test_split(
        x[:, None], y, test_size=0.35, random_state=seed
    )
    degrees = np.arange(1, max_degree + 1)
    train_errors = []
    test_errors = []
    for degree in degrees:
        transformer = PolynomialFeatures(degree=int(degree), include_bias=False)
        train_basis = transformer.fit_transform(x_train)
        test_basis = transformer.transform(x_test)
        model = LinearRegression().fit(train_basis, y_train)
        train_errors.append(mean_squared_error(y_train, model.predict(train_basis)))
        test_errors.append(mean_squared_error(y_test, model.predict(test_basis)))
    test_array = np.asarray(test_errors)
    return ComplexityCurve(
        degrees=degrees,
        train_mse=np.asarray(train_errors),
        test_mse=test_array,
        selected_degree=int(degrees[np.argmin(test_array)]),
    )
