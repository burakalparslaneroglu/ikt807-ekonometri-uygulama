"""Sharp ve fuzzy RDD için yerel polinom tahmin ve tanı araçları."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
import statsmodels.api as sm


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RDDDGPConfig:
    nobs: int = 1600
    seed: int = 809
    cutoff: float = 0.0
    treatment_effect: float = -2.0
    first_stage_jump: float = 0.55
    manipulation_strength: float = 0.0

    def __post_init__(self) -> None:
        if self.nobs < 200:
            raise ValueError("RDD DGP'si için en az 200 gözlem gerekir.")
        if not 0.05 <= self.first_stage_jump <= 0.75:
            raise ValueError("İlk aşama sıçraması 0.05 ile 0.75 arasında olmalıdır.")
        if not 0 <= self.manipulation_strength <= 1:
            raise ValueError("Manipülasyon gücü 0 ile 1 arasında olmalıdır.")


@dataclass(frozen=True)
class RDDEstimate:
    estimate: float
    standard_error: float
    confidence_interval: tuple[float, float]
    left_limit: float
    right_limit: float
    n_left: int
    n_right: int
    bandwidth: float
    cutoff: float
    degree: int
    kernel: str
    left_coefficients: FloatArray
    right_coefficients: FloatArray

    @property
    def effective_n(self) -> int:
        return self.n_left + self.n_right


@dataclass(frozen=True)
class FuzzyRDDResult:
    local_wald: float
    reduced_form: float
    first_stage: float
    outcome_fit: RDDEstimate
    treatment_fit: RDDEstimate


def simulate_rdd_data(config: RDDDGPConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    running = rng.uniform(-10, 10, config.nobs)
    near_left = (running < config.cutoff) & (running > config.cutoff - 1.5)
    moved = near_left & (
        rng.uniform(size=config.nobs) < config.manipulation_strength
    )
    running[moved] = config.cutoff + 0.12 + np.abs(running[moved] - config.cutoff) * 0.35

    assigned = (running >= config.cutoff).astype(float)
    probability = 0.18 + config.first_stage_jump * assigned
    treatment = rng.binomial(1, probability, config.nobs).astype(float)
    centered = running - config.cutoff
    baseline = 5.2 + 0.22 * centered + 0.018 * centered**2
    noise = rng.normal(scale=1.0 + 0.025 * np.abs(centered), size=config.nobs)
    return pd.DataFrame(
        {
            "running": running,
            "assigned": assigned,
            "treatment": treatment,
            "treatment_probability": probability,
            "outcome_sharp": baseline + config.treatment_effect * assigned + noise,
            "outcome_fuzzy": baseline + config.treatment_effect * treatment + noise,
            "baseline": baseline,
        }
    )


def _kernel_values(values: FloatArray, kernel: str) -> FloatArray:
    distance = np.abs(values)
    if kernel == "Üçgensel":
        return np.maximum(1 - distance, 0)
    if kernel == "Uniform":
        return (distance <= 1).astype(float)
    if kernel == "Epanechnikov":
        return 0.75 * np.maximum(1 - values**2, 0) * (distance <= 1)
    raise ValueError(f"Desteklenmeyen RDD kernel'i: {kernel}")


def _side_fit(
    outcome: FloatArray,
    running: FloatArray,
    cutoff: float,
    bandwidth: float,
    side: str,
    degree: int,
    kernel: str,
):
    centered = running - cutoff
    side_mask = centered < 0 if side == "left" else centered >= 0
    mask = side_mask & (np.abs(centered) <= bandwidth)
    local_x = centered[mask]
    local_y = outcome[mask]
    if local_y.size < degree + 4:
        raise ValueError("Seçilen bandwidth bir RDD tarafında çok az gözlem bırakıyor.")
    design = np.column_stack([local_x**power for power in range(degree + 1)])
    weights = _kernel_values(local_x / bandwidth, kernel)
    result = sm.WLS(local_y, design, weights=weights).fit(cov_type="HC1")
    return result, int(local_y.size)


def fit_sharp_rdd(
    outcome: ArrayLike,
    running: ArrayLike,
    *,
    cutoff: float = 0.0,
    bandwidth: float = 4.0,
    degree: int = 1,
    kernel: str = "Üçgensel",
) -> RDDEstimate:
    y = np.asarray(outcome, dtype=float).reshape(-1)
    x = np.asarray(running, dtype=float).reshape(-1)
    if y.size != x.size or y.size == 0:
        raise ValueError("RDD sonuç ve eşik değişkeni aynı uzunlukta olmalıdır.")
    if not (np.isfinite(y).all() and np.isfinite(x).all()):
        raise ValueError("RDD girdileri sonlu olmalıdır.")
    if bandwidth <= 0:
        raise ValueError("Bandwidth pozitif olmalıdır.")
    if degree not in (1, 2):
        raise ValueError("RDD öğretim çekirdeği derece 1 veya 2 destekler.")

    left, n_left = _side_fit(y, x, cutoff, bandwidth, "left", degree, kernel)
    right, n_right = _side_fit(y, x, cutoff, bandwidth, "right", degree, kernel)
    left_limit = float(left.params[0])
    right_limit = float(right.params[0])
    estimate = right_limit - left_limit
    variance = float(left.cov_params()[0, 0] + right.cov_params()[0, 0])
    standard_error = float(np.sqrt(max(variance, 0)))
    return RDDEstimate(
        estimate=estimate,
        standard_error=standard_error,
        confidence_interval=(
            estimate - 1.96 * standard_error,
            estimate + 1.96 * standard_error,
        ),
        left_limit=left_limit,
        right_limit=right_limit,
        n_left=n_left,
        n_right=n_right,
        bandwidth=float(bandwidth),
        cutoff=float(cutoff),
        degree=degree,
        kernel=kernel,
        left_coefficients=np.asarray(left.params, dtype=float),
        right_coefficients=np.asarray(right.params, dtype=float),
    )


def fitted_rdd_lines(
    fit: RDDEstimate,
    points_per_side: int = 80,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    left_x = np.linspace(fit.cutoff - fit.bandwidth, fit.cutoff, points_per_side)
    right_x = np.linspace(fit.cutoff, fit.cutoff + fit.bandwidth, points_per_side)

    def evaluate(grid: FloatArray, coefficients: FloatArray) -> FloatArray:
        centered = grid - fit.cutoff
        design = np.column_stack(
            [centered**power for power in range(fit.degree + 1)]
        )
        return design @ coefficients

    return (
        left_x,
        evaluate(left_x, fit.left_coefficients),
        right_x,
        evaluate(right_x, fit.right_coefficients),
    )


def fit_fuzzy_rdd(
    outcome: ArrayLike,
    treatment: ArrayLike,
    running: ArrayLike,
    *,
    cutoff: float = 0.0,
    bandwidth: float = 4.0,
    degree: int = 1,
    kernel: str = "Üçgensel",
    minimum_first_stage: float = 0.03,
) -> FuzzyRDDResult:
    outcome_fit = fit_sharp_rdd(
        outcome,
        running,
        cutoff=cutoff,
        bandwidth=bandwidth,
        degree=degree,
        kernel=kernel,
    )
    treatment_fit = fit_sharp_rdd(
        treatment,
        running,
        cutoff=cutoff,
        bandwidth=bandwidth,
        degree=degree,
        kernel=kernel,
    )
    if abs(treatment_fit.estimate) < minimum_first_stage:
        raise ValueError("Fuzzy RDD yerel Wald oranı için ilk aşama sıçraması çok zayıf.")
    return FuzzyRDDResult(
        local_wald=outcome_fit.estimate / treatment_fit.estimate,
        reduced_form=outcome_fit.estimate,
        first_stage=treatment_fit.estimate,
        outcome_fit=outcome_fit,
        treatment_fit=treatment_fit,
    )


def density_ratio_near_cutoff(
    running: ArrayLike,
    *,
    cutoff: float = 0.0,
    window: float = 1.0,
) -> tuple[float, int, int]:
    x = np.asarray(running, dtype=float).reshape(-1)
    if window <= 0:
        raise ValueError("Yoğunluk penceresi pozitif olmalıdır.")
    left = int(((x >= cutoff - window) & (x < cutoff)).sum())
    right = int(((x >= cutoff) & (x <= cutoff + window)).sum())
    if left == 0:
        raise ValueError("Yoğunluk oranı için eşik solunda gözlem yok.")
    return right / left, left, right


def placebo_estimates(
    outcome: ArrayLike,
    running: ArrayLike,
    cutoffs: ArrayLike,
    *,
    bandwidth: float,
    kernel: str = "Üçgensel",
) -> tuple[RDDEstimate, ...]:
    values = np.asarray(cutoffs, dtype=float).reshape(-1)
    return tuple(
        fit_sharp_rdd(
            outcome,
            running,
            cutoff=float(cutoff),
            bandwidth=bandwidth,
            degree=1,
            kernel=kernel,
        )
        for cutoff in values
    )
