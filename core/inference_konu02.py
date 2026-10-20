"""Deterministic reference calculations for notes §§2.9 and 2.12.3.

These functions do not fit a model or choose its covariance estimator. The normal
test assumes a valid N(0,1) reference; the exact F calibration assumes the normal,
homoskedastic linear model. Data-based diagnostics remain in the lab core.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from scipy import stats

Alternative = Literal["two-sided", "greater", "less"]


@dataclass(frozen=True)
class NormalTest:
    observed: float
    alternative: Alternative
    alpha: float
    p_value: float
    critical_values: tuple[float, ...]
    reject: bool


def normal_test(observed: float, alternative: Alternative, alpha: float = .05) -> NormalTest:
    if not np.isfinite(observed):
        raise ValueError("Gözlenen istatistik sonlu olmalıdır.")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Anlamlılık düzeyi 0 ile 1 arasında olmalıdır.")
    if alternative == "two-sided":
        p_value = 2 * stats.norm.sf(abs(observed))
        upper = stats.norm.isf(alpha / 2)
        critical = (-upper, upper)
    elif alternative == "greater":
        p_value = stats.norm.sf(observed)
        critical = (stats.norm.isf(alpha),)
    elif alternative == "less":
        p_value = stats.norm.cdf(observed)
        critical = (stats.norm.ppf(alpha),)
    else:
        raise ValueError("Alternatif iki taraflı, sağ veya sol yönlü olmalıdır.")
    return NormalTest(float(observed), alternative, float(alpha), float(p_value),
                      tuple(float(c) for c in critical), bool(p_value < alpha))


def normal_reference_curve(result: NormalTest) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Include cut points exactly; the shaded density represents the p event."""
    limit = max(4.2, abs(result.observed) + .4)
    x = np.sort(np.unique(np.r_[np.linspace(-limit, limit, 1201),
                               result.observed, -result.observed, result.critical_values]))
    density = stats.norm.pdf(x)
    if result.alternative == "two-sided":
        mask = np.abs(x) >= abs(result.observed)
    elif result.alternative == "greater":
        mask = x >= result.observed
    else:
        mask = x <= result.observed
    return x, density, np.where(mask, density, 0.0)


@dataclass(frozen=True)
class ClassicalF:
    statistic: float
    numerator_df: int
    denominator_df: int
    p_value: float
    critical_value: float
    reject: bool


def classical_f_test(sse_unrestricted: float, sse_restricted: float,
                     n: int, k: int, q: int, alpha: float = .05) -> ClassicalF:
    """Nested-model SSE formula; k is the unrestricted coefficient count."""
    if any(not isinstance(v, Integral) or isinstance(v, bool) for v in (n, k, q)):
        raise ValueError("Gözlem, katsayı ve kısıt sayıları tam sayı olmalıdır.")
    if not 1 <= q <= k < n:
        raise ValueError("1 ≤ q ≤ k < n koşulu gerekir.")
    if not (np.isfinite(sse_unrestricted) and np.isfinite(sse_restricted)):
        raise ValueError("Artık kareler toplamları sonlu olmalıdır.")
    if not 0 < sse_unrestricted <= sse_restricted:
        raise ValueError("0 < SSE_u ≤ SSE_r koşulu gerekir.")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Anlamlılık düzeyi 0 ile 1 arasında olmalıdır.")
    value = ((sse_restricted - sse_unrestricted) / q) / (sse_unrestricted / (n-k))
    p_value = stats.f.sf(value, q, n-k)
    return ClassicalF(float(value), int(q), int(n-k), float(p_value),
                      float(stats.f.isf(alpha, q, n-k)), bool(p_value < alpha))


def vif_summary(r_squared: float) -> tuple[float, float]:
    """Variance and homoskedastic-SE factors, holding sigma² and SST fixed."""
    if not np.isfinite(r_squared) or not 0 <= r_squared < 1:
        raise ValueError("Yardımcı R² için 0 ≤ R² < 1 gerekir; R²=1 tam bağlantıdır.")
    value = 1 / (1-r_squared)
    return float(value), float(np.sqrt(value))
