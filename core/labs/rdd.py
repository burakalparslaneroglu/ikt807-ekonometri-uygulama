"""Regresyon süreksizliği (RDD): eşikte yerel doğrusal tahmin ve eşik çevresi eğrileri.

Hansen (Econometrics, Bölüm 19 ve 21) çekirdekleri birim varyansa ölçekler: bant genişliği h
çekirdeğin standart sapmasıdır. Üçgen çekirdek K(u) = (1 − |u|/√6)/√6 (|u| ≤ √6) olduğundan
h bant genişliği eşikten ±h√6 uzaklığa kadar pozitif ağırlık verir; dikdörtgen (düzgün)
çekirdekte pencere ±h√3'tür. ``scale="window"`` seçeneğinde h doğrudan pencerenin yarı
genişliğidir (standart, ölçeklenmemiş çekirdek).

Eşikteki sıçrama, pencere içindeki gözlemlerde Y'nin D = 1{X ≥ c}, R = X − c ve D·R
üzerine çekirdek ağırlıklı en küçük kareler regresyonundaki D katsayısıdır; bu, eşiğin iki
yanında ayrı yerel doğrusal regresyonların eşikteki farkına eşittir. Standart hata ağırlıklı
regresyonun HC1 sandviçidir: Python ``WLS(...).fit(cov_type="HC1")``, R ``vcovHC(type = "HC1")``,
Stata ``regress ... [aw = w], vce(robust)`` ile aynı sayı.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scipy import stats

INTERCEPT = "Intercept"
TERMS = (INTERCEPT, "D", "R", "DR")
"""Terim adları (statsmodels adlandırmasıyla): sabit, eşik göstergesi, merkezlenmiş eşik değişkeni, etkileşim."""
KERNELS = ("triangular", "rectangular")
SCALES = ("hansen", "window")
CI_MULTIPLIER = 1.96


def window(bandwidth: float, kernel: str = "triangular", scale: str = "hansen") -> float:
    """Pozitif ağırlık alan gözlemlerin eşiğe en büyük uzaklığı."""

    if kernel not in KERNELS or scale not in SCALES:
        raise ValueError(f"Desteklenmeyen çekirdek/ölçek: {kernel}, {scale}")
    if scale == "window":
        return float(bandwidth)
    return float(bandwidth) * float(np.sqrt(6.0 if kernel == "triangular" else 3.0))


def kernel_weights(r, bandwidth: float, kernel: str = "triangular", scale: str = "hansen") -> np.ndarray:
    """Göreli çekirdek ağırlıkları (sabit çarpanlar ağırlıklı EKK'yi değiştirmez)."""

    distance = np.abs(np.asarray(r, dtype=float))
    width = window(bandwidth, kernel, scale)
    if kernel == "triangular":
        return np.maximum(1.0 - distance / width, 0.0)
    return (distance <= width).astype(float)


@dataclass
class RDDFit:
    """Ağırlıklı EKK sonucu; statsmodels sonuçlarıyla aynı adlı alanlar (params, bse, nobs)."""

    params: pd.Series
    bse: pd.Series
    nobs: int
    covariance: pd.DataFrame
    cutoff: float
    bandwidth: float
    kernel: str
    scale: str
    n_left: int
    n_right: int

    def cov_params(self) -> pd.DataFrame:
        return self.covariance

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.bse

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series(2.0 * stats.norm.sf(np.abs(self.tvalues)), index=self.params.index)

    def conf_int(self) -> pd.DataFrame:
        return pd.DataFrame({0: self.params - CI_MULTIPLIER * self.bse, 1: self.params + CI_MULTIPLIER * self.bse})

    @property
    def window(self) -> float:
        return window(self.bandwidth, self.kernel, self.scale)

    @property
    def jump(self) -> float:
        return float(self.params["D"])

    @property
    def left(self) -> float:
        return float(self.params[INTERCEPT])

    @property
    def right(self) -> float:
        return float(self.params[INTERCEPT] + self.params["D"])


def weighted_hc1(design: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ağırlıklı EKK katsayıları ve HC1 kovaryansı: n/(n−k)·A⁻¹(Σ wᵢ²eᵢ²zᵢzᵢ')A⁻¹, A = Σ wᵢzᵢzᵢ'."""

    n, k = design.shape
    if n <= k:
        raise ValueError("Pencerede tahmin için yeterli gözlem yok; bant genişliğini büyütün.")
    weighted = design * weights[:, None]
    a_inverse = np.linalg.inv(design.T @ weighted)
    beta = a_inverse @ (weighted.T @ y)
    residual = y - design @ beta
    scores = design * (weights * residual)[:, None]
    covariance = n / (n - k) * a_inverse @ (scores.T @ scores) @ a_inverse
    return beta, covariance


def rdd_fit(x, y, cutoff: float, bandwidth: float, kernel: str = "triangular", scale: str = "hansen") -> RDDFit:
    """Eşikte yerel doğrusal sıçrama: Y ~ 1 + D + R + D·R, çekirdek ağırlıklı, pencere içindeki gözlemler."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    complete = np.isfinite(x) & np.isfinite(y)
    r = x[complete] - cutoff
    outcome = y[complete]
    weights = kernel_weights(r, bandwidth, kernel, scale)
    inside = weights > 0
    r, outcome, weights = r[inside], outcome[inside], weights[inside]
    d = (r >= 0).astype(float)
    design = np.column_stack([np.ones_like(r), d, r, d * r])
    if d.sum() < 2 or (1 - d).sum() < 2:
        raise ValueError("Eşiğin iki yanında da en az iki gözlem gerekir; bant genişliğini büyütün.")
    beta, covariance = weighted_hc1(design, outcome, weights)
    return RDDFit(
        params=pd.Series(beta, index=TERMS),
        bse=pd.Series(np.sqrt(np.diag(covariance)), index=TERMS),
        nobs=int(len(outcome)),
        covariance=pd.DataFrame(covariance, index=TERMS, columns=TERMS),
        cutoff=float(cutoff),
        bandwidth=float(bandwidth),
        kernel=kernel,
        scale=scale,
        n_left=int((d == 0).sum()),
        n_right=int((d == 1).sum()),
    )


def rdd_table(fits: dict[float, RDDFit]) -> pd.DataFrame:
    """Bant genişliği duyarlılık tablosu: n, tahmin, SH ve normal yaklaşımla %95 güven aralığı."""

    rows = {
        h: {
            "n": float(fit.nobs),
            "tahmin": fit.jump,
            "sh": float(fit.bse["D"]),
            "alt": fit.jump - CI_MULTIPLIER * float(fit.bse["D"]),
            "ust": fit.jump + CI_MULTIPLIER * float(fit.bse["D"]),
        }
        for h, fit in fits.items()
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "h"
    return table


def local_linear_side(x, y, points, bandwidth: float, kernel: str = "triangular", scale: str = "hansen"):
    """Tek taraflı yerel doğrusal tahmin ve HC1 standart hatası, her değerlendirme noktasında.

    Her x₀ için pencere içindeki gözlemlerde Y ~ 1 + (X − x₀) ağırlıklı EKK'sinin sabit terimi;
    SH, o yerel regresyonun HC1 sandviçidir (n/(n−2) çarpanıyla). Kapalı biçimde, bütün
    noktalar için birlikte hesaplanır.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    points = np.asarray(points, dtype=float)
    d = x[None, :] - points[:, None]
    w = kernel_weights(d, bandwidth, kernel, scale)
    count = (w > 0).sum(axis=1)
    s0, s1, s2 = w.sum(axis=1), (w * d).sum(axis=1), (w * d * d).sum(axis=1)
    t0, t1 = w @ y, (w * d) @ y
    determinant = s0 * s2 - s1**2
    with np.errstate(invalid="ignore", divide="ignore"):
        intercept = (s2 * t0 - s1 * t1) / determinant
        slope = (s0 * t1 - s1 * t0) / determinant
        residual = y[None, :] - intercept[:, None] - slope[:, None] * d
        u = (w * residual) ** 2
        m00, m01, m11 = u.sum(axis=1), (u * d).sum(axis=1), (u * d * d).sum(axis=1)
        # (A⁻¹ M A⁻¹)₀₀, A⁻¹ = [[s2, −s1], [−s1, s0]] / det
        variance = (s2**2 * m00 - 2 * s1 * s2 * m01 + s1**2 * m11) / determinant**2
        variance = variance * count / (count - 2)
    valid = count > 2
    intercept = np.where(valid, intercept, np.nan)
    standard_error = np.where(valid, np.sqrt(variance), np.nan)
    return intercept, standard_error


def rdd_curve(x, y, cutoff: float, bandwidth: float, left_points, right_points,
              kernel: str = "triangular", scale: str = "hansen") -> pd.DataFrame:
    """Eşiğin iki yanında ayrı yerel doğrusal eğriler ve noktasal %95 güven bantları."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    complete = np.isfinite(x) & np.isfinite(y)
    x, y = x[complete], y[complete]
    parts = []
    for side, points, mask in (("sol", left_points, x < cutoff), ("sag", right_points, x >= cutoff)):
        estimate, standard_error = local_linear_side(x[mask], y[mask], points, bandwidth, kernel, scale)
        parts.append(pd.DataFrame({
            "x": np.asarray(points, dtype=float),
            "tahmin": estimate,
            "alt": estimate - CI_MULTIPLIER * standard_error,
            "ust": estimate + CI_MULTIPLIER * standard_error,
            "taraf": side,
        }))
    return pd.concat(parts, ignore_index=True)
