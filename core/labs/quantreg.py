"""Kantil regresyon: kesin doğrusal programlama çözümü ve Hendricks–Koenker kovaryansı.

Kantil regresyon tahmin edicisi bir doğrusal programlama probleminin çözümüdür. Burada
Koenker'in dual problemi HiGHS ile çözülür; çözüm tekse sonuç R ``quantreg::rq`` (Barrodale–Roberts)
ve Stata ``qreg`` ile aynı köşe çözümüdür. statsmodels ``QuantReg`` yinelemeli yeniden
ağırlıklandırma (IRLS) kullanır ve kesin çözüme yalnız yaklaşır: büyük örneklemde katsayılar
beşinci ondalıkta farklılaşabilir.

Standart hatalar Hendricks–Koenker (1992) sandviçidir: R ``summary.rq(se = "nid")``
varsayılanı (n > 1000). Koşullu yoğunluk her gözlem için τ ± h kantil doğrularının
farkından tahmin edilir; h Hall–Sheather bant genişliğidir.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog

INTERCEPT = "Intercept"
EPS = float(np.finfo(float).eps ** 0.5)
"""R quantreg: ``.Machine$double.eps^(1/2)``; yoğunluk paydasından çıkarılan küçük sabit."""
VCOV_TYPES = ("none", "nid")

_HIGHS_LOCK = threading.Lock()
"""HiGHS aynı süreçte eşzamanlı çağrılmaz. Çözücünün iş zamanlayıcısı süreç genelindedir; Windows'ta
eşzamanlı çağrılar erişim ihlaline (access violation) yol açabiliyor. Streamlit her oturumu ayrı bir
iş parçacığında çalıştırdığı için kilit, farklı oturumlardaki çözümleri de sıraya koyar."""


def rq_fit(x: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Kesin kantil regresyon çözümü.

    Dual problem: max y'a, X'a = (1 − τ)X'1, 0 ≤ a ≤ 1. Eşitlik kısıtlarının gölge fiyatları
    (işaret değiştirilerek) primal çözüm β'dır.
    """

    x = np.asarray(x, dtype=float)
    with _HIGHS_LOCK:
        result = linprog(
            -np.asarray(y, dtype=float),
            A_eq=x.T,
            b_eq=(1.0 - tau) * x.sum(axis=0),
            bounds=(0.0, 1.0),
            method="highs",
        )
    if result.status != 0:
        raise RuntimeError(f"Kantil regresyon çözülemedi (τ = {tau}): {result.message}")
    return -np.asarray(result.eqlin.marginals, dtype=float)


def check_loss_sum(x: np.ndarray, y: np.ndarray, beta: np.ndarray, tau: float) -> float:
    residual = y - x @ beta
    return float(np.sum(residual * (tau - (residual < 0))))


def hall_sheather(tau: float, n: int, alpha: float = 0.05) -> float:
    """Hall–Sheather bant genişliği (τ ölçeğinde); R ``bandwidth.rq(hs = TRUE)``."""

    z = stats.norm.ppf(tau)
    density = stats.norm.pdf(z)
    return float(n ** (-1.0 / 3.0) * stats.norm.ppf(1.0 - alpha / 2.0) ** (2.0 / 3.0)
                 * (1.5 * density**2 / (2.0 * z**2 + 1.0)) ** (1.0 / 3.0))


def hk_bandwidth(tau: float, n: int) -> float:
    """Hall–Sheather bandı; τ ± h aralığı (0, 1) dışına taşarsa yarıya indirilir."""

    h = hall_sheather(tau, n)
    while tau - h < 0.0 or tau + h > 1.0:
        h /= 2.0
    return h


@dataclass
class QuantileFit:
    """Kantil regresyon sonucu; statsmodels sonuçlarıyla aynı adlı alanlar.

    ``hinv`` = (X'FX)⁻¹ ve ``xtx`` = X'X, iki kantil tahmininin ortak kovaryansı için saklanır.
    """

    params: pd.Series
    q: float
    nobs: int
    exog: pd.DataFrame
    endog: np.ndarray
    vcov: str = "none"
    hinv: np.ndarray | None = None
    xtx: np.ndarray | None = None
    bandwidth: float | None = None
    covariance: pd.DataFrame | None = None

    def cov_params(self) -> pd.DataFrame:
        if self.covariance is None:
            raise ValueError("Bu kantil regresyon için standart hata hesaplanmadı (vcov='none').")
        return self.covariance

    @property
    def bse(self) -> pd.Series:
        if self.covariance is None:
            return pd.Series(np.nan, index=self.params.index)
        return pd.Series(np.sqrt(np.diag(self.covariance.to_numpy())), index=self.params.index)

    @property
    def fittedvalues(self) -> pd.Series:
        return pd.Series(self.exog.to_numpy() @ self.params.to_numpy(), index=self.exog.index)

    @property
    def resid(self) -> pd.Series:
        return pd.Series(self.endog - self.fittedvalues.to_numpy(), index=self.exog.index)

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.bse

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series(2.0 * stats.norm.sf(np.abs(self.tvalues)), index=self.params.index)

    def conf_int(self) -> pd.DataFrame:
        return pd.DataFrame({0: self.params - 1.96 * self.bse, 1: self.params + 1.96 * self.bse})


def _design(frame: pd.DataFrame, outcome: str, regressors: tuple[str, ...]) -> tuple[pd.DataFrame, np.ndarray]:
    used = list(dict.fromkeys([outcome, *regressors]))
    block = frame[used]
    data = frame[~block.isna().any(axis=1)]
    design = data[list(regressors)].astype(float).copy()
    design.insert(0, INTERCEPT, 1.0)
    return design, data[outcome].to_numpy(dtype=float)


def quantile_regression(design: pd.DataFrame, y: np.ndarray, tau: float, vcov: str = "none") -> QuantileFit:
    """Kesin çözüm; ``vcov="nid"`` ise Hendricks–Koenker kovaryansı (R ``summary.rq(se = "nid")``)."""

    if vcov not in VCOV_TYPES:
        raise ValueError(f"Desteklenmeyen kovaryans türü: {vcov}")
    x = design.to_numpy(dtype=float)
    n = len(y)
    if vcov == "none":
        beta = rq_fit(x, y, tau)
        return QuantileFit(pd.Series(beta, index=design.columns), tau, n, design, y)
    h = hk_bandwidth(tau, n)
    beta, high, low = (rq_fit(x, y, t) for t in (tau, tau + h, tau - h))
    change = x @ (high - low)
    density = np.maximum(0.0, (2.0 * h) / (change - EPS))
    hinv = np.linalg.inv(x.T @ (density[:, None] * x))
    xtx = x.T @ x
    covariance = tau * (1.0 - tau) * hinv @ xtx @ hinv
    names = design.columns
    return QuantileFit(
        pd.Series(beta, index=names), tau, n, design, y, vcov, hinv, xtx, h,
        pd.DataFrame(covariance, index=names, columns=names),
    )


def fit_quantile(op, frame: pd.DataFrame) -> QuantileFit:
    design, y = _design(frame, op.outcome, op.regressors)
    return quantile_regression(design, y, op.q, getattr(op, "vcov", "none"))


def quantile_difference(low: QuantileFit, high: QuantileFit, term: str) -> tuple[float, float]:
    """β̂(τ₂) − β̂(τ₁) ve standart hatası; iki tahminin ortak asimptotik kovaryansıyla.

    Cov(β̂τ₁, β̂τ₂) = (min(τ₁, τ₂) − τ₁τ₂) H₁⁻¹ X'X H₂⁻¹ (Koenker 2005, Kısım 3.2.4).
    """

    for fit in (low, high):
        if fit.hinv is None:
            raise ValueError("Kantiller arası fark için iki modelde de vcov='nid' gerekir.")
    if low.nobs != high.nobs or not np.array_equal(low.xtx, high.xtx):
        raise ValueError("İki kantil regresyonu aynı örneklemde ve aynı regresörlerle tahmin edilmelidir.")
    j = list(low.params.index).index(term)
    cross = (min(low.q, high.q) - low.q * high.q) * low.hinv @ low.xtx @ high.hinv
    difference = float(high.params.iloc[j] - low.params.iloc[j])
    variance = (low.covariance.to_numpy()[j, j] + high.covariance.to_numpy()[j, j] - 2.0 * cross[j, j])
    return difference, float(np.sqrt(variance))
