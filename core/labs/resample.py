"""Bootstrap: bir OLS modelini yeniden örneklenmiş verilerde yeniden tahmin etmek.

Üç yeniden örnekleme şeması (Notlar §10.5, §10.10, §10.14):

* ``pairs``   gözlem satırları yerine koyarak çekilir: her tekrarda ``rng.integers(0, n, n)``.
* ``wild``    regresörler sabit; Y* = Xβ̂ + ê·ξ, ξ Rademacher: ``rng.choice([-1.0, 1.0], n)``.
* ``cluster`` kümeler yerine koyarak çekilir (``rng.integers(0, G, G)``); seçilen kümenin bütün
  gözlemleri birlikte gelir.

Çekiliş sırası üretilen Python koduyla aynıdır; bu yüzden Python betiği uygulamadaki sayıların
aynısını verir. R ve Stata aynı dağılımdan farklı çekiliş yapar (Monte Carlo hatası kadar fark).
Katsayılar ``numpy.linalg.lstsq`` ile, tekrar başına HC1 standart hataları (percentile-t için)
sandviç formülüyle hesaplanır.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

METHODS = ("pairs", "wild", "cluster")


@dataclass
class Replicate:
    """Bir bootstrap tekrarındaki tahmin; statsmodels sonuçlarıyla aynı adlı alanlar."""

    params: pd.Series
    bse: pd.Series


def ols_fit(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(design, y, rcond=None)[0]


def hc1_errors(design: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """HC1 standart hataları: n/(n−k)·(X'X)⁻¹(Σ eᵢ²xᵢxᵢ')(X'X)⁻¹."""

    n, k = design.shape
    inverse = np.linalg.inv(design.T @ design)
    scores = design * (y - design @ beta)[:, None]
    covariance = n / (n - k) * inverse @ (scores.T @ scores) @ inverse
    return np.sqrt(np.diag(covariance))


def replicates(design: np.ndarray, y: np.ndarray, names, reps: int, rng: np.random.Generator,
               method: str = "pairs", clusters=None, need_se: bool = False, fitted=None):
    """Bootstrap tekrarlarını sırayla üretir (``Replicate`` nesneleri).

    ``fitted``: wild bootstrap'ta orijinal modelin uyum değerleri (verilmezse EKK ile hesaplanır).
    """

    if method not in METHODS:
        raise ValueError(f"Desteklenmeyen bootstrap yöntemi: {method}")
    design = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    names = list(names)
    if method == "wild":
        fitted = design @ ols_fit(design, y) if fitted is None else np.asarray(fitted, dtype=float)
        residual = y - fitted
    if method == "cluster":
        if clusters is None:
            raise ValueError("Küme bootstrap'ı için küme değişkeni gerekir.")
        labels, codes = np.unique(np.asarray(clusters), return_inverse=True)
        members = [np.flatnonzero(codes == g) for g in range(len(labels))]
    for _ in range(reps):
        if method == "pairs":
            index = rng.integers(0, n, n)
            x_b, y_b = design[index], y[index]
        elif method == "wild":
            x_b, y_b = design, fitted + residual * rng.choice([-1.0, 1.0], n)
        else:
            chosen = rng.integers(0, len(members), len(members))
            index = np.concatenate([members[g] for g in chosen])
            x_b, y_b = design[index], y[index]
        beta = ols_fit(x_b, y_b)
        errors = hc1_errors(x_b, y_b, beta) if need_se else np.full(len(names), np.nan)
        yield Replicate(pd.Series(beta, index=names), pd.Series(errors, index=names))


def summary(values) -> dict[str, float]:
    """Bootstrap standart hatası (ddof = 1) ve yüzde 2,5 ile 97,5 yüzdelikleri (numpy varsayılanı)."""

    values = np.asarray(values, dtype=float)
    return {
        "se": float(values.std(ddof=1)),
        "lo": float(np.quantile(values, 0.025)),
        "hi": float(np.quantile(values, 0.975)),
    }
