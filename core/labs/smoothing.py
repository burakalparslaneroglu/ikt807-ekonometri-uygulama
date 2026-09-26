"""Parametrik olmayan regresyon: Gauss çekirdekli yerel doğrusal (ve yerel sabit) tahmin.

Çekirdek K(u) = φ(u), u = (Xᵢ − x)/h; h çekirdeğin standart sapmasıdır. Normalleştirme
sabiti oranlarda sadeleştiği için ağırlık exp(−u²/2) alınır. x noktasında

    m̂(x) = (S₂T₀ − S₁T₁) / (S₀S₂ − S₁²),  S_k = Σ wᵢ(Xᵢ − x)^k,  T_k = Σ wᵢ(Xᵢ − x)^k Yᵢ,

yani ağırlıklı en küçük kareler probleminin sabit terimi (yerel doğrusal). Yerel sabit
(Nadaraya–Watson) tahmin T₀/S₀'dır. Aynı formüller üretilen Python, R ve Stata (Mata)
kodunda kullanılır.

Çapraz doğrulama (CV) ölçütü dışarıda bırakılan tahmin hatalarının kareler ortalamasıdır.
Birini-dışarıda-bırak CV'de gözlem kendi tahmininden çıkarılır; küme-silmeli CV'de gözlemin
bütün kümesi (ör. okulu) çıkarılır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_BLOCK = 512
"""Değerlendirme noktaları bu büyüklükte bloklar hâlinde işlenir (bellek sınırı)."""


def _moments(x: np.ndarray, y: np.ndarray, points: np.ndarray, h: float):
    d = x[None, :] - points[:, None]
    w = np.exp(-0.5 * (d / h) ** 2)
    wd = w * d
    return w.sum(axis=1), wd.sum(axis=1), (wd * d).sum(axis=1), w @ y, wd @ y


def local_fit(x, y, points, h: float, degree: int = 1) -> np.ndarray:
    """Gauss çekirdekli yerel doğrusal (``degree=1``) veya yerel sabit (``degree=0``) tahmin."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    points = np.atleast_1d(np.asarray(points, dtype=float))
    if degree not in (0, 1):
        raise ValueError("Yalnız yerel sabit (0) ve yerel doğrusal (1) desteklenir.")
    out = np.empty(len(points))
    for start in range(0, len(points), _BLOCK):
        block = points[start:start + _BLOCK]
        s0, s1, s2, t0, t1 = _moments(x, y, block, h)
        out[start:start + _BLOCK] = t0 / s0 if degree == 0 else (s2 * t0 - s1 * t1) / (s0 * s2 - s1**2)
    return out


def local_linear(x, y, points, h: float) -> np.ndarray:
    return local_fit(x, y, points, h, degree=1)


def local_residuals(x, v, h: float) -> np.ndarray:
    """v − m̂(x): v'nin x üzerindeki yerel doğrusal tahminden sapması, örneklem noktalarında."""

    v = np.asarray(v, dtype=float)
    return v - local_linear(x, v, x, h)


def bandwidth_grid(low: float, high: float, step: float) -> np.ndarray:
    count = int(round((high - low) / step)) + 1
    return np.round(low + step * np.arange(count), 10)


def cv_curve(x, y, bandwidths, cluster=None) -> pd.DataFrame:
    """Her h için CV(h); ``cluster`` verilirse küme-silmeli CV de hesaplanır.

    Dönüş tablosu: indeks h, sütunlar ``cv`` ve (küme varsa) ``cv_kume``.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x[None, :] - x[:, None]
    d2 = d * d
    same = None
    if cluster is not None:
        groups = pd.factorize(np.asarray(cluster))[0]
        same = groups[:, None] == groups[None, :]
    loo, clustered = [], []
    for h in np.asarray(bandwidths, dtype=float):
        w = np.exp(-0.5 * d2 / (h * h))
        np.fill_diagonal(w, 0.0)
        loo.append(_cv_error(w, d, y))
        if same is not None:
            w[same] = 0.0
            clustered.append(_cv_error(w, d, y))
    table = pd.DataFrame({"cv": loo}, index=pd.Index(np.asarray(bandwidths, dtype=float), name="h"))
    if same is not None:
        table["cv_kume"] = clustered
    return table


def _cv_error(w: np.ndarray, d: np.ndarray, y: np.ndarray) -> float:
    wd = w * d
    s0, s1, s2 = w.sum(axis=1), wd.sum(axis=1), (wd * d).sum(axis=1)
    t0, t1 = w @ y, wd @ y
    fitted = (s2 * t0 - s1 * t1) / (s0 * s2 - s1**2)
    return float(np.mean((y - fitted) ** 2))


def binned_means(x, y, bins: int) -> pd.DataFrame:
    """Eşit genişlikli aralıklarda x ve y ortalamaları (grafikte verinin özeti)."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    low, high = float(np.min(x)), float(np.max(x))
    width = (high - low) / bins
    index = np.minimum(np.floor((x - low) / width), bins - 1).astype(int)
    frame = pd.DataFrame({"aralik": index, "x": x, "y": y})
    grouped = frame.groupby("aralik").agg(x=("x", "mean"), y=("y", "mean"), n=("y", "size"))
    return grouped.reset_index(drop=True)
