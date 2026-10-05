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

import hashlib
import threading
from collections import OrderedDict

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


RELIABLE = 1e-6
"""Yerel doğrusal tahminin sayısal olarak güvenilir sayıldığı en küçük göreli belirleyici ρ = (S₀S₂ − S₁²)/(S₀S₂).

ρ ∈ [0, 1] (Cauchy–Schwarz): ağırlıklı yerel tasarımın tekilliğe uzaklığıdır. Formüldeki iki farkın yuvarlama hatası
yaklaşık ε·max|Y|/ρ'dur (ε ≈ 2,2·10⁻¹⁶); ρ ≥ 10⁻⁶ iken tahminin hatası 10⁻¹⁰·max|Y| düzeyinde kalır ve Python, R ve
Stata aynı sonucu verir. Bir noktanın çevresinde ağırlığı anlamlı tek bir farklı X değeri kalırsa (ör. küçük h ile
uçtaki seyrek bir değer ya da değerler arasındaki büyük bir boşluk) ρ sıfıra iner: formül 0/0 ya da yuvarlama gürültüsü
üretir ve diller farklı sayılar (R'de grafik hatası) verebilir."""


def _determinant(s0: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """ρ = (S₀S₂ − S₁²)/(S₀S₂); tanımsızsa (ağırlıkların tamamı ya da ikinci momenti sıfır) 0."""

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        product = s0 * s2
        value = (product - s1 * s1) / product
    return np.where(np.isfinite(value), value, 0.0)


def local_conditioning(x, points, bandwidths) -> np.ndarray:
    """Her h için noktalardaki en küçük ρ (bütün veriyle yerel doğrusal tahmin, ör. grafik eğrisi); bkz. ``RELIABLE``."""

    x = np.asarray(x, dtype=float)
    points = np.atleast_1d(np.asarray(points, dtype=float))
    out = []
    for h in np.atleast_1d(np.asarray(bandwidths, dtype=float)):
        smallest = np.inf
        for start in range(0, len(points), _BLOCK):
            d = x[None, :] - points[start:start + _BLOCK, None]
            w = np.exp(-0.5 * (d / h) ** 2)
            wd = w * d
            smallest = min(smallest, float(_determinant(w.sum(axis=1), wd.sum(axis=1), (wd * d).sum(axis=1)).min()))
        out.append(smallest)
    return np.array(out)


_CV_CACHE: OrderedDict[str, dict[float, tuple[float, ...]]] = OrderedDict()
_CV_LOCK = threading.Lock()
_CV_SIZE = 16
"""Son hesaplanan CV değerlerinin önbelleği (veri başına, h başına): kendi verinde bant genişliği tanım kurulurken
seçilir; aynı ölçüt laboratuvar çalışırken ya da ızgaranın bir parçası için yeniden hesaplanmasın (n×n ağırlık matrisi;
birkaç bin gözlemde saniyeler sürer)."""


def _cache_key(x: np.ndarray, y: np.ndarray, codes: np.ndarray | None) -> str:
    digest = hashlib.sha1()
    parts = (x, y) if codes is None else (x, y, codes)
    for part in parts:
        data = np.ascontiguousarray(part).tobytes()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest() + ("k" if codes is not None else "")


def _cv_rows(x, y, bandwidths, cluster) -> tuple[list[float], list[tuple[float, ...]]]:
    """Izgaradaki her h için (CV, ρ) ya da küme varsa (CV, ρ, küme-silmeli CV, küme-silmeli ρ); eksikler hesaplanır."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = [round(float(h), 10) for h in np.atleast_1d(np.asarray(bandwidths, dtype=float))]
    codes = None if cluster is None else pd.factorize(np.asarray(cluster))[0].astype(np.int64)
    key = _cache_key(x, y, codes)
    with _CV_LOCK:
        rows = _CV_CACHE.get(key, {})
        missing = [h for h in dict.fromkeys(grid) if h not in rows]
    if missing:
        computed = _cv_compute(x, y, np.asarray(missing), codes)
    with _CV_LOCK:
        rows = _CV_CACHE.setdefault(key, rows)
        if missing:
            rows.update(computed)
        _CV_CACHE.move_to_end(key)
        while len(_CV_CACHE) > _CV_SIZE:
            _CV_CACHE.popitem(last=False)
        return grid, [rows[h] for h in grid]


def cv_curve(x, y, bandwidths, cluster=None) -> pd.DataFrame:
    """Her h için CV(h); ``cluster`` verilirse küme-silmeli CV de hesaplanır.

    Dönüş tablosu: indeks h, sütunlar ``cv`` ve (küme varsa) ``cv_kume``. Aynı veriyle yeniden çağrıldığında daha önce
    hesaplanan h'ler önbellekten gelir.
    """

    grid, rows = _cv_rows(x, y, bandwidths, cluster)
    table = pd.DataFrame({"cv": [row[0] for row in rows]}, index=pd.Index(grid, dtype=float, name="h"))
    if cluster is not None:
        table["cv_kume"] = [row[2] for row in rows]
    return table


def cv_conditioning(x, y, bandwidths, cluster=None) -> pd.DataFrame:
    """Her h için CV'deki dışarıda bırakılan tahminlerin en küçük ρ değeri (``RELIABLE``): sütunlar ``kosul`` ve (küme
    varsa) ``kosul_kume``. CV ile aynı hesapta bulunur ve önbelleği paylaşır."""

    grid, rows = _cv_rows(x, y, bandwidths, cluster)
    table = pd.DataFrame({"kosul": [row[1] for row in rows]}, index=pd.Index(grid, dtype=float, name="h"))
    if cluster is not None:
        table["kosul_kume"] = [row[3] for row in rows]
    return table


def _cv_compute(x: np.ndarray, y: np.ndarray, bandwidths: np.ndarray,
                codes: np.ndarray | None) -> dict[float, tuple[float, ...]]:
    d = x[None, :] - x[:, None]
    d2 = d * d
    same = None if codes is None else codes[:, None] == codes[None, :]
    rows = {}
    for h in np.asarray(bandwidths, dtype=float):
        w = np.exp(-0.5 * d2 / (h * h))
        np.fill_diagonal(w, 0.0)
        row = _cv_error(w, d, y)
        if same is not None:
            w[same] = 0.0
            row += _cv_error(w, d, y)
        rows[round(float(h), 10)] = row
    return rows


def _cv_error(w: np.ndarray, d: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(CV, en küçük ρ): dışarıda bırakılan tahmin hatalarının kareler ortalaması ve tahminlerin koşulu."""

    wd = w * d
    s0, s1, s2 = w.sum(axis=1), wd.sum(axis=1), (wd * d).sum(axis=1)
    t0, t1 = w @ y, wd @ y
    with np.errstate(divide="ignore", invalid="ignore"):
        fitted = (s2 * t0 - s1 * t1) / (s0 * s2 - s1**2)
    return float(np.mean((y - fitted) ** 2)), float(_determinant(s0, s1, s2).min())


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
