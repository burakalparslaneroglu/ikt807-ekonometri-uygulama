"""Bootstrap örnekleme indeksleri ve yeniden örnekleme birimi araçları."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


IntArray = NDArray[np.int64]


def draw_resample_indices(
    nobs: int,
    *,
    seed: int,
    groups: ArrayLike | None = None,
) -> IntArray:
    if nobs < 2:
        raise ValueError("Yeniden örnekleme için en az iki gözlem gerekir.")
    rng = np.random.default_rng(seed)
    if groups is None:
        return rng.integers(0, nobs, nobs, dtype=np.int64)

    labels = np.asarray(groups).reshape(-1)
    if labels.size != nobs:
        raise ValueError("Küme kimlikleri gözlem sayısıyla eşleşmelidir.")
    unique = np.unique(labels)
    if unique.size < 2:
        raise ValueError("Küme bootstrap için en az iki küme gerekir.")
    sampled = rng.choice(unique, size=unique.size, replace=True)
    pieces = [np.flatnonzero(labels == label) for label in sampled]
    return np.concatenate(pieces).astype(np.int64)


def resample_frequencies(indices: ArrayLike, nobs: int) -> IntArray:
    values = np.asarray(indices, dtype=int).reshape(-1)
    if values.size == 0 or (values < 0).any() or (values >= nobs).any():
        raise ValueError("Örnekleme indeksleri geçerli gözlem aralığında olmalıdır.")
    return np.bincount(values, minlength=nobs).astype(np.int64)
