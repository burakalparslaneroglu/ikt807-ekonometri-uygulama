"""Kurgusal örnek veri: "Kendi verini yükle" seçeneğinin örnek dosyası.

Veri gerçek kişilere ait değildir; aşağıdaki veri üretim süreciyle (DGP) sabit bir tohumla üretilir ve bellekte Excel
dosyasına yazılır (depoda veri dosyası tutulmaz). Amaç öğrencinin dosya biçimini görmesi ve seçenekleri denemesidir;
sayılar herhangi bir anakütleyi temsil etmez.

DGP (``n`` = 400 çalışan, tohum 807):

* eğitim yılı ∈ {5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 20}; yaş ~ Tam sayı[22, 60]; kariyer arası ~ Tam sayı[0, 3];
* deneyim = max(yaş − eğitim − 6 − kariyer arası, 0); kıdem = ⌊deneyim · U(0, 1)⌋;
* kadın ~ Bernoulli(0,45); evli ~ Bernoulli(0,55);
* log ücret = 3,6 + 0,085·eğitim + 0,040·deneyim − 0,00065·deneyim² − 0,15·kadın + 0,01·kıdem + 0,04·evli + u,
  u ~ N(0, σᵢ²), σᵢ = 0,25 + 0,015·eğitim (heteroskedastik);
* saatlik ücret = exp(log ücret), iki ondalığa yuvarlanır (TL).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 807
EDUCATION_LEVELS = (5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 20)
EDUCATION_SHARES = (0.04, 0.12, 0.05, 0.26, 0.05, 0.06, 0.04, 0.22, 0.05, 0.07, 0.04)


def ucret_verisi(n: int = 400, seed: int = SEED) -> pd.DataFrame:
    """Kurgusal çalışan verisi (yukarıdaki DGP); sütun adları Excel'deki gibi Türkçedir."""

    rng = np.random.default_rng(seed)
    egitim = rng.choice(np.array(EDUCATION_LEVELS), size=n, p=np.array(EDUCATION_SHARES))
    yas = rng.integers(22, 61, size=n)
    ara = rng.integers(0, 4, size=n)
    deneyim = np.maximum(yas - egitim - 6 - ara, 0)
    kidem = np.floor(deneyim * rng.random(n)).astype(int)
    kadin = rng.random(n) < 0.45
    evli = (rng.random(n) < 0.55).astype(int)
    sigma = 0.25 + 0.015 * egitim
    log_ucret = (3.6 + 0.085 * egitim + 0.040 * deneyim - 0.00065 * deneyim ** 2 - 0.15 * kadin + 0.01 * kidem
                 + 0.04 * evli + rng.normal(0.0, 1.0, size=n) * sigma)
    return pd.DataFrame({
        "Saatlik ücret (TL)": np.round(np.exp(log_ucret), 2),
        "Eğitim yılı": egitim.astype(int),
        "Deneyim (yıl)": deneyim.astype(int),
        "Cinsiyet": np.where(kadin, "Kadın", "Erkek"),
        "Evli (1/0)": evli,
        "Kıdem (yıl)": kidem,
    })
