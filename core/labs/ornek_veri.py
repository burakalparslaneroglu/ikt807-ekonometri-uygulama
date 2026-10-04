"""Kurgusal örnek veri: "Kendi verini yükle" seçeneğinin örnek dosyası.

Veri gerçek kişilere ait değildir; aşağıdaki veri üretim süreçleriyle (DGP) sabit bir tohumla üretilir ve bellekte Excel
dosyasına yazılır (depoda veri dosyası tutulmaz). Amaç öğrencinin dosya biçimini görmesi ve seçenekleri denemesidir;
sayılar herhangi bir anakütleyi temsil etmez.

Ücret verisi (Konu 1–2; ``n`` = 400 çalışan, tohum 807):

* eğitim yılı ∈ {5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 20}; yaş ~ Tam sayı[22, 60]; kariyer arası ~ Tam sayı[0, 3];
* deneyim = max(yaş − eğitim − 6 − kariyer arası, 0); kıdem = ⌊deneyim · U(0, 1)⌋;
* kadın ~ Bernoulli(0,45); evli ~ Bernoulli(0,55);
* log ücret = 3,6 + 0,085·eğitim + 0,040·deneyim − 0,00065·deneyim² − 0,15·kadın + 0,01·kıdem + 0,04·evli + u,
  u ~ N(0, σᵢ²), σᵢ = 0,25 + 0,015·eğitim (heteroskedastik);
* saatlik ücret = exp(log ücret), iki ondalığa yuvarlanır (TL).

Okul deneyi (Konu 3; 40 okul, tohum 807):

* her okulda Tam sayı[18, 32] öğrenci; okulların 20'si rastgele seçilir ve programa alınır (Program = Var);
* okul etkileri a_g ~ N(0, 0,3²) (başlangıç puanında) ve s_g ~ N(0, 3²) (son sınav puanında);
* başlangıç puanı = a_g + N(0, 1); kız öğrenci ~ Bernoulli(0,5); yaş = 9 + Tam sayı[0, 2], öğrencilerin %7'sinde boş;
* son sınav puanı = 50 + 6·başlangıç + 1·kız + 2·program + s_g + N(0, 8²), bir ondalığa yuvarlanır: programın
  gerçek etkisi 2 puandır;
* düşük başlangıç grubu: program okullarında başlangıç puanı okul medyanının altındaysa Evet, değilse Hayır; öteki
  okullarda boş (gruplar yalnız program okullarında oluşur).

Araç değişkeni verisi (Konu 4; ``n`` = 1.500 çalışan, tohum 807):

* yetenek a ~ N(0, 1) (gözlenmez); üniversiteye yakınlık ~ Bernoulli(0,45); kentte yaşıyor ~ Bernoulli(0,6); kadın ~
  Bernoulli(0,45); deneyim ~ Tam sayı[0, 25];
* eğitim yılı = 11 + 1,0·yakınlık + 1,2·a + 0,4·kent + N(0, 1,5²), tam sayıya yuvarlanıp [5, 20] aralığına kırpılır;
* log ücret = 3,2 + 0,08·eğitim + 0,03·deneyim − 0,0005·deneyim² − 0,15·kadın + 0,10·kent + 0,15·a + N(0, 0,35²);
  saatlik ücret = exp(log ücret), iki ondalığa yuvarlanır (TL). Eğitimin log ücrete gerçek etkisi 0,08'dir; yetenek
  hem eğitimi hem ücreti etkilediği için OLS yukarı yanlıdır, yakınlık ücreti yalnız eğitim üzerinden etkiler.
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


def deney_verisi(schools: int = 40, seed: int = SEED) -> pd.DataFrame:
    """Kurgusal okul deneyi (yukarıdaki DGP): okul düzeyinde rastgele atanan program."""

    rng = np.random.default_rng(seed)
    sizes = rng.integers(18, 33, size=schools)
    treated = np.zeros(schools, dtype=bool)
    treated[rng.permutation(schools)[: schools // 2]] = True
    school_base = rng.normal(0.0, 0.3, size=schools)
    school_end = rng.normal(0.0, 3.0, size=schools)
    school = np.repeat(np.arange(101, 101 + schools), sizes)
    index = np.repeat(np.arange(schools), sizes)
    n = int(sizes.sum())
    baseline = school_base[index] + rng.normal(0.0, 1.0, size=n)
    girl = (rng.random(n) < 0.5).astype(int)
    age = (9 + rng.integers(0, 3, size=n)).astype(float)
    age[rng.random(n) < 0.07] = np.nan
    program = treated[index]
    final = 50 + 6 * baseline + 1.0 * girl + 2.0 * program + school_end[index] + rng.normal(0.0, 8.0, size=n)
    medians = pd.Series(baseline).groupby(index).transform("median").to_numpy()
    low = np.where(program, np.where(baseline < medians, "Evet", "Hayır"), None)
    return pd.DataFrame({
        "Son sınav puanı": np.round(final, 1),
        "Program": np.where(program, "Var", "Yok"),
        "Okul": school,
        "Başlangıç puanı": np.round(baseline, 2),
        "Kız öğrenci (1/0)": girl,
        "Yaş": age,
        "Düşük başlangıç grubu": low,
    })


def arac_verisi(n: int = 1500, seed: int = SEED) -> pd.DataFrame:
    """Kurgusal araç değişkeni verisi (yukarıdaki DGP): koleje yakınlık eğitimi kaydırır."""

    rng = np.random.default_rng(seed)
    ability = rng.normal(0.0, 1.0, size=n)
    near = (rng.random(n) < 0.45).astype(int)
    city = (rng.random(n) < 0.6).astype(int)
    woman = (rng.random(n) < 0.45).astype(int)
    experience = rng.integers(0, 26, size=n)
    education = np.clip(np.round(11 + 1.0 * near + 1.2 * ability + 0.4 * city + rng.normal(0.0, 1.5, size=n)), 5, 20)
    log_wage = (3.2 + 0.08 * education + 0.03 * experience - 0.0005 * experience ** 2 - 0.15 * woman + 0.10 * city
                + 0.15 * ability + rng.normal(0.0, 0.35, size=n))
    return pd.DataFrame({
        "Saatlik ücret (TL)": np.round(np.exp(log_wage), 2),
        "Eğitim yılı": education.astype(int),
        "Üniversiteye yakınlık (1/0)": near,
        "Deneyim (yıl)": experience.astype(int),
        "Kadın (1/0)": woman,
        "Kentte yaşıyor (1/0)": city,
    })
