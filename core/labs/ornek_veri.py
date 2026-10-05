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

İkili sonuç verisi (Konu 5; ``n`` = 1.200 kişi, tohum 807):

* yaş ~ Tam sayı[22, 60]; eğitim yılı ∈ {8, 11, 12, 14, 16} (olasılıklar 0,20; 0,15; 0,30; 0,15; 0,20), kişilerin %2'sinde
  boş; kadın ~ Bernoulli(0,48); il merkezinde yaşıyor ~ Bernoulli(0,6); meslek kursu ~ Bernoulli(0,25 + 0,15·1{yaş < 35});
* P(işe yerleşti) = Λ(−3,2 + 0,06·yaş + 0,15·(eğitim − 12) − 0,5·kadın + 0,3·il merkezi + 0,9·kurs), Λ lojistik
  dağılım fonksiyonu: Logit modeli doğru tanımlıdır, kursun gerçek katsayısı 0,9'dur.

Bağış verisi (Konu 6; ``n`` = 800 hane, tohum 807):

* hane geliri (bin TL/ay) = exp(N(log 25, 0,5²)), bir ondalığa yuvarlanır; yaş ~ Tam sayı[22, 70]; hane büyüklüğü =
  min(1 + Poisson(2), 9), hanelerin %2'sinde boş; kentte yaşıyor ~ Bernoulli(0,65);
* gizli bağış B* = −380 + 22·gelir − 12·(gelir − 40)₊ + 3·(yaş − 40) + 40·kent − 15·(hane − 3) + N(0, 250²);
* aylık bağış (TL) = max(B*, 0), 10 TL'ye yuvarlanır. Gizli model (yuvarlama dışında) normal ve homoskedastik
  hatalı bir Tobit'tir; gelirin gizli bağışa etkisi 40 bin TL'ye kadar 22, sonrasında 10 TL'dir. Uygulamanın spline
  düğümleri gelirin çeyreklerindedir; gerçek kırılma noktasıyla (40) çakışmaz.

Çalışma saati verisi (Konu 8; 40 okul × 20 öğrenci = 800 öğrenci, tohum 807):

* okulun ortalama çalışma saati m_g ~ U(5, 35); öğrencinin haftalık çalışma saati = m_g + N(0, 5²), [0, 45]
  aralığına kırpılıp yarım saate yuvarlanır (aynı okuldaki öğrencilerin çalışma saatleri birbirine yakındır);
* okul etkisi u_g ~ N(0, 5²); sınav puanı = 35 + 2,4·saat − 0,05·saat² + u_g + N(0, 7²), bir ondalığa yuvarlanır.
  Gerçek koşullu ortalama tepe noktası 24 saatte olan bir paraboldür. Aynı okuldaki öğrenciler iki yoldan
  bağımlıdır: saatleri ortak okul ortalaması m_g çevresinde, puanları ortak okul etkisi u_g ile (u_g saatleri
  etkilemez; m_g ile bağımsızdır).
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


def ikili_veri(n: int = 1200, seed: int = SEED) -> pd.DataFrame:
    """Kurgusal işe yerleşme verisi (yukarıdaki DGP); sütun adları Excel'deki gibi Türkçedir."""

    rng = np.random.default_rng(seed)
    yas = rng.integers(22, 61, size=n)
    egitim = rng.choice(np.array((8, 11, 12, 14, 16)), size=n, p=np.array((0.20, 0.15, 0.30, 0.15, 0.20)))
    kadin = (rng.random(n) < 0.48).astype(int)
    merkez = (rng.random(n) < 0.6).astype(int)
    kurs = rng.random(n) < 0.25 + 0.15 * (yas < 35)
    indeks = -3.2 + 0.06 * yas + 0.15 * (egitim - 12) - 0.5 * kadin + 0.3 * merkez + 0.9 * kurs
    yerlesti = rng.random(n) < 1.0 / (1.0 + np.exp(-indeks))
    egitim_bos = np.where(rng.random(n) < 0.02, np.nan, egitim.astype(float))
    return pd.DataFrame({
        "İşe yerleşti": np.where(yerlesti, "Evet", "Hayır"),
        "Yaş": yas,
        "Meslek kursu": np.where(kurs, "Var", "Yok"),
        "Eğitim yılı": egitim_bos,
        "Kadın (1/0)": kadin,
        "İl merkezinde yaşıyor (1/0)": merkez,
    })


def bagis_verisi(n: int = 800, seed: int = SEED) -> pd.DataFrame:
    """Kurgusal hane bağış verisi (yukarıdaki DGP); sütun adları Excel'deki gibi Türkçedir."""

    rng = np.random.default_rng(seed)
    gelir = np.round(np.exp(rng.normal(np.log(25.0), 0.5, size=n)), 1)
    yas = rng.integers(22, 71, size=n)
    hane = np.minimum(1 + rng.poisson(2.0, size=n), 9)
    kent = (rng.random(n) < 0.65).astype(int)
    gizli = (-380 + 22 * gelir - 12 * np.maximum(gelir - 40, 0) + 3 * (yas - 40) + 40 * kent - 15 * (hane - 3)
             + rng.normal(0, 250, size=n))
    bagis = np.round(np.maximum(gizli, 0) / 10) * 10
    hane_bos = np.where(rng.random(n) < 0.02, np.nan, hane.astype(float))
    return pd.DataFrame({
        "Aylık bağış (TL)": bagis,
        "Hane geliri (bin TL)": gelir,
        "Yaş": yas,
        "Hane büyüklüğü": hane_bos,
        "Kentte yaşıyor (1/0)": kent,
    })


def calisma_verisi(schools: int = 40, students: int = 20, seed: int = SEED) -> pd.DataFrame:
    """Kurgusal okul–öğrenci verisi (yukarıdaki DGP); sütun adları Excel'deki gibi Türkçedir."""

    rng = np.random.default_rng(seed)
    school = np.repeat(np.arange(1, schools + 1), students)
    mean_hours = rng.uniform(5, 35, schools)[school - 1]
    hours = np.round(np.clip(mean_hours + rng.normal(0, 5, len(school)), 0, 45) * 2) / 2
    effect = rng.normal(0, 5, schools)[school - 1]
    score = np.round(35 + 2.4 * hours - 0.05 * hours ** 2 + effect + rng.normal(0, 7, len(school)), 1)
    return pd.DataFrame({"Sınav puanı": score, "Haftalık çalışma saati": hours, "Okul": school})
