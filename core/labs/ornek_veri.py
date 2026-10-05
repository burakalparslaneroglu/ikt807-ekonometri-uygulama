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

Burs verisi (Konu 9; 40 okul × 30 öğrenci = 1.200 öğrenci, tohum 807):

* okul etkileri a_g ~ N(0, 6²) (sınav puanında) ve b_g ~ N(0, 0,25²) (not ortalamasında);
* bursluluk sınavı puanı = a_g + 69 + N(0, 12²), tam sayıya yuvarlanıp [20, 100] aralığına kırpılır; puanı 70 ve üstü
  olan öğrenci burs alır (keskin kural; öğrencilerin %53'ü). Puanların medyanı 70'tir: uygulamanın eşik alanı ilk
  değer olarak medyanı yazdığı için örnek dosya gerçek eşikle açılır;
* mezuniyet not ortalaması = 1,2 + 0,025·puan + 0,20·burs + b_g + N(0, 0,35²), [0, 4] aralığına kırpılıp iki ondalığa
  yuvarlanır: eşikteki gerçek sıçrama 0,20 puandır. Aynı okulun öğrencileri ortak okul etkileriyle bağımlıdır.

Sınıf verisi (Konu 10; 30 okul × Tam sayı[18, 32] öğrenci, tohum 807):

* okulların 15'i rastgele seçilir ve kodlama kursu programına alınır (Program = 1, okul düzeyinde atama);
* okul etkisi s_g ~ N(0, 6²); başlangıç puanı = 50 + N(0, 10²), bir ondalığa yuvarlanır; kız öğrenci ~ Bernoulli(0,5);
* matematik puanı = 20 + 0,6·başlangıç + 1,5·kız + 3·program + s_g + N(0, 8²), bir ondalığa yuvarlanır: programın
  gerçek etkisi 3 puandır. Program okul düzeyinde atandığı ve okul etkisi ortak olduğu için öğrenci düzeyindeki
  bootstrap belirsizliği küçük gösterir.

Konut verisi (Konu 11; ``n`` = 800 daire, tohum 807):

* alan (m²) = 40 + Gamma(şekil 3, ölçek 25), tam sayıya yuvarlanır; oda sayısı = min(1 + ⌊alan/40⌋, 6); bina yaşı ~
  Tam sayı[0, 40]; kat ~ Tam sayı[0, 12]; merkeze uzaklık (km) = Gamma(şekil 2, ölçek 3), bir ondalığa yuvarlanır; asansör ~
  Bernoulli(0,4 + 0,4·1{kat ≥ 4}); otopark ~ Bernoulli(0,45); manzara ~ Bernoulli(0,15); ilçe altı ilçeden biri (A–F,
  eşit olasılıkla); ısıtma Kombi, Merkezi ya da Soba (olasılıklar 0,60; 0,25; 0,15; ayrı bir üreteçle, tohum (807, 11));
* log fiyat = 7,4 + 0,85·log(alan/100) + 0,04·oda − 0,012·yaş + 0,00015·yaş² − 0,035·uzaklık + 0,10·asansör + 0,06·otopark
  + 0,22·manzara + 0,015·kat·asansör + ilçe etkisi (A 0, B 0,10, C −0,08, D 0,18, E −0,15, F 0,05) + ısıtma etkisi
  (Kombi 0, Merkezi 0,04, Soba −0,12) + N(0, 0,20²);
* satış fiyatı (bin TL) = exp(log fiyat), bir ondalığa yuvarlanır. Doğrusal olmayan terimler (log alan, yaş², kat ×
  asansör) gerçek modelde vardır; basit doğrusal model onları kaçırır.

Kurs verisi (Konu 12; 50 firma × 30 çalışan = 1.500 çalışan, tohum 807):

* yaş ~ Tam sayı[22, 60]; eğitim yılı ∈ {8, 11, 12, 14, 16} (eşit olasılıkla); kadın ~ Bernoulli(0,45); firma etkisi
  f_g ~ N(0, 0,15²);
* önceki log kazanç p = 2,5 + 0,06·(eğitim − 12) + 0,02·(yaş − 40) − 0,0008·(yaş − 40)² + N(0, 0,3²);
* P(kursa katıldı) = Λ(−0,8 + 3·(p − 2,5)² + 0,04·(eğitim − 12) − 0,02·(yaş − 40)), Λ lojistik: önceki kazancı 2,5'ten
  uzak (çok düşük ya da çok yüksek) olanlar kursa daha çok katılır;
* log kazanç = 0,3 + 0,9·p + 1,0·(p − 2,5)² + 0,03·(eğitim − 12) − 0,05·kadın + 0,08·kurs + f_g + N(0, 0,25²): kursun
  gerçek etkisi 0,08'dir. Katılım ve kazanç önceki kazancın aynı kare terimine bağlıdır: önceki kazancı doğrusal giren
  OLS yukarı yanlıdır, kare terimleri içeren esnek ayarlama (DML) yanlılığı büyük ölçüde giderir. Bütün karıştırıcılar
  gözlenir (koşullu bağımsızlık bu kurgusal veride doğrudur).
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


def burs_verisi(schools: int = 40, students: int = 30, seed: int = SEED) -> pd.DataFrame:
    """Kurgusal bursluluk sınavı verisi (yukarıdaki DGP); sütun adları Excel'deki gibi Türkçedir."""

    rng = np.random.default_rng(seed)
    school = np.repeat(np.arange(1, schools + 1), students)
    score_effect = rng.normal(0, 6, schools)[school - 1]
    grade_effect = rng.normal(0, 0.25, schools)[school - 1]
    score = np.clip(np.round(69 + score_effect + rng.normal(0, 12, len(school))), 20, 100)
    scholarship = (score >= 70).astype(float)
    gpa = np.round(np.clip(1.2 + 0.025 * score + 0.20 * scholarship + grade_effect
                           + rng.normal(0, 0.35, len(school)), 0, 4), 2)
    return pd.DataFrame({
        "Mezuniyet not ortalaması": gpa,
        "Sınav puanı": score.astype(int),
        "Burs aldı": np.where(scholarship == 1, "Evet", "Hayır"),
        "Okul": school,
        "Öğrenci no": np.arange(1, len(school) + 1),
    })


def sinif_verisi(schools: int = 30, seed: int = SEED) -> pd.DataFrame:
    """Kurgusal okul düzeyinde program verisi (yukarıdaki DGP); sütun adları Excel'deki gibi Türkçedir."""

    rng = np.random.default_rng(seed)
    sizes = rng.integers(18, 33, size=schools)
    school = np.repeat(np.arange(1, schools + 1), sizes)
    program = np.zeros(schools, dtype=int)
    program[rng.choice(schools, size=schools // 2, replace=False)] = 1
    effect = rng.normal(0, 6, schools)[school - 1]
    n = len(school)
    baseline = np.round(50 + rng.normal(0, 10, n), 1)
    girl = (rng.random(n) < 0.5).astype(int)
    score = np.round(20 + 0.6 * baseline + 1.5 * girl + 3 * program[school - 1] + effect + rng.normal(0, 8, n), 1)
    return pd.DataFrame({
        "Matematik puanı": score,
        "Program (1/0)": program[school - 1],
        "Başlangıç puanı": baseline,
        "Kız (1/0)": girl,
        "Okul": school,
    })


def konut_verisi(n: int = 800, seed: int = SEED) -> pd.DataFrame:
    """Kurgusal konut verisi (yukarıdaki DGP); sütun adları Excel'deki gibi Türkçedir."""

    rng = np.random.default_rng(seed)
    area = np.round(40 + rng.gamma(3.0, 25.0, size=n))
    rooms = np.clip(1 + np.floor(area / 40), 1, 6)
    age = rng.integers(0, 41, size=n)
    floor = rng.integers(0, 13, size=n)
    distance = np.round(rng.gamma(2.0, 3.0, size=n), 1)
    lift = (rng.random(n) < 0.4 + 0.4 * (floor >= 4)).astype(int)
    parking = (rng.random(n) < 0.45).astype(int)
    view = (rng.random(n) < 0.15).astype(int)
    district = rng.choice(np.array(list("ABCDEF")), size=n)
    effects = {"A": 0.0, "B": 0.10, "C": -0.08, "D": 0.18, "E": -0.15, "F": 0.05}
    heating = np.random.default_rng([seed, 11]).choice(np.array(["Kombi", "Merkezi", "Soba"]), size=n,
                                                       p=[0.60, 0.25, 0.15])
    heating_effects = {"Kombi": 0.0, "Merkezi": 0.04, "Soba": -0.12}
    log_price = (7.4 + 0.85 * np.log(area / 100) + 0.04 * rooms - 0.012 * age + 0.00015 * age**2 - 0.035 * distance
                 + 0.10 * lift + 0.06 * parking + 0.22 * view + 0.015 * floor * lift
                 + np.array([effects[d] for d in district]) + np.array([heating_effects[h] for h in heating])
                 + rng.normal(0, 0.20, size=n))
    return pd.DataFrame({
        "Satış fiyatı (bin TL)": np.round(np.exp(log_price), 1),
        "Alan (m²)": area.astype(int),
        "Oda sayısı": rooms.astype(int),
        "Bina yaşı": age,
        "Kat": floor,
        "Merkeze uzaklık (km)": distance,
        "Asansör (1/0)": lift,
        "Otopark (1/0)": parking,
        "Manzara (1/0)": view,
        "İlçe": district,
        "Isıtma": heating,
    })


def kurs_verisi(firms: int = 50, workers: int = 30, seed: int = SEED) -> pd.DataFrame:
    """Kurgusal mesleki kurs verisi (yukarıdaki DGP); sütun adları Excel'deki gibi Türkçedir."""

    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(1, firms + 1), workers)
    n = len(firm)
    age = rng.integers(22, 61, size=n)
    education = rng.choice(np.array((8, 11, 12, 14, 16)), size=n)
    female = (rng.random(n) < 0.45).astype(int)
    effect = rng.normal(0, 0.15, firms)[firm - 1]
    previous = (2.5 + 0.06 * (education - 12) + 0.02 * (age - 40) - 0.0008 * (age - 40) ** 2
                + rng.normal(0, 0.3, size=n))
    index = -0.8 + 3.0 * (previous - 2.5) ** 2 + 0.04 * (education - 12) - 0.02 * (age - 40)
    course = (rng.random(n) < 1.0 / (1.0 + np.exp(-index))).astype(int)
    earnings = (0.3 + 0.9 * previous + 1.0 * (previous - 2.5) ** 2 + 0.03 * (education - 12) - 0.05 * female
                + 0.08 * course + effect + rng.normal(0, 0.25, size=n))
    return pd.DataFrame({
        "Log kazanç": np.round(earnings, 4),
        "Kursa katıldı (1/0)": course,
        "Önceki log kazanç": np.round(previous, 4),
        "Yaş": age,
        "Eğitim yılı": education,
        "Kadın (1/0)": female,
        "Firma": firm,
    })
