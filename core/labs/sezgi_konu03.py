"""Konu 3 Sezgi deneyleri: tanımlama sorunlarını bilinen bir veri üretim süreciyle görmek.

Deney 1  Ham grup farkı = tedavi etkisi + seçim farkı            (Notlar §3.3, §3.4)
Deney 2  Ölçüm hatası: eğim neden sıfıra çekilir?                 (Notlar §3.8.4)
Deney 3  Örneklem büyüdükçe içsellik kaybolur mu? (Monte Carlo)   (Notlar §3.9, §3.12)
"""

from __future__ import annotations

import numpy as np

from core.labs import expr as E
from core.labs.runner import LabState, coverage_key
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, number, plain
from core.labs.spec import (
    OLS,
    Curve,
    Derive,
    Draw,
    Histogram,
    MeanPoints,
    ModelLine,
    MonteCarlo,
    NewSample,
    NoteRef,
    Plot,
    Scatter,
)

SEED = 803
FRAME = "sim"
TOPIC = "konu03"
EFFECT = 1.0
SELECTION_SCALE = 2.0 * 2.0 * 0.3989422804014327 / 0.5
"""Deney 1'de E[Y(0) | D=1] − E[Y(0) | D=0] = 2·(E[A | D=1] − E[A | D=0]) = 3,1915·s/√(1+s²)."""

SAMPLE_SIZE = SimParameter(
    "n", "Gözlem sayısı", 200, 5000, 2000, 200,
    "Örneklem büyüdükçe örnekleme dalgalanması küçülür.", integer=True, decimals=0,
)


# --- Deney 1: seçim ayrıştırması ------------------------------------------------------

def expected_selection(s: float) -> float:
    return SELECTION_SCALE * s / float(np.sqrt(1.0 + s * s))


def _build_selection(p: Parameters) -> tuple:
    s = round(p["s"], 6)
    return (
        NewSample(FRAME, int(p["n"]), SEED),
        Draw(FRAME, "a", "normal", 0, 1, "Başlangıç potansiyeli A (araştırmacı gözlemez)"),
        Draw(FRAME, "e", "normal", 0, 1, "Sonucu etkileyen diğer etkenler"),
        Derive(FRAME, "y0", E.add(E.add(10, E.mul(2, E.var("a"))), E.var("e")),
               "Potansiyel sonuç Y(0): tedavi olmasaydı gözlenecek sonuç"),
        Draw(FRAME, "nu", "normal", 0, 1, "Tedavi kararındaki rastlantısal kısım"),
        Derive(FRAME, "d", E.positive(E.add(E.mul(s, E.var("a")), E.var("nu"))),
               "Tedavi kararı: D = 1{s·A + ν > 0}; s = 0 rastgele atamadır"),
        Derive(FRAME, "y", E.add(E.var("y0"), E.mul(EFFECT, E.var("d"))), "Gözlenen sonuç: Y = Y(0) + τ·D, τ = 1"),
        OLS("gozlenen", FRAME, "y", ("d",)),
        OLS("secim", FRAME, "y0", ("d",)),
        Plot(FRAME, "d",
             (MeanPoints("y", "Gözlenen Y ortalaması"),
              MeanPoints("y0", "Y(0) ortalaması (yalnız simülasyonda görülür)"),
              ModelLine("gozlenen", "Gözlenen fark"),
              ModelLine("secim", "Seçim farkı", dashed=True)),
             "Tedavi D (0: almadı, 1: aldı)", "Ortalama sonuç", "Gözlenen fark = tedavi etkisi + seçim farkı"),
    )


def _selection_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    observed = float(state.models["gozlenen"].params["d"])
    selection = float(state.models["secim"].params["d"])
    return (
        SimMetric("Gözlenen fark", plain(observed), "Ȳ(D=1) − Ȳ(D=0); OLS Y ~ D eğimi."),
        SimMetric("Tedavi etkisi τ", plain(EFFECT), "DGP: her bireyde τ = 1; bu yüzden ATT = ATE = 1."),
        SimMetric("Seçim farkı", plain(selection),
                  "Y(0) ortalamalarının farkı. Gerçek veride hesaplanamaz: tedavi alanların Y(0)'ı karşı-olgusaldır."),
        SimMetric("Gözlenen − seçim", plain(observed - selection), "Gözlenen fark eksi seçim farkı: ayrıştırma örneklemde birebir tutar, τ = 1."),
        SimMetric("Seçim farkı (DGP)", plain(expected_selection(p["s"])), "Beklenen seçim farkı: 3,1915·s/√(1+s²)."),
    )


def _selection_takeaway(state: LabState, p: Parameters) -> str:
    observed = float(state.models["gozlenen"].params["d"])
    selection = float(state.models["secim"].params["d"])
    if abs(p["s"]) < 1e-9:
        return (
            "s = 0: tedavi rastgele atanıyor, potansiyel sonuçlardan bağımsız. İki grubun Y(0) ortalamaları yalnız "
            f"örnekleme dalgalanması kadar farklı ({plain(selection, 3)}); gözlenen fark ({plain(observed, 3)}) "
            "ATE = 1'i tahmin eder. Rassal atama seçim farkını beklentide sıfırlar."
        )
    direction = "yüksek" if p["s"] > 0 else "düşük"
    effect = "abartır" if selection > 0 else "küçümser"
    return (
        f"s = {plain(p['s'], 2)}: başlangıç potansiyeli {direction} olanlar tedaviyi daha çok seçiyor. Tedavi alanlar, "
        f"tedavi olmasaydı bile {plain(abs(selection), 3)} puan {'daha iyi' if selection > 0 else 'daha kötü'} "
        f"olacaktı. Gözlenen fark ({plain(observed, 3)}) tedavi etkisini (1) bu seçim farkı kadar {effect}. Örneklemi "
        "büyütmek bunu düzeltmez; seçim farkı gerçek veride gözlenemez, bu yüzden tanımlama bir tasarım argümanı ister."
    )


SELECTION = SimExperiment(
    topic_key=TOPIC, number=1,
    title="Ham grup farkı = tedavi etkisi + seçim farkı",
    question="Tedavi alanlarla almayanların ortalama sonuç farkı tedavinin etkisini mi ölçer?",
    note=NoteRef("3.3", 0),
    parameters=(
        SAMPLE_SIZE,
        SimParameter("s", "Seçim gücü s", -1.5, 1.5, 1.0, 0.25,
                     "s = 0 rastgele atama; s > 0 potansiyeli yüksek olanlar, s < 0 düşük olanlar tedaviyi seçer.",
                     decimals=2),
    ),
    dgp=lambda p: (
        r"A_i\sim N(0,1)\ \text{(gözlenmez)}, \quad e_i\sim N(0,1), \quad \nu_i\sim N(0,1)",
        r"Y_i(0)=10+2A_i+e_i, \qquad Y_i(1)=Y_i(0)+\tau, \quad \tau=1",
        rf"D_i=\mathbf 1\{{s\,A_i+\nu_i>0\}}, \quad s={number(p['s'], 2)}, \qquad Y_i=Y_i(0)+\tau D_i",
    ),
    dgp_note=(
        "Tedavinin etkisi herkes için aynı: τ = 1. Tedavi kararı, araştırmacının gözlemediği başlangıç potansiyeli A'ya "
        "bağlı olabilir. Simülasyonda her bireyin Y(0)'ı bilindiği için gözlenen farkı iki bileşenine ayırabiliriz; "
        "gerçek veride tedavi alanların Y(0)'ı karşı-olgusaldır."
    ),
    look_at=(
        "**Grafik** — D = 0 ve D = 1 gruplarında gözlenen Y ortalaması ile Y(0) ortalaması. D = 0 grubunda ikisi "
        "aynıdır. D = 1'deki dikey fark tedavi etkisidir (τ = 1); kesikli çizginin eğimi seçim farkıdır.",
        "**Ölçüler** — gözlenen fark, tedavi etkisi ve seçim farkı: "
        "$\\Delta_{obs}=ATT+\\{E[Y(0)\\mid D=1]-E[Y(0)\\mid D=0]\\}$.",
    ),
    build=_build_selection, metrics=_selection_metrics, takeaway=_selection_takeaway,
    labels=(("d", "Tedavi"), ("y", "Gözlenen sonuç"), ("y0", "Y(0)")),
)


# --- Deney 2: ölçüm hatası -----------------------------------------------------------

def attenuation(variance: float) -> float:
    return 1.0 / (1.0 + variance)


def _build_measurement(p: Parameters) -> tuple:
    variance = round(p["var_u"], 6)
    factor = round(attenuation(variance), 6)
    x = E.var("x")
    return (
        NewSample(FRAME, int(p["n"]), SEED),
        Draw(FRAME, "z", "normal", 0, 1, "Gerçek açıklayıcı değişken Z (gözlenmez)"),
        Draw(FRAME, "u", "normal", 0, round(float(np.sqrt(variance)), 6), "Klasik ölçüm hatası u"),
        Derive(FRAME, "x", E.add(E.var("z"), E.var("u")), "Gözlenen değişken: X = Z + u"),
        Draw(FRAME, "eps", "normal", 0, 1, "Yapısal hata"),
        Derive(FRAME, "y", E.add(E.add(1, E.var("z")), E.var("eps")), "Y = 1 + β·Z + ε, β = 1"),
        OLS("gercek", FRAME, "y", ("z",)),
        OLS("olculen", FRAME, "y", ("x",)),
        Plot(FRAME, "x",
             (Scatter("y", "Gözlemler"),
              Curve(E.add(1, x), "Yapısal ilişki: eğim β = 1"),
              ModelLine("olculen", "OLS, gözlenen X ile"),
              Curve(E.add(1, E.mul(factor, x)), "Olasılık limiti: eğim λβ", dashed=True)),
             "Gözlenen X", "Y", "Ölçüm hatası altında OLS doğrusu"),
    )


def _measurement_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    true_slope = float(state.models["gercek"].params["z"])
    measured = float(state.models["olculen"].params["x"])
    factor = attenuation(p["var_u"])
    return (
        SimMetric("OLS, gerçek Z ile", plain(true_slope), "Yalnız simülasyonda mümkün: Z gözlenmez."),
        SimMetric("OLS, gözlenen X ile", plain(measured), "Uygulamada elde edilen eğim."),
        SimMetric("Zayıflama çarpanı λ (DGP)", plain(factor), "Var(Z)/(Var(Z)+Var(u)), Var(Z) = 1."),
        SimMetric("Olasılık limiti λβ", plain(factor * 1.0), "n → ∞ iken gözlenen X ile OLS eğimi."),
    )


def _measurement_takeaway(state: LabState, p: Parameters) -> str:
    measured = float(state.models["olculen"].params["x"])
    factor = attenuation(p["var_u"])
    if p["var_u"] <= 0:
        return "Var(u) = 0: ölçüm hatası yok, X = Z. OLS eğimi yapısal β = 1'i tahmin eder."
    return (
        f"Var(u) = {plain(p['var_u'], 2)}: X'teki varyansın yalnız %{100 * factor:.0f} kadarı gerçek sinyaldir. OLS eğimi "
        f"({plain(measured, 3)}) yapısal etkiyi (1) λ = {plain(factor, 3)} çarpanıyla sıfıra doğru çeker. X, u'yu "
        "içerdiği için yeni hata ε − βu ile ilişkilidir: ölçüm hatası bir içsellik kaynağıdır. Örneklemi büyütmek "
        "yalnız bu yanlış hedef etrafındaki belirsizliği azaltır."
    )


MEASUREMENT = SimExperiment(
    topic_key=TOPIC, number=2,
    title="Ölçüm hatası: eğim neden sıfıra çekilir?",
    question="Açıklayıcı değişken hatalı ölçülürse OLS eğimi neyi tahmin eder?",
    note=NoteRef("3.8.4", 0),
    parameters=(
        SAMPLE_SIZE,
        SimParameter("var_u", "Ölçüm hatası varyansı Var(u)", 0.0, 3.0, 1.0, 0.25,
                     "Var(u) = 0 iken X gerçek değişkene eşittir.", decimals=2),
    ),
    dgp=lambda p: (
        r"Z_i\sim N(0,1)\ \text{(gözlenmez)}, \qquad u_i\sim N(0,\operatorname{Var}(u)), \qquad \varepsilon_i\sim N(0,1)",
        rf"X_i=Z_i+u_i, \qquad \operatorname{{Var}}(u)={number(p['var_u'], 2)}",
        r"Y_i=1+\beta Z_i+\varepsilon_i, \quad \beta=1, \qquad "
        r"\operatorname{plim}\hat\beta_{OLS}=\beta\,\frac{\operatorname{Var}(Z)}{\operatorname{Var}(Z)+\operatorname{Var}(u)}",
    ),
    dgp_note=(
        "Klasik ölçüm hatası: u, gerçek değişken Z'den ve yapısal hatadan bağımsız, ortalaması sıfır. Araştırmacı "
        "yalnız X = Z + u'yu gözler. Simülasyonda Z bilindiği için iki regresyon yan yana tahmin edilebilir."
    ),
    look_at=(
        "**Grafik** — gözlenen X ile Y; yapısal ilişki (eğim 1), OLS doğrusu ve DGP'den bilinen olasılık limiti "
        "(kesikli, eğim λβ).",
        "**Ölçüler** — gerçek Z ile ve gözlenen X ile OLS eğimleri, zayıflama çarpanı λ.",
    ),
    build=_build_measurement, metrics=_measurement_metrics, takeaway=_measurement_takeaway,
    labels=(("x", "Gözlenen X"), ("y", "Y"), ("z", "Gerçek Z")),
)


# --- Deney 3: büyük örneklem ve içsellik ---------------------------------------------

REPS = 300
STRUCTURAL = 2.0


def ols_limit(a: float) -> float:
    """U dışarıda: β + 1,5·Cov(X,U)/Var(X) = 2 + 1,5a/(a²+1)."""

    return STRUCTURAL + 1.5 * a / (a * a + 1.0)


def _build_large_sample(p: Parameters) -> tuple:
    a = round(p["a"], 6)
    limit = round(ols_limit(a), 4)
    body = (
        NewSample(FRAME, int(p["n"]), None),
        Draw(FRAME, "u", "normal", 0, 1, "Gözlenmeyen karıştırıcı U"),
        Draw(FRAME, "v", "normal", 0, 1, "X'in U'dan bağımsız kısmı"),
        Derive(FRAME, "x", E.add(E.mul(a, E.var("u")), E.var("v")), "X = a·U + v"),
        Draw(FRAME, "eps", "normal", 0, 1, "Yapısal hata"),
        Derive(FRAME, "y", E.add(E.add(E.add(1, E.mul(STRUCTURAL, E.var("x"))), E.mul(1.5, E.var("u"))), E.var("eps")),
               "Y = 1 + 2X + 1,5U + ε"),
        OLS("kisa", FRAME, "y", ("x",), vcov="HC1"),
        OLS("uzun", FRAME, "y", ("x", "u"), vcov="HC1"),
    )
    return (
        MonteCarlo(
            FRAME, REPS, SEED, body,
            (
                ("b_kisa", E.coef("kisa", "x")), ("se_kisa", E.se("kisa", "x")),
                ("b_uzun", E.coef("uzun", "x")), ("se_uzun", E.se("uzun", "x")),
            ),
            "mc",
            f"Monte Carlo: her tekrarda yeni örneklem, U dışarıda ve U kontrol edilerek iki OLS",
            coverage=(("b_kisa", "se_kisa", STRUCTURAL), ("b_uzun", "se_uzun", STRUCTURAL)),
        ),
        Histogram(
            "mc", (("b_kisa", "U dışarıda: Y ~ X"), ("b_uzun", "U kontrol: Y ~ X + U")),
            1.4, 3.2,
            ((STRUCTURAL, "Yapısal β = 2"), (limit, "OLS olasılık limiti (U dışarıda)")),
            "X katsayısının tahmini", f"{REPS} örneklemde OLS katsayısının dağılımı", bins=36,
        ),
    )


def _large_sample_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    table = state.tables["mc"]
    return (
        SimMetric("U dışarıda: ortalama", plain(float(table["b_kisa"].mean())), f"{REPS} tahminin ortalaması."),
        SimMetric("Olasılık limiti (DGP)", plain(ols_limit(p["a"])), "2 + 1,5a/(a²+1)."),
        SimMetric("SS, U dışarıda", plain(float(table["b_kisa"].std())), "Tahminlerin standart sapması: örnekleme dağılımının genişliği."),
        SimMetric("Kapsama, U dışarıda", f"%{100 * state.scalars[coverage_key('mc', 'b_kisa')]:.1f}".replace(".", ","),
                  "Tahmin ± 1,96·SH aralığının β = 2'yi içerdiği tekrarların payı; nominal %95."),
        SimMetric("Kapsama, U kontrol", f"%{100 * state.scalars[coverage_key('mc', 'b_uzun')]:.1f}".replace(".", ","),
                  "U gözlenebilseydi: nominal %95'e yakın."),
    )


def _large_sample_takeaway(state: LabState, p: Parameters) -> str:
    table = state.tables["mc"]
    coverage = 100 * state.scalars[coverage_key("mc", "b_kisa")]
    if p["a"] <= 0:
        return (
            "a = 0: X, U'dan bağımsız; U'yu dışarıda bırakmak yalnız hata varyansını artırır, katsayıyı kaydırmaz. İki "
            "dağılım da β = 2 etrafında toplanır."
        )
    return (
        f"n = {int(p['n'])}: U dışarıda bırakılınca tahminler {plain(ols_limit(p['a']), 3)} etrafında toplanıyor "
        f"(SS {plain(float(table['b_kisa'].std()), 3)}); güven aralığının β = 2'yi kapsadığı tekrarların payı "
        f"%{coverage:.0f}. n'yi büyütün: dağılım daralır ama yeri değişmez, kapsama sıfıra iner. Büyük örneklem tesadüfi "
        "belirsizliği azaltır; sistematik tanımlama hatasını ortadan kaldırmaz. U gözlenseydi kontrol etmek sorunu "
        "çözerdi; gözlenmediğinde başka bir tanımlama stratejisi gerekir."
    )


LARGE_SAMPLE = SimExperiment(
    topic_key=TOPIC, number=3,
    title="Örneklem büyüdükçe içsellik kaybolur mu?",
    question="Gözlenmeyen bir karıştırıcı varken örneklemi büyütmek OLS'yi gerçek etkiye yaklaştırır mı?",
    note=NoteRef("3.9", 0),
    parameters=(
        SimParameter("n", "Her örneklemdeki gözlem sayısı", 100, 5000, 100, 100,
                     "Her Monte Carlo tekrarı bu büyüklükte yeni bir örneklem çeker.", integer=True, decimals=0),
        SimParameter("a", "U'nun X'e etkisi a", 0.0, 1.5, 0.1, 0.05,
                     "a = 0 iken X ile U ilişkisizdir. Notlar §3.12'deki süreçte a = 0,8.", decimals=2),
    ),
    dgp=lambda p: (
        r"U_i\sim N(0,1), \qquad v_i\sim N(0,1), \qquad \varepsilon_i\sim N(0,1)",
        rf"X_i=a\,U_i+v_i, \quad a={number(p['a'], 2)}, \qquad Y_i=1+2X_i+1{{,}}5\,U_i+\varepsilon_i",
        r"\operatorname{plim}\hat\beta_{OLS}^{\,U\ \text{dışarıda}}=2+\frac{1{,}5\,a}{a^2+1}",
    ),
    dgp_note=(
        f"Notlar §3.12'deki süreç (orada a = 0,8). {REPS} tekrarın her birinde yeni bir örneklem çekilir ve iki regresyon "
        "tahmin edilir: U dışarıda (gerçek uygulamadaki durum) ve U kontrol edilerek (yalnız simülasyonda mümkün)."
    ),
    look_at=(
        "**Grafik** — iki tahmin edicinin örnekleme dağılımı; yapısal β = 2 ve U dışarıdayken OLS'nin olasılık "
        "limiti (dikey çizgiler).",
        "**Ölçüler** — tahminlerin ortalaması ve standart sapması; güven aralığının β = 2'yi kapsama oranı.",
    ),
    build=_build_large_sample, metrics=_large_sample_metrics, takeaway=_large_sample_takeaway,
)


KONU03_EXPERIMENTS = (SELECTION, MEASUREMENT, LARGE_SAMPLE)
