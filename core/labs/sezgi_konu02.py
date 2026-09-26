"""Konu 2 Sezgi deneyleri: güvenilir çıkarım için bilinen gerçekle karşılaştırma.

Deney 1  Heteroskedastisite: katsayı aynı, belirsizlik farklı    (Notlar §2.6)
Deney 2  FWL ve eksik değişken: çoklu regresyon katsayısı neyi ölçer? (Notlar §2.1.1, §2.4)
Deney 3  Kümelenmiş veri: gözlem sayısı neden yanıltabilir?        (Notlar §2.7)
"""

from __future__ import annotations

import numpy as np

from core.labs import expr as E
from core.labs.runner import LabState
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, number, plain
from core.labs.spec import (
    OLS,
    ClusterDraw,
    Curve,
    Derive,
    Draw,
    MeanPoints,
    ModelLine,
    NewSample,
    NoteRef,
    Plot,
    Predict,
    Scatter,
)

SEED = 807
FRAME = "sim"
TOPIC = "konu02"
SLOPE = 0.5
CLUSTER_SIZE = 25

SAMPLE_SIZE = SimParameter(
    "n", "Gözlem sayısı", 200, 5000, 2000, 200,
    "Örneklem büyüdükçe tahminler DGP'deki gerçek değerlere yaklaşır.", integer=True, decimals=0,
)


def _true_slope_se(state: LabState) -> float:
    """X'e koşullu gerçek SH: √(Σ(x−x̄)²σ²(x)) / Σ(x−x̄)² — DGP'deki σ(x) bilindiği için hesaplanabilir."""

    frame = state.frames[FRAME]
    deviation = frame["x"] - frame["x"].mean()
    return float(np.sqrt((deviation**2 * frame["s"] ** 2).sum()) / (deviation**2).sum())


# --- Deney 1: heteroskedastisite ----------------------------------------------------

def _build_hetero(p: Parameters) -> tuple:
    gamma = round(p["gamma"], 6)
    x = E.var("x")
    return (
        NewSample(FRAME, int(p["n"]), SEED),
        Draw(FRAME, "u", "uniform", 0, 1, "Tek düze şok"),
        Derive(FRAME, "x", E.minimum(E.maximum(E.rounded(E.add(E.mul(10, E.power(E.var("u"), 2)), 0.5)), 1), 10),
               "Açıklayıcı değişken: 1 ile 10 arası tam sayı, küçük değerlerde yoğun (çarpık)"),
        Derive(FRAME, "s", E.add(0.5, E.mul(gamma, x)), "Gerçek koşullu standart sapma σ(x); DGP'den bilinir"),
        Draw(FRAME, "e", "normal", 0, 1, "Standart normal hata"),
        Derive(FRAME, "y", E.add(E.add(1, E.mul(SLOPE, x)), E.mul(E.var("s"), E.var("e"))), "Y = 1 + 0,5 x + σ(x) e"),
        OLS("klasik", FRAME, "y", ("x",)),
        OLS("hc1", FRAME, "y", ("x",), vcov="HC1"),
        Predict("klasik", FRAME, "ehat", "residual"),
        Derive(FRAME, "ehat2", E.power(E.var("ehat"), 2), "Artık kareleri"),
        Plot(FRAME, "x", (Scatter("y", "Gözlemler"), ModelLine("klasik", "OLS doğrusu")),
             "x", "Y", "Veri ve OLS doğrusu: yayılım x ile büyüyor mu?"),
        Plot(FRAME, "x",
             (MeanPoints("ehat2", "Artık karelerinin ortalaması"),
              Curve(E.power(E.add(0.5, E.mul(gamma, x)), 2), "Gerçek koşullu varyans σ²(x)")),
             "x", "Varyans", "Her x değerinde artık kareleri ve gerçek σ²(x)"),
    )


def _hetero_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    classic, robust = state.models["klasik"], state.models["hc1"]
    same = abs(classic.params["x"] - robust.params["x"]) < 1e-12
    return (
        SimMetric("OLS eğimi", plain(float(classic.params["x"])) + (" (iki fitte aynı)" if same else ""),
                  "HC1 seçeneği katsayıyı değiştirmez."),
        SimMetric("Gerçek SH (DGP'den)", plain(_true_slope_se(state)), "σ(x) bilindiği için X'e koşullu olarak hesaplanır."),
        SimMetric("Klasik SH", plain(float(classic.bse["x"])), "Sabit varyans varsayar."),
        SimMetric("HC1 SH", plain(float(robust.bse["x"])), "Varyansın x ile değişmesine izin verir."),
    )


def _hetero_takeaway(state: LabState, p: Parameters) -> str:
    true_se = _true_slope_se(state)
    classic, robust = float(state.models["klasik"].bse["x"]), float(state.models["hc1"].bse["x"])
    text = "İki tahminde de eğim aynıdır: HC1 katsayıyı değil, yalnız belirsizlik ölçüsünü değiştirir. "
    if p["gamma"] > 0:
        text += (
            f"γ > 0 iken klasik SH gerçek değerin yüzde {abs(100 * (classic / true_se - 1)):.0f} "
            f"{'altında' if classic < true_se else 'üstünde'}, HC1 ise yüzde {abs(100 * (robust / true_se - 1)):.0f} "
            "uzağında. Heteroskedastisite katsayıyı yanlı yapmaz, klasik standart hatayı yanıltıcı yapar."
        )
    else:
        text += "γ = 0 iken varyans sabittir; klasik ve HC1 standart hataları gerçek değere birlikte yakındır."
    return text


HETERO = SimExperiment(
    topic_key=TOPIC, number=1,
    title="Heteroskedastisite: katsayı aynı, belirsizlik farklı",
    question="Hatanın varyansı x ile büyüyorsa OLS katsayısı mı bozulur, standart hatası mı?",
    note=NoteRef("2.6", 0),
    parameters=(SAMPLE_SIZE, SimParameter("gamma", "Varyansın x ile büyüme hızı γ", 0.0, 0.3, 0.2, 0.05,
                                          "γ = 0 iken hata varyansı sabittir (homoskedastik).", decimals=2)),
    dgp=lambda p: (
        r"U_i\sim U(0,1), \quad X_i=\min\{\max\{\mathrm{yuvarla}(10\,U_i^2+0{,}5),\,1\},\,10\}, \quad e_i\sim N(0,1)",
        rf"\sigma(x)=0{{,}}5+\gamma\,x, \qquad \gamma={number(p['gamma'], 2)}",
        r"Y_i=1+0{,}5\,X_i+\sigma(X_i)\,e_i",
    ),
    dgp_note=(
        "X çarpık: gözlemlerin çoğu küçük x değerlerinde, azı büyük değerlerde (eğitim yılı yüksek az sayıda "
        "kişi gibi). Hata varyansı x ile büyüyor; yani seyrek ve bu yüzden eğimi en çok belirleyen (yüksek "
        "kaldıraçlı) gözlemler aynı zamanda en gürültülü olanlar. σ(x) bilindiği için eğimin X'e koşullu "
        "gerçek standart hatası hesaplanabilir; klasik ve HC1 bununla karşılaştırılır."
    ),
    look_at=(
        "**Üst grafik** — gözlemler ve OLS doğrusu: γ > 0 iken yayılım x ile büyür.",
        "**Alt grafik** — her x değerinde artık karelerinin ortalaması (noktalar) ve gerçek "
        "$\\sigma^2(x)$ (eğri): heteroskedastisitenin kendisi.",
        "**Ölçüler** — gerçek SH ile klasik ve HC1 standart hataları.",
    ),
    build=_build_hetero, metrics=_hetero_metrics, takeaway=_hetero_takeaway,
)


# --- Deney 2: FWL ve eksik değişken ----------------------------------------------

def _build_fwl(p: Parameters) -> tuple:
    rho, beta2 = round(p["rho"], 6), round(p["beta2"], 6)
    return (
        NewSample(FRAME, int(p["n"]), SEED),
        Draw(FRAME, "x2", "normal", 0, 1, "Kontrol değişkeni x2"),
        Draw(FRAME, "v", "normal", 0, 1, "x1'in x2'den bağımsız kısmı"),
        Derive(FRAME, "x1", E.add(E.mul(rho, E.var("x2")), E.mul(round(float(np.sqrt(1 - rho**2)), 6), E.var("v"))),
               "İlgilenilen değişken x1: x2 ile korelasyonu ρ"),
        Draw(FRAME, "e", "normal", 0, 1, "Hata terimi"),
        Derive(FRAME, "y", E.add(E.add(E.add(1, E.mul(SLOPE, E.var("x1"))), E.mul(beta2, E.var("x2"))), E.var("e")),
               "Y = 1 + 0,5 x1 + β2 x2 + e"),
        OLS("uzun", FRAME, "y", ("x1", "x2")),
        OLS("kisa", FRAME, "y", ("x1",)),
        OLS("yardimci", FRAME, "x1", ("x2",)),
        Predict("yardimci", FRAME, "x1_art", "residual"),
        OLS("fwl", FRAME, "y", ("x1_art",)),
        Plot(FRAME, "x1_art", (Scatter("y", "Gözlemler"), ModelLine("fwl", "FWL doğrusu")),
             "x1'in x2'den arındırılmış kısmı", "Y", "FWL: Y'nin, x1'in x2'den arındırılmış kısmı üzerindeki regresyonu"),
    )


def _fwl_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    long = float(state.models["uzun"].params["x1"])
    fwl = float(state.models["fwl"].params["x1_art"])
    short = float(state.models["kisa"].params["x1"])
    target = SLOPE + p["beta2"] * p["rho"]
    return (
        SimMetric("Çoklu regresyon β̂₁", plain(long), "Y ~ x1 + x2 regresyonunda x1 katsayısı."),
        SimMetric("FWL eğimi", plain(fwl) + (" (birebir)" if abs(fwl - long) < 1e-10 else ""),
                  "Y ~ (x1'in x2'den arındırılmış kısmı)."),
        SimMetric("Kısa regresyon", plain(short), "Y ~ x1: x2 dışarıda."),
        SimMetric("Kısa regresyonun hedefi (DGP)", plain(target), "0,5 + β2·ρ: eksik değişken yanlılığı formülü."),
    )


def _fwl_takeaway(state: LabState, p: Parameters) -> str:
    text = (
        "FWL: çoklu regresyondaki x1 katsayısı, Y'nin yalnız x1'in x2'den arındırılmış kısmı üzerindeki "
        "regresyonunun eğimine birebir eşittir. Katsayı, x2 ile ortak olmayan değişkenlikten öğrenir. "
    )
    bias = p["beta2"] * p["rho"]
    if abs(bias) > 1e-9:
        text += (
            f"x2 dışarıda bırakılınca kısa regresyon 0,5 yerine 0,5 + β2·ρ = {plain(SLOPE + bias, 2)} değerini hedefler; "
            f"yanlılığın işareti β2 ile ρ'nun işaretlerinin çarpımıdır ({'pozitif' if bias > 0 else 'negatif'})."
        )
    else:
        text += "β2 = 0 ya da ρ = 0: dışarıda bırakılan değişken ya Y'yi etkilemiyor ya da x1 ile ilişkisiz; yanlılık yok."
    return text


FWL = SimExperiment(
    topic_key=TOPIC, number=2,
    title="FWL ve eksik değişken: çoklu regresyon katsayısı neyi ölçer?",
    question="Bir kontrol eklendiğinde katsayı neden değişir ve eklenmezse kısa regresyon neyi tahmin eder?",
    note=NoteRef("2.4", 0),
    parameters=(
        SAMPLE_SIZE,
        SimParameter("rho", "x1 ile x2 arasındaki korelasyon ρ", -0.9, 0.9, 0.6, 0.1, "ρ = 0 iken iki değişken ilişkisiz.", decimals=1),
        SimParameter("beta2", "x2'nin Y'ye etkisi β2", -1.0, 1.0, 0.8, 0.1, "β2 = 0 iken x2 Y'yi etkilemez.", decimals=1),
    ),
    dgp=lambda p: (
        r"x_{2i}\sim N(0,1), \qquad v_i\sim N(0,1), \qquad e_i\sim N(0,1)",
        rf"x_{{1i}}=\rho\,x_{{2i}}+\sqrt{{1-\rho^2}}\,v_i, \qquad \rho={number(p['rho'], 1)}",
        rf"Y_i=1+0{{,}}5\,x_{{1i}}+\beta_2\,x_{{2i}}+e_i, \qquad \beta_2={number(p['beta2'], 1)}",
    ),
    dgp_note=(
        "x1'in Y'ye etkisi 0,5'tir. x1 ile x2 ilişkili (korelasyon ρ) ve x2 de Y'yi etkiliyorsa (β2 ≠ 0), "
        "x2'yi dışarıda bırakan kısa regresyon 0,5'i tahmin etmez."
    ),
    look_at=(
        "**Grafik** — Y ile x1'in x2'den arındırılmış kısmı: x1'i x2 üzerine regres edip artığını alıyoruz.",
        "**Ölçüler** — çoklu regresyon katsayısı, FWL eğimi, kısa regresyon ve kısa regresyonun DGP'den "
        "bilinen hedefi $0{,}5+\\beta_2\\rho$.",
    ),
    build=_build_fwl, metrics=_fwl_metrics, takeaway=_fwl_takeaway,
)


# --- Deney 3: kümelenmiş veri ------------------------------------------------------

def _build_cluster(p: Parameters) -> tuple:
    groups, rho = int(p["groups"]), round(p["rho"], 6)
    return (
        NewSample(FRAME, groups * CLUSTER_SIZE, SEED),
        Derive(FRAME, "kume", E.add(E.floor(E.div(E.sub(E.var("id"), 1), CLUSTER_SIZE)), 1),
               f"Küme numarası: her kümede {CLUSTER_SIZE} gözlem"),
        ClusterDraw(FRAME, "x", "kume", groups, 0, 1, "Küme düzeyinde açıklayıcı değişken (ör. sınıf büyüklüğü)"),
        ClusterDraw(FRAME, "u", "kume", groups, 0, 1, "Küme şoku"),
        Draw(FRAME, "eps", "normal", 0, 1, "Bireysel şok"),
        Derive(FRAME, "e", E.add(E.mul(round(float(np.sqrt(rho)), 6), E.var("u")),
                                 E.mul(round(float(np.sqrt(1 - rho)), 6), E.var("eps"))),
               "Hata: küme içi korelasyonu ρ, varyansı 1"),
        Derive(FRAME, "y", E.add(E.add(1, E.mul(0.3, E.var("x"))), E.var("e")), "Y = 1 + 0,3 x + e"),
        OLS("klasik", FRAME, "y", ("x",)),
        OLS("hc1", FRAME, "y", ("x",), vcov="HC1"),
        OLS("kume_sh", FRAME, "y", ("x",), vcov="cluster", cluster="kume"),
        Plot(FRAME, "x", (Scatter("y", "Gözlemler"), MeanPoints("y", "Küme ortalamaları"), ModelLine("klasik", "OLS doğrusu")),
             "x (küme düzeyinde)", "Y", "Her kümede tek x değeri: bağımsız bilgi küme sayısı kadar"),
    )


def _moulton(p: Parameters) -> float:
    return float(np.sqrt(1 + (CLUSTER_SIZE - 1) * p["rho"]))


def _cluster_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    classic = float(state.models["klasik"].bse["x"])
    robust = float(state.models["hc1"].bse["x"])
    cluster = float(state.models["kume_sh"].bse["x"])
    return (
        SimMetric("Klasik SH", plain(classic), "Gözlemleri bağımsız sayar."),
        SimMetric("HC1 SH", plain(robust), "Heteroskedastisiteye dayanıklı, küme bağımlılığına değil."),
        SimMetric("Küme SH", plain(cluster), "Küme içi bağımlılığa dayanıklı."),
        SimMetric("Küme SH / klasik SH", plain(cluster / classic, 2), "Tahmin edilen şişme oranı."),
        SimMetric("Moulton çarpanı (DGP)", plain(_moulton(p), 2), f"√(1 + (m−1)ρ), m = {CLUSTER_SIZE}."),
    )


def _cluster_takeaway(state: LabState, p: Parameters) -> str:
    text = (
        f"{int(p['groups']) * CLUSTER_SIZE} gözlem var ama x küme düzeyinde değiştiği için bağımsız bilgi "
        f"{int(p['groups'])} küme kadardır. "
    )
    if p["rho"] > 0:
        text += (
            "Hata küme içinde ilişkiliyken klasik ve HC1 standart hataları belirsizliği küçümser; küme SH'nin "
            "klasiğe oranı DGP'den bilinen Moulton çarpanına yakındır. HC1 bu sorunu çözmez: "
            "heteroskedastisiteye karşı dayanıklıdır, küme içi bağımlılığa karşı değil."
        )
    else:
        text += (
            "ρ = 0 iken hatalar bağımsızdır ve üç standart hata ortalamada aynıdır. Yine de küme SH tek bir "
            "örneklemde klasikten belirgin biçimde sapabilir: küme SH bağımsız bilgiyi küme sayısından alır, "
            "az sayıda kümeyle gürültülüdür. Küme sayısını artırıp farkın küçüldüğüne bakın."
        )
    return text


CLUSTER = SimExperiment(
    topic_key=TOPIC, number=3,
    title="Kümelenmiş veri: gözlem sayısı neden yanıltabilir?",
    question="Aynı kümedeki gözlemlerin hataları ilişkiliyse standart hata ne kadar yanılır?",
    note=NoteRef("2.7", 0),
    parameters=(
        SimParameter("groups", "Küme sayısı G", 10, 80, 40, 10, "Her kümede 25 gözlem vardır.", integer=True, decimals=0),
        SimParameter("rho", "Hatanın küme içi korelasyonu ρ", 0.0, 0.8, 0.3, 0.1, "ρ = 0 iken hatalar bağımsız.", decimals=1),
    ),
    dgp=lambda p: (
        rf"g=1,\dots,G, \quad G={int(p['groups'])}, \quad m={CLUSTER_SIZE}\ \text{{gözlem/küme}}",
        r"X_{ig}=x_g,\quad x_g\sim N(0,1), \qquad e_{ig}=\sqrt{\rho}\,u_g+\sqrt{1-\rho}\,\varepsilon_{ig}",
        rf"Y_{{ig}}=1+0{{,}}3\,X_{{ig}}+e_{{ig}}, \qquad \rho={number(p['rho'], 1)}",
    ),
    dgp_note=(
        "x küme düzeyinde (ör. sınıf büyüklüğü), hata ise küme şoku u ile bireysel şok ε'nin karışımı. "
        "Eşit küme büyüklüğünde standart hatanın şişme oranı DGP'den bilinir: √(1 + (m−1)ρ)."
    ),
    look_at=(
        "**Grafik** — her kümede tek bir x değeri: noktalar dikey sütunlar halinde, küme ortalamaları "
        "sütunların merkezinde.",
        "**Ölçüler** — klasik, HC1 ve küme standart hataları; küme SH'nin klasiğe oranı ve DGP'den "
        "bilinen Moulton çarpanı.",
    ),
    build=_build_cluster, metrics=_cluster_metrics, takeaway=_cluster_takeaway,
)


KONU02_EXPERIMENTS = (HETERO, FWL, CLUSTER)
