"""Konu 7 Sezgi deneyleri: kantil doğruları, sağlamlık ve kuyruklarda belirsizlik.

Deney 1  Konum kayması mı, ölçek değişimi mi? Kantil doğruları ve gerçek kantiller  (Notlar §7.8)
Deney 2  Kalın kuyruklu hata: OLS ve LAD (medyan regresyonu), Monte Carlo           (Notlar §7.2)
Deney 3  Kuyruk kantillerinde örnekleme belirsizliği, Monte Carlo                  (Notlar §7.11)

Kantil tahminleri doğrusal programlamanın kesin çözümüdür (Python'da HiGHS, R'de ``rq``,
Stata'da ``qreg``).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from core.labs import expr as E
from core.labs.runner import LabState
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, number, plain
from core.labs.spec import (
    OLS,
    Curve,
    Derive,
    Draw,
    Histogram,
    MonteCarlo,
    NewSample,
    NoteRef,
    Plot,
    QuantileRegression,
    Scatter,
)

SEED = 807
MC_SEED = 707
FRAME = "sim"
TOPIC = "konu07"
REPS = 300
TAUS = ((0.10, "q10"), (0.50, "q50"), (0.90, "q90"))


def _line(model: str) -> E.Expr:
    return E.add(E.coef(model, E.INTERCEPT), E.mul(E.coef(model, "x"), E.var("x")))


def _tau_text(tau: float) -> str:
    return f"{tau:.2f}".replace(".", ",")


# --- Deney 1: konum kayması ve ölçek değişimi ---------------------------------------------------

def true_slope(tau: float, gamma: float) -> float:
    """Y = 1 + X + (1 + γX)e, e ~ N(0,1) için Q_τ(Y|X) = 1 + z_τ + (1 + γz_τ)X: eğim 1 + γz_τ."""

    return 1.0 + gamma * float(stats.norm.ppf(tau))


def _build_location_scale(p: Parameters) -> tuple:
    gamma = round(p["gamma"], 6)
    truths = []
    for index, (tau, _) in enumerate(TAUS):
        z = float(stats.norm.ppf(tau))
        truth = E.add(round(1 + z, 8), E.mul(round(1 + gamma * z, 8), E.var("x")))
        truths.append((index, tau, truth))
    layers = [Scatter("y", "Gözlemler")]
    for (index, tau, truth), (_, model) in zip(truths, TAUS):
        layers.append(Curve(_line(model), f"τ = {_tau_text(tau)}: kantil regresyonu", color=index))
        layers.append(Curve(truth, f"τ = {_tau_text(tau)}: gerçek koşullu kantil", dashed=True, color=index))
    return (
        NewSample(FRAME, int(p["n"]), SEED),
        Draw(FRAME, "x", "uniform", 0, 4, "X ~ U[0, 4]"),
        Draw(FRAME, "e", "normal", 0, 1, "e ~ N(0, 1), X'ten bağımsız"),
        Derive(FRAME, "y", E.add(E.add(1, E.var("x")), E.mul(E.add(1, E.mul(gamma, E.var("x"))), E.var("e"))),
               "Y = 1 + X + (1 + γX)·e: γ > 0 iken yayılım X ile artar"),
        *(QuantileRegression(model, FRAME, "y", ("x",), tau) for tau, model in TAUS),
        OLS("ols", FRAME, "y", ("x",)),
        Plot(FRAME, "x", tuple(layers), "X", "Y", "Kantil doğruları ve gerçek koşullu kantiller"),
    )


def _location_scale_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    metrics = []
    for tau, model in TAUS:
        estimate = float(state.models[model].params["x"])
        metrics.append(
            SimMetric(f"τ = {_tau_text(tau)} eğimi", plain(estimate, 3),
                      f"Gerçek eğim 1 + γz_τ = {plain(true_slope(tau, p['gamma']), 3)}.")
        )
    metrics.append(SimMetric("OLS eğimi", plain(float(state.models["ols"].params["x"]), 3),
                             "Koşullu ortalamanın eğimi; gerçek değer 1."))
    return tuple(metrics)


def _location_scale_takeaway(state: LabState, p: Parameters) -> str:
    low = float(state.models["q10"].params["x"])
    high = float(state.models["q90"].params["x"])
    if p["gamma"] < 1e-9:
        return (
            f"γ = 0: hata X'ten bağımsız, koşullu dağılım X ile yalnız yukarı kayar. Kantil eğimleri ({plain(low, 3)} ve "
            f"{plain(high, 3)}) aynı gerçek değer 1'in etrafında: doğrular paraleldir, yalnız sabitleri farklıdır. Bu "
            "durumda kantil regresyonu OLS'in söylediğinden fazlasını söylemez (§7.8)."
        )
    return (
        f"γ = {plain(p['gamma'], 2)}: koşullu yayılım X ile büyüyor. Üst kantil eğimi {plain(high, 3)}, alt kantil eğimi "
        f"{plain(low, 3)}; gerçek değerler {plain(true_slope(0.9, p['gamma']), 3)} ve {plain(true_slope(0.1, p['gamma']), 3)}. "
        "Kantil doğruları yelpaze gibi açılır; OLS ve medyan eğimi ise 1 civarında kalır. Farklı kantil eğimleri, "
        "X'in koşullu dağılımın şeklini de değiştirdiğinin işaretidir; bireylerin \"eğitimden farklı fayda "
        "gördüğünün\" kanıtı değildir (§7.13)."
    )


LOCATION_SCALE = SimExperiment(
    topic_key=TOPIC, number=1,
    title="Konum kayması mı, ölçek değişimi mi?",
    question="X koşullu dağılımın yalnız konumunu mu, yoksa yayılımını da mı değiştiriyor? Kantil doğruları bunu nasıl gösterir?",
    note=NoteRef("7.8", 0),
    parameters=(
        SimParameter("gamma", "Ölçek eğimi γ", 0.0, 1.0, 0.5, 0.05,
                     "Hatanın standart sapması 1 + γX: γ = 0 iken homoskedastik (konum kayması modeli).", decimals=2),
        SimParameter("n", "Gözlem sayısı", 500, 5000, 2000, 500, "Örneklem büyüklüğü.", integer=True, decimals=0),
    ),
    dgp=lambda p: (
        r"X\sim U[0,4], \qquad e\sim N(0,1)\ \text{bağımsız}",
        rf"Y=1+X+(1+\gamma X)\,e, \qquad \gamma={number(p['gamma'], 2)}",
        r"Q_\tau(Y\mid X)=1+z_\tau+(1+\gamma z_\tau)X, \qquad z_\tau=\Phi^{-1}(\tau)",
    ),
    dgp_note=(
        "Gerçek koşullu kantiller X'te doğrusaldır; eğimleri 1 + γz_τ. γ = 0 iken bütün kantil eğimleri 1'dir (konum "
        "kayması). Koşullu ortalama ve medyan her γ için 1 + X'tir. Tohum 807."
    ),
    look_at=(
        "**Grafik** — gözlemler, τ = 0,10; 0,50; 0,90 kantil regresyonu doğruları (düz) ve aynı renkte gerçek koşullu "
        "kantiller (kesikli).",
        "**Ölçüler** — üç kantil eğimi ve OLS eğimi; gerçek değerler açıklamada.",
    ),
    build=_build_location_scale, metrics=_location_scale_metrics, takeaway=_location_scale_takeaway,
)


# --- Deney 2: kalın kuyruklu hata, OLS ve LAD ---------------------------------------------------

def _contaminated_body(n: int, share: float, scale: float) -> tuple:
    share, scale = round(share, 6), round(scale, 6)
    return (
        NewSample(FRAME, n, None),
        Draw(FRAME, "x", "normal", 0, 1, "X ~ N(0, 1)"),
        Draw(FRAME, "z1", "normal", 0, 1, "Olağan şok N(0, 1)"),
        Draw(FRAME, "z2", "normal", 0, scale, "Uç şok N(0, κ²)"),
        Draw(FRAME, "u", "uniform", 0, 1, "Hangi şokun geleceğini belirleyen tek düze çekiliş"),
        Derive(FRAME, "uc", E.compare("lt", E.var("u"), share), "Uç şok göstergesi: olasılık ε"),
        Derive(FRAME, "e", E.add(E.var("z1"), E.mul(E.var("uc"), E.sub(E.var("z2"), E.var("z1")))),
               "Karışık normal hata: 1 − ε olasılıkla N(0, 1), ε olasılıkla N(0, κ²); simetrik"),
        Derive(FRAME, "y", E.add(E.add(1, E.mul(2, E.var("x"))), E.var("e")), "Y = 1 + 2X + e (gerçek eğim 2)"),
        OLS("ols", FRAME, "y", ("x",)),
        QuantileRegression("lad", FRAME, "y", ("x",), 0.5),
    )


def _build_contamination(p: Parameters) -> tuple:
    return (
        MonteCarlo(
            FRAME, REPS, MC_SEED, _contaminated_body(int(p["n"]), p["share"], p["scale"]),
            (("b_ols", E.coef("ols", "x")), ("b_lad", E.coef("lad", "x"))),
            "mc",
            "Monte Carlo: her tekrarda yeni örneklem; OLS ve LAD eğimi",
        ),
        Histogram(
            "mc", (("b_lad", "LAD (medyan regresyonu)"), ("b_ols", "OLS")), 1.0, 3.0, ((2.0, "Gerçek eğim 2"),),
            "X eğimi tahmini", f"{REPS} örneklemde eğim tahminlerinin dağılımı", bins=50,
        ),
    )


def _contamination_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    table = state.tables["mc"]
    ols_sd, lad_sd = float(table["b_ols"].std()), float(table["b_lad"].std())
    return (
        SimMetric("OLS: ortalama", plain(float(table["b_ols"].mean()), 3), "Gerçek eğim 2."),
        SimMetric("OLS: std. sapma", plain(ols_sd, 3), "Tekrarlar arası dağılım."),
        SimMetric("LAD: ortalama", plain(float(table["b_lad"].mean()), 3), "Gerçek eğim 2."),
        SimMetric("LAD: std. sapma", plain(lad_sd, 3), "Tekrarlar arası dağılım."),
        SimMetric("SS(LAD) / SS(OLS)", plain(lad_sd / ols_sd, 2),
                  "1'den küçükse LAD daha kesin. Normal hatada kuramsal değer √(π/2) ≈ 1,25."),
    )


def _contamination_takeaway(state: LabState, p: Parameters) -> str:
    table = state.tables["mc"]
    ratio = float(table["b_lad"].std()) / float(table["b_ols"].std())
    if p["share"] < 1e-9 or p["scale"] <= 1.0 + 1e-9:
        return (
            f"Uç şok yok: hata normal. İki tahmin edici de 2'yi hedefler; LAD'ın standart sapması OLS'inkinin "
            f"{plain(ratio, 2)} katı (kuram: √(π/2) ≈ 1,25). Normal hatada OLS daha kesindir. ε ve κ'yı artırın."
        )
    verdict = "daha kesin" if ratio < 1 else "henüz daha kesin değil"
    return (
        f"Her gözlem ε = {plain(p['share'], 2)} olasılıkla κ = {plain(p['scale'], 0)} ölçekli uç şok alıyor. Hata "
        f"simetrik olduğu için ortalama ve medyan aynıdır: iki tahmin edici de 2'yi hedefler. Ancak LAD'ın standart "
        f"sapması OLS'inkinin {plain(ratio, 2)} katı; LAD {verdict}. Kareli kayıpta uç artıklar tahmini sürükler, "
        "mutlak kayıpta etkileri sınırlıdır (§7.2). Hata çarpık olsaydı iki yöntem farklı hedefleri (ortalama ve "
        "medyan) tahmin ederdi: sağlamlık, estimand seçiminin yerine geçmez."
    )


CONTAMINATION = SimExperiment(
    topic_key=TOPIC, number=2,
    title="Kalın kuyruklu hata: OLS ve LAD",
    question="Hatanın küçük bir kısmı çok büyük olduğunda OLS ve medyan regresyonunun örnekleme dağılımları nasıl değişir?",
    note=NoteRef("7.2", 0),
    parameters=(
        SimParameter("share", "Uç şok payı ε", 0.0, 0.3, 0.1, 0.02, "Hatanın uç dağılımdan gelme olasılığı.",
                     decimals=2),
        SimParameter("scale", "Uç şokun ölçeği κ", 1.0, 20.0, 10.0, 1.0, "Uç şokun standart sapması.", decimals=0),
        SimParameter("n", "Her örneklemdeki gözlem sayısı", 100, 1000, 200, 100, "Örneklem büyüklüğü.",
                     integer=True, decimals=0),
    ),
    dgp=lambda p: (
        r"X\sim N(0,1), \qquad Y=1+2X+e",
        rf"e\sim(1-\varepsilon)\,N(0,1)+\varepsilon\,N(0,\kappa^2), \qquad \varepsilon={number(p['share'], 2)}, \ \kappa={number(p['scale'], 0)}",
        r"\mathbb E[e\mid X]=\operatorname{med}(e\mid X)=0 \ \Rightarrow\ \text{OLS ve LAD aynı eğimi (2) hedefler}",
    ),
    dgp_note=(
        f"{REPS} tekrarın her birinde yeni örneklem çekilir; OLS ve LAD (τ = 0,5 kantil regresyonu) eğimleri toplanır. "
        "Karışık normal hata simetriktir: ortalama ile medyan çakışır, fark yalnız kesinliktedir."
    ),
    look_at=(
        "**Grafik** — OLS ve LAD eğim tahminlerinin dağılımı; gerçek eğim 2 (dikey çizgi). [1; 3] dışındaki tahminler "
        "çizilmez.",
        "**Ölçüler** — iki tahmin edicinin ortalaması ve standart sapması; standart sapmaların oranı.",
    ),
    build=_build_contamination, metrics=_contamination_metrics, takeaway=_contamination_takeaway,
)


# --- Deney 3: kuyruk kantillerinde belirsizlik -------------------------------------------------

def theoretical_ratio(tau: float) -> float:
    """Normal hatada √{τ(1−τ)}/φ(z_τ) oranı, medyandakine göre: kuyruk kantilinin göreli standart sapması."""

    z = float(stats.norm.ppf(tau))
    return float(np.sqrt(tau * (1 - tau)) / stats.norm.pdf(z) / (0.5 / stats.norm.pdf(0.0)))


def _tail_body(n: int, tau: float) -> tuple:
    return (
        NewSample(FRAME, n, None),
        Draw(FRAME, "x", "uniform", 0, 1, "X ~ U[0, 1]"),
        Draw(FRAME, "e", "normal", 0, 1, "e ~ N(0, 1), X'ten bağımsız"),
        Derive(FRAME, "y", E.add(E.add(1, E.mul(2, E.var("x"))), E.var("e")),
               "Y = 1 + 2X + e: bütün kantil eğimleri 2"),
        QuantileRegression("medyan", FRAME, "y", ("x",), 0.5),
        QuantileRegression("kuyruk", FRAME, "y", ("x",), tau),
    )


def _build_tail(p: Parameters) -> tuple:
    tau = round(p["tau"], 2)
    return (
        MonteCarlo(
            FRAME, REPS, MC_SEED, _tail_body(int(p["n"]), tau),
            (("b_medyan", E.coef("medyan", "x")), ("b_kuyruk", E.coef("kuyruk", "x"))),
            "mc",
            f"Monte Carlo: her tekrarda yeni örneklem; τ = 0,5 ve τ = {_tau_text(tau)} kantil eğimleri",
        ),
        Histogram(
            "mc", (("b_medyan", "τ = 0,50"), ("b_kuyruk", f"τ = {_tau_text(tau)}")), 0.0, 4.0,
            ((2.0, "Gerçek eğim 2"),),
            "X eğimi tahmini", f"{REPS} örneklemde iki kantil eğiminin dağılımı", bins=50,
        ),
    )


def _tail_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    table = state.tables["mc"]
    median_sd, tail_sd = float(table["b_medyan"].std()), float(table["b_kuyruk"].std())
    tau = round(p["tau"], 2)
    return (
        SimMetric("τ = 0,50: std. sapma", plain(median_sd, 3), f"Ortalama {plain(float(table['b_medyan'].mean()), 3)}."),
        SimMetric(f"τ = {_tau_text(tau)}: std. sapma", plain(tail_sd, 3),
                  f"Ortalama {plain(float(table['b_kuyruk'].mean()), 3)}."),
        SimMetric("Oran (Monte Carlo)", plain(tail_sd / median_sd, 2), "Kuyruk kantilinin göreli standart sapması."),
        SimMetric("Oran (kuram)", plain(theoretical_ratio(tau), 2),
                  "√{τ(1−τ)}/φ(z_τ) ÷ √{0,25}/φ(0): normal hatada asimptotik değer."),
    )


def _tail_takeaway(state: LabState, p: Parameters) -> str:
    table = state.tables["mc"]
    ratio = float(table["b_kuyruk"].std()) / float(table["b_medyan"].std())
    tau = round(p["tau"], 2)
    return (
        f"İki kantilde de gerçek eğim 2 ve iki tahmin edici de onu hedefliyor; ama τ = {_tau_text(tau)} eğiminin standart "
        f"sapması medyandakinin {plain(ratio, 2)} katı (kuram {plain(theoretical_ratio(tau), 2)}). Varyans "
        "τ(1−τ)/f(Q_τ)² ile orantılıdır: kuyrukta koşullu yoğunluk f düşüktür, hedef kantilin çevresinde az gözlem vardır. "
        "Uç kantillerde daha \"dramatik\" katsayılar görmek, orada daha çok bilgi olduğu anlamına gelmez (§7.11, §7.16.5)."
    )


TAIL = SimExperiment(
    topic_key=TOPIC, number=3,
    title="Kuyruk kantillerinde örnekleme belirsizliği",
    question="Aynı gerçek eğimi hedefleyen iki kantil regresyonundan hangisi daha kesin: medyan mı, üst kuyruk mu?",
    note=NoteRef("7.11", 0),
    parameters=(
        SimParameter("tau", "Üst kantil τ", 0.60, 0.95, 0.90, 0.05, "Medyanla karşılaştırılan kantil.", decimals=2),
        SimParameter("n", "Her örneklemdeki gözlem sayısı", 200, 2000, 500, 100, "Örneklem büyüklüğü.",
                     integer=True, decimals=0),
    ),
    dgp=lambda p: (
        r"X\sim U[0,1], \qquad e\sim N(0,1)\ \text{bağımsız}, \qquad Y=1+2X+e",
        rf"Q_\tau(Y\mid X)=1+z_\tau+2X \ \text{{her }}\tau\text{{ için}}, \qquad \tau={number(p['tau'], 2)}",
        r"\operatorname{Var}(\hat\beta_\tau)\ \propto\ \tau(1-\tau)\,/\,f_e(z_\tau)^2",
    ),
    dgp_note=(
        f"{REPS} tekrarın her birinde yeni örneklem çekilir; aynı veride τ = 0,5 ve seçilen üst kantil tahmin edilir. "
        "Hata homoskedastik olduğu için bütün kantil eğimleri 2'dir: fark yalnız kesinliktedir."
    ),
    look_at=(
        "**Grafik** — iki kantil eğiminin örnekleme dağılımı; gerçek eğim 2 (dikey çizgi).",
        "**Ölçüler** — iki standart sapma ve oranları; kuramsal oran normal hata için asimptotik değerdir.",
    ),
    build=_build_tail, metrics=_tail_metrics, takeaway=_tail_takeaway,
)


KONU07_EXPERIMENTS = (LOCATION_SCALE, CONTAMINATION, TAIL)
