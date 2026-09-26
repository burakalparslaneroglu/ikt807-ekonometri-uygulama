"""Konu 9 Sezgi deneyleri: yerel doğrusal RDD, bant genişliği ve bulanık tasarım.

Deney 1  Yerel doğrusal ve global polinom: eşikteki sıçrama                  (Notlar §9.4, §9.11)
Deney 2  Bant genişliği: yanlılık, varyans ve kapsama (Monte Carlo)          (Notlar §9.6, §9.5)
Deney 3  Bulanık RDD ve zayıf ilk aşama (Monte Carlo)                        (Notlar §9.12)

Yerel doğrusal tahmin uygulamadaki gibi Hansen ölçeğinde birim varyanslı üçgen çekirdekle yapılır:
h çekirdeğin standart sapmasıdır, pencere ±h√6.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from core.labs import expr as E
from core.labs.runner import LabState, coverage_key
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, number, plain
from core.labs.spec import (
    OLS,
    RDD,
    Curve,
    Derive,
    Draw,
    Histogram,
    MonteCarlo,
    NewSample,
    NoteRef,
    Plot,
    RDDCurve,
    Scatter,
    VLine,
)

SEED = 809
MC_SEED = 909
FRAME = "sim"
TOPIC = "konu09"
REPS = 300
TAU = 1.0
SIGMA = 0.5


def _window(h: float) -> str:
    return plain(h * np.sqrt(6.0), 2)


# --- Deney 1: yerel doğrusal ve global polinom --------------------------------------------------

STEP_LOCATION, STEP_HEIGHT, STEP_SLOPE = 0.5, 1.5, 20.0
GLOBAL_SIGMA = 0.3


def true_untreated_mean(x) -> np.ndarray:
    """m₀(x) = 0,5x + 1,5·Λ(20(x − 0,5)): eşikten uzakta, sağ tarafta dik bir yükseliş."""

    x = np.asarray(x, dtype=float)
    return 0.5 * x + STEP_HEIGHT * expit(STEP_SLOPE * (x - STEP_LOCATION))


def _untreated(x: E.Expr) -> E.Expr:
    return E.add(E.mul(0.5, x), E.mul(STEP_HEIGHT, E.logistic(E.mul(STEP_SLOPE, E.sub(x, STEP_LOCATION)))))


def _global_terms(degree: int) -> tuple[str, ...]:
    powers = tuple("x" if k == 1 else f"x{k}" for k in range(1, degree + 1))
    return ("d", *powers, *(f"d{name}" for name in powers))


def _global_fit(degree: int) -> E.Expr:
    """Global polinom uyumu: eşiğin iki yanında ayrı p. dereceden polinom (grafik eğrisi)."""

    x = E.var("x")
    side = E.compare("ge", x, 0)
    fit: E.Expr = E.add(E.coef("polinom", E.INTERCEPT), E.mul(E.coef("polinom", "d"), side))
    for k in range(1, degree + 1):
        name = "x" if k == 1 else f"x{k}"
        term = x if k == 1 else E.power(x, k)
        fit = E.add(fit, E.mul(E.coef("polinom", name), term))
        fit = E.add(fit, E.mul(E.mul(E.coef("polinom", f"d{name}"), side), term))
    return fit


MC_FRAME = "mc_ornek"


def _sample(frame: str, n: int, seed: int | None, h: float, degree: int, suffix: str = "") -> tuple:
    """Bir örneklem: veri, yerel doğrusal sıçrama ve global polinom (Monte Carlo'da aynı adımlar)."""

    x = E.var("x")
    derived = []
    for k in range(2, degree + 1):
        derived.append(Derive(frame, f"x{k}", E.power(x, k), f"X^{k}"))
    for k in range(1, degree + 1):
        name = "x" if k == 1 else f"x{k}"
        derived.append(Derive(frame, f"d{name}", E.mul(E.var("d"), E.var(name)), f"D·X^{k}: sağ tarafın ayrı polinomu"))
    return (
        NewSample(frame, n, seed),
        Draw(frame, "x", "uniform", -1, 1, "Eşik değişkeni X ~ U[−1, 1], eşik c = 0"),
        Draw(frame, "e", "normal", 0, GLOBAL_SIGMA, "e ~ N(0; 0,3²)"),
        Derive(frame, "d", E.compare("ge", x, 0), "Keskin tasarım: D = 1{X ≥ 0}"),
        Derive(frame, "y", E.add(E.add(_untreated(x), E.mul(TAU, E.var("d"))), E.var("e")),
               "Y = m0(X) + τD + e, m0(x) = 0,5x + 1,5·Λ(20(x − 0,5)), τ = 1"),
        RDD(f"yerel{suffix}", frame, "x", "y", 0, h),
        *derived,
        OLS(f"polinom{suffix}", frame, "y", _global_terms(degree), vcov="HC1"),
    )


def _build_global(p: Parameters) -> tuple:
    h, degree, n = round(p["h"], 6), int(p["p"]), int(p["n"])
    x = E.var("x")
    truth = E.add(_untreated(x), E.mul(TAU, E.compare("ge", x, 0)))
    return (
        *_sample(FRAME, n, SEED, h, degree),
        Plot(
            FRAME, "x",
            (
                Scatter("y", "Gözlemler"),
                Curve(truth, "Gerçek E[Y | X] (sıçrama τ = 1)", dashed=True),
                RDDCurve("y", 0, h, f"Yerel doğrusal, h = {plain(h, 2)} ve %95 bant"),
                Curve(_global_fit(degree), f"Global polinom, p = {degree}"),
                VLine(0, "Eşik c = 0"),
            ),
            "Eşik değişkeni X", "Y", "Bir örneklemde eşikteki sıçrama: yerel doğrusal ve global polinom",
        ),
        MonteCarlo(
            MC_FRAME, REPS, MC_SEED, _sample(MC_FRAME, n, None, h, degree, "_t"),
            (("tau_yerel", E.coef("yerel_t", "D")), ("tau_polinom", E.coef("polinom_t", "d"))), "mc",
            "Monte Carlo: her tekrarda yeni örneklem; aynı iki tahmin edici",
        ),
        Histogram(
            "mc", (("tau_yerel", f"Yerel doğrusal, h = {plain(h, 2)}"), ("tau_polinom", f"Global polinom, p = {degree}")),
            0.0, 2.0, ((TAU, "Gerçek τ = 1"),), "Sıçrama tahmini τ̂", f"{REPS} örneklemde τ̂'nin dağılımı", bins=50,
        ),
    )


def _global_numbers(state: LabState) -> dict[str, float]:
    table = state.tables["mc"]
    return {
        "local_bias": float(table["tau_yerel"].mean() - TAU), "local_sd": float(table["tau_yerel"].std()),
        "global_bias": float(table["tau_polinom"].mean() - TAU), "global_sd": float(table["tau_polinom"].std()),
    }


def _global_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    values = _global_numbers(state)
    return (
        SimMetric("Yerel doğrusal: yanlılık", plain(values["local_bias"], 3), f"{REPS} örneklemde τ̂ − 1 ortalaması."),
        SimMetric("Yerel doğrusal: std. sapma", plain(values["local_sd"], 3), "Tekrarlar arası."),
        SimMetric("Global polinom: yanlılık", plain(values["global_bias"], 3), f"p = {int(p['p'])}."),
        SimMetric("Global polinom: std. sapma", plain(values["global_sd"], 3), "Tekrarlar arası."),
    )


def _global_takeaway(state: LabState, p: Parameters) -> str:
    values = _global_numbers(state)
    local, polynomial = state.models["yerel"], state.models["polinom"]
    reach = p["h"] * np.sqrt(6.0) > STEP_LOCATION
    text = (
        f"{REPS} örneklemde yerel doğrusal tahminin yanlılığı {plain(values['local_bias'], 3)} (std. sapma "
        f"{plain(values['local_sd'], 3)}), global p = {int(p['p'])} polinomunun yanlılığı "
        f"{plain(values['global_bias'], 3)} (std. sapma {plain(values['global_sd'], 3)}). Global polinom eşikteki "
        "değeri bütün veriden öğrenir: sağ taraftaki x = 0,5 civarındaki dik yükseliş eşiğin hemen sağındaki uyumu da "
        "büker; derece arttıkça uçlardaki salınım ve varyans büyür (Gelman ve Imbens 2019; §9.4, §9.11). Grafikteki tek "
        f"örneklemde tahminler {plain(local.jump, 3)} (yerel, pencerede {local.nobs} gözlem) ve "
        f"{plain(float(polynomial.params['d']), 3)} (global): tek örneklem gürültülüdür, fark tekrarların dağılımında "
        "görünür. "
    )
    if reach:
        text += (
            f"Pencere ±{_window(p['h'])} artık x = 0,5'teki yükselişe ulaşıyor: yerel doğrusal tahmin de yanlılaşır. "
            "Yerellik bu yüzden bir model kararıdır."
        )
    else:
        text += (
            "Yerel doğrusal tahmin yalnız eşiğe yakın gözlemleri kullanır; uzaktaki yapı sıçrama tahminine karışmaz. "
            "h'yi 0,2'nin üstüne çıkarın ve pencere yükselişe ulaştığında ne olduğuna bakın."
        )
    return text


GLOBAL = SimExperiment(
    topic_key=TOPIC, number=1,
    title="Yerel doğrusal ve global polinom",
    question=(
        "Eşikteki sıçramayı eşiğe yakın gözlemlerle yerel doğrusal tahmin etmek ile bütün veriye yüksek dereceli "
        "global polinom uydurmak neden farklı sonuç verir?"
    ),
    note=NoteRef("9.4", 0, ("§9.11",)),
    parameters=(
        SimParameter("h", "Bant genişliği h", 0.05, 0.4, 0.15, 0.05,
                     "Hansen ölçeği: üçgen çekirdekte pencere ±h√6.", decimals=2),
        SimParameter("p", "Global polinom derecesi p", 1, 5, 4, 1, "Eşiğin iki yanında ayrı p. dereceden polinom.",
                     integer=True, decimals=0),
        SimParameter("n", "Gözlem sayısı", 500, 5000, 1000, 500, "Örneklem büyüklüğü.", integer=True, decimals=0),
    ),
    dgp=lambda p: (
        r"X\sim U[-1,1],\qquad D=\mathbf 1\{X\ge 0\},\qquad e\sim N(0,\,0{,}3^2)",
        r"Y=m_0(X)+\tau D+e,\qquad m_0(x)=0{,}5x+1{,}5\,\Lambda\bigl(20(x-0{,}5)\bigr),\qquad \tau=1",
        rf"h={number(p['h'], 2)}\ (\text{{pencere}}\ \pm h\sqrt6={number(p['h'] * np.sqrt(6), 2)}),"
        rf"\qquad p={int(p['p'])}",
    ),
    dgp_note=(
        "Tedavi yokken koşullu ortalama m0(x) eşikte düzgündür; sağ tarafta, eşikten uzakta (x = 0,5) dik bir "
        "yükseliş vardır (Λ lojistik fonksiyon). Gerçek sıçrama τ = 1 bilinir. Global polinom her tarafta ayrı "
        f"p. dereceden polinomdur: Y ~ D + X + … + X^p + D·X + … + D·X^p. Grafik tek örneklemdir (tohum 809); "
        f"yanlılık ve standart sapma {REPS} yeni örneklemden (tohum 909)."
    ),
    look_at=(
        "**Grafik 1** — bir örneklemde gözlemler, gerçek koşullu ortalama (kesikli), eşiğin iki yanında yerel doğrusal "
        "tahmin ve %95 bandı, global polinom uyumu.",
        f"**Grafik 2** — {REPS} örneklemde iki tahmin edicinin sıçrama tahminleri; gerçek τ = 1 (dikey çizgi).",
        "**Ölçüler** — iki tahmin edicinin yanlılığı ve standart sapması.",
    ),
    build=_build_global, metrics=_global_metrics, takeaway=_global_takeaway,
)


# --- Deney 2: bant genişliği Monte Carlo ------------------------------------------------------

CURVATURE = 2.0
"""m₀(x) = x + 2x|x|: eşiğin solunda m₀'' = −4, sağında +4."""
BIAS_CONSTANT = -0.1 * 6.0 * 2 * CURVATURE
"""Üçgen çekirdekli tek taraflı yerel doğrusal tahminin sınırdaki yanlılığı (m''/2)·B·W², B = −0,1 (çekirdek
momentlerinden), W = h√6. İki taraf ters işaretli eğrilikle toplanır: yaklaşık yanlılık −2,4h²."""


def approximate_bias(h: float) -> float:
    return BIAS_CONSTANT * h * h


def _bandwidth_body(n: int, h: float) -> tuple:
    x = E.var("x")
    return (
        NewSample(FRAME, n, None),
        Draw(FRAME, "x", "uniform", -1, 1, "X ~ U[−1, 1], eşik c = 0"),
        Draw(FRAME, "e", "normal", 0, SIGMA, "e ~ N(0; 0,5²)"),
        Derive(FRAME, "y", E.add(E.add(E.add(x, E.mul(CURVATURE, E.mul(x, E.absolute(x)))),
                                       E.mul(TAU, E.compare("ge", x, 0))), E.var("e")),
               "Y = X + 2X|X| + τ·1{X ≥ 0} + e, τ = 1"),
        RDD("rdd", FRAME, "x", "y", 0, h),
    )


def _build_bandwidth(p: Parameters) -> tuple:
    h = round(p["h"], 6)
    return (
        MonteCarlo(
            FRAME, REPS, MC_SEED, _bandwidth_body(int(p["n"]), h),
            (("tau", E.coef("rdd", "D")), ("tau_se", E.se("rdd", "D"))), "mc",
            "Monte Carlo: her tekrarda yeni örneklem ve aynı h ile yerel doğrusal sıçrama tahmini",
            coverage=(("tau", "tau_se", TAU),),
        ),
        Histogram("mc", (("tau", f"Yerel doğrusal τ̂, h = {plain(h, 2)}"),), 0.0, 2.0,
                  ((TAU, "Gerçek τ = 1"),), "Sıçrama tahmini τ̂", f"{REPS} örneklemde τ̂'nin dağılımı", bins=50),
    )


def _bandwidth_numbers(state: LabState) -> dict[str, float]:
    table = state.tables["mc"]
    return {
        "bias": float(table["tau"].mean() - TAU),
        "sd": float(table["tau"].std()),
        "se": float(table["tau_se"].mean()),
        "coverage": float(state.scalars[coverage_key("mc", "tau")]),
    }


def _bandwidth_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    values = _bandwidth_numbers(state)
    return (
        SimMetric("Yanlılık (Monte Carlo)", plain(values["bias"], 3), "τ̂'lerin ortalaması − 1."),
        SimMetric("Yaklaşık yanlılık −2,4h²", plain(approximate_bias(p["h"]), 3),
                  "Eğrilik ve çekirdekten hesaplanan birinci derece yanlılık."),
        SimMetric("Std. sapma", plain(values["sd"], 3), "Tekrarlar arası."),
        SimMetric("Ortalama SH", plain(values["se"], 3), "HC1 standart hatalarının ortalaması."),
        SimMetric("%95 GA kapsama", f"%{plain(100 * values['coverage'], 1)}", "τ̂ ± 1,96·SH aralığının τ = 1'i kapsama oranı."),
    )


def _bandwidth_takeaway(state: LabState, p: Parameters) -> str:
    values = _bandwidth_numbers(state)
    ratio = abs(values["bias"]) / values["sd"]
    text = (
        f"h = {plain(p['h'], 2)} ile yanlılık {plain(values['bias'], 3)}, standart sapma {plain(values['sd'], 3)}; "
        f"yanlılık standart sapmanın {plain(ratio, 2)} katı ve kapsama %{plain(100 * values['coverage'], 1)}. "
        "Yanlılık kabaca h² ile, standart sapma 1/√(nh) ile ölçeklenir; oranları √(nh⁵) mertebesindedir (§9.6). "
    )
    if values["coverage"] < 0.9:
        text += (
            "Güven aralığı yanlılığı hesaba katmadığı için nominal %95'in altında kalıyor. h'yi küçültün "
            "(undersmoothing): yanlılık hızla düşer, aralık genişler ama kapsama düzelir."
        )
    else:
        text += (
            "Bu bantta yanlılık standart hataya göre küçük; kapsama %95'e yakın. n'yi büyütün: aynı h'de standart hata "
            "küçülür, yanlılık aynı kalır ve kapsama bozulmaya başlar. Nokta tahmini için iyi bant, çıkarım için "
            "yeterince küçük olmayabilir."
        )
    return text


BANDWIDTH = SimExperiment(
    topic_key=TOPIC, number=2,
    title="Bant genişliği: yanlılık, varyans ve kapsama",
    question=(
        "Bant genişliği büyüdükçe yerel doğrusal sıçrama tahmininin yanlılığı, standart sapması ve güven aralığının "
        "kapsama oranı nasıl değişir?"
    ),
    note=NoteRef("9.6", 0, ("§9.5",)),
    parameters=(
        SimParameter("h", "Bant genişliği h", 0.05, 0.4, 0.15, 0.05,
                     "Hansen ölçeği: üçgen çekirdekte pencere ±h√6.", decimals=2),
        SimParameter("n", "Her örneklemdeki gözlem sayısı", 500, 4000, 1000, 500, "Örneklem büyüklüğü.",
                     integer=True, decimals=0),
    ),
    dgp=lambda p: (
        r"X\sim U[-1,1],\qquad e\sim N(0,\,0{,}5^2),\qquad \tau=1",
        r"Y=X+2X|X|+\tau\,\mathbf 1\{X\ge0\}+e \quad (m_0''=-4\ \text{solda},\ +4\ \text{sağda})",
        rf"h={number(p['h'], 2)},\qquad \text{{yaklaşık yanlılık}}\ -2{{,}}4h^2={number(approximate_bias(p['h']), 3)}",
    ),
    dgp_note=(
        f"{REPS} tekrarın her birinde yeni örneklem çekilir ve sıçrama aynı h ile tahmin edilir. Eğrilik eşiğin iki "
        "yanında ters işaretli olduğu için iki tarafın sınır yanlılıkları farkta birbirini götürmez, toplanır: "
        "birinci derece yanlılık (m''/2)·B·W² farkıdır; üçgen çekirdekte B = −0,1 ve W = h√6, sonuç −2,4h². "
        "Tohum 909."
    ),
    look_at=(
        "**Grafik** — τ̂'lerin dağılımı; gerçek τ = 1 (dikey çizgi).",
        "**Ölçüler** — Monte Carlo yanlılığı ve kuramsal yaklaşığı, standart sapma, ortalama SH ve %95 aralığının "
        "kapsama oranı.",
    ),
    build=_build_bandwidth, metrics=_bandwidth_metrics, takeaway=_bandwidth_takeaway,
)


# --- Deney 3: bulanık RDD ve zayıf ilk aşama -----------------------------------------------------

FUZZY_H = 0.2
BASE_PROBABILITY = 0.3
SELECTION = 1.5


def _fuzzy_body(n: int, jump: float) -> tuple:
    x = E.var("x")
    return (
        NewSample(FRAME, n, None),
        Draw(FRAME, "x", "uniform", -1, 1, "X ~ U[−1, 1], eşik c = 0"),
        Draw(FRAME, "v", "uniform", 0, 1, "V ~ U[0, 1]: tedaviye yatkınlık (araştırmacı gözlemez)"),
        Draw(FRAME, "e", "normal", 0, SIGMA, "e ~ N(0; 0,5²)"),
        Derive(FRAME, "d", E.compare("lt", E.var("v"), E.add(BASE_PROBABILITY, E.mul(jump, E.compare("ge", x, 0)))),
               "Tedavi: D = 1{V < 0,3 + Δ·1{X ≥ 0}}; eşik tedavi olasılığını Δ kadar artırır"),
        Derive(FRAME, "y", E.add(E.add(E.add(E.mul(0.5, x), E.mul(TAU, E.var("d"))),
                                       E.mul(SELECTION, E.sub(0.5, E.var("v")))), E.var("e")),
               "Y = 0,5X + θD + 1,5(0,5 − V) + e, θ = 1: V hem D'yi hem Y'yi etkiler"),
        RDD("rdd_y", FRAME, "x", "y", 0, FUZZY_H),
        RDD("rdd_d", FRAME, "x", "d", 0, FUZZY_H),
        OLS("naif", FRAME, "y", ("d",)),
    )


def _build_fuzzy(p: Parameters) -> tuple:
    jump = round(p["delta"], 6)
    return (
        MonteCarlo(
            FRAME, REPS, MC_SEED, _fuzzy_body(int(p["n"]), jump),
            (
                ("wald", E.div(E.coef("rdd_y", "D"), E.coef("rdd_d", "D"))),
                ("ilk_asama", E.coef("rdd_d", "D")),
                ("ilk_asama_f", E.power(E.div(E.coef("rdd_d", "D"), E.se("rdd_d", "D")), 2)),
                ("naif", E.coef("naif", "d")),
            ),
            "mc",
            "Monte Carlo: her tekrarda yeni örneklem; sonuç ve tedavi sıçramaları aynı h ile, Wald oranı ve naif OLS",
        ),
        Histogram("mc", (("wald", "Yerel Wald oranı θ̂ = τ̂_Y / τ̂_D"),), -3.0, 5.0, ((TAU, "Gerçek θ = 1"),),
                  "Tedavi etkisi tahmini θ̂", f"{REPS} örneklemde yerel Wald oranının dağılımı", bins=80),
    )


def _fuzzy_numbers(state: LabState) -> dict[str, float]:
    table = state.tables["mc"]
    wald = table["wald"].to_numpy(dtype=float)
    q10, q25, q50, q75, q90 = np.quantile(wald, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "median": float(q50), "iqr": float(q75 - q25), "q10": float(q10), "q90": float(q90),
        "far": float(np.mean(np.abs(wald - TAU) > 1.0)),
        "first": float(table["ilk_asama"].mean()), "f": float(table["ilk_asama_f"].median()),
        "naive": float(table["naif"].mean()),
    }


def _fuzzy_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    values = _fuzzy_numbers(state)
    return (
        SimMetric("İlk aşama sıçraması", plain(values["first"], 3), f"Tekrarların ortalaması; gerçek Δ = {plain(p['delta'], 2)}."),
        SimMetric("Medyan ilk aşama F", plain(values["f"], 1), "(τ̂_D / SH)²: zayıf araç ölçüsü (Konu 4)."),
        SimMetric("θ̂: medyan", plain(values["median"], 3), "Gerçek θ = 1; oranın ortalaması tanımsız olabilir."),
        SimMetric("θ̂: %10 yüzdelik", plain(values["q10"], 2), "Tekrarların yüzde 10'u bu değerin altında."),
        SimMetric("θ̂: %90 yüzdelik", plain(values["q90"], 2), "Tekrarların yüzde 10'u bu değerin üstünde."),
    )


def _fuzzy_takeaway(state: LabState, p: Parameters) -> str:
    values = _fuzzy_numbers(state)
    text = (
        f"Eşik tedavi olasılığını Δ = {plain(p['delta'], 2)} artırıyor; medyan ilk aşama F {plain(values['f'], 1)}. "
        f"Wald oranının medyanı {plain(values['median'], 3)}, %10–%90 aralığı [{plain(values['q10'], 2)}; "
        f"{plain(values['q90'], 2)}]; tekrarların %{plain(100 * values['far'], 1)}'inde tahmin gerçek değerden 1'den "
        f"fazla sapıyor. Naif OLS ortalamada {plain(values['naive'], 3)} veriyor: V hem tedaviyi hem sonucu etkilediği "
        "için seçim yanlılığı taşır. "
    )
    if values["f"] < 10:
        text += (
            "Payda (ilk aşama sıçraması) küçük ve gürültülü: oran çok oynak, dağılımı kalın kuyruklu ve naif OLS'e doğru "
            "kayık. Bu, zayıf araç sorununun bulanık RDD karşılığıdır; delta yöntemi standart hatası yanıltıcı olur "
            "(§9.12). Tedavi olasılığı grafiği sonuç grafiği kadar raporlanmalıdır."
        )
    else:
        text += (
            "İlk aşama güçlü: oran gerçek θ = 1 çevresinde toplanır. Δ'yı 0,1'e düşürün ve dağılımın nasıl dağıldığına "
            "bakın."
        )
    return text


FUZZY = SimExperiment(
    topic_key=TOPIC, number=3,
    title="Bulanık RDD ve zayıf ilk aşama",
    question="Eşikte tedavi olasılığı az sıçradığında yerel Wald oranı neden çok değişken olur?",
    note=NoteRef("9.12", 0),
    parameters=(
        SimParameter("delta", "İlk aşama sıçraması Δ", 0.05, 0.6, 0.3, 0.05,
                     "Eşikte tedavi olasılığındaki artış.", decimals=2),
        SimParameter("n", "Her örneklemdeki gözlem sayısı", 1000, 5000, 2000, 500, "Örneklem büyüklüğü.",
                     integer=True, decimals=0),
    ),
    dgp=lambda p: (
        r"X\sim U[-1,1],\quad V\sim U[0,1],\quad e\sim N(0,\,0{,}5^2),\quad Z=\mathbf 1\{X\ge0\}",
        rf"D=\mathbf 1\{{V<0{{,}}3+\Delta Z\}},\qquad \Delta={number(p['delta'], 2)}",
        r"Y=0{,}5X+\theta D+1{,}5\,(0{,}5-V)+e,\qquad \theta=1,\qquad "
        r"\hat\theta=\hat\tau_Y/\hat\tau_D\ (h=0{,}2)",
    ),
    dgp_note=(
        f"{REPS} tekrarın her birinde yeni örneklem çekilir. Sonuç sıçraması τ̂_Y ve tedavi sıçraması τ̂_D aynı "
        "yerel doğrusal tahminle (h = 0,2, pencere ±0,49) bulunur; yerel Wald oranı τ̂_Y/τ̂_D'dir. Tedavi etkisi "
        "herkeste aynı (θ = 1) olduğu için oran θ'yı hedefler. Tohum 909."
    ),
    look_at=(
        "**Grafik** — yerel Wald oranının dağılımı; gerçek θ = 1 (dikey çizgi). [−3, 5] dışındaki tekrarlar sayılır.",
        "**Ölçüler** — ilk aşama sıçraması ve F istatistiği; oranın medyanı ve yüzde 10 ile 90 yüzdelikleri. Naif OLS "
        "(Y'nin D üzerine) karşılaştırma için sonuç metnindedir.",
    ),
    build=_build_fuzzy, metrics=_fuzzy_metrics, takeaway=_fuzzy_takeaway,
)


KONU09_EXPERIMENTS = (GLOBAL, BANDWIDTH, FUZZY)
