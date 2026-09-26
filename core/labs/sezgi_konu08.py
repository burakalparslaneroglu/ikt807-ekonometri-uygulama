"""Konu 8 Sezgi deneyleri: bant genişliği, sınır yanlılığı ve kısmen doğrusal model.

Deney 1  Bant genişliği: yanlılık–varyans dengesi ve çapraz doğrulama   (Notlar §8.5, §8.6)
Deney 2  Sınırda Nadaraya–Watson ve yerel doğrusal tahmin                (Notlar §8.3, §8.4)
Deney 3  Kısmen doğrusal model: doğrusal kontrol ve Robinson (Monte Carlo) (Notlar §8.10)

Yerel tahminler Gauss çekirdeğiyle; h çekirdeğin standart sapmasıdır.
"""

from __future__ import annotations

import numpy as np

from core.labs import expr as E
from core.labs import smoothing as S
from core.labs.runner import LabState
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, number, plain
from core.labs.spec import (
    OLS,
    BandwidthCV,
    Curve,
    Derive,
    Draw,
    Histogram,
    LocalCurve,
    LocalResidual,
    MonteCarlo,
    NewSample,
    NoteRef,
    Plot,
    Scatter,
)

SEED = 808
MC_SEED = 708
FRAME = "sim"
TOPIC = "konu08"
REPS = 200


def _h_text(h: float) -> str:
    return plain(h, 2)


# --- Deney 1: bant genişliği ve çapraz doğrulama ---------------------------------------------------

def true_mean_sine(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sin(x)


CV_GRID = (0.1, 3.0, 0.05)


def _build_bandwidth(p: Parameters) -> tuple:
    h, sigma = round(p["h"], 6), round(p["sigma"], 6)
    truth = E.mul(2, E.sin(E.var("x")))
    return (
        NewSample(FRAME, int(p["n"]), SEED),
        Draw(FRAME, "x", "uniform", 0, 10, "X ~ U[0, 10]"),
        Draw(FRAME, "e", "normal", 0, sigma, "e ~ N(0, σ²)"),
        Derive(FRAME, "y", E.add(truth, E.var("e")), "Y = 2·sin(X) + e: gerçek koşullu ortalama m(x) = 2·sin(x)"),
        Plot(FRAME, "x",
             (Scatter("y", "Gözlemler"),
              Curve(truth, "Gerçek m(x) = 2·sin(x)", dashed=True),
              LocalCurve("y", h, f"Yerel doğrusal, h = {_h_text(h)}")),
             "X", "Y", "Yerel doğrusal tahmin ve gerçek koşullu ortalama"),
        BandwidthCV("cv", FRAME, "x", "y", CV_GRID, "cv_tablosu", "Bant genişliği h",
                    "Çapraz doğrulama ölçütü (birini dışarıda bırak)"),
    )


def fit_error(state: LabState, h: float) -> float:
    """Tahmin edilen eğri ile gerçek m(x) arasındaki ortalama kare fark, [0, 10] ızgarasında."""

    frame = state.frames[FRAME]
    grid = np.linspace(0.0, 10.0, 201)
    fitted = S.local_fit(frame["x"], frame["y"], grid, h)
    return float(np.mean((fitted - true_mean_sine(grid)) ** 2))


def _bandwidth_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    chosen = float(state.scalars["cv_h"])
    return (
        SimMetric("Seçtiğiniz h", _h_text(p["h"]), "Çekirdeğin standart sapması."),
        SimMetric("Sapma: sizin h", plain(fit_error(state, p["h"]), 4),
                  "Tahmin edilen eğrinin gerçek m(x)'ten ortalama kare sapması, x ∈ [0, 10]; simülasyonda bilinir."),
        SimMetric("CV ile seçilen h", _h_text(chosen), "Birini dışarıda bırakan CV, h ∈ {0,10; 0,15; …; 3,00}."),
        SimMetric("Sapma: CV h", plain(fit_error(state, chosen), 4),
                  "CV'nin seçtiği bantla aynı ölçü: gerçek m(x)'ten ortalama kare sapma."),
    )


def _bandwidth_takeaway(state: LabState, p: Parameters) -> str:
    chosen = float(state.scalars["cv_h"])
    h = p["h"]
    if h < 0.6 * chosen:
        verdict = "eğri gürültüyü izliyor: yanlılık küçük, varyans büyük"
    elif h > 1.6 * chosen:
        verdict = "eğri tepeleri ve çukurları düzleştiriyor: varyans küçük, yanlılık büyük"
    else:
        verdict = "CV'nin seçimine yakın: yanlılık ile varyans dengede"
    return (
        f"h = {_h_text(h)} ile {verdict}. Çapraz doğrulama gerçek eğriyi bilmeden h = {_h_text(chosen)} seçti; "
        f"bu bantla gerçek eğriden sapma {plain(fit_error(state, chosen), 4)}, sizin bandınızla "
        f"{plain(fit_error(state, h), 4)}. Gerçek veride m(x) bilinmez; CV bu dengeyi dışarıda bırakılan tahmin "
        "hatasıyla kurar (§8.5–8.6). h'yi 0,1'e ve 3'e çekerek iki ucu görün."
    )


BANDWIDTH = SimExperiment(
    topic_key=TOPIC, number=1,
    title="Bant genişliği: yanlılık–varyans dengesi",
    question="Bant genişliği küçüldükçe ve büyüdükçe yerel doğrusal tahmin nasıl değişir; çapraz doğrulama hangi h'yi seçer?",
    note=NoteRef("8.5", 0, ("§8.6",)),
    parameters=(
        SimParameter("h", "Bant genişliği h", 0.1, 3.0, 1.0, 0.05, "Gauss çekirdeğinin standart sapması.",
                     decimals=2),
        SimParameter("sigma", "Hata standart sapması σ", 0.25, 2.0, 1.0, 0.25, "Gürültünün büyüklüğü.", decimals=2),
        SimParameter("n", "Gözlem sayısı", 200, 2000, 500, 100, "Örneklem büyüklüğü.", integer=True, decimals=0),
    ),
    dgp=lambda p: (
        r"X\sim U[0,10], \qquad e\sim N(0,\sigma^2)",
        rf"Y=2\sin(X)+e, \qquad \sigma={number(p['sigma'], 2)}, \ h={number(p['h'], 2)}",
        r"\hat m(x)=\arg\min_{a}\min_{b}\sum_i K\!\left(\tfrac{X_i-x}{h}\right)\{Y_i-a-b(X_i-x)\}^2, \quad K=\phi",
    ),
    dgp_note=(
        "Gerçek koşullu ortalama m(x) = 2·sin(x) bilindiği için tahminin gerçek eğriden sapması doğrudan ölçülebilir. "
        "Çapraz doğrulama h ∈ {0,10; 0,15; …; 3,00} ızgarasında birini dışarıda bırakan ölçütle yapılır. Tohum 808."
    ),
    look_at=(
        "**Grafik** — gözlemler, gerçek m(x) (kesikli) ve seçtiğiniz h ile yerel doğrusal tahmin.",
        "**CV grafiği** — ölçütün en küçük değerinden farkı; dikey çizgi CV'nin seçtiği h.",
        "**Ölçüler** — iki bantla gerçek eğriden ortalama kare sapma.",
    ),
    build=_build_bandwidth, metrics=_bandwidth_metrics, takeaway=_bandwidth_takeaway,
)


# --- Deney 2: sınırda NW ve yerel doğrusal ------------------------------------------------------

def true_line(x) -> np.ndarray:
    return 1.0 + 3.0 * np.asarray(x, dtype=float)


def boundary_errors(state: LabState, h: float) -> dict[str, np.ndarray]:
    frame = state.frames[FRAME]
    points = np.array([0.0, 0.5, 1.0])
    truth = true_line(points)
    return {
        "nw": S.local_fit(frame["x"], frame["y"], points, h, degree=0) - truth,
        "ll": S.local_fit(frame["x"], frame["y"], points, h, degree=1) - truth,
    }


def _build_boundary(p: Parameters) -> tuple:
    h = round(p["h"], 6)
    truth = E.add(1, E.mul(3, E.var("x")))
    return (
        NewSample(FRAME, int(p["n"]), SEED),
        Draw(FRAME, "x", "uniform", 0, 1, "X ~ U[0, 1]: desteğin sınırları 0 ve 1"),
        Draw(FRAME, "e", "normal", 0, 0.5, "e ~ N(0; 0,5²)"),
        Derive(FRAME, "y", E.add(truth, E.var("e")), "Y = 1 + 3X + e: gerçek m(x) doğrusal"),
        Plot(FRAME, "x",
             (Scatter("y", "Gözlemler"),
              Curve(truth, "Gerçek m(x) = 1 + 3x", dashed=True),
              LocalCurve("y", h, f"Nadaraya–Watson (yerel sabit), h = {_h_text(h)}", degree=0),
              LocalCurve("y", h, f"Yerel doğrusal, h = {_h_text(h)}")),
             "X", "Y", "Sınırda yerel sabit ve yerel doğrusal tahmin"),
    )


def _boundary_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    errors = boundary_errors(state, p["h"])
    return (
        SimMetric("NW hatası, x = 0", plain(errors["nw"][0], 3), "m̂(0) − m(0); sol sınır."),
        SimMetric("Yerel doğrusal hatası, x = 0", plain(errors["ll"][0], 3), "m̂(0) − m(0)."),
        SimMetric("NW hatası, x = 0,5", plain(errors["nw"][1], 3), "İç nokta."),
        SimMetric("Yerel doğrusal hatası, x = 0,5", plain(errors["ll"][1], 3), "İç nokta."),
        SimMetric("NW hatası, x = 1", plain(errors["nw"][2], 3), "Sağ sınır."),
    )


def _boundary_takeaway(state: LabState, p: Parameters) -> str:
    errors = boundary_errors(state, p["h"])
    return (
        f"Sol sınırda Nadaraya–Watson {plain(errors['nw'][0], 3)}, sağ sınırda {plain(errors['nw'][2], 3)} hata yapıyor: "
        "sınırda komşuların hepsi tek taraftadır ve yerel ortalama eğimli bir eğriyi içeri doğru çeker. Yerel doğrusal "
        f"tahminin hatası x = 0'da {plain(errors['ll'][0], 3)}: yerel eğim tahmin edildiği için bu birinci derece "
        "yanlılık ortadan kalkar; doğrusal bir m(x)'i herhangi bir h ile yanlılıksız yeniden üretir. İç noktalarda iki "
        "tahmin birbirine yakındır. h'yi büyütün: NW'nin sınır hatası h ile orantılı büyür (§8.4)."
    )


BOUNDARY = SimExperiment(
    topic_key=TOPIC, number=2,
    title="Sınırda Nadaraya–Watson ve yerel doğrusal",
    question="Desteğin sınırında yerel sabit (Nadaraya–Watson) ve yerel doğrusal tahmin neden farklı davranır?",
    note=NoteRef("8.4", 0, ("§8.3",)),
    parameters=(
        SimParameter("h", "Bant genişliği h", 0.02, 0.3, 0.1, 0.02, "Gauss çekirdeğinin standart sapması.",
                     decimals=2),
        SimParameter("n", "Gözlem sayısı", 200, 3000, 1000, 100, "Örneklem büyüklüğü.", integer=True, decimals=0),
    ),
    dgp=lambda p: (
        r"X\sim U[0,1], \qquad e\sim N(0,\,0{,}5^2)",
        rf"Y=1+3X+e, \qquad h={number(p['h'], 2)}",
        r"\text{NW: } \hat m(x)=\frac{\sum_i K_iY_i}{\sum_i K_i}, \qquad \text{yerel doğrusal: } \hat m(x)=\hat a(x)",
    ),
    dgp_note=(
        "Gerçek koşullu ortalama doğrusaldır; yerel doğrusal tahmin doğruları tam yeniden üretir. Hatalar tek örneklemde "
        "hesaplanır; büyük n'de örnekleme gürültüsü küçüktür, fark yanlılıktan gelir. Tohum 808."
    ),
    look_at=(
        "**Grafik** — gözlemler, gerçek doğru (kesikli), aynı h ile Nadaraya–Watson ve yerel doğrusal eğriler.",
        "**Ölçüler** — iki sınırda ve ortada tahmin hatası m̂(x) − m(x).",
    ),
    build=_build_boundary, metrics=_boundary_metrics, takeaway=_boundary_takeaway,
)


# --- Deney 3: kısmen doğrusal model ----------------------------------------------------------------

VAR_X2 = 16.0 / 5.0 - (4.0 / 3.0) ** 2
"""X ~ U[−2, 2] için Var(X²) = E[X⁴] − E[X²]² = 16/5 − 16/9."""


def linear_control_bias(strength: float) -> float:
    """Y'yi D ve X üzerine regres etmenin olasılık limiti θ + a·Var(X²)/{Var(X²) + 1}; X² doğrusal X ile ilişkisizdir."""

    return strength * VAR_X2 / (VAR_X2 + 1.0)


def _partially_linear_body(n: int, strength: float, h: float) -> tuple:
    strength, h = round(strength, 6), round(h, 6)
    square = E.power(E.var("x"), 2)
    return (
        NewSample(FRAME, n, None),
        Draw(FRAME, "x", "uniform", -2, 2, "X ~ U[−2, 2]"),
        Draw(FRAME, "v", "normal", 0, 1, "v ~ N(0, 1)"),
        Draw(FRAME, "e", "normal", 0, 1, "e ~ N(0, 1)"),
        Derive(FRAME, "d", E.add(square, E.var("v")), "D = X² + v: temel değişken X ile doğrusal olmayan biçimde ilişkili"),
        Derive(FRAME, "y", E.add(E.add(E.var("d"), E.mul(strength, square)), E.var("e")),
               "Y = θD + g(X) + e, θ = 1, g(X) = a·X²"),
        OLS("dogrusal_kontrol", FRAME, "y", ("d", "x")),
        LocalResidual(FRAME, "y_art", "y", "x", h, "Ỹ = Y − m̂_Y(X): Y'nin X üzerindeki yerel doğrusal tahminden sapması"),
        LocalResidual(FRAME, "d_art", "d", "x", h, "D̃ = D − m̂_D(X): D'nin X üzerindeki yerel doğrusal tahminden sapması"),
        OLS("robinson", FRAME, "y_art", ("d_art",)),
    )


def _build_partially_linear(p: Parameters) -> tuple:
    return (
        MonteCarlo(
            FRAME, REPS, MC_SEED, _partially_linear_body(int(p["n"]), p["strength"], p["h"]),
            (("theta_dogrusal", E.coef("dogrusal_kontrol", "d")), ("theta_robinson", E.coef("robinson", "d_art"))),
            "mc",
            "Monte Carlo: her tekrarda yeni örneklem; X'i doğrusal kontrol eden OLS ve Robinson artıklaştırması",
        ),
        Histogram(
            "mc", (("theta_robinson", "Robinson (yerel doğrusal artıklaştırma)"), ("theta_dogrusal", "OLS, X doğrusal")),
            0.5, 2.5, ((1.0, "Gerçek θ = 1"),), "θ tahmini", f"{REPS} örneklemde θ tahminlerinin dağılımı", bins=50,
        ),
    )


def _partially_linear_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    table = state.tables["mc"]
    return (
        SimMetric("Doğrusal kontrol: ortalama", plain(float(table["theta_dogrusal"].mean()), 3),
                  f"Olasılık limiti 1 + {plain(linear_control_bias(p['strength']), 3)}."),
        SimMetric("Doğrusal kontrol: std. sapma", plain(float(table["theta_dogrusal"].std()), 3), "Tekrarlar arası."),
        SimMetric("Robinson: ortalama", plain(float(table["theta_robinson"].mean()), 3), "Gerçek θ = 1."),
        SimMetric("Robinson: std. sapma", plain(float(table["theta_robinson"].std()), 3), "Tekrarlar arası."),
    )


def _partially_linear_takeaway(state: LabState, p: Parameters) -> str:
    table = state.tables["mc"]
    linear = float(table["theta_dogrusal"].mean())
    robinson = float(table["theta_robinson"].mean())
    if p["strength"] < 1e-9:
        return (
            f"a = 0: g(X) = 0, X'in Y üzerinde doğrudan etkisi yok. İki tahmin de 1 civarında (doğrusal kontrol "
            f"{plain(linear, 3)}, Robinson {plain(robinson, 3)}). a'yı artırın."
        )
    return (
        f"g(X) = a·X² ve E[D|X] = X² doğrusal olmayan: X'i doğrusal kontrol eden OLS ortalamada {plain(linear, 3)} "
        f"veriyor (olasılık limiti {plain(1 + linear_control_bias(p['strength']), 3)}). X²'nin etkisi D üzerinden θ'ya "
        f"yükleniyor: fonksiyonel biçim hatası eksik değişken yanlılığı gibi çalışır. Robinson artıklaştırması Y'yi ve D'yi "
        f"X'ten esnek biçimde arındırır ve ortalamada {plain(robinson, 3)} verir (gerçek θ = 1). h'yi çok büyütün: "
        "yerel doğrusal tahmin X²'yi izleyemez ve yanlılık geri gelir (§8.10). Konu 12'deki DML aynı fikri makine "
        "öğrenmesi ve çapraz uydurma ile genelleştirir."
    )


PARTIALLY_LINEAR = SimExperiment(
    topic_key=TOPIC, number=3,
    title="Kısmen doğrusal model: doğrusal kontrol ve Robinson",
    question="Kontrol değişkeninin etkisi doğrusal değilse onu doğrusal kontrol etmek θ'yı nasıl bozar; Robinson artıklaştırması neyi düzeltir?",
    note=NoteRef("8.10", 0),
    parameters=(
        SimParameter("strength", "Doğrusal olmama a", 0.0, 2.0, 1.0, 0.25, "g(X) = a·X²: X'in Y'ye doğrudan etkisi.",
                     decimals=2),
        SimParameter("h", "Artıklaştırmada h", 0.1, 2.0, 0.3, 0.05, "Yerel doğrusal tahminin bant genişliği.",
                     decimals=2),
        SimParameter("n", "Her örneklemdeki gözlem sayısı", 200, 1000, 500, 100, "Örneklem büyüklüğü.",
                     integer=True, decimals=0),
    ),
    dgp=lambda p: (
        r"X\sim U[-2,2], \qquad v,e\sim N(0,1)\ \text{bağımsız}",
        rf"D=X^2+v, \qquad Y=\theta D+g(X)+e, \quad \theta=1,\ g(X)=aX^2,\ a={number(p['strength'], 2)}",
        r"\tilde Y=Y-\hat m_Y(X),\quad \tilde D=D-\hat m_D(X), \qquad \hat\theta=\frac{\sum\tilde D_i\tilde Y_i}{\sum\tilde D_i^2}",
    ),
    dgp_note=(
        f"{REPS} tekrarın her birinde yeni örneklem çekilir. Doğrusal kontrol: Y'nin D ve X üzerine OLS regresyonu. "
        "Robinson: Y ve D, X üzerinde yerel doğrusal tahminle artıklaştırılır, artıklar birbirine regres edilir."
    ),
    look_at=(
        "**Grafik** — iki tahmin edicinin θ dağılımı; gerçek θ = 1 (dikey çizgi).",
        "**Ölçüler** — ortalama ve standart sapma; doğrusal kontrolün olasılık limiti.",
    ),
    build=_build_partially_linear, metrics=_partially_linear_metrics, takeaway=_partially_linear_takeaway,
)


KONU08_EXPERIMENTS = (BANDWIDTH, BOUNDARY, PARTIALLY_LINEAR)
