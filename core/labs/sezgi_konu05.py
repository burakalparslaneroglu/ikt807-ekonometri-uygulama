"""Konu 5 Sezgi deneyleri: ikili sonuçlarda olasılık, ölçek ve marjinal etki.

Deney 1  LPM ve Logit: olasılık sınırları ve ortalama etki     (Notlar §5.3, §5.4)
Deney 2  Ölçek normalizasyonu: Logit ve Probit katsayıları     (Notlar §5.5)
Deney 3  Kukla değişkende türev mi, sonlu fark mı?             (Notlar §5.8.2)
"""

from __future__ import annotations

import numpy as np
from scipy import special, stats

from core.labs import expr as E
from core.labs.runner import LabState
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, number, plain
from core.labs.spec import (
    OLS,
    BinaryChoice,
    Curve,
    Derive,
    Draw,
    MarginalEffects,
    MeanPoints,
    ModelLine,
    NewSample,
    NoteRef,
    Plot,
    ZeroLine,
)

SEED = 805
FRAME = "sim"
TOPIC = "konu05"
STEP = 0.25
"""X, sürekli bir çekilişin 0,25'lik adıma yuvarlanmasıyla elde edilir: her değerde Y = 1 payı görülebilsin."""


def _grid_x(low: float, high: float) -> tuple:
    return (
        Draw(FRAME, "ux", "uniform", low, high, f"Sürekli çekiliş, U[{E.format_number(low)}, {E.format_number(high)}]"),
        Derive(FRAME, "x", E.mul(E.rounded(E.div(E.var("ux"), STEP)), STEP), "X: 0,25'lik adıma yuvarlanmış değer"),
    )


def _logistic_curve(model: str) -> E.Expr:
    return E.logistic(E.add(E.coef(model, E.INTERCEPT), E.mul(E.coef(model, "x"), E.var("x"))))


def _x(state: LabState) -> np.ndarray:
    return state.frames[FRAME]["x"].to_numpy(dtype=float)


# --- Deney 1: LPM ve Logit ------------------------------------------------------------

def true_ame_logit(x: np.ndarray, alpha: float, beta: float) -> float:
    """Örneklemdeki X değerleri üzerinden gerçek ortalama türev: ortalama β·Λ(α+βX){1−Λ(α+βX)}."""

    p = special.expit(alpha + beta * x)
    return float(np.mean(beta * p * (1.0 - p)))


def _build_lpm(p: Parameters) -> tuple:
    alpha, beta = round(p["alpha"], 6), round(p["beta"], 6)
    index = E.add(alpha, E.mul(beta, E.var("x")))
    return (
        NewSample(FRAME, int(p["n"]), SEED),
        *_grid_x(-3.0, 3.0),
        Draw(FRAME, "u", "uniform", 0, 1, "Y'yi üretmek için tek düze çekiliş"),
        Derive(FRAME, "p", E.logistic(index), "Gerçek olasılık P(Y=1|X) = Λ(α + βX)"),
        Derive(FRAME, "y", E.compare("lt", E.var("u"), E.var("p")), "Y = 1{U < P(Y=1|X)}: Bernoulli sonuç"),
        OLS("lpm", FRAME, "y", ("x",), vcov="HC1"),
        BinaryChoice("logit", FRAME, "y", ("x",), "logit"),
        MarginalEffects("logit", "ame_logit", ("x",)),
        Plot(FRAME, "x",
             (MeanPoints("y", "Y = 1 payı (her X değerinde)"),
              Curve(E.logistic(index), "Gerçek P(Y=1|x)", dashed=True),
              ModelLine("lpm", "LPM doğrusu"),
              Curve(_logistic_curve("logit"), "Logit tahmini"),
              ZeroLine("Olasılığın alt sınırı")),
             "X", "Olasılık", "Aynı veriye doğrusal ve S biçimli olasılık modeli"),
    )


def _lpm_outside(state: LabState) -> float:
    params = state.models["lpm"].params
    fitted = params["Intercept"] + params["x"] * _x(state)
    return float(np.mean((fitted < 0) | (fitted > 1)))


def _lpm_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    truth = true_ame_logit(_x(state), p["alpha"], p["beta"])
    return (
        SimMetric("Gerçek AME", plain(truth), "Örneklemdeki X'ler üzerinden ortalama β·Λ'(α + βX): DGP'den bilinir."),
        SimMetric("LPM eğimi", plain(float(state.models["lpm"].params["x"])),
                  "Doğrusal olasılık modelinin tek katsayısı; olasılık ölçeğinde."),
        SimMetric("Logit AME", plain(float(state.models["ame_logit"].params["x"])),
                  "Logit katsayısı × ortalama Λ(1 − Λ)."),
        SimMetric("Logit β̂", plain(float(state.models["logit"].params["x"]), 3), f"Gerçek β = {plain(p['beta'], 2)}."),
        SimMetric("LPM sınır dışı tahmin", f"%{100 * _lpm_outside(state):.1f}".replace(".", ","),
                  "LPM'nin tahmin ettiği olasılığın [0, 1] dışında kaldığı gözlemlerin payı."),
    )


def _lpm_takeaway(state: LabState, p: Parameters) -> str:
    truth = true_ame_logit(_x(state), p["alpha"], p["beta"])
    slope = float(state.models["lpm"].params["x"])
    logit = float(state.models["ame_logit"].params["x"])
    outside = 100 * _lpm_outside(state)
    if abs(slope - truth) <= 0.1 * abs(truth):
        text = (
            f"LPM eğimi ({plain(slope, 3)}) gerçek ortalama etkiye ({plain(truth, 3)}) yakın. "
        )
    else:
        text = (
            f"LPM eğimi ({plain(slope, 3)}) gerçek ortalama etkiden ({plain(truth, 3)}) belirgin biçimde farklı: LPM "
            "katsayısı olasılık eğrisinin en iyi doğrusal yaklaşımının eğimidir, ortalama türev değil. İkisinin ne kadar "
            "ayrışacağı X'in dağılımına bağlıdır (X normal dağılsaydı anakütlede çakışırlardı; burada X tek düze). "
        )
    text += f"Logit AME'si ({plain(logit, 3)}) doğru modelden geldiği için gerçeğe yakındır ve eğrinin biçimini de yakalar. "
    if outside > 0.5:
        text += (
            f"Gözlemlerin %{plain(outside, 1)} kadarında LPM'nin tahmini [0, 1] dışında: bireysel olasılık tahmini "
            "hedefse bu ciddi bir sorundur. "
        )
    else:
        text += "Bu ayarda LPM tahminleri [0, 1] içinde kalıyor; β'yı artırın ve sınır ihlallerini izleyin. "
    return text + "Seçim araştırma hedefine bağlıdır: ortalama bir özet mi, olasılık eğrisinin kendisi mi (§5.3)?"


LPM = SimExperiment(
    topic_key=TOPIC, number=1,
    title="LPM ve Logit: olasılık sınırları ve ortalama etki",
    question="Gerçek olasılık S biçimliyse doğrusal olasılık modeli neyi doğru, neyi yanlış yapar?",
    note=NoteRef("5.3", 0, ("§5.4",)),
    parameters=(
        SimParameter("n", "Gözlem sayısı", 500, 5000, 2000, 500, "Örneklem büyüklüğü.", integer=True, decimals=0),
        SimParameter("alpha", "Sabit α", -2.0, 2.0, 0.0, 0.25, "Olasılık eğrisini sağa–sola kaydırır.", decimals=2),
        SimParameter("beta", "Eğim β", 0.25, 3.0, 1.5, 0.25, "Eğrinin dikliği; büyüdükçe olasılıklar 0 ve 1'e yığılır.",
                     decimals=2),
    ),
    dgp=lambda p: (
        r"X\in\{-3;\,-2{,}75;\,\dots;\,3\}\ \text{(tek düze çekilişin 0,25 adıma yuvarlanması)}",
        rf"P(Y=1\mid X)=\Lambda(\alpha+\beta X), \quad \alpha={number(p['alpha'], 2)}, \ \beta={number(p['beta'], 2)}",
        r"Y=\mathbf 1\{U<\Lambda(\alpha+\beta X)\}, \quad U\sim U(0,1)",
    ),
    dgp_note=(
        "Veri Logit modelinden üretilir; Logit doğru model, LPM ise doğrusal bir yaklaşımdır. X'in 0,25 adımlı "
        "değerleri, her X'te gözlenen Y = 1 payını (koşullu ortalamayı) grafikte göstermeyi sağlar."
    ),
    look_at=(
        "**Grafik** — noktalar her X değerinde Y = 1 payıdır (nokta büyüklüğü gözlem sayısı). Kesikli eğri gerçek "
        "olasılık, düz doğru LPM, düz eğri Logit tahmini.",
        "**Ölçüler** — DGP'den bilinen ortalama marjinal etki, LPM eğimi, Logit AME ve LPM'nin [0, 1] dışına taşan "
        "tahminlerinin payı.",
    ),
    build=_build_lpm, metrics=_lpm_metrics, takeaway=_lpm_takeaway,
)


# --- Deney 2: ölçek normalizasyonu ----------------------------------------------------------

CONSTANT = 0.3


def true_ame_probit(x: np.ndarray, b1: float, sigma: float) -> float:
    """Gerçek AME: ortalama (β₁/σ)·φ((β₀ + β₁X)/σ)."""

    return float(np.mean(b1 / sigma * stats.norm.pdf((CONSTANT + b1 * x) / sigma)))


def _build_scale(p: Parameters) -> tuple:
    b1, sigma = round(p["b1"], 6), round(p["sigma"], 6)
    return (
        NewSample(FRAME, int(p["n"]), SEED),
        *_grid_x(-3.0, 3.0),
        Draw(FRAME, "eps", "normal", 0, 1, "Gizli hata ε ~ N(0, 1)"),
        Derive(FRAME, "ystar", E.add(E.add(CONSTANT, E.mul(b1, E.var("x"))), E.mul(sigma, E.var("eps"))),
               "Gizli eğilim Y* = β₀ + β₁X + σε"),
        Derive(FRAME, "y", E.positive(E.var("ystar")), "Gözlenen Y = 1{Y* > 0}"),
        BinaryChoice("probit", FRAME, "y", ("x",), "probit"),
        BinaryChoice("logit", FRAME, "y", ("x",), "logit"),
        MarginalEffects("probit", "ame_probit", ("x",)),
        MarginalEffects("logit", "ame_logit", ("x",)),
        Plot(FRAME, "x",
             (MeanPoints("y", "Y = 1 payı (her X değerinde)"),
              Curve(E.normcdf(E.div(E.add(CONSTANT, E.mul(b1, E.var("x"))), sigma)), "Gerçek P(Y=1|x)", dashed=True),
              Curve(E.normcdf(E.add(E.coef("probit", E.INTERCEPT), E.mul(E.coef("probit", "x"), E.var("x")))),
                    "Probit tahmini"),
              Curve(_logistic_curve("logit"), "Logit tahmini", dashed=False)),
             "X", "Olasılık", "Farklı katsayılar, neredeyse aynı olasılık eğrisi"),
    )


def _scale_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    probit = float(state.models["probit"].params["x"])
    logit = float(state.models["logit"].params["x"])
    truth = true_ame_probit(_x(state), p["b1"], p["sigma"])
    return (
        SimMetric("β₁/σ (tanımlanan)", plain(p["b1"] / p["sigma"], 3), "Veri yalnız β₁/σ oranını belirler."),
        SimMetric("Probit β̂", plain(probit, 3), "Doğru model: β₁/σ'yı tahmin eder."),
        SimMetric("Logit β̂", plain(logit, 3), "Lojistik ölçekte; tipik olarak Probit'in 1,6–1,8 katı."),
        SimMetric("Logit / Probit", plain(logit / probit, 2), "Katsayı oranı: ölçek normalizasyonunun izi."),
        SimMetric("AME: Probit · Logit · gerçek",
                  f"{plain(float(state.models['ame_probit'].params['x']), 3)} · "
                  f"{plain(float(state.models['ame_logit'].params['x']), 3)} · {plain(truth, 3)}",
                  "Olasılık ölçeğinde iki model aynı etkiyi verir."),
    )


def _scale_takeaway(state: LabState, p: Parameters) -> str:
    probit = float(state.models["probit"].params["x"])
    logit = float(state.models["logit"].params["x"])
    return (
        f"σ = {plain(p['sigma'], 2)} iken Probit katsayısı {plain(probit, 3)}, gerçek β₁/σ = "
        f"{plain(p['b1'] / p['sigma'], 3)}: β₁ ve σ ayrı ayrı tanımlanamaz, yalnız oranları. σ'yı değiştirin: "
        "katsayılar ölçekle birlikte değişir, olasılık eğrisi de. Logit katsayısı Probit'inkinin yaklaşık "
        f"{plain(logit / probit, 2)} katı; bu fark iki modelin gizli hatayı farklı ölçeklemesinden gelir, ekonomik bir fark "
        "değildir. Karşılaştırma katsayılarla değil, olasılıklar ve marjinal etkilerle yapılır (§5.5)."
    )


SCALE = SimExperiment(
    topic_key=TOPIC, number=2,
    title="Ölçek normalizasyonu: Logit ve Probit katsayıları",
    question="Logit ve Probit aynı veriye neden farklı katsayılar ama neredeyse aynı olasılıklar verir?",
    note=NoteRef("5.5", 0),
    parameters=(
        SimParameter("n", "Gözlem sayısı", 500, 10000, 3000, 500, "Örneklem büyüklüğü.", integer=True, decimals=0),
        SimParameter("b1", "Gizli eğim β₁", 0.25, 2.0, 1.0, 0.25, "Gizli denklemdeki eğim.", decimals=2),
        SimParameter("sigma", "Hata ölçeği σ", 0.5, 3.0, 1.0, 0.25,
                     "Gizli hatanın standart sapması; veride gözlenmez.", decimals=2),
    ),
    dgp=lambda p: (
        r"X\in\{-3;\,-2{,}75;\,\dots;\,3\}, \qquad \varepsilon\sim N(0,1)",
        rf"Y^*=0{{,}}3+\beta_1X+\sigma\varepsilon, \quad \beta_1={number(p['b1'], 2)}, \ \sigma={number(p['sigma'], 2)}",
        r"Y=\mathbf 1\{Y^*>0\} \ \Rightarrow\ P(Y=1\mid X)=\Phi\!\left(\frac{0{,}3+\beta_1X}{\sigma}\right)",
    ),
    dgp_note=(
        "Gizli hata normal olduğu için Probit doğru modeldir; Logit yakın bir yaklaşımdır. β₁ ile σ'yı aynı oranda "
        "büyütmek gözlenen veriyi değiştirmez: tanımlanan yalnız β₁/σ'dır."
    ),
    look_at=(
        "**Grafik** — her X'te Y = 1 payı, gerçek olasılık (kesikli) ve iki modelin tahmin ettiği olasılık eğrileri. "
        "Eğriler neredeyse üst üste biner.",
        "**Ölçüler** — tanımlanan β₁/σ, iki modelin ham katsayısı ve oranı; olasılık ölçeğinde AME'ler.",
    ),
    build=_build_scale, metrics=_scale_metrics, takeaway=_scale_takeaway,
)


# --- Deney 3: kukla değişken ----------------------------------------------------------------

SLOPE = 0.5


def true_dummy_effect(x: np.ndarray, alpha: float, gamma: float) -> float:
    """Gerçek sonlu fark: ortalama Λ(α + 0,5X + γ) − Λ(α + 0,5X)."""

    return float(np.mean(special.expit(alpha + SLOPE * x + gamma) - special.expit(alpha + SLOPE * x)))


def _build_dummy(p: Parameters) -> tuple:
    alpha, gamma = round(p["alpha"], 6), round(p["gamma"], 6)
    base = E.add(alpha, E.mul(SLOPE, E.var("x")))
    fitted = E.add(E.coef("logit", E.INTERCEPT), E.mul(E.coef("logit", "x"), E.var("x")))
    return (
        NewSample(FRAME, int(p["n"]), SEED),
        Draw(FRAME, "x", "normal", 0, 1, "Sürekli kovaryat X ~ N(0, 1)"),
        Draw(FRAME, "ud", "uniform", 0, 1, "Göstergeyi üretmek için tek düze çekiliş"),
        Derive(FRAME, "d", E.positive(E.sub(E.var("ud"), 0.5)), "Gösterge D: gözlemlerin yarısında 1"),
        Draw(FRAME, "u", "uniform", 0, 1, "Y'yi üretmek için tek düze çekiliş"),
        Derive(FRAME, "y", E.compare("lt", E.var("u"), E.logistic(E.add(base, E.mul(gamma, E.var("d"))))),
               "Y = 1{U < Λ(α + 0,5X + γD)}"),
        BinaryChoice("logit", FRAME, "y", ("x", "d"), "logit"),
        MarginalEffects("logit", "me_turev", ("d",)),
        MarginalEffects("logit", "me_fark", ("d",), discrete=("d",)),
        Plot(FRAME, "x",
             (Curve(E.logistic(base), "Gerçek, D = 0", dashed=True, color=0),
              Curve(E.logistic(E.add(base, gamma)), "Gerçek, D = 1", dashed=True, color=1),
              Curve(E.logistic(fitted), "Logit, D = 0", color=0),
              Curve(E.logistic(E.add(fitted, E.coef("logit", "d"))), "Logit, D = 1", color=1)),
             "X", "P(Y = 1)", "Göstergenin etkisi: iki eğri arasındaki dikey uzaklık"),
    )


def _dummy_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    derivative = float(state.models["me_turev"].params["d"])
    difference = float(state.models["me_fark"].params["d"])
    truth = true_dummy_effect(_x(state), p["alpha"], p["gamma"])
    return (
        SimMetric("Türev tabanlı", plain(derivative), "β̂_D × ortalama Λ(1 − Λ): D'yi sürekli değişken gibi ele alır."),
        SimMetric("Sonlu fark (AME)", plain(difference), "Ortalama Λ(x'β̂ + β̂_D) − Λ(x'β̂): D = 1 ve D = 0 senaryoları."),
        SimMetric("Gerçek sonlu fark", plain(truth), "DGP'den: ortalama Λ(α + 0,5X + γ) − Λ(α + 0,5X)."),
        SimMetric("Türev − sonlu fark", plain(derivative - difference), "Yaklaşım hatası."),
    )


def _dummy_takeaway(state: LabState, p: Parameters) -> str:
    derivative = float(state.models["me_turev"].params["d"])
    difference = float(state.models["me_fark"].params["d"])
    gap = derivative - difference
    if abs(gap) < 0.01:
        return (
            f"γ = {plain(p['gamma'], 2)}: türev tabanlı yaklaşım ({plain(derivative, 3)}) ile sonlu fark "
            f"({plain(difference, 3)}) hemen hemen aynı. Katsayı küçük ve olasılıklar orta bölgedeyken türev iyi bir "
            "yaklaşımdır. γ'yı büyütün veya α'yı −3'e çekin: fark açılır (§5.8.2)."
        )
    return (
        f"γ = {plain(p['gamma'], 2)}: türev tabanlı yaklaşım {plain(derivative, 3)}, doğru tanım olan sonlu fark "
        f"{plain(difference, 3)}; fark {plain(gap, 3)}. Türev, Λ eğrisinin D = 0 noktasındaki eğimini 0'dan 1'e kadar "
        "sabit sayar; eğri büküldükçe (büyük katsayı, uç olasılıklar) bu yaklaşım bozulur. Kukla değişkende etki "
        "tanımı D = 1 ve D = 0 senaryoları arasındaki olasılık farkıdır."
    )


DUMMY = SimExperiment(
    topic_key=TOPIC, number=3,
    title="Kukla değişkende türev mi, sonlu fark mı?",
    question="0/1 bir göstergenin olasılık etkisini türevle ölçmek ne zaman yanıltır?",
    note=NoteRef("5.8.2", 0),
    parameters=(
        SimParameter("gamma", "Göstergenin katsayısı γ", 0.0, 4.0, 2.0, 0.25, "D'nin gizli indeksteki etkisi.",
                     decimals=2),
        SimParameter("alpha", "Sabit α", -3.0, 1.0, -1.0, 0.25,
                     "Negatif α olasılıkları 0'a yaklaştırır; eğri bükülmesi artar.", decimals=2),
        SimParameter("n", "Gözlem sayısı", 1000, 10000, 4000, 1000, "Örneklem büyüklüğü.", integer=True, decimals=0),
    ),
    dgp=lambda p: (
        r"X\sim N(0,1), \qquad D\sim\text{Bernoulli}(0{,}5)\ \text{(bağımsız)}",
        rf"P(Y=1\mid X,D)=\Lambda(\alpha+0{{,}}5X+\gamma D), \quad \alpha={number(p['alpha'], 2)}, \ "
        rf"\gamma={number(p['gamma'], 2)}",
        r"\text{sonlu fark}=\mathbb E[\Lambda(\alpha+0{,}5X+\gamma)-\Lambda(\alpha+0{,}5X)]",
    ),
    dgp_note=(
        "Model doğru: Logit tahmini tutarlıdır. Soru tahmin değil, etkinin nasıl tanımlandığıdır: aynı Logit tahmininden "
        "iki farklı \"marjinal etki\" hesaplanır."
    ),
    look_at=(
        "**Grafik** — D = 0 ve D = 1 için gerçek (kesikli) ve tahmin edilen (düz) olasılık eğrileri. Göstergenin etkisi "
        "iki eğri arasındaki dikey uzaklıktır.",
        "**Ölçüler** — türev tabanlı yaklaşım, sonlu fark AME ve DGP'den bilinen gerçek sonlu fark.",
    ),
    build=_build_dummy, metrics=_dummy_metrics, takeaway=_dummy_takeaway,
)


KONU05_EXPERIMENTS = (LPM, SCALE, DUMMY)
