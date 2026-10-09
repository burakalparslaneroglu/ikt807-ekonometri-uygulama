"""Konu 12 Sezgi deneyleri: yüksek boyutlu kontroller altında hedef katsayı, ortogonal moment, DML'in sınırı.

Deney 1  Yalnız sonuç seçimi, double selection, artıklaştırma ve DML               (Notlar §12.11, §12.2–§12.4, §12.7)
Deney 2  Ortogonal moment: yardımcı parametre hatasına birinci dereceden duyarsızlık  (Notlar §12.5)
Deney 3  Gözlenmeyen karıştırıcı: DML tanımlama sorununu çözmez                       (Notlar §12.10)

Deney 1 varsayılan ayarlarıyla notlardaki Tablo 12.1'i birebir yeniden üretir: tek üreteç
``np.random.default_rng(80712)``; önce X₁, …, X₁₈₀ sütunları, sonra V ve ε çekilir.
"""

from __future__ import annotations

import numpy as np

from core.labs import expr as E
from core.labs.runner import LabState, coverage_key
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, number, plain
from core.labs.spec import (
    OLS,
    THETA,
    CrossFitDML,
    Curve,
    Derive,
    DoubleSelection,
    Draw,
    DrawColumns,
    EstimatePlot,
    Histogram,
    MonteCarlo,
    NewSample,
    NoteRef,
    Plot,
    Scatter,
    VLine,
)

FRAME = "sim"
TOPIC = "konu12"


def _count(value: int) -> str:
    return f"{int(value):,}".replace(",", ".")


def _signed(value: float, decimals: int = 3) -> str:
    """Yuvarlanınca sıfır olan küçük negatif sayılar "−0,000" değil "0,000" yazılır."""

    return plain(round(value, decimals) + 0.0, decimals)


def _linear(terms, start: E.Expr | None = None) -> E.Expr:
    """Σ katsayı·değişken, soldan sağa (üretilen kodda aynı toplama sırası); sıfır katsayılar atlanır."""

    total = start
    for value, name in terms:
        if value == 0:
            continue
        if total is None:
            total = E.mul(value, E.var(name))
        elif value > 0:
            total = E.add(total, E.mul(value, E.var(name)))
        else:
            total = E.sub(total, E.mul(-value, E.var(name)))
    return total


# --- Deney 1: §12.11 kontrollü yüksek boyutlu örnek --------------------------------------------------

DS_SEED = 80712
CONTROLS = 180
GAMMA = (0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6)
"""İlk sekiz kontrolün D denklemindeki katsayıları (c = 1 iken)."""
DIRECT_SIGNS = (1, -1, 1, -1, 1, -1, 1, -1)
"""İlk sekiz kontrolün Y denklemindeki doğrudan etkilerinin işaretleri; büyüklük β_c."""
OUTCOME_ONLY = ((0.9, "x21"), (-0.8, "x22"), (0.7, "x23"), (-0.6, "x24"))
GRID = (0.5, -3, 80)
CONFOUNDERS = tuple(f"x{j}" for j in range(1, 9))
ESTIMATES = (
    ("Kısa OLS", "kisa", "d"),
    ("Yalnız sonuç Post-Lasso", "secim_sonuc", "d"),
    ("Double selection", "secim", "d"),
    ("Artıklaştırma", "artik", THETA),
    ("DML (10 kat)", "dml", THETA),
)


def omitted_bias(bc: float, cg: float) -> float:
    """Kısa OLS'nin olasılık limitindeki yanlılık: Σγⱼβⱼ / (Σγⱼ² + 1) (Var(Xⱼ) = Var(V) = 1)."""

    gamma = cg * np.array(GAMMA)
    beta = bc * np.array(DIRECT_SIGNS)
    return float(gamma @ beta / (gamma @ gamma + 1))


def _build_double(p: Parameters) -> tuple:
    nobs, bc, cg = int(p["n"]), round(p["bc"], 6), round(p["cg"], 6)
    controls = tuple(f"x{j}" for j in range(1, CONTROLS + 1))
    treatment = _linear(((round(cg * g, 6), f"x{j}") for j, g in enumerate(GAMMA, start=1)))
    treatment = E.var("v") if treatment is None else E.add(treatment, E.var("v"))
    direct = ((round(bc * s, 6), f"x{j}") for j, s in enumerate(DIRECT_SIGNS, start=1))
    outcome = _linear((*direct, *OUTCOME_ONLY), start=E.var("d"))
    return (
        NewSample(FRAME, nobs, DS_SEED),
        DrawColumns(FRAME, "x", CONTROLS, 0.0, f"X₁, …, X_{CONTROLS}: bağımsız standart normal kontroller"),
        Draw(FRAME, "v", "normal", 0, 1, "V ~ N(0, 1): D'nin kontrollerle açıklanamayan kısmı"),
        Draw(FRAME, "e", "normal", 0, 1, "ε ~ N(0, 1)"),
        Derive(FRAME, "d", treatment, "D: ilk sekiz kontrol D'yi güçlü biçimde açıklar"),
        Derive(FRAME, "y", E.add(outcome, E.var("e")),
               "Y = θD + ...: θ = 1; ilk sekiz kontrolün Y'deki doğrudan etkisi zayıf, X21–X24 yalnız Y'yi etkiler"),
        Derive(FRAME, "kat5", E.sub(E.var("id"), E.mul(5, E.floor(E.div(E.sub(E.var("id"), 1), 5)))),
               "Lasso cezası için 5 CV katı: (sıra − 1) mod 5 + 1"),
        Derive(FRAME, "dis10", E.add(E.floor(E.div(E.mul(E.sub(E.var("id"), 1), 10), nobs)), 1),
               "DML dış katları: ardışık 10 blok"),
        OLS("kisa", FRAME, "y", ("d",), vcov="HC1"),
        DoubleSelection("secim", FRAME, "y", "d", controls, GRID, "kat5", "1se", track=CONFOUNDERS),
        CrossFitDML("artik", FRAME, "y", "d", controls, None, "kat5", GRID, "1se"),
        CrossFitDML("dml", FRAME, "y", "d", controls, "dis10", "kat5", GRID, "1se", residuals=("u_hat", "v_hat")),
        EstimatePlot(ESTIMATES, "tahminler", "θ tahmini ve yaklaşık %95 güven aralığı (HC1)",
                     "Yüksek boyutlu kontrol örneğinde yöntem karşılaştırması", truth=1.0, truth_label="Gerçek θ = 1"),
        Plot(FRAME, "v_hat",
             (Scatter("u_hat", "Gözlemler"),
              Curve(E.mul(E.coef("dml", THETA), E.var("v_hat")), "Eğim θ̂ (DML), sabitsiz")),
             "Çapraz uyarlanmış tedavi artığı V̂", "Çapraz uyarlanmış sonuç artığı Û",
             "DML'in son aşaması: artıkların artıklar üzerine regresyonu"),
    )


def _theta(state: LabState, model: str) -> float:
    result = state.models[model]
    return float(result.params[THETA if model in ("artik", "dml") else "d"])


def _double_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    scalars = state.scalars
    chosen = f"{int(scalars['secim_ny_iz'])} · {int(scalars['secim_nd_iz'])} · {int(scalars['secim_n_iz'])}"
    return (
        SimMetric("Yalnız sonuç Post-Lasso θ̂", plain(_theta(state, "secim_sonuc"), 3), "Kontroller yalnız Y denkleminden."),
        SimMetric("Double selection θ̂", plain(_theta(state, "secim"), 3), "Y ve D denklemlerinin seçimlerinin birleşimi."),
        SimMetric("DML θ̂", plain(_theta(state, "dml"), 3), "10 dış katla çapraz uyarlanmış Lasso artıkları."),
        SimMetric("Seçilen karıştırıcı (8'den)", chosen,
                  "Sırasıyla yalnız-sonuç seçimine (S_Y), D denklemi seçimine (S_D) ve double selection birleşimine "
                  "giren karıştırıcı (X₁–X₈) sayısı."),
    )


def _double_takeaway(state: LabState, p: Parameters) -> str:
    scalars = state.scalars
    bias = omitted_bias(p["bc"], p["cg"])
    if abs(bias) < 1e-12:
        return (
            f"Karıştırma kanalı kapalı (β_c·c = 0): X₁–X₈'i dışarıda bırakmak θ̂'yı yanlı yapmaz. Kısa OLS "
            f"{plain(_theta(state, 'kisa'), 3)}, DML {plain(_theta(state, 'dml'), 3)}. Kaydırıcıları açın."
        )
    outcome_count, treatment_count = int(scalars["secim_ny_iz"]), int(scalars["secim_nd_iz"])
    outcome_text = "hiçbirini almadı" if outcome_count == 0 else f"yalnız {outcome_count} tanesini aldı"
    treatment_text = "hiçbirini seçmedi" if treatment_count == 0 else f"{treatment_count} tanesini seçti"
    return (
        f"Yalnız sonuç denkleminde seçim ilk sekiz karıştırıcının {outcome_text}: Y'deki doğrudan etkileri küçük "
        "olduğu için Lasso onları Y'yi tahmin etmekte gereksiz görür. Oysa D ile güçlü ilişkilidirler; dışarıda "
        f"kalınca kısa regresyonun yanlılığı Σγⱼβⱼ/(Σγⱼ² + 1) = {plain(bias, 3)} düzeyindedir (§12.2). D denklemi "
        f"Lasso'su {treatment_text}; birleşimle double selection {plain(_theta(state, 'secim'), 3)} verdi. "
        f"Artıklaştırma {plain(_theta(state, 'artik'), 3)}, DML {plain(_theta(state, 'dml'), 3)}. Tek bir örneklemde "
        "yöntem sıralaması yapılmaz; mekanizma önemlidir: kontrol seçimi hedef katsayının moment koşuluna göre "
        "yapılmalıdır (§12.3–§12.7)."
    )


DOUBLE = SimExperiment(
    topic_key=TOPIC, number=1,
    title="Yalnız sonuç seçimi, double selection, artıklaştırma ve DML",
    question=(
        "180 aday kontrolün birkaçı hem D'yi hem Y'yi etkiliyorsa, yalnız Y'yi iyi tahmin eden kontrolleri seçmek "
        "hedef katsayı θ'yı neden yanıltır; double selection ve DML bunu nasıl düzeltir?"
    ),
    note=NoteRef("12.11", 0, ("§12.2", "§12.3", "§12.4", "§12.7")),
    parameters=(
        SimParameter("bc", "Karıştırıcıların Y'deki doğrudan etkisi β_c", 0.0, 0.3, 0.1, 0.02,
                     "X₁–X₈'in Y denklemindeki katsayı büyüklüğü (işaretler +, −, +, …).", decimals=2),
        SimParameter("cg", "Karıştırıcıların D'deki gücü c", 0.0, 1.5, 1.0, 0.1,
                     "D denklemindeki katsayılar c·(0,9; −0,9; 0,8; …; −0,6).", decimals=1),
        SimParameter("n", "Gözlem sayısı n", 400, 1600, 800, 200, "Dış katlar her zaman ardışık 10 blok.",
                     integer=True, decimals=0),
    ),
    dgp=lambda p: (
        rf"X_1,\dots,X_{{{CONTROLS}}},\ V,\ \varepsilon\ \sim N(0,1)\ \text{{bağımsız}},\qquad n={int(p['n'])}",
        rf"D=c\,(0{{,}}9X_1-0{{,}}9X_2+0{{,}}8X_3-0{{,}}8X_4+0{{,}}7X_5-0{{,}}7X_6+0{{,}}6X_7-0{{,}}6X_8)+V,"
        rf"\quad c={number(p['cg'], 1)}",
        rf"Y=\theta D+\beta_c(X_1-X_2+X_3-\dots-X_8)+0{{,}}9X_{{21}}-0{{,}}8X_{{22}}+0{{,}}7X_{{23}}-0{{,}}6X_{{24}}"
        rf"+\varepsilon,\quad \theta=1,\ \beta_c={number(p['bc'], 2)}",
    ),
    dgp_note=(
        "Rastgele sayılar tek üreteçten (np.random.default_rng(80712)) sırayla çekilir: önce 180 kontrol sütunu, sonra "
        "V ve ε. Varsayılan ayarlar (β_c = 0,10, c = 1, n = 800) notlardaki Tablo 12.1'i birebir verir. Bütün Lasso "
        "adımlarında ceza 5-katlı CV ve bir standart hata kuralıyla seçilir (kat: (sıra − 1) mod 5 + 1); ölçekleme her "
        "katın eğitim verisinden öğrenilir. Yalnız-sonuç Post-Lasso'da D cezalandırılmasın diye Y ve her X önce D "
        "üzerinde artıklaştırılır. DML'in dış katları ardışık 10 bloktur; son aşama sabitsizdir, SH HC1'dir."
    ),
    look_at=(
        "**Grafik 1** — beş yöntemin θ tahmini ve %95 güven aralığı; kesikli çizgi gerçek θ = 1.",
        "**Grafik 2** — DML'in son aşaması: çapraz uyarlanmış artıklar ve orijinden geçen doğru.",
        "**Ölçüler ve tablo** — yöntemlerin tahminleri ve ilk sekiz karıştırıcıdan kaçının seçildiği "
        "(notlarda Tablo 12.1).",
    ),
    build=_build_double, metrics=_double_metrics, takeaway=_double_takeaway,
    tables=(("tahminler", "Hedef katsayı tahminleri (notlarda Tablo 12.1)"),),
    labels=(("etiket", "Yöntem"), ("tahmin", "θ̂"), ("sh", "HC1 SH"), ("alt", "%95 GA alt"), ("ust", "%95 GA üst")),
)


# --- Deney 2: ortogonal moment --------------------------------------------------------------------

ORTH_SEED = 81202
ORTH_REPS = 2000


def naive_bias(delta: float, beta: float, gamma: float) -> float:
    """β̃ = (1 − δ)β iken naif momentin yanlılığı: δβγ/(1 + γ²)."""

    return delta * beta * gamma / (1 + gamma**2)


def orthogonal_bias(delta: float, beta: float, gamma: float) -> float:
    """γ̃ = (1 − δ)γ, η̃ = (1 − δ)η iken ortogonal momentin yanlılığı: δ²βγ/(1 + δ²γ²)."""

    return delta**2 * beta * gamma / (1 + delta**2 * gamma**2)


def _build_orthogonal(p: Parameters) -> tuple:
    delta, beta, gamma, nobs = round(p["delta"], 6), round(p["beta"], 6), round(p["gamma"], 6), int(p["n"])
    keep = round(1 - delta, 6)
    x = E.var("x")
    body = (
        NewSample(FRAME, nobs, None),
        Draw(FRAME, "x", "normal", 0, 1, "X ~ N(0, 1)"),
        Draw(FRAME, "v", "normal", 0, 1, "V ~ N(0, 1)"),
        Draw(FRAME, "e", "normal", 0, 1, "ε ~ N(0, 1)"),
        Derive(FRAME, "d", E.add(E.mul(gamma, x), E.var("v")), "D = γX + V"),
        Derive(FRAME, "y", E.add(E.add(E.var("d"), E.mul(beta, x)), E.var("e")), "Y = θD + βX + ε, θ = 1"),
        Derive(FRAME, "y_naif", E.sub(E.var("y"), E.mul(round(keep * beta, 6), x)),
               "Naif moment: Y − β̃X, β̃ = (1 − δ)β"),
        Derive(FRAME, "u_hat", E.sub(E.var("y"), E.mul(round(keep * (beta + gamma), 6), x)),
               "Ortogonal moment: Û = Y − η̃X, η = β + γθ, η̃ = (1 − δ)η"),
        Derive(FRAME, "v_hat", E.sub(E.var("d"), E.mul(round(keep * gamma, 6), x)), "V̂ = D − γ̃X, γ̃ = (1 − δ)γ"),
        OLS("naif", FRAME, "y_naif", ("d",), constant=False),
        OLS("ortogonal", FRAME, "u_hat", ("v_hat",), constant=False),
    )
    sd = float(np.sqrt((1 + (delta * (beta + gamma)) ** 2) / nobs))
    lower = round(1 - 5 * sd, 2)
    upper = round(1 + naive_bias(delta, beta, gamma) + 5 * sd, 2)
    naive = E.div(E.mul(E.mul(E.var("delta"), beta), gamma), 1 + gamma**2)
    orthogonal = E.div(E.mul(E.mul(E.mul(E.var("delta"), E.var("delta")), beta), gamma),
                       E.add(1, E.mul(E.mul(E.var("delta"), E.var("delta")), gamma**2)))
    return (
        MonteCarlo(
            FRAME, ORTH_REPS, ORTH_SEED, body,
            (("naif", E.coef("naif", "d")), ("sh_naif", E.se("naif", "d")),
             ("ortogonal", E.coef("ortogonal", "v_hat")), ("sh_ortogonal", E.se("ortogonal", "v_hat"))),
            "ortogonallik",
            f"{_count(ORTH_REPS)} tekrar: yardımcı parametreler aynı δ oranında hatalıyken iki momentten θ tahmini",
            coverage=(("naif", "sh_naif", 1.0), ("ortogonal", "sh_ortogonal", 1.0)),
        ),
        Histogram(
            "ortogonallik", (("naif", "Naif moment"), ("ortogonal", "Ortogonal moment")), lower, upper,
            ((1.0, "Gerçek θ = 1"),), "θ̂", f"İki momentten θ tahminleri ({_count(ORTH_REPS)} tekrar)", bins=60,
        ),
        Plot(
            "", "delta",
            (Curve(naive, "Naif: δβγ/(1 + γ²)"), Curve(orthogonal, "Ortogonal: δ²βγ/(1 + δ²γ²)"),
             VLine(delta, "Seçilen δ")),
            "Yardımcı parametre hatası δ", "θ̂'nın yanlılığı (kuram)",
            "Yanlılık: naif moment δ ile, ortogonal moment δ² ile büyür", x_range=(0.0, 0.8),
        ),
    )


def orthogonal_numbers(state: LabState) -> dict[str, float]:
    table = state.tables["ortogonallik"]
    return {
        "naive": float(table["naif"].mean() - 1),
        "orthogonal": float(table["ortogonal"].mean() - 1),
        "naive_cover": state.scalars[coverage_key("ortogonallik", "naif")],
        "orthogonal_cover": state.scalars[coverage_key("ortogonallik", "ortogonal")],
    }


def _orthogonal_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    values = orthogonal_numbers(state)
    theory_n = naive_bias(p["delta"], p["beta"], p["gamma"])
    theory_o = orthogonal_bias(p["delta"], p["beta"], p["gamma"])
    return (
        SimMetric("Naif yanlılık", _signed(values["naive"]), f"Tekrarların ortalaması − 1; kuram {plain(theory_n, 3)}."),
        SimMetric("Ortogonal yanlılık", _signed(values["orthogonal"]),
                  f"Tekrarların ortalaması − 1; kuram {plain(theory_o, 3)}."),
        SimMetric("Naif kapsama", plain(values["naive_cover"], 3), "Klasik %95 güven aralığının kapsama oranı."),
        SimMetric("Ortogonal kapsama", plain(values["orthogonal_cover"], 3), "Aynı aralık, ortogonal moment."),
    )


def _orthogonal_takeaway(state: LabState, p: Parameters) -> str:
    values = orthogonal_numbers(state)
    delta = p["delta"]
    if delta < 1e-9:
        return (
            "δ = 0: yardımcı parametreler doğru; iki moment de yansızdır. δ'yı artırın: yardımcı parametre hatası "
            "düzenlileştirmenin küçültme yanlılığına benzer."
        )
    if p["beta"] * p["gamma"] < 1e-9:
        return (
            "β·γ = 0: X ya D'yi ya da Y'yi etkilemiyor; yardımcı parametre hatası θ'ya sızmaz. β ve γ'yı birlikte "
            "artırın."
        )
    return (
        f"Yardımcı parametreler %{plain(100 * delta, 0)} küçültülmüşken naif moment θ'yı "
        f"{_signed(values['naive'])} kaydırır (kuram {plain(naive_bias(delta, p['beta'], p['gamma']), 3)}): yanlılık "
        f"δ ile doğrusal büyür, çünkü ∂m/∂β = −E[DX] ≠ 0. Ortogonal momentte yanlılık "
        f"{_signed(values['orthogonal'])} (kuram {plain(orthogonal_bias(delta, p['beta'], p['gamma']), 3)}): iki "
        "yardımcı parametre hatasının çarpımıyla orantılıdır, bu yüzden δ² ile büyür (§12.5, §12.8). n'yi büyütün: "
        "yanlılık sabit kalırken standart hata küçülür; naif aralığın kapsaması "
        f"({plain(values['naive_cover'], 3)}) daha da düşer."
    )


ORTHOGONAL = SimExperiment(
    topic_key=TOPIC, number=2,
    title="Ortogonal moment: yardımcı parametre hatasına birinci dereceden duyarsızlık",
    question=(
        "Yardımcı parametreler aynı oranda hatalı tahmin edildiğinde, naif moment ile ortogonal (artıklaştırma) "
        "moment θ'yı ne kadar yanıltır?"
    ),
    note=NoteRef("12.5", 0, ("§12.4", "§12.8")),
    parameters=(
        SimParameter("delta", "Yardımcı parametre hatası δ", 0.0, 0.8, 0.15, 0.05,
                     "β̃ = (1 − δ)β, γ̃ = (1 − δ)γ, η̃ = (1 − δ)η: düzenlileştirmedeki küçültmeye benzer.",
                     decimals=2),
        SimParameter("gamma", "X'in D'deki etkisi γ", 0.0, 2.0, 1.0, 0.1, "D = γX + V.", decimals=1),
        SimParameter("beta", "X'in Y'deki doğrudan etkisi β", 0.0, 2.0, 1.0, 0.1, "Y = θD + βX + ε.", decimals=1),
        SimParameter("n", "Gözlem sayısı n", 100, 2000, 500, 100, "Her tekrarda yeni örneklem.", integer=True,
                     decimals=0),
    ),
    dgp=lambda p: (
        rf"X,\ V,\ \varepsilon\sim N(0,1),\quad D=\gamma X+V,\quad Y=\theta D+\beta X+\varepsilon,\quad \theta=1,\ "
        rf"\gamma={number(p['gamma'], 1)},\ \beta={number(p['beta'], 1)}",
        r"\text{naif: }\hat\theta=\frac{\sum D_i(Y_i-\tilde\beta X_i)}{\sum D_i^2};\qquad "
        r"\text{ortogonal: }\hat\theta=\frac{\sum \hat V_i\hat U_i}{\sum \hat V_i^2},\ \hat V_i=D_i-\tilde\gamma X_i,\ "
        r"\hat U_i=Y_i-\tilde\eta X_i",
        rf"\tilde\beta=(1-\delta)\beta,\ \tilde\gamma=(1-\delta)\gamma,\ \tilde\eta=(1-\delta)(\beta+\gamma\theta),"
        rf"\quad \delta={number(p['delta'], 2)},\ n={int(p['n'])}",
    ),
    dgp_note=(
        "Yardımcı parametre hatası kasıtlı ve bilinen biçimde verilir; böylece yalnız momentin yapısının etkisi "
        "görülür. Kuram: naif yanlılık δβγ/(1 + γ²), ortogonal yanlılık δ²βγ/(1 + δ²γ²). "
        f"{_count(ORTH_REPS)} tekrar, tohum 81202; iki tahmin de sabitsiz EKK'dir, SH klasik."
    ),
    look_at=(
        "**Grafik 1** — iki momentten θ tahminlerinin dağılımı; kesikli çizgi gerçek θ = 1.",
        "**Grafik 2** — kuramsal yanlılık eğrileri: naif moment δ ile doğrusal, ortogonal moment δ² ile büyür.",
        "**Ölçüler** — simülasyondaki yanlılıklar (yardımda kuramsal değerler) ve güven aralıklarının kapsaması.",
    ),
    build=_build_orthogonal, metrics=_orthogonal_metrics, takeaway=_orthogonal_takeaway,
)


# --- Deney 3: gözlenmeyen karıştırıcı -------------------------------------------------------------

CONF_SEED = 81203
CONF_REPS = 500
OBSERVED = 5
TREATMENT_X = ((0.5, "x1"), (-0.5, "x2"), (0.5, "x3"))
OUTCOME_X = ((0.5, "x1"), (0.5, "x2"), (-0.5, "x4"), (0.5, "x5"))


def confounding_bias(kappa: float) -> float:
    """U dışarıda kalınca DML'in olasılık limitindeki yanlılık: κ²/(1 + κ²)."""

    return kappa**2 / (1 + kappa**2)


def _build_confounding(p: Parameters) -> tuple:
    kappa, nobs = round(p["kappa"], 6), int(p["n"])
    observed = tuple(f"x{j}" for j in range(1, OBSERVED + 1))
    treatment = _linear(TREATMENT_X)
    outcome = _linear(OUTCOME_X, start=E.var("d"))
    if kappa > 0:
        treatment = E.add(treatment, E.mul(kappa, E.var("u")))
        outcome = E.add(outcome, E.mul(kappa, E.var("u")))
    body = (
        NewSample(FRAME, nobs, None),
        DrawColumns(FRAME, "x", OBSERVED, 0.0, f"X₁, …, X_{OBSERVED}: gözlenen kontroller"),
        Draw(FRAME, "u", "normal", 0, 1, "U ~ N(0, 1): gözlenmeyen karıştırıcı"),
        Draw(FRAME, "v", "normal", 0, 1, "V ~ N(0, 1)"),
        Draw(FRAME, "e", "normal", 0, 1, "ε ~ N(0, 1)"),
        Derive(FRAME, "d", E.add(treatment, E.var("v")), "D: gözlenen X'lere ve U'ya bağlı"),
        Derive(FRAME, "y", E.add(outcome, E.var("e")), "Y = θD + ... + κU + ε, θ = 1"),
        Derive(FRAME, "kat", E.sub(E.var("id"), E.mul(5, E.floor(E.div(E.sub(E.var("id"), 1), 5)))),
               "5 dış kat: (sıra − 1) mod 5 + 1"),
        CrossFitDML("dml", FRAME, "y", "d", observed, "kat", learner="ols"),
        CrossFitDML("dml_u", FRAME, "y", "d", (*observed, "u"), "kat", learner="ols"),
    )
    shift = confounding_bias(kappa)
    spread = 5 * float(np.sqrt(2.0 / nobs))
    return (
        MonteCarlo(
            FRAME, CONF_REPS, CONF_SEED, body,
            (("dml", E.coef("dml", THETA)), ("sh_dml", E.se("dml", THETA)),
             ("dml_u", E.coef("dml_u", THETA)), ("sh_dml_u", E.se("dml_u", THETA))),
            "karistirici",
            f"{_count(CONF_REPS)} tekrar: gözlenen X'lerle DML ve (gerçekte mümkün olmayan) U'yu da içeren DML",
            coverage=(("dml", "sh_dml", 1.0), ("dml_u", "sh_dml_u", 1.0)),
        ),
        Histogram(
            "karistirici", (("dml", "DML: yalnız gözlenen X"), ("dml_u", "DML: U da gözlenseydi")),
            round(1 - spread, 2), round(1 + shift + spread, 2),
            ((1.0, "Gerçek θ = 1"), (1 + shift, "Olasılık limiti 1 + κ²/(1 + κ²)")), "θ̂",
            f"DML tahminlerinin dağılımı ({_count(CONF_REPS)} tekrar)", bins=50,
        ),
    )


def confounding_numbers(state: LabState) -> dict[str, float]:
    table = state.tables["karistirici"]
    return {
        "bias": float(table["dml"].mean() - 1),
        "bias_u": float(table["dml_u"].mean() - 1),
        "cover": state.scalars[coverage_key("karistirici", "dml")],
        "cover_u": state.scalars[coverage_key("karistirici", "dml_u")],
    }


def _confounding_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    values = confounding_numbers(state)
    return (
        SimMetric("DML yanlılığı", _signed(values["bias"]), "Tekrarların ortalaması − 1."),
        SimMetric("Kuramsal yanlılık κ²/(1 + κ²)", plain(confounding_bias(p["kappa"]), 3),
                  "U dışarıda kalınca olasılık limiti; n'den bağımsız."),
        SimMetric("DML kapsama", plain(values["cover"], 3), "%95 güven aralığının kapsama oranı."),
        SimMetric("U gözlenseydi: kapsama", plain(values["cover_u"], 3), "Gerçek uygulamada mümkün olmayan karşılaştırma."),
    )


def _confounding_takeaway(state: LabState, p: Parameters) -> str:
    values = confounding_numbers(state)
    if p["kappa"] < 1e-9:
        return (
            f"κ = 0: gözlenmeyen karıştırıcı yok; gözlenen X'ler yeterlidir. DML yanlılığı {_signed(values['bias'])}, "
            f"kapsama {plain(values['cover'], 3)}. κ'yı artırın."
        )
    return (
        f"U hem D'yi hem Y'yi etkiliyor ve veride yok. DML gözlenen X'lerin etkisini esnek biçimde ayıklar ama U'yu "
        f"ayıklayamaz: yanlılık {_signed(values['bias'])} (kuram {plain(confounding_bias(p['kappa']), 3)}), güven "
        f"aralığının kapsaması {plain(values['cover'], 3)}. Aynı hesap U'yu da görseydi yanlılık "
        f"{_signed(values['bias_u'])}, kapsama {plain(values['cover_u'], 3)} olurdu. n'yi büyütmek yanlılığı azaltmaz, "
        "yalnız güven aralığını daraltır; kapsama daha da düşer. DML bir tahmin tekniğidir, tanımlama stratejisi "
        "değildir: E[ε | D, X] ≠ 0 ise IV, RDD veya deneysel tasarım gerekir (§12.10)."
    )


CONFOUNDING = SimExperiment(
    topic_key=TOPIC, number=3,
    title="Gözlenmeyen karıştırıcı: DML tanımlama sorununu çözmez",
    question=(
        "Hem tedaviyi hem sonucu etkileyen bir değişken veride yoksa, çapraz uyarlanmış DML hedef katsayıyı yine de "
        "doğru tahmin eder mi?"
    ),
    note=NoteRef("12.10", 0, ("§12.1", "§12.7")),
    parameters=(
        SimParameter("kappa", "Gözlenmeyen karıştırıcının gücü κ", 0.0, 1.0, 0.5, 0.05,
                     "U, D'ye ve Y'ye κ katsayısıyla girer.", decimals=2),
        SimParameter("n", "Gözlem sayısı n", 200, 1000, 400, 100, "Her tekrarda yeni örneklem.", integer=True,
                     decimals=0),
    ),
    dgp=lambda p: (
        rf"X_1,\dots,X_{OBSERVED},\ U,\ V,\ \varepsilon\sim N(0,1)\ \text{{bağımsız}};\ U\ \text{{gözlenmez}}",
        rf"D=0{{,}}5X_1-0{{,}}5X_2+0{{,}}5X_3+\kappa U+V,\qquad \kappa={number(p['kappa'], 2)}",
        rf"Y=\theta D+0{{,}}5X_1+0{{,}}5X_2-0{{,}}5X_4+0{{,}}5X_5+\kappa U+\varepsilon,\quad \theta=1,\ n={int(p['n'])}",
    ),
    dgp_note=(
        "Yardımcı modeller cezasız EKK'dir (düşük boyut); 5 dış katla çapraz uyarlanır, son aşama sabitsiz, SH HC1. "
        "U dışarıda kalınca V̂ = κU + V olur ve θ̂ → 1 + κ²/(1 + κ²). İkinci tahmin U'yu da kontrol eder; gerçek "
        f"uygulamada yapılamayan bir karşılaştırmadır. {_count(CONF_REPS)} tekrar, tohum 81203."
    ),
    look_at=(
        "**Grafik** — gözlenen X'lerle DML ve U'yu da içeren DML tahminlerinin dağılımı; kesikli çizgiler gerçek θ ve "
        "kuramsal olasılık limiti.",
        "**Ölçüler** — DML yanlılığı, kuramsal değeri ve iki güven aralığının kapsaması.",
    ),
    build=_build_confounding, metrics=_confounding_metrics, takeaway=_confounding_takeaway,
)


KONU12_EXPERIMENTS = (DOUBLE, ORTHOGONAL, CONFOUNDING)
