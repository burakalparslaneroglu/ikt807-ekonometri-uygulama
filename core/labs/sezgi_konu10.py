"""Konu 10 Sezgi deneyleri: bootstrap standart hata ve güven aralıkları, dönüşümler, yeniden örnekleme birimi.

Deney 1  Heteroskedastik regresyon: analitik, pairs ve wild bootstrap       (Notlar §10.11, §10.8, §10.10)
Deney 2  Doğrusal olmayan dönüşüm: delta yöntemi ve percentile aralığı      (Notlar §10.7, §10.6)
Deney 3  Kümeli veri: gözlem mi küme mi yeniden örneklenmeli?               (Notlar §10.14, §10.5)

Deney 1 varsayılan ayarlarıyla notlardaki Tablo 10.1'i birebir yeniden üretir: tek üreteç
``np.random.default_rng(807)``; önce X ve ε, sonra pairs (B = 3.000), en son wild (B = 3.000) çekilişleri.
"""

from __future__ import annotations

import numpy as np

from core.labs import expr as E
from core.labs.runner import LabState, bootstrap_key
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, number, plain
from core.labs.spec import (
    BOOT,
    OLS,
    Bootstrap,
    ClusterDraw,
    DeltaMethod,
    Derive,
    Draw,
    Histogram,
    NewSample,
    NoteRef,
    Scalar,
    ScalarTable,
)

SEED = 807
FRAME = "sim"
TOPIC = "konu10"
MULTIPLIER = 1.96


def _key(result: str, column: str, statistic: str) -> str:
    return bootstrap_key(result, column, statistic)


def _count(value: int) -> str:
    """Türkçe binlik ayraçlı tamsayı: 3000 → "3.000"."""

    return f"{int(value):,}".replace(",", ".")


# --- Deney 1: §10.11 heteroskedastik örnek ---------------------------------------------------------

SLOPE = 1.8


def _build_hetero(p: Parameters) -> tuple:
    gamma, reps = round(p["gamma"], 6), int(p["reps"])
    x = E.var("x")
    slope, hc1_se = E.coef("ols", "x"), E.se("ols_hc1", "x")
    return (
        NewSample(FRAME, int(p["n"]), SEED),
        Draw(FRAME, "x", "normal", 0, 1, "X ~ N(0, 1)"),
        Draw(FRAME, "e", "normal", 0, 1, "ε ~ N(0, 1)"),
        Derive(FRAME, "y", E.add(E.add(1, E.mul(SLOPE, x)), E.mul(E.add(0.45, E.mul(gamma, E.absolute(x))), E.var("e"))),
               "Y = 1 + 1,8X + (0,45 + γ|X|)ε: hata varyansı X'e bağlı"),
        OLS("ols", FRAME, "y", ("x",)),
        OLS("ols_hc1", FRAME, "y", ("x",), vcov="HC1"),
        Bootstrap(
            "ols", FRAME, reps, None,
            (("b", E.coef(BOOT, "x")), ("t", E.div(E.sub(E.coef(BOOT, "x"), slope), E.se(BOOT, "x")))),
            "pairs", f"Pairs bootstrap, B = {_count(reps)}; her tekrarda eğim ve percentile-t için t* = (b* − b̂)/SH*(HC1)",
        ),
        Bootstrap(
            "ols", FRAME, reps, None, (("b", E.coef(BOOT, "x")),), "wild",
            f"Wild bootstrap, B = {_count(reps)}: regresörler sabit, artıklar Rademacher çarpanıyla", method="wild",
        ),
        Scalar("normal_alt", E.sub(slope, E.mul(MULTIPLIER, hc1_se)), "HC1 normal %95 GA alt", percent=False, decimals=3),
        Scalar("normal_ust", E.add(slope, E.mul(MULTIPLIER, hc1_se)), "HC1 normal %95 GA üst", percent=False, decimals=3),
        Scalar("pt_alt", E.sub(slope, E.mul(E.ref(_key("pairs", "t", "hi")), hc1_se)),
               "Percentile-t %95 GA alt: b̂ − q*(0,975)·SH", percent=False, decimals=3),
        Scalar("pt_ust", E.sub(slope, E.mul(E.ref(_key("pairs", "t", "lo")), hc1_se)),
               "Percentile-t %95 GA üst: b̂ − q*(0,025)·SH", percent=False, decimals=3),
        ScalarTable(
            (
                ("Eğim tahmini", slope),
                ("Geleneksel OLS SH", E.se("ols", "x")),
                ("HC1 SH", hc1_se),
                ("Pairs bootstrap SH", E.ref(_key("pairs", "b", "se"))),
                ("Wild bootstrap SH", E.ref(_key("wild", "b", "se"))),
                ("HC1 normal %95 GA: alt", E.ref("normal_alt")),
                ("HC1 normal %95 GA: üst", E.ref("normal_ust")),
                ("Percentile %95 GA: alt", E.ref(_key("pairs", "b", "lo"))),
                ("Percentile %95 GA: üst", E.ref(_key("pairs", "b", "hi"))),
                ("Percentile-t %95 GA: alt", E.ref("pt_alt")),
                ("Percentile-t %95 GA: üst", E.ref("pt_ust")),
            ),
            "ozet",
        ),
        Histogram(
            "pairs", (("b", "Pairs bootstrap eğimi"),), 1.3, 2.3, ((SLOPE, "Gerçek eğim β₁ = 1,8"),),
            "Bootstrap eğim tahmini", f"Pairs bootstrap eğimlerinin dağılımı (B = {_count(reps)})", bins=50,
        ),
    )


def _hetero_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    return (
        SimMetric("Geleneksel SH", plain(float(state.models["ols"].bse["x"]), 3), "Homoskedastisite varsayar."),
        SimMetric("HC1 SH", plain(float(state.models["ols_hc1"].bse["x"]), 3), "Heteroskedastisiteye dayanıklı."),
        SimMetric("Pairs bootstrap SH", plain(state.scalars[_key("pairs", "b", "se")], 3), "Satırlar yeniden örneklenir."),
        SimMetric("Wild bootstrap SH", plain(state.scalars[_key("wild", "b", "se")], 3),
                  "Regresörler sabit, artıkların işareti rastgele."),
    )


def _hetero_takeaway(state: LabState, p: Parameters) -> str:
    classic = float(state.models["ols"].bse["x"])
    hc1 = float(state.models["ols_hc1"].bse["x"])
    pairs = state.scalars[_key("pairs", "b", "se")]
    wild = state.scalars[_key("wild", "b", "se")]
    if p["gamma"] < 1e-9:
        return (
            f"γ = 0: hata homoskedastik. Geleneksel SH {plain(classic, 3)}, HC1 {plain(hc1, 3)}, pairs {plain(pairs, 3)}, "
            f"wild {plain(wild, 3)}: dördü de aynı belirsizliği ölçer ve yakındır. γ'yı artırın."
        )
    return (
        f"Hata varyansı |X| ile büyüyor: geleneksel SH {plain(classic, 3)} belirsizliği küçümser; HC1 {plain(hc1, 3)}, "
        f"pairs bootstrap {plain(pairs, 3)}, wild bootstrap {plain(wild, 3)}. Pairs bootstrap her gözlemin (X, Y) çiftini "
        "birlikte taşıdığı, wild bootstrap her gözlemin artığını kendi yerinde tuttuğu için heteroskedastisiteyi korur "
        "(§10.5, §10.10). İyi kurulmuş bootstrap ile doğru analitik formül aynı örnekleme belirsizliğini hedefler; "
        "benzer çıkmaları beklenir. Varsayılan ayarlar notlardaki Tablo 10.1'i yeniden üretir (§10.11)."
    )


HETERO = SimExperiment(
    topic_key=TOPIC, number=1,
    title="Heteroskedastik regresyon: analitik, pairs ve wild bootstrap",
    question=(
        "Hata varyansı X ile değiştiğinde geleneksel standart hata, HC1 ve bootstrap standart hataları nasıl "
        "karşılaştırılır; üç güven aralığı ne kadar farklıdır?"
    ),
    note=NoteRef("10.11", 0, ("§10.8", "§10.10")),
    parameters=(
        SimParameter("gamma", "Heteroskedastisite γ", 0.0, 1.5, 0.55, 0.05, "u = (0,45 + γ|X|)ε; γ = 0 homoskedastik.",
                     decimals=2),
        SimParameter("n", "Gözlem sayısı", 60, 960, 240, 60, "Örneklem büyüklüğü.", integer=True, decimals=0),
        SimParameter("reps", "Bootstrap tekrarı B", 1000, 5000, 3000, 500, "Her iki bootstrap için.",
                     integer=True, decimals=0),
    ),
    dgp=lambda p: (
        r"X_i,\ \varepsilon_i\sim N(0,1),\qquad u_i=(0{,}45+\gamma|X_i|)\,\varepsilon_i",
        rf"Y_i=1+1{{,}}8X_i+u_i,\qquad \gamma={number(p['gamma'], 2)},\ n={int(p['n'])},\ B={int(p['reps'])}",
        r"\text{wild: }Y_i^*=X_i'\hat\beta+\hat e_i\xi_i^*,\quad P(\xi_i^*=\pm1)=1/2;\qquad "
        r"\text{percentile-}t\text{: }[\hat\beta-q^*_{0{,}975}s,\ \hat\beta-q^*_{0{,}025}s]",
    ),
    dgp_note=(
        "Rastgele sayılar tek bir üreteçten (np.random.default_rng(807)) sırayla çekilir: önce X ve ε, sonra pairs, "
        "en son wild bootstrap çekilişleri. Varsayılan ayarlar (γ = 0,55, n = 240, B = 3.000) notlardaki Tablo 10.1'i "
        "birebir verir. Percentile-t'de her tekrarın standart hatası HC1'dir; s orijinal HC1 SH'sidir."
    ),
    look_at=(
        "**Grafik** — pairs bootstrap eğimlerinin dağılımı; gerçek eğim 1,8 (dikey çizgi).",
        "**Ölçüler** — geleneksel, HC1, pairs ve wild bootstrap standart hataları.",
        "**Tablo** — notlardaki Tablo 10.1'in bütün satırları: eğim, dört standart hata ve üç güven aralığı.",
    ),
    build=_build_hetero, metrics=_hetero_metrics, takeaway=_hetero_takeaway,
    tables=(("ozet", "Analitik ve bootstrap belirsizlik ölçüleri (notlarda Tablo 10.1)"),),
)


# --- Deney 2: doğrusal olmayan dönüşüm ---------------------------------------------------------------

TRANSFORM_REPS = 2000
TRANSFORM_SEED = 1007


def percent_effect(beta: float) -> float:
    return 100.0 * (np.exp(beta) - 1.0)


def _percent(coefficient: E.Expr) -> E.Expr:
    return E.mul(100, E.sub(E.exp(coefficient), 1))


def _build_transform(p: Parameters) -> tuple:
    beta = round(p["beta"], 6)
    return (
        NewSample(FRAME, int(p["n"]), TRANSFORM_SEED),
        Draw(FRAME, "u", "uniform", 0, 1, "U ~ U[0, 1]"),
        Derive(FRAME, "d", E.compare("lt", E.var("u"), 0.5), "D = 1{U < 0,5}: grup göstergesi"),
        Draw(FRAME, "e", "normal", 0, 1, "e ~ N(0, 1)"),
        Derive(FRAME, "y", E.add(E.add(1, E.mul(beta, E.var("d"))), E.var("e")),
               "log sonuç: Y = 1 + βD + e"),
        OLS("ols", FRAME, "y", ("d",), vcov="HC1"),
        DeltaMethod("yuzde", "ols", _percent(E.coef("ols", "d")), "Yüzde etki g(β) = 100·(exp(β) − 1)"),
        Bootstrap(
            "ols", FRAME, TRANSFORM_REPS, None,
            (("b", E.coef(BOOT, "d")), ("g", _percent(E.coef(BOOT, "d")))), "boot",
            f"Pairs bootstrap, B = {_count(TRANSFORM_REPS)}: her tekrarda β* ve g(β*) = 100·(exp(β*) − 1)",
        ),
        Scalar("delta_alt", E.sub(E.ref("yuzde"), E.mul(MULTIPLIER, E.ref("yuzde_se"))),
               "Delta yöntemi %95 GA alt", percent=False, decimals=1),
        Scalar("delta_ust", E.add(E.ref("yuzde"), E.mul(MULTIPLIER, E.ref("yuzde_se"))),
               "Delta yöntemi %95 GA üst", percent=False, decimals=1),
        Scalar("donusum_alt", _percent(E.ref(_key("boot", "b", "lo"))),
               "β'nın percentile alt sınırına g uygulanmış", percent=False, decimals=1),
        Scalar("donusum_ust", _percent(E.ref(_key("boot", "b", "hi"))),
               "β'nın percentile üst sınırına g uygulanmış", percent=False, decimals=1),
        ScalarTable(
            (
                ("Tahmin g(β̂)", E.ref("yuzde")),
                ("Delta yöntemi %95 GA: alt", E.ref("delta_alt")),
                ("Delta yöntemi %95 GA: üst", E.ref("delta_ust")),
                ("Percentile %95 GA: alt", E.ref(_key("boot", "g", "lo"))),
                ("Percentile %95 GA: üst", E.ref(_key("boot", "g", "hi"))),
                ("β'nın percentile alt ucuna g uygulanmış", E.ref("donusum_alt")),
                ("β'nın percentile üst ucuna g uygulanmış", E.ref("donusum_ust")),
            ),
            "araliklar", decimals=1,
        ),
        Histogram(
            "boot", (("g", "Bootstrap yüzde etki g(β*)"),), -100.0, 500.0,
            ((percent_effect(beta), f"Gerçek g(β) = {plain(percent_effect(beta), 1)}"),),
            "Yüzde etki g(β*) = 100·(exp(β*) − 1)", f"g(β*)'nin bootstrap dağılımı (B = {_count(TRANSFORM_REPS)})", bins=60,
        ),
    )


def _transform_numbers(state: LabState) -> dict[str, float]:
    scalars = state.scalars
    return {
        "estimate": scalars["yuzde"],
        "delta": (scalars["delta_alt"], scalars["delta_ust"]),
        "percentile": (scalars[_key("boot", "g", "lo")], scalars[_key("boot", "g", "hi")]),
        "mapped": (scalars["donusum_alt"], scalars["donusum_ust"]),
    }


def _interval(pair: tuple[float, float]) -> str:
    return f"[{plain(pair[0], 1)}; {plain(pair[1], 1)}]"


def _transform_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    values = _transform_numbers(state)
    low, high = values["percentile"]
    return (
        SimMetric("Gerçek g(β)", plain(percent_effect(p["beta"]), 1), "DGP'deki yüzde etki."),
        SimMetric("Tahmin g(β̂)", plain(values["estimate"], 1), "Bu örneklemde."),
        SimMetric("Delta SH", plain(state.scalars["yuzde_se"], 1), "g'(β̂)·SH(β̂), HC1 ile."),
        SimMetric("Percentile asimetrisi", plain((high - values["estimate"]) / (values["estimate"] - low), 2),
                  "Üst ucun tahmine uzaklığının alt uca oranı; 1 simetrik aralıktır."),
    )


def _transform_takeaway(state: LabState, p: Parameters) -> str:
    values = _transform_numbers(state)
    return (
        f"Delta yöntemi g(β̂) = {plain(values['estimate'], 1)} çevresinde simetrik bir aralık kurar: "
        f"{_interval(values['delta'])}. Percentile aralığı {_interval(values['percentile'])} bootstrap dağılımının sağa "
        "çarpık biçimini taşır: üstel dönüşüm büyük β*'ları daha çok büyütür. β'nın percentile uçlarına g uygulamak "
        f"aynı aralığı verir ({_interval(values['mapped'])}): percentile aralığı monoton dönüşümlere uyumludur (§10.7). "
        "Delta yöntemi alt sınırı gerçekçi olmayan biçimde aşağı çekebilir. n'yi büyütün veya β'yı küçültün: dönüşüm "
        "yerel olarak doğrusala yaklaşır ve iki aralık birbirine yaklaşır."
    )


TRANSFORM = SimExperiment(
    topic_key=TOPIC, number=2,
    title="Doğrusal olmayan dönüşüm: delta yöntemi ve percentile aralığı",
    question=(
        r"Log modelde katsayının yüzde etkiye dönüşümü $g(\beta)=100\,(e^{\beta}-1)$ için delta yöntemi ve bootstrap "
        "percentile aralıkları neden farklıdır?"
    ),
    note=NoteRef("10.7", 0, ("§10.6",)),
    parameters=(
        SimParameter("beta", "Gerçek β", 0.0, 1.2, 0.8, 0.1, "Grup farkı (log ölçek).", decimals=1),
        SimParameter("n", "Gözlem sayısı", 40, 400, 80, 20, "Küçük n'de dönüşümün eğriliği belirginleşir.",
                     integer=True, decimals=0),
    ),
    dgp=lambda p: (
        r"U\sim U[0,1],\quad D=\mathbf 1\{U<0{,}5\},\quad e\sim N(0,1)",
        rf"Y=1+\beta D+e,\qquad \beta={number(p['beta'], 1)},\qquad g(\beta)=100\,(e^{{\beta}}-1)={number(percent_effect(p['beta']), 1)}",
        r"\text{delta: } g(\hat\beta)\pm1{,}96\,g'(\hat\beta)\,SH(\hat\beta),\qquad \text{percentile: } "
        r"g(\hat\beta^*)\ \text{dağılımının}\ [q^*_{0{,}025},\,q^*_{0{,}975}]\ \text{yüzdelikleri}",
    ),
    dgp_note=(
        r"Y log ölçekte bir sonuçtur; $g(\beta)=100\,(e^{\beta}-1)$ iki grup arasındaki yüzde farktır. Delta yöntemi "
        f"standart hatası HC1 kovaryansıyla hesaplanır. Pairs bootstrap B = {_count(TRANSFORM_REPS)}, aynı üreteçle "
        "(tohum 1007) veri çekilişlerinin ardından."
    ),
    look_at=(
        "**Grafik** — g(β*)'nin bootstrap dağılımı ve gerçek g(β).",
        "**Ölçüler** — gerçek ve tahmin edilen yüzde etki, delta standart hatası ve percentile aralığının asimetrisi.",
        "**Tablo** — delta yöntemi ve percentile aralıkları; β aralığının uçlarına g uygulanmış hali.",
    ),
    build=_build_transform, metrics=_transform_metrics, takeaway=_transform_takeaway,
    tables=(("araliklar", "Yüzde etki için iki %95 güven aralığı"),),
)


# --- Deney 3: kümeli veride yeniden örnekleme birimi ------------------------------------------------

CLUSTER_SIZE = 20
CLUSTER_SEED = 1010
CLUSTER_REPS = 1000


def _build_cluster(p: Parameters) -> tuple:
    groups, rho = int(p["groups"]), round(p["rho"], 6)
    return (
        NewSample(FRAME, groups * CLUSTER_SIZE, CLUSTER_SEED),
        Derive(FRAME, "kume", E.add(E.floor(E.div(E.sub(E.var("id"), 1), CLUSTER_SIZE)), 1),
               f"Küme numarası: her kümede {CLUSTER_SIZE} gözlem"),
        ClusterDraw(FRAME, "x", "kume", groups, 0, 1, "Küme düzeyinde açıklayıcı değişken (ör. okulun bir özelliği)"),
        ClusterDraw(FRAME, "u", "kume", groups, 0, 1, "Küme şoku"),
        Draw(FRAME, "eps", "normal", 0, 1, "Bireysel şok"),
        Derive(FRAME, "e", E.add(E.mul(round(float(np.sqrt(rho)), 6), E.var("u")),
                                 E.mul(round(float(np.sqrt(1 - rho)), 6), E.var("eps"))),
               "Hata: küme içi korelasyonu ρ, varyansı 1"),
        Derive(FRAME, "y", E.add(E.add(1, E.mul(0.5, E.var("x"))), E.var("e")), "Y = 1 + 0,5x + e"),
        OLS("ols", FRAME, "y", ("x",), vcov="HC1"),
        OLS("kume_sh", FRAME, "y", ("x",), vcov="cluster", cluster="kume"),
        Bootstrap("ols", FRAME, CLUSTER_REPS, None, (("b", E.coef(BOOT, "x")),), "cift",
                  f"Pairs bootstrap, B = {_count(CLUSTER_REPS)}: gözlemler tek tek yeniden örneklenir"),
        Bootstrap("ols", FRAME, CLUSTER_REPS, None, (("b", E.coef(BOOT, "x")),), "kume",
                  f"Küme bootstrap'ı, B = {_count(CLUSTER_REPS)}: kümeler bütün gözlemleriyle yeniden örneklenir",
                  method="cluster", cluster="kume"),
        Histogram("cift", (("b", "Pairs bootstrap (gözlem)"),), -0.5, 1.5, ((0.5, "Gerçek eğim 0,5"),),
                  "Bootstrap eğim tahmini", "Gözlemleri yeniden örneklemek", bins=50),
        Histogram("kume", (("b", "Küme bootstrap'ı"),), -0.5, 1.5, ((0.5, "Gerçek eğim 0,5"),),
                  "Bootstrap eğim tahmini", "Kümeleri yeniden örneklemek", bins=50),
    )


def conditional_sd(state: LabState, rho: float) -> float:
    """DGP'den bilinen koşullu standart sapma: √[(X'X)⁻¹X'ΩX(X'X)⁻¹]₂₂, Ω = (1 − ρ)I + ρ·(küme içi birler)."""

    frame = state.frames[FRAME]
    design = np.column_stack([np.ones(len(frame)), frame["x"].to_numpy(dtype=float)])
    inverse = np.linalg.inv(design.T @ design)
    sums = frame.assign(sabit=1.0).groupby("kume")[["sabit", "x"]].sum().to_numpy(dtype=float)
    middle = (1 - rho) * design.T @ design + rho * sums.T @ sums
    return float(np.sqrt((inverse @ middle @ inverse)[1, 1]))


def _cluster_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    return (
        SimMetric("HC1 SH", plain(float(state.models["ols"].bse["x"]), 3), "Gözlemleri bağımsız sayar."),
        SimMetric("Pairs bootstrap SH", plain(state.scalars[_key("cift", "b", "se")], 3), "Gözlem yeniden örnekleme."),
        SimMetric("Küme SH (CR1)", plain(float(state.models["kume_sh"].bse["x"]), 3), "Küme-dayanıklı analitik."),
        SimMetric("Küme bootstrap SH", plain(state.scalars[_key("kume", "b", "se")], 3), "Küme yeniden örnekleme."),
        SimMetric("Gerçek SD (DGP)", plain(conditional_sd(state, p["rho"]), 3),
                  "Bu x'ler verildiğinde eğimin gerçek (koşullu) standart sapması."),
    )


def _cluster_takeaway(state: LabState, p: Parameters) -> str:
    pairs = state.scalars[_key("cift", "b", "se")]
    cluster = state.scalars[_key("kume", "b", "se")]
    truth = conditional_sd(state, p["rho"])
    if p["rho"] < 1e-9:
        return (
            f"ρ = 0: gözlemler bağımsız. Pairs bootstrap {plain(pairs, 3)}, küme bootstrap'ı {plain(cluster, 3)}; gerçek "
            f"{plain(truth, 3)}. Bu durumda ikisi de geçerlidir; küme bootstrap'ı yalnız daha gürültülüdür. ρ'yu artırın."
        )
    return (
        f"Hatalar küme içinde ilişkili ve x küme düzeyinde: {int(p['groups']) * CLUSTER_SIZE} gözlemin taşıdığı bağımsız "
        f"bilgi {int(p['groups'])} küme kadardır. Pairs bootstrap gözlemleri tek tek yeniden örneklediği için bağımlılığı "
        f"bozar ve belirsizliği küçümser: {plain(pairs, 3)}; küme bootstrap'ı {plain(cluster, 3)}, gerçek koşullu standart "
        f"sapma {plain(truth, 3)}. Yeniden örnekleme birimi bağımsız birimle aynı olmalıdır (§10.14). Küme sayısını "
        "azaltın: küme bootstrap'ı da, küme SH'si de az kümeyle gürültülüleşir."
    )


CLUSTER = SimExperiment(
    topic_key=TOPIC, number=3,
    title="Kümeli veri: gözlem mi küme mi yeniden örneklenmeli?",
    question="Hatalar küme içinde ilişkiliyken gözlemleri tek tek yeniden örneklemek standart hatayı nasıl yanıltır?",
    note=NoteRef("10.14", 0, ("§10.5",)),
    parameters=(
        SimParameter("groups", "Küme sayısı G", 20, 100, 40, 10, f"Her kümede {CLUSTER_SIZE} gözlem.", integer=True,
                     decimals=0),
        SimParameter("rho", "Hatanın küme içi korelasyonu ρ", 0.0, 0.8, 0.3, 0.1, "ρ = 0 iken hatalar bağımsız.",
                     decimals=1),
    ),
    dgp=lambda p: (
        rf"g=1,\dots,G,\quad G={int(p['groups'])},\quad m={CLUSTER_SIZE}\ \text{{gözlem/küme}}",
        r"X_{ig}=x_g,\quad x_g\sim N(0,1),\qquad e_{ig}=\sqrt{\rho}\,u_g+\sqrt{1-\rho}\,\varepsilon_{ig}",
        rf"Y_{{ig}}=1+0{{,}}5\,X_{{ig}}+e_{{ig}},\qquad \rho={number(p['rho'], 1)}",
    ),
    dgp_note=(
        f"Konu 2'deki kümelenmiş veri deneyinin bootstrap karşılığı. Her iki bootstrap B = {_count(CLUSTER_REPS)}; aynı üreteç "
        "(tohum 1010) veri çekilişlerinin ardından önce pairs, sonra küme bootstrap'ı için kullanılır. Gerçek koşullu "
        "standart sapma, bu x'ler verildiğinde DGP'nin kovaryans matrisinden hesaplanır."
    ),
    look_at=(
        "**Grafikler** — aynı veride gözlemleri ve kümeleri yeniden örneklemenin verdiği eğim dağılımları.",
        "**Ölçüler** — HC1, pairs bootstrap, küme SH (CR1), küme bootstrap'ı ve DGP'den bilinen gerçek koşullu standart "
        "sapma.",
    ),
    build=_build_cluster, metrics=_cluster_metrics, takeaway=_cluster_takeaway,
)


KONU10_EXPERIMENTS = (HETERO, TRANSFORM, CLUSTER)
