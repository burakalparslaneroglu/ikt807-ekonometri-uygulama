"""Konu 4 Sezgi deneyleri: araçsal değişkenin gücü, geçerliliği ve yorumu.

Deney 1  Wald oranı ve dışlama kısıtı                            (Notlar §4.4, §4.2.3)
Deney 2  Zayıf araç: 2SLS'in örnekleme dağılımı (Monte Carlo)     (Notlar §4.9)
Deney 3  Heterojen etkiler: IV kimin etkisini ölçer? (LATE)       (Notlar §4.12)
"""

from __future__ import annotations

import numpy as np

from core.labs import expr as E
from core.labs.runner import LabState, coverage_key
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, number, plain
from core.labs.spec import (
    IV,
    OLS,
    Derive,
    Draw,
    Histogram,
    MeanPoints,
    ModelLine,
    MonteCarlo,
    NewSample,
    NoteRef,
    Plot,
    Scalar,
)

SEED = 804
FRAME = "sim"
TOPIC = "konu04"
BETA = 1.0


def _binary_instrument() -> tuple:
    return (
        Draw(FRAME, "uz", "uniform", 0, 1, "Aracı üretmek için tek düze çekiliş"),
        Derive(FRAME, "z", E.positive(E.sub(E.var("uz"), 0.5)), "İkili araç Z: gözlemlerin yarısında 1 (rastgele)"),
    )


# --- Deney 1: Wald oranı ve dışlama kısıtı ------------------------------------------

def iv_target(pi: float, kappa: float) -> float:
    return BETA + kappa / pi


def ols_limit_binary(pi: float) -> float:
    """β + Cov(X,U)/Var(X); Var(X) = π²/4 + 2 (Z Bernoulli(0,5), U ve v standart normal)."""

    return BETA + 1.0 / (pi * pi / 4.0 + 2.0)


def _build_wald(p: Parameters) -> tuple:
    pi, kappa = round(p["pi"], 6), round(p["kappa"], 6)
    return (
        NewSample(FRAME, int(p["n"]), SEED),
        *_binary_instrument(),
        Draw(FRAME, "u", "normal", 0, 1, "Gözlenmeyen yetenek U: hem X'i hem Y'yi etkiler"),
        Draw(FRAME, "v", "normal", 0, 1, "X'in diğer belirleyicileri"),
        Derive(FRAME, "x", E.add(E.add(E.mul(pi, E.var("z")), E.var("u")), E.var("v")), "İçsel regresör: X = πZ + U + v"),
        Draw(FRAME, "eps", "normal", 0, 1, "Yapısal hata"),
        Derive(FRAME, "y", E.add(E.add(E.add(E.mul(BETA, E.var("x")), E.mul(kappa, E.var("z"))), E.var("u")), E.var("eps")),
               "Y = βX + κZ + U + ε; κ ≠ 0 dışlama kısıtının ihlalidir"),
        OLS("ilk", FRAME, "x", ("z",)),
        OLS("indirgenmis", FRAME, "y", ("z",)),
        Scalar("wald", E.div(E.coef("indirgenmis", "z"), E.coef("ilk", "z")), "Wald oranı", percent=False, decimals=4),
        IV("iv", FRAME, "y", ("x",), ("z",)),
        OLS("ols", FRAME, "y", ("x",)),
        Plot(FRAME, "z",
             (MeanPoints("x", "X ortalaması"), MeanPoints("y", "Y ortalaması"),
              ModelLine("ilk", "İlk aşama farkı"), ModelLine("indirgenmis", "İndirgenmiş biçim farkı", dashed=True)),
             "Araç Z (0 veya 1)", "Grup ortalaması", "Wald oranı: indirgenmiş biçim farkı / ilk aşama farkı"),
    )


def _wald_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    first = float(state.models["ilk"].params["z"])
    reduced = float(state.models["indirgenmis"].params["z"])
    iv = float(state.models["iv"].params["x"])
    return (
        SimMetric("İlk aşama π̂", plain(first), "X̄(Z=1) − X̄(Z=0)."),
        SimMetric("İndirgenmiş biçim", plain(reduced), "Ȳ(Z=1) − Ȳ(Z=0); DGP'de βπ + κ."),
        SimMetric("Wald = 2SLS", plain(state.scalars["wald"]),
                  f"İndirgenmiş biçim / ilk aşama. 2SLS komutu aynı sayıyı verir: {plain(iv)}."),
        SimMetric("Hedef β + κ/π", plain(iv_target(p["pi"], p["kappa"])), "DGP'den bilinen olasılık limiti; κ = 0 iken β = 1."),
        SimMetric("OLS", plain(float(state.models["ols"].params["x"])),
                  f"U yüzünden yanlı; olasılık limiti {plain(ols_limit_binary(p['pi']), 3)}."),
    )


def _interval(state: LabState, model: str, term: str) -> tuple[float, float, str]:
    estimate = float(state.models[model].params[term])
    error = float(state.models[model].bse[term])
    low, high = estimate - 1.96 * error, estimate + 1.96 * error
    return low, high, f"%95 GA [{plain(low, 2)}; {plain(high, 2)}]"


def _wald_takeaway(state: LabState, p: Parameters) -> str:
    iv = float(state.models["iv"].params["x"])
    ols = float(state.models["ols"].params["x"])
    low, high, interval = _interval(state, "iv", "x")
    if abs(p["kappa"]) < 1e-9:
        covers = "kapsıyor" if low <= BETA <= high else "bu örneklemde kapsamıyor (tekrarların yaklaşık %5'inde beklenir)"
        return (
            f"κ = 0: araç Y'yi yalnız X üzerinden etkiliyor. Wald oranı {plain(iv, 3)}, {interval}; aralık gerçek "
            f"β = 1'i {covers}. OLS ({plain(ols, 3)}) gözlenmeyen U yüzünden yukarı yanlı. π'yi küçültün: aralık "
            "genişler, çünkü küçük bir paydaya bölüyoruz."
        )
    bias = p["kappa"] / p["pi"]
    return (
        f"κ = {plain(p['kappa'], 2)}: araç Y'yi X dışından da etkiliyor. Wald oranı β = 1 yerine β + κ/π = "
        f"{plain(iv_target(p['pi'], p['kappa']), 3)} hedefler (tahmin {plain(iv, 3)}, {interval}); yanlılık κ/π = "
        f"{plain(bias, 3)}. Aynı ihlal zayıf ilk aşamada büyür: paydaki doğrudan etki küçük bir π'ye bölünür. Güçlü ilk "
        "aşama dışlama ihlalini düzeltmez, yalnız büyütmez; dışlama kısıtı veriden test edilemez, kurumsal argümanla "
        "savunulur."
    )


WALD = SimExperiment(
    topic_key=TOPIC, number=1,
    title="Wald oranı ve dışlama kısıtı",
    question="Araç sonucu yalnız X üzerinden etkilemiyorsa Wald oranı neyi tahmin eder?",
    note=NoteRef("4.4", 0),
    parameters=(
        SimParameter("n", "Gözlem sayısı", 500, 5000, 2000, 500, "Örneklem büyüklüğü.", integer=True, decimals=0),
        SimParameter("pi", "İlk aşama farkı π", 0.1, 1.0, 0.5, 0.1, "Z = 1 grubunda X ortalaması π kadar yüksek.",
                     decimals=1),
        SimParameter("kappa", "Z'nin Y'ye doğrudan etkisi κ", -0.3, 0.3, 0.0, 0.05,
                     "κ = 0 iken dışlama kısıtı sağlanır.", decimals=2),
    ),
    dgp=lambda p: (
        r"Z_i\sim\text{Bernoulli}(0{,}5), \quad U_i,\,v_i,\,\varepsilon_i\sim N(0,1)\ \text{bağımsız}",
        rf"X_i=\pi Z_i+U_i+v_i, \quad \pi={number(p['pi'], 1)}",
        rf"Y_i=\beta X_i+\kappa Z_i+U_i+\varepsilon_i, \quad \beta=1, \ \kappa={number(p['kappa'], 2)}, \qquad "
        r"\operatorname{plim}\hat\beta_{IV}=\beta+\kappa/\pi",
    ),
    dgp_note=(
        "Araç rastgele atanmıştır (dışsal) ve π > 0 (ilgili). Gözlenmeyen U hem X'i hem Y'yi etkilediği için OLS yanlıdır. "
        "κ ≠ 0, aracın sonucu X dışındaki bir kanaldan da etkilediği durumdur: dışlama kısıtı ihlali (§4.2.3)."
    ),
    look_at=(
        "**Grafik** — Z = 0 ve Z = 1 gruplarında X ve Y ortalamaları. İlk aşama doğrusunun eğimi π̂, kesikli "
        "indirgenmiş biçim doğrusunun eğimi βπ + κ'dır; Wald oranı iki eğimin oranıdır.",
        "**Ölçüler** — ilk aşama, indirgenmiş biçim, Wald oranı ve 2SLS; DGP'den bilinen hedef β + κ/π.",
    ),
    build=_build_wald, metrics=_wald_metrics, takeaway=_wald_takeaway,
)


# --- Deney 2: zayıf araç ---------------------------------------------------------

REPS = 500


def ols_limit_weak(pi: float, rho: float) -> float:
    """β + Cov(X,u)/Var(X) = 1 + ρ_uv/(π² + 1)."""

    return BETA + rho / (pi * pi + 1.0)


def _build_weak(p: Parameters) -> tuple:
    pi, rho = round(p["pi"], 6), round(p["rho"], 6)
    body = (
        NewSample(FRAME, int(p["n"]), None),
        Draw(FRAME, "z", "normal", 0, 1, "Araç Z (dışsal)"),
        Draw(FRAME, "v", "normal", 0, 1, "İlk aşama hatası v"),
        Draw(FRAME, "eps", "normal", 0, 1, "Bağımsız şok"),
        Derive(FRAME, "u", E.add(E.mul(rho, E.var("v")), E.mul(round(float(np.sqrt(1 - rho**2)), 6), E.var("eps"))),
               "Yapısal hata u: v ile korelasyonu ρ_uv (içsellik)"),
        Derive(FRAME, "x", E.add(E.mul(pi, E.var("z")), E.var("v")), "X = πZ + v"),
        Derive(FRAME, "y", E.add(E.mul(BETA, E.var("x")), E.var("u")), "Y = βX + u, β = 1"),
        OLS("ilk", FRAME, "x", ("z",), vcov="HC1"),
        OLS("ols", FRAME, "y", ("x",)),
        IV("iv", FRAME, "y", ("x",), ("z",)),
    )
    limit = round(ols_limit_weak(pi, rho), 4)
    return (
        MonteCarlo(
            FRAME, REPS, SEED, body,
            (
                ("b_2sls", E.coef("iv", "x")),
                ("se_2sls", E.se("iv", "x")),
                ("b_ols", E.coef("ols", "x")),
                ("F", E.power(E.div(E.coef("ilk", "z"), E.se("ilk", "z")), 2)),
            ),
            "mc",
            "Monte Carlo: her tekrarda yeni örneklem; 2SLS, OLS ve ilk aşama F",
            coverage=(("b_2sls", "se_2sls", BETA),),
        ),
        Histogram(
            "mc", (("b_2sls", "2SLS"), ("b_ols", "OLS")), -1.0, 3.0,
            ((BETA, "Gerçek β = 1"), (limit, "OLS olasılık limiti")),
            "Eğim tahmini", f"{REPS} örneklemde 2SLS ve OLS tahminlerinin dağılımı", bins=40,
        ),
    )


def _outside(table, column: str) -> int:
    values = table[column]
    return int(((values < -1.0) | (values > 3.0)).sum())


def _weak_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    table = state.tables["mc"]
    concentration = p["n"] * p["pi"] ** 2
    return (
        SimMetric("Medyan ilk aşama F", plain(float(table["F"].median()), 1),
                  f"Beklenen değer yaklaşık 1 + nπ² = {plain(1 + concentration, 1)}."),
        SimMetric("2SLS medyanı", plain(float(table["b_2sls"].median()), 3), "Gerçek β = 1. Zayıf araçta OLS'ye doğru kayar."),
        SimMetric("OLS medyanı", plain(float(table["b_ols"].median()), 3),
                  f"Olasılık limiti {plain(ols_limit_weak(p['pi'], p['rho']), 3)}."),
        SimMetric("Kapsama (2SLS)", f"%{100 * state.scalars[coverage_key('mc', 'b_2sls')]:.1f}".replace(".", ","),
                  "2SLS ± 1,96·SH aralığının β = 1'i içerdiği tekrarların payı; nominal %95."),
        SimMetric("Aralık dışı 2SLS", str(_outside(table, "b_2sls")),
                  "[−1; 3] aralığının dışındaki tahmin sayısı: kalın kuyruğun göstergesi."),
    )


def _weak_takeaway(state: LabState, p: Parameters) -> str:
    table = state.tables["mc"]
    median_f = float(table["F"].median())
    median_iv = float(table["b_2sls"].median())
    limit = ols_limit_weak(p["pi"], p["rho"])
    coverage = 100 * state.scalars[coverage_key("mc", "b_2sls")]
    outside = _outside(table, "b_2sls")
    if median_f >= 10:
        return (
            f"Medyan F = {plain(median_f, 1)}: ilk aşama güçlü. 2SLS dağılımı β = 1 etrafında ve yaklaşık normal; "
            f"kapsama %{plain(coverage, 1)}. π'yi küçültün veya n'yi azaltın: yoğunlaşma parametresi nπ² düştükçe "
            "dağılım yayılır, kuyruklar kalınlaşır. Yine de F > 10 evrensel bir geçerlilik eşiği değildir."
        )
    text = (
        f"Medyan F = {plain(median_f, 1)}: araç geçerli ama zayıf. 2SLS tahminleri çok geniş ve kalın kuyruklu bir "
        f"dağılıma sahip: {outside} tekrar grafiğin [−1; 3] aralığının dışında kaldı. "
    )
    if abs(median_iv - BETA) > 0.1 and (median_iv - BETA) * (limit - BETA) > 0:
        text += f"Medyan ({plain(median_iv, 3)}) OLS'nin limitine ({plain(limit, 3)}) doğru kaymış. "
    if coverage < 93:
        text += (
            f"Güven aralığı β'yı tekrarların yalnız %{plain(coverage, 1)} kadarında kapsıyor: nominal %95'lik çıkarım "
            "güvenilmez. "
        )
    else:
        text += (
            f"Kapsama %{plain(coverage, 1)}; ancak tek tek aralıklar çok geniş ve normal yaklaşım zayıf. İçselliği ($\\rho_{{uv}}$) "
            "artırın: bozulma belirginleşir. "
        )
    return text + (
        "Zayıf araç yalnız \"standart hata biraz büyür\" sorunu değildir; tahmin edicinin dağılımı ve klasik çıkarım "
        "birlikte bozulabilir."
    )


WEAK = SimExperiment(
    topic_key=TOPIC, number=2,
    title="Zayıf araç: 2SLS'in örnekleme dağılımı",
    question="Araç geçerli ama X'i çok az hareket ettiriyorsa 2SLS tahminleri nasıl dağılır?",
    note=NoteRef("4.9", 0),
    parameters=(
        SimParameter("pi", "İlk aşama katsayısı π", 0.02, 0.5, 0.06, 0.02, "π küçüldükçe araç zayıflar.", decimals=2),
        SimParameter("n", "Her örneklemdeki gözlem sayısı", 100, 2000, 500, 100,
                     "Araç gücünü π ile birlikte belirler: yoğunlaşma parametresi nπ².", integer=True, decimals=0),
        SimParameter("rho", "İçsellik $\\rho_{uv}=\\mathrm{Corr}(u,v)$", 0.0, 0.95, 0.9, 0.05,
                     "Korelasyon büyüdükçe OLS yanlılığı ve zayıf araçta 2SLS çıkarımının bozulması artar.", decimals=2),
    ),
    dgp=lambda p: (
        rf"Z_i,\,v_i,\,\varepsilon_i\sim N(0,1)\ \text{{bağımsız}}, \qquad "
        rf"u_i=\rho_{{uv}}\,v_i+\sqrt{{1-\rho_{{uv}}^2}}\,\varepsilon_i, \quad \rho_{{uv}}={number(p['rho'], 2)}",
        rf"X_i=\pi Z_i+v_i, \quad \pi={number(p['pi'], 2)}, \qquad Y_i=\beta X_i+u_i, \quad \beta=1",
        rf"\text{{yoğunlaşma}}\ n\pi^2={number(p['n'] * p['pi'] ** 2, 1)}, \qquad "
        r"\operatorname{plim}\hat\beta_{OLS}=1+\frac{\rho_{uv}}{\pi^2+1}",
    ),
    dgp_note=(
        f"Araç tamamen geçerli: dışsal ve dışlama kısıtı sağlanıyor. Tek sorun gücü: π küçükse Z, X'teki değişimin "
        f"çok küçük bir kısmını üretir. {REPS} tekrarın her birinde yeni örneklem çekilir; 2SLS dayanıklı (HC1) "
        "standart hatayla, ilk aşama F dayanıklı t istatistiğinin karesiyle hesaplanır."
    ),
    look_at=(
        "**Grafik** — 2SLS ve OLS tahminlerinin dağılımı; gerçek β = 1 ve OLS'nin olasılık limiti (dikey çizgiler). "
        "[−1; 3] dışındaki tahminler çizilmez, sayıları ölçülerde verilir.",
        "**Ölçüler** — medyan ilk aşama F, iki tahmin edicinin medyanı ve 2SLS güven aralığının kapsama oranı. "
        "Ortalama yerine medyan: tam tanımlı 2SLS'in ortalaması sonlu olmayabilir.",
    ),
    build=_build_weak, metrics=_weak_metrics, takeaway=_weak_takeaway,
)


# --- Deney 3: LATE ------------------------------------------------------------------

EFFECT_ALWAYS = 1.5
EFFECT_NEVER = 0.5


def ate(p: Parameters) -> float:
    other = (1.0 - p["p_c"]) / 2.0
    return other * EFFECT_ALWAYS + p["p_c"] * p["tau_c"] + other * EFFECT_NEVER


def _build_late(p: Parameters) -> tuple:
    share = round((1.0 - p["p_c"]) / 2.0, 6)
    tau_c = round(p["tau_c"], 6)
    t = E.var("t")
    return (
        NewSample(FRAME, int(p["n"]), SEED),
        *_binary_instrument(),
        Draw(FRAME, "t", "uniform", 0, 1, "Tip belirleyici: her zaman alan, uyumlu veya hiç almayan"),
        Derive(FRAME, "her_zaman", E.positive(E.sub(share, t)), "Her zaman alan (always-taker): Z ne olursa olsun D = 1"),
        Derive(FRAME, "hic", E.positive(E.sub(t, 1 - share)), "Hiç almayan (never-taker): Z ne olursa olsun D = 0"),
        Derive(FRAME, "uyumlu", E.sub(E.sub(1, E.var("her_zaman")), E.var("hic")), "Uyumlu (complier): D = Z"),
        Derive(FRAME, "d", E.add(E.var("her_zaman"), E.mul(E.var("uyumlu"), E.var("z"))), "Tedavi: D = D(Z)"),
        Draw(FRAME, "e", "normal", 0, 1, "Diğer etkenler"),
        Derive(FRAME, "y0", E.add(E.sub(E.mul(0.5, E.var("her_zaman")), E.mul(0.5, E.var("hic"))), E.var("e")),
               "Y(0): her zaman alanlar daha yüksek, hiç almayanlar daha düşük başlar"),
        Derive(FRAME, "etki",
               E.add(E.add(E.mul(EFFECT_ALWAYS, E.var("her_zaman")), E.mul(tau_c, E.var("uyumlu"))),
                     E.mul(EFFECT_NEVER, E.var("hic"))),
               "Bireysel etki: tipe göre farklı (heterojen)"),
        Derive(FRAME, "y", E.add(E.var("y0"), E.mul(E.var("etki"), E.var("d"))), "Y = Y(0) + etki·D"),
        OLS("ilk", FRAME, "d", ("z",)),
        OLS("indirgenmis", FRAME, "y", ("z",)),
        IV("iv", FRAME, "y", ("d",), ("z",)),
        OLS("ols", FRAME, "y", ("d",)),
        Plot(FRAME, "z",
             (MeanPoints("d", "Tedavi alma oranı"), MeanPoints("y", "Y ortalaması"),
              ModelLine("ilk", "İlk aşama: uyumlu payı"), ModelLine("indirgenmis", "İndirgenmiş biçim", dashed=True)),
             "Araç Z (davet: 0 veya 1)", "Grup ortalaması", "Araç yalnız uyumluların tedavi durumunu değiştirir"),
    )


def _late_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    return (
        SimMetric("IV (Wald)", plain(float(state.models["iv"].params["d"])), "İndirgenmiş biçim / ilk aşama."),
        SimMetric("LATE (DGP)", plain(p["tau_c"]), "Uyumluların ortalama etkisi."),
        SimMetric("ATE (DGP)", plain(ate(p)), "Bütün anakütlenin ortalama etkisi."),
        SimMetric("İlk aşama", plain(float(state.models["ilk"].params["z"])), f"Uyumlu payı; DGP'de {plain(p['p_c'], 2)}."),
        SimMetric("OLS", plain(float(state.models["ols"].params["d"])), "Tedavi alan ve almayanların ham farkı."),
    )


def _late_takeaway(state: LabState, p: Parameters) -> str:
    iv = float(state.models["iv"].params["d"])
    gap = ate(p) - p["tau_c"]
    _, _, interval = _interval(state, "iv", "d")
    text = (
        f"IV ({plain(iv, 3)}, {interval}) uyumluların etkisini ({plain(p['tau_c'], 2)}) tahmin ediyor. Araç yalnız uyumluların "
        "tedavi durumunu değiştirdiği için, her zaman alan ve hiç almayanların etkileri Wald oranına girmez. "
    )
    if abs(gap) > 1e-9:
        text += (
            f"Bu anakütlede ATE {plain(ate(p), 3)}: IV genel ortalama etkiyi değil, LATE'i ölçer. Farklı bir araç "
            "(farklı uyumlular) farklı bir LATE verebilir. "
        )
    else:
        text += "Bu ayarda LATE ile ATE rastlantısal olarak eşit. "
    return text + (
        "OLS ise tedavi seçimi Y(0) ile ilişkili olduğu için hiçbirini tahmin etmez. Yorum monotonluğa dayanır: "
        "bu DGP'de davete rağmen tedaviden vazgeçen (defier) yoktur."
    )


LATE = SimExperiment(
    topic_key=TOPIC, number=3,
    title="Heterojen etkiler: IV kimin etkisini ölçer?",
    question="Tedavi etkisi kişiden kişiye değişiyorsa ikili araçla elde edilen Wald oranı hangi ortalamayı verir?",
    note=NoteRef("4.12", 0),
    parameters=(
        SimParameter("n", "Gözlem sayısı", 1000, 10000, 4000, 1000, "Örneklem büyüklüğü.", integer=True, decimals=0),
        SimParameter("p_c", "Uyumlu payı", 0.2, 0.8, 0.4, 0.1,
                     "Kalan pay her zaman alan ve hiç almayanlar arasında eşit bölünür.", decimals=1),
        SimParameter("tau_c", "Uyumluların etkisi $\\tau_C$", 0.0, 2.0, 0.5, 0.25,
                     f"Her zaman alanlarda etki {EFFECT_ALWAYS:g}, hiç almayanlarda {EFFECT_NEVER:g}.".replace(".", ","),
                     decimals=2),
    ),
    dgp=lambda p: (
        r"Z_i\sim\text{Bernoulli}(0{,}5)\ \text{(rastgele davet)}, \qquad D_i=D_i(Z_i)",
        rf"\text{{uyumlu}}: D(1)=1,\,D(0)=0 \ (\text{{pay}}={number(p['p_c'], 1)}); \quad "
        r"\text{her zaman alan}: D\equiv1; \quad \text{hiç almayan}: D\equiv0",
        rf"Y_i=Y_i(0)+\tau_i D_i, \qquad \tau_i\in\{{1{{,}}5;\ \tau_C;\ 0{{,}}5\}}, \quad \tau_C={number(p['tau_c'], 2)}",
    ),
    dgp_note=(
        "Etkiler tipe göre farklı: her zaman alanlar tedaviden en çok, hiç almayanlar en az yararlanıyor. Başlangıç "
        "düzeyi Y(0) de tipe göre değişiyor; tedavi seçimi bu yüzden içsel. Defier yok (monotonluk sağlanıyor)."
    ),
    look_at=(
        "**Grafik** — davet edilen (Z = 1) ve edilmeyen (Z = 0) gruplarda tedavi alma oranı ve Y ortalaması. "
        "Tedavi alma oranındaki fark uyumlu payıdır; Y'deki fark yalnız uyumluların tedavisinden gelir.",
        "**Ölçüler** — IV (Wald), DGP'den bilinen LATE ve ATE, ilk aşama ve OLS.",
    ),
    build=_build_late, metrics=_late_metrics, takeaway=_late_takeaway,
)


KONU04_EXPERIMENTS = (WALD, WEAK, LATE)
