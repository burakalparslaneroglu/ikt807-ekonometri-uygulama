"""Konu 6 Sezgi deneyleri: sansürleme, örneklem seçimi ve dışlama kısıtı.

Deney 1  Sansürleme: tam örneklem OLS, pozitif OLS ve Tobit      (Notlar §6.4.1, Tablo 6.2)
Deney 2  Endojen seçim ve Heckman iki aşama                     (Notlar §6.13, Tablo 6.3)
Deney 3  Dışlama kısıtı olmadan Heckman (Monte Carlo)            (Notlar §6.12)

Deney 1 ve 2'nin varsayılan ayarları (tohum 807) notlardaki Tablo 6.2 ve 6.3'ü birebir üretir.
"""

from __future__ import annotations

import numpy as np

from core.labs import expr as E
from core.labs.runner import LabState
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, number, plain
from core.labs.spec import (
    OLS,
    BinaryChoice,
    Curve,
    Derive,
    Draw,
    Histogram,
    ModelLine,
    MonteCarlo,
    NewSample,
    NoteRef,
    Plot,
    Predict,
    Scatter,
    Tobit,
)

SEED = 807
"""Notlardaki benzetim tablolarının tohumu: varsayılan ayarlar Tablo 6.2 ve 6.3'ü üretir."""
MC_SEED = 806
FRAME = "sim"
TOPIC = "konu06"


# --- Deney 1: sansürleme ----------------------------------------------------------------------

def _build_censoring(p: Parameters) -> tuple:
    alpha, sigma = round(p["alpha"], 6), round(p["sigma"], 6)
    latent = E.add(alpha, E.var("x"))
    z = E.div(latent, sigma)
    observed_mean = E.add(E.mul(E.normcdf(z), latent), E.mul(sigma, E.normpdf(z)))
    return (
        NewSample(FRAME, int(p["n"]), SEED),
        Draw(FRAME, "x", "uniform", -3, 3, "X ~ U[−3, 3]"),
        Draw(FRAME, "e", "normal", 0, sigma, "Gizli hata e ~ N(0, σ²)"),
        Derive(FRAME, "ystar", E.add(latent, E.var("e")), "Gizli sonuç Y* = α + X + e (gerçek eğim 1)"),
        Derive(FRAME, "y", E.maximum(E.var("ystar"), 0), "Gözlenen sonuç Y = max{Y*, 0}: sıfırda sansürlü"),
        Derive(FRAME, "pozitif", E.positive(E.var("y")), "Sansürlenmemiş gözlem göstergesi"),
        OLS("tam", FRAME, "y", ("x",)),
        OLS("pozitif_ols", FRAME, "y", ("x",), where=("pozitif", 1.0)),
        Tobit("tobit", FRAME, "y", ("x",), 0.0),
        Plot(FRAME, "x",
             (Scatter("y", "Gözlenen Y"),
              Curve(latent, "Gizli ortalama m*(x) = α + x (gerçek)", dashed=True, color=0),
              Curve(observed_mean, "Gözlenen ortalama m(x) (gerçek)", color=1),
              ModelLine("tam", "Tam örneklem OLS"),
              ModelLine("pozitif_ols", "Yalnız Y > 0 OLS"),
              ModelLine("tobit", "Tobit (gizli ortalama)")),
             "X", "Y", "Sansürlü veri: üç tahmin ve iki gerçek koşullu ortalama"),
    )


def censored_share(state: LabState) -> float:
    return 1.0 - float(state.frames[FRAME]["pozitif"].mean())


def _censoring_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    share = censored_share(state)
    return (
        SimMetric("Sansürlenme oranı", f"%{100 * share:.1f}".replace(".", ","), "Y* ≤ 0 olan gözlemlerin payı."),
        SimMetric("Tam örneklem OLS eğimi", plain(float(state.models["tam"].params["x"]), 3),
                  "m(x)'in doğrusal projeksiyonu; gerçek gizli eğim 1."),
        SimMetric("Yalnız Y > 0 OLS eğimi", plain(float(state.models["pozitif_ols"].params["x"]), 3),
                  "Kesilmiş (truncated) ortalamanın projeksiyonu."),
        SimMetric("Tobit eğimi", plain(float(state.models["tobit"].params["x"]), 3),
                  "Doğru belirtilmiş olabilirlik: gizli eğimi hedefler."),
        SimMetric("Tobit σ̂", plain(float(state.models["tobit"].sigma), 3), f"Gerçek σ = {plain(p['sigma'], 2)}."),
    )


def _censoring_takeaway(state: LabState, p: Parameters) -> str:
    share = censored_share(state)
    full = float(state.models["tam"].params["x"])
    positive = float(state.models["pozitif_ols"].params["x"])
    tobit = float(state.models["tobit"].params["x"])
    return (
        f"Gözlemlerin %{plain(100 * share, 1)} kadarı sıfırda sansürlü. Tam örneklem OLS ({plain(full, 3)}) gözlenen "
        f"ortalama m(x)'e bir doğru uydurur; sıfırları silmek ({plain(positive, 3)}) kesilmiş ortalamayı hedefler. İkisi de "
        f"gizli eğimi (1) geri getirmez. Tobit ({plain(tobit, 3)}) sansürleme mekanizmasını olabilirliğe kattığı için "
        "hedefe yakındır; ancak bu üstünlük normallik ve homoskedastisite varsayımlarına dayanır. α'yı düşürün: "
        "sansürlenme arttıkça OLS eğimleri düzleşir (§6.4)."
    )


CENSORING = SimExperiment(
    topic_key=TOPIC, number=1,
    title="Sansürleme: tam örneklem OLS, pozitif OLS ve Tobit",
    question="Sonuç sıfırda sansürlüyse OLS, sıfırları silmek ve Tobit neyi tahmin eder?",
    note=NoteRef("6.4.1", 0, ("Tablo 6.2",)),
    parameters=(
        SimParameter("n", "Gözlem sayısı", 1000, 10000, 5000, 1000, "Örneklem büyüklüğü.", integer=True, decimals=0),
        SimParameter("alpha", "Sabit α", -2.0, 3.0, 1.0, 0.25,
                     "Gizli ortalamayı kaydırır; küçüldükçe sansürlenme artar.", decimals=2),
        SimParameter("sigma", "Hata standart sapması σ", 0.5, 2.0, 1.0, 0.25, "Gizli hatanın ölçeği.", decimals=2),
    ),
    dgp=lambda p: (
        r"X\sim U[-3,3], \qquad e\sim N(0,\sigma^2)",
        rf"Y^*=\alpha+X+e, \quad \alpha={number(p['alpha'], 2)}, \ \sigma={number(p['sigma'], 2)}",
        r"Y=\max\{Y^*,0\}, \qquad m(x)=\Phi(z)(\alpha+x)+\sigma\phi(z),\ z=(\alpha+x)/\sigma",
    ),
    dgp_note=(
        "Varsayılan ayarlar (n = 5.000, α = 1, σ = 1, tohum 807) notlardaki Tablo 6.2'nin benzetimidir: Python kodu "
        "tablodaki sayıların aynısını verir. Gizli eğim her zaman 1'dir."
    ),
    look_at=(
        "**Grafik** — gözlenen Y (sıfırdaki yığılma), gerçek gizli ortalama (kesikli) ve gerçek gözlenen ortalama "
        "m(x) eğrisi; üç tahmin doğrusu.",
        "**Ölçüler** — sansürlenme oranı ve üç eğim tahmini; gerçek gizli eğim 1.",
    ),
    build=_build_censoring, metrics=_censoring_metrics, takeaway=_censoring_takeaway,
)


# --- Deney 2 ve 3: örneklem seçimi --------------------------------------------------------------

def _selection_body(n: int, seed: int | None, rho: float, gz: float) -> tuple:
    rho, gz = round(rho, 6), round(gz, 6)
    scale = round(float(np.sqrt(1.0 - rho * rho)), 6)
    selection = E.add(E.add(E.add(0.2, E.mul(0.8, E.var("x"))), E.mul(gz, E.var("z"))), E.var("u"))
    return (
        NewSample(FRAME, n, seed),
        Draw(FRAME, "x", "normal", 0, 1, "Sonuç denklemindeki regresör X"),
        Draw(FRAME, "z", "normal", 0, 1, "Dışlama değişkeni Z: yalnız seçim denkleminde"),
        Draw(FRAME, "u", "normal", 0, 1, "Seçim hatası u"),
        Draw(FRAME, "v", "normal", 0, 1, "Bağımsız şok"),
        Derive(FRAME, "e", E.add(E.mul(rho, E.var("u")), E.mul(scale, E.var("v"))),
               "Sonuç hatası e: u ile korelasyonu ρ, Var(e) = 1"),
        Derive(FRAME, "ystar", E.add(E.add(1, E.mul(2, E.var("x"))), E.var("e")), "Gizli sonuç Y* = 1 + 2X + e"),
        Derive(FRAME, "s", E.positive(selection), "Seçim S = 1{0,2 + 0,8X + γ_Z·Z + u > 0}: Y yalnız S = 1 iken gözlenir"),
        OLS("naif", FRAME, "ystar", ("x",), where=("s", 1.0)),
        BinaryChoice("secim", FRAME, "s", ("x", "z"), "probit", vcov="classic"),
        Predict("secim", FRAME, "indeks", "index"),
        Derive(FRAME, "mills", E.div(E.normpdf(E.var("indeks")), E.normcdf(E.var("indeks"))),
               "Ters Mills oranı λ̂ = φ(Z'γ̂)/Φ(Z'γ̂)"),
        OLS("heckman", FRAME, "ystar", ("x", "mills"), where=("s", 1.0)),
    )


def _selection_dgp(p: Parameters) -> tuple[str, ...]:
    return (
        r"X,\,Z,\,u,\,v\sim N(0,1)\ \text{bağımsız}, \qquad e=\rho u+\sqrt{1-\rho^2}\,v",
        rf"Y^*=1+2X+e, \qquad S=\mathbf 1\{{0{{,}}2+0{{,}}8X+\gamma_Z Z+u>0\}}, \quad \rho={number(p['rho'], 2)}, \ "
        rf"\gamma_Z={number(p['gz'], 2)}",
        r"Y=Y^*\ \text{yalnız } S=1 \text{ iken}, \qquad \mathbb E[Y\mid X,Z,S=1]=1+2X+\rho\,\lambda(Z'\gamma)",
    )


def _build_selection(p: Parameters) -> tuple:
    return (
        *_selection_body(int(p["n"]), SEED, p["rho"], p["gz"]),
        Plot(FRAME, "x",
             (Curve(E.add(1, E.mul(2, E.var("x"))), "Gerçek sonuç denklemi 1 + 2x", dashed=True),
              ModelLine("naif", "Seçilmiş örneklemde OLS"),
              ModelLine("heckman", "Heckman iki aşama (X eğimi)")),
             "X", "Y", "Seçilmiş örneklem: naif OLS ve seçim düzeltmesi"),
    )


def mills_correlation(state: LabState) -> float:
    frame = state.frames[FRAME]
    selected = frame[frame["s"] == 1]
    return float(np.corrcoef(selected["mills"], selected["x"])[0, 1])


def _selection_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    heckman = state.models["heckman"].params
    return (
        SimMetric("Seçilme oranı", f"%{100 * float(state.frames[FRAME]['s'].mean()):.1f}".replace(".", ","),
                  "Y'nin gözlendiği gözlemlerin payı."),
        SimMetric("Naif OLS eğimi", plain(float(state.models["naif"].params["x"]), 3), "Gerçek eğim 2."),
        SimMetric("Heckman eğimi", plain(float(heckman["x"]), 3), "Seçim düzeltmeli sonuç denklemi."),
        SimMetric("λ̂ katsayısı", plain(float(heckman["mills"]), 3),
                  f"σ_eu = Cov(e, u)'yu tahmin eder; DGP'de ρ = {plain(p['rho'], 2)}."),
        SimMetric("Corr(λ̂, X | S = 1)", plain(mills_correlation(state), 3),
                  "Seçilmiş örneklemde ters Mills oranı ile X'in korelasyonu; 1'e yaklaştıkça tanımlama kırılganlaşır."),
    )


def _selection_takeaway(state: LabState, p: Parameters) -> str:
    naive = float(state.models["naif"].params["x"])
    heckman = float(state.models["heckman"].params["x"])
    correlation = mills_correlation(state)
    if abs(p["rho"]) < 1e-9:
        return (
            f"ρ = 0: seçim sonuç hatasıyla ilişkisiz. Naif OLS ({plain(naive, 3)}) da gerçek eğime (2) yakın; seçim "
            "yanlılığı seçimin kendisinden değil, seçim ile gözlenmeyenler arasındaki ilişkiden doğar (§6.9)."
        )
    text = (
        f"Seçilmiş örneklemde naif OLS eğimi {plain(naive, 3)}: seçim, hata teriminin koşullu ortalamasını X ile "
        f"değiştirdiği için eğim bozuluyor. Ters Mills oranı eklendiğinde eğim {plain(heckman, 3)}: eksik değişken geri "
        "konmuş gibi. "
    )
    if p["gz"] < 0.05:
        text += (
            f"Ancak γ_Z ≈ 0: dışlama değişkeni seçimi etkilemiyor. Seçilmiş örneklemde λ̂ ile X'in korelasyonu "
            f"{plain(correlation, 3)}; düzeltme yalnız normal dağılımın doğrusal olmayan biçimine dayanıyor ve kırılgan "
            "(Deney 3'te dağılımı görün, §6.12)."
        )
    else:
        text += f"λ̂ ile X'in korelasyonu {plain(correlation, 3)}: dışlama değişkeni λ̂'ya X'ten bağımsız varyasyon sağlıyor."
    return text


SELECTION = SimExperiment(
    topic_key=TOPIC, number=2,
    title="Endojen seçim ve Heckman iki aşama",
    question="Sonuç yalnız seçilenlerde gözleniyorsa naif OLS neden yanlıdır ve ters Mills oranı neyi düzeltir?",
    note=NoteRef("6.13", 0, ("Tablo 6.3",)),
    parameters=(
        SimParameter("n", "Gözlem sayısı", 2000, 20000, 10000, 2000, "Örneklem büyüklüğü (seçilmeyenler dahil).",
                     integer=True, decimals=0),
        SimParameter("rho", "Korelasyon ρ = Corr(e, u)", -0.9, 0.9, 0.6, 0.1,
                     "Seçim ile sonuç gözlenmeyenleri arasındaki ilişki; 0 iken seçim dışsaldır.", decimals=1),
        SimParameter("gz", "Dışlama değişkeninin etkisi γ_Z", 0.0, 1.5, 0.6, 0.1,
                     "Z seçimi etkiler ama sonucu doğrudan etkilemez; 0 iken dışlama kısıtı yoktur.", decimals=1),
    ),
    dgp=_selection_dgp,
    dgp_note=(
        "Varsayılan ayarlar (n = 10.000, ρ = 0,6, γ_Z = 0,6, tohum 807) notlardaki Tablo 6.3'ün benzetimidir. Birinci "
        "aşama tam örneklemde Probit, ikinci aşama yalnız S = 1 gözlemlerinde X ve λ̂ ile OLS'dir. λ̂'nın katsayısı "
        "σ_eu = Cov(e, u) = ρ'yu tahmin eder (Var(u) = 1)."
    ),
    look_at=(
        "**Grafik** — gerçek sonuç denklemi (kesikli), seçilmiş örneklemde naif OLS doğrusu ve Heckman düzeltmesinin "
        "sonuç denklemi (λ̂ terimi hariç).",
        "**Ölçüler** — seçilme oranı, iki eğim, λ̂ katsayısı ve seçilmiş örneklemde λ̂ ile X'in korelasyonu.",
    ),
    build=_build_selection, metrics=_selection_metrics, takeaway=_selection_takeaway,
)


REPS = 300


def _build_exclusion(p: Parameters) -> tuple:
    body = _selection_body(int(p["n"]), None, p["rho"], p["gz"])
    return (
        MonteCarlo(
            FRAME, REPS, MC_SEED, body,
            (
                ("b_naif", E.coef("naif", "x")),
                ("b_heckman", E.coef("heckman", "x")),
                ("lambda_katsayisi", E.coef("heckman", "mills")),
            ),
            "mc",
            "Monte Carlo: her tekrarda yeni örneklem; naif OLS ve Heckman iki aşama",
        ),
        Histogram(
            "mc", (("b_heckman", "Heckman iki aşama"), ("b_naif", "Naif OLS")), 1.0, 3.0,
            ((2.0, "Gerçek eğim 2"),),
            "X eğimi tahmini", f"{REPS} örneklemde eğim tahminlerinin dağılımı", bins=50,
        ),
    )


def _exclusion_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    table = state.tables["mc"]
    return (
        SimMetric("Heckman: ortalama", plain(float(table["b_heckman"].mean()), 3), "Gerçek eğim 2."),
        SimMetric("Heckman: std. sapma", plain(float(table["b_heckman"].std()), 3), "Tekrarlar arası dağılım."),
        SimMetric("Naif OLS: ortalama", plain(float(table["b_naif"].mean()), 3), "Seçim yanlılığı."),
        SimMetric("Naif OLS: std. sapma", plain(float(table["b_naif"].std()), 3), "Yanlı ama kararlı."),
        SimMetric("λ̂ katsayısı: std. sapma", plain(float(table["lambda_katsayisi"].std()), 3),
                  f"Ortalaması {plain(float(table['lambda_katsayisi'].mean()), 3)}; hedef ρ = {plain(p['rho'], 2)}."),
    )


def _exclusion_takeaway(state: LabState, p: Parameters) -> str:
    table = state.tables["mc"]
    spread = float(table["b_heckman"].std())
    naive = float(table["b_naif"].std())
    if p["gz"] < 0.05:
        return (
            f"γ_Z = 0: dışlama kısıtı yok. Heckman tahmini ortalamada hâlâ 2 civarında, ama standart sapması "
            f"{plain(spread, 3)}: naif OLS'nin ({plain(naive, 3)}) yaklaşık {plain(spread / naive, 1)} katı. Tanımlama yalnız "
            "normal dağılımın biçiminden geliyor; λ̂ seçilmiş örneklemde X'in neredeyse doğrusal bir fonksiyonu olduğu "
            "için ikinci aşama çoklu doğrusallıktan etkileniyor. γ_Z'yi artırın: dağılım daralır (§6.12)."
        )
    return (
        f"γ_Z = {plain(p['gz'], 1)}: dışlama değişkeni seçimi etkiliyor. Heckman tahminlerinin standart sapması "
        f"{plain(spread, 3)}; γ_Z'yi 0'a çekin ve dağılımın nasıl genişlediğini görün. İyi bir dışlama değişkeni, "
        "seçimi güçlü biçimde etkileyen fakat sonucu seçim dışı bir kanaldan etkilemeyen değişkendir; bu iddia veriden "
        "değil ekonomik argümandan gelir."
    )


EXCLUSION = SimExperiment(
    topic_key=TOPIC, number=3,
    title="Dışlama kısıtı olmadan Heckman",
    question="Seçim denkleminde sonuçtan dışlanan bir değişken yoksa Heckman düzeltmesine ne olur?",
    note=NoteRef("6.12", 0),
    parameters=(
        SimParameter("gz", "Dışlama değişkeninin etkisi γ_Z", 0.0, 1.2, 0.0, 0.1,
                     "0 iken tanımlama yalnız fonksiyonel biçimden gelir.", decimals=1),
        SimParameter("n", "Her örneklemdeki gözlem sayısı", 500, 4000, 1000, 500, "Örneklem büyüklüğü.",
                     integer=True, decimals=0),
        SimParameter("rho", "Korelasyon ρ = Corr(e, u)", 0.0, 0.9, 0.6, 0.1, "Seçimin içsellik derecesi.", decimals=1),
    ),
    dgp=_selection_dgp,
    dgp_note=(
        f"Deney 2 ile aynı veri üretim süreci; {REPS} tekrarın her birinde yeni örneklem çekilir ve naif OLS ile Heckman "
        "iki aşama tahmin edilir. Probit'e Z her zaman girer; γ_Z = 0 iken Z seçimi etkilemez."
    ),
    look_at=(
        "**Grafik** — Heckman ve naif OLS eğim tahminlerinin dağılımı; gerçek eğim 2 (dikey çizgi). [1; 3] dışındaki "
        "tahminler çizilmez.",
        "**Ölçüler** — iki tahmin edicinin ortalaması ve standart sapması; λ̂ katsayısının dağılımı.",
    ),
    build=_build_exclusion, metrics=_exclusion_metrics, takeaway=_exclusion_takeaway,
)


KONU06_EXPERIMENTS = (CENSORING, SELECTION, EXCLUSION)
