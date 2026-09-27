"""Konu 11 Sezgi deneyleri: düzenlileştirme, model karmaşıklığı, model seçimi sonrası çıkarım.

Deney 1  Ridge, Lasso, Elastic Net ve Post-Lasso: dış-örneklem tahmin hatası     (Notlar §11.14, §11.8–§11.12)
Deney 2  Polinom derecesi: eğitim hatası, LOOCV, AIC, BIC ve test hatası          (Notlar §11.2–§11.5)
Deney 3  Ön testle model seçimi sonrasında güven aralığının kapsaması             (Notlar §11.13)

Deney 1 varsayılan ayarlarıyla notlardaki Tablo 11.1'i birebir yeniden üretir: tek üreteç
``np.random.default_rng(80711)``; önce X₁, …, X_p sütunları, sonra ε çekilir.
"""

from __future__ import annotations

import numpy as np

from core.labs import expr as E
from core.labs.runner import LabState, coverage_key
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, number, plain
from core.labs.spec import (
    OLS,
    CoefPath,
    ComplexityCurve,
    CVCurve,
    Derive,
    DotPlot,
    Draw,
    DrawColumns,
    Histogram,
    LinePlot,
    ModelMetrics,
    MonteCarlo,
    NewSample,
    NoteRef,
    Penalized,
    PostSelection,
)

FRAME = "sim"
TOPIC = "konu11"


def _count(value: int) -> str:
    """Türkçe binlik ayraçlı tamsayı: 2000 → "2.000"."""

    return f"{int(value):,}".replace(",", ".")


_LOCATIVE = {0: "da", 1: "de", 2: "de", 3: "te", 4: "te", 5: "te", 6: "da", 7: "de", 8: "de", 9: "da"}


def _at(value: int) -> str:
    """Bulunma eki: 3 → "3'te", 6 → "6'da", 10 → "10'da" (1–10 arası dereceler için)."""

    suffix = "da" if value == 10 else _LOCATIVE[value % 10]
    return f"{value}'{suffix}"


def _training_size(nobs: int) -> int:
    """İlk %70 eğitim örneklemi; notlardaki ``round(0.7 * n)`` ile aynı."""

    return int(round(0.7 * nobs))


# --- Deney 1: §11.14 kontrollü düzenlileştirme örneği ---------------------------------------------

REG_SEED = 80711
SIGNALS = ((1, 1.5), (2, -1.2), (5, 0.9), (10, 0.7), (20, -0.5))
"""Gerçek katsayısı sıfırdan farklı değişkenler (sıra, katsayı); diğer bütün X'lerin katsayısı sıfırdır."""
SIGNAL_NAMES = tuple(f"x{index}" for index, _ in SIGNALS)
REG_MODELS = (
    ("ols_tum", "OLS (tüm X)"),
    ("ridge", "Ridge"),
    ("lasso", "Lasso"),
    ("enet", "Elastic Net"),
    ("post", "Post-Lasso"),
)
RIDGE_GRID = (2, -3, 80)
LASSO_GRID = (1, -3, 100)
ENET_GRID = (1, -3, 80)
ENET_RATIOS = (0.2, 0.5, 0.8)


def _signal() -> E.Expr:
    """1,5X₁ − 1,2X₂ + 0,9X₅ + 0,7X₁₀ − 0,5X₂₀ (soldan sağa; notlardaki toplamla aynı sıra)."""

    total: E.Expr | None = None
    for index, value in SIGNALS:
        term = E.mul(abs(value), E.var(f"x{index}"))
        if total is None:
            total = term if value > 0 else E.mul(value, E.var(f"x{index}"))
        else:
            total = E.add(total, term) if value > 0 else E.sub(total, term)
    return total


def _build_regularization(p: Parameters) -> tuple:
    nobs, count = int(p["n"]), int(p["p"])
    rho, sigma = round(p["rho"], 6), round(p["sigma"], 6)
    names = tuple(f"x{j}" for j in range(1, count + 1))
    train = _training_size(nobs)
    common = dict(sample="egitim", standardize=False)
    return (
        NewSample(FRAME, nobs, REG_SEED),
        DrawColumns(FRAME, "x", count, rho, f"X₁, …, X_{count}: standart normal, Corr(Xj, Xk) = ρ^|j − k|"),
        Draw(FRAME, "e", "normal", 0, 1, "ε ~ N(0, 1)"),
        Derive(FRAME, "y", E.add(_signal(), E.mul(sigma, E.var("e"))),
               "Y: yalnız X1, X2, X5, X10, X20'nin gerçek katsayısı sıfırdan farklı"),
        Derive(FRAME, "egitim", E.compare("le", E.var("id"), train),
               f"İlk {train} gözlem (%70) eğitim, son {nobs - train} gözlem test örneklemi"),
        Derive(FRAME, "kat", E.sub(E.var("id"), E.mul(10, E.floor(E.div(E.sub(E.var("id"), 1), 10)))),
               "10 CV katı: (sıra − 1) mod 10 + 1; yalnız eğitim satırlarında kullanılır"),
        Penalized("ols_tum", FRAME, "y", names, penalty="ols", **common),
        Penalized("ridge", FRAME, "y", names, penalty="ridge", grid=RIDGE_GRID, folds="kat", **common),
        Penalized("lasso", FRAME, "y", names, penalty="lasso", grid=LASSO_GRID, folds="kat", path=True, **common),
        Penalized("enet", FRAME, "y", names, penalty="enet", grid=ENET_GRID, folds="kat", l1_ratios=ENET_RATIOS,
                  **common),
        PostSelection("post", "lasso"),
        ModelMetrics(REG_MODELS, "karsilastirma"),
        CVCurve("ridge", "Ceza parametresi λ (Ridge, SSE ölçeği)", "Ridge: 10-katlı CV ile ceza seçimi"),
        CoefPath("lasso", SIGNAL_NAMES, "Gerçek katsayısı sıfırdan farklı X", "Gerçek katsayısı sıfır olan X",
                 "Ceza parametresi λ (Lasso, yazılım ölçeği; büyükten küçüğe)",
                 "Lasso katsayı yolu: ceza azaldıkça değişkenler modele girer"),
        DotPlot("karsilastirma", "test_mse", REG_MODELS, f"Test örneklemi MSE ({nobs - train} gözlem)",
                "Aynı veri üretim sürecinde dış-örneklem tahmin hatası", decimals=3),
    )


def captured_signals(state: LabState) -> int:
    """Lasso'nun seçtiği değişkenlerden kaçının gerçek katsayısı sıfırdan farklıdır."""

    return len(set(state.models["lasso"].selected) & set(SIGNAL_NAMES))


def _regularization_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    models = state.models
    lasso = models["lasso"]
    return (
        SimMetric("OLS test MSE", plain(models["ols_tum"].test_mse, 3), f"Bütün {int(p['p'])} X ile cezasız EKK."),
        SimMetric("Lasso test MSE", plain(lasso.test_mse, 3), "Ceza 10-katlı CV ile, en küçük CV kuralı."),
        SimMetric("Post-Lasso test MSE", plain(models["post"].test_mse, 3),
                  "Lasso'nun seçtiği değişkenlerle cezasız EKK."),
        SimMetric("Lasso'nun seçtiği X", f"{lasso.nonzero} ({captured_signals(state)}/5)",
                  "Sıfırdan farklı katsayı sayısı; parantez içinde beş gerçek sinyalden kaçının yakalandığı."),
    )


def _regularization_takeaway(state: LabState, p: Parameters) -> str:
    table = state.tables["karsilastirma"]
    labels = dict(REG_MODELS)
    best = str(table["test_mse"].idxmin())
    ridge, lasso = state.models["ridge"], state.models["lasso"]
    train = _training_size(int(p["n"]))
    hansen = 2 * train * lasso.lam
    text = (
        f"Bu örneklemde en düşük test MSE'yi {labels[best]} verdi ({plain(table.loc[best, 'test_mse'], 3)}); "
        f"bütün X'lerle OLS {plain(table.loc['ols_tum', 'test_mse'], 3)}. Lasso {lasso.nonzero} değişken seçti, beş "
        f"gerçek sinyalin {captured_signals(state)} tanesini yakaladı; Ridge {int(p['p'])} katsayının hiçbirini sıfırlamaz, "
        f"yalnız küçültür (λ = {plain(ridge.lam, 1)}, SSE ölçeği). Lasso cezası λ = {plain(lasso.lam, 3)} yazılım "
        f"ölçeğindedir; Hansen'in SSE ölçeğinde 2·{train}·λ ≈ {plain(hansen, 0)}. Post-Lasso seçilen değişkenlerde "
        "küçültmeyi geri alır. Bu tek bir örneklemdir: sıralama sinyal gücüne, korelasyona ve n/p oranına bağlıdır."
    )
    if p["rho"] >= 0.8:
        text += " Korelasyon yüksek: Lasso ilişkili X'lerden birini seçip komşusunu dışarıda bırakabilir."
    if int(p["p"]) >= 0.6 * train:
        text += " p eğitim örneklemine yaklaştıkça OLS'nin varyansı büyür ve düzenlileştirmenin kazancı artar."
    return text


REGULARIZATION = SimExperiment(
    topic_key=TOPIC, number=1,
    title="Ridge, Lasso, Elastic Net ve Post-Lasso: dış-örneklem tahmin hatası",
    question=(
        "Çok sayıda aday açıklayıcının yalnız birkaçı önemliyken cezalı tahmin ediciler test örnekleminde OLS'den daha "
        "iyi tahmin yapar mı; hangisi hangi değişkenleri tutar?"
    ),
    note=NoteRef("11.14", 0, ("§11.8", "§11.9", "§11.11", "§11.12")),
    parameters=(
        SimParameter("rho", "X'ler arası korelasyon ρ", 0.0, 0.9, 0.45, 0.05, "Corr(Xj, Xk) = ρ^|j − k|.",
                     decimals=2),
        SimParameter("sigma", "Hata standart sapması σ", 0.5, 3.0, 1.6, 0.1, "Sinyal/gürültü dengesini belirler.",
                     decimals=1),
        SimParameter("p", "Aday değişken sayısı p", 20, 120, 40, 10, "İlk 20'nin beşi sinyaldir; gerisi gürültü.",
                     integer=True, decimals=0),
        SimParameter("n", "Gözlem sayısı n", 240, 960, 320, 80, "İlk %70'i eğitim, kalan %30'u test.",
                     integer=True, decimals=0),
    ),
    dgp=lambda p: (
        rf"X_{{ij}}\sim N(0,1),\quad \operatorname{{Corr}}(X_j,X_k)=\rho^{{|j-k|}},\quad j=1,\dots,p;\qquad "
        rf"\rho={number(p['rho'], 2)},\ p={int(p['p'])}",
        rf"Y_i=1{{,}}5X_{{i1}}-1{{,}}2X_{{i2}}+0{{,}}9X_{{i5}}+0{{,}}7X_{{i10}}-0{{,}}5X_{{i20}}+\sigma\varepsilon_i,"
        rf"\quad \varepsilon_i\sim N(0,1),\quad \sigma={number(p['sigma'], 1)}",
        rf"n={int(p['n'])}:\ \text{{ilk }}{_training_size(int(p['n']))}\text{{ gözlem eğitim, son }}"
        rf"{int(p['n']) - _training_size(int(p['n']))}\text{{ gözlem test}};\quad "
        r"\text{CV katı}=(i-1)\bmod 10+1",
    ),
    dgp_note=(
        "Rastgele sayılar tek üreteçten (np.random.default_rng(80711)) sırayla çekilir: önce X₁, …, Xₚ sütunları, "
        "sonra ε. Varsayılan ayarlar (ρ = 0,45, σ = 1,6, p = 40, n = 320) notlardaki Tablo 11.1'i birebir verir. "
        "Cezalar yalnız eğitim örnekleminde 10-katlı CV ile seçilir (en küçük CV); Elastic Net'te L1 ağırlığı "
        "r ∈ {0,2; 0,5; 0,8} da CV ile seçilir. Ridge cezası SSE ölçeğindedir (Hansen; scikit-learn Ridge alpha), Lasso "
        "ve Elastic Net yazılım ölçeğindedir (scikit-learn, glmnet, Stata lasso). X'ler birim varyansla üretildiği "
        "için standartlaştırılmaz; sabit terim cezalandırılmaz."
    ),
    look_at=(
        "**Grafik 1** — Ridge cezasının 10-katlı CV eğrisi ve ±1 SH bandı; kesikli çizgi seçilen λ.",
        "**Grafik 2** — Lasso katsayı yolu: ceza azaldıkça değişkenler modele girer; renkli çizgiler gerçek katsayısı "
        "sıfırdan farklı beş X.",
        "**Grafik 3 ve tablo** — beş yöntemin test MSE'si, katsayı normu ve sıfırdan farklı katsayı sayısı "
        "(notlarda Tablo 11.1).",
    ),
    build=_build_regularization, metrics=_regularization_metrics, takeaway=_regularization_takeaway,
    tables=(("karsilastirma", "Test performansı (notlarda Tablo 11.1)"),),
)


# --- Deney 2: model karmaşıklığı ------------------------------------------------------------------

COMPLEXITY_SEED = 81102
TEST_SIZE = 2000
MAX_DEGREE = 10
PHASE = 0.5


def _truth(a: float, x: E.Expr) -> E.Expr:
    return E.sin(E.add(E.mul(a, x), PHASE))


def _build_complexity(p: Parameters) -> tuple:
    train, a, sigma = int(p["n"]), round(p["a"], 6), round(p["sigma"], 6)
    x = E.var("x")
    return (
        NewSample(FRAME, train + TEST_SIZE, COMPLEXITY_SEED),
        Draw(FRAME, "x", "uniform", -1, 1, "X ~ U(−1, 1)"),
        Draw(FRAME, "e", "normal", 0, 1, "ε ~ N(0, 1)"),
        Derive(FRAME, "y", E.add(_truth(a, x), E.mul(sigma, E.var("e"))), "Y = sin(aX + 0,5) + σε"),
        Derive(FRAME, "egitim", E.compare("le", E.var("id"), train),
               f"İlk {train} gözlem eğitim, son {_count(TEST_SIZE)} gözlem test örneklemi"),
        ComplexityCurve("karmasiklik", FRAME, "x", "y", MAX_DEGREE, "egitim", "olcutler"),
        LinePlot("olcutler", (("egitim_mse", "Eğitim MSE"), ("loocv", "LOOCV"), ("test_mse", "Test MSE")),
                 "Polinom derecesi d", "Ortalama karesel hata", "Eğitim hatası düşer, test hatası U çizer"),
        LinePlot("olcutler", (("aic", "AIC"), ("bic", "BIC")), "Polinom derecesi d",
                 "Ölçüt − en küçük değeri", "AIC ve BIC: en küçük değerden uzaklık", relative=True),
    )


def _selected(state: LabState, column: str) -> int:
    return int(state.scalars[f"karmasiklik_d_{column}"])


def _complexity_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    return (
        SimMetric("LOOCV'nin seçtiği d", str(_selected(state, "loocv")), "Kaldıraçla kesin dışarıda bırakma hatası."),
        SimMetric("AIC'nin seçtiği d", str(_selected(state, "aic")), "Ek parametre cezası 2."),
        SimMetric("BIC'in seçtiği d", str(_selected(state, "bic")), "Ek parametre cezası log n."),
        SimMetric("Testte en iyi d", str(_selected(state, "test_mse")),
                  f"{_count(TEST_SIZE)} yeni gözlemde en küçük MSE; yalnız değerlendirme içindir, seçimde kullanılmaz."),
    )


def _complexity_takeaway(state: LabState, p: Parameters) -> str:
    table = state.tables["olcutler"]
    train_mse = table["egitim_mse"].to_numpy()
    monotone = bool(np.all(np.diff(train_mse) <= 1e-12))
    loocv, aic, bic, test = (_selected(state, column) for column in ("loocv", "aic", "bic", "test_mse"))
    text = (
        f"Eğitim MSE derece arttıkça {'hiç artmadan ' if monotone else ''}düşer: d = 1'de {plain(train_mse[0], 3)}, "
        f"d = {_at(MAX_DEGREE)} {plain(train_mse[-1], 3)}. Test MSE ise d = {_at(test)} en küçüktür; daha karmaşık modeller "
        f"gürültüyü öğrenir. LOOCV d = {loocv}, AIC d = {aic}, BIC d = {bic} seçti. BIC'in cezası log n = "
        f"{plain(float(np.log(int(p['n']))), 2)} > 2 olduğu için BIC genellikle daha az parametreli modeli seçer (§11.3). "
        "Hiçbir ölçüt test verisine bakmaz; test örneklemi yalnız son değerlendirme içindir (§11.4)."
    )
    if int(p["n"]) <= 60:
        text += " Eğitim örneklemi küçük: ölçütler arasındaki fark ve seçimdeki oynaklık büyür; n'yi artırın."
    return text


COMPLEXITY = SimExperiment(
    topic_key=TOPIC, number=2,
    title="Polinom derecesi: eğitim hatası, LOOCV, AIC, BIC ve test hatası",
    question=(
        "Model karmaşıklaştıkça eğitim hatası neden hep düşer; LOOCV, AIC ve BIC test hatasını en küçük yapan "
        "dereceyi ne kadar iyi bulur?"
    ),
    note=NoteRef("11.3", 0, ("§11.2", "§11.4", "§11.5")),
    parameters=(
        SimParameter("n", "Eğitim gözlem sayısı n", 40, 400, 80, 20, f"Test örneklemi her zaman {_count(TEST_SIZE)}.",
                     integer=True, decimals=0),
        SimParameter("sigma", "Hata standart sapması σ", 0.1, 1.0, 0.4, 0.05, "Gürültü düzeyi.", decimals=2),
        SimParameter("a", "Eğrilik a", 1.0, 6.0, 3.0, 0.5, "m(x) = sin(ax + 0,5): a büyüdükçe fonksiyon kıvrılır.",
                     decimals=1),
    ),
    dgp=lambda p: (
        rf"X\sim U(-1,1),\qquad m(x)=\sin(a x+0{{,}}5),\qquad a={number(p['a'], 1)}",
        rf"Y=m(X)+\sigma\varepsilon,\quad \varepsilon\sim N(0,1),\quad \sigma={number(p['sigma'], 2)};\qquad "
        rf"n_{{\text{{eğitim}}}}={int(p['n'])},\ n_{{\text{{test}}}}={TEST_SIZE}",
        rf"\text{{aday modeller: }}Y=\beta_0+\beta_1X+\dots+\beta_dX^d+e,\qquad d=1,\dots,{MAX_DEGREE}",
        r"AIC=n+n\log(2\pi\hat\sigma^2)+2K,\qquad BIC=n+n\log(2\pi\hat\sigma^2)+K\log n,\qquad K=d+2",
    ),
    dgp_note=(
        f"Gerçek koşullu ortalama bir polinom değildir; her derece onu yalnız yaklaşık temsil eder. σ̂² = SSR/n (n'e "
        "bölünür) ve K, σ²'yi de sayar (§11.3). LOOCV kaldıraç değerleriyle kesin hesaplanır: ortalama [êᵢ/(1 − hᵢᵢ)]². "
        f"Test MSE {_count(TEST_SIZE)} yeni gözlemde, gerçek dış-örneklem hatasının iyi bir yaklaşığıdır. Tohum 81102."
    ),
    look_at=(
        "**Grafik 1** — eğitim MSE, LOOCV ve test MSE'nin polinom derecesiyle değişimi.",
        "**Grafik 2** — AIC ve BIC'in kendi en küçük değerlerinden uzaklığı; sıfır olan derece seçilir.",
        "**Ölçüler ve tablo** — her ölçütün seçtiği derece ve bütün ölçütlerin değerleri.",
    ),
    build=_build_complexity, metrics=_complexity_metrics, takeaway=_complexity_takeaway,
    tables=(("olcutler", "Dereceye göre model seçim ölçütleri"),),
    labels=(("derece", "Derece d"), ("egitim_mse", "Eğitim MSE"), ("loocv", "LOOCV"), ("test_mse", "Test MSE"),
            ("aic", "AIC"), ("bic", "BIC")),
)


# --- Deney 3: model seçimi sonrası çıkarım --------------------------------------------------------

SELECTION_SEED = 81103
SELECTION_REPS = 2000
CRITICAL = 1.96


def _selection_indicator() -> E.Expr:
    return E.compare("ge", E.absolute(E.div(E.coef("uzun", "x2"), E.se("uzun", "x2"))), CRITICAL)


def _selected_value(long: E.Expr, short: E.Expr) -> E.Expr:
    chosen = _selection_indicator()
    return E.add(E.mul(chosen, long), E.mul(E.sub(1, chosen), short))


def long_sd(nobs: int, rho: float) -> float:
    """Uzun modelde β̂₁'in yaklaşık standart sapması: 1/√[n(1 − ρ²)] (σ = 1, Var(X₁) = 1)."""

    return float(1 / np.sqrt(nobs * (1 - rho**2)))


def _build_selection(p: Parameters) -> tuple:
    nobs, beta2, rho = int(p["n"]), round(p["beta2"], 6), round(p["rho"], 6)
    scale = round(float(np.sqrt(1 - rho**2)), 6)
    spread = 4.5 * long_sd(nobs, rho)
    lower, upper = round(1 - spread, 2), round(1 + beta2 * rho + spread, 2)
    body = (
        NewSample(FRAME, nobs, None),
        Draw(FRAME, "x1", "normal", 0, 1, "X₁ ~ N(0, 1)"),
        Draw(FRAME, "z", "normal", 0, 1, "Z ~ N(0, 1)"),
        Derive(FRAME, "x2", E.add(E.mul(rho, E.var("x1")), E.mul(scale, E.var("z"))),
               "X₂ = ρX₁ + √(1 − ρ²)Z: Corr(X₁, X₂) = ρ"),
        Draw(FRAME, "e", "normal", 0, 1, "ε ~ N(0, 1)"),
        Derive(FRAME, "y", E.add(E.add(E.var("x1"), E.mul(beta2, E.var("x2"))), E.var("e")),
               "Y = X₁ + β₂X₂ + ε: hedef β₁ = 1"),
        OLS("uzun", FRAME, "y", ("x1", "x2")),
        OLS("kisa", FRAME, "y", ("x1",)),
    )
    return (
        MonteCarlo(
            FRAME, SELECTION_REPS, SELECTION_SEED, body,
            (
                ("b_uzun", E.coef("uzun", "x1")),
                ("sh_uzun", E.se("uzun", "x1")),
                ("secildi", _selection_indicator()),
                ("b_secim", _selected_value(E.coef("uzun", "x1"), E.coef("kisa", "x1"))),
                ("sh_secim", _selected_value(E.se("uzun", "x1"), E.se("kisa", "x1"))),
            ),
            "secim",
            f"{_count(SELECTION_REPS)} tekrar: her tekrarda X₂ için t-testi (|t| ≥ 1,96 ise uzun model), sonra β₁'in "
            "%95 güven aralığı",
            coverage=(("b_uzun", "sh_uzun", 1.0), ("b_secim", "sh_secim", 1.0)),
        ),
        Histogram(
            "secim", (("b_uzun", "Uzun model (seçimsiz)"), ("b_secim", "Ön test sonrası seçilen model")),
            lower, upper, ((1.0, "Gerçek β₁ = 1"),), "β̂₁",
            f"β̂₁'in örnekleme dağılımı ({_count(SELECTION_REPS)} tekrar)", bins=60,
        ),
    )


def selection_numbers(state: LabState) -> dict[str, float]:
    table = state.tables["secim"]
    return {
        "long": state.scalars[coverage_key("secim", "b_uzun")],
        "post": state.scalars[coverage_key("secim", "b_secim")],
        "rate": float(table["secildi"].mean()),
        "bias": float(table["b_secim"].mean() - 1.0),
        "long_bias": float(table["b_uzun"].mean() - 1.0),
    }


def _selection_metrics(state: LabState, p: Parameters) -> tuple[SimMetric, ...]:
    values = selection_numbers(state)
    return (
        SimMetric("Kapsama: uzun model", plain(values["long"], 3), "X₂ her zaman modelde; seçim yok."),
        SimMetric("Kapsama: seçim sonrası", plain(values["post"], 3),
                  "Ön testle seçilen modelin klasik %95 güven aralığı."),
        SimMetric("X₂'nin seçilme oranı", plain(values["rate"], 3), "|t₂| ≥ 1,96 olan tekrarların payı."),
        SimMetric("Seçim sonrası yanlılık", plain(round(values["bias"], 3) + 0.0, 3), "Ortalama β̂₁ − 1."),
    )


def _selection_takeaway(state: LabState, p: Parameters) -> str:
    values = selection_numbers(state)
    beta2, rho = p["beta2"], p["rho"]
    if beta2 < 1e-9:
        return (
            f"β₂ = 0: kısa model doğrudur; X₂'nin seçilme oranı {plain(values['rate'], 3)} (testin anlamlılık düzeyi "
            f"kadar, yaklaşık 0,05). Seçim sonrası kapsama {plain(values['post'], 3)}, uzun modelinki "
            f"{plain(values['long'], 3)}. Sorun ortaya çıkmaz; β₂'yi artırın."
        )
    if rho < 1e-9:
        return (
            f"ρ = 0: X₂, X₁ ile ilişkisiz; X₂'yi dışarıda bırakmak β̂₁'i yanlı yapmaz. Seçim sonrası kapsama "
            f"{plain(values['post'], 3)}, uzun modelinki {plain(values['long'], 3)}. Korelasyonu artırın."
        )
    return (
        f"Uzun modelin güven aralığı nominal düzeydedir (kapsama {plain(values['long'], 3)}). Ön testle seçilen modelde "
        f"kapsama {plain(values['post'], 3)}: X₂'nin seçilme oranı yalnız {plain(values['rate'], 3)}; seçilmediğinde kısa "
        f"model β₁'i β₂ρ = {plain(beta2 * rho, 3)} kadar yanlı tahmin eder ve standart hatası bu yanlılığı bilmez. "
        "Seçim kuralı tahmin edicinin parçasıdır; seçimden sonra klasik standart hata seçim belirsizliğini içermez "
        "(§11.13). Sorun, β₂ sıfır değil ama örneklem gürültüsüne göre küçükken en büyüktür; n'yi büyütün: X₂ büyük "
        "olasılıkla seçilir ve kapsama düzelir."
    )


POST_SELECTION = SimExperiment(
    topic_key=TOPIC, number=3,
    title="Ön testle model seçimi sonrasında güven aralığının kapsaması",
    question=(
        "Bir kontrol değişkeni t-testiyle modele alınıp alınmadığında, seçilen modelin klasik %95 güven aralığı hedef "
        "katsayıyı gerçekten %95 olasılıkla kapsar mı?"
    ),
    note=NoteRef("11.13", 0, ("§11.1", "§11.12")),
    parameters=(
        SimParameter("beta2", "X₂'nin katsayısı β₂", 0.0, 0.5, 0.2, 0.02, "β₂ = 0 ise kısa model doğrudur.",
                     decimals=2),
        SimParameter("rho", "Corr(X₁, X₂) = ρ", 0.0, 0.9, 0.7, 0.05, "X₂'yi dışlamanın β̂₁'e yanlılığı β₂ρ.",
                     decimals=2),
        SimParameter("n", "Gözlem sayısı n", 50, 400, 100, 25, "Her tekrarda yeni örneklem.", integer=True,
                     decimals=0),
    ),
    dgp=lambda p: (
        r"X_1\sim N(0,1),\quad X_2=\rho X_1+\sqrt{1-\rho^2}\,Z,\quad Z,\ \varepsilon\sim N(0,1)",
        rf"Y=\beta_0+\beta_1X_1+\beta_2X_2+\varepsilon,\quad \beta_0=0,\ \beta_1=1,\ \beta_2={number(p['beta2'], 2)},"
        rf"\ \rho={number(p['rho'], 2)},\ n={int(p['n'])}",
        r"|t_2|\ge1{,}96\ \Rightarrow\ Y\sim 1+X_1+X_2,\qquad \text{aksi hâlde}\ Y\sim 1+X_1",
        r"\text{seçilen modelde GA: }\hat\beta_1\pm1{,}96\,SH(\hat\beta_1)",
    ),
    dgp_note=(
        f"Hansen (§28.17) simülasyonunun sade biçimi. {_count(SELECTION_REPS)} tekrar, tek üreteç (tohum 81103). "
        "Standart hatalar klasik (homoskedastik) EKK standart hatasıdır; hatalar homoskedastik olduğu için doğru "
        "formüldür. Sorun formülde değil, modelin veriye bakılarak seçilmesindedir."
    ),
    look_at=(
        "**Grafik** — uzun modelin ve ön test sonrası seçilen modelin β̂₁ dağılımları; seçim sonrası dağılım iki "
        "modelin karışımıdır.",
        "**Ölçüler** — iki güven aralığının kapsama oranı, X₂'nin seçilme oranı ve seçim sonrası yanlılık.",
    ),
    build=_build_selection, metrics=_selection_metrics, takeaway=_selection_takeaway,
)


KONU11_EXPERIMENTS = (REGULARIZATION, COMPLEXITY, POST_SELECTION)
