"""Konu 1 Sezgi deneyleri: bilinen bir DGP ile koşullu ortalama, OLS ve nedensellik.

Deney 1  Koşullu ortalama ile OLS doğrusu            (Notlar §1.6, Şekil 1.3)
Deney 2  OLS artıklarının cebirsel özellikleri       (Notlar §1.8.2, Şekil 1.4)
Deney 3  OLS eğimi nedensel etki mi?                 (Notlar §1.11, Şekil 1.5)
"""

from __future__ import annotations


from core.labs import expr as E
from core.labs.runner import LabState
from core.labs.sezgi import Parameters, SimExperiment, SimMetric, SimParameter, number, plain
from core.labs.spec import (
    OLS,
    Curve,
    Derive,
    Draw,
    GroupSummary,
    MeanPoints,
    ModelLine,
    NewSample,
    NoteRef,
    Plot,
    Predict,
    Scatter,
    Summaries,
    ZeroLine,
)

SEED = 807
FRAME = "sim"
TOPIC = "konu01"

X_LABEL = "Eğitim yılı (x)"
LABELS = (
    ("x", "Eğitim (x)"),
    ("n", "Gözlem"),
    ("ort_y", "Hücre ortalaması m̂(x)"),
    ("m", "Gerçek m(x)"),
    ("yhat_doymus", "Ŷ doymuş model"),
    ("yhat_dogrusal", "Ŷ doğrusal OLS"),
    ("ort_ehat", "Artık ortalaması"),
)

SAMPLE_SIZE = SimParameter(
    "n", "Gözlem sayısı", 200, 5000, 2000, 200,
    "Örneklem büyüdükçe noktalar gerçek eğriye, OLS doğrusu hedef doğruya yaklaşır.",
    integer=True, decimals=0,
)
CURVATURE = SimParameter(
    "gamma", "Koşullu ortalamanın eğriliği γ", 0.0, 0.02, 0.01, 0.002,
    "γ = 0 iken gerçek koşullu ortalama doğrusaldır.", decimals=3,
)


# --- Ortak DGP: Deney 1 ve 2 ---------------------------------------------------

def _cef(gamma: float) -> E.Expr:
    """Gerçek koşullu ortalama m(x) = 1,25 + 0,08 x + γ (x − 14)²."""

    x = E.var("x")
    return E.add(E.add(1.25, E.mul(0.08, x)), E.mul(round(gamma, 6), E.power(E.sub(x, 14), 2)))


def _cef_sample(parameters: Parameters) -> tuple:
    return (
        NewSample(FRAME, int(parameters["n"]), SEED),
        Draw(FRAME, "z", "normal", 0, 1, "Eğitimi belirleyen standart normal şok"),
        Derive(
            FRAME, "x",
            E.minimum(E.maximum(E.rounded(E.add(13.5, E.mul(2.2, E.var("z")))), 8), 20),
            "Eğitim yılı: 8 ile 20 arasında tam sayı",
        ),
        Draw(FRAME, "e", "normal", 0, 1, "Hata terimi"),
        Derive(FRAME, "m", _cef(parameters["gamma"]), "Gerçek koşullu ortalama m(x); DGP'den bilinir"),
        Derive(FRAME, "y", E.add(E.var("m"), E.mul(0.45, E.var("e"))), "Sonuç: Y = m(X) + 0,45 e"),
    )


def _cef_dgp(parameters: Parameters) -> tuple[str, ...]:
    return (
        r"Z_i \sim N(0,1), \qquad X_i=\min\{\max\{\mathrm{yuvarla}(13{,}5+2{,}2\,Z_i),\,8\},\,20\}",
        rf"m(x)=1{{,}}25+0{{,}}08\,x+\gamma\,(x-14)^2, \qquad \gamma={number(parameters['gamma'])}",
        r"Y_i=m(X_i)+0{,}45\,e_i, \qquad e_i\sim N(0,1)\ \text{ve } X_i\text{'den bağımsız}",
    )


CEF_DGP_NOTE = (
    "X eğitim yılı, Y log saatlik ücret gibi düşünülebilir. Gerçek veride m(x) bilinmez; burada "
    "DGP'yi biz yazdığımız için biliyoruz ve tahminleri onunla karşılaştırabiliyoruz. "
    "γ = 0 iken koşullu ortalama doğrusaldır."
)


# --- Deney 1 ------------------------------------------------------------------

def _build_projection(parameters: Parameters) -> tuple:
    return _cef_sample(parameters) + (
        OLS("dogrusal", FRAME, "y", ("x",)),
        OLS("hedef", FRAME, "m", ("x",)),
        OLS("doymus", FRAME, "y", ("x",), categorical=("x",)),
        Predict("dogrusal", FRAME, "yhat_dogrusal", "fitted"),
        Predict("doymus", FRAME, "yhat_doymus", "fitted"),
        GroupSummary(
            FRAME,
            "x",
            (
                ("n", "y", "count"),
                ("ort_y", "y", "mean"),
                ("m", "m", "mean"),
                ("yhat_doymus", "yhat_doymus", "mean"),
                ("yhat_dogrusal", "yhat_dogrusal", "mean"),
            ),
            "karsilastirma",
        ),
        Plot(
            FRAME,
            "x",
            (
                MeanPoints("y", "Hücre ortalaması m̂(x)"),
                Curve(_cef(parameters["gamma"]), "Gerçek koşullu ortalama m(x)"),
                ModelLine("dogrusal", "OLS doğrusu Ŷ"),
                ModelLine("hedef", "m(x)'in en iyi doğrusal yaklaşımı", dashed=True),
            ),
            X_LABEL,
            "Y",
            "Koşullu ortalama, hücre ortalamaları ve OLS doğrusu",
        ),
    )


def _projection_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    table = state.tables["karsilastirma"]
    ols_slope = float(state.models["dogrusal"].params["x"])
    target_slope = float(state.models["hedef"].params["x"])
    gap = float((table["ort_y"] - table["yhat_dogrusal"]).abs().max())
    saturated = float((table["ort_y"] - table["yhat_doymus"]).abs().max())
    saturated_text = "0 (makine duyarlığında)" if saturated < 1e-9 else plain(saturated, 6)
    return (
        SimMetric("OLS eğimi β̂₁", plain(ols_slope), "Örneklemden tahmin edilen doğrunun eğimi."),
        SimMetric("Hedef eğim β₁", plain(target_slope), "m(X)'in X üzerindeki en iyi doğrusal yaklaşımının eğimi."),
        SimMetric("En büyük |m̂(x) − Ŷ(x)|", plain(gap), "Doğrusal OLS'in hücre ortalamalarından en uzak olduğu nokta."),
        SimMetric("Doymuş model: |m̂(x) − Ŷ(x)|", saturated_text, "Her eğitim düzeyine ayrı kukla: Ŷ hücre ortalamasına eşit."),
    )


def _projection_takeaway(state: LabState, parameters: Parameters) -> str:
    text = (
        "Koşullu ortalama ile OLS uyum değeri farklı nesnelerdir. Hücre ortalaması m̂(x), m(x)'in "
        "doğrudan tahminidir. OLS doğrusu ise m(x)'in en iyi doğrusal yaklaşımını (yeşil kesikli) "
        "tahmin eder; ikisi yalnız m(x) doğrusalsa aynı yere yakınsar. Her eğitim düzeyi için ayrı "
        "kukla içeren doymuş modelde Ŷ, hücre ortalamalarına tam olarak eşittir. OLS doğrusu "
        "hücrelere gözlem sayısıyla ağırlıklı uyar: kalabalık hücreler doğruyu kendine çeker."
    )
    if parameters["gamma"] == 0:
        text += " Şu an γ = 0: gerçek m(x) doğrusal, dolayısıyla üç doğru da aynı nesneyi hedefliyor."
    return text


PROJECTION = SimExperiment(
    topic_key=TOPIC,
    number=1,
    title="Koşullu ortalama ile OLS doğrusu",
    question="OLS doğrusu koşullu ortalamanın kendisi mi, yoksa ona yapılan en iyi doğrusal yaklaşım mı?",
    note=NoteRef("1.6", 0, ("Şekil 1.3",)),
    parameters=(SAMPLE_SIZE, CURVATURE),
    dgp=_cef_dgp,
    dgp_note=CEF_DGP_NOTE,
    look_at=(
        "**Siyah eğri** — gerçek koşullu ortalama $m(x)=\\mathbb{E}[Y\\mid X=x]$. DGP'den bilinir.",
        "**Noktalar** — hücre ortalamaları $\\hat m(x)$: aynı eğitim düzeyindeki gözlemlerin $Y$ "
        "ortalaması. Biçim varsaymayan tahmindir; nokta boyutu gözlem sayısıdır.",
        "**Kırmızı doğru** — OLS uyum değerleri $\\hat Y=\\hat\\beta_0+\\hat\\beta_1 x$.",
        "**Yeşil kesikli doğru** — $m(X)$'in $X$ üzerindeki en iyi doğrusal yaklaşımı. OLS'in "
        "tahmin ettiği hedef budur.",
    ),
    build=_build_projection,
    metrics=_projection_metrics,
    takeaway=_projection_takeaway,
    tables=(("karsilastirma", "Her eğitim düzeyinde dört nesne"),),
    labels=LABELS,
)


# --- Deney 2 ------------------------------------------------------------------

def _build_residuals(parameters: Parameters) -> tuple:
    return _cef_sample(parameters) + (
        OLS("dogrusal", FRAME, "y", ("x",)),
        Predict("dogrusal", FRAME, "yhat", "fitted"),
        Predict("dogrusal", FRAME, "ehat", "residual"),
        Derive(FRAME, "x_ehat", E.mul(E.var("x"), E.var("ehat")), "Eğitim × artık"),
        Summaries(
            FRAME,
            (
                ("Artıkların toplamı  Σ êᵢ", "ehat", "sum"),
                ("Eğitim × artık toplamı  Σ xᵢ êᵢ", "x_ehat", "sum"),
                ("Ŷ ortalaması", "yhat", "mean"),
                ("Y ortalaması", "y", "mean"),
            ),
            "ozellikler",
        ),
        GroupSummary(FRAME, "x", (("n", "ehat", "count"), ("ort_ehat", "ehat", "mean")), "artik_ortalamalari"),
        Plot(
            FRAME,
            "x",
            (
                Scatter("ehat", "OLS artıkları êᵢ"),
                MeanPoints("ehat", "Her eğitim düzeyinde artık ortalaması"),
                ZeroLine("Sıfır"),
            ),
            X_LABEL,
            "Artık ê",
            "OLS artıkları ve eğitim düzeylerine göre ortalamaları",
        ),
    )


def _residual_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    values = state.tables["ozellikler"]["Değer"]
    means = state.tables["artik_ortalamalari"]["ort_ehat"]
    return (
        SimMetric("Σ êᵢ", "0 (makine duyarlığında)" if abs(values.iloc[0]) < 1e-8 else plain(values.iloc[0], 8),
                  "Sabit terim sayesinde her örneklemde sıfır."),
        SimMetric("Σ xᵢ êᵢ", "0 (makine duyarlığında)" if abs(values.iloc[1]) < 1e-6 else plain(values.iloc[1], 8),
                  "Normal denklem: her örneklemde sıfır."),
        SimMetric("En büyük hücre içi artık ortalaması", plain(float(means.abs().max())),
                  "E[e | X = x] = 0 olsaydı bu sayı yalnız örnekleme gürültüsü kadar olurdu."),
    )


def _residual_takeaway(state: LabState, parameters: Parameters) -> str:
    text = (
        "Σ êᵢ = 0 ve Σ xᵢ êᵢ = 0 her örneklemde, γ ne olursa olsun sağlanır: bunlar OLS'in normal "
        "denklemleridir, bir varsayım değildir. "
    )
    if parameters["gamma"] > 0:
        text += (
            "Ama γ > 0 iken artıkların eğitim düzeylerine göre ortalaması U biçimlidir: "
            "E[e | X] = 0 sağlanmıyor. "
        )
    else:
        text += "γ = 0 iken artık ortalamaları sıfır çevresinde rastgele dağılır: E[e | X] = 0 sağlanıyor. "
    text += (
        "Örneklem ortogonalliği (X′ê = 0), projeksiyon özelliği (E[Xe] = 0) ve doğrusal koşullu "
        "ortalama varsayımı (E[e | X] = 0) üç ayrı ifadedir; ilki ikincisini, ikincisi üçüncüsünü içermez."
    )
    return text


RESIDUALS = SimExperiment(
    topic_key=TOPIC,
    number=2,
    title="OLS artıklarının cebirsel özellikleri",
    question=(
        "OLS artıklarının toplamı ve eğitimle çarpımlarının toplamı neden hep sıfır çıkar? "
        "Bu, hatanın koşullu ortalamasının sıfır olduğu anlamına mı gelir?"
    ),
    note=NoteRef("1.8.2", 0, ("Şekil 1.4",)),
    parameters=(SAMPLE_SIZE, CURVATURE),
    dgp=_cef_dgp,
    dgp_note=CEF_DGP_NOTE,
    look_at=(
        "**Gri noktalar** — OLS artıkları $\\hat e_i=Y_i-\\hat Y_i$.",
        "**Mavi noktalar** — her eğitim düzeyinde artıkların ortalaması; "
        "$\\mathbb{E}[e\\mid X=x]$'in örneklem karşılığı.",
        "**Ölçüler** — $\\sum_i \\hat e_i$ ve $\\sum_i X_i\\hat e_i$ (normal denklemler).",
    ),
    build=_build_residuals,
    metrics=_residual_metrics,
    takeaway=_residual_takeaway,
    tables=(("ozellikler", "OLS'in cebirsel özellikleri"),),
    labels=LABELS,
)


# --- Deney 3 ------------------------------------------------------------------

ABILITY_TO_SCHOOLING = SimParameter(
    "pi", "Yeteneğin eğitime etkisi π", 0.0, 2.0, 1.2, 0.2,
    "Yetenekli bireyler daha çok okuyorsa π > 0.", decimals=1,
)
ABILITY_TO_WAGE = SimParameter(
    "delta", "Yeteneğin ücrete etkisi δ", 0.0, 0.3, 0.15, 0.05,
    "Yetenek eğitimden bağımsız olarak da ücreti artırıyorsa δ > 0.", decimals=2,
)
CAUSAL_RETURN = 0.08


def _build_confounding(parameters: Parameters) -> tuple:
    pi, delta = round(parameters["pi"], 6), round(parameters["delta"], 6)
    return (
        NewSample(FRAME, int(parameters["n"]), SEED),
        Draw(FRAME, "a", "normal", 0, 1, "Gözlenmeyen yetenek A"),
        Draw(FRAME, "v", "normal", 0, 1, "Eğitimi etkileyen diğer etkenler"),
        Derive(
            FRAME, "x",
            E.minimum(
                E.maximum(E.rounded(E.add(E.add(13.5, E.mul(pi, E.var("a"))), E.mul(1.7, E.var("v")))), 8),
                20,
            ),
            "Eğitim yılı: yetenekten π kadar etkilenir",
        ),
        Draw(FRAME, "e", "normal", 0, 1, "Hata terimi"),
        Derive(
            FRAME, "y",
            E.add(E.add(E.add(1.25, E.mul(CAUSAL_RETURN, E.var("x"))), E.mul(delta, E.var("a"))),
                  E.mul(0.35, E.var("e"))),
            "Log ücret: eğitimin nedensel etkisi 0,08; yetenek δ kadar ekler",
        ),
        OLS("kisa", FRAME, "y", ("x",)),
        OLS("uzun", FRAME, "y", ("x", "a")),
        Plot(
            FRAME,
            "x",
            (
                Scatter("y", "Gözlemler"),
                Curve(E.add(1.25, E.mul(CAUSAL_RETURN, E.var("x"))), "Nedensel ilişki (eğim 0,08)"),
                ModelLine("kisa", "OLS doğrusu (yalnız eğitim)"),
            ),
            X_LABEL,
            "Y",
            "OLS doğrusu ve nedensel ilişki",
        ),
    )


def _confounding_metrics(state: LabState, parameters: Parameters) -> tuple[SimMetric, ...]:
    short = float(state.models["kisa"].params["x"])
    long = float(state.models["uzun"].params["x"])
    return (
        SimMetric("Nedensel etki (DGP)", plain(CAUSAL_RETURN), "Yetenek sabitken bir yıl fazla eğitimin etkisi."),
        SimMetric("OLS eğimi: yalnız eğitim", plain(short), "Gerçek veride çalıştırabildiğimiz regresyon."),
        SimMetric("OLS eğimi: yetenek de modelde", plain(long), "Yalnız simülasyonda mümkün: A burada gözleniyor."),
    )


def _confounding_takeaway(state: LabState, parameters: Parameters) -> str:
    both = parameters["pi"] > 0 and parameters["delta"] > 0
    if both:
        text = (
            "π ve δ ikisi birden sıfırdan farklı: OLS eğimi 0,08'den büyük, çünkü eğitim gözlenmeyen "
            "yeteneğin ücrete etkisini de taşıyor. Yetenek modele girince eğim 0,08'e döner. "
        )
    else:
        text = (
            "π veya δ sıfır: yetenek ya eğitimle ya da ücretle ilişkisiz, bu yüzden OLS eğimi 0,08 "
            "çevresinde. Farkın oluşması için ikisinin birden sıfırdan farklı olması gerekir. "
        )
    text += (
        "OLS yanlış hesap yapmıyor: eğitim ile ücret arasındaki en iyi doğrusal projeksiyonu doğru "
        "tahmin ediyor. Ama bu projeksiyon katsayısı, ek bir tanımlama varsayımı olmadan nedensel "
        "etki olarak okunamaz. Bu farkın formülü Konu 2'de eksik değişken yanlılığı olarak gelecek."
    )
    return text


CONFOUNDING = SimExperiment(
    topic_key=TOPIC,
    number=3,
    title="OLS eğimi nedensel etki mi?",
    question="Gözlenmeyen yetenek hem eğitimi hem ücreti etkiliyorsa OLS eğimi neyi ölçer?",
    note=NoteRef("1.11", 0, ("Şekil 1.5",)),
    parameters=(SAMPLE_SIZE, ABILITY_TO_SCHOOLING, ABILITY_TO_WAGE),
    dgp=lambda p: (
        r"A_i\sim N(0,1)\ \text{(gözlenmeyen yetenek)}, \qquad V_i\sim N(0,1)",
        rf"X_i=\min\{{\max\{{\mathrm{{yuvarla}}(13{{,}}5+\pi A_i+1{{,}}7\,V_i),\,8\}},\,20\}}, \qquad \pi={number(p['pi'], 1)}",
        rf"Y_i=1{{,}}25+0{{,}}08\,X_i+\delta A_i+0{{,}}35\,e_i, \qquad \delta={number(p['delta'], 2)}",
    ),
    dgp_note=(
        "Eğitimin ücrete nedensel etkisi 0,08'dir: yetenek sabitken bir yıl fazla eğitim log ücreti "
        "0,08 artırır. π yeteneğin eğitime, δ ücrete etkisidir. Gerçek veride A gözlenmez; "
        "simülasyonda gözlediğimiz için onu sabit tutan regresyonu da çalıştırabiliyoruz."
    ),
    look_at=(
        "**Gri noktalar** — gözlemler.",
        "**Siyah doğru** — nedensel ilişki: yetenek ortalamadayken $Y=1{,}25+0{,}08\\,x$.",
        "**Kırmızı doğru** — OLS doğrusu: $Y$'nin yalnız $X$ üzerindeki projeksiyonu. Gerçek veride "
        "çalıştırabildiğimiz regresyon budur.",
    ),
    build=_build_confounding,
    metrics=_confounding_metrics,
    takeaway=_confounding_takeaway,
    labels=LABELS,
)


KONU01_EXPERIMENTS = (PROJECTION, RESIDUALS, CONFOUNDING)
