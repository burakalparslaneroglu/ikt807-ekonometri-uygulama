"""Konu 10 genel uygulaması: aynı katsayı için analitik ve bootstrap belirsizliği.

Notlardaki §10.17'nin beş adımı aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde yazılır (``build``):
orijinal OLS tahmini ve HC1 standart hatası, 1.000 tekrarlı pairs bootstrap (tohum 807), yeniden örnekleme birimi,
bootstrap tablosunu okurken dört soru ve bootstrap'ın çözemediği problem. Notlarda 3. adım yalnız metindir; veri kümeliyse
genel uygulama bu adımda küme bootstrap'ını ve analitik karşılığı olan küme-dayanıklı standart hatayı da hesaplar.

Alternatif örnek Duflo, Dupas ve Kremer (2011) verisidir (Hansen'in arşivindeki ``DDK2011.dta``): Konu 3'teki tracking
ham farkı. Tracking okul düzeyinde atandığı için öğrenci düzeyinde (pairs) bootstrap belirsizliği küçük gösterir; okul
bootstrap'ı okul-kümeli standart hatayla aynı büyüklüktedir. "Kendi verini yükle" seçeneğinde aynı adımlar öğrencinin
dosyasıyla kurulur. Notlardaki laboratuvar (``core.labs.konu10``) değişmez.

Rollerin notlardaki karşılıkları: sonuç log ücret, hedef açıklayıcı eğitim, kontroller deneyim, deneyim²/100 ve kadın;
küme notlarda yoktur (Adım 3 metin).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cache

import numpy as np
import pandas as pd

from core.labs import expr as E
from core.labs import kendi_veri as K
from core.labs.ornek import (
    SCALE_MESSAGE,
    Case,
    CustomLab,
    Role,
    TopicVariants,
    exact_fit,
    full_rank,
    md,
    sayi,
    sayim,
    stable_design,
    with_app_values,
    with_expected,
)
from core.labs.ornek_veri import sinif_verisi
from core.labs.runner import run_lab
from core.labs.spec import (
    BOOT,
    OLS,
    Bootstrap,
    Check,
    CoefTarget,
    DropMissing,
    Histogram,
    LabSpec,
    LabStep,
    LoadHansen,
    NoteRef,
    Operation,
    ReproClass,
    ScalarTarget,
    StatTarget,
)


def _sayi(value: float, decimals: int = 0) -> str:
    """Metinlerde Türkçe sayı, binlik ayırıcıyla (12.345,6)."""

    return sayi(value, decimals, binlik=True)


TOPIC = "konu10"
SECTION = "10.17"
SONUC, HEDEF, KUME = "sonuc", "hedef", "kume"
REPS = 1000
SEED = 807
COLUMN = "hedef"
"""Bootstrap tablosunda hedef katsayının sütun adı (skalerler ``boot_hedef_se`` vb.)."""
SE_FACTOR = 6.0 / math.sqrt(2.0 * REPS)
"""R ve Stata için bootstrap SH toleransı, SH'nin katı: Monte Carlo standart hatası yaklaşık SH/√(2B); tolerans altı
katıdır (notlardaki Konu 10 kuralı)."""
PERCENTILE_FACTOR = 6.0 * math.sqrt(0.025 * 0.975 / REPS) / (math.exp(-0.5 * 1.959964 ** 2) / math.sqrt(2 * math.pi))
"""Yüzde 2,5 ve 97,5 yüzdeliklerinin toleransı, bootstrap SH'nin katı: Monte Carlo standart hatası
√(p(1−p)/B)/f(q), dağılım yaklaşık normalken f(q) = φ(1,96)/SH; tolerans altı katıdır (≈ 0,51·SH). İki bağımsız
çekilişin farkının standart sapması √2 kat büyük olduğu için bu, farkın yaklaşık 4,2 standart sapmasıdır."""


@dataclass(frozen=True)
class Plan:
    """Rollerin koddaki sütunları ve gösterim ayarları.

    ``decimals``: katsayı, standart hata ve yüzdeliklerin ondalığı. ``histogram``: iki histogramın ortak aralığı (pairs
    ve küme bootstrap'ı aynı eksende karşılaştırılır).
    """

    frame: str
    y: str
    x: str
    controls: tuple[str, ...]
    cluster: str | None
    prepare: tuple[Operation, ...]
    labels: dict
    x_label: str
    decimals: int
    histogram: tuple[float, float]
    estimate: float
    titles: dict
    """``pairs`` ve ``kume`` histogram başlıkları."""
    unit: str = "gözlem"
    cluster_unit: str = "küme"


def _key(result: str, statistic: str) -> str:
    return f"{result}_{COLUMN}_{statistic}"


def _bootstrap_checks(result: str, decimals: int, label: str) -> tuple[Check, ...]:
    return (
        Check(f"{label} SH", ScalarTarget(_key(result, "se")), 0.0, decimals=decimals, mc_tolerance=1.0),
        Check(f"{label} percentile alt sınır", ScalarTarget(_key(result, "lo")), 0.0, decimals=decimals,
              mc_tolerance=1.0),
        Check(f"{label} percentile üst sınır", ScalarTarget(_key(result, "hi")), 0.0, decimals=decimals,
              mc_tolerance=1.0),
    )


def _steps(plan: Plan, texts: dict) -> tuple[LabStep, ...]:
    f, d = plan.frame, plan.decimals
    regressors = (plan.x, *plan.controls)
    low, high = plan.histogram
    references = ((plan.estimate, f"Orijinal OLS tahmini ({_sayi(plan.estimate, d)})"),)
    if low < 0 < high:
        references += ((0.0, "Sıfır"),)
    step3_ops: tuple = ()
    step3_checks: tuple = ()
    if plan.cluster:
        step3_ops = (
            OLS("ols_kume", f, plan.y, regressors, vcov="cluster", cluster=plan.cluster),
            Bootstrap("ols", f, REPS, SEED, ((COLUMN, E.coef(BOOT, plan.x)),), "kume_boot",
                      f"{plan.x_label}: 1.000 tekrarlı küme bootstrap'ı ({plan.cluster_unit} düzeyinde)",
                      method="cluster", cluster=plan.cluster),
            Histogram("kume_boot", ((COLUMN, f"Küme bootstrap'ı: {plan.x_label}"),), low, high, references,
                      f"Bootstrap {plan.x_label.lower()} katsayısı", plan.titles["kume"]),
        )
        step3_checks = (
            Check("Küme-dayanıklı analitik SH" if plan.cluster_unit == "küme" else f"{plan.cluster_unit.capitalize()}-kümeli analitik SH", CoefTarget("ols_kume", plan.x, "se"), 0.0,
                  decimals=d),
            *_bootstrap_checks("kume_boot", d, "Küme bootstrap'ı"),
        )
    return (
        LabStep(
            number=1,
            title="Orijinal tahmin",
            note=NoteRef(SECTION, 1),
            explanation=texts[1][0],
            operations=(*plan.prepare, OLS("ols", f, plan.y, regressors, vcov="HC1")),
            checks=(
                Check("Analiz örneklemi (N)", StatTarget(f, plan.y, "count"), 0.0, decimals=0),
                Check(f"{plan.x_label} katsayısı", CoefTarget("ols", plan.x), 0.0, decimals=d),
                Check(f"{plan.x_label} katsayısının HC1 SH'si", CoefTarget("ols", plan.x, "se"), 0.0, decimals=d),
            ),
            reproducibility=ReproClass.EXACT,
            takeaway=texts[1][1],
            note_for=texts.get((1, "not")),
        ),
        LabStep(
            number=2,
            title="Pairs bootstrap ile 1.000 yeniden örnekleme",
            note=NoteRef(SECTION, 2, ("Tablo 10.3", "Şekil 10.3")),
            explanation=texts[2][0],
            operations=(
                Bootstrap("ols", f, REPS, SEED, ((COLUMN, E.coef(BOOT, plan.x)),), "boot",
                          f"{plan.x_label}: 1.000 tekrarlı pairs bootstrap ({plan.unit} satırları)"),
                Histogram("boot", ((COLUMN, f"Pairs bootstrap: {plan.x_label}"),), low, high, references,
                          f"Bootstrap {plan.x_label.lower()} katsayısı", plan.titles["pairs"]),
            ),
            checks=_bootstrap_checks("boot", d, "Pairs bootstrap"),
            reproducibility=ReproClass.DISTRIBUTIONAL,
            takeaway=texts[2][1],
            code_note=texts.get("kod2", ""),
            note_for=texts.get((2, "not")),
        ),
        LabStep(
            number=3,
            title="Bootstrap kodunda yeniden örnekleme birimini açıkça görmek",
            note=NoteRef(SECTION, 3),
            explanation=texts[3][0],
            operations=step3_ops,
            checks=step3_checks,
            reproducibility=ReproClass.DISTRIBUTIONAL if plan.cluster else ReproClass.EXACT,
            takeaway=texts[3][1],
            code_note=texts.get("kod3", ""),
            note_for=texts.get((3, "not")),
        ),
        LabStep(
            number=4,
            title="Bootstrap tablosunu okurken dört soru",
            note=NoteRef(SECTION, 4),
            explanation=QUESTIONS,
            takeaway=texts[4][1],
            note_for=texts.get((4, "not")),
        ),
        LabStep(
            number=5,
            title="Bootstrap'ın çözemediği problemi görünür kılmak",
            note=NoteRef(SECTION, 5),
            explanation=texts[5][0],
            takeaway=texts[5][1],
            note_for=texts.get((5, "not")),
        ),
    )


def _spec(plan: Plan, texts: dict, *, source: str, title: str, dataset: str) -> LabSpec:
    labels = {**plan.labels, E.INTERCEPT: "Sabit", "ols": "OLS (HC1)", "ols_kume": "OLS (küme SH)",
              COLUMN: plan.x_label}
    return LabSpec(topic_key=TOPIC, title=title, dataset=dataset, note_section=SECTION, steps=_steps(plan, texts),
                   labels=tuple(labels.items()), source=source)


def with_tolerances(spec: LabSpec, run=None) -> LabSpec:
    """Bootstrap kontrollerinin R ve Stata toleransı: SH için ``SE_FACTOR``·SH, yüzdelikler için
    ``PERCENTILE_FACTOR``·SH (SH o bootstrap'ın uygulamadaki standart hatası), en az gösterim basamağının yarısı."""

    run = run_lab(spec) if run is None else run
    steps = []
    for step in spec.steps:
        checks = []
        for check in step.checks:
            if check.mc_tolerance is not None:
                result = check.target.name.rsplit(f"_{COLUMN}_", 1)[0]
                se = float(run.state.scalars[_key(result, "se")])
                factor = SE_FACTOR if check.target.name.endswith("_se") else PERCENTILE_FACTOR
                check = replace(check, mc_tolerance=max(factor * se, 0.5 * 10 ** (-check.decimals)))
            checks.append(check)
        steps.append(replace(step, checks=tuple(checks)))
    return replace(spec, steps=tuple(steps))


QUESTIONS = (
    "1. **Ne yeniden örneklenmiş?** Satırlar mı, kümeler mi, artıklar mı?\n"
    "2. **Kaç tekrar yapılmış?** Nihai sonuç için Monte Carlo gürültüsü yeterince küçük mü?\n"
    "3. **Hangi güven aralığı?** Normal, percentile, basic, BCa veya percentile-$t$ mi?\n"
    "4. **Bootstrap hangi varsayımı taklit ediyor?** Bağımsız örnekleme mi, heteroskedastisite mi, küme "
    "bağımlılığı mı?"
)
SE_FORMULA = (
    "$B=1.000$ tekrarda elde edilen katsayıların standart sapması bootstrap standart hatasıdır:\n\n"
    "$$\\widehat{se}_{boot}=\\Bigl\\{\\frac{1}{B-1}\\sum_{b=1}^B\\bigl(\\widehat\\beta_{1b}^*"
    "-\\bar\\beta_1^*\\bigr)^2\\Bigr\\}^{1/2}.$$\n\n"
    "Percentile güven aralığı bootstrap katsayılarının yüzde 2,5 ve 97,5 yüzdelikleridir. Rastgele sayı üreteci tohumu "
    "807'dir; her tekrarda önce satır numaraları çekilir, sonra katsayılar tasarım matrisiyle çözülür."
)
UNIT_TEXT = (
    "Adım 2'nin kodunda yeniden örnekleme birimi tek bir satırdır: Python'da `i = rng.integers(0, len(y_b), len(y_b))`, "
    "R'de `sample.int(nrow(X_b), nrow(X_b), replace = TRUE)`, Stata'da `bsample`. Bu satır bireysel gözlemleri yeniden "
    "örnekler; gözlemler bağımsızsa doğrudur.\n\n"
    "Veri okul, firma, hane veya bölge içinde kümeliyse bireyleri tek tek yeniden örneklemek bağımlılık yapısını bozar. O "
    "durumda yeniden örnekleme birimi küme olmalıdır: kümeler yerine koyarak çekilir ve seçilen kümenin bütün gözlemleri "
    "birlikte gelir (Stata'da `bsample, cluster(küme)`)."
)


# --- Alternatif örnek: Duflo, Dupas ve Kremer (2011) ----------------------------------------------------------

ALT_DATA = "ddk2011"
ALT_FRAME = "ddk"
ALT_TITLE = "Tracking Etkisi İçin Analitik ve Bootstrap Belirsizliği"
ALT_ESTIMATE = 1.258


def alternative_plan() -> Plan:
    frame = ALT_FRAME
    return Plan(
        frame=frame, y="totalscore", x="tracking", controls=(), cluster="schoolid",
        prepare=(
            LoadHansen(ALT_DATA, "DDK2011.dta", frame),
            DropMissing(frame, ("totalscore", "tracking", "schoolid"),
                        "Analiz örneklemi: toplam puanı, tedavi durumu ve okulu gözlenen öğrenciler"),
        ),
        labels={"totalscore": "Toplam test puanı", "tracking": "Tracking", "schoolid": "Okul"},
        x_label="Tracking", decimals=3, histogram=(-1.0, 3.5), estimate=ALT_ESTIMATE,
        titles={"pairs": "DDK2011: öğrenci düzeyinde pairs bootstrap (B = 1.000)",
                "kume": "DDK2011: okul düzeyinde küme bootstrap'ı (B = 1.000)"},
        unit="öğrenci", cluster_unit="okul",
    )


ALT_TEXTS = {
    1: (
        "Bootstrap'ı soyut bir yeniden örnekleme algoritması olarak değil gerçek bir regresyon katsayısında uyguluyoruz: "
        "Duflo, Dupas ve Kremer'in (2011) Kenya tracking deneyi (Hansen'in `DDK2011.dta` dosyası). Konu 3'teki ham "
        "fark\n\n"
        "$$totalscore_i=\\beta_0+\\beta_1 tracking_i+u_i$$\n\n"
        "121 okulda toplam test puanı gözlenen 5.795 öğrenciyle tahmin edilir; $\\widehat\\beta_1$ tracking uygulanan "
        "okulların öğrencileriyle uygulanmayanların ortalama puan farkıdır. Tracking okul düzeyinde rastgele atanmıştır. "
        "Amaç katsayıyı değiştirmek değil, $\\widehat\\beta_1$'in örnekleme belirsizliğini iki yolla ölçmektir: "
        "heteroskedastisiteye dayanıklı (HC1) analitik standart hata ve bootstrap. Notlardaki gibi gözlem düzeyinde "
        "başlıyoruz; Adım 3 yeniden örnekleme birimini okula taşır.",
        "Ham fark 1,258 test puanı, HC1 standart hatası 0,239. Bu bir rastgele deney tahminidir: nedensel yorum "
        "tasarımdan gelir; bootstrap yorumu değiştirmez, yalnız belirsizliği yeniden ölçer.",
    ),
    2: (
        "Her bootstrap tekrarında 5.795 öğrenci satırı yerine koyarak yeniden örneklenir ve aynı regresyon baştan tahmin "
        "edilir. " + SE_FORMULA,
        "Pairs bootstrap standart hatası 0,236, HC1 standart hatası 0,239: ikisi de öğrencileri birbirinden bağımsız sayar "
        "ve birbirine yakındır. Percentile %95 aralığı [0,798; 1,745] sıfırı içermez. Bu yakınlık iki yöntemin aynı "
        "bağımsızlık varsayımını paylaştığını gösterir; varsayımın kendisini sınamaz.",
    ),
    3: (
        UNIT_TEXT + "\n\nBu veride tracking okul düzeyinde atanmıştır ve aynı okulun öğrencileri ortak öğretmen, yönetim ve "
        "çevre koşullarını paylaşır: bağımsız birim öğrenci değil okuldur. Küme bootstrap'ında 121 okul yerine koyarak "
        "çekilir, seçilen okulun bütün öğrencileri birlikte gelir. Analitik karşılığı Konu 3'teki okul-kümeli standart "
        "hatadır.",
        "Okul bootstrap'ının standart hatası 0,726, okul-kümeli analitik standart hata 0,704: yine birbirine yakın ve "
        "ikisi de pairs bootstrap standart hatasının (0,236) yaklaşık üç katı. Percentile aralığı [−0,200; 2,736] sıfırı "
        "içerir. Aynı veri, aynı katsayı, "
        "aynı B = 1.000; yalnız yeniden örnekleme birimi değişti ve aralık sıfırı dışlamaktan içermeye döndü. Yeniden "
        "örnekleme birimi, verinin hangi bileşenlerinin bağımsız kabul edildiğinin kod karşılığıdır.",
    ),
    4: (
        QUESTIONS,
        "Bu laboratuvarın cevapları: Adım 2'de öğrenci satırları, Adım 3'te okullar; B = 1.000; percentile; Adım 2 "
        "bağımsız öğrencileri (heteroskedastisiteye izin vererek), Adım 3 okul içi bağımlılığı taklit eder. Atama birimi okul "
        "olduğu için tasarıma uyan şema okul bootstrap'ıdır.",
    ),
    5: (
        "Bu veride tanımlama rastgele atamadan gelir; bootstrap yalnız belirsizliği ölçer. Gözlemsel bir veride, örneğin "
        "notlardaki eğitim katsayısında, gözlenmeyen yetenek nedeniyle içsellik varsa pairs ya da küme bootstrap'ı içsel "
        "regresyonu 1.000 kez yeniden tahmin eder: örnekleme dağılımını daha iyi yaklaşıklar, yanlış hedefi doğru hedefe "
        "dönüştürmez. Aynı mantık zayıf araç veya yanlış kurulmuş bir RDD için de geçerlidir.",
        "Tez yazımında örnek raporlama: \"Tracking etkisinin okul-kümeli standart hatası 0,704, okul düzeyinde 1.000 "
        "tekrarlı küme bootstrap standart hatası 0,726'dır; bootstrap percentile yüzde 95 aralığı [−0,20; 2,74]'tür. "
        "Öğrenci düzeyinde bootstrap (0,236) atama birimini yok saydığı için belirsizliği yaklaşık üç kat küçük "
        "gösterir. Nedensel yorum okul düzeyindeki rastgele atamaya dayanır.\"",
    ),
}

ALT_CODE_NOTES = {
    "kod2": (
        "Regresyon her tekrarda doğrudan tasarım matrisiyle çözülür (`np.linalg.lstsq`, R'de `lm.fit`). Python betiği "
        "uygulamadaki çekilişin aynısını yapar ve sayıları birebir verir. R (`set.seed(807)`) ve Stata (`set seed 807`) "
        "başka rastgele sayı üreteci kullanır; betik bootstrap SH'sini ±{se_tol}, percentile sınırlarını ±{pct_tol} Monte "
        "Carlo toleransıyla denetler (R'de SH {r_se}, sınırlar {r_lo} ve {r_hi} çıkar)."
    ),
    "kod3": (
        "Küme bootstrap'ında okullar yerine koyarak çekilir (Python `rng.integers(0, G, G)`, R `sample.int(G, G, replace = "
        "TRUE)`, Stata `bsample, cluster(schoolid)`); seçilen okulun bütün satırları eklenir. R ve Stata yine farklı "
        "çekilişler yapar: SH ±{se_tol}, sınırlar ±{pct_tol} toleransla denetlenir (R'de SH {r_se}, sınırlar {r_lo} ve "
        "{r_hi})."
    ),
}

ALT_EXPECTED = {
    (1, "Analiz örneklemi (N)"): 5795,
    (1, "Tracking katsayısı"): 1.258,
    (1, "Tracking katsayısının HC1 SH'si"): 0.239,
    (2, "Pairs bootstrap SH"): 0.236,
    (2, "Pairs bootstrap percentile alt sınır"): 0.798,
    (2, "Pairs bootstrap percentile üst sınır"): 1.745,
    (3, "Okul-kümeli analitik SH"): 0.704,
    (3, "Küme bootstrap'ı SH"): 0.726,
    (3, "Küme bootstrap'ı percentile alt sınır"): -0.200,
    (3, "Küme bootstrap'ı percentile üst sınır"): 2.736,
}
"""Kontrollerin DDK2011 örneklemindeki (5.795 öğrenci, 121 okul) değerleri; bootstrap değerleri tohum 807 ile uygulamanın
(ve Python betiğinin) çekilişleridir. Testler bağımsız bir hesapla doğrular."""

ALT_TOLERANCES = {
    result: tuple(math.ceil(factor * ALT_EXPECTED[(step, f"{label} SH")] * 100) / 100
                  for factor in (SE_FACTOR, PERCENTILE_FACTOR))
    for result, step, label in (("boot", 2, "Pairs bootstrap"), ("kume_boot", 3, "Küme bootstrap'ı"))
}
"""(SH, yüzdelik) Monte Carlo toleransları: ``SE_FACTOR``·SH ve ``PERCENTILE_FACTOR``·SH, iki ondalığa yukarı
yuvarlanmış (pairs 0,04 ve 0,12; küme 0,10 ve 0,37)."""
ALT_R_VALUES = {"boot": ("0,237", "0,785", "1,721"), "kume_boot": ("0,688", "−0,138", "2,449")}
"""R'nin (set.seed(807)) bootstrap sonuçları; kod notlarında gösterilir, testle doğrulanır."""


def _alt_code_notes() -> dict:
    notes = {}
    for key, result in (("kod2", "boot"), ("kod3", "kume_boot")):
        se_tol, pct_tol = ALT_TOLERANCES[result]
        r_se, r_lo, r_hi = ALT_R_VALUES[result]
        notes[key] = ALT_CODE_NOTES[key].format(se_tol=_sayi(se_tol, 2), pct_tol=_sayi(pct_tol, 2), r_se=r_se, r_lo=r_lo,
                                                r_hi=r_hi)
    return notes


@cache
def alternative() -> LabSpec:
    """Alternatif örneğin tanımı; beklenen değerler DDK2011 örneklemindeki sayılardır (testle doğrulanır)."""

    texts = dict(ALT_TEXTS)
    texts.update(_alt_code_notes())
    spec = with_expected(_spec(alternative_plan(), texts, source="alternatif", title=ALT_TITLE, dataset=ALT_DATA),
                         ALT_EXPECTED)
    steps = []
    for step in spec.steps:
        checks = []
        for check in step.checks:
            if check.mc_tolerance is not None:
                result = check.target.name.rsplit(f"_{COLUMN}_", 1)[0]
                se_tol, pct_tol = ALT_TOLERANCES[result]
                check = replace(check, mc_tolerance=se_tol if check.target.name.endswith("_se") else pct_tol)
            checks.append(check)
        steps.append(replace(step, checks=tuple(checks)))
    return replace(spec, steps=tuple(steps))


STORY = (
    "Alternatif örnek Duflo, Dupas ve Kremer (2011) verisidir: Kenya'da 121 okulda 5.795 öğrenci (Hansen'in arşivindeki "
    "DDK2011.dta). Konu 3'teki tracking ham farkının belirsizliği HC1, öğrenci düzeyinde pairs bootstrap, okul-kümeli "
    "standart hata ve okul düzeyinde küme bootstrap'ı ile ölçülür. Adımlar notlardaki gibidir; 3. adımda yeniden "
    "örnekleme birimi okula taşınır."
)


# --- Kendi verin --------------------------------------------------------------------------------------------

MIN_ROWS = 30
MIN_CLUSTERS = 20
"""Küme bootstrap'ı için en az küme sayısı."""
MIN_GROUP = 10
"""Hedef açıklayıcı iki değerliyse her değerde en az gözlem (bootstrap tekrarlarında değişken sabit kalmasın)."""
MAX_DECIMALS = 8


def _fits(case: Case) -> tuple[float, float, float | None]:
    """Uygulamanın OLS tahmini, HC1 SH'si ve (küme varsa) küme SH'si (ondalık ve histogram aralığı için)."""

    import statsmodels.api as sm

    data = case.data
    y, x = case.roles[SONUC], case.roles[HEDEF]
    cluster = case.roles.get(KUME)
    columns = [x, *case.extras]
    design = sm.add_constant(data[columns].astype(float), has_constant="add")
    model = sm.OLS(data[y].astype(float), design)
    hc1 = model.fit(cov_type="HC1")
    clustered = None
    if cluster:
        codes = pd.factorize(data[cluster])[0]
        clustered = float(model.fit(cov_type="cluster", cov_kwds={"groups": codes}).bse[x])
    return float(hc1.params[x]), float(hc1.bse[x]), clustered


def _nice_range(center: float, half: float) -> tuple[float, float]:
    """Histogram aralığı: merkez ± yarı genişlik, dışa doğru düzgün bir adıma yuvarlanmış."""

    step = 10.0 ** math.floor(math.log10(half)) if half > 0 else 1.0
    return (float(math.floor((center - half) / step) * step), float(math.ceil((center + half) / step) * step))


def own_plan(case: Case) -> Plan:
    y, x = case.roles[SONUC], case.roles[HEDEF]
    cluster = case.roles.get(KUME)
    estimate, hc1, clustered = _fits(case)
    widest = max(hc1, clustered or 0.0)
    decimals = int(min(MAX_DECIMALS, max(2, 2 - math.floor(math.log10(hc1))))) if hc1 > 0 else 2
    labels = {column: case.name(column) for column in (y, x, *case.extras) + ((cluster,) if cluster else ())}
    return Plan(
        frame=case.frame, y=y, x=x, controls=case.extras, cluster=cluster, prepare=case.load, labels=labels,
        x_label=case.name(x), decimals=decimals, histogram=_nice_range(estimate, 4.5 * widest),
        estimate=round(estimate, decimals),
        titles={"pairs": "Kendi veriniz: pairs bootstrap (B = 1.000)",
                "kume": "Kendi veriniz: küme bootstrap'ı (B = 1.000)"},
    )


def _interval(state, result: str, d: int) -> tuple[str, bool]:
    lo, hi = state.scalars[_key(result, "lo")], state.scalars[_key(result, "hi")]
    return f"[{_sayi(lo, d)}; {_sayi(hi, d)}]", bool(lo <= 0 <= hi)


def _own_texts(case: Case, plan: Plan) -> dict:
    y, x = case.md(SONUC), case.md(HEDEF)
    cluster = case.md(KUME) if plan.cluster else None
    d = plan.decimals
    controls = ", ".join(f"“{md(case.name(column))}”" for column in plan.controls) if plan.controls else ""
    model = (f"“{y}” sonucunun “{x}” üzerine OLS regresyonu"
             + (f" (kontroller: {controls})" if controls else " (kontrol yok)"))

    def step1(state) -> str:
        fit = state.models["ols"]
        return (f"Analiz örneklemi {sayim(fit.nobs)} gözlem. “{x}” katsayısı {_sayi(float(fit.params[plan.x]), d)}, HC1 "
                f"standart hatası {_sayi(float(fit.bse[plan.x]), d)}.")

    def step2(state) -> str:
        hc1 = float(state.models["ols"].bse[plan.x])
        se = float(state.scalars[_key("boot", "se")])
        interval, zero = _interval(state, "boot", d)
        return (f"Pairs bootstrap standart hatası {_sayi(se, d)}, HC1 {_sayi(hc1, d)} (oran {_sayi(se / hc1, 2)}). "
                f"Percentile %95 aralığı {interval}" + (" sıfırı içerir." if zero else " sıfırı içermez.")
                + " İki yöntem de gözlemleri birbirinden bağımsız sayar.")

    def step3(state) -> str:
        if not plan.cluster:
            return ("Küme seçilmedi: bu analiz gözlemleri bağımsız kabul eder. Veriniz kümeliyse (ör. okul, firma, il) "
                    "küme sütununu seçin; Adım 3'te küme bootstrap'ı ve küme-dayanıklı standart hata hesaplanır.")
        analytic = float(state.models["ols_kume"].bse[plan.x])
        se, pairs = float(state.scalars[_key("kume_boot", "se")]), float(state.scalars[_key("boot", "se")])
        interval, zero = _interval(state, "kume_boot", d)
        _, pairs_zero = _interval(state, "boot", d)
        text = (f"Küme bootstrap'ının standart hatası {_sayi(se, d)} (pairs bootstrap'ınkinin {_sayi(se / pairs, 2)} katı); "
                f"“{cluster}” düzeyinde kümelenmiş analitik standart hata {_sayi(analytic, d)}. Percentile %95 aralığı "
                f"{interval}" + (" sıfırı içerir." if zero else " sıfırı içermez."))
        if zero != pairs_zero:
            text += " Yalnız yeniden örnekleme birimi değişti ve aralığın sıfıra göre konumu değişti."
        return text

    def step4(state) -> str:
        unit = (f"Adım 2'de gözlem satırları, Adım 3'te “{cluster}” kümeleri" if plan.cluster else "gözlem satırları")
        assumption = ("Adım 2 bağımsız gözlemleri, Adım 3 küme içi bağımlılığı taklit eder" if plan.cluster else
                      "bağımsız gözlemler (heteroskedastisiteye izin vererek)")
        return f"Bu analizin cevapları: {unit}; B = 1.000; percentile; {assumption}."

    cluster_text = (
        f"\n\nBu veride gözlemler “{cluster}” kümeleri içinde bağımlı olabilir. Küme bootstrap'ında kümeler yerine koyarak "
        "çekilir, seçilen kümenin bütün gözlemleri birlikte gelir; analitik karşılığı küme-dayanıklı standart hatadır."
        if plan.cluster else
        "\n\nKüme seçilmedi; bu adım yalnız açıklamadır. Veriniz kümeliyse küme sütununu seçin.")
    return {
        1: (f"Model: {model}. Amaç katsayıyı değiştirmek değil, “{x}” katsayısının örnekleme belirsizliğini iki yolla "
            "ölçmektir: heteroskedastisiteye dayanıklı (HC1) analitik standart hata ve bootstrap.",
            "Bootstrap tahminin nedensel yorumunu değiştirmez: yorum, katsayının hangi tanımlama varsayımına dayandığına "
            "bağlıdır."),
        (1, "not"): step1,
        2: ("Her bootstrap tekrarında gözlem satırları yerine koyarak yeniden örneklenir ve aynı regresyon baştan tahmin "
            "edilir. " + SE_FORMULA,
            "Bootstrap ile HC1'in yakın çıkması yararlı bir sağlamlık bulgusudur; ama iki yöntem aynı regresyon "
            "spesifikasyonuna ve aynı bağımsızlık varsayımına dayanır."),
        (2, "not"): step2,
        3: (UNIT_TEXT + cluster_text,
            "Yeniden örnekleme birimi, verinin hangi bileşenlerinin bağımsız kabul edildiğinin kod karşılığıdır."),
        (3, "not"): step3,
        4: (QUESTIONS, "Makalede yalnız \"standart hatalar bootstrap ile hesaplandı\" ifadesi çoğu zaman yetersizdir."),
        (4, "not"): step4,
        5: ("Hedef katsayıda içsellik varsa (gözlenmeyen bir değişken hem “" + x + "” değişkenini hem “" + y + "” "
            "sonucunu etkiliyorsa) bootstrap içsel regresyonu 1.000 kez yeniden tahmin eder: örnekleme dağılımını daha iyi "
            "yaklaşıklar, yanlış hedefi doğru hedefe dönüştürmez. Aynı mantık zayıf araç veya yanlış kurulmuş bir RDD "
            "için de geçerlidir.",
            "Bootstrap belirsizliği yeniden hesaplar; tanımlama sorununu onarmaz."),
        "kod2": ("Regresyon her tekrarda doğrudan tasarım matrisiyle çözülür (`np.linalg.lstsq`, R'de `lm.fit`). Python "
                 "betiği uygulamadaki çekilişin aynısını yapar. R (`set.seed(807)`) başka rastgele sayı üreteci kullanır; "
                 "betik bootstrap SH'sini yaklaşık ±0,13·SH, percentile sınırlarını ±0,51·SH Monte Carlo toleransıyla "
                 "denetler."),
        "kod3": ("Küme bootstrap'ında kümeler yerine koyarak çekilir (Python `rng.integers(0, G, G)`, R `sample.int(G, G, "
                 "replace = TRUE)`); seçilen kümenin bütün satırları eklenir. R farklı çekilişler yapar ve aynı "
                 "toleranslarla denetlenir." if plan.cluster else ""),
    }


def build(case: Case) -> LabSpec:
    """Kendi verinle Konu 10 uygulaması; beklenen değerler uygulamanın hesabıdır, bootstrap toleransları SH'ye göre."""

    plan = own_plan(case)
    spec = with_app_values(_spec(plan, _own_texts(case, plan), source="kendi",
                                 title=f"{case.label(HEDEF)} katsayısının belirsizliği", dataset=""))
    return with_tolerances(spec)


def _replicates_identified(design: np.ndarray, target: int, clusters=None) -> bool:
    """Tohum 807 ile uygulamanın (ve Python betiğinin) bütün bootstrap tekrarlarında hedef sütun diğer sütunlardan
    doğrusal bağımsız mı (hedef katsayı tanımlı mı)."""

    rng = np.random.default_rng(SEED)
    n = len(design)
    others = np.delete(design, target, axis=1)
    if clusters is not None:
        _, codes = np.unique(np.asarray(clusters), return_inverse=True)
        members = [np.flatnonzero(codes == g) for g in range(int(codes.max()) + 1)]
    for _ in range(REPS):
        if clusters is None:
            index = rng.integers(0, n, n)
        else:
            chosen = rng.integers(0, len(members), len(members))
            index = np.concatenate([members[g] for g in chosen])
        sample, rest = design[index], others[index]
        if np.linalg.matrix_rank(sample) != np.linalg.matrix_rank(rest) + 1:
            return False
    return True


def validate(case: Case) -> None:
    y, x = case.roles[SONUC], case.roles[HEDEF]
    cluster = case.roles.get(KUME)
    if y == x or cluster in (y, x):
        raise K.UploadError("Sonuç, hedef açıklayıcı ve küme için farklı sütunlar seçin.")
    data = case.data
    if len(data) < MIN_ROWS:
        raise K.UploadError(f"Analiz için en az {MIN_ROWS} gözlem gerekir; seçilen sütunlarda {len(data)} gözlem var.")
    for column in (y, x):
        if data[column].nunique() < 2:
            raise K.UploadError(f"“{case.name(column)}” sütununda en az iki farklı değer olmalı.")
    columns = (x, *case.extras)
    if not full_rank(data, columns):
        raise K.UploadError("Seçilen açıklayıcılar arasında tam doğrusal bağlantı var (biri diğerlerinden hesaplanabiliyor). "
                            "Kontrollerden birini çıkarın.")
    if exact_fit(data, y, columns):
        raise K.UploadError(f"“{case.name(y)}”, açıklayıcıların (neredeyse) tam doğrusal bir fonksiyonu (R² ≈ 1): standart "
                            "hatalar yuvarlama hatasına duyarlıdır ve yazılımlar arasında aynı çıkmaz.")
    if not stable_design(data, y, (columns,)):
        raise K.UploadError(SCALE_MESSAGE)
    values = data[x].astype(float)
    if values.nunique() == 2 and values.value_counts().min() < MIN_GROUP:
        raise K.UploadError(f"“{case.name(x)}” iki değerli ve değerlerinden biri {MIN_GROUP}'dan az gözlemde: bootstrap "
                            "tekrarlarının bazılarında değişken sabit kalabilir ve katsayısı tanımsız olur.")
    groups = None
    if cluster:
        count = data[cluster].nunique()
        if count < MIN_CLUSTERS:
            raise K.UploadError(f"“{case.name(cluster)}” sütununda {count} küme var; küme bootstrap'ı için en az "
                                f"{MIN_CLUSTERS} küme gerekir.")
        if count == len(data):
            raise K.UploadError(f"“{case.name(cluster)}” sütununda her gözlem ayrı bir küme: küme bootstrap'ı pairs "
                                "bootstrap'ıyla aynı olur. Küme seçmeyin ya da doğru küme sütununu seçin.")
        groups = data[cluster].to_numpy()
    design = np.column_stack([np.ones(len(data)), data[list(columns)].to_numpy(dtype=float)])
    if not _replicates_identified(design, 1):
        raise K.UploadError(f"Pairs bootstrap tekrarlarının bazılarında “{case.name(x)}” katsayısı tanımsız kalıyor "
                            "(değişken yeniden örneklemede sabit ya da kontrollerle bağlantılı oluyor). Değişkende daha çok "
                            "çeşitlilik gerekir.")
    if groups is not None and not _replicates_identified(design, 1, groups):
        raise K.UploadError(f"Küme bootstrap'ının bazı tekrarlarında “{case.name(x)}” katsayısı tanımsız kalıyor (ör. "
                            "değişken küme düzeyinde ve seçilen kümelerde sabit). Daha çok küme gerekir.")
    _, hc1, _ = _fits(case)
    if not hc1 > 0 or 2 - math.floor(math.log10(hc1)) > MAX_DECIMALS:
        raise K.UploadError(f"“{case.name(x)}” katsayısının standart hatası çok küçük: değerler {MAX_DECIMALS} ondalıkta "
                            "ayırt edilemiyor. Değişkenlerin birimini değiştirin.")


def suggest(table: K.UploadedTable) -> dict[str, str]:
    """Küme rolü için adında okul, küme, firma, sınıf, il, köy ya da kurum sözcüğü geçen ve 20 ile gözlem sayısının
    yarısı arasında farklı değer alan sütun önerilir."""

    words = {"okul", "küme", "kume", "firma", "sınıf", "sinif", "school", "cluster", "köy", "koy", "kurum", "il"}
    rows = len(table.frame)
    for column in table.columns:
        if words & set(K.name_words(column)):
            count = table.frame[column].map(K.clean_text).dropna().nunique()
            if MIN_CLUSTERS <= count <= rows / 2:
                return {KUME: column}
    return {}


def sample() -> pd.DataFrame:
    """Örnek dosya: kurgusal okul düzeyinde program verisi (``core.labs.ornek_veri``)."""

    return sinif_verisi()


ROLES = (
    Role(SONUC, "Sonuç değişkeni", "sayisal", True, (1, 2, 3),
         "Regresyonun bağımlı değişkeni (notlarda log ücret)."),
    Role(HEDEF, "Hedef açıklayıcı", "sayisal", True, (1, 2, 3),
         "Belirsizliği ölçülen katsayının değişkeni (notlarda eğitim yılı)."),
    Role(KUME, "Küme (ör. okul)", "serbest", False, (3,),
         "Gözlemlerin bağımlı olduğu birim. Seçilirse Adım 3'te küme bootstrap'ı ve küme-dayanıklı standart hata "
         "hesaplanır. Boş hücre olamaz; en az 20 küme.", complete=True),
)

CUSTOM = CustomLab(
    roles=ROLES,
    build=build,
    sample=sample,
    intro=(
        "Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç ve belirsizliği ölçülecek hedef açıklayıcı seçilir; isterseniz "
        "kontroller ve gözlemlerin bağımlı olduğu küme (ör. okul) de seçilir. Seçilen sütunlarda boş hücresi olan satırlar "
        "analizden çıkarılır. Bootstrap 1.000 tekrar ve tohum 807 ile yapılır."
    ),
    min_rows=MIN_ROWS,
    extra_columns=True,
    extra_label="Kontroller (isteğe bağlı)",
    extra_help="Regresyona eklenen sayısal kontroller (notlarda deneyim, deneyim²/100 ve kadın); en çok 8.",
    max_extra=8,
    extra_required=True,
    validate=validate,
    suggest=suggest,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
