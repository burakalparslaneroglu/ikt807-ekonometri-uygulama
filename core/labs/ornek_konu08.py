"""Konu 8 genel uygulaması: bant genişliği bir model kararıdır.

Notlardaki §8.13'ün beş adımı aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde yazılır (``build``):
doğrusal model ve aralık ortalamaları, birini dışarıda bırakan ve küme-silmeli çapraz doğrulama, kübik polinom, kübik
spline ve iki bant genişliğinde yerel doğrusal tahmin, kümeli yapının tuning aşamasına taşınması ve okuma soruları.
Alternatif örnek Angrist ve Lavy (1999) verisidir (Hansen'in arşivindeki ``AL1999.dta``): İsrail'de 4. sınıfların
ortalama matematik puanı ile okulun dezavantajlı öğrenci yüzdesi. "Kendi verini yükle" seçeneğinde aynı adımlar
öğrencinin dosyasıyla kurulur. Notlardaki laboratuvar (``core.labs.konu08``) değişmez.

Rollerin notlardaki karşılıkları: sonuç toplam test puanı, sürekli açıklayıcı başlangıç yüzdeliği, küme okul. Küme
seçilmezse yalnız birini dışarıda bırakan CV hesaplanır ve CV'nin seçtiği h, iki katıyla karşılaştırılır.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from core.labs import expr as E
from core.labs import kendi_veri as K
from core.labs import smoothing as S
from core.labs.ornek import (
    Case,
    CustomLab,
    Role,
    TopicVariants,
    exact_fit,
    full_rank,
    katsayi,
    sayi,
    sayim,
    with_app_values,
    with_expected,
)
from core.labs.ornek_veri import calisma_verisi
from core.labs.spec import (
    OLS,
    BandwidthCV,
    BinMeans,
    Check,
    Curve,
    Derive,
    DropMissing,
    KeepIf,
    LabSpec,
    LabStep,
    LoadHansen,
    LocalCurve,
    LocalLinear,
    ModelLine,
    NoteRef,
    Operation,
    Plot,
    ProfileCurves,
    ReproClass,
    ScalarTarget,
    StatTarget,
    TableTarget,
)

TOPIC = "konu08"
SECTION = "8.13"
SONUC, ACIKLAYICI, KUME = "sonuc", "aciklayici", "kume"
BINS = 20
MAX_ROWS = 2500
"""Kendi verinde en çok gözlem: çapraz doğrulama n×n ağırlık matrisiyle hesaplanır (indirilen kodda da)."""
MIN_CLUSTERS = 10
MODELS = (("dogrusal", "Doğrusal"), ("kubik", "Kübik polinom"), ("spline", "Kübik spline"))
_LABELS = dict(MODELS)


@dataclass(frozen=True)
class Plan:
    """Rollerin koddaki sütunları, bant genişlikleri, baz fonksiyonları ve etiketler.

    ``h_main`` Adım 3'ün ana bant genişliğidir (küme varsa küme-silmeli CV'nin, yoksa birini dışarıda bırakan CV'nin
    seçtiği); ``h_other`` karşılaştırma bandıdır (küme varsa birini dışarıda bırakan CV'nin seçtiği; küme yoksa ya da iki
    ölçüt aynı h'yi seçerse ana bandın iki katı: ``doubled``). ``basis``: kübik ve spline terimleri (ad, ifade,
    açıklama); ``point``: kontrol etiketinde değerlendirme noktasının yazımı (ör. "dezavantajlı %{}"; ``str.format``
    kalıbı). ``raised``: CV ızgarasının alt ucu sayısal güvenilirlik için yükseltildi mi (kendi verin).
    """

    frame: str
    y: str
    x: str
    cluster: str | None
    prepare: tuple[Operation, ...]
    grid: tuple[float, float, float]
    h_main: float
    h_other: float
    knots: tuple[float, ...]
    points: tuple[float, ...]
    basis: tuple[tuple[str, E.Expr, str], ...]
    labels: dict
    point: str
    x_label: str
    y_label: str
    titles: dict
    """Grafik başlıkları: ``veri`` (Adım 1), ``cv`` ve ``profil`` (Adım 2), ``parametrik`` ve ``uyum`` (Adım 3)."""
    plot_grid: tuple[float, float, int]
    decimals: int = 2
    h_decimals: int = 1
    doubled: bool = False
    raised: bool = False

    @property
    def cubic(self) -> tuple[str, ...]:
        return (self.x, *(name for name, _, _ in self.basis[:2]))

    @property
    def spline(self) -> tuple[str, ...]:
        return (self.x, *(name for name, _, _ in self.basis))

    def expression(self, name: str) -> E.Expr:
        return next(expression for term, expression, _ in self.basis if term == name)


def _value(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return ("−" if value < 0 else "") + sayim(abs(value))
    return np.format_float_positional(value, trim="-").replace("-", "−").replace(".", ",")


def _column(h: float) -> str:
    """Yerel doğrusal tablo sütununun adı: h = 6,5 → ``h6_5``."""

    return "h" + np.format_float_positional(float(h), trim="-").replace(".", "_").replace("-", "m")


def _fit(plan: Plan, model: str, terms: tuple[str, ...]) -> E.Expr:
    """Tahmin edilen seri regresyonu m̂(x) = b₀ + Σ bⱼ pⱼ(x), grafik eğrisi için."""

    expression: E.Expr = E.coef(model, E.INTERCEPT)
    for term in terms:
        base = E.var(plan.x) if term == plan.x else plan.expression(term)
        expression = E.add(expression, E.mul(E.coef(model, term), base))
    return expression


def _titles(data: str, pair: str) -> dict:
    return {"veri": f"{data}: {pair}", "cv": f"{data}: çapraz doğrulama ölçütü",
            "profil": f"{data}: bant genişliği ve tahmin edilen profil", "parametrik": f"{data}: parametrik uyumlar",
            "uyum": f"{data}: doğrusal, polinom, spline ve yerel doğrusal uyumlar"}


def _h_label(h: float) -> str:
    return f"Yerel doğrusal h = {_value(h)}"


def _steps(plan: Plan, texts: dict) -> tuple[LabStep, ...]:
    f, x, y = plan.frame, plan.x, plan.y
    vcov = {"vcov": "cluster", "cluster": plan.cluster} if plan.cluster else {"vcov": "HC1"}
    main, other = _column(plan.h_main), _column(plan.h_other)
    prediction_checks = [
        Check(f"{label}, {plan.point.format(_value(point))}", TableTarget("parametrik", point, model), 0.0,
              plan.decimals)
        for model, label in MODELS for point in plan.points
    ] + [
        Check(f"{_h_label(h)}, {plan.point.format(_value(point))}", TableTarget("yerel", point, column), 0.0,
              plan.decimals)
        for column, h in ((main, plan.h_main), (other, plan.h_other)) for point in plan.points
    ]
    cv_checks = [Check("Birini dışarıda bırakan CV ile h", ScalarTarget("cv_h"), 0.0, decimals=plan.h_decimals)]
    if plan.cluster:
        cv_checks.append(Check("Küme-silmeli CV ile h", ScalarTarget("cv_h_kume"), 0.0, decimals=plan.h_decimals))
    other_dashed = LocalCurve(y, plan.h_other, _h_label(plan.h_other), dashed=True)
    return (
        LabStep(
            number=1,
            title="Araştırma sorusu: ilişki doğrusal mı?",
            note=NoteRef(SECTION, 1),
            explanation=texts[1][0],
            operations=(
                *plan.prepare,
                OLS("dogrusal", f, y, (x,), **vcov),
                Plot(f, x, (BinMeans(y, BINS, "Yirmi aralığın ortalaması"), ModelLine("dogrusal", "Doğrusal OLS")),
                     plan.x_label, plan.y_label, plan.titles["veri"]),
            ),
            checks=(Check("Analiz örneklemi (N)", StatTarget(f, y, "count"), 0.0, decimals=0),),
            reproducibility=ReproClass.EXACT,
            takeaway=texts[1][1],
            note_for=texts.get((1, "not")),
        ),
        LabStep(
            number=2,
            title="İki farklı CV yaklaşımı iki farklı h seçer",
            note=NoteRef(SECTION, 2, ("Şekil 8.4",)),
            explanation=texts[2][0],
            operations=(
                BandwidthCV("cv", f, x, y, plan.grid, "cv_tablosu", "Bant genişliği h", plan.titles["cv"],
                            cluster=plan.cluster),
                Plot(
                    f, x,
                    (BinMeans(y, BINS, "Yirmi aralığın ortalaması"), ModelLine("dogrusal", "Doğrusal OLS"),
                     LocalCurve(y, plan.h_main, _h_label(plan.h_main)), other_dashed),
                    plan.x_label, plan.y_label, plan.titles["profil"],
                ),
            ),
            checks=tuple(cv_checks),
            reproducibility=ReproClass.EXACT,
            takeaway=texts[2][1],
            code_note=texts.get("kod2", CODE_NOTE_2),
            note_for=texts.get((2, "not")),
        ),
        LabStep(
            number=3,
            title="Aynı veri, farklı düzgünlük",
            note=NoteRef(SECTION, 3, ("Tablo 8.3", "Tablo 8.1", "Şekil 8.3")),
            explanation=texts[3][0],
            operations=(
                *(Derive(f, name, expression, comment) for name, expression, comment in plan.basis),
                OLS("kubik", f, y, plan.cubic, **vcov),
                OLS("spline", f, y, plan.spline, **vcov),
                ProfileCurves(
                    models=MODELS,
                    frame=f,
                    variable=x,
                    derived=tuple((name, expression) for name, expression, _ in plan.basis),
                    values=plan.points,
                    result="parametrik",
                    plot_grid=plan.plot_grid,
                    x_label=plan.x_label,
                    y_label=plan.y_label,
                    title=plan.titles["parametrik"],
                ),
                LocalLinear("ll", f, x, y, ((main, plan.h_main), (other, plan.h_other)), plan.points, "yerel"),
                Plot(
                    f, x,
                    (
                        BinMeans(y, BINS, "Yirmi aralığın ortalaması"),
                        ModelLine("dogrusal", "Doğrusal"),
                        Curve(_fit(plan, "kubik", plan.cubic), "Kübik polinom"),
                        Curve(_fit(plan, "spline", plan.spline), "Kübik spline (düğümler çeyreklerde)", dashed=True),
                        LocalCurve(y, plan.h_main, _h_label(plan.h_main)),
                    ),
                    plan.x_label, plan.y_label, plan.titles["uyum"],
                ),
            ),
            checks=tuple(prediction_checks),
            reproducibility=ReproClass.EXACT,
            takeaway=texts[3][1],
            code_note=texts.get("kod3", CODE_NOTE_3),
            note_for=texts.get((3, "not")),
        ),
        LabStep(
            number=4,
            title="Kümeli yapıyı tuning aşamasına da taşımak",
            note=NoteRef(SECTION, 4),
            explanation=texts[4][0],
            takeaway=texts[4][1],
        ),
        LabStep(
            number=5,
            title="Bir parametrik olmayan makale çıktısını okurken",
            note=NoteRef("8.13.5", 0),
            explanation=QUESTIONS,
            takeaway=texts[5][1],
            note_for=texts.get((5, "not")),
        ),
    )


def _spec(plan: Plan, texts: dict, *, source: str, title: str, dataset: str) -> LabSpec:
    labels = {**plan.labels, **_LABELS}
    return LabSpec(topic_key=TOPIC, title=title, dataset=dataset, note_section=SECTION, steps=_steps(plan, texts),
                   labels=tuple(labels.items()), source=source)


# --- Ortak metinler -------------------------------------------------------------------------------------------

CV_RULE = (
    "Yerel doğrusal tahminde Gauss çekirdeği $K(u)=\\phi(u)$, $u=(X_i-x)/h$ kullanılır; $h$ çekirdeğin standart "
    "sapmasıdır. Çapraz doğrulama ölçütü dışarıda bırakılan tahmin hatalarının kareler ortalamasıdır:\n\n"
    "$$CV(h)=\\frac1n\\sum_{i=1}^n\\{Y_i-\\tilde m_{-i}(X_i;h)\\}^2.$$\n\n"
)
QUESTIONS = (
    "1. Tahmin edilen nesne koşullu ortalama mı, yoğunluk mu, olasılık mı?\n"
    "2. Çekirdek ve yerel polinom derecesi nedir?\n"
    "3. Bant genişliği nasıl seçilmiştir?\n"
    "4. Veri bağımlıysa CV ve standart hata kümeyi koruyor mu?\n"
    "5. Grafikte sınır bölgelerinde güven aralıkları genişliyor mu?\n"
    "6. Farklı makul $h$ değerleri ana ekonomik sonucu değiştiriyor mu?\n"
    "7. Çok boyutlu $X$ kullanılıyorsa boyutluluk laneti nasıl yönetiliyor?"
)
CLUSTER_RULE = (
    "Bağımlılık yalnız standart hata sorununa indirgenemez. Veri kümeli olduğunda standart hata, çapraz doğrulama, "
    "bootstrap ve eğitim/test bölmesi gibi bütün yeniden örnekleme veya doğrulama adımlarının kümeyi koruması gerekir."
)
CODE_NOTE_2 = (
    "Çapraz doğrulama n×n ağırlık matrisiyle vektörel hesaplanır; aynı fonksiyonlar Python'da NumPy, R'de matris "
    "işlemleri, Stata'da Mata ile yazılmıştır. Eşit CV değerlerinde küçük h seçilir."
)
CODE_NOTE_3 = (
    "Kübik polinom ve spline OLS'dir; baz terimleri açıkça türetilir, tahminler seçilmiş noktalarda x'β olarak "
    "hesaplanır. Standart hatalar küme düzeyinde kümelenmiştir (tahminleri etkilemez)."
)


# --- Alternatif örnek: Angrist ve Lavy (1999) -----------------------------------------------------------------

ALT_DATA = "al1999"
ALT_FRAME = "al"
ALT_TITLE = "Angrist–Lavy Verisinde Bant Genişliği Bir Model Kararıdır"
ALT_KNOTS = (4.0, 9.0, 19.0)
ALT_POINTS = (2.0, 9.0, 35.0)
ALT_H_CLUSTER, ALT_H_LOO = 6.5, 3.5


def _alt_basis() -> tuple[tuple[str, E.Expr, str], ...]:
    """Notlardaki ölçekleme: kare terim 100'e, kübik terimler 10⁴'e bölünür (x yüzde ölçeğinde, 0–76)."""

    x = E.var("disadvantaged")
    terms = [("dis2", E.div(E.power(x, 2), 100), "Dezavantajlı yüzdesinin karesi / 100"),
             ("dis3", E.div(E.power(x, 3), 10000), "Dezavantajlı yüzdesinin küpü / 10⁴")]
    terms += [(f"s{int(k)}", E.div(E.power(E.maximum(E.sub(x, k), 0), 3), 10000),
               f"Spline terimi (dezavantajlı − {int(k)})₊³ / 10⁴") for k in ALT_KNOTS]
    return tuple(terms)


def alternative_plan() -> Plan:
    frame = ALT_FRAME
    return Plan(
        frame=frame, y="avgmath", x="disadvantaged", cluster="schlcode",
        prepare=(
            LoadHansen(ALT_DATA, "AL1999.dta", frame),
            KeepIf(frame, (("grade", "==", 4),), "Analiz örneklemi: 4. sınıflar"),
            DropMissing(frame, ("avgmath", "disadvantaged", "schlcode"),
                        "Matematik puanı, dezavantajlı öğrenci yüzdesi veya okul kodu eksik sınıflar"),
        ),
        grid=(2.0, 20.0, 0.5), h_main=ALT_H_CLUSTER, h_other=ALT_H_LOO, knots=ALT_KNOTS, points=ALT_POINTS,
        basis=_alt_basis(),
        labels={
            E.INTERCEPT: "Sabit",
            "avgmath": "Ortalama matematik puanı",
            "disadvantaged": "Dezavantajlı öğrenci yüzdesi",
            "dis2": "Dezavantajlı²/100",
            "dis3": "Dezavantajlı³/10⁴",
            **{f"s{int(k)}": f"(dezavantajlı − {int(k)})₊³/10⁴" for k in ALT_KNOTS},
        },
        point="dezavantajlı %{}",
        x_label="Dezavantajlı öğrenci yüzdesi",
        y_label="Sınıfın ortalama matematik puanı",
        titles=_titles("AL1999", "dezavantajlı öğrenci yüzdesi ve matematik puanı"),
        plot_grid=(0.0, 76.0, 77),
    )


ALT_TEXTS = {
    1: (
        "Angrist ve Lavy (1999), İsrail'de 1991'de 4. ve 5. sınıflarda yapılan ulusal sınavları sınıf düzeyinde "
        "kullanarak sınıf mevcudunun başarıya etkisini Maimonides kuralıyla inceler. Hansen'in `AL1999.dta` dosyası iki "
        "sınıf düzeyini birleştirir (4.067 sınıf); burada 4. sınıflar kullanılır: 1.013 okulda 2.049 sınıf. Bu "
        "uygulamanın sorusu sınıf mevcudu değildir: sınıfın ortalama matematik puanı (`avgmath`) okulun dezavantajlı "
        "öğrenci yüzdesiyle (`disadvantaged`, okul düzeyinde bir sosyoekonomik endeks) doğrusal mı ilişkili, yoksa "
        "eğrilik var mı?\n\n"
        "Parametrik doğrusal model $\\mathbb E[Y\\mid X=x]=\\beta_0+\\beta_1x$ tek bir eğim dayatır. Grafikte 2.049 "
        "tekil nokta yerine $x$'in yirmi eşit genişlikli aralığındaki ortalamalar gösterilir; doğru bütün gözlemlerle "
        "tahmin edilmiştir. Aynı okulun sınıfları bağımlı olduğu için standart hatalar okul düzeyinde kümelenmiştir. "
        "Dezavantajlı yüzdesi okul düzeyinde ölçüldüğü için aynı okulun bütün sınıfları aynı $x$ değerini taşır."
    ),
    2: (
        CV_RULE + "**Birini dışarıda bırakan** CV'de $\\tilde m_{-i}$ yalnız $i$. sınıf çıkarılarak, **küme-silmeli** "
        "CV'de $i$'nin okulundaki bütün sınıflar çıkarılarak hesaplanır. Ölçüt "
        "$h\\in\\{2{,}0;\\,2{,}5;\\,\\ldots;\\,20{,}0\\}$ ızgarasında değerlendirilir. Bu veride açıklayıcı okul "
        "düzeyinde olduğu için birini dışarıda bırakan CV, dışarıda bırakılan sınıfı aynı okulun aynı $x$ değerindeki "
        "diğer sınıflarıyla tahmin edebilir."
    ),
    3: (
        "Aynı koşullu ortalamayı beş biçimde tahmin ediyoruz: doğrusal model; kübik polinom "
        "$\\beta_0+\\beta_1x+\\beta_2x^2+\\beta_3x^3$; düğümleri dezavantajlı yüzdesinin çeyreklerinde (%4, %9 ve %19) "
        "olan kübik spline (kübik polinoma $(x-4)_+^3$, $(x-9)_+^3$, $(x-19)_+^3$ terimleri eklenir); iki bant "
        "genişliğinde yerel doğrusal tahmin ($h=6{,}5$ ve $h=3{,}5$). Polinom ve spline sıradan OLS ile tahmin edilir; "
        "esneklik baz fonksiyonlarından gelir (katsayılar okunur ölçekte olsun diye kare terim 100'e, kübik terimler "
        "$10^4$'e bölünür; tahminler değişmez). Tahminler dezavantajlı yüzdesinin 10., 50. ve 90. yüzdeliklerinde (%2, %9 "
        "ve %35) karşılaştırılır."
    ),
    4: (
        "Aynı okulun sınıfları bağımlıdır: okulun öğrenci profili, öğretmenleri ve yönetimi bütün sınıflarını etkiler. Bu "
        "veride açıklayıcı da okul düzeyindedir. Birini dışarıda bırakan CV bir sınıfı çıkarırken aynı okulun aynı $x$ "
        "değerindeki sınıflarını tahminde tutar; küçük $h$ bu sınıflara çok ağırlık verir ve okul etkisini \"öğrenir\". "
        "Küme-silmeli yaklaşım okulun bütün sınıflarını birlikte dışarıda bırakır (Adım 2).\n\n" + CLUSTER_RULE
    ),
    5: QUESTIONS,
}
"""Adım açıklamaları; anlatılan sayılar kontrollerdir (``ALT_EXPECTED``) ya da testte bağımsız olarak hesaplanır."""

ALT_TAKEAWAYS = {
    1: (
        "Doğrusal modelin eğimi −0,296: dezavantajlı yüzdesi 10 puan yüksek okulların sınıflarında ortalama matematik "
        "puanı yaklaşık 3 puan düşük. Aralık ortalamaları %7,6'nın altında doğrunun üstünde, %7,6–38 aralığında (%26,6–30,4 "
        "aralığı dışında) altında, %38–60,8 aralığında yeniden üstünde kalır: ilişki düşük yüzdelerde dik, orta yüzdelerde "
        "daha yatıktır. %49,4'ün üstündeki yedi aralığın her birinde yalnız 1–22 sınıf vardır; orada aralık ortalamaları "
        "doğrunun iki yanına dağılır. Yerel doğrusal tahmin her $x$ noktasında yakın gözlemlere daha çok ağırlık vererek "
        "koşullu ortalama eğrisini tahmin eder."
    ),
    2: (
        "Birini dışarıda bırakan CV h = 3,5, küme-silmeli CV h = 6,5 seçer. Yön notlardaki DDK örneğinin tersidir (orada "
        "12,3 ve 6,2): hangi ölçütün daha dar bant seçeceği verinin yapısına bağlıdır. Burada küçük h, dışarıda bırakılan "
        "sınıfı aynı okulun sınıflarıyla tahmin ederek okul etkisinden yararlanır; okulun bütün sınıfları birlikte "
        "çıkarılınca bu avantaj kaybolur. Küme-silmeli ölçüt yaklaşık 5–8,5 aralığında çok yassıdır (en küçük değerden "
        "farkı 0,01'in altında): veri tek bir bant genişliğini keskin biçimde ayırt etmez. Yazılımın seçtiği tek sayı "
        "\"doğru bant genişliği\" değildir; ölçütün en küçük değer çevresindeki şekli de raporlanmalıdır."
    ),
    3: (
        "Ortada (%9) yöntemler birbirine yakındır (69,89–70,29). Alt uçta (%2) esnek yöntemler doğrusal modelden yüksek "
        "tahmin üretir: doğrusal 72,36, yerel doğrusal 72,87–73,02, kübik polinom 73,18, spline 73,61. Üst uçta (%35) "
        "bütün tahminler 62,58–62,81 aralığındadır. Küçük h yerel değişimleri izler, büyük h eğriyi düzleştirir. "
        "Karşılaştırılan üç noktada iki bant arasındaki fark en çok 0,23 puandır (%35'te); verinin seyrek olduğu uçlarda "
        "fark büyür: %0'da (87 sınıf) 0,84, %76'da (tek sınıf) 2,36 puan. Farklılıkların hiçbiri tek başına \"doğru "
        "model\" kanıtı değildir: veri desteği, tuning ölçütü ve makul alternatiflerde sonucun değişip değişmediği "
        "birlikte değerlendirilir."
    ),
    4: (
        "Tez cümlesi örneği: \"Yerel doğrusal tahminde ana bant genişliği okul düzeyinde küme-silmeli çapraz doğrulama "
        "ile seçilmiş (h = 6,5), birini dışarıda bırakan çapraz doğrulamanın seçtiği daha dar bantla (h = 3,5) duyarlılık "
        "analizi yapılmıştır. Dezavantajlı yüzdesinin 10., 50. ve 90. yüzdeliklerinde iki bandın tahminleri en çok 0,23 "
        "puan farklıdır; bu aralığın dışında, gözlemlerin seyrek olduğu uçlarda fark daha büyüktür.\""
    ),
    5: (
        "Bu laboratuvarın cevapları: koşullu ortalama; Gauss çekirdeği, yerel doğrusal; küme-silmeli CV (h = 6,5) ve "
        "duyarlılık için h = 3,5; kümeli yapı hem CV'de hem standart hatada korunur. Karşılaştırılan üç nokta içinde beş "
        "yöntem en çok %2'de ayrışır (72,36–73,61); gözlemlerin seyrek olduğu uçlarda ayrışma daha da büyüktür (en büyük "
        "ve en küçük tahmin arasındaki fark %0'da 3,07, %55'te 2,75, %76'da 8,78 puan)."
    ),
}

ALT_CODE_NOTES = {
    "kod2": (
        "Çapraz doğrulama n×n ağırlık matrisiyle vektörel hesaplanır (2.049 sınıf için yaklaşık 4,2 milyon öğe); 37 bant "
        "genişliği ve iki ölçüt birkaç saniye sürer. Aynı fonksiyonlar Python'da NumPy, R'de matris işlemleri, Stata'da "
        "Mata ile yazılmıştır. Eşit CV değerlerinde küçük h seçilir."
    ),
    "kod3": (
        "Kübik polinom ve spline OLS'dir; baz terimleri açıkça türetilir, tahminler %2, %9 ve %35'te x'β olarak "
        "hesaplanır. Standart hatalar okul düzeyinde kümelenmiştir (tahminleri etkilemez)."
    ),
}

ALT_EXPECTED = {
    (1, "Analiz örneklemi (N)"): 2049,
    (2, "Birini dışarıda bırakan CV ile h"): ALT_H_LOO,
    (2, "Küme-silmeli CV ile h"): ALT_H_CLUSTER,
    **{(3, f"{label}, dezavantajlı %{int(point)}"): value
       for label, values in (("Doğrusal", (72.36, 70.29, 62.60)), ("Kübik polinom", (73.18, 69.89, 62.66)),
                             ("Kübik spline", (73.61, 70.04, 62.71)),
                             ("Yerel doğrusal h = 6,5", (73.02, 70.18, 62.81)),
                             ("Yerel doğrusal h = 3,5", (72.87, 70.09, 62.58)))
       for point, value in zip(ALT_POINTS, values)},
}
"""Kontrollerin AL1999 4. sınıf örneklemindeki (2.049 sınıf) değerleri, gösterim basamağında: (adım, etiket) → değer.
Metinlerdeki sayılar bunlardır; testler bağımsız bir hesapla ve iki dilde doğrular."""


@cache
def alternative() -> LabSpec:
    """Alternatif örneğin tanımı; beklenen değerler AL1999'un 4. sınıf örneklemindeki sayılardır (testle doğrulanır)."""

    texts = {number: (ALT_TEXTS[number], ALT_TAKEAWAYS[number]) for number in ALT_TEXTS}
    texts.update(ALT_CODE_NOTES)
    spec = _spec(alternative_plan(), texts, source="alternatif", title=ALT_TITLE, dataset=ALT_DATA)
    return with_expected(spec, ALT_EXPECTED)


STORY = (
    "Alternatif örnek Angrist ve Lavy (1999) verisidir: İsrail'de 1.013 okulda 2.049 dördüncü sınıf (Hansen'in "
    "arşivindeki AL1999.dta). Sınıfın ortalama matematik puanının okulun dezavantajlı öğrenci yüzdesiyle ilişkisi esnek "
    "yöntemlerle tahmin edilir. Adımlar notlardaki gibidir: doğrusal model, birini dışarıda bırakan ve küme-silmeli CV, "
    "polinom, spline ve yerel doğrusal tahmin, kümeli yapı ve okuma soruları."
)


# --- Kendi verin --------------------------------------------------------------------------------------------

MAX_DECIMALS = 6
"""Tahminlerin en çok ondalığı. Sonucun ölçeği daha fazlasını gerektiriyorsa (standart sapma 10⁻⁴'ün altında) dosya
reddedilir: kontroller anlamsızlaşır (0,000000)."""
SPLINE_CONDITION = 1e6
"""Doğrusal, kübik ve spline tasarımlarının (sütunları birim uzunlukta, sabit terim dahil) en büyük koşul sayısı. R'nin
QR çözümü bir sütunu yalnız kalan normu ilk normunun 10⁻⁷'sinden küçükse düşürür; koşul sayısı 10⁶'nın altındayken bu
olmaz. Kontroller katsayılar değil seçilmiş noktalardaki tahminler olduğu için Konu 1–2'nin sınırından
(``CONDITION_LIMIT``) gevşektir: çarpık bir açıklayıcının kübik terimleri doğal olarak yüksek koşul sayısı verir
(AL1999'da 2·10⁴). Tahminlerin yazılımlar arasında aynılığı ayrıca sınanır (``_stable_fits``)."""
PREDICTION_LIMIT = 1e-9
"""Uygulamanın tahmini (statsmodels, sözde ters) ile sütunları ortalanıp ölçeklenmiş tasarımdaki tahmin arasında izin
verilen en büyük fark, sonucun standart sapmasına (büyük tahminlerde tahmine) göre."""
MIN_GRID = 5
"""Sayısal olarak güvenilir CV ızgarasında en az bu kadar bant genişliği kalmalı."""
CURVE_POINTS = 200
"""Grafik eğrilerinin nokta sayısı: uygulamada, Python'da ve R'de verinin en küçük ve en büyük değeri arasında 200 nokta."""


def _math(value: float) -> str:
    """Sayının matematik kipindeki yazımı: ondalık virgül ``{,}`` ile (KaTeX virgülden sonra boşluk bırakmasın)."""

    return _value(value).replace(",", "{,}")


def _shift(value: float) -> str:
    """"x − 4,7", "x + 4,7" ya da "x": negatif merkez ya da düğüm "x − −4,7" diye yazılmaz."""

    if value == 0:
        return "x"
    return f"x − {_value(value)}" if value > 0 else f"x + {_value(-value)}"


def _digits(values: pd.Series) -> int:
    """Nokta ve düğümlerin ondalığı: değişkenin standart sapmasına göre dört anlamlı basamak."""

    spread = float(values.std())
    if not math.isfinite(spread) or spread <= 0:
        return 2
    return int(max(0, 3 - math.floor(math.log10(spread))))


def _quantiles(values: pd.Series, levels: tuple[float, ...]) -> tuple[float, ...]:
    data = values.dropna().astype(float)
    return tuple(float(v) for v in np.round(np.quantile(data, levels), _digits(data)))


def _grid(values: pd.Series) -> tuple[tuple[float, float, float], int]:
    """Aday CV ızgarası: değişkenin aralığının 1/50'sinden 1/4'üne, yaklaşık 40 nokta; adım 1, 2 ya da 5 × 10ᵏ. Dönüş:
    (alt, üst, adım) ve h'nin ondalığı. Alt uç sayısal güvenilirlik için yükseltilebilir (``bandwidths``)."""

    data = values.dropna().astype(float)
    spread = float(data.max() - data.min())
    low, high = spread / 50, spread / 4
    raw = (high - low) / 40
    power = 10 ** math.floor(math.log10(raw))
    step = min((factor * power for factor in (1, 2, 5, 10) if factor * power >= raw), default=10 * power)
    decimals = max(0, -math.floor(math.log10(step) + 1e-12))
    low = math.ceil(low / step - 1e-9) * step
    high = math.floor(high / step + 1e-9) * step
    return (float(round(low, decimals)), float(round(high, decimals)), float(round(step, decimals))), decimals


def _basis(x: str, center: float, scale: float, knots: tuple[float, ...]) -> tuple[tuple[str, E.Expr, str], ...]:
    """Kübik ve spline terimleri ortalanmış ve ölçeklenmiş u = (x − c)/s üzerinden: kare ve küp terimler ile
    (x − k)₊³/s³ terimleri. Tahminler (aynı fonksiyon uzayı) notlardaki ölçeklemeyle aynıdır; değişkenin birimi büyük
    ya da küçük olsa da tasarım iyi koşulludur."""

    u = E.div(E.sub(E.var(x), center), scale)
    s = _value(scale)
    terms = [("kare_terim", E.power(u, 2), f"(({_shift(center)}) / {s})²"),
             ("kup_terim", E.power(u, 3), f"(({_shift(center)}) / {s})³")]
    terms += [(f"dugum{i}", E.power(E.div(E.maximum(E.sub(E.var(x), knot), 0), scale), 3),
               f"Spline terimi (({_shift(knot)})₊ / {s})³") for i, knot in enumerate(knots, start=1)]
    return tuple(terms)


def _decimals(values: pd.Series) -> int:
    """Tahminlerin ondalığı: sonucun standart sapmasına göre üç anlamlı basamak (en az 0; üst sınır ``MAX_DECIMALS``,
    ``validate`` denetler)."""

    spread = float(values.astype(float).std())
    if not math.isfinite(spread) or spread <= 0:
        return 2
    return int(max(0, 2 - math.floor(math.log10(spread))))


def _terms(case: Case) -> tuple[tuple[float, ...], tuple[float, ...], tuple[tuple[str, E.Expr, str], ...]]:
    """Düğümler (çeyrekler), karşılaştırma noktaları (10., 50. ve 90. yüzdelikler) ve kübik/spline terimleri."""

    x = case.roles[ACIKLAYICI]
    values = case.data[x].astype(float)
    knots = tuple(dict.fromkeys(_quantiles(values, (0.25, 0.5, 0.75))))
    points = tuple(dict.fromkeys(_quantiles(values, (0.10, 0.5, 0.90))))
    center = _quantiles(values, (0.5,))[0]
    scale = float(10 ** math.floor(math.log10(float(values.std()))))
    return knots, points, _basis(x, center, scale, knots)


def _plot_points(values: pd.Series) -> np.ndarray:
    data = values.astype(float)
    return np.linspace(float(data.min()), float(data.max()), CURVE_POINTS)


def _gap_message(case: Case) -> str:
    x = case.roles[ACIKLAYICI]
    values = np.unique(case.data[x].astype(float))
    i = int(np.argmax(np.diff(values)))
    digits = _digits(case.data[x].astype(float))
    left, right = (_value(round(float(v), digits)) for v in (values[i], values[i + 1]))
    return (f"“{case.name(x)}” sütununda değerler arasında çok büyük boşluklar var (en büyüğü {left} ile {right} "
            "arasında): bant genişliklerinin çoğunda yerel doğrusal tahmin seyrek değerlerin çevresinde sayısal olarak "
            "hesaplanamıyor (yakında ağırlığı anlamlı tek bir farklı değer kalıyor). Uç değerleri çıkarın ya da "
            "değişkeni dönüştürün (ör. logaritmasını alın).")


@dataclass(frozen=True)
class Bandwidths:
    """Kendi verinde CV ızgarası ve seçilen bant genişlikleri."""

    grid: tuple[float, float, float]
    decimals: int
    raised: bool
    loo: float
    clustered: float | None


def bandwidths(case: Case) -> Bandwidths:
    """CV ızgarası ve CV'nin seçtiği h'ler.

    Aday ızgara değişkenin aralığının 1/50'sinden 1/4'üne uzanır (``_grid``). Alt ucu, ızgaranın o noktadan sonraki
    bütün h'lerinde dışarıda bırakılan bütün tahminlerin ve grafik eğrisinin sayısal olarak güvenilir olduğu
    (ρ ≥ ``S.RELIABLE``) en küçük h'ye yükseltilir: daha küçük h'lerde seyrek değerlerin çevresinde formül 0/0 ya da
    yuvarlama gürültüsü üretir, diller farklı sayılar verir (R grafiği çizemez). CV hesapları önbelleklidir; laboratuvar
    çalışırken yeniden yapılmaz."""

    data = case.data
    x, y = case.roles[ACIKLAYICI], case.roles[SONUC]
    cluster = case.roles.get(KUME)
    groups = data[cluster] if cluster else None
    (low, high, step), decimals = _grid(data[x])
    values = S.bandwidth_grid(low, high, step)
    conditioning = S.cv_conditioning(data[x], data[y], values, groups).min(axis=1).to_numpy()
    curve = S.local_conditioning(data[x], _plot_points(data[x]), values)
    unreliable = np.flatnonzero((conditioning < S.RELIABLE) | (curve < S.RELIABLE))
    start = int(unreliable[-1]) + 1 if len(unreliable) else 0
    if len(values) - start < MIN_GRID:
        raise K.UploadError(_gap_message(case))
    grid = (round(float(values[start]), decimals), high, step)
    table = S.cv_curve(data[x], data[y], S.bandwidth_grid(*grid), groups)
    loo = float(table["cv"].idxmin())
    clustered = float(table["cv_kume"].idxmin()) if cluster else None
    return Bandwidths(grid, decimals, start > 0, loo, clustered)


def own_plan(case: Case) -> Plan:
    """Kendi verinin planı. Küme varsa ana bant küme-silmeli CV'nin, karşılaştırma bandı birini dışarıda bırakan CV'nin
    seçtiği h'dir; küme yoksa ya da iki ölçüt aynı h'yi seçerse karşılaştırma bandı ana bandın iki katıdır."""

    name = case.name
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    cluster = case.roles.get(KUME)
    data = case.data
    chosen = bandwidths(case)
    h_main = chosen.clustered if cluster else chosen.loo
    doubled = not cluster or chosen.loo == chosen.clustered
    h_other = round(2 * h_main, chosen.decimals) if doubled else chosen.loo
    knots, points, basis = _terms(case)
    values = data[x].astype(float)
    x_name = name(x)
    labels = {column: name(column) for column in (y, x) + ((cluster,) if cluster else ())}
    labels.update({E.INTERCEPT: "Sabit", **{term: comment for term, _, comment in basis}})
    template = x_name.replace("{", "{{").replace("}", "}}")  # etiket kalıbı str.format ile doldurulur
    return Plan(
        frame=case.frame, y=y, x=x, cluster=cluster, prepare=case.load, grid=chosen.grid, h_main=h_main,
        h_other=h_other, knots=knots, points=points, basis=basis, labels=labels, point=f"“{template}” = {{}}",
        x_label=x_name, y_label=name(y), titles=_titles("Kendi veriniz", f"“{x_name}” ve “{name(y)}”"),
        plot_grid=(float(values.min()), float(values.max()), 201), decimals=_decimals(data[y]),
        h_decimals=chosen.decimals, doubled=doubled, raised=chosen.raised,
    )


def _own_texts(case: Case, plan: Plan) -> dict:
    y, x = case.md(SONUC), case.md(ACIKLAYICI)
    cluster = case.md(KUME) if plan.cluster else None
    h_main, h_other = _value(plan.h_main), _value(plan.h_other)
    low, high, step = plan.grid
    d = plan.decimals
    grid_text = (f"Ölçüt $h\\in\\{{{_math(low)};\\,\\ldots;\\,{_math(high)}\\}}$ ızgarasında, {_value(step)} adımla "
                 "değerlendirilir.")
    if plan.raised:
        grid_text += (f" Izgaranın alt ucu h = {_value(low)}: daha küçük h'lerde bazı gözlemlerin çevresinde ağırlığı "
                      "anlamlı tek bir farklı değer kalıyor (ör. uçtaki seyrek değerler) ve yerel doğrusal tahmin "
                      "sayısal olarak hesaplanamıyor. Uç değerleri çıkarmak ya da değişkeni dönüştürmek (ör. logaritma) "
                      "daha küçük h'leri mümkün kılar.")
    if plan.cluster:
        main_source, other_source = "küme-silmeli CV ile", (
            "ana bandın iki katı olan" if plan.doubled else "birini dışarıda bırakan CV'nin seçtiği")
    else:
        main_source, other_source = "birini dışarıda bırakan CV ile", "ana bandın iki katı olan"
    width = "daha geniş" if plan.h_other > plan.h_main else "daha dar"

    def step1(state) -> str:
        frame = state.frames[plan.frame]
        groups = f" ({sayim(frame[plan.cluster].nunique())} küme)" if plan.cluster else ""
        slope = float(state.models["dogrusal"].params[plan.x])
        return (f"Analiz örneklemi {sayim(len(frame))} gözlem{groups}. Doğrusal modelin eğimi {katsayi(slope)}: “{x}” bir "
                f"birim yüksekken “{y}” ortalamada bu kadar farklı. Grafikte aralık ortalamalarının doğrunun çevresinde "
                "sistematik biçimde (ör. uçlarda üstte, ortada altta) dağılıp dağılmadığına bakın.")

    def step2(state) -> str:
        s = state.scalars
        chosen = [s["cv_h"]] + ([s["cv_h_kume"]] if plan.cluster else [])
        text = f"Birini dışarıda bırakan CV h = {_value(s['cv_h'])} seçiyor"
        if plan.cluster:
            text += f", küme-silmeli CV h = {_value(s['cv_h_kume'])}"
        text += ". Ölçütün en küçük değer çevresinde ne kadar yassı olduğu da grafikten okunur."
        if plan.cluster and plan.doubled:
            text += (" İki ölçüt aynı h'yi seçtiği için Adım 3'teki karşılaştırma bandı ana bandın iki katıdır "
                     f"(h = {h_other}).")
        if plan.grid[0] in chosen and plan.raised:
            text += (" Seçilen h ızgaranın alt ucunda: ölçüt daha küçük bir h'yi tercih edebilirdi, ama daha küçük h'lerde "
                     "tahmin sayısal olarak hesaplanamıyor. Sonucu bu sınırla birlikte okuyun.")
        elif plan.grid[0] in chosen or plan.grid[1] in chosen:
            text += (" Seçilen h ızgaranın ucunda: ölçüt ızgara içinde en küçük değerine ulaşmıyor olabilir; sonucu bu "
                     "sınırla birlikte okuyun.")
        return text

    def step3(state) -> str:
        table, local = state.tables["parametrik"], state.tables["yerel"]
        main, other = _column(plan.h_main), _column(plan.h_other)
        columns = [(table, model) for model, _ in MODELS] + [(local, main), (local, other)]
        spreads = []
        for point in plan.points:
            values = [float(frame.loc[point, column]) for frame, column in columns]
            spreads.append((max(values) - min(values), point, min(values), max(values)))
        _, widest, smallest, largest = max(spreads)
        low_point, high_point = plan.points[0], plan.points[-1]
        parts = [f"{label}: {sayi(float(table.loc[low_point, model]), d)} ve "
                 f"{sayi(float(table.loc[high_point, model]), d)}" for model, label in MODELS]
        parts.append(f"yerel doğrusal (h = {h_main}): {sayi(float(local.loc[low_point, main]), d)} ve "
                     f"{sayi(float(local.loc[high_point, main]), d)}")
        return (f"“{x}” = {_value(low_point)} ve {_value(high_point)} noktalarında tahminler — " + "; ".join(parts)
                + f". Karşılaştırılan üç noktada beş tahmin en çok “{x}” = {_value(widest)} noktasında ayrışır "
                  f"({sayi(smallest, d)}–{sayi(largest, d)}).")

    answers = ("Bu laboratuvarın cevapları: koşullu ortalama; Gauss çekirdeği, yerel doğrusal; bant genişliği CV ile "
               + ("(küme-silmeli ve birini dışarıda bırakan); kümeli yapı hem CV'de hem standart hatada korunur."
                  if plan.cluster else "(birini dışarıda bırakan); küme seçilmedi, standart hatalar HC1."))

    def step5(state) -> str:
        frame = state.frames[plan.frame][[plan.x, plan.y]].dropna()
        xs, ys = frame[plan.x].to_numpy(dtype=float), frame[plan.y].to_numpy(dtype=float)
        grid = np.linspace(xs.min(), xs.max(), CURVE_POINTS)
        gap = np.abs(S.local_fit(xs, ys, grid, plan.h_main) - S.local_fit(xs, ys, grid, plan.h_other))
        where = float(grid[int(np.argmax(gap))])
        local = state.tables["yerel"]
        at_points = float((local[_column(plan.h_main)] - local[_column(plan.h_other)]).abs().max())
        region = ("10. ve 90. yüzdeliklerin dışında, gözlemlerin daha seyrek olduğu bölgede"
                  if where < plan.points[0] or where > plan.points[-1] else "10. ile 90. yüzdelikler arasında")
        spread = float(xs.max() - xs.min())  # konum yaklaşık: aralığın yaklaşık yüzde biri duyarlılığında
        place = _value(round(where, max(0, 1 - math.floor(math.log10(spread))) if spread > 0 else 2))
        return (answers + f" Bu veride h = {h_main} ve h = {h_other} eğrileri “{x}” değişkeninin gözlenen aralığında en "
                f"çok {sayi(float(gap.max()), d)} farklıdır (“{x}” ≈ {place}; {region}); karşılaştırılan üç noktada "
                f"en çok {sayi(at_points, d)}.")

    if plan.cluster:
        cv_text = (CV_RULE + "**Birini dışarıda bırakan** CV'de $\\tilde m_{-i}$ yalnız $i$. gözlem çıkarılarak, "
                   f"**küme-silmeli** CV'de $i$'nin kümesindeki (“{cluster}”) bütün gözlemler çıkarılarak hesaplanır. "
                   + grid_text + " Adım 3'te ana bant küme-silmeli CV'nin, karşılaştırma bandı birini dışarıda bırakan "
                   "CV'nin seçtiği h'dir; iki ölçüt aynı h'yi seçerse karşılaştırma bandı ana bandın iki katıdır.")
        cluster_text = (f"Gözlemler “{cluster}” kümeleri içinde bağımlı olabilir. Birini dışarıda bırakan CV tek bir "
                        "gözlemi çıkarırken aynı kümeden çok benzer gözlemleri tahminde tutar; bu nedenle tahmin "
                        "performansını gereğinden iyimser değerlendirebilir. Küme-silmeli yaklaşım bir kümenin bütün "
                        "gözlemlerini birlikte dışarıda bırakır (Adım 2).\n\n" + CLUSTER_RULE)
    else:
        cv_text = (CV_RULE + "**Birini dışarıda bırakan** CV'de $\\tilde m_{-i}$ yalnız $i$. gözlem çıkarılarak "
                   "hesaplanır. Küme seçilmediği için küme-silmeli CV hesaplanmaz; Adım 3'te CV'nin seçtiği h, iki "
                   "katıyla karşılaştırılır. " + grid_text)
        cluster_text = ("Küme seçilmedi: gözlemlerin birbirinden bağımsız olduğu varsayılıyor. Veri kümeliyse (ör. okul, "
                        "firma, bölge) küme sütununu seçin; CV ve standart hatalar kümeyi korur.\n\n" + CLUSTER_RULE)
    return {
        1: (
            f"Soru: “{y}” ile “{x}” arasındaki ilişki doğrusal mı? Parametrik doğrusal model "
            "$\\mathbb E[Y\\mid X=x]=\\beta_0+\\beta_1x$ tek bir eğim dayatır. Grafikte tekil noktalar yerine $x$'in yirmi "
            "eşit genişlikli aralığındaki ortalamalar gösterilir; doğru bütün gözlemlerle tahmin edilmiştir."
            + (f" Standart hatalar “{cluster}” düzeyinde kümelenmiştir." if plan.cluster else
               " Standart hatalar HC1'dir (küme seçilmedi)."),
            "Yerel doğrusal tahmin her $x$ noktasında yakın gözlemlere daha çok ağırlık vererek koşullu ortalama eğrisini "
            "tahmin eder.",
        ),
        (1, "not"): step1,
        2: (cv_text, "Yazılımın seçtiği tek sayı \"doğru bant genişliği\" değildir; ölçütün en küçük değer çevresindeki "
                     "şekli de raporlanmalıdır."),
        (2, "not"): step2,
        3: (
            "Aynı koşullu ortalamayı beş biçimde tahmin ediyoruz: doğrusal model; kübik polinom; düğümleri “" + x
            + "” değişkeninin çeyreklerinde (" + "; ".join(_value(k) for k in plan.knots) + ") olan kübik spline; iki bant "
            f"genişliğinde yerel doğrusal tahmin (ana bant h = {h_main}, karşılaştırma bandı h = {h_other}). Polinom ve "
            "spline sıradan OLS ile tahmin edilir; kare ve küp terimler ortalanmış ve ölçeklenmiş değişkenden kurulur "
            "(terim adlarında x = “" + x + "”; tahminler değişmez). Tahminler “" + x + "” değişkeninin 10., 50. ve 90. "
            "yüzdeliklerinde ("
            + "; ".join(_value(p) for p in plan.points) + ") karşılaştırılır.",
            "Farklılıkların hiçbiri tek başına \"doğru model\" kanıtı değildir: veri desteği, tuning ölçütü ve makul "
            "alternatiflerde sonucun değişip değişmediği birlikte değerlendirilir.",
        ),
        (3, "not"): step3,
        4: (cluster_text, f"Ana bant genişliği {main_source} seçilir (h = {h_main}); duyarlılık {other_source} {width} "
                          f"bir bantla (h = {h_other}) gösterilir."),
        5: ("", answers),
        (5, "not"): step5,
        "kod2": CODE_NOTE_2.replace(", Stata'da Mata ile yazılmıştır", " ile yazılmıştır").replace(
            "Python'da NumPy, R'de", "Python'da NumPy ve R'de"),
        "kod3": CODE_NOTE_3 if plan.cluster else CODE_NOTE_3.replace(
            "Standart hatalar küme düzeyinde kümelenmiştir", "Standart hatalar HC1'dir"),
    }


def build(case: Case) -> LabSpec:
    """Kendi verinle Konu 8 uygulaması; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    plan = own_plan(case)
    spec = _spec(plan, _own_texts(case, plan), source="kendi",
                 title=f"{case.label(SONUC)} ve {case.label(ACIKLAYICI)}", dataset="")
    return with_app_values(spec)


def _stable_fits(frame: pd.DataFrame, y: str, x: str, basis, points: tuple[float, ...]) -> bool:
    """Doğrusal, kübik ve spline tahminleri seçilmiş noktalarda yazılımlar arasında aynı hesaplanabilir mi: tasarımın
    koşul sayısı ``SPLINE_CONDITION`` altında ve uygulamanın tahmini (statsmodels) sütunları ortalanıp ölçeklenmiş
    tasarımdaki tahminle ``PREDICTION_LIMIT`` içinde aynı."""

    import statsmodels.api as sm

    names = [name for name, _, _ in basis]
    rows = pd.DataFrame({x: np.asarray(points, dtype=float)})
    for name, expression, _ in basis:
        rows[name] = np.asarray(E.evaluate(expression, rows), dtype=float) * np.ones(len(rows))
    target = frame[y].to_numpy(dtype=float)
    spread = float(np.std(target))
    for columns in ([x], [x, *names[:2]], [x, *names]):
        values = frame[columns].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(values)), values])
        if not np.isfinite(design).all():
            return False
        norms = np.linalg.norm(design, axis=0)
        norms[norms == 0] = 1.0
        if float(np.linalg.cond(design / norms)) > SPLINE_CONDITION:
            return False
        at = rows[columns].to_numpy(dtype=float)
        app = np.column_stack([np.ones(len(at)), at]) @ np.asarray(sm.OLS(target, design).fit().params, dtype=float)
        means, scales = values.mean(axis=0), values.std(axis=0)
        scales[scales == 0] = 1.0
        standard = np.column_stack([np.ones(len(values)), (values - means) / scales])
        solution, *_ = np.linalg.lstsq(standard, target, rcond=None)
        reference = solution[0] + ((at - means) / scales) @ solution[1:]
        if np.any(np.abs(app - reference) > PREDICTION_LIMIT * np.maximum(spread, np.abs(reference))):
            return False
    return True


def _reliable_curves(case: Case, plan: Plan) -> bool:
    """İki bant genişliğinde yerel doğrusal tahmin grafik ızgarasında ve karşılaştırma noktalarında sayısal olarak
    güvenilir mi (ρ ≥ ``S.RELIABLE``)."""

    values = case.data[plan.x]
    points = np.concatenate([_plot_points(values), np.asarray(plan.points, dtype=float)])
    return bool(np.all(S.local_conditioning(values, points, sorted({plan.h_main, plan.h_other})) >= S.RELIABLE))


def validate(case: Case) -> None:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    cluster = case.roles.get(KUME)
    if y == x or cluster in (y, x):
        raise K.UploadError("Sonuç, açıklayıcı ve küme için farklı sütunlar seçin.")
    data = case.data
    if len(data) > MAX_ROWS:
        raise K.UploadError(f"Analizde {sayim(len(data))} gözlem var; bant genişliği seçimi n×n ağırlık matrisiyle "
                            f"hesaplandığı için en çok {sayim(MAX_ROWS)} gözlem kullanılabilir.")
    if data[x].nunique() < 20:
        raise K.UploadError(f"“{case.name(x)}” sütununda en az 20 farklı değer olmalı (yerel doğrusal tahmin ve "
                            "çeyreklerde düğümler için).")
    if data[y].nunique() < 2:
        raise K.UploadError(f"“{case.name(y)}” sütununda en az iki farklı değer olmalı.")
    if cluster:
        groups = data[cluster].nunique()
        if groups < MIN_CLUSTERS:
            raise K.UploadError(f"“{case.name(cluster)}” sütununda {groups} küme var; küme-silmeli CV için en az "
                                f"{MIN_CLUSTERS} küme gerekir.")
        if groups == len(data):
            raise K.UploadError(f"“{case.name(cluster)}” sütununda her gözlem ayrı bir küme: küme-silmeli CV birini "
                                "dışarıda bırakan CV ile aynı olur. Küme seçmeyin ya da doğru küme sütununu seçin.")
    if _decimals(data[y]) > MAX_DECIMALS:
        raise K.UploadError(f"“{case.name(y)}” sütununun ölçeği çok küçük: tahminler {MAX_DECIMALS} ondalıkta bile "
                            "ayırt edilemiyor. Dosyada birimini değiştirin (ör. ton yerine kilogram).")
    if exact_fit(data, y, (x,)):
        raise K.UploadError(f"“{case.name(y)}”, “{case.name(x)}” değişkeninin (neredeyse) tam doğrusal bir fonksiyonu "
                            "(R² ≈ 1): yerel doğrusal tahmin doğruyu tam olarak yeniden üretir, çapraz doğrulama ölçütü "
                            "yalnız yuvarlama hatasından oluşur ve bant genişliği seçilemez.")
    knots, points, basis = _terms(case)
    if len(knots) < 3 or len(points) < 3:
        raise K.UploadError(f"“{case.name(x)}” sütununun çeyrekleri ya da 10./50./90. yüzdelikleri ayrışmıyor "
                            "(değerlerin çoğu birkaç değerde toplanmış): spline ve karşılaştırma noktaları kurulamıyor.")
    frame = data.copy()
    for name, expression, _ in basis:
        frame[name] = np.asarray(E.evaluate(expression, frame), dtype=float)
    if not full_rank(frame, (x, *(name for name, _, _ in basis))):
        raise K.UploadError("Spline terimleri arasında tam doğrusal bağlantı var: açıklayıcının değerleri çok az.")
    if not _stable_fits(frame, y, x, basis, points):
        raise K.UploadError(f"“{case.name(x)}” ile kurulan kübik polinom ve spline terimleri neredeyse doğrusal "
                            "bağlantılı (ör. değerlerin çoğu dar bir aralıkta, birkaç değer çok uzakta): tahminler "
                            "yazılımlar arasında aynı hassasiyetle hesaplanamıyor. Değişkeni dönüştürün (ör. logaritmasını "
                            "alın) ya da uç değerleri çıkarın.")
    plan = own_plan(case)  # CV (önbellekli): güvenilir ızgara ve bant genişlikleri; ızgara kalmazsa UploadError
    if not _reliable_curves(case, plan):
        raise K.UploadError(_gap_message(case))


def suggest(table: K.UploadedTable) -> dict[str, str]:
    """Küme rolü için adında okul, küme, firma, sınıf, il, köy ya da kurum sözcüğü geçen ve 10 ile gözlem sayısının
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
    """Örnek dosya: kurgusal okul–öğrenci verisi (``core.labs.ornek_veri``)."""

    return calisma_verisi()


ROLES = (
    Role(SONUC, "Sonuç değişkeni", "sayisal", True, (1, 2, 3),
         "Koşullu ortalaması tahmin edilen sayısal değişken (notlarda toplam test puanı)."),
    Role(ACIKLAYICI, "Sürekli açıklayıcı değişken", "sayisal", True, (1, 2, 3),
         "Eğrinin çizildiği değişken (notlarda başlangıç yüzdeliği); en az 20 farklı değer."),
    Role(KUME, "Küme (ör. okul)", "serbest", False, (1, 2, 3, 4),
         "Gözlemlerin bağımlı olduğu birim (notlarda okul). Seçilirse küme-silmeli CV de hesaplanır ve standart "
         "hatalar kümelenir; seçilmezse yalnız birini dışarıda bırakan CV ve HC1 kullanılır. Boş hücre olamaz.",
         complete=True),
)

CUSTOM = CustomLab(
    roles=ROLES,
    build=build,
    sample=sample,
    intro=(
        "Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç ve sürekli bir açıklayıcı seçilir; gözlemler kümeler (ör. "
        "okul) içinde bağımlıysa küme sütunu da seçilir. Seçilen sütunlarda boş hücresi olan satırlar analizden "
        f"çıkarılır. Bant genişliği seçimi n×n ağırlık matrisiyle hesaplandığı için en çok {sayim(MAX_ROWS)} gözlem "
        "kullanılır."
    ),
    min_rows=60,
    validate=validate,
    suggest=suggest,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
