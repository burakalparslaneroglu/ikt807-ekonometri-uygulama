"""Konu 1 genel uygulaması: koşullu ortalamadan doğrusal projeksiyona ve çok sütunlu regresyon tablosuna.

Notlardaki §1.13'ün dokuz adımı aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde yazılır
(``build``). Alternatif örnek Card (1995) verisidir: Hansen'in arşivindeki ``Card1995.dta`` (NLSYM; 1976'da 24–34
yaşındaki genç erkekler). "Kendi verini yükle" seçeneğinde aynı adımlar öğrencinin dosyasıyla kurulur. Notlardaki
laboratuvar (``core.labs.konu01``) değişmez.

Rollerin koddaki karşılığı ``Plan``'dadır: sonucun düzeyi ve regresyondaki biçimi (notlardaki gibi log), koşullu
ortalaması alınan açıklayıcı değişken, karesiyle eklenen sürekli kontrol ve 0/1 grup göstergesi. Notlardaki karşılıkları
sırasıyla saatlik ücret ve log ücret, eğitim yılı, potansiyel deneyim ve kadın göstergesidir.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from core.labs import expr as E
from core.labs import kendi_veri as K
from core.labs.ornek import (
    EXACT_FIT_NOTE,
    SCALE_MESSAGE,
    Case,
    CustomLab,
    Option,
    Role,
    TopicVariants,
    exact_fit,
    full_rank,
    katsayi,
    md,
    sayi,
    sayim,
    stable_checks,
    stable_design,
    with_app_values,
    with_expected,
    yuzde,
)
from core.labs.ornek_veri import ucret_verisi
from core.labs.spec import (
    OLS,
    Check,
    CoefTarget,
    Derive,
    Describe,
    DropMissing,
    GroupMeanPlot,
    GroupSummary,
    Indicator,
    LabSpec,
    LabStep,
    LoadHansen,
    ModelTarget,
    NoteRef,
    Operation,
    ProjectionPlot,
    RegressionTable,
    Scalar,
    ScalarTarget,
    ShowModel,
    StatTarget,
)

TOPIC = "konu01"
SECTION = "1.13"
TITLE = "CPS Ücret–Eğitim İlişkisi"
SONUC, ACIKLAYICI, KONTROL, GRUP = "sonuc", "aciklayici", "kontrol", "grup"
LOG = "log"
LOG_Y, Z2, G01 = "log_sonuc", "kontrol_kare_100", "grup01"
"""Kendi verinde türetilen sütunlar (``kendi_veri.RESERVED_CODES``: öğrencinin sütunlarına verilmez)."""
MAX_LEVELS = 30
"""Koşullu ortalama tablosu için açıklayıcı değişkenin en çok farklı değer sayısı."""
SKEW_LIMIT = 0.5
"""Kendi verinin yorumunda "sağa çarpık" denmesi için örneklem çarpıklık katsayısının alt sınırı."""
INTERCEPT = E.INTERCEPT


@dataclass(frozen=True)
class Plan:
    """Rollerin koddaki sütunları, kontrol etiketlerindeki adları ve ekrandaki etiketler.

    ``words``: kontrol etiketlerinde ve metinlerde kullanılan adlar (cümle içinde), ``titles``: cümle başındaki
    biçimleri. Anahtarlar: ``x``, ``z``, ``z2``, ``group``, ``outcome``, ``level``. ``points``: Adım 4'te gözlem sayısı ve
    ortalaması kontrol edilen açıklayıcı değişken değerleri; üç değer verilirse ortadakinde düzey ortalaması ve medyanı da
    kontrol edilir (notlarda eğitim = 12, 16, 20).
    """

    frame: str
    level: str
    outcome: str
    log: bool
    x: str
    z: str
    z2: str
    group: str
    group_names: tuple[str, str]
    group_checks: tuple[str, str]
    describe: tuple[str, ...]
    means: tuple[tuple[str, str], ...]
    points: tuple[float, ...]
    prepare: tuple[Operation, ...]
    words: dict
    titles: dict
    labels: dict
    plot_title: str
    projection_title: str
    y_axis: str
    table_title: str
    percent: tuple[tuple[str, str], tuple[str, str]]


def _value(value: float) -> str:
    """Açıklayıcı değişkenin değeri, kontrol etiketinde (12; 12,5)."""

    return E.format_number(value).replace(".", ",")


# --- Adımlar -----------------------------------------------------------------------------------------------

def _step1(plan: Plan, texts: dict) -> LabStep:
    return LabStep(number=1, title="Araştırma sorusunu ve tahmin hedefini yazmak", note=NoteRef(SECTION, 1),
                   explanation=texts[1][0], takeaway=texts[1][1])


def _step2(plan: Plan, texts: dict, code_note: str) -> LabStep:
    return LabStep(
        number=2,
        title="Ham veriden analiz örneklemine",
        note=NoteRef(SECTION, 2),
        explanation=texts[2][0],
        operations=plan.prepare,
        checks=(Check("Analiz örneklemi (N)", StatTarget(plan.frame, plan.outcome, "count"), 0.0, decimals=0),),
        takeaway=texts[2][1],
        code_note=code_note,
        note_for=texts.get((2, "not")),
    )


def _stat(plan: Plan, variable: str, stat: str, label: str, *, where=None, decimals: int = 4) -> Check:
    return Check(label, StatTarget(plan.frame, variable, stat, where), 0.0, decimals)


def _step3(plan: Plan, texts: dict) -> LabStep:
    w = plan.titles
    checks = [
        _stat(plan, plan.level, "mean", f"{w['level']} ortalaması"),
        _stat(plan, plan.level, "sd", f"{w['level']} std. sapması"),
        _stat(plan, plan.level, "median", f"{w['level']} medyanı"),
    ]
    if plan.log:
        checks += [
            _stat(plan, plan.outcome, "mean", f"{w['outcome']} ortalaması"),
            _stat(plan, plan.outcome, "sd", f"{w['outcome']} std. sapması"),
            _stat(plan, plan.outcome, "min", f"{w['outcome']} en küçük değeri"),
        ]
    checks += [
        _stat(plan, plan.x, "mean", f"{w['x']} ortalaması"),
        _stat(plan, plan.z, "mean", f"{w['z']} ortalaması"),
        _stat(plan, plan.group, "mean", f"{plan.group_names[1]} payı"),
        *(_stat(plan, column, "mean", label) for column, label in plan.means),
    ]
    return LabStep(
        number=3,
        title="Regresyondan önce veriyi tanımak",
        note=NoteRef(SECTION, 3),
        explanation=texts[3][0],
        operations=(Describe(plan.frame, plan.describe, "betimsel"),),
        checks=tuple(checks),
        takeaway=texts[3][1],
        note_for=texts.get((3, "not")),
    )


def _step4(plan: Plan, texts: dict) -> LabStep:
    w = plan.words
    columns = [("N", plan.outcome, "count"), ("ort_sonuc", plan.outcome, "mean")]
    if plan.log:
        columns.append(("ort_duzey", plan.level, "mean"))
    columns.append(("medyan_duzey", plan.level, "median"))
    x_name = plan.titles["x"]
    first, *middle, last = plan.points
    checks = [
        _stat(plan, plan.outcome, "mean", plan.group_checks[0], where=(plan.group, 0)),
        _stat(plan, plan.outcome, "mean", plan.group_checks[1], where=(plan.group, 1)),
        _stat(plan, plan.outcome, "count", f"{x_name}={_value(first)} gözlem sayısı", where=(plan.x, first), decimals=0),
        _stat(plan, plan.outcome, "mean", f"{x_name}={_value(first)} ortalama {w['outcome']}", where=(plan.x, first)),
    ]
    for value in middle:
        checks.append(_stat(plan, plan.outcome, "mean", f"{x_name}={_value(value)} ortalama {w['outcome']}",
                            where=(plan.x, value)))
        if plan.log:
            checks.append(_stat(plan, plan.level, "mean", f"{x_name}={_value(value)} ortalama {w['level']}",
                                where=(plan.x, value)))
        checks.append(_stat(plan, plan.level, "median", f"{x_name}={_value(value)} medyan {w['level']}",
                            where=(plan.x, value)))
    checks.append(_stat(plan, plan.outcome, "mean", f"{x_name}={_value(last)} ortalama {w['outcome']}",
                        where=(plan.x, last)))
    return LabStep(
        number=4,
        title="Koşullu ortalamaları önce grafik ve tabloyla görmek",
        note=NoteRef(SECTION, 4),
        explanation=texts[4][0],
        operations=(
            GroupSummary(plan.frame, plan.x, tuple(columns), "kosullu_ortalama"),
            GroupMeanPlot(plan.frame, plan.x, plan.outcome, plan.group,
                          ((0, plan.group_names[0]), (1, plan.group_names[1])),
                          plan.labels[plan.x], plan.y_axis, plan.plot_title),
        ),
        checks=tuple(checks),
        takeaway=texts[4][1],
        note_for=texts.get((4, "not")),
    )


def _coef(model: str, term: str, label: str, quantity: str = "coef") -> Check:
    return Check(label, CoefTarget(model, term, quantity), 0.0)


def _step5(plan: Plan, texts: dict) -> LabStep:
    return LabStep(
        number=5,
        title="Koşullu ortalamadan doğrusal projeksiyona",
        note=NoteRef(SECTION, 5),
        explanation=texts[5][0],
        operations=(
            OLS("m1", plan.frame, plan.outcome, (plan.x,)),
            ProjectionPlot(plan.frame, plan.x, plan.outcome, "m1", plan.labels[plan.x], plan.y_axis,
                           plan.projection_title, relative_size=True),
        ),
        checks=(
            _coef("m1", plan.x, f"Model (1) {plan.words['x']} katsayısı"),
            _coef("m1", INTERCEPT, "Model (1) sabit terim"),
        ),
        takeaway=texts[5][1],
        note_for=texts.get((5, "not")),
    )


def _step6(plan: Plan, texts: dict) -> LabStep:
    w = plan.words
    m2 = (plan.x, plan.z, plan.z2)
    m3 = (*m2, plan.group)
    operations: list[Operation] = [
        OLS("m2", plan.frame, plan.outcome, m2),
        OLS("m3", plan.frame, plan.outcome, m3),
        RegressionTable(("m1", "m2", "m3"), (*m3, INTERCEPT), "regresyon_tablosu", title=plan.table_title),
    ]
    checks = [
        _coef("m2", plan.x, f"Model (2) {w['x']}"),
        _coef("m2", plan.z, f"Model (2) {w['z']}"),
        _coef("m2", plan.z2, f"Model (2) {w['z2']}"),
        _coef("m3", plan.x, f"Model (3) {w['x']}"),
        _coef("m3", plan.z, f"Model (3) {w['z']}"),
        _coef("m3", plan.z2, f"Model (3) {w['z2']}"),
        _coef("m3", plan.group, f"Model (3) {w['group']}"),
        _coef("m3", INTERCEPT, "Model (3) sabit terim"),
        Check("Model (1) R²", ModelTarget("m1", "r2"), 0.0),
        Check("Model (2) R²", ModelTarget("m2", "r2"), 0.0),
        Check("Model (3) R²", ModelTarget("m3", "r2"), 0.0),
    ]
    if plan.log:
        (x_name, x_comment), (g_name, g_comment) = plan.percent
        operations += [
            Scalar(x_name, E.mul(100, E.sub(E.exp(E.coef("m3", plan.x)), 1)), x_comment),
            Scalar(g_name, E.mul(100, E.sub(E.exp(E.coef("m3", plan.group)), 1)), g_comment),
        ]
        checks += [
            Check(f"{plan.titles['x']}: kesin yüzde etki", ScalarTarget(x_name), 0.0, decimals=2),
            Check(f"{plan.titles['group']}: kesin yüzde fark", ScalarTarget(g_name), 0.0, decimals=2),
        ]
    return LabStep(
        number=6,
        title="Bir makaledeki çok sütunlu regresyon tablosunu okumak",
        note=NoteRef(SECTION, 6),
        explanation=texts[6][0],
        operations=tuple(operations),
        checks=tuple(checks),
        takeaway=texts[6][1],
        note_for=texts.get((6, "not")),
    )


def _step7(plan: Plan, texts: dict, exact: bool) -> LabStep:
    t = plan.titles
    checks = (
        _coef("m3", plan.x, f"{t['x']} standart hatası", "se"),
        _coef("m3", plan.z, f"{t['z']} standart hatası", "se"),
        _coef("m3", plan.z2, f"{t['z2']} standart hatası", "se"),
        _coef("m3", plan.group, f"{t['group']} standart hatası", "se"),
        _coef("m3", INTERCEPT, "Sabit terim standart hatası", "se"),
    )
    return LabStep(
        number=7,
        title="Yazılım çıktısından makale tablosuna geçmek",
        note=NoteRef(SECTION, 7),
        explanation=texts[7][0],
        operations=(ShowModel("m3"),),
        checks=stable_checks(checks, exact),
        takeaway=texts[7][1],
        note_for=texts.get((7, "not")),
    )


def _step8(texts: dict) -> LabStep:
    return LabStep(number=8, title="Sonuçları ampirik metne dönüştürmek", note=NoteRef(SECTION, 8),
                   explanation=texts[8][0], takeaway=texts[8][1], note_for=texts.get((8, "not")))


def _step9(texts: dict) -> LabStep:
    return LabStep(number=9, title="Aynı analizi kendiniz nasıl kurarsınız?", note=NoteRef(SECTION, 9),
                   explanation=texts[9][0], takeaway=texts[9][1])


def _spec(plan: Plan, texts: dict, *, source: str, title: str, dataset: str, code_note: str,
          exact: bool = False) -> LabSpec:
    return LabSpec(
        topic_key=TOPIC,
        title=title,
        dataset=dataset,
        note_section=SECTION,
        steps=(
            _step1(plan, texts), _step2(plan, texts, code_note), _step3(plan, texts), _step4(plan, texts),
            _step5(plan, texts), _step6(plan, texts), _step7(plan, texts, exact), _step8(texts), _step9(texts),
        ),
        labels=tuple(plan.labels.items()),
        source=source,
    )


# --- Ortak metinler ---------------------------------------------------------------------------------------

READING = (
    "Regresyon tablosunun amacı sayıları tekrar etmek değil, araştırma sorusuna kontrollü bir cevap vermektir. İyi "
    "bir paragraf dört şey yapar: veri ve örneklemi belirtir, katsayının spesifikasyonlar arasında nasıl değiştiğini "
    "söyler, büyüklüğü ekonomik birime çevirir ve modelin izin verdiği yorum sınırını korur."
)
PROTOCOL = (
    "Tez için başlangıç protokolü: araştırma sorusunu yazın; tahmin hedefini belirleyin; ham veriyi koruyun ve "
    "temizliği kodla yapın; analiz örneklemini ve türetilmiş değişkenleri belgeleyin; önce betimleyin, sonra tahmin "
    "edin; sonuçları yorum sınırıyla birlikte yazın.\n\n"
    "Aşağıdan bütün laboratuvarı seçtiğiniz dilde tek dosya olarak indirebilirsiniz. Betik sonunda uygulamanın aynı "
    "veriyle verdiği sayılarla karşılaştırma yapar; bir sayı tutmazsa hangisinin tutmadığını söyleyerek durur."
)
CLASSICAL_SE = (
    "Bu bölümdeki standart hatalar klasik OLS standart hatalarıdır ve yalnız tablo okuma pratiği içindir. Güvenilir "
    "çıkarım ve heteroskedastisiteye dayanıklı standart hatalar Konu 2'de ele alınır."
)
PROJECTION = (
    "OLS doğrusu bütün koşullu ortalamalardan geçmek zorunda değildir. Koşullu ortalama doğrusal değilse OLS, onun "
    "en iyi doğrusal yaklaşımını (projeksiyonunu) verir."
)
HANSEN_NOTE = (
    "Kod veriyi Hansen'in sayfasındaki `Econometrics Data.zip` arşivinden indirir. İnternet yoksa dosyayı kendiniz "
    "indirip betiğin başındaki yerel dosya satırına yolunu yazın."
)
OWN_NOTE = (
    "Kod veri dosyanızı okur ve uygulamanın yaptığı temizliği aynı kurallarla yapar. Dosyayı betikle aynı klasöre "
    "koyun ya da betiğin başındaki dosya yolunu değiştirin. Kendi verinizle kod Python ve R'da üretilir."
)


# --- Alternatif örnek: Card (1995) ----------------------------------------------------------------------------

ALT_DATA = "card1995"
ALT_FRAME = "card"
ALT_TITLE = "Card (1995) Verisiyle Ücret–Eğitim İlişkisi"


def alternative_plan() -> Plan:
    frame = ALT_FRAME
    prepare = (
        LoadHansen(ALT_DATA, "Card1995.dta", frame),
        DropMissing(frame, ("lwage76",),
                    "Analiz örneklemi: 1976 log ücreti (lwage76) gözlenen erkekler; Card saatlik ücreti 1 doların "
                    "altındaki 7 gözlemde bu değişkeni eksik bırakır"),
        Derive(frame, "hrwage", E.div(E.var("wage76"), 100), "Saatlik ücret, dolar (wage76 sent cinsindendir)"),
        Derive(frame, "lwage", E.log(E.var("hrwage")), "Log saatlik ücret"),
        Derive(frame, "experience", E.sub(E.sub(E.var("age76"), E.var("ed76")), 6),
               "Potansiyel deneyim = yaş - eğitim - 6 (örneklemde negatif değer yoktur)"),
        Derive(frame, "experience2_100", E.div(E.power(E.var("experience"), 2), 100),
               "Deneyimin karesi / 100 (katsayıyı okunur ölçeğe getirir)"),
    )
    words = {"x": "eğitim", "z": "deneyim", "z2": "deneyim²/100", "group": "siyahi", "outcome": "log ücret",
             "level": "ücret"}
    titles = {"x": "Eğitim", "z": "Deneyim", "z2": "Deneyim²/100", "group": "Siyahi", "outcome": "Log ücret",
              "level": "Saatlik ücret"}
    labels = {
        "hrwage": "Saatlik ücret (dolar)",
        "lwage": "Log saatlik ücret",
        "ed76": "Eğitim yılı",
        "experience": "Potansiyel deneyim",
        "experience2_100": "Deneyim²/100",
        "black": "Siyahi göstergesi",
        "age76": "Yaş",
        INTERCEPT: "Sabit",
        "N": "N",
        "ort_sonuc": "Ortalama log ücret",
        "ort_duzey": "Ortalama ücret",
        "medyan_duzey": "Medyan ücret",
    }
    return Plan(
        frame=frame, level="hrwage", outcome="lwage", log=True, x="ed76", z="experience", z2="experience2_100",
        group="black", group_names=("Siyahi olmayan", "Siyahi"),
        group_checks=("Siyahi olmayanlarda ortalama log ücret", "Siyahilerde ortalama log ücret"),
        describe=("hrwage", "lwage", "ed76", "experience", "black", "age76"),
        means=(("age76", "Yaş ortalaması"),),
        points=(12.0, 16.0, 18.0),
        prepare=prepare, words=words, titles=titles, labels=labels,
        plot_title="Eğitim düzeyine ve ırka göre ortalama log saatlik ücret (NLSYM, 1976)",
        projection_title="Eğitim düzeyine göre koşullu ortalama ve OLS doğrusal projeksiyonu (NLSYM, 1976)",
        y_axis="Ortalama log saatlik ücret",
        table_title="Log saatlik ücret için OLS regresyonları, Card (1995) verisi",
        percent=(("egitim_yuzde", "Model (3) eğitim katsayısının kesin yüzde karşılığı"),
                 ("siyahi_yuzde", "Model (3) siyahi göstergesinin kesin yüzde karşılığı")),
    )


ALT_TEXTS = {
    1: (
        "Araştırma sorusu: *1976'da 24–34 yaşındaki genç erkeklerde eğitim düzeyi yükseldikçe saatlik ücretin koşullu "
        "ortalaması nasıl değişmektedir?*\n\n"
        "Veri Card (1995)'in kullandığı Ulusal Boylamsal Genç Erkekler Araştırması'dır (NLSYM): 1966'da 14–24 yaşında "
        "olan erkekler 1976'da yeniden görüşülmüştür. Saatlik ücret $wage_i$ dolar cinsinden, sonuç değişkeni "
        "$Y_i=\\log(wage_i)$ ve ilk tahmin hedefi koşullu beklenti fonksiyonu "
        "$m(x)=\\mathbb{E}[\\log(wage_i)\\mid ed76_i=x]$'tir; $ed76$ 1976'da tamamlanmış eğitim yılıdır.",
        "Araştırma sorusu \"eğitim katsayısı kaçtır?\" değildir. Önce hangi anakütle nesnesini öğrenmek istediğimizi "
        "yazarız; katsayı ondan sonra gelir. Notlardaki CPS örnekleminden farklı olarak burada yalnız erkekler ve dar "
        "bir yaş aralığı vardır; sonuçlar bu anakütleye aittir.",
    ),
    2: (
        "Hansen'in `Card1995` dosyasında 3.613 erkek vardır. 1976 saatlik ücreti (`wage76`, sent) 3.017 kişide "
        "gözlenir; Card'ın log ücret değişkeni (`lwage76`) saatlik ücreti 1 doların altında olan 7 gözlemde eksiktir. "
        "Analiz örneklemi `lwage76`'nın gözlendiği 3.010 erkektir. Dört değişken türetilir: dolar cinsinden saatlik "
        "ücret, log saatlik ücret, potansiyel deneyim ve deneyimin karesinin 100'e bölümü.",
        "Ücret sent cinsinden olduğu için önce dolara çevrilir. Logaritmada birim değişikliği yalnız sabit terimi "
        "kaydırır: $\\log(wage76/100)=\\log(wage76)-\\log 100$; eğim katsayıları değişmez. Örneklem kuralı (burada 1 "
        "doların altındaki ücretler) kodla belgelenmelidir.",
    ),
    3: (
        "Regresyondan önce temel değişkenlerin dağılımına bakarız. 1976 doları ile ortalama saatlik ücret 5,77, "
        "medyanı 5,375'tir: ücret dağılımı burada da sağa çarpıktır. Örneklem dar bir yaş aralığını (24–34) kapsadığı "
        "için ortalama potansiyel deneyim yalnız 8,9 yıldır.",
        "Siyahi göstergesinin ortalaması 0,2336'dır: bir kukla değişkenin ortalaması, o grubun örneklemdeki payıdır. Log "
        "ücretin en küçük değeri 0'dır: saatlik ücreti tam 1 dolar olan iki gözlem, yani örneklem kuralının alt sınırı.",
    ),
    4: (
        "Regresyondan önce grup örüntülerini görünür kılarız. Siyahi olmayan ve siyahi erkekler için ortalama log ücret "
        "1,7309 ve 1,4129'dur: yaklaşık 0,32 log puanı fark. Eğitim arttıkça ortalama log ücret genel olarak yükselir; "
        "12 yılın altındaki kategorilerde gözlem az ve örüntü düzensizdir. 12 yıl (lise) 992 gözlemle en kalabalık "
        "kategoridir.",
        "Bu sayılar grup ortalamalarıdır, nedensel eğitim etkileri değildir.",
    ),
    5: (
        "Yalnız eğitimi kullanan doğrusal projeksiyon $\\log(wage_i)=\\beta_0+\\beta_1 ed76_i+e_i$ tahmin edilir. Tahmin "
        "edilen denklem $\\widehat{\\log(wage_i)}=0{,}9657+0{,}0521\\,ed76_i$ olur. Nokta büyüklükleri o eğitim "
        "kategorisindeki gözlem sayısını gösterir.",
        PROJECTION + " OLS her gözleme eşit ağırlık verir; koşullu ortalamalar gözlem sayılarıyla ağırlıklanır. "
        "Ortalamadan uzak düzeylerin etkisi (kaldıracı) ise büyüktür: 8 yılın altındaki eğitim düzeyleri gözlemlerin "
        "yalnız %2,1'idir, ama eğitimin örneklemdeki değişkenliğinin, $\\sum_i (x_i-\\bar x)^2$'nin %16,9'unu "
        "taşır; bu gözlemler çıkarılırsa eğim 0,0521'den 0,0502'ye iner.",
    ),
    6: (
        "Aynı araştırma sorusu üç spesifikasyonla yan yana raporlanır: (1) yalnız eğitim, (2) deneyim profili "
        "eklenmiş, (3) siyahi göstergesi de eklenmiş. Tablo satır satır değil, sütunlar arasında katsayının nasıl "
        "değiştiğine bakılarak okunur.",
        "Eğitim katsayısı deneyim profili eklenince 0,0521'den 0,0932'ye yükselir. Örneklemdeki erkekler benzer "
        "yaştadır ve potansiyel deneyim yaş − eğitim − 6 olduğu için uzun eğitim daha az deneyim demektir "
        "(korelasyon −0,65). Deneyim modelde yokken eğitim katsayısı, daha az deneyimin ücretle negatif ilişkisini de "
        "taşır ve küçük çıkar. Siyahi göstergesi de eklenince katsayı 0,0818'e düşer. Model (3)'te eğitim katsayısı 0,0818: eğitim yılı bir birim daha yüksek "
        "gözlemlerin koşullu ortalama log ücreti 0,0818 daha yüksektir; bu, ücrette kesin dönüşümle %8,52 daha "
        "yüksek değere karşılık gelir. Siyahi göstergesinin katsayısı −0,2319: diğer değişkenler aynıyken siyahi "
        "erkeklerde bu, ücrette kesin dönüşümle %20,70 daha düşük değere karşılık gelir. Bunlar koşullu "
        "ilişkilerdir; nedensel getiri ya da ayrımcılık ölçüsü olarak yorumlanmamalıdır.",
    ),
    7: (
        "Makale tablosu ile yazılım çıktısı farklı nesneler değildir: tablo, çıktının araştırma sorusu için önemli "
        "kısmının okunur biçimidir. `ed76` satırında katsayı, standart hata, $t$ oranı (katsayı / standart hata), "
        "$p$-değeri ve %95 güven aralığı yer alır.",
        CLASSICAL_SE,
    ),
    8: (
        READING,
        "1976 NLSYM örnekleminde (Card, 1995) eğitim ile log saatlik ücret arasında pozitif bir ilişki bulunmaktadır. "
        "Yalnız eğitim değişkenini içeren spesifikasyonda eğitim katsayısı 0,0521'dir. Potansiyel deneyim ve "
        "deneyimin karesi eklendiğinde katsayı 0,0932'ye yükselmekte, siyahi göstergesi de kontrol edildiğinde "
        "0,0818'e düşmektedir. Bu ilişki, eğitim kararının içselliğini ele alan bir tanımlama stratejisi kurulmadığı "
        "için nedensel eğitim getirisi olarak yorumlanmamalıdır.",
    ),
    9: (PROTOCOL, ""),
}

ALT_EXPECTED = {
    (2, "Analiz örneklemi (N)"): 3010,
    (3, "Saatlik ücret ortalaması"): 5.7728,
    (3, "Saatlik ücret std. sapması"): 2.6296,
    (3, "Saatlik ücret medyanı"): 5.375,
    (3, "Log ücret ortalaması"): 1.6567,
    (3, "Log ücret std. sapması"): 0.4438,
    (3, "Log ücret en küçük değeri"): 0.0,
    (3, "Eğitim ortalaması"): 13.2635,
    (3, "Deneyim ortalaması"): 8.8561,
    (3, "Siyahi payı"): 0.2336,
    (3, "Yaş ortalaması"): 28.1196,
    (4, "Siyahi olmayanlarda ortalama log ücret"): 1.7309,
    (4, "Siyahilerde ortalama log ücret"): 1.4129,
    (4, "Eğitim=12 gözlem sayısı"): 992,
    (4, "Eğitim=12 ortalama log ücret"): 1.6439,
    (4, "Eğitim=16 ortalama log ücret"): 1.7794,
    (4, "Eğitim=16 ortalama ücret"): 6.4289,
    (4, "Eğitim=16 medyan ücret"): 6.18,
    (4, "Eğitim=18 ortalama log ücret"): 1.9532,
    (5, "Model (1) eğitim katsayısı"): 0.0521,
    (5, "Model (1) sabit terim"): 0.9657,
    (6, "Model (2) eğitim"): 0.0932,
    (6, "Model (2) deneyim"): 0.0898,
    (6, "Model (2) deneyim²/100"): -0.2486,
    (6, "Model (3) eğitim"): 0.0818,
    (6, "Model (3) deneyim"): 0.0882,
    (6, "Model (3) deneyim²/100"): -0.2484,
    (6, "Model (3) siyahi"): -0.2319,
    (6, "Model (3) sabit terim"): 0.0824,
    (6, "Model (1) R²"): 0.0987,
    (6, "Model (2) R²"): 0.1958,
    (6, "Model (3) R²"): 0.2411,
    (6, "Eğitim: kesin yüzde etki"): 8.52,
    (6, "Siyahi: kesin yüzde fark"): -20.70,
    (7, "Eğitim standart hatası"): 0.0036,
    (7, "Deneyim standart hatası"): 0.0069,
    (7, "Deneyim²/100 standart hatası"): 0.0328,
    (7, "Siyahi standart hatası"): 0.0173,
    (7, "Sabit terim standart hatası"): 0.0687,
}
"""Kontrollerin Card1995 tam örneklemindeki (3.010 erkek) değerleri, gösterim basamağında: (adım, etiket) → değer.
Metinlerdeki sayılar bunlardır; testler bağımsız bir hesapla (numpy ve statsmodels) ve üç dilde doğrular."""


@cache
def alternative() -> LabSpec:
    """Alternatif örneğin tanımı; beklenen değerler Card1995'in tam örneklemindeki sayılardır (testle doğrulanır)."""

    spec = _spec(alternative_plan(), ALT_TEXTS, source="alternatif", title=ALT_TITLE, dataset=ALT_DATA,
                 code_note=HANSEN_NOTE)
    return with_expected(spec, ALT_EXPECTED)


STORY = (
    "Alternatif örnek Card (1995) verisidir: 1976'da 24–34 yaşındaki 3.010 genç erkek (NLSYM; Hansen'in arşivindeki "
    "Card1995.dta). Soru notlardaki gibidir: eğitim ile log saatlik ücret arasındaki koşullu ilişki; deneyim profili "
    "ve bir grup göstergesi (burada siyahi) eklenerek."
)


# --- Kendi verin --------------------------------------------------------------------------------------------

def _derived_data(case: Case) -> pd.DataFrame:
    """Kendi verinde türetilen sütunlarla veri (doğrulama ve metinler için; hesap uygulamada ayrıca yapılır)."""

    data = case.data.copy()
    y, z, g = case.roles[SONUC], case.roles[KONTROL], case.roles[GRUP]
    if case.options.get(LOG, True):
        data[LOG_Y] = np.log(data[y].astype(float))
    data[Z2] = data[z].astype(float) ** 2 / 100
    data[G01] = (data[g] == case.levels[GRUP]).astype(float)
    return data


def _points(values: pd.Series) -> tuple[float, float]:
    """Adım 4'ün kontrol noktaları: en sık görülen değer (eşitlikte küçüğü) ve ondan farklı uç değer (en büyük ya da
    en küçük)."""

    counts = values.value_counts()
    top = counts.max()
    mode = float(min(value for value, count in counts.items() if count == top))
    edge = float(values.max()) if float(values.max()) != mode else float(values.min())
    return mode, edge


def own_plan(case: Case) -> Plan:
    y, x, z, g = (case.roles[role] for role in (SONUC, ACIKLAYICI, KONTROL, GRUP))
    log = bool(case.options.get(LOG, True))
    pick = case.levels[GRUP]
    other = next(item for item in case.orders[g] if item != pick)
    name = case.name
    outcome = LOG_Y if log else y
    outcome_name = f"log({name(y)})" if log else name(y)
    prepare: list[Operation] = list(case.load)
    if log:
        prepare.append(Derive(case.frame, LOG_Y, E.log(E.var(y)), "Sonucun doğal logaritması"))
    prepare += [
        Derive(case.frame, Z2, E.div(E.power(E.var(z), 2), 100), "Kontrol değişkeninin karesi / 100"),
        Indicator(case.frame, G01, g, pick, "Grup göstergesi: seçilen kategori 1, diğeri 0"),
    ]
    words = {"x": name(x), "z": name(z), "z2": f"{name(z)}²/100", "group": f"“{pick}” göstergesi",
             "outcome": outcome_name, "level": name(y)}
    titles = dict(words)
    labels = {column: name(column) for column in (y, x, z, g)}
    labels.update({
        Z2: f"{name(z)}²/100",
        G01: f"{name(g)} = {pick}",
        INTERCEPT: "Sabit",
        "N": "N",
        "ort_sonuc": f"Ortalama {outcome_name}",
        "medyan_duzey": f"Medyan {name(y)}",
    })
    if log:
        labels.update({LOG_Y: outcome_name, "ort_duzey": f"Ortalama {name(y)}"})
    mode, edge = _points(case.data[x])
    return Plan(
        frame=case.frame, level=y, outcome=outcome, log=log, x=x, z=z, z2=Z2, group=G01,
        group_names=(other, pick),
        group_checks=(f"“{other}” grubunda ortalama {outcome_name}", f"“{pick}” grubunda ortalama {outcome_name}"),
        describe=tuple(dict.fromkeys((y, outcome, x, z, G01))),
        means=(),
        points=(mode, edge),
        prepare=tuple(prepare), words=words, titles=titles, labels=labels,
        plot_title=f"{name(x)} düzeyine ve gruba göre ortalama {outcome_name}",
        projection_title=f"{name(x)} düzeyine göre koşullu ortalama ve OLS doğrusal projeksiyonu",
        y_axis=f"Ortalama {outcome_name}",
        table_title=f"{outcome_name} için OLS regresyonları",
        percent=(("x_yuzde", "Model (3) temel katsayının kesin yüzde karşılığı"),
                 ("grup_yuzde", "Model (3) grup göstergesinin kesin yüzde karşılığı")),
    )


def _direction(value: float) -> str:
    return "yükselir" if value > 0 else "düşer"


def _own_texts(case: Case, plan: Plan) -> dict:
    """Kendi verin: adım metinleri (sütun adları ``md`` ile) ve sonuçlardan yazılan yorumlar (``note_for``)."""

    y, x, z = (md(case.label(role)) for role in (SONUC, ACIKLAYICI, KONTROL))
    group, pick = md(case.label(GRUP)), md(case.levels[GRUP])
    outcome = f"log({y})" if plan.log else y

    def step3(state) -> str:
        table = state.tables["betimsel"]
        frame = state.frames[plan.frame]
        mean, median = table.loc[plan.level, "mean"], table.loc[plan.level, "median"]
        text = f"Ortalama “{y}” {sayi(mean, 2)}, medyanı {sayi(median, 2)}."
        skew = float(frame[plan.level].skew())
        if mean > median and skew > SKEW_LIMIT:
            text += f" Ortalama medyandan büyüktür ve çarpıklık katsayısı {sayi(skew, 2)}: dağılım sağa çarpıktır."
            if plan.log:
                log_skew = float(frame[plan.outcome].skew())
                if abs(log_skew) < skew:
                    text += f" Log dönüşümünden sonra çarpıklık {sayi(log_skew, 2)}: log bu çarpıklığı azaltır."
        else:
            text += f" Çarpıklık katsayısı {sayi(skew, 2)}: dağılımda belirgin bir sağa çarpıklık yoktur."
        share = table.loc[G01, "mean"]
        text += (f" “{pick}” göstergesinin ortalaması {sayi(share, 4)}: bir kukla değişkenin ortalaması, o grubun "
                 "örneklemdeki payıdır.")
        return text

    def step4(state) -> str:
        frame = state.frames[plan.frame]
        means = frame.groupby(G01)[plan.outcome].mean()
        mode, edge = plan.points
        count = int((frame[plan.x] == mode).sum())
        return (f"“{md(plan.group_names[1])}” grubunda ortalama {outcome} {sayi(means.loc[1.0], 4)}, "
                f"“{md(plan.group_names[0])}” grubunda {sayi(means.loc[0.0], 4)}. “{x}” değişkeninin en sık görülen "
                f"değeri {md(_value(mode))} ({sayim(count)} gözlem). Bu sayılar grup ortalamalarıdır, nedensel etkiler "
                "değildir.")

    def step5(state) -> str:
        result = state.models["m1"]
        b0, b1 = float(result.params["Intercept"]), float(result.params[plan.x])
        sign = "−" if b1 < 0 else "+"
        return (f"Tahmin edilen denklem: ŷ = {katsayi(b0)} {sign} {katsayi(abs(b1))}·x (y: {outcome}, x: "
                f"“{x}”). {PROJECTION}")

    def step6(state) -> str:
        b = [float(state.models[m].params[plan.x]) for m in ("m1", "m2", "m3")]
        text = (f"“{x}” katsayısı Model (1)'de {katsayi(b[0])}; “{z}” ve karesi eklenince {katsayi(b[1])} olur, grup "
                f"göstergesi de eklenince {katsayi(b[2])}. Kontroller eklendikçe katsayının değişmesi, kontrollerin "
                f"hem “{x}” ile hem sonuçla ilişkili olduğunu gösterir.")
        if plan.log:
            x_percent, g_percent = (state.scalars[name] for name, _ in plan.percent)
            text += (f" Model (3)'te “{x}” katsayısı, sonuçta kesin dönüşümle {yuzde(x_percent, 2)} farka karşılık "
                     f"gelir; “{pick}” grubunun katsayısı {yuzde(g_percent, 2)} farka.")
        else:
            text += (" Sonuç logaritmik değildir: katsayılar sonucun kendi birimindeki farklardır; yüzde dönüşüm "
                     "gerekmez.")
        return text + " Bunlar koşullu ilişkilerdir, nedensel etkiler değildir."

    def step7(state) -> str:
        if not exact_fit(_derived_data(case), plan.outcome, (plan.x, plan.z, plan.z2, plan.group)):
            return CLASSICAL_SE
        return CLASSICAL_SE + " " + EXACT_FIT_NOTE

    def step8(state) -> str:
        b = [float(state.models[m].params[plan.x]) for m in ("m1", "m2", "m3")]
        n = int(state.models["m3"].nobs)
        sign = "pozitif" if b[0] > 0 else "negatif"
        return (f"Yüklenen veride ({sayim(n)} gözlem) “{x}” ile {outcome} arasında {sign} bir ilişki "
                f"bulunmaktadır. Yalnız “{x}” değişkenini içeren spesifikasyonda katsayı {katsayi(b[0])} olarak "
                f"tahmin edilmektedir. “{z}” ve karesi eklendiğinde katsayı {katsayi(b[1])}, grup göstergesi de kontrol "
                f"edildiğinde {katsayi(b[2])} olmaktadır. Bu ilişki, bir tanımlama stratejisi kurulmadığı için nedensel etki olarak "
                "yorumlanmamalıdır.")

    return {
        1: (
            f"Araştırma sorusu: *“{x}” yükseldikçe “{y}” değişkeninin koşullu ortalaması nasıl değişmektedir?*\n\n"
            f"Sonuç değişkeni $Y_i$ = {outcome}" + (" (notlardaki gibi logaritma)" if plan.log else "") + f", "
            f"açıklayıcı değişken $X_i$ = “{x}”. İlk tahmin hedefi anakütledeki koşullu beklenti fonksiyonudur: "
            "$m(x)=\\mathbb{E}[Y_i\\mid X_i=x]$. Örneklemde her düzeyin grup ortalamasıyla tahmin edilir.",
            "Araştırma sorusu \"katsayı kaçtır?\" değildir. Önce hangi anakütle nesnesini öğrenmek istediğimizi yazarız; "
            "katsayı ondan sonra gelir.",
        ),
        2: (
            "Dosyanızdan seçilen sütunlar okunur ve temizlenir. Analiz için türetilen değişkenler: "
            + ("sonucun logaritması, " if plan.log else "")
            + f"kontrol değişkeninin (“{z}”) karesinin 100'e bölümü ve “{group}” sütunundan “{pick}” = 1 olan grup "
            "göstergesi.",
            "Hangi gözlemlerin örnekleme girdiği ve türetilmiş değişkenlerin nasıl kurulduğu kodla belgelenmelidir. "
            "Boş hücresi olan satırlar analizden çıkarılır; sayısı veri panelinde yazılıdır.",
        ),
        (2, "not"): None,
        3: ("Regresyondan önce temel değişkenlerin dağılımına bakarız: ortalama, standart sapma, medyan, en küçük ve "
            "en büyük değer.", ""),
        (3, "not"): step3,
        4: (f"Regresyondan önce grup örüntülerini görünür kılarız: “{x}” düzeylerine göre sonucun koşullu "
            f"ortalamaları ve iki grubun (“{pick}” ve diğeri) ortalama eğrileri.", ""),
        (4, "not"): step4,
        5: (f"Yalnız “{x}” değişkenini kullanan doğrusal projeksiyon tahmin edilir. Nokta büyüklükleri o "
            "düzeydeki gözlem sayısını gösterir.", PROJECTION),
        (5, "not"): step5,
        6: (f"Aynı soru üç spesifikasyonla yan yana raporlanır: (1) yalnız “{x}”, (2) “{z}” ve karesi eklenmiş, "
            "(3) grup göstergesi de eklenmiş. Tablo sütunlar arasında katsayının nasıl değiştiğine bakılarak okunur.",
            ""),
        (6, "not"): step6,
        7: ("Makale tablosu ile yazılım çıktısı farklı nesneler değildir: tablo, çıktının araştırma sorusu için önemli "
            "kısmının okunur biçimidir. Her satırda katsayı, standart hata, $t$ oranı, $p$-değeri ve %95 güven "
            "aralığı yer alır.", CLASSICAL_SE),
        (7, "not"): step7,
        8: (READING, ""),
        (8, "not"): step8,
        9: (PROTOCOL, ""),
    }


def build(case: Case) -> LabSpec:
    """Kendi verinle Konu 1 uygulaması; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    plan = own_plan(case)
    texts = _own_texts(case, plan)
    exact = exact_fit(_derived_data(case), plan.outcome, (plan.x, plan.z, plan.z2, plan.group))
    spec = _spec(plan, texts, source="kendi", title=f"{case.label(SONUC)} ve {case.label(ACIKLAYICI)}", dataset="",
                 code_note=OWN_NOTE, exact=exact)
    return with_app_values(spec)


def validate(case: Case) -> None:
    y, x, z, g = (case.roles[role] for role in (SONUC, ACIKLAYICI, KONTROL, GRUP))
    if len({y, x, z}) < 3:
        raise K.UploadError("Sonuç, açıklayıcı ve kontrol değişkeni için üç farklı sütun seçin.")
    data = case.data
    if data[y].nunique() < 2:
        raise K.UploadError(f"“{case.name(y)}” sütununda en az iki farklı değer olmalı.")
    if case.options.get(LOG, True) and (data[y] <= 0).any():
        raise K.UploadError(f"“{case.name(y)}” sütununda sıfır ya da negatif değer var; logaritma alınamaz. "
                            "“Sonucun logaritmasını al” seçeneğini kapatın ya da başka bir sütun seçin.")
    levels = data[x].nunique()
    if levels < 2:
        raise K.UploadError(f"“{case.name(x)}” sütununda en az iki farklı değer olmalı.")
    if levels > MAX_LEVELS:
        raise K.UploadError(f"“{case.name(x)}” sütununda {levels} farklı değer var. Koşullu ortalama tablosu için "
                            f"açıklayıcı değişken en çok {MAX_LEVELS} farklı değer almalı (ör. eğitim yılı, çocuk "
                            "sayısı).")
    if data[z].nunique() < 3:
        raise K.UploadError(f"“{case.name(z)}” sütununda en az üç farklı değer olmalı (değişken karesiyle eklenir).")
    derived = _derived_data(case)
    counts = derived[G01].value_counts()
    if counts.min() < 2:
        raise K.UploadError(f"“{case.name(g)}” sütununun iki kategorisinde de en az iki gözlem olmalı.")
    if not full_rank(derived, (x, z, Z2, G01)):
        raise K.UploadError("Seçilen değişkenler arasında tam doğrusal bağlantı var (ör. bir sütun diğerlerinin "
                            "doğrusal birleşimi ya da grup içinde sabit). Model (3) tahmin edilemez; başka sütunlar "
                            "seçin.")
    outcome = LOG_Y if case.options.get(LOG, True) else y
    if not stable_design(derived, outcome, ((x,), (x, z, Z2), (x, z, Z2, G01))):
        raise K.UploadError(SCALE_MESSAGE)


def sample() -> pd.DataFrame:
    """Örnek dosya: kurgusal ücret verisi (``core.labs.ornek_veri``)."""

    return ucret_verisi()


def _positive(series: pd.Series) -> bool:
    """Sütunun bütün değerleri pozitif mi; metin olarak yazılmış sayılar (ondalık virgül dahil) uygulamanın okuduğu gibi
    sayıya çevrilir."""

    if pd.api.types.is_numeric_dtype(series):
        values = series.dropna().astype(float)
    else:
        texts = series.map(K.clean_text).dropna().astype(str).str.replace(",", ".", regex=False)
        values = pd.to_numeric(texts, errors="coerce")
        if values.isna().any():
            return True  # sayı olmayan hücre: sütun zaten reddedilir; seçenek kendi iletisiyle engellenmez
    return bool(len(values)) and bool((values > 0).all())


ROLES = (
    Role(SONUC, "Sonuç değişkeni", "sayisal", True, (2, 3, 4, 5, 6, 7, 8),
         "Açıklanan sayısal değişken (ör. saatlik ücret). Notlardaki gibi logaritması alınabilir."),
    Role(ACIKLAYICI, "Temel açıklayıcı değişken", "sayisal", True, (3, 4, 5, 6, 7, 8),
         f"Koşullu ortalaması alınacak, az sayıda değer alan değişken (ör. eğitim yılı); en çok {MAX_LEVELS} farklı "
         "değer."),
    Role(KONTROL, "Sürekli kontrol değişkeni", "sayisal", True, (3, 6, 7, 8),
         "Model (2)'de doğrusal ve karesel olarak eklenir (ör. deneyim, yaş)."),
    Role(GRUP, "İki kategorili grup değişkeni", "kategorik", True, (3, 4, 6, 7, 8),
         "Grafikte iki grup ayrı çizilir; Model (3)'te gösterge olarak eklenir (ör. cinsiyet).",
         levels=(2, 2), pick="1 ile kodlanan grup"),
)

OPTIONS = (
    Option(LOG, "Sonucun logaritmasını al (log Y)",
           "Notlardaki gibi sonucun doğal logaritması kullanılır; katsayılar yaklaşık yüzde farklar olarak okunur.",
           default=True, role=SONUC, allowed=_positive,
           blocked="Sonuç sütununda sıfır ya da negatif değer olduğu için logaritma alınamaz; sonuç kendi biriminde "
                   "kullanılır."),
)

CUSTOM = CustomLab(
    roles=ROLES,
    build=build,
    sample=sample,
    intro=(
        "Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Dört sütun seçilir: sonuç, az sayıda değer alan bir açıklayıcı "
        "değişken (ör. eğitim yılı), karesiyle eklenen sürekli bir kontrol (ör. deneyim) ve iki kategorili bir grup "
        "değişkeni (ör. cinsiyet). Bu sütunlarda boş hücresi olan satırlar analizden çıkarılır."
    ),
    options=OPTIONS,
    min_rows=10,
    validate=validate,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
