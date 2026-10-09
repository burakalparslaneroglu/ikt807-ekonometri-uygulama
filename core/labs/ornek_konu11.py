"""Konu 11 genel uygulaması: model seçimini gerçekten dış-örneklemde değerlendirmek.

Notlardaki §11.15'in dört adımı aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde yazılır (``build``):
açık kurallı eğitim/test bölmesi ve dört aday model (basit OLS, zengin OLS, Ridge CV, Lasso CV), test setini tuning için
kullanmamak (CV eğrileri), model seçimi tablosunda görünmesi gereken sayılar ve tahmin ile yapısal çıkarım ayrımı.

Alternatif örnek Card (1995) verisidir (Hansen'in arşivindeki ``Card1995.dta``): 1976'da ücreti ve IQ dahil aday
değişkenlerin hepsi gözlenen 2.034 genç erkek. Zengin model 15 değişkenden kurulan 132 terimli bir sözlüktür (sürekli
değişkenlerin kare ve küpleri, bütün ikili etkileşimler); eğitim örneklemi 1.524 gözlem olduğu için zengin OLS aşırı uyum
gösterir, Ridge ve Lasso test hatasını düşürür. Notlardaki büyük örneklemde (38.055 eğitim gözlemi, 30 terim) bu kazanç
yoktu. Sürekli değişkenler ortalamalarına yakın sabitlerden farkları olarak girer (ör. eğitim − 13): kuvvetler ve
çarpımlar daha az bağlantılı olur, koordinat inişi hızlı yakınsar; doğrusal uzay aynıdır. "Kendi verini yükle"
seçeneğinde aynı adımlar öğrencinin dosyasıyla kurulur. Notlardaki laboratuvar (``core.labs.konu11``) değişmez.

Rollerin notlardaki karşılıkları: sonuç log ücret, basit modelin değişkenleri eğitim, deneyim, deneyim² ve kadın, ek
adaylar eğitim², deneyim³ ve etkileşimler, kategorik değişkenler bölge, ırk ve Hispanik.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from core.labs import expr as E
from core.labs import kendi_veri as K
from core.labs import penalized as P
from core.labs.ornek import (
    Case,
    CustomLab,
    Option,
    Role,
    TopicVariants,
    deger,
    exact_fit,
    full_rank,
    md,
    sayi,
    sayim,
    with_app_values,
    with_expected,
)
from core.labs.ornek_veri import konut_verisi
from core.labs.penalized import GOLDEN
from core.labs.spec import (
    CVCurve,
    Check,
    Derive,
    Dictionary,
    DotPlot,
    DropMissing,
    LabSpec,
    LabStep,
    LoadHansen,
    ModelMetrics,
    NoteRef,
    Operation,
    Penalized,
    ReproClass,
    RowNumber,
    ScalarTarget,
    StatTarget,
    TableTarget,
    dictionary_terms,
    penalized_key,
)


def _sayi(value: float, decimals: int = 0) -> str:
    """Metinlerde Türkçe sayı, binlik ayırıcıyla (12.345,6)."""

    return sayi(value, decimals, binlik=True)


TOPIC = "konu11"
SECTION = "11.15"
TEST_SHARE = 0.25
FOLD_WIDTH = 0.15
MODELS = (("basit", "Basit OLS"), ("zengin", "Zengin OLS"), ("ridge", "Ridge CV"), ("lasso", "Lasso CV"))
SONUC = "sonuc"
TEMEL = ("temel1", "temel2", "temel3", "temel4", "temel5", "temel6")
"""Basit modelin değişken rolleri (notlarda eğitim, deneyim, deneyim²/100, kadın: dört değişken)."""
KATEGORILER = ("kategori", "kategori2", "kategori3")
"""Zengin modelin kategorik değişken rolleri (notlarda bölge, ırk, Hispanik)."""
TEMEL1, KATEGORI = TEMEL[0], KATEGORILER[0]
SOZLUK = "sozluk"


@dataclass(frozen=True)
class Plan:
    """Rollerin koddaki sütunları, sözlük, ızgaralar ve gösterim ayarları.

    ``prepare``: veriyi ve türetilmiş değişkenleri kuran işlemler (sözlük dahil); ``basic`` basit modelin, ``rich``
    zengin modelin sayısal terimleri; ``categorical`` zengin modelin kategorik değişkenleri. ``decimals``: MSE'lerin,
    ``lambda_decimals``: seçilen cezaların ondalığı (ridge, lasso).
    """

    frame: str
    y: str
    prepare: tuple[Operation, ...]
    basic: tuple[str, ...]
    rich: tuple[str, ...]
    categorical: tuple[str, ...]
    ridge_grid: tuple[float, float, int]
    lasso_grid: tuple[float, float, int]
    labels: dict
    decimals: int = 4
    lambda_decimals: tuple[int, int] = (3, 6)
    test_label: str = "test örneklemi"


def _fraction(expression: E.Expr) -> E.Expr:
    return E.sub(expression, E.floor(expression))


def split_operations(frame: str) -> tuple[Operation, ...]:
    """Notlardaki açık bölme kuralı: u = {i·φ}; u < 0,25 test, kalan eğitim; eğitimde kat ⌊(u − 0,25)/0,15⌋ + 1."""

    key = E.var("u")
    return (
        RowNumber(frame, "sira", "Gözlemin dosyadaki sıra numarası i = 1, …, n"),
        Derive(frame, "u", _fraction(E.mul(E.var("sira"), GOLDEN)),
               "u = {i·φ}, φ = (√5 − 1)/2: [0, 1) aralığına düzgün yayılan deterministik anahtar"),
        Derive(frame, "egitim", E.compare("ge", key, TEST_SHARE), "Eğitim örneklemi: u ≥ 0,25 (test: u < 0,25)"),
        Derive(frame, "kat", E.mul(E.var("egitim"), E.add(E.floor(E.div(E.sub(key, TEST_SHARE), FOLD_WIDTH)), 1)),
               "CV katı: eğitimde ⌊(u − 0,25)/0,15⌋ + 1 (1, …, 5); testte 0"),
    )


def _steps(plan: Plan, texts: dict) -> tuple[LabStep, ...]:
    f, y = plan.frame, plan.y
    md_d, (ridge_d, lasso_d) = plan.decimals, plan.lambda_decimals

    def metric(model: str, column: str, label: str, decimals: int) -> Check:
        return Check(label, TableTarget("secim", model, column), 0.0, decimals=decimals)

    return (
        LabStep(
            number=1,
            title="Aday modeller",
            note=NoteRef(SECTION, 1, ("Tablo 11.2", "Şekil 11.4")),
            explanation=texts[1][0],
            operations=(
                *plan.prepare,
                *split_operations(f),
                Penalized("basit", f, y, plan.basic, sample="egitim"),
                Penalized("zengin", f, y, plan.rich, plan.categorical, sample="egitim"),
                Penalized("ridge", f, y, plan.rich, plan.categorical, "ridge", plan.ridge_grid, "egitim", "kat"),
                Penalized("lasso", f, y, plan.rich, plan.categorical, "lasso", plan.lasso_grid, "egitim", "kat"),
                ModelMetrics(MODELS, "secim"),
                DotPlot("secim", "test_mse", MODELS, f"Test MSE ({plan.test_label})", texts["grafik"],
                        decimals=md_d),
            ),
            checks=(
                Check("Eğitim örneklemi (gözlem)", StatTarget(f, "egitim", "sum"), 0.0, decimals=0),
                Check("Test örneklemi (gözlem)", StatTarget(f, y, "count", ("egitim", 0)), 0.0, decimals=0),
                metric("basit", "test_mse", "Basit OLS test MSE", md_d),
                metric("zengin", "test_mse", "Zengin OLS test MSE", md_d),
                metric("ridge", "test_mse", "Ridge CV test MSE", md_d),
                metric("lasso", "test_mse", "Lasso CV test MSE", md_d),
                Check("Zengin OLS eğitim MSE", ScalarTarget(penalized_key("zengin", "egitim_mse")), 0.0,
                      decimals=md_d),
                metric("basit", "sifirdan", "Basit OLS katsayı sayısı", 0),
                metric("zengin", "sifirdan", "Zengin OLS katsayı sayısı", 0),
                metric("ridge", "sifirdan", "Ridge sıfırdan farklı katsayı", 0),
                metric("lasso", "sifirdan", "Lasso sıfırdan farklı katsayı", 0),
                Check("Ridge seçilen λ (SSE ölçeği)", ScalarTarget(penalized_key("ridge", "lambda")), 0.0,
                      decimals=ridge_d),
                Check("Lasso seçilen λ (yazılım ölçeği)", ScalarTarget(penalized_key("lasso", "lambda")), 0.0,
                      decimals=lasso_d),
            ),
            reproducibility=ReproClass.CONVENTION,
            takeaway=texts[1][1],
            code_note=texts.get("kod1", CODE_NOTE),
            note_for=texts.get((1, "not")),
        ),
        LabStep(
            number=2,
            title="Test setini tuning için kullanmamak",
            note=NoteRef(SECTION, 2),
            explanation=texts[2][0],
            operations=(
                CVCurve("lasso", "Ceza parametresi λ (Lasso, yazılım ölçeği)", "Lasso: 5-katlı CV ile ceza seçimi"),
                CVCurve("ridge", "Ceza parametresi λ (Ridge, SSE ölçeği)", "Ridge: 5-katlı CV ile ceza seçimi"),
            ),
            reproducibility=ReproClass.CONVENTION,
            takeaway=texts[2][1],
            note_for=texts.get((2, "not")),
        ),
        LabStep(
            number=3,
            title="Model seçimi tablosunda hangi sayılar görünmeli?",
            note=NoteRef(SECTION, 3),
            explanation=REPORTING,
            takeaway=texts[3][1],
            note_for=texts.get((3, "not")),
        ),
        LabStep(
            number=4,
            title="Tahmin ile yapısal çıkarım ayrımını korumak",
            note=NoteRef(SECTION, 4),
            explanation=texts[4][0],
            takeaway=texts[4][1],
            note_for=texts.get((4, "not")),
        ),
    )


def _spec(plan: Plan, texts: dict, *, source: str, title: str, dataset: str) -> LabSpec:
    labels = {**plan.labels, "egitim": "Eğitim örneklemi (1) / test (0)", "kat": "CV katı",
              **{model: label for model, label in MODELS}}
    return LabSpec(topic_key=TOPIC, title=title, dataset=dataset, note_section=SECTION, steps=_steps(plan, texts),
                   labels=tuple(labels.items()), source=source)


# --- Ortak metinler -------------------------------------------------------------------------------------------

SPLIT_TEXT = (
    "Bölme rastgele sayı üreteciyle değil açık bir kuralla yapılır: gözlemin dosyadaki sıra numarası $i$ için\n\n"
    "$$u_i=\\{i\\varphi\\},\\qquad \\varphi=\\frac{\\sqrt5-1}{2}\\simeq0{,}618,$$\n\n"
    "$\\{\\cdot\\}$ kesirli kısımdır. $u_i<0{,}25$ olan gözlemler test, kalanlar eğitim örneklemidir; eğitimde çapraz "
    "doğrulama katı $\\lfloor(u_i-0{,}25)/0{,}15\\rfloor+1$'dir. Kural açık olduğu için Python, R ve Stata aynı bölmeyi ve "
    "aynı sayıları üretir."
)
REPORTING = (
    "Tahmin amaçlı bir makalede şu bilgiler önemlidir:\n\n"
    "1. eğitim, değerlendirme ve test örneklemi tanımı,\n"
    "2. kayıp fonksiyonu (MSE, MAE, log-loss vb.),\n"
    "3. tuning yöntemi ve kat sayısı,\n"
    "4. ön işleme zinciri,\n"
    "5. seçilen ceza parametresi ve ölçeği,\n"
    "6. dış-örneklem performans ve mümkünse belirsizliği,\n"
    "7. farklı seed veya bölmelerde duyarlılık.\n\n"
    "Tek bir test MSE'si model seçiminin mutlak sıralamasını kanıtlamaz; test MSE birkaç aşırı gözleme de duyarlıdır."
)
TUNING_TEXT = (
    "Lasso ve Ridge ceza parametreleri eğitim verisi içinde 5-katlı çapraz doğrulama ile, aynı katlarda seçilir: "
    "logaritmik ızgarada kat ortalama karesel hatalarının ortalamasını en küçük yapan ceza. Test örneklemi yalnız nihai "
    "performans karşılaştırması için kullanılır. Araştırmacı farklı $\\lambda$ değerlerini test setinde deneyip en iyi "
    "sonucu seçerse test seti artık bağımsız değerlendirme verisi değildir.\n\n"
    "Önemli olan Lasso komutunun kendisinden çok, standartlaştırma ve kategori kodlamanın da çapraz doğrulama katının "
    "içinde yapılmasıdır: her katta ölçek ve kategori düzeyleri yalnız o katın eğitim verisinden öğrenilir."
)
CODE_NOTE = (
    "Ceza ölçekleri: Ridge SSE ölçeğinde, $(\\boldsymbol Y-\\boldsymbol X\\beta)'(\\boldsymbol Y-\\boldsymbol X\\beta)+\\lambda\\beta'\\beta$ (scikit-learn `Ridge`, Hansen); "
    "Lasso yazılım ölçeğinde, $\\frac{1}{2n}\\|\\boldsymbol Y-\\boldsymbol X\\beta\\|^2+\\lambda_y\\|\\beta\\|_1$ (scikit-learn, R `glmnet`, Stata "
    "`lasso`; Hansen'in SSE ölçeğinde ceza $2n\\lambda_y$). Ridge kapalı biçimle, Lasso koordinat inişiyle çözülür; tolerans "
    "çok küçük tutulduğu için diller aynı optimuma yakınsar. Izgara, ölçekleme ve katlar kodda açıkça verilir."
)


def _grid_text(grid: tuple[float, float, int], symbol: str = r"\lambda") -> str:
    high, low, count = grid
    return (f"${symbol}\\in[10^{{{_sayi(low, 1 if low % 1 else 0).replace(',', '{,}')}}},"
            f"10^{{{_sayi(high, 1 if high % 1 else 0).replace(',', '{,}')}}}]$ aralığında {count} noktalı logaritmik ızgara")


# --- Alternatif örnek: Card (1995) ----------------------------------------------------------------------------

ALT_DATA = "card1995"
ALT_FRAME = "card"
ALT_TITLE = "Card Verisinde Model Seçimini Dış-Örneklemde Değerlendirmek"
ALT_RAW = ("ed76", "age76", "momed", "daded", "kww", "iq", "black", "south66", "smsa66r", "smsa76r", "reg76r", "momdad14",
           "libcrd14", "enroll76", "nearc4")
ALT_CENTERS = (("ogrenim", "ed76", 13), ("deneyim_m", "exp", 8), ("anne", "momed", 10), ("baba", "daded", 10),
               ("kww_m", "kww", 34), ("iq_m", "iq", 100))
"""Sürekli değişkenler: (kod adı, kaynak, merkez). Merkezler örneklem ortalamalarına yakın yuvarlak sayılardır."""
ALT_BINARY = ("black", "south66", "smsa66r", "smsa76r", "reg76r", "momdad14", "libcrd14", "enroll76", "nearc4")
ALT_DICTIONARY = Dictionary(
    ALT_FRAME, tuple(name for name, _, _ in ALT_CENTERS) + ALT_BINARY, tuple(name for name, _, _ in ALT_CENTERS), 3,
    True, "Aday terim sözlüğü: 15 değişken, altı sürekli değişkenin 2. ve 3. kuvvetleri, 105 ikili etkileşim",
)
ALT_RIDGE_GRID = (4.0, -2.0, 40)
ALT_LASSO_GRID = (-1.0, -4.0, 40)
ALT_LABELS = {
    "lwage76": "Log saatlik ücret (1976)", "ed76": "Eğitim yılı (1976)", "exp": "Deneyim", "exp2_100": "Deneyim²/100",
    "black": "Siyahi", "south66": "Güney (1966)", "smsa66r": "Metropol (1966)", "smsa76r": "Metropol (1976)",
    "reg76r": "Güney (1976)", "momdad14": "14 yaşında anne ve babayla", "libcrd14": "14 yaşında kütüphane kartı",
    "enroll76": "1976'da okula kayıtlı", "nearc4": "Dört yıllık koleje yakın", "momed": "Anne eğitimi",
    "daded": "Baba eğitimi", "kww": "KWW bilgi testi", "iq": "IQ", "ogrenim": "Eğitim − 13", "deneyim_m": "Deneyim − 8",
    "anne": "Anne eğitimi − 10", "baba": "Baba eğitimi − 10", "kww_m": "KWW − 34", "iq_m": "IQ − 100",
}


def alternative_plan() -> Plan:
    frame = ALT_FRAME
    prepare = (
        LoadHansen(ALT_DATA, "Card1995.dta", frame),
        DropMissing(frame, ("lwage76", *ALT_RAW),
                    "Analiz örneklemi: 1976 log ücreti ve aday değişkenlerin hepsi (IQ dahil) gözlenen erkekler"),
        Derive(frame, "exp", E.sub(E.sub(E.var("age76"), E.var("ed76")), 6), "Potansiyel deneyim = yaş − eğitim − 6"),
        Derive(frame, "exp2_100", E.div(E.power(E.var("exp"), 2), 100), "Deneyim²/100"),
        *(Derive(frame, name, E.sub(E.var(source), center), f"{ALT_LABELS[source]} − {center}")
          for name, source, center in ALT_CENTERS),
        ALT_DICTIONARY,
    )
    return Plan(
        frame=frame, y="lwage76", prepare=prepare, basic=("ed76", "exp", "exp2_100", "black"),
        rich=dictionary_terms(ALT_DICTIONARY), categorical=(), ridge_grid=ALT_RIDGE_GRID, lasso_grid=ALT_LASSO_GRID,
        labels=ALT_LABELS, decimals=4, lambda_decimals=(2, 4), test_label="510 gözlemlik test örneklemi",
    )


ALT_TEXTS = {
    1: (
        "Araştırma amacı **nedensel eğitim getirisi değil, log ücret tahmin performansıdır**. Card (1995) NLSYM verisinde "
        "1976'da ücreti ve aday değişkenlerin hepsi gözlenen 2.034 genç erkeği kullanıyoruz (IQ yalnız bir kısmında "
        "ölçülmüştür; örneklem bu yüzden Konu 1–2 alternatifindekinden küçüktür). " + SPLIT_TEXT + "\n\n"
        "Dört aday: eğitim, deneyim, deneyim²/100 ve siyahi göstergesini kullanan **Basit OLS**; 132 terimli bir sözlükle "
        "**Zengin OLS**; aynı sözlükte **Ridge** ve **Lasso**. Sözlük 15 değişkenden kurulur: altı sürekli değişken "
        "(eğitim, deneyim, anne ve baba eğitimi, KWW bilgi testi, IQ; yuvarlak bir sabitten farkları, ör. eğitim − 13) ve "
        "dokuz gösterge (siyahi; 1966'da güney ve metropol; 1976'da metropol ve güney; 14 yaşında anne ve babayla yaşama ve "
        "kütüphane kartı; 1976'da okula kayıt; dört yıllık koleje yakınlık). Sürekli değişkenlerin kare ve küpleri (12) ve "
        "15 değişkenin bütün ikili çarpımları (105) eklenir. Eğitim örnekleminde terim başına yaklaşık 12 gözlem vardır; "
        "notlardaki zengin modelde (30 terim, 38.055 eğitim gözlemi) yaklaşık 1.270.",
        "Zengin OLS'in test MSE'si (0,1368) basit modelinkinden (0,1357) bile yüksektir; eğitim MSE'si ise 0,1217'ye iner: "
        "132 terim eğitim verisindeki gürültüyü de öğrenir (aşırı uyum). Ridge (0,1276) ve Lasso (0,1277) aynı sözlükle "
        "test hatasını zengin OLS'e göre yaklaşık %7 düşürür; Lasso 132 terimden 39'unu tutar. Notlarda düzenlileştirme "
        "neredeyse hiç kazanç sağlamıyordu: fark büyük ölçüde eğitim gözlemi sayısının aday terim sayısına oranıyla "
        "açıklanır.",
    ),
    2: (
        TUNING_TEXT + " Ridge için " + _grid_text(ALT_RIDGE_GRID) + ", Lasso için " + _grid_text(ALT_LASSO_GRID, r"\lambda_y")
        + " kullanılır. Ridge ızgarası notlardakinden bir on yıl yukarıdadır (132 terimde iyi Ridge cezası daha "
        "büyüktür); Lasso ızgarası notlardakiyle aynıdır.",
        "Notlardaki düz CV eğrilerinin aksine burada eğriler belirgin bir çukur oluşturur: çok küçük cezalar (OLS'e yakın) "
        "eğitim katlarında aşırı uyar, çok büyük cezalar katsayıları gereğinden fazla küçültür. Seçilen cezalar ızgaranın "
        "içindedir. Test örneklemi bu grafiğin hiçbir noktasında kullanılmadı.",
    ),
    3: (
        REPORTING,
        "Bu laboratuvarın cevapları: eğitim/test bölmesi açık kuralla (1.524/510); kayıp MSE; 5-katlı CV, aynı katlar; "
        "ölçekleme kat içinde; Ridge λ = 1.193,78 (SSE ölçeği), Lasso λ_y = 0,01 (yazılım ölçeği). 510 gözlemlik test "
        "örnekleminde Ridge ile Lasso arasındaki fark (yaklaşık 0,00004) bir sıralama kanıtı değildir.",
    ),
    4: (
        "Lasso'nun bir kontrol değişkenini sıfırlaması, bu değişkenin nedensel analizde \"gereksiz\" olduğu anlamına "
        "gelmez; tutması da \"gerekli\" olduğunu göstermez. Tahmin algoritması $Y$'yi en iyi öngören seti arar. Bu "
        "sözlükte Lasso IQ ve KWW terimlerini, 1976'daki metropol ve güney göstergelerini ve okula kayıt göstergesini "
        "tutar: hepsi ücreti iyi öngörür. Ama bunlar eğitimden etkilenebilir (IQ ve KWW okul yıllarında ölçülmüştür; "
        "1976'daki konum ve okula kayıt eğitimden sonra gelir). Eğitimin etkisini tahmin ederken bunları kontrol etmek "
        "\"kötü kontrol\" olur: Konu 12'nin alternatif örneği aynı veride yalnız tedavi öncesi değişkenleri kullanır.",
        "Tez yazımında doğru sonuç dili: \"Küçük bir eğitim örnekleminde 132 terimli zengin OLS aşırı uyum göstermiş, Ridge "
        "ve Lasso test hatasını yaklaşık %7 azaltmıştır. Bu karşılaştırma öngörü performansına ilişkindir; Lasso'nun "
        "seçtiği değişkenler nedensel kontrol seti olarak otomatik kabul edilmemiştir.\"",
    ),
    "grafik": "Card1995: dış-örneklem tahmin performansı",
    "kod1": CODE_NOTE + " Sözlük 132 terim olduğu için Zengin OLS'in katsayıları yorumlanmaz; tahminleri karşılaştırılır.",
}

ALT_EXPECTED = {
    (1, "Eğitim örneklemi (gözlem)"): 1524,
    (1, "Test örneklemi (gözlem)"): 510,
    (1, "Basit OLS test MSE"): 0.1357,
    (1, "Zengin OLS test MSE"): 0.1368,
    (1, "Ridge CV test MSE"): 0.1276,
    (1, "Lasso CV test MSE"): 0.1277,
    (1, "Zengin OLS eğitim MSE"): 0.1217,
    (1, "Basit OLS katsayı sayısı"): 4,
    (1, "Zengin OLS katsayı sayısı"): 132,
    (1, "Ridge sıfırdan farklı katsayı"): 132,
    (1, "Lasso sıfırdan farklı katsayı"): 39,
    (1, "Ridge seçilen λ (SSE ölçeği)"): 1193.78,
    (1, "Lasso seçilen λ (yazılım ölçeği)"): 0.01,
}
"""Kontrollerin Card1995 örneklemindeki (2.034 erkek) değerleri; testler bağımsız bir hesapla doğrular."""


@cache
def alternative() -> LabSpec:
    """Alternatif örneğin tanımı; beklenen değerler Card1995 örneklemindeki sayılardır (testle doğrulanır)."""

    spec = _spec(alternative_plan(), ALT_TEXTS, source="alternatif", title=ALT_TITLE, dataset=ALT_DATA)
    return with_expected(spec, ALT_EXPECTED)


STORY = (
    "Alternatif örnek Card (1995) verisidir: 1976'da ücreti ve IQ dahil aday değişkenlerin hepsi gözlenen 2.034 genç erkek "
    "(Hansen'in arşivindeki Card1995.dta). Log ücret basit OLS, 132 terimli bir sözlükle zengin OLS, Ridge ve Lasso ile "
    "tahmin edilir; bölme notlardaki açık kuraldır. Eğitim örneklemi küçük olduğu için zengin OLS aşırı uyar, Ridge ve "
    "Lasso test hatasını düşürür."
)


# --- Kendi verin --------------------------------------------------------------------------------------------

MIN_ROWS = 100
MAX_TERMS = 200
MIN_LEVEL = 10
"""Kategorik değişkenin her düzeyinde en az gözlem (göstergeler eğitim katlarında sabit kalmasın)."""
MAX_LEVELS = 20
STABLE_LIMIT = 1e-8
"""Zengin OLS'in test MSE'si ile sütunları ortalanıp ölçeklenmiş tasarımdaki çözümün test MSE'si arasındaki en büyük
göreli fark (neredeyse doğrusal bağlantıda R ile Python ayrışır)."""


def _significant(value: float, digits: int = 2) -> float:
    if value == 0 or not math.isfinite(value):
        return 0.0
    return float(round(value, digits - 1 - math.floor(math.log10(abs(value)))))


def _value(value: float) -> str:
    return deger(value)


def split_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Notlardaki kural: eğitim göstergesi ve CV katı (1, …, 5; testte 0)."""

    u = P.golden_key(np.arange(1, n + 1))
    train = u >= TEST_SHARE
    folds = np.where(train, np.floor((u - TEST_SHARE) / FOLD_WIDTH) + 1, 0).astype(int)
    return train, folds


def _subsets(train: np.ndarray, folds: np.ndarray) -> list[np.ndarray]:
    """Ölçeklerin öğrenildiği satır kümeleri: bütün eğitim örneklemi ve her CV katının eğitim kısmı."""

    return [train] + [train & (folds != k) for k in range(1, 6)]


@dataclass(frozen=True)
class OwnDesign:
    """Kendi verinin terimleri.

    ``centered``: (kod adı, kaynak sütun, merkez) — sürekli değişkenler yuvarlak bir merkezden farkları olarak sözlüğe
    girer; ``indicators``: (kod adı, kaynak sütun, düzey) — kategorik değişkenlerin göstergeleri (her birinde ilk düzey
    referans); ``frame``: bütün türetilmiş sütunlarıyla veri (denetimler ve ızgara için).
    """

    derive: tuple[Operation, ...]
    basic: tuple[str, ...]
    rich: tuple[str, ...]
    dictionary: Dictionary | None
    excluded: tuple[str, ...]
    centered: tuple[tuple[str, str, float], ...]
    indicators: tuple[tuple[str, str, str], ...]
    frame: pd.DataFrame


def _new_name(base: str, suffix: str, taken: set[str]) -> str:
    name = f"{base}_{suffix}"
    while name in taken:
        name += "_"
    taken.add(name)
    return name


def _basics(case: Case) -> tuple[str, ...]:
    return tuple(case.roles[key] for key in TEMEL if case.has(key))


def _categoricals(case: Case) -> tuple[str, ...]:
    return tuple(case.roles[key] for key in KATEGORILER if case.has(key))


def own_design(case: Case) -> OwnDesign:
    from core.labs.spec import Indicator

    data, frame_name = case.data, case.frame
    basics = _basics(case)
    base_vars = tuple(dict.fromkeys(basics + case.extras))
    taken = set(data.columns)
    frame = data.copy()
    derive: list[Operation] = []
    centered: list[tuple[str, str, float]] = []
    names: list[str] = []
    for column in base_vars:
        values = frame[column].astype(float)
        if values.nunique() >= 3:
            center = _significant(float(values.mean()))
            name = _new_name(column, "m", taken)
            derive.append(Derive(frame_name, name, E.sub(E.var(column), center),
                                 f"{case.name(column)} − {_value(center)}"))
            frame[name] = values - center
            centered.append((name, column, center))
            names.append(name)
        else:
            names.append(column)
    indicators: list[tuple[str, str, str]] = []
    for column in _categoricals(case):
        for position, level in enumerate(case.orders[column][1:], start=2):
            name = _new_name(column, f"d{position}", taken)
            derive.append(Indicator(frame_name, name, column, level,
                                    f"Gösterge: “{case.name(column)}” = “{level}”"))
            frame[name] = (frame[column] == level).astype(float)
            indicators.append((name, column, level))
    train, folds = split_rule(len(frame))
    subsets = _subsets(train, folds)
    dictionary, excluded = None, []
    if case.options.get(SOZLUK, True):
        powers = tuple(name for name, _, _ in centered)
        candidate = Dictionary(frame_name, tuple(names), powers, 2, True,
                               "Aday terim sözlüğü: seçilen sayısal değişkenler, sürekli olanların kareleri ve bütün ikili "
                               "çarpımlar")
        frame = dictionary_frame(frame, candidate)
        excluded = [term for term in dictionary_terms(candidate)
                    if any(np.ptp(frame[term].to_numpy(dtype=float)[mask]) == 0 for mask in subsets)]
        excluded += aliased(frame.loc[train], [term for term in dictionary_terms(candidate) if term not in excluded],
                            set(candidate.base) | {name for name, _, _ in indicators})
        dictionary = Dictionary(frame_name, candidate.base, candidate.powers, 2, True, candidate.comment,
                                exclude=tuple(term for term in dictionary_terms(candidate) if term in set(excluded)))
        numeric = dictionary_terms(dictionary)
    else:
        numeric = tuple(names)
    rich = tuple(numeric) + tuple(name for name, _, _ in indicators)
    return OwnDesign(tuple(derive), basics, rich, dictionary, tuple(excluded), tuple(centered), tuple(indicators),
                     frame)


ALIAS_TOLERANCE = 1e-8
"""Sözlük terimi, önceki terimlerin uzayına uzaklığı (standartlaştırılmış sütunda) bu oranın altındaysa onların doğrusal
birleşimidir (ör. seyrek bir göstergenin başka bir göstergeyle çarpımı kendisine eşit)."""


def aliased(train: pd.DataFrame, terms: list[str], keep: set[str]) -> list[str]:
    """Eğitim örnekleminde önceki terimlerin tam doğrusal birleşimi olan sözlük terimleri (sırayla Gram–Schmidt).
    ``keep`` terimleri (sözlüğün tabanı) hiçbir zaman çıkarılmaz; onların bağlantısı ayrıca denetlenir."""

    values = train[terms].to_numpy(dtype=float)
    values = (values - values.mean(axis=0)) / values.std(axis=0)
    basis: list[np.ndarray] = []
    dropped: list[str] = []
    for index, term in enumerate(terms):
        column = values[:, index].copy()
        norm = float(np.linalg.norm(column))
        for vector in basis:
            column -= (vector @ column) * vector
        remainder = float(np.linalg.norm(column))
        if remainder <= ALIAS_TOLERANCE * norm and term not in keep:
            dropped.append(term)
            continue
        if remainder > ALIAS_TOLERANCE * norm:
            basis.append(column / remainder)
    return dropped


def dictionary_frame(frame: pd.DataFrame, dictionary: Dictionary) -> pd.DataFrame:
    """Sözlüğün sütunları (uygulamanın ``Dictionary`` hesabıyla aynı)."""

    created = {}
    for name in dictionary.powers:
        for power in range(2, dictionary.degree + 1):
            created[f"{name}_{power}"] = frame[name].to_numpy(dtype=float) ** power
    for index, first in enumerate(dictionary.base):
        for second in dictionary.base[index + 1:]:
            created[f"{first}_x_{second}"] = frame[first].to_numpy(dtype=float) * frame[second].to_numpy(dtype=float)
    return pd.concat([frame, pd.DataFrame(created, index=frame.index)], axis=1)


def lasso_max(frame: pd.DataFrame, y: str, numeric, train: np.ndarray) -> float:
    """Lasso'nun bütün katsayıları sıfır yapan en küçük cezası (yazılım ölçeği), eğitim örnekleminde:
    max_j |x_j'(y − ȳ)|/n, x_j eğitim verisiyle standartlaştırılmış."""

    x = frame.loc[train, list(numeric)].to_numpy(dtype=float)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    target = frame.loc[train, y].to_numpy(dtype=float)
    return float(np.max(np.abs(x.T @ (target - target.mean()))) / len(target))


def own_grids(n_train: int, largest: float) -> tuple[tuple[float, float, int], tuple[float, float, int]]:
    """Kendi verinin ızgaraları. Ridge (SSE ölçeği, sütunlar standart): 10^(log n + 2)'den 10^(log n − 5)'e 36 nokta
    (on yılda beş); Lasso: λ_max'tan dört on yıl aşağı 41 nokta (on yılda on). Üsler bir ondalığa yuvarlanır."""

    scale = math.log10(n_train)
    ridge = (round(scale + 2, 1), round(scale - 5, 1), 36)
    top = math.ceil(math.log10(largest) * 10) / 10
    return ridge, (top, round(top - 4, 1), 41)


def _lambda_decimals(grid: tuple[float, float, int]) -> int:
    """Seçilen cezanın ondalığı: ızgaranın en küçük iki noktasını ayıracak kadar."""

    return int(max(0, math.ceil(-grid[1] + 1)))


def own_plan(case: Case) -> Plan:
    data = case.data
    y = case.roles[SONUC]
    design = own_design(case)
    train, _ = split_rule(len(data))
    ridge, lasso = own_grids(int(train.sum()), lasso_max(design.frame, y, design.rich, train))
    variance = float(data[y].astype(float).var())
    decimals = int(min(10, max(0, 3 - math.floor(math.log10(variance))))) if variance > 0 else 4
    labels = {column: case.name(column) for column in data.columns}
    labels.update({name: f"{case.name(source)} − {_value(center)}" for name, source, center in design.centered})
    labels.update({name: f"{case.name(column)} = {level}" for name, column, level in design.indicators})
    prepare = case.load + design.derive + ((design.dictionary,) if design.dictionary is not None else ())
    return Plan(
        frame=case.frame, y=y, prepare=prepare, basic=design.basic, rich=design.rich, categorical=(),
        ridge_grid=ridge, lasso_grid=lasso, labels=labels, decimals=decimals,
        lambda_decimals=(_lambda_decimals(ridge), _lambda_decimals(lasso)),
        test_label=f"{sayim(len(data) - int(train.sum()))} gözlemlik test örneklemi",
    )


def _own_texts(case: Case, plan: Plan, design: OwnDesign) -> dict:
    y = case.md(SONUC)
    basics = ", ".join(f"“{md(case.name(column))}”" for column in plan.basic)
    d = plan.decimals
    pieces = [f"basit modelin değişkenleri ({basics})"]
    if case.extras:
        pieces.append("ek sayısal adaylar (" + ", ".join(f"“{md(case.name(c))}”" for c in case.extras) + ")")
    if design.indicators:
        counts = {column: sum(source == column for _, source, _ in design.indicators) for column in _categoricals(case)}
        pieces.append(", ".join(f"“{md(case.name(column))}” değişkeninin {count} göstergesi" for column, count in counts.items())
                      + (" (ilk kategori referans)" if len(counts) == 1 else " (her birinde ilk kategori referans)"))
    if design.dictionary is not None:
        rich_text = (f"Zengin modelin {len(plan.rich)} terimi: " + "; ".join(pieces) + "; sürekli değişkenlerin kareleri ve "
                     "sayısal değişkenlerin bütün ikili çarpımları (terim sözlüğü; kategorik değişkenlerin göstergeleri çarpımlara "
                     "girmez). "
                     "Sürekli değişkenler yuvarlak bir merkezden farkları olarak girer (ör. x − ortalamaya yakın bir "
                     "sabit): kare ve çarpımlar daha az bağlantılı olur. Zengin OLS'in tahminleri bundan etkilenmez (aynı "
                     "doğrusal uzay); Ridge ve Lasso'nunkiler merkeze bağlıdır, çünkü ceza ölçeklenmiş kare ve çarpım "
                     "sütunlarına uygulanır.")
        if design.excluded:
            rich_text += (f" {len(design.excluded)} çarpım ya da kare sözlükten çıkarıldı: eğitim örnekleminde veya bir CV "
                          "katının eğitim kısmında sabit kalıyor (ölçeklenemez) ya da önceki terimlerin tam doğrusal "
                          "birleşimi (ör. seyrek bir göstergenin başka bir göstergeyle çarpımı kendisine eşit).")
    else:
        rich_text = f"Zengin modelin {len(plan.rich)} terimi: " + "; ".join(pieces) + "."
    if design.indicators:
        rich_text += " Göstergeler diğer sütunlar gibi eğitim verisiyle ölçeklenir."

    def step1(state) -> str:
        table = state.tables["secim"]
        mse = {model: float(table.loc[model, "test_mse"]) for model, _ in MODELS}
        best = min(mse, key=mse.get)
        labels = dict(MODELS)
        train_mse = float(state.scalars[penalized_key("zengin", "egitim_mse")])
        text = ("Test MSE: " + "; ".join(f"{labels[m]} {_sayi(mse[m], d)}" for m, _ in MODELS)
                + f". En küçük test hatası {labels[best]} modelindedir. Zengin OLS'in eğitim MSE'si "
                f"{_sayi(train_mse, d)}.")
        if mse["zengin"] > mse["basit"]:
            text += " Zengin OLS basit modelden kötüdür: aşırı uyum."
        nonzero = int(table.loc["lasso", "sifirdan"])
        text += f" Lasso {len(plan.rich)} terimden {nonzero} tanesini tutar."
        return text

    def step2(state) -> str:
        parts = []
        for model, label in (("ridge", "Ridge"), ("lasso", "Lasso")):
            fit = state.models[model]
            grid = list(fit.cv.grid)
            index = grid.index(fit.lam)
            place = ("ızgaranın üst ucunda" if index == 0 else "ızgaranın alt ucunda" if index == len(grid) - 1
                     else "ızgaranın içinde")
            parts.append(f"{label} cezası {place}")
        text = "; ".join(parts) + "."
        if "ucunda" in text:
            text += " Uçta seçilen bir ceza, ölçütün ızgara dışında daha küçük olabileceğini gösterir; sonucu bu sınırla okuyun."
        return text

    def step3(state) -> str:
        s = state.scalars
        frame = state.frames[plan.frame]
        train = int(frame["egitim"].sum())
        return (f"Bu analizin cevapları: eğitim/test bölmesi açık kuralla ({sayim(train)}/{sayim(len(frame) - train)}); kayıp "
                "MSE; 5-katlı CV, aynı katlar; ölçekleme kat içinde; Ridge λ = "
                f"{_value(round(float(s[penalized_key('ridge', 'lambda')]), plan.lambda_decimals[0]))} (SSE ölçeği), "
                f"Lasso λ_y = {_value(round(float(s[penalized_key('lasso', 'lambda')]), plan.lambda_decimals[1]))} "
                "(yazılım ölçeği).")

    return {
        1: (f"Amaç “{y}” sonucunu dış-örneklemde iyi tahmin etmektir; nedensel bir katsayı aranmaz. "
            + SPLIT_TEXT.replace("Python, R ve Stata", "Python ve R")
            + "\n\nDört aday: basit modelin değişkenleriyle **Basit OLS**; zengin terim setiyle **Zengin OLS**, **Ridge** ve "
            "**Lasso**. " + rich_text,
            "Ek esnekliğin değeri eğitim gözlemi sayısının aday terim sayısına oranına bağlıdır: terim başına gözlem azken "
            "zengin OLS aşırı uyar ve düzenlileştirme test hatasını düşürür; bol veride kazanç küçüktür."),
        (1, "not"): step1,
        2: (TUNING_TEXT + " Izgaralar veriden kurulur: Ridge için " + _grid_text(plan.ridge_grid) + " (eğitim gözlemi "
            "sayısına göre), Lasso için " + _grid_text(plan.lasso_grid, r"\lambda_y") + " (bütün katsayıları sıfır yapan en küçük "
            "cezadan dört on yıl aşağı).",
            "Test örneklemi CV eğrilerinin hiçbir noktasında kullanılmaz."),
        (2, "not"): step2,
        3: (REPORTING, "Tek bir test MSE'si model seçiminin mutlak sıralamasını kanıtlamaz."),
        (3, "not"): step3,
        4: ("Lasso'nun bir değişkeni sıfırlaması, o değişkenin nedensel analizde \"gereksiz\" olduğu anlamına gelmez; "
            "tutması da \"gerekli\" olduğunu göstermez. Tahmin algoritması sonucu en iyi öngören seti arar; bir hedef "
            "katsayının eksik değişken yanlılığından korunması farklı bir problemdir (Konu 12).",
            "Bu karşılaştırma öngörü performansına ilişkindir; seçilen değişkenler nedensel kontrol seti olarak "
            "otomatik kabul edilmez."),
        "grafik": "Kendi veriniz: dış-örneklem tahmin performansı",
        "kod1": CODE_NOTE.replace(", Stata `lasso`", ""),
    }


def build(case: Case) -> LabSpec:
    """Kendi verinle Konu 11 uygulaması; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    design = own_design(case)
    plan = own_plan(case)
    spec = _spec(plan, _own_texts(case, plan, design), source="kendi",
                 title=f"{case.label(SONUC)}: dış-örneklem model seçimi", dataset="")
    return with_app_values(spec)


def _stable_rich(frame: pd.DataFrame, y: str, numeric, train: np.ndarray) -> bool:
    """Zengin OLS tahminleri yazılımlar arasında aynı hesaplanabilir mi: uygulamanın çözümünün (standartlaştırılmış
    tasarımda ``lstsq``) test tahminleri, QR çözümünün tahminleriyle göreli olarak ``STABLE_LIMIT`` içinde aynı."""

    built = P.design(frame[train], frame, numeric, ())
    target = frame.loc[train, y].to_numpy(dtype=float)
    path = P.penalty_path(built.train, target, built.other, np.array([0.0]), "ols")
    xa = np.column_stack([np.ones(int(train.sum())), built.train])
    q, r = np.linalg.qr(xa)
    beta = np.linalg.solve(r, q.T @ target)
    reference = np.column_stack([np.ones(len(frame)), built.other]) @ beta
    spread = float(np.std(target)) or 1.0
    return bool(np.max(np.abs(path.predictions[:, 0] - reference)) <= STABLE_LIMIT * spread)


def validate(case: Case) -> None:
    data = case.data
    y = case.roles[SONUC]
    basics = list(_basics(case))
    categoricals = _categoricals(case)
    if y in basics or y in case.extras:
        raise K.UploadError("Sonuç değişkeni açıklayıcılar arasında seçilemez.")
    if len(set(basics)) != len(basics):
        raise K.UploadError("Basit modelin değişkenleri için farklı sütunlar seçin.")
    if len(set(categoricals)) != len(categoricals):
        raise K.UploadError("Kategorik değişkenler için farklı sütunlar seçin.")
    if len(data) < MIN_ROWS:
        raise K.UploadError(f"Eğitim/test bölmesi ve 5-katlı CV için en az {MIN_ROWS} gözlem gerekir; seçilen sütunlarda "
                            f"{len(data)} gözlem var.")
    if data[y].nunique() < 3:
        raise K.UploadError(f"“{case.name(y)}” sütununda en az üç farklı değer olmalı.")
    for column in (*basics, *case.extras):
        if data[column].nunique() < 2:
            raise K.UploadError(f"“{case.name(column)}” sütunu sabit; çıkarın.")
    for column in categoricals:
        counts = data[column].value_counts()
        if len(counts) > MAX_LEVELS:
            raise K.UploadError(f"“{case.name(column)}” sütununda {len(counts)} kategori var; en çok {MAX_LEVELS}.")
        if counts.min() < MIN_LEVEL:
            raise K.UploadError(f"“{case.name(column)}” sütununun bazı kategorilerinde {MIN_LEVEL}'dan az gözlem var "
                                f"(ör. “{counts.idxmin()}”): göstergeleri bazı eğitim katlarında sabit kalır. Seyrek "
                                "kategorileri birleştirin.")
    design = own_design(case)
    train, folds = split_rule(len(data))
    subsets = _subsets(train, folds)
    for name in design.basic + tuple(n for n, _, _ in design.indicators) + tuple(
            n for n in design.rich if design.dictionary is None or n in design.dictionary.base):
        if any(np.ptp(design.frame[name].to_numpy(dtype=float)[mask]) == 0 for mask in subsets):
            label = case.name(name) if name in case.labels else name
            raise K.UploadError(f"“{label}” eğitim örnekleminde ya da bir CV katının eğitim kısmında sabit kalıyor; "
                                "ölçeklenemez. Bu değişkeni çıkarın ya da seyrek değerleri birleştirin.")
    terms = len(design.rich)
    if terms > MAX_TERMS:
        raise K.UploadError(f"Zengin modelde {terms} terim var; en çok {MAX_TERMS}. Daha az değişken seçin ya da "
                            "\"Kareler ve ikili etkileşimler\" seçeneğini kapatın.")
    if int(train.sum()) < 2 * (terms + 1):
        raise K.UploadError(f"Eğitim örnekleminde {int(train.sum())} gözlem var; {terms} terimli zengin OLS için en az "
                            f"{2 * (terms + 1)} gözlem gerekir. Daha az değişken seçin ya da sözlüğü kapatın.")
    frame = design.frame
    if not full_rank(frame.loc[train], design.rich):
        raise K.UploadError("Zengin modelin terimleri arasında tam doğrusal bağlantı var (biri diğerlerinden "
                            "hesaplanabiliyor). Değişkenlerden birini çıkarın ya da sözlüğü kapatın.")
    if not full_rank(frame.loc[train], design.basic):
        raise K.UploadError("Basit modelin değişkenleri arasında tam doğrusal bağlantı var.")
    if exact_fit(frame.loc[train], y, design.rich):
        raise K.UploadError(f"“{case.name(y)}” zengin modelin terimlerinin (neredeyse) tam doğrusal bir fonksiyonu (R² ≈ "
                            "1): test hataları yuvarlama hatasından oluşur.")
    if not _stable_rich(frame, y, design.rich, train):
        raise K.UploadError("Zengin modelin terimleri neredeyse doğrusal bağlantılı: zengin OLS'in tahminleri yazılımlar "
                            "arasında aynı hassasiyetle hesaplanamıyor. Daha az değişken seçin ya da sözlüğü kapatın.")


def suggest(table: K.UploadedTable) -> dict[str, str]:
    """1. kategorik rol için sayı olmayan, 2–20 kategorili ve her kategorisinde en az 10 gözlem bulunan ilk sütun önerilir
    (öteki kategorik rolleri öğrenci seçer)."""

    for column in table.columns:
        values = table.frame[column]
        if pd.api.types.is_numeric_dtype(values):
            continue
        counts = values.map(K.clean_text).dropna().value_counts()
        if 2 <= len(counts) <= MAX_LEVELS and counts.min() >= MIN_LEVEL and K.usable(table, column, "kategorik"):
            return {KATEGORI: column}
    return {}


def sample() -> pd.DataFrame:
    """Örnek dosya: kurgusal konut verisi (``core.labs.ornek_veri``)."""

    return konut_verisi()


_NOTES_BASIC = ("eğitim", "deneyim", "deneyim²/100", "kadın", None, None)
_NOTES_CATEGORICAL = ("bölge", "ırk", "Hispanik")

ROLES = (
    Role(SONUC, "Sonuç değişkeni", "sayisal", True, (1, 2, 3),
         "Tahmin edilen sayısal değişken (notlarda log ücret)."),
    *(Role(key, f"Basit modelin {index}. değişkeni", "sayisal", index == 1, (1,),
           f"Basit OLS modelinin {index}. açıklayıcısı" + (f" (notlarda {note})" if note else "")
           + "; zengin modele de girer.")
      for index, (key, note) in enumerate(zip(TEMEL, _NOTES_BASIC), start=1)),
    *(Role(key, f"{index}. kategorik değişken", "kategorik", False, (1,),
           f"Zengin modele göstergeleriyle girer (notlarda {note}); ilk kategori referanstır. Her kategoride en az "
           f"{MIN_LEVEL} gözlem.", levels=(2, MAX_LEVELS))
      for index, (key, note) in enumerate(zip(KATEGORILER, _NOTES_CATEGORICAL), start=1)),
)

OPTIONS = (
    Option(SOZLUK, "Kareler ve ikili etkileşimler (terim sözlüğü)",
           "Açıksa zengin model, sayısal değişkenlerin kendileri, sürekli olanların kareleri ve bütün ikili çarpımlarıyla "
           "kurulur (alternatif örnekteki gibi); kapalıysa yalnız değişkenlerin kendileriyle.", default=True),
)

CUSTOM = CustomLab(
    roles=ROLES,
    build=build,
    sample=sample,
    intro=(
        "Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Tahmin edilecek sonuç ve basit modelin 1–6 değişkeni seçilir; "
        "zengin model için ek sayısal adaylar ve en çok üç kategorik değişken eklenebilir. Seçilen sütunlarda boş hücresi "
        "olan satırlar analizden çıkarılır. Gözlemler notlardaki açık kuralla eğitim (%75) ve test (%25) örneklemine "
        "ayrılır."
    ),
    options=OPTIONS,
    order_roles=KATEGORILER,
    min_rows=MIN_ROWS,
    extra_columns=True,
    extra_label="Zengin model için ek sayısal adaylar (isteğe bağlı)",
    extra_help="Zengin modele eklenen sayısal değişkenler; en çok 12.",
    max_extra=12,
    extra_required=True,
    validate=validate,
    suggest=suggest,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
