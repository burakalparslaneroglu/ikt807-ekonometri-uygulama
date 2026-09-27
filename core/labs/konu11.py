"""Konu 11 uygulama laboratuvarı: model seçimini gerçekten dış-örneklemde değerlendirmek.

Ders notları §11.15'in (Adım 1–4) birebir karşılığıdır. Veri: Hansen'in cps09mar dosyası, Konu 1'deki analiz
örneklemi (50.742 tam zamanlı çalışan). Bölme rastgele sayı üreteciyle değil açık bir kuralla yapılır: gözlemin
dosyadaki sıra numarası i için u = {i·φ}, φ = (√5 − 1)/2; u < 0,25 test, kalan eğitim; eğitimde CV katı
⌊(u − 0,25)/0,15⌋ + 1. Böylece Python, R ve Stata aynı bölmeyi kurar ve aynı sayıları verir.

Ceza ölçekleri: Ridge SSE ölçeğinde (Hansen), Lasso yazılım ölçeğinde (scikit-learn, glmnet, Stata); ızgaralar
logaritmik, 40 nokta. Ölçekleme ve kategori göstergeleri her CV katında o katın eğitim verisinden öğrenilir.
"""

from __future__ import annotations

from core.labs import expr as E
from core.labs.konu01 import KONU01_LAB
from core.labs.penalized import GOLDEN
from core.labs.spec import (
    CVCurve,
    Check,
    Derive,
    DotPlot,
    LabSpec,
    LabStep,
    ModelMetrics,
    NoteRef,
    Penalized,
    ReproClass,
    RowNumber,
    ScalarTarget,
    StatTarget,
    TableTarget,
    penalized_key,
)

SECTION = "11.15"
FRAME = "cps"
BASE = ("education", "experience", "experience2_100", "female")
RICH = ("education", "education2", "experience", "experience2_100", "experience3_1000", "female", "edu_female",
        "exp_female")
CATEGORICAL = ("region", "race", "hisp")
RIDGE_GRID = (3.0, -3.0, 40)
LASSO_GRID = (-1.0, -4.0, 40)
TEST_SHARE = 0.25
FOLD_WIDTH = 0.15
MODELS = (
    ("basit", "Basit OLS"),
    ("zengin", "Zengin OLS"),
    ("ridge", "Ridge CV"),
    ("lasso", "Lasso CV"),
)


def _fraction(expression: E.Expr) -> E.Expr:
    """Kesirli kısım {a} = a − ⌊a⌋."""

    return E.sub(expression, E.floor(expression))


_KEY = E.var("u")

SAMPLE_OPERATIONS = (
    Derive(FRAME, "education2", E.power(E.var("education"), 2), "Eğitimin karesi"),
    Derive(FRAME, "experience3_1000", E.div(E.power(E.var("experience"), 3), 1000),
           "Deneyimin küpü / 1000"),
    Derive(FRAME, "edu_female", E.mul(E.var("education"), E.var("female")), "Eğitim × kadın etkileşimi"),
    Derive(FRAME, "exp_female", E.mul(E.var("experience"), E.var("female")), "Deneyim × kadın etkileşimi"),
    RowNumber(FRAME, "sira", "Gözlemin dosyadaki sıra numarası i = 1, …, n"),
    Derive(FRAME, "u", _fraction(E.mul(E.var("sira"), GOLDEN)),
           "u = {i·φ}, φ = (√5 − 1)/2: [0, 1) aralığına düzgün yayılan deterministik anahtar"),
    Derive(FRAME, "egitim", E.compare("ge", _KEY, TEST_SHARE), "Eğitim örneklemi: u ≥ 0,25 (test: u < 0,25)"),
    Derive(FRAME, "kat", E.mul(E.var("egitim"), E.add(E.floor(E.div(E.sub(_KEY, TEST_SHARE), FOLD_WIDTH)), 1)),
           "CV katı: eğitimde ⌊(u − 0,25)/0,15⌋ + 1 (1, …, 5); testte 0"),
)

MODEL_OPERATIONS = (
    Penalized("basit", FRAME, "lwage", BASE, sample="egitim"),
    Penalized("zengin", FRAME, "lwage", RICH, CATEGORICAL, sample="egitim"),
    Penalized("ridge", FRAME, "lwage", RICH, CATEGORICAL, "ridge", RIDGE_GRID, "egitim", "kat"),
    Penalized("lasso", FRAME, "lwage", RICH, CATEGORICAL, "lasso", LASSO_GRID, "egitim", "kat"),
    ModelMetrics(MODELS, "secim"),
    DotPlot("secim", "test_mse", MODELS, "Test MSE (12.687 gözlemlik test örneklemi)",
            "CPS: dış-örneklem tahmin performansı"),
)


def _metric(model: str, column: str, expected: float, label: str, decimals: int) -> Check:
    return Check(label, TableTarget("secim", model, column), expected, decimals=decimals)


STEPS = (
    LabStep(
        number=1,
        title="Aday modeller",
        note=NoteRef(SECTION, 1, ("Tablo 11.2", "Şekil 11.4")),
        explanation=(
            "Araştırma amacı bu kez **nedensel eğitim getirisi değil, log ücret tahmin performansıdır**. Hansen'in CPS "
            "verisinin tamamını (50.742 gözlem) kullanıyor, gözlemlerin yaklaşık yüzde 75'ini eğitim ve yüzde 25'ini "
            "test örneklemine ayırıyoruz. Bölme rastgele sayı üreteciyle değil açık bir kuralla yapılır: gözlemin "
            "dosyadaki sıra numarası $i$ için\n\n"
            "$$u_i=\\{i\\varphi\\},\\qquad \\varphi=\\frac{\\sqrt5-1}{2}\\simeq0{,}618,$$\n\n"
            "$\\{\\cdot\\}$ kesirli kısımdır. $u_i<0{,}25$ olan gözlemler test, kalanlar eğitim örneklemidir; eğitimde "
            "çapraz doğrulama katı $\\lfloor(u_i-0{,}25)/0{,}15\\rfloor+1$'dir. $\\{i\\varphi\\}$ dizisi $[0,1)$ aralığına "
            "düzgün yayıldığı için bu bölme dengeli bir rastgele bölme gibi davranır; kural açık olduğu için Python, R ve "
            "Stata aynı bölmeyi ve aynı sayıları üretir.\n\n"
            "Dört aday: yalnız eğitim, deneyim, deneyim karesi ve cinsiyet kullanan **Basit OLS**; sekiz sürekli terim "
            "(eğitim, eğitimin karesi, deneyim, deneyim²/100, deneyim³/1000, kadın, eğitim×kadın, deneyim×kadın) ve bölge, "
            "ırk ve Hispanik kökenli olma göstergeleriyle **Zengin OLS**; aynı zengin özellik setinde **Ridge** ve "
            "**Lasso**. Göstergeler eğitim örnekleminde görülen düzeylerden kurulur, ilk düzey referanstır: bölge 3, ırk "
            "18 (eğitimde görülen 19 düzeyden), Hispanik 1 gösterge; toplam 30 terim. Ridge ve Lasso'da sürekli özellikler "
            "yalnız eğitim verisinden öğrenilen ortalama ve standart sapmayla ölçeklenir."
        ),
        operations=KONU01_LAB.step(2).operations + SAMPLE_OPERATIONS + MODEL_OPERATIONS,
        checks=(
            Check("Eğitim örneklemi (gözlem)", StatTarget(FRAME, "egitim", "sum"), 38055, decimals=0),
            Check("Test örneklemi (gözlem)", StatTarget(FRAME, "lwage", "count", ("egitim", 0)), 12687, decimals=0),
            _metric("basit", "test_mse", 0.3215, "Basit OLS test MSE", 4),
            _metric("zengin", "test_mse", 0.3121, "Zengin OLS test MSE", 4),
            _metric("ridge", "test_mse", 0.3120, "Ridge CV test MSE", 4),
            _metric("lasso", "test_mse", 0.3119, "Lasso CV test MSE", 4),
            _metric("basit", "sifirdan", 4, "Basit OLS katsayı sayısı", 0),
            _metric("zengin", "sifirdan", 30, "Zengin OLS katsayı sayısı", 0),
            _metric("ridge", "sifirdan", 30, "Ridge sıfırdan farklı katsayı", 0),
            _metric("lasso", "sifirdan", 16, "Lasso sıfırdan farklı katsayı", 0),
            Check("Ridge seçilen λ (SSE ölçeği)", ScalarTarget(penalized_key("ridge", "lambda")), 3.455, decimals=3),
            Check("Lasso seçilen λ (yazılım ölçeği)", ScalarTarget(penalized_key("lasso", "lambda")), 0.000143,
                  decimals=6),
        ),
        reproducibility=ReproClass.CONVENTION,
        takeaway=(
            "Zengin OLS, Ridge ve Lasso'nun test MSE'leri birbirine son derece yakındır (0,3121; 0,3120; 0,3119). Ek "
            "esneklik basit modele göre küçük bir kazanım sağlar (0,3215'ten 0,3121'e); düzenlileştirme bu özellik setinde "
            "büyük bir üstünlük yaratmaz: 38.055 eğitim gözlemi ve 30 terimle cezalar çok küçük seçilir. Lasso 30 yerine 16 "
            "katsayıyı sıfırdan farklı tutar; sıfırladığı 14 terim eğitim×kadın etkileşimi ile gözlem sayısı küçük 13 ırk "
            "göstergesidir. Seyrek modelin test hatası Zengin OLS'ten yalnız dördüncü ondalıkta farklıdır."
        ),
        code_note=(
            "Ceza ölçekleri: Ridge SSE ölçeğinde, $(Y-X\\beta)'(Y-X\\beta)+\\lambda\\beta'\\beta$ (scikit-learn `Ridge`, "
            "Hansen); Lasso yazılım ölçeğinde, $\\frac{1}{2n}\\|Y-X\\beta\\|^2+\\lambda\\|\\beta\\|_1$ (scikit-learn, R "
            "`glmnet`, Stata `lasso`; Hansen'in SSE ölçeğinde ceza $2n\\lambda$). Ridge kapalı biçimle, Lasso koordinat "
            "inişiyle çözülür (Python `enet_path`, R `glmnet`, Stata'da kodda yazılı Mata fonksiyonu); tolerans çok küçük "
            "tutulduğu için üç dil aynı optimuma yakınsar. Paketlerin varsayılan ayarları (otomatik ızgara, ölçekleme, "
            "rastgele katlar) farklıdır; ızgara, ölçekleme ve katlar kodda açıkça verildiği için sayılar aynıdır."
        ),
    ),
    LabStep(
        number=2,
        title="Test setini tuning için kullanmamak",
        note=NoteRef(SECTION, 2),
        explanation=(
            "Lasso ve Ridge ceza parametreleri eğitim verisi içinde 5-katlı çapraz doğrulama ile, aynı katlarda seçildi: "
            "Ridge için $\\lambda\\in[10^{-3},10^{3}]$, Lasso için $\\lambda\\in[10^{-4},10^{-1}]$ aralığında logaritmik "
            "40 noktalı ızgarada kat ortalama karesel hatalarının ortalamasını en küçük yapan ceza. Test örneklemi yalnız "
            "nihai performans karşılaştırması için kullanılır. Araştırmacı farklı $\\lambda$ değerlerini test setinde "
            "deneyip en iyi sonucu seçerse test seti artık bağımsız değerlendirme verisi değildir.\n\n"
            "Önemli olan Lasso komutunun kendisinden çok, standartlaştırma ve kategori kodlamanın da çapraz doğrulama "
            "katının içinde yapılmasıdır: her katta ölçek ve kategori düzeyleri yalnız o katın eğitim verisinden öğrenilir. "
            "scikit-learn'de aynı hesap `ColumnTransformer` ve `Lasso` içeren bir `Pipeline` nesnesini `GridSearchCV` ile, "
            "katları `PredefinedSplit(kat)` ile vererek de yapılabilir; aynı cezayı seçer, fakat her cezada modeli sıfırdan "
            "tahmin ettiği için ızgara boyunca sıcak başlangıçlı yoldan belirgin biçimde yavaştır."
        ),
        operations=(
            CVCurve("lasso", "Ceza parametresi λ (Lasso, yazılım ölçeği)", "Lasso: 5-katlı CV ile ceza seçimi"),
            CVCurve("ridge", "Ceza parametresi λ (Ridge, SSE ölçeği)", "Ridge: 5-katlı CV ile ceza seçimi"),
        ),
        reproducibility=ReproClass.CONVENTION,
        takeaway=(
            "CV eğrisi seçilen cezanın çevresinde çok düzdür: bu kadar büyük örneklemde düzenlileştirme OLS'i ancak az "
            "değiştirir ve pek çok ceza neredeyse aynı dış-örneklem hatasını verir. Test örneklemi bu grafiğin hiçbir "
            "noktasında kullanılmadı."
        ),
    ),
    LabStep(
        number=3,
        title="Model seçimi tablosunda hangi sayılar görünmeli?",
        note=NoteRef(SECTION, 3),
        explanation=(
            "Tahmin amaçlı bir makalede şu bilgiler önemlidir:\n\n"
            "1. eğitim, değerlendirme ve test örneklemi tanımı,\n"
            "2. kayıp fonksiyonu (MSE, MAE, log-loss vb.),\n"
            "3. tuning yöntemi ve kat sayısı,\n"
            "4. ön işleme zinciri,\n"
            "5. seçilen ceza parametresi ve ölçeği,\n"
            "6. dış-örneklem performans ve mümkünse belirsizliği,\n"
            "7. farklı seed veya bölmelerde duyarlılık.\n\n"
            "Tek bir test MSE'si model seçiminin mutlak sıralamasını kanıtlamaz. Test MSE birkaç aşırı gözleme de "
            "duyarlıdır: log ücreti çok düşük birkaç gözlemin test ya da eğitim örneklemine düşmesi MSE düzeyini belirgin "
            "biçimde değiştirebilir."
        ),
        takeaway=(
            "Bu laboratuvarın cevapları: eğitim/test bölmesi açık kuralla (38.055/12.687); kayıp MSE; 5-katlı CV, aynı "
            "katlar; ölçekleme ve göstergeler kat içinde; Ridge λ = 3,455 (SSE ölçeği), Lasso λ = 0,000143 (yazılım "
            "ölçeği). Benzer sonuçlarda tekrarlı çapraz doğrulama veya bootstrap ile farkın kararlılığı incelenebilir."
        ),
    ),
    LabStep(
        number=4,
        title="Tahmin ile yapısal çıkarım ayrımını korumak",
        note=NoteRef(SECTION, 4),
        explanation=(
            "Lasso'nun bir kontrol değişkenini sıfırlaması, bu değişkenin nedensel analizde \"gereksiz\" olduğu anlamına "
            "gelmez. Tahmin algoritması $Y$'yi en iyi öngören seti arar. Bir hedef katsayının eksik değişken yanlılığından "
            "korunması farklı bir problemdir. Bu nedenle Konu 12'de yalnız sonuç denklemine dayalı seçim yerine double "
            "selection ve ortogonal moment yaklaşımına geçiyoruz."
        ),
        takeaway=(
            "Tez yazımında doğru sonuç dili: \"Zengin OLS ve düzenlileştirilmiş modeller test örnekleminde benzer hata "
            "üretmiştir. Lasso daha seyrek bir model elde etmiş, ancak tahmin performansında belirgin üstünlük "
            "sağlamamıştır. Bu karşılaştırma öngörü performansına ilişkindir; seçilen değişkenler nedensel kontrol seti "
            "olarak otomatik kabul edilmemiştir.\""
        ),
    ),
)


KONU11_LAB = LabSpec(
    topic_key="konu11",
    title="Model Seçimini Gerçekten Dış-Örneklemde Değerlendirmek",
    dataset="cps09mar",
    note_section=SECTION,
    steps=STEPS,
    labels=KONU01_LAB.labels + (
        ("female", "Kadın"), ("education2", "Eğitim²"), ("experience3_1000", "Deneyim³/1000"),
        ("edu_female", "Eğitim × kadın"), ("exp_female", "Deneyim × kadın"), ("region", "Bölge"), ("race", "Irk"),
        ("hisp", "Hispanik"), ("egitim", "Eğitim örneklemi (1) / test (0)"), ("kat", "CV katı"),
        ("basit", "Basit OLS"), ("zengin", "Zengin OLS"), ("ridge", "Ridge CV"), ("lasso", "Lasso CV"),
    ),
    consistency_notes=(
        "Notlardaki laboratuvar 20.000 gözlemlik bir alt örneklem, kodu verilmeyen bir eğitim/test bölmesi ve LassoCV'nin "
        "kendi rastgele katlarını kullanıyordu; Python, R ve Stata aynı bölmeyi kuramıyordu. Laboratuvar Hansen'in tam "
        "örneklemiyle (50.742 gözlem) ve açık bir bölme kuralıyla (u = {i·φ}) yeniden kuruldu; tabloya seçilen ceza "
        "sütunu eklendi, şekil ve metin güncellendi.",
        "Notlar Lasso ve Elastic Net cezasının hangi ölçekte olduğunu (Hansen'in SSE ölçeği mi, yazılımların 1/(2n) "
        "ölçeği mi) belirtmiyordu; iki ölçek ve aralarındaki dönüşüm (SSE ölçeğinde 2nλ) notlara ve sunuma eklendi "
        "(§11.8–§11.11).",
    ),
)
