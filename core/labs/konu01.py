"""Konu 1 uygulama laboratuvarı: CPS ücret–eğitim ilişkisi.

Ders notları §1.13'ün (Adım 1–9) birebir karşılığıdır. Buradaki her ``Check``
notlarda basılı bir sayıdır; değer notlardan kopyalanmıştır, hesaplanmamıştır.
"""

from __future__ import annotations

from core.hansen_data import CPS09MAR_COLUMNS
from core.labs import expr as E
from core.labs.spec import (
    OLS,
    Check,
    CoefTarget,
    Derive,
    Describe,
    GroupMeanPlot,
    GroupSummary,
    LabSpec,
    LabStep,
    LoadHansen,
    ModelTarget,
    NoteRef,
    ProjectionPlot,
    RegressionTable,
    ReproClass,
    Scalar,
    ScalarTarget,
    ShowModel,
    StatTarget,
)

SECTION = "1.13"
FRAME = "cps"

M1 = ("education",)
M2 = ("education", "experience", "experience2_100")
M3 = ("education", "experience", "experience2_100", "female")


def _stat(variable: str, stat: str, expected: float, label: str, *, where=None, decimals=4) -> Check:
    return Check(label, StatTarget(FRAME, variable, stat, where), expected, decimals)


def _coef(model: str, term: str, expected: float, label: str, quantity: str = "coef") -> Check:
    return Check(label, CoefTarget(model, term, quantity), expected)


STEPS = (
    LabStep(
        number=1,
        title="Araştırma sorusunu ve tahmin hedefini yazmak",
        note=NoteRef(SECTION, 1),
        explanation=(
            "Araştırma sorusu: *Tam zamanlı çalışanların eğitim düzeyi yükseldikçe saatlik "
            "ücretlerinin koşullu ortalaması nasıl değişmektedir?*\n\n"
            "Saatlik ücret $wage_i = earnings_i/(hours_i \\times weeks_i)$, sonuç değişkeni "
            "$Y_i=\\log(wage_i)$ ve ilk tahmin hedefi koşullu beklenti fonksiyonu "
            "$m(x)=\\mathbb{E}[\\log(wage_i)\\mid education_i=x]$'tir."
        ),
        takeaway=(
            "Araştırma sorusu \"eğitim katsayısı kaçtır?\" değildir. Önce hangi anakütle "
            "nesnesini öğrenmek istediğimizi yazarız; katsayı ondan sonra gelir."
        ),
    ),
    LabStep(
        number=2,
        title="Ham veriden analiz örneklemine",
        note=NoteRef(SECTION, 2),
        explanation=(
            "Hansen'in `cps09mar` dosyası tam zamanlı çalışan (haftada en az 36 saat, yılda en az "
            "48 hafta), askerî olmayan ve tahsis edilmemiş 50.742 kişiyi içerir. Analiz için dört "
            "değişken türetilir: saatlik ücret, log saatlik ücret, potansiyel deneyim ve deneyimin "
            "karesinin 100'e bölümü."
        ),
        operations=(
            LoadHansen("cps09mar", "cps09mar.txt", FRAME, CPS09MAR_COLUMNS),
            Derive(
                FRAME,
                "hrwage",
                E.div(E.var("earnings"), E.mul(E.var("hours"), E.var("week"))),
                "Saatlik ücret = yıllık ücret geliri / (haftalık saat × yıllık hafta)",
            ),
            Derive(FRAME, "lwage", E.log(E.var("hrwage")), "Log saatlik ücret"),
            Derive(
                FRAME,
                "experience",
                E.maximum(E.sub(E.sub(E.var("age"), E.var("education")), 6), 0),
                "Potansiyel deneyim = yaş - eğitim - 6; negatif değerler sıfıra çekilir",
            ),
            Derive(
                FRAME,
                "experience2_100",
                E.div(E.power(E.var("experience"), 2), 100),
                "Deneyimin karesi / 100 (katsayıyı okunur ölçeğe getirir)",
            ),
        ),
        checks=(
            Check("Analiz örneklemi (N)", StatTarget(FRAME, "lwage", "count"), 50742, decimals=0),
        ),
        takeaway=(
            "Bir tezde yalnız \"CPS verisi kullanılmıştır\" demek yetmez. Hangi gözlemlerin "
            "örnekleme girdiği ve türetilmiş değişkenlerin nasıl kurulduğu kodla belgelenmelidir."
        ),
        code_note=(
            "Kod veriyi Hansen'in sayfasındaki `Econometrics Data.zip` arşivinden indirir. İnternet "
            "yoksa dosyayı kendiniz indirip betiğin başındaki yerel dosya satırına yolunu yazın."
        ),
    ),
    LabStep(
        number=3,
        title="Regresyondan önce veriyi tanımak",
        note=NoteRef(SECTION, 3, ("Tablo 1.2",)),
        explanation=(
            "Regresyondan önce temel değişkenlerin dağılımına bakarız. Ortalama saatlik ücret "
            "medyandan belirgin biçimde yüksektir: ücret dağılımı sağa çarpıktır. Log dönüşümü bu "
            "çarpıklığı azaltır ve oransal farkların yorumunu kolaylaştırır."
        ),
        operations=(
            Describe(
                FRAME,
                ("hrwage", "lwage", "education", "experience", "female", "age"),
                "betimsel",
            ),
        ),
        checks=(
            _stat("hrwage", "mean", 23.9027, "Saatlik ücret ortalaması"),
            _stat("hrwage", "sd", 20.7111, "Saatlik ücret std. sapması"),
            _stat("hrwage", "median", 19.2308, "Saatlik ücret medyanı"),
            _stat("lwage", "mean", 2.9462, "Log ücret ortalaması"),
            _stat("lwage", "sd", 0.6759, "Log ücret std. sapması"),
            _stat("lwage", "min", -7.8633, "Log ücret en küçük değeri"),
            _stat("education", "mean", 13.9246, "Eğitim ortalaması"),
            _stat("experience", "mean", 22.2082, "Deneyim ortalaması"),
            _stat("female", "mean", 0.4257, "Kadın payı"),
            _stat("age", "mean", 42.1317, "Yaş ortalaması"),
        ),
        takeaway=(
            "Kadın göstergesinin ortalaması 0,4257'dir: bir kukla değişkenin ortalaması, o grubun "
            "örneklemdeki payıdır. Log ücretin en küçük değeri (−7,86) çok düşük bir saatlik ücrete "
            "karşılık gelir; aykırı gözlemleri regresyondan önce görmek bu yüzden önemlidir."
        ),
    ),
    LabStep(
        number=4,
        title="Koşullu ortalamaları önce grafik ve tabloyla görmek",
        note=NoteRef(SECTION, 4, ("Tablo 1.3", "Şekil 1.6")),
        explanation=(
            "Regresyondan önce grup örüntülerini görünür kılarız. Erkekler ve kadınlar için ortalama "
            "log ücret 3,0459 ve 2,8116'dır; bu değerler Hansen'in raporladığı 3,05 ve 2,81'i yeniden "
            "üretir. Eğitim arttıkça ortalama log ücret yükselir, ancak ilişki tam doğrusal değildir."
        ),
        operations=(
            GroupSummary(
                FRAME,
                "education",
                (
                    ("N", "lwage", "count"),
                    ("ort_lwage", "lwage", "mean"),
                    ("ort_ucret", "hrwage", "mean"),
                    ("medyan_ucret", "hrwage", "median"),
                ),
                "kosullu_ortalama",
            ),
            GroupMeanPlot(
                FRAME,
                "education",
                "lwage",
                "female",
                ((0, "Erkek"), (1, "Kadın")),
                "Eğitim yılı",
                "Ortalama log saatlik ücret",
                "Eğitim düzeyine ve cinsiyete göre ortalama log saatlik ücret",
            ),
        ),
        checks=(
            _stat("lwage", "mean", 3.0459, "Erkeklerde ortalama log ücret", where=("female", 0)),
            _stat("lwage", "mean", 2.8116, "Kadınlarda ortalama log ücret", where=("female", 1)),
            _stat("lwage", "count", 13896, "Eğitim=12 gözlem sayısı", where=("education", 12), decimals=0),
            _stat("lwage", "mean", 2.7121, "Eğitim=12 ortalama log ücret", where=("education", 12)),
            _stat("lwage", "mean", 3.2011, "Eğitim=16 ortalama log ücret", where=("education", 16)),
            _stat("hrwage", "mean", 30.1216, "Eğitim=16 ortalama ücret", where=("education", 16)),
            _stat("hrwage", "median", 24.0385, "Eğitim=16 medyan ücret", where=("education", 16)),
            _stat("lwage", "mean", 3.6890, "Eğitim=20 ortalama log ücret", where=("education", 20)),
        ),
        takeaway="Bu sayılar grup ortalamalarıdır, nedensel eğitim etkileri değildir.",
    ),
    LabStep(
        number=5,
        title="Koşullu ortalamadan doğrusal projeksiyona",
        note=NoteRef(SECTION, 5, ("Şekil 1.7",)),
        explanation=(
            "Yalnız eğitimi kullanan doğrusal projeksiyon "
            "$\\log(wage_i)=\\beta_0+\\beta_1 education_i+e_i$ tahmin edilir. Tahmin edilen denklem "
            "$\\widehat{\\log(wage_i)}=1{,}4396+0{,}1082\\,education_i$ olur. Nokta büyüklükleri o "
            "eğitim kategorisindeki gözlem sayısıyla orantılıdır."
        ),
        operations=(
            OLS("m1", FRAME, "lwage", M1),
            ProjectionPlot(
                FRAME,
                "education",
                "lwage",
                "m1",
                "Eğitim yılı",
                "Ortalama log saatlik ücret",
                "Eğitim düzeyine göre koşullu ortalama ve OLS doğrusal projeksiyonu",
            ),
        ),
        checks=(
            _coef("m1", "education", 0.1082, "Model (1) eğitim katsayısı"),
            _coef("m1", E.INTERCEPT, 1.4396, "Model (1) sabit terim"),
        ),
        takeaway=(
            "OLS doğrusu bütün koşullu ortalamalardan geçmek zorunda değildir. Koşullu ortalama "
            "doğrusal değilse OLS, onun en iyi doğrusal yaklaşımını (projeksiyonunu) verir."
        ),
    ),
    LabStep(
        number=6,
        title="Bir makaledeki çok sütunlu regresyon tablosunu okumak",
        note=NoteRef(SECTION, 6, ("Tablo 1.4", "Şekil 1.8")),
        explanation=(
            "Aynı araştırma sorusu üç spesifikasyonla yan yana raporlanır: (1) yalnız eğitim, "
            "(2) deneyim profili eklenmiş, (3) kadın göstergesi de eklenmiş. Tablo satır satır değil, "
            "sütunlar arasında katsayının nasıl değiştiğine bakılarak okunur."
        ),
        operations=(
            OLS("m2", FRAME, "lwage", M2),
            OLS("m3", FRAME, "lwage", M3),
            RegressionTable(
                ("m1", "m2", "m3"),
                ("education", "experience", "experience2_100", "female", E.INTERCEPT),
                "regresyon_tablosu",
            ),
            Scalar(
                "egitim_yuzde",
                E.mul(100, E.sub(E.exp(E.coef("m3", "education")), 1)),
                "Model (3) eğitim katsayısının kesin yüzde karşılığı",
            ),
            Scalar(
                "kadin_yuzde",
                E.mul(100, E.sub(E.exp(E.coef("m3", "female")), 1)),
                "Model (3) kadın katsayısının kesin yüzde karşılığı",
            ),
        ),
        checks=(
            _coef("m2", "education", 0.1126, "Model (2) eğitim"),
            _coef("m2", "experience", 0.0363, "Model (2) deneyim"),
            _coef("m2", "experience2_100", -0.0578, "Model (2) deneyim²/100"),
            _coef("m3", "education", 0.1148, "Model (3) eğitim"),
            _coef("m3", "experience", 0.0356, "Model (3) deneyim"),
            _coef("m3", "experience2_100", -0.0563, "Model (3) deneyim²/100"),
            _coef("m3", "female", -0.2596, "Model (3) kadın"),
            _coef("m3", E.INTERCEPT, 1.0211, "Model (3) sabit terim"),
            Check("Model (1) R²", ModelTarget("m1", "r2"), 0.1930),
            Check("Model (2) R²", ModelTarget("m2", "r2"), 0.2368),
            Check("Model (3) R²", ModelTarget("m3", "r2"), 0.2728),
            Check("Eğitim: kesin yüzde etki", ScalarTarget("egitim_yuzde"), 12.16, decimals=2),
            Check("Kadın: kesin yüzde fark", ScalarTarget("kadin_yuzde"), -22.87, decimals=2),
        ),
        takeaway=(
            "Model (3)'te eğitim katsayısı 0,1148: eğitim yılı bir birim daha yüksek gözlemlerin "
            "koşullu ortalama log ücreti 0,1148 daha yüksektir; bu, ücrette yaklaşık %11,5, kesin dönüşümle "
            "%12,16 daha yüksek değere karşılık gelir. \"Bir kişiye "
            "bir yıl daha eğitim verirsek ücreti %12 artar\" demiyoruz: model eğitimde dışsal bir "
            "müdahaleyi tanımlamıyor. Daha yüksek R² fonksiyonel biçimin doğruluğunu, nedenselliği "
            "veya daha iyi öngörüyü tek başına kanıtlamaz."
        ),
    ),
    LabStep(
        number=7,
        title="Yazılım çıktısından makale tablosuna geçmek",
        note=NoteRef(SECTION, 7),
        explanation=(
            "Makale tablosu ile yazılım çıktısı farklı nesneler değildir: tablo, çıktının araştırma "
            "sorusu için önemli kısmının okunur biçimidir. `education` satırında katsayı, standart "
            "hata, $t$ oranı (katsayı / standart hata), $p$-değeri ve %95 güven aralığı yer alır."
        ),
        operations=(ShowModel("m3"),),
        checks=(
            _coef("m3", "education", 0.0010, "Eğitim standart hatası", "se"),
            _coef("m3", "experience", 0.0008, "Deneyim standart hatası", "se"),
            _coef("m3", "experience2_100", 0.0016, "Deneyim²/100 standart hatası", "se"),
            _coef("m3", "female", 0.0052, "Kadın standart hatası", "se"),
            _coef("m3", E.INTERCEPT, 0.0162, "Sabit terim standart hatası", "se"),
        ),
        takeaway=(
            "Bu bölümdeki standart hatalar klasik OLS standart hatalarıdır ve yalnız tablo okuma "
            "pratiği içindir. Güvenilir çıkarım ve heteroskedastisiteye dayanıklı standart hatalar "
            "Konu 2'de ele alınır."
        ),
    ),
    LabStep(
        number=8,
        title="Sonuçları ampirik metne dönüştürmek",
        note=NoteRef(SECTION, 8),
        explanation=(
            "Regresyon tablosunun amacı sayıları tekrar etmek değil, araştırma sorusuna kontrollü bir "
            "cevap vermektir. İyi bir paragraf dört şey yapar: veri ve örneklemi belirtir, katsayının "
            "spesifikasyonlar arasında nasıl değiştiğini söyler, büyüklüğü ekonomik birime çevirir ve "
            "modelin izin verdiği yorum sınırını korur."
        ),
        takeaway=(
            "Mart 2009 CPS tam zamanlı çalışan örnekleminde eğitim ile log saatlik ücret arasında "
            "pozitif bir ilişki bulunmaktadır. Yalnız eğitim değişkenini içeren spesifikasyonda eğitim "
            "katsayısı 0,1082'dir. Potansiyel deneyim ve deneyimin karesi eklendiğinde katsayı "
            "0,1126'ya, kadın göstergesi de kontrol edildiğinde 0,1148'e yükselmektedir. Bu ilişki, "
            "eğitim kararının içselliğini ele alan bir tanımlama stratejisi kurulmadığı için nedensel "
            "eğitim getirisi olarak yorumlanmamalıdır."
        ),
    ),
    LabStep(
        number=9,
        title="Aynı analizi kendiniz nasıl kurarsınız?",
        note=NoteRef(SECTION, 9),
        explanation=(
            "Tez için başlangıç protokolü: araştırma sorusunu yazın; tahmin hedefini belirleyin; ham "
            "veriyi koruyun ve temizliği kodla yapın; analiz örneklemini ve türetilmiş değişkenleri "
            "belgeleyin; önce betimleyin, sonra tahmin edin; sonuçları yorum sınırıyla birlikte yazın.\n\n"
            "Aşağıdan bütün laboratuvarı seçtiğiniz dilde tek dosya olarak indirebilirsiniz. Betik "
            "sonunda notlardaki sayılarla karşılaştırma yapar; bir sayı tutmazsa hangisinin "
            "tutmadığını söyleyerek durur."
        ),
    ),
)


KONU01_LAB = LabSpec(
    topic_key="konu01",
    title="CPS Ücret–Eğitim İlişkisi",
    dataset="cps09mar",
    note_section=SECTION,
    steps=STEPS,
    labels=(
        ("hrwage", "Saatlik ücret"),
        ("lwage", "Log saatlik ücret"),
        ("education", "Eğitim yılı"),
        ("experience", "Potansiyel deneyim"),
        ("experience2_100", "Deneyim²/100"),
        ("female", "Kadın göstergesi"),
        ("age", "Yaş"),
        (E.INTERCEPT, "Sabit"),
        ("N", "N"),
        ("ort_lwage", "Ortalama log ücret"),
        ("ort_ucret", "Ortalama ücret"),
        ("medyan_ucret", "Medyan ücret"),
    ),
    consistency_notes=(
        "Notlar potansiyel deneyimi sıfırın altında kırpar (44 gözlem etkilenir) ve bunu kitabın veri "
        "hazırlama betiğine atfeder. Hansen'in kitap metnindeki tanım ve Stata örneği kırpmasızdır "
        "(age − education − 6). Laboratuvar notları izler. Kırpma kaldırılırsa Tablo 1.4'te dört hücre "
        "dördüncü ondalıkta değişir: Model (2) sabit 0,9368→0,9372, deneyim 0,0363→0,0362, "
        "deneyim²/100 −0,0578→−0,0577; Model (3) deneyim²/100 −0,0563→−0,0562.",
    ),
)
