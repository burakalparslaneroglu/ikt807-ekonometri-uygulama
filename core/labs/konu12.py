"""Konu 12 uygulama laboratuvarı: DML'i tanımlama stratejisinin yerine koymamak.

Ders notları §12.15'in (Adım 1–4 ve §12.15.5) birebir karşılığıdır. Veri: Hansen'in DDK2011 dosyası (Duflo, Dupas
ve Kremer 2011, Kenya tracking deneyi); eksiksiz kovaryat örneklemi 108 okulda 5.135 öğrenci. Tracking okul
düzeyinde rastgele atanmıştır: nedensel tanımlama tasarımdan gelir, DML'den değil.

DML: kısmen doğrusal model, yardımcı modeller 27 terimli bir sözlük üzerinde Lasso. Okullar dış katlara açık bir
kuralla atanır: okul sıra numarası r için ⌊5{rφ}⌋ + 1; Lasso cezası her eğitim katında okul düzeyindeki iç katlarla
(⌊5{r√2}⌋ + 1) CV'de seçilir. Standart hata okul düzeyinde kümelenmiştir. Bölme duyarlılığı: dış katlarda φ yerine
√3, …, √31 çarpanlı on kural daha; medyan birleştirme (Chernozhukov vd., 2018).
"""

from __future__ import annotations

import math

from core.labs import expr as E
from core.labs.penalized import GOLDEN
from core.labs.spec import (
    OLS,
    THETA,
    Check,
    CoefTarget,
    CrossFitDML,
    Derive,
    Dictionary,
    DMLSplits,
    DropMissing,
    EffectTable,
    EstimatePlot,
    GroupRank,
    GroupSummary,
    LabSpec,
    LabStep,
    LoadHansen,
    NoteRef,
    ReproClass,
    ScalarTable,
    ScalarTarget,
    StatTarget,
    dictionary_terms,
    dml_fold_key,
)

SECTION = "12.15"
FRAME = "ddk"
Y = "totalscore"
D = "tracking"
CLUSTER = "schoolid"
COVARIATES = ("std_mark", "girl", "agetest", "sbm", "etpteacher", "percentile")
BASE = ("std_mark", "girl", "yas", "sbm", "etpteacher", "yuzdelik")
DICTIONARY = Dictionary(FRAME, BASE, ("std_mark", "yas", "yuzdelik"), 3, True,
                        "Aday terim sözlüğü: altı değişken, üç sürekli değişkenin 2. ve 3. kuvvetleri, 15 ikili etkileşim")
TERMS = dictionary_terms(DICTIONARY)
GRID = (1.0, -3.0, 41)
FOLDS = 5
SQRT2 = math.sqrt(2.0)
RULES = (("phi", "φ", GOLDEN),) + tuple((f"kok{p}", f"√{p}", math.sqrt(float(p))) for p in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31))
ROWS = (
    ("Ham fark, okul-küme SH", "ham", D),
    ("Kovaryat ayarlı OLS, okul-küme SH", "ayarli", D),
    ("DML, Lasso, okul-kümeli çapraz uyarlama", "dml", THETA),
    ("DML, 11 kat kuralının medyanı", "bolmeler", THETA),
)


def _fold(multiplier: float) -> E.Expr:
    """⌊5{r·c}⌋ + 1."""

    scaled = E.mul(E.var("okul_sira"), multiplier)
    return E.add(E.floor(E.mul(FOLDS, E.sub(scaled, E.floor(scaled)))), 1)


def _coef(model: str, term: str, expected: float, label: str, quantity: str = "coef") -> Check:
    return Check(label, CoefTarget(model, term, quantity), expected, decimals=3)


STEPS = (
    LabStep(
        number=1,
        title="Aynı estimand için üç tahmin yolu",
        note=NoteRef(SECTION, 1, ("Tablo 12.2", "Şekil 12.4")),
        explanation=(
            "DML'in en iyi öğretilebileceği bağlamlardan biri tanımlamanın zaten açık olduğu bir veri setidir: Hansen'in "
            "Duflo–Dupas–Kremer tracking deneyi. Tracking okul düzeyinde rastgele atanmıştır; nedensel tanımlama makine "
            "öğrenmesinden gelmez. DML burada esnek kovaryat ayarlaması ve çapraz uyarlama mantığını öğretmek için "
            "kullanılır.\n\n"
            "Eksiksiz kovaryat örnekleminde 108 okuldan 5.135 öğrenci vardır. Yardımcı modeller $m_Y(X)=E[Y\\mid X]$ ve "
            "$m_D(X)=E[D\\mid X]$ Lasso'dur ve 27 terimli bir sözlük kullanır: altı değişkenin kendisi (başlangıç standart "
            "puanı, cinsiyet, test yaşının 9'dan sapması, okul yönetim komitesi, ek öğretmen, "
            "$(\\text{yüzdelik}-50)/25$), üç sürekli değişkenin ikinci ve üçüncü kuvvetleri ve altı değişkenin 15 ikili "
            "etkileşimi. Hedef katsayı artıkların artıklar üzerine sabitsiz regresyonudur:\n\n"
            "$$\\widehat\\theta=\\frac{\\sum_i\\widehat V_i\\widehat U_i}{\\sum_i\\widehat V_i^2},\\qquad "
            "\\widehat U_i=Y_i-\\widehat m_Y^{(-k)}(X_i),\\quad \\widehat V_i=D_i-\\widehat m_D^{(-k)}(X_i).$$\n\n"
            "Dört satırın standart hataları okul düzeyinde kümelenmiştir. Son satır, dış katları 11 farklı kuralla kuran "
            "bölmelerin medyanıdır (Adım 3)."
        ),
        operations=(
            LoadHansen("ddk2011", "DDK2011.dta", FRAME),
            DropMissing(FRAME, (Y, D, CLUSTER) + COVARIATES,
                        "Eksiksiz kovaryat örneklemi: sonuç, tedavi, okul veya kovaryatı eksik gözlemler çıkarılır"),
            Derive(FRAME, "yas", E.sub(E.var("agetest"), 9), "Test yaşının 9'dan sapması"),
            Derive(FRAME, "yuzdelik", E.div(E.sub(E.var("percentile"), 50), 25),
                   "Başlangıç yüzdelik dilimi, (yüzdelik − 50)/25 olarak ölçeklenmiş"),
            DICTIONARY,
            GroupRank(FRAME, "okul_sira", CLUSTER, "Okul sıra numarası r = 1, …, 108 (okul numarasına göre)"),
            Derive(FRAME, "dis_kat", _fold(GOLDEN), "DML dış katı: ⌊5{rφ}⌋ + 1, φ = (√5 − 1)/2; bütün okul aynı katta"),
            Derive(FRAME, "ic_kat", _fold(SQRT2), "Lasso cezası için iç kat: ⌊5{r√2}⌋ + 1"),
            OLS("ham", FRAME, Y, (D,), vcov="cluster", cluster=CLUSTER),
            OLS("ayarli", FRAME, Y, (D, "std_mark", "girl", "agetest", "sbm", "etpteacher", "percentile"),
                vcov="cluster", cluster=CLUSTER),
            CrossFitDML("dml", FRAME, Y, D, TERMS, "dis_kat", "ic_kat", GRID, "min", CLUSTER),
            DMLSplits("bolmeler", "dml", "okul_sira", RULES, FOLDS, "bolme_tablosu"),
            EffectTable(ROWS, "karsilastirma", "DDK2011: tracking etkisi için OLS ve DML karşılaştırması"),
            EstimatePlot(ROWS, "tahminler", "Tracking etkisi (test puanı)",
                         "DDK2011: aynı araştırma sorusu, farklı tahmin yolları", splits="bolme_tablosu", splits_row=3),
        ),
        checks=(
            Check("Eksiksiz kovaryat örneklemi (N)", StatTarget(FRAME, Y, "count"), 5135, decimals=0),
            Check("Okul sayısı", StatTarget(FRAME, "okul_sira", "max"), 108, decimals=0),
            _coef("ham", D, 1.134, "Ham fark"),
            _coef("ham", D, 0.708, "Ham fark, okul-küme SH", "se"),
            _coef("ayarli", D, 1.383, "Kovaryat ayarlı OLS"),
            _coef("ayarli", D, 0.699, "Kovaryat ayarlı OLS, okul-küme SH", "se"),
            _coef("dml", THETA, 1.497, "DML (Lasso, φ kuralı)"),
            _coef("dml", THETA, 0.704, "DML, okul-küme SH", "se"),
            _coef("bolmeler", THETA, 1.464, "DML, 11 bölmenin medyanı"),
            _coef("bolmeler", THETA, 0.705, "DML medyan SH", "se"),
            Check("D için Lasso: sıfırdan farklı katsayı (en çok)", ScalarTarget(dml_fold_key("dml", "d", "max")), 0,
                  decimals=0),
            Check("Y için Lasso: sıfırdan farklı katsayı (en az)", ScalarTarget(dml_fold_key("dml", "y", "min")), 17,
                  decimals=0),
            Check("Y için Lasso: sıfırdan farklı katsayı (en çok)", ScalarTarget(dml_fold_key("dml", "y", "max")), 26,
                  decimals=0),
        ),
        reproducibility=ReproClass.CONVENTION,
        takeaway=(
            "Üç yöntem aynı büyüklük bölgesindedir: ayarlı OLS 1,38, DML 1,50 test puanı; standart hatalar neredeyse "
            "aynıdır (0,699 ve 0,704). Fark standart hatanın beşte birinden azdır; bu tek uygulamadan \"DML daha "
            "hassastır\" ya da \"daha az hassastır\" sonucu çıkarılamaz. Tasarımın izi yardımcı modellerde de görünür: "
            "tracking okul düzeyinde rastgele atandığı için iç CV bütün katlarda D için sabit modeli seçer (sıfırdan farklı "
            "katsayı yok); V̂ tracking'in eğitim katındaki ortalamasından sapmasıdır. Y için Lasso 27 terimden 17 ile 26 "
            "arasını tutar."
        ),
        code_note=(
            "Lasso yazılım ölçeğinde, $\\lambda_y\\in[10^{-3},10]$ aralığında 41 noktalı logaritmik ızgarada iç CV "
            "hatasını en küçük yapan cezadır; özellikler her eğitim katında ölçeklenir. Python `enet_path`, R `glmnet`, "
            "Stata kodda yazılı Mata koordinat inişi aynı optimuma yakınsar. scikit-learn'deki `GroupKFold` okulları "
            "büyüklüklerine göre dengeler; eşit büyüklükteki okulların hangi kata gideceği kütüphanenin sıralama "
            "ayrıntısına bağlıdır ve R ile Stata aynı katları kuramaz. Açık kural üç dilde aynı katları verir. 11 bölmenin "
            "her biri 5 × 2 yardımcı model ve her birinde 5 katlı iç CV demektir; hesap birkaç saniye sürer."
        ),
    ),
    LabStep(
        number=2,
        title="Çapraz uyarlamayı okul kümelerine saygılı yapmak",
        note=NoteRef(SECTION, 2),
        explanation=(
            "Tedavi okul düzeyinde atandığı için rastgele kat oluşturup aynı okulun bazı öğrencilerini eğitim, bazılarını "
            "değerlendirme katına koymak uygun değildir. Okullar beş kata açık bir kuralla atanır: okulun numarasına göre "
            "sıra numarası $r=1,\\ldots,108$ olmak üzere\n\n"
            "$$\\text{kat}=\\lfloor5\\{r\\varphi\\}\\rfloor+1,\\qquad \\varphi=\\frac{\\sqrt5-1}{2}.$$\n\n"
            "Böylece bütün okul aynı kat içinde kalır ve yardımcı model aynı okulun değerlendirme örneklemindeki "
            "öğrencileri hakkında doğrudan eğitim görmez. Her eğitim katında Lasso cezası da okul düzeyinde katlarla "
            "seçilir: iç kat $\\lfloor5\\{r\\sqrt2\\}\\rfloor+1$'dir. Aşağıdaki tablo dış katların büyüklüğünü ve her "
            "kattaki tracking payını gösterir."
        ),
        operations=(
            GroupSummary(FRAME, "dis_kat", (("ogrenci", Y, "count"), ("tracking_payi", D, "mean")), "dis_katlar"),
        ),
        reproducibility=ReproClass.EXACT,
        takeaway=(
            "Bu kod iki farklı bağımlılık sorununu ayrı yerlerde ele alır. Okul-kümeli katlar veri sızıntısını azaltır; son "
            "aşamadaki cluster-robust kovaryans okul içi hata bağımlılığını çıkarıma taşır. Birini yapmak diğerinin yerini "
            "tutmaz."
        ),
    ),
    LabStep(
        number=3,
        title="DML makalesindeki sonuç tablosunu nasıl okumalıyız?",
        note=NoteRef(SECTION, 3),
        explanation=(
            "DML çıktısında yalnız hedef katsayı ve p-değerini görmek yeterli değildir. En az şu ayrıntılar "
            "raporlanmalıdır:\n\n"
            "1. hedef parametre ve tanımlama varsayımı,\n"
            "2. treatment ve outcome tanımları,\n"
            "3. kullanılan yardımcı model öğrenicileri,\n"
            "4. tuning prosedürü,\n"
            "5. kat sayısı ve bölme birimi,\n"
            "6. ön işlemenin kat içinde yapılıp yapılmadığı,\n"
            "7. hedef katsayı için standart hata türü,\n"
            "8. farklı öğrenici, kat ve seed seçimlerinde duyarlılık.\n\n"
            "Son maddenin önemi bu veride açıkça görülür. Aynı hesabı dış katlarda $\\varphi$ yerine 3 ile 31 arasındaki "
            "on asal sayının kareköküyle kurulan on farklı kuralla tekrarladığımızda tahmin değişir; yalnız okulların "
            "katlara dağılımı değişmiştir, veri, öğrenici ve kat sayısı aynıdır. On bir bölme Chernozhukov vd. (2018) "
            "gibi birleştirilir:\n\n"
            "$$\\widehat\\theta_{med}=\\operatorname{medyan}_s\\widehat\\theta_s,\\qquad "
            "\\widehat V_{\\widehat\\theta,med}=\\operatorname{medyan}_s\\{\\widehat V_{\\widehat\\theta,s}+(\\widehat\\theta_s-"
            "\\widehat\\theta_{med})^2\\}.$$"
        ),
        operations=(
            ScalarTable(
                (
                    ("En küçük θ̂ (11 bölme)", E.ref("bolmeler_min")),
                    ("En büyük θ̂ (11 bölme)", E.ref("bolmeler_max")),
                    ("Medyan θ̂", E.coef("bolmeler", THETA)),
                    ("Medyan SH", E.se("bolmeler", THETA)),
                ),
                "bolme_ozeti",
            ),
        ),
        checks=(
            Check("11 bölmede en küçük tahmin", ScalarTarget("bolmeler_min"), 1.267, decimals=3),
            Check("11 bölmede en büyük tahmin", ScalarTarget("bolmeler_max"), 1.554, decimals=3),
        ),
        reproducibility=ReproClass.CONVENTION,
        takeaway=(
            "Tahmin 1,267 ile 1,554 arasında değişir; medyan 1,464, medyan standart hatası 0,705. Tek bir bölmenin "
            "sonucunu \"en iyi\" bölme olarak seçmek yerine bölme kuralı önceden kaydedilmeli ve bölmeler arası değişim "
            "raporlanmalıdır. \"Lasso kullanıldı\" ifadesi yöntem tanımı değildir: aday terim sözlüğü, ceza ızgarası, ceza "
            "seçim kuralı ve cross-fitting yapısı yeniden üretilebilir biçimde verilmelidir."
        ),
    ),
    LabStep(
        number=4,
        title="Benzer OLS ve DML sonuçlarını nasıl yorumlamalıyız?",
        note=NoteRef(SECTION, 4),
        explanation=(
            "Bu örnekte ayarlı OLS ile DML nokta tahminleri birbirine yakındır. Bunun iki olası okuması vardır. "
            "Birincisi, doğrusal kontrol spesifikasyonu bu veri için yeterli olabilir. İkincisi, randomized design "
            "nedeniyle kovaryat fonksiyonel biçiminin hedef katsayı üzerindeki etkisi zaten sınırlı olabilir. Her iki "
            "durumda da yakınlık, gözlenmeyen confounding olmadığına ilişkin yeni bir test değildir; randomization "
            "tasarımından gelen bilgiyle uyumludur.\n\n"
            "Gözlemsel bir tezde OLS ve DML'in yakın çıkması \"içsellik yoktur\" anlamına gelmez. Her ikisi de yalnız "
            "gözlenen $X$ üzerinde koşullandırıyorsa gözlenmeyen karıştırıcı problemi ortak kalabilir."
        ),
        takeaway=(
            "DML, karmaşık fonksiyonel biçimi ve yüksek boyutlu gözlenen kontrolleri yönetmek için güçlüdür; fakat geçerli "
            "araç, randomization, unconfoundedness veya başka bir tanımlama kaynağının yerine geçmez. Örnek sonuç "
            "paragrafı: \"Tracking etkisi kovaryat ayarlı OLS'de 1,38, okul-kümeli cross-fitting kullanan Lasso-DML'de 1,50 "
            "olarak tahmin edilmiştir; on bir farklı kat kuralında DML tahminleri 1,27 ile 1,55 arasında değişmiş, "
            "medyanları 1,46 olmuştur. Nedensel yorum DML'den değil, programın okul düzeyindeki rastgele atama tasarımından "
            "gelmektedir.\""
        ),
    ),
    LabStep(
        number=5,
        title="Kendi tezinizde DML için uygulama protokolü",
        note=NoteRef("12.15.5", 0),
        explanation=(
            "1. Önce estimand ve tanımlama varsayımını düz yazıyla açıklayın.\n"
            "2. Hangi değişkenlerin treatment sonrası olduğunu kontrol edin; post-treatment kontrolleri nuisance setine "
            "mekanik biçimde eklemeyin.\n"
            "3. Fold yapısını bağımlılık birimine göre kurun.\n"
            "4. Ön işleme ve tuning'i her eğitim katı içinde yapın.\n"
            "5. En az iki makul yardımcı model öğrenicisi ile duyarlılık düşünün.\n"
            "6. Hedef katsayı için veri yapısına uygun robust/küme standart hata kullanın.\n"
            "7. DML sonucunu basit benchmark modellerle yan yana raporlayın.\n"
            "8. Replikasyon paketinde kat kuralını veya seed'i, öğrenici ayarlarını ve paket sürümlerini saklayın; birkaç "
            "bölme altında duyarlılığı raporlayın."
        ),
        takeaway=(
            "Bu laboratuvarın bütün seçimleri kodda açıktır: sözlük, ızgara, iç ve dış kat kuralları, okul-küme standart "
            "hatası ve 11 bölmelik duyarlılık. Aynı kod Python, R ve Stata'da aynı sayıları verir."
        ),
    ),
)


KONU12_LAB = LabSpec(
    topic_key="konu12",
    title="DML'i Tanımlama Stratejisinin Yerine Koymamak",
    dataset="ddk2011",
    note_section=SECTION,
    steps=STEPS,
    labels=(
        (Y, "Toplam test puanı"), (D, "Tracking"), ("std_mark", "Başlangıç standart puanı"), ("girl", "Kız"),
        ("agetest", "Test yaşı"), ("sbm", "Okul yönetim komitesi"), ("etpteacher", "Ek öğretmen"),
        ("percentile", "Başlangıç yüzdelik dilimi"), ("yas", "Test yaşı − 9"), ("yuzdelik", "(Yüzdelik − 50)/25"),
        ("okul_sira", "Okul sıra numarası"), ("dis_kat", "Dış kat"), ("ic_kat", "İç kat"),
        ("ham", "Ham fark"), ("ayarli", "Kovaryat ayarlı OLS"), ("dml", "DML (φ kuralı)"),
        ("bolmeler", "DML, 11 bölmenin medyanı"), (THETA, "θ (tracking)"), ("ogrenci", "Öğrenci"),
        ("tracking_payi", "Tracking payı"),
    ),
    consistency_notes=(
        "Notlardaki DML satırı rassal orman yardımcı modelleri ve scikit-learn GroupKFold ile hesaplanmıştı (1,397; SH "
        "0,654); ağaçlar ve katlar R ile Stata'da aynı biçimde kurulamıyordu. DML, 27 terimli sözlük üzerinde Lasso "
        "yardımcı modelleriyle ve okulları açık bir kuralla katlara ayırarak yeniden kuruldu (1,497; SH 0,704); 11 kat "
        "kuralı altında duyarlılık ve medyan birleştirme eklendi. Notlar ve sunum güncellendi.",
    ),
)
