"""Konu 6 "Kendini sına" soru seti: sansürleme, kesilme ve örneklem seçimi.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Bölüm sonu egzersizlerindeki
hesaplar (sansürlenme olasılığı, üç koşullu ortalama, Greene oranı, olabilirlik katkıları, Tobit
gözlenen ortalama etkisi, ters Mills oranı) tekrar edilmez; buradaki sorular farklı yönleri sınar.
"""

from __future__ import annotations

from core.labs.spec import NoteRef
from core.quiz.expression import Symbol
from core.quiz.model import (
    Equation,
    FillBlanks,
    MultipleChoice,
    NumberBlank,
    Question,
    QuestionSet,
    TextBlank,
    TrueFalse,
)


def _note(section: str, step: int = 0, *objects: str) -> NoteRef:
    return NoteRef(section, step, tuple(objects))


QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="sansurde-bilinen", note=_note("6.1"),
        prompt="Soldan sıfırda sansürlü bir veri setinde, sınırdaki ($Y=0$) bir birim hakkında ne biliriz?",
        answer=MultipleChoice((
            "Hiçbir şey; bu birimlerin veri setinde satırı yoktur",
            "Gizli $Y^*$ değerinin kendisini",
            "Birimin varlığını ve $X$ değerlerini; $Y^*$'ın eşiğin altında kaldığını",
            "Yalnız bir seçim göstergesini",
        ), correct=2),
        explanation=(
            "Sansürlemede birim ve açıklayıcıları gözlenir, yalnız $Y^*$'ın tam değeri gözlenmez. Kesilmede (truncation) "
            "birimin satırı hiç yoktur; seçimde $Y$ yalnız $S=1$ iken gözlenir (§6.1)."
        ),
    ),
    Question(
        key="k02", concept="tam-orneklem-ols-hedefi", note=_note("6.3"),
        prompt="Sansürlü veride tam örneklem OLS genel olarak hangi fonksiyonun en iyi doğrusal yaklaşımını hedefler?",
        answer=MultipleChoice((
            "Gizli ortalama $m^*(x)=x'\\beta$",
            "Gözlenen sansürlü sonucun ortalaması $m(x)=\\mathbb E[Y\\mid X=x]$",
            "Pozitif gözlemlerin ortalaması $m^{\\#}(x)$",
            "Sansürlenmeme olasılığı $P(Y>0\\mid X=x)$",
        ), correct=1),
        explanation=(
            "OLS gözlenen $Y$'nin koşullu ortalamasına bir doğru uydurur; bu $m(x)$'tir ve $x'\\beta$ ile aynı nesne "
            "değildir. Yalnız pozitif gözlemlerle OLS ise $m^{\\#}(x)$'i hedefler (§6.3)."
        ),
    ),
    Question(
        key="k03", concept="tobit-sansurlu-katki", note=_note("6.5"),
        prompt="Tobit olabilirliğinde $Y_i=0$ olan bir gözlemin katkısı nedir?",
        answer=MultipleChoice((
            "Normal yoğunluk $\\sigma^{-1}\\phi((0-x_i'\\beta)/\\sigma)$",
            "Hiçbir katkısı yoktur; bu gözlemler atılır",
            "Her zaman 1",
            "Sansürlenme olasılığı $\\Phi(-x_i'\\beta/\\sigma)$",
        ), correct=3),
        explanation=(
            "Sansürlü gözlem yalnız $Y_i^*\\le0$ bilgisini taşır; katkısı bir yoğunluk değil olasılıktır. Pozitif "
            "gözlemler ise normal yoğunlukla katkı yapar (§6.5)."
        ),
    ),
    Question(
        key="k04", concept="tobit-varsayimlar-baglar", note=_note("6.7"),
        prompt="Klasik Tobit'in normallik ve homoskedastisite varsayımları neyi birbirine bağlar?",
        answer=MultipleChoice((
            "Sansürlenme olasılığını ve pozitif gözlemlerin koşullu dağılımını aynı parametre setine",
            "Seçim denklemini ve sonuç denklemini",
            "OLS ve LAD tahminlerini",
            "Standart hataları ve katsayıları",
        ), correct=0),
        explanation=(
            "Aynı $\\beta$ hem sıfırda olup olmamayı hem pozitif sonucun büyüklüğünü yönetir. İki mekanizma farklıysa "
            "(katılım ve miktar) tek denklemli Tobit aşırı kısıtlayıcıdır; iki parçalı modeller düşünülür (§6.7)."
        ),
    ),
    Question(
        key="k05", concept="mills-kontrol-fonksiyonu", note=_note("6.11"),
        prompt="Heckman iki aşamasında ikinci aşamaya eklenen $\\hat\\lambda_i$ ne işlev görür?",
        answer=MultipleChoice((
            "Seçilmeyen bireylerin $Y$ değerlerini doldurur (imputasyon)",
            "Seçilmiş örneklemde hata teriminin sıfır olmayan koşullu ortalamasını temsil eden kontrol fonksiyonudur",
            "Standart hataları üretilmiş regresör için düzeltir",
            "Seçim denkleminin gücünü ölçer",
        ), correct=1),
        explanation=(
            "$\\mathbb E[e\\mid X,Z,S=1]=\\sigma_{eu}\\lambda(Z'\\gamma)$: $\\hat\\lambda$ eksik değişkeni geri koyar. "
            "Bireysel imputasyon yapmaz; standart hataları da kendiliğinden düzeltmez (§6.11)."
        ),
    ),
    Question(
        key="k06", concept="dislama-yoksa-kirilganlik", note=_note("6.12"),
        prompt="Seçim denkleminde sonuçtan dışlanan hiçbir değişken yoksa Heckman modeli için ne söylenebilir?",
        answer=MultipleChoice((
            "Hiç tahmin edilemez",
            "Tahmini OLS ile birebir aynıdır",
            "Yalnız normal dağılımın doğrusal olmayan biçiminden tanımlanır; $\\hat\\lambda$ ile $X$ yüksek ilişkili olur ve "
            "sonuçlar kırılganlaşır",
            "Standart hataları küçülür",
        ), correct=2),
        explanation=(
            "Ters Mills oranı $X$'in doğrusal olmayan bir fonksiyonudur, fakat seçilmiş örneklemde neredeyse doğrusal "
            "olabilir. Sezgi Deney 3'te $\\gamma_Z=0$ iken Heckman eğiminin dağılımı belirgin biçimde genişler (§6.12)."
        ),
    ),
    Question(
        key="k07", concept="tobit-model-kontrolu", note=_note("6.16", 3),
        prompt=(
            "CHJ uygulamasında Tobit ortalamada $P(Y>0)=0{,}579$ öngörürken verideki pozitif transfer payı 0,824. Bu "
            "karşılaştırma neyi gösterir?"
        ),
        answer=MultipleChoice((
            "Tobit katsayıları yanlış hesaplanmıştır",
            "Veri setinde kodlama hatası vardır",
            "OLS'nin doğru model olduğunu",
            "Normal-homoskedastik Tobit, sıfır kütlesini ve pozitif kısmın çarpıklığını birlikte yakalayamıyor",
        ), correct=3),
        explanation=(
            "İma edilen ortalama da (11,46) gözlenenden (7,71) büyüktür: dağılım varsayımı veriyle çelişiyor. Bu, "
            "Hansen'in CLAD gibi daha az varsayımlı yöntemleri tartışmasının gerekçesidir (§6.16, Adım 3)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="sifirlari-silmek", note=_note("6.4"),
        prompt="Sansürlü veride sıfırları silip yalnız pozitif gözlemlerle OLS yapmak gizli eğimi genel olarak geri getirir.",
        answer=TrueFalse(False),
        explanation=(
            "Pozitif olma koşulu hata teriminin dağılımını seçer: $\\mathbb E[e\\mid X, Y^*>0]$ sıfır değildir ve $X$ ile "
            "değişir. Benzetimde (Tablo 6.2) eğim 0,745'te kalır; gerçek değer 1 (§6.4)."
        ),
    ),
    Question(
        key="d02", concept="gizli-sonuc-anlamli-olmali", note=_note("6.6.1"),
        prompt=(
            "Tobit'in gizli sonuç yorumu, araştırma bağlamında negatif değer alabilen bir eğilimin ekonomik olarak anlamlı "
            "olmasını gerektirir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "$\\beta$ gizli $Y^*$'ın eğimidir. İstenen çalışma saati gibi bir eğilim savunulabiliyorsa yorum anlamlıdır; "
            "savunulamıyorsa Tobit yalnız matematiksel bir kolaylıktır (§6.6.1)."
        ),
    ),
    Question(
        key="d03", concept="ikinci-asama-sh-uretilmis-regresor", note=_note("6.11"),
        prompt=(
            "Heckman ikinci aşamasındaki sıradan OLS standart hataları, $\\hat\\lambda$'nın tahmin edilmiş bir regresör "
            "olmasını hesaba katar."
        ),
        answer=TrueFalse(False),
        explanation=(
            "$\\hat\\lambda$ birinci aşamadan gelir; sıradan standart hatalar bu belirsizliği içermez. Düzeltilmiş kovaryans "
            "veya ortak maksimum olabilirlik gerekir (§6.11)."
        ),
    ),
    Question(
        key="d04", concept="secim-yanliligi-kalici", note=_note("6.9"),
        prompt="Endojen seçimden doğan yanlılık, örneklem büyüdükçe kaybolur.",
        answer=TrueFalse(False),
        explanation=(
            "Sorun eksik gözlemlerin sayısı değil, neden eksik olduklarıdır. $\\mathbb E[e\\mid X,S=1]\\neq0$ olduğu sürece "
            "naif OLS yanlış hedefe yakınsar (§6.9)."
        ),
    ),
    Question(
        key="d05", concept="tobit-heckman-veri-ayrimi", note=_note("6.14"),
        prompt=(
            "Tobit'te $Y$ bütün birimler için gözlenir (bazıları sınır değerine dönüştürülmüştür); Heckman seçiminde ise "
            "$Y$ seçilmeyen birimler için eksiktir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "İki model benzer araçlar (normal dağılım, Probit, ters Mills oranı) kullanır ama farklı veri üretim "
            "süreçlerine karşılık gelir. Yanlış sınıflandırma yanlış olabilirlik üretir (§6.14)."
        ),
    ),
    Question(
        key="d06", concept="chj-arsiv-duzeltilmis-gelir", note=_note("6.16", 1),
        prompt=(
            "Hansen'in veri arşivindeki CHJ2004 dosyasında gelir değişkeni, transferler ve emekli maaşı çıkarılmış "
            "düzeltilmiş gelirdir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Arşivdeki dosya oluşturma adımlarının sonucudur (8.684 hane); düzeltmeyi yeniden uygulamak geliri ikinci kez "
            "azaltır. Ham `urban.dta` ile çalışılıyorsa adımlar bir kez uygulanır (§6.16, Adım 1)."
        ),
    ),
    Question(
        key="d07", concept="pozitif-ortalama-gizliden-buyuk", note=_note("6.3"),
        prompt=(
            "Tobit modelinde pozitif gözlemlerin ortalaması $m^{\\#}(x)$, gizli ortalama $m^*(x)=x'\\beta$'dan küçüktür."
        ),
        answer=TrueFalse(False),
        explanation=(
            "$m^{\\#}(x)=x'\\beta+\\sigma\\lambda(x'\\beta/\\sigma)$ ve $\\lambda>0$ olduğundan her zaman büyüktür: pozitif "
            "olma koşulu dağılımın alt kısmını keser (§6.3)."
        ),
    ),
    # --- Boşluk doldurma ---------------------------------------------------------
    Question(
        key="b01", concept="kesilme-tanimi", note=_note("6.1"),
        prompt=(
            "Sınırın dışındaki birimler veri setine hiç girmiyorsa (sayıları ve açıklayıcıları dahi gözlenmiyorsa) veri "
            "sansürlü değil, **(1)** ______ (truncated) veridir."
        ),
        answer=FillBlanks((TextBlank(
            ("kesilmiş", "kesilmis", "kesik", "truncate", "truncated", "truncate edilmiş", "kesilme"),
            "kesilmiş",
        ),)),
        explanation=(
            "Sansürlemede birim görünür ama değeri sınıra dönüştürülür; kesilmede birim hiç görünmez. Olabilirlik ve "
            "tanımlama iki durumda farklı kurulur (§6.1)."
        ),
    ),
    Question(
        key="b02", concept="ters-mills-adi", note=_note("6.3"),
        prompt="$\\lambda(a)=\\phi(a)/\\Phi(a)$ oranına **(1)** ______ oranı denir.",
        answer=FillBlanks((TextBlank(
            ("ters Mills", "ters mills", "inverse Mills", "inverse mills", "Mills", "ters Mill"),
            "ters Mills",
        ),)),
        explanation=(
            "Kesilmiş ortalamada $m^{\\#}(x)=x'\\beta+\\sigma\\lambda(x'\\beta/\\sigma)$ ve Heckman düzeltmesinde aynı oran "
            "ortaya çıkar (§6.3)."
        ),
    ),
    Question(
        key="b03", concept="dislama-kisiti-adi", note=_note("6.12"),
        prompt=(
            "Seçim denkleminde yer alıp sonuç denkleminde yer almayan değişkene dayanan tanımlama varsayımına **(1)** "
            "______ kısıtı denir."
        ),
        answer=FillBlanks((TextBlank(("dışlama", "dislama", "exclusion", "dışlanma"), "dışlama (exclusion)"),)),
        explanation=(
            "IV'deki gibi ilgililik (seçimi etkilemesi) ve dışlama (sonucu doğrudan etkilememesi) ayrı iddialardır; ikincisi "
            "veriden değil ekonomik argümandan gelir (§6.12)."
        ),
    ),
    Question(
        key="b04", concept="chj-tobit-gizli-150", note=_note("6.16", 2),
        prompt=(
            "CHJ uygulamasında Tobit'in gizli ortalaması 150 bin peso gelirde yaklaşık **(1)** ______ bin pesodur "
            "(iki ondalık)."
        ),
        answer=FillBlanks((NumberBlank(3.60, 0.006, "3,60"),)),
        explanation=(
            "Aynı profilde OLS 7,72, LAD 2,24 verir; Tobit'in ima ettiği gözlenen ortalama ise 9,81'dir. Hedefler farklı "
            "olduğu için düzeyler farklıdır (§6.16, Adım 2–3)."
        ),
    ),
    Question(
        key="b05", concept="heckman-birinci-asama", note=_note("6.11"),
        prompt=(
            "Heckman iki aşamasının ilk aşamasında seçim göstergesi $S$, tam örneklemde **(1)** ______ modeliyle "
            "tahmin edilir."
        ),
        answer=FillBlanks((TextBlank(("Probit", "probit", "probit modeli"), "Probit"),)),
        explanation=(
            "Seçim hatası normal ve $\\operatorname{Var}(u)=1$ normalizasyonuyla Probit kullanılır; tahmin edilen indeksten "
            "$\\hat\\lambda_i$ hesaplanır (§6.11)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="f01", concept="tobit-olasilik-marjinal-etki", note=_note("6.6.2"),
        prompt=(
            "Tobit'te sansürlenmeme olasılığı $P(Y>0\\mid x)=\\Phi(x'\\beta/\\sigma)$ üzerindeki marjinal etkiyi "
            "$\\beta_j$, $\\sigma$ ve $f=\\phi(x'\\beta/\\sigma)$ cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\partial P(Y>0\mid x)/\partial x_j",
            symbols=(
                Symbol("b", r"\beta_j", "Tobit katsayısı", -2.0, 2.0, aliases=("β_j", "βj", "β", "beta")),
                Symbol("s", r"\sigma", "hata standart sapması", 0.5, 3.0, aliases=("σ", "sigma")),
                Symbol("f", "f", "yoğunluk değeri φ(x'β/σ)", 0.05, 0.4, aliases=("φ", "phi")),
            ),
            answer="b*f/s",
            shown=r"\dfrac{\beta_j}{\sigma}\,\phi\!\left(\dfrac{x'\beta}{\sigma}\right)",
        ),
        explanation=(
            "Probit'teki mantığın aynısı, ölçek $1/\\sigma$ ile: etki indeksin bulunduğu noktaya göre değişir (§6.6.2)."
        ),
    ),
    Question(
        key="f02", concept="heckman-kosullu-ortalama", note=_note("6.10"),
        prompt=(
            "Normal seçim modelinde ($\\operatorname{Var}(u)=1$) seçilmiş örneklem koşullu ortalamasını $\\mu=X'\\beta$, "
            "$c=\\sigma_{eu}$ ve $\\lambda=\\lambda(Z'\\gamma)$ cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\mathbb E[Y\mid X,Z,S=1]",
            symbols=(
                Symbol("m", r"\mu", "sonuç denkleminin indeksi X'β", -2.0, 2.0, aliases=("μ", "mu")),
                Symbol("c", r"\sigma_{eu}", "Cov(e, u)", -1.0, 1.0, aliases=("σ_eu", "sigma_eu")),
                Symbol("l", r"\lambda", "ters Mills oranı", 0.1, 2.0, aliases=("λ", "lambda")),
            ),
            answer="m + c*l",
            shown=r"X'\beta+\sigma_{eu}\,\lambda(Z'\gamma)",
        ),
        explanation=(
            "Naif OLS $\\sigma_{eu}\\lambda$ terimini dışarıda bırakır; bu terim $X$ ile ilişkili olduğunda eksik değişken "
            "yanlılığı doğar (§6.10)."
        ),
    ),
    Question(
        key="f03", concept="mills-katsayisi-hedefi", note=_note("6.13"),
        prompt=(
            "$\\operatorname{Var}(u)=1$, $\\operatorname{Var}(e)=\\sigma_e^2$ ve $\\operatorname{Corr}(e,u)=\\rho$ ise "
            "$\\hat\\lambda$'nın katsayısının hedeflediği $\\sigma_{eu}$'yu $\\rho$ ve $\\sigma_e$ cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\sigma_{eu}",
            symbols=(
                Symbol("r", r"\rho", "korelasyon", -0.9, 0.9, aliases=("ρ", "rho")),
                Symbol("s", r"\sigma_e", "sonuç hatasının standart sapması", 0.5, 3.0, aliases=("σ_e", "σe", "σ", "sigma")),
            ),
            answer="r*s",
            shown=r"\sigma_{eu}=\rho\,\sigma_e",
        ),
        explanation=(
            "Kovaryans = korelasyon × standart sapmalar çarpımı ve $\\operatorname{sd}(u)=1$. Benzetimde (Tablo 6.3) "
            "$\\rho=0{,}6$, $\\sigma_e=1$; $\\hat\\lambda$ katsayısı 0,632 (§6.13)."
        ),
    ),
    Question(
        key="f04", concept="sansur-olasiligi-simetri", note=_note("6.3"),
        prompt="Tobit'te sansürlenme olasılığı $P(Y=0\\mid x)$'i $F=\\Phi(x'\\beta/\\sigma)$ cinsinden yazın.",
        answer=Equation(
            lhs=r"P(Y=0\mid x)",
            symbols=(Symbol("F", "F", "Φ(x'β/σ)", 0.05, 0.95, aliases=("Φ",)),),
            answer="1 - F",
            shown=r"1-\Phi(x'\beta/\sigma)=\Phi(-x'\beta/\sigma)",
        ),
        explanation=(
            "Normal dağılımın simetrisi: $\\Phi(-a)=1-\\Phi(a)$. Sansürlenmeme olasılığı $F$ olduğundan sansürlenme "
            "olasılığı tümleyenidir (§6.3)."
        ),
    ),
    Question(
        key="f05", concept="net-transfer-tanimi", note=_note("6.16", 1),
        prompt=(
            "Hansen'in oluşturma dosyasındaki net hükümet dışı transferi; yurt dışından ($a$), yurt içinden ($d$) ve ayni "
            "($k$) alınan transferler ile verilen hediyeler ($g$) cinsinden yazın."
        ),
        answer=Equation(
            lhs="transfers",
            symbols=(
                Symbol("a", "a", "yurt dışından alınan (tabroad)", 0.0, 50.0),
                Symbol("d", "d", "yurt içinden alınan (tdomestic)", 0.0, 50.0),
                Symbol("k", "k", "ayni alınan (tinkind)", 0.0, 50.0),
                Symbol("g", "g", "verilen hediyeler (tgifts)", 0.0, 50.0),
            ),
            answer="a + d + k - g",
            shown=r"tabroad+tdomestic+tinkind-tgifts",
        ),
        explanation=(
            "Net transfer gelir düzeltmesinde kullanılır; uygulamadaki sansürlü sonuç ise yalnız alınanların toplamıdır "
            "($a+d+k\\ge0$), çünkü net transfer negatif olabilir (§6.16, Adım 1)."
        ),
    ),
)


KONU06_QUIZ = QuestionSet(
    topic_key="konu06",
    title="Konu 6: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set sansürleme, kesilme ve örneklem seçimi kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki "
        "bir bölüme bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Bölüm sonu "
        "egzersizlerini tekrar etmez, onları tamamlar."
    ),
    sections=("6.1", "6.3", "6.4", "6.5", "6.6.1", "6.6.2", "6.7", "6.9", "6.10", "6.11", "6.12", "6.13", "6.14", "6.16"),
)
