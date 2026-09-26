"""Konu 2 "Kendini sına" soru seti.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Bölüm sonu egzersizleri
uzun türetmeler ister; buradaki sorular aynı konuların farklı ve kısa yönlerini sınar.
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
        key="k01", concept="fwl-arindirilmis-degiskenlik", note=_note("2.1.1"),
        prompt="Frisch–Waugh–Lovell teoremine göre çoklu regresyondaki $x_1$ katsayısı hangi regresyondan **birebir** elde edilir?",
        answer=MultipleChoice((
            "$Y$'nin yalnız $x_1$ üzerindeki basit regresyonundan",
            "$Y$'nin, $x_1$'in diğer regresörler üzerindeki regresyonundan kalan artıklar üzerindeki regresyonundan",
            "$x_1$'in $Y$ üzerindeki regresyonundan",
            "$Y$'nin diğer regresörler üzerindeki regresyonunun uyum değerlerinden",
        ), correct=1),
        explanation=(
            "Katsayı yalnız $x_1$'in diğer regresörlerle ortak olmayan değişkenliğinden öğrenir. Sezgi "
            "Deney 2'de FWL eğimi ile çoklu regresyon katsayısı makine duyarlığında eşittir (§2.1.1)."
        ),
    ),
    Question(
        key="k02", concept="duzey-log-yorumu", note=_note("2.2.3"),
        prompt=r"$Y=\beta_0+\beta_1\log X+e$ modelinde $\beta_1=5$ ise $X$'teki yüzde 1'lik artış yaklaşık olarak neyle ilişkilidir?",
        answer=MultipleChoice(("$Y$'de yüzde 5 artış", "$Y$'de 5 birim artış", "$Y$'de 0,05 birim artış", "$Y$'de yüzde 0,05 artış"), correct=2),
        explanation=r"Düzey–log modelinde $X$'te yaklaşık yüzde 1'lik değişim $Y$'de $\beta_1/100=0{,}05$ birimlik değişimle ilişkilidir (§2.2.3).",
    ),
    Question(
        key="k03", concept="log-log-esneklik", note=_note("2.2.4"),
        prompt=r"$\log Y=\beta_0+\beta_1\log X+e$ modelinde $\beta_1=0{,}8$. Hangisi doğru yorumdur?",
        answer=MultipleChoice((
            "$X$'te yüzde 1'lik artış, $Y$'nin koşullu geometrik ortalamasında yaklaşık yüzde 0,8 artışla ilişkilidir",
            "$X$'te 1 birim artış $Y$'yi 0,8 birim artırır",
            "$X$'te 1 birim artış $Y$'yi yüzde 80 artırır",
            "$X$ ile $Y$ arasındaki korelasyon 0,8'dir",
        ), correct=0),
        explanation="Log–log modelinde katsayı esnekliktir; birimden bağımsızdır ve oransal değişimleri bağlar (§2.2.4).",
    ),
    Question(
        key="k04", concept="eksik-degisken-iki-kosul", note=_note("2.4"),
        prompt="Dışarıda bırakılan $x_2$, kısa regresyonda $x_1$'in katsayısını hangi durumda yanlı yapar?",
        answer=MultipleChoice((
            "$x_2$ $Y$'yi etkiliyorsa, $x_1$ ile ilişkisi ne olursa olsun",
            "$x_2$ $x_1$ ile ilişkiliyse, $Y$'yi etkilemese bile",
            "Örneklem küçükse",
            "$x_2$ hem $Y$'yi etkiliyor hem de $x_1$ ile ilişkiliyse",
        ), correct=3),
        explanation=(
            r"Yanlılık $\beta_2\cdot\operatorname{Cov}(x_1,x_2)/\operatorname{Var}(x_1)$ biçimindedir; iki çarpandan "
            "biri sıfırsa yanlılık yoktur. Sezgi Deney 2'de β2 veya ρ sıfıra çekilince fark kaybolur (§2.4)."
        ),
    ),
    Question(
        key="k05", concept="hc1-hc3-farki", note=_note("2.6"),
        prompt="HC1 ile HC3 dayanıklı standart hataları arasındaki fark nedir?",
        answer=MultipleChoice((
            "HC1 katsayıları da yeniden hesaplar, HC3 hesaplamaz",
            r"HC1 artık karelerini $n/(n-k)$ ile çarpar; HC3 her artık karesini $(1-h_{ii})^2$ ile böler ve yüksek kaldıraçlı gözlemlerde daha ihtiyatlıdır",
            "HC3 küme içi bağımlılığa da dayanıklıdır",
            "İkisi yalnız homoskedastisite altında geçerlidir",
        ), correct=1),
        explanation="İkisi de katsayıyı değiştirmez; yalnız varyans tahmininde küçük örneklem düzeltmeleri farklıdır (§2.6).",
    ),
    Question(
        key="k06", concept="kumeleme-duzeyi", note=_note("2.7.4"),
        prompt=(
            "Bir eğitim politikası **okul** düzeyinde atanıyor; veri **öğrenci** düzeyinde. Standart hatalar "
            "hangi düzeyde kümelenmelidir?"
        ),
        answer=MultipleChoice((
            "Okul düzeyinde: politikanın atandığı ve hataların ortak şoklar taşıdığı düzey",
            "Öğrenci düzeyinde",
            "Sınıf düzeyinde, çünkü daha çok küme vardır",
            "Kümelemeye gerek yoktur; HC1 yeterlidir",
        ), correct=0),
        explanation=(
            "Kümeleme, regresörün ve hatanın ortak şok taşıdığı düzeyde yapılır. Daha çok küme elde etmek için "
            "daha dar düzey seçmek bağımlılığı gizler (§2.7.4)."
        ),
    ),
    Question(
        key="k07", concept="ortak-hipotez-wald", note=_note("2.9.2"),
        prompt="Üç bölge kuklasının katsayılarının **hepsinin birlikte** sıfır olduğunu test etmek için hangisi uygundur?",
        answer=MultipleChoice((
            "Her kukla için ayrı t-testi yapmak; üçü de anlamsızsa sıfır hipotezini kabul etmek",
            "$R^2$'nin yüksek olup olmadığına bakmak",
            "Üç kısıtlı ortak Wald (veya F) testi yapmak",
            "Üç katsayının ortalamasına t-testi uygulamak",
        ), correct=2),
        explanation=(
            "Ortak hipotez, katsayıların kovaryansını da hesaba katan tek bir testle sınanır; ayrı t-testleri "
            "ortak hipotezin boyutunu kontrol etmez (§2.9.2)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="heteroskedastisite-yanlilik-yaratmaz", note=_note("2.6.1"),
        prompt="Heteroskedastisite OLS katsayılarını yanlı yapar.",
        answer=TrueFalse(False),
        explanation=(
            "Heteroskedastisite katsayının yanlılığını değil, klasik standart hatanın geçerliliğini bozar. Sezgi "
            "Deney 1'de iki tahminde eğim aynıdır; değişen yalnız belirsizlik ölçüsüdür (§2.6.1)."
        ),
    ),
    Question(
        key="d02", concept="robust-sh-icselligi-cozmez", note=_note("2.6.1"),
        prompt="Dayanıklı (robust) standart hata kullanmak eksik değişken yanlılığını da giderir.",
        answer=TrueFalse(False),
        explanation="Dayanıklı standart hata yalnız belirsizlik hesabını değiştirir; tahmin hedefini ve katsayıyı değiştirmez (§2.6.1; Uygulama, Adım 4).",
    ),
    Question(
        key="d03", concept="hc1-kume-bagimliligini-gormez", note=_note("2.7.1"),
        prompt="Hatalar aynı küme içinde ilişkiliyse HC1 standart hataları bu bağımlılığı hesaba katmaz.",
        answer=TrueFalse(True),
        explanation=(
            "HC1 gözlemleri bağımsız kabul eder, yalnız varyanslarının farklı olmasına izin verir. Küme içi "
            "bağımlılık için küme-dayanıklı kovaryans gerekir; Sezgi Deney 3'te HC1 klasikle neredeyse aynıdır (§2.7.1)."
        ),
    ),
    Question(
        key="d04", concept="p-degeri-hipotez-olasiligi-degildir", note=_note("2.9.1"),
        prompt="p = 0,03 ise sıfır hipotezinin doğru olma olasılığı yüzde 3'tür.",
        answer=TrueFalse(False),
        explanation=(
            "p-değeri, sıfır hipotezi **doğruyken** en az gözlenen kadar aşırı bir istatistik elde etme olasılığıdır. "
            "Koşullandırmanın yönü terstir; hipotezin olasılığını vermez (§2.9.1)."
        ),
    ),
    Question(
        key="d05", concept="guc-orneklemle-artar", note=_note("2.9.1"),
        prompt="Diğer her şey sabitken örneklem büyüdükçe bir testin gücü artar.",
        answer=TrueFalse(True),
        explanation="Standart hata $1/\\sqrt n$ hızıyla küçülür; aynı gerçek etki daha kolay ayırt edilir (§2.9.1).",
    ),
    Question(
        key="d06", concept="kucuk-p-buyukluk-soylemez", note=_note("2.10"),
        prompt="Bir katsayının p-değerinin çok küçük olması, etkinin ekonomik büyüklüğü hakkında tek başına bilgi vermez.",
        answer=TrueFalse(True),
        explanation=(
            "Çok büyük örneklemde ekonomik olarak önemsiz bir etki bile istatistiksel olarak anlamlı çıkabilir. "
            "Büyüklük, katsayının birimi ve güven aralığıyla değerlendirilir (§2.10)."
        ),
    ),
    Question(
        key="d07", concept="vif-nedeniyle-cikarma", note=_note("2.11.3"),
        prompt="Yüksek VIF gördüğümüz bir kontrol değişkenini yalnız bu nedenle modelden çıkarmak doğru bir uygulamadır.",
        answer=TrueFalse(False),
        explanation=(
            "Yüksek VIF, katsayının az bir kısmi değişkenlikten tanımlandığını gösterir; kontrolü çıkarmak tahmin "
            "hedefini değiştirir ve eksik değişken yanlılığı yaratabilir (§2.11.3)."
        ),
    ),
    # --- Boşluk doldurma ---------------------------------------------------------
    Question(
        key="b01", concept="karesel-marjinal-etki", note=_note("2.3.1"),
        prompt=(
            r"$\mathbb{E}[Y\mid X]=\beta_0+\beta_1X+\beta_2X^2$, $\beta_1=0{,}04$ ve $\beta_2=-0{,}001$ ise "
            r"$X=10$'daki marjinal etki **(1)** ______'dir."
        ),
        answer=FillBlanks((NumberBlank(0.02, 0.0005, "0,02"),)),
        explanation=r"Marjinal etki $\beta_1+2\beta_2X=0{,}04+2(-0{,}001)(10)=0{,}02$; karesel modelde etki $X$ düzeyine bağlıdır (§2.3.1).",
    ),
    Question(
        key="b02", concept="kukla-tuzagi", note=_note("2.3.2"),
        prompt="Dört bölgeli bir değişken için modelde sabit terim varken en fazla **(1)** ____ bölge kuklası kullanılabilir.",
        answer=FillBlanks((NumberBlank(3, 0.001, "3"),)),
        explanation=(
            "Dört kuklanın toplamı sabit terime eşittir; birini dışarıda bırakmazsak tam çoklu doğrusal bağlantı "
            "(kukla tuzağı) oluşur. Dışarıdaki bölge referanstır (§2.3.2)."
        ),
    ),
    Question(
        key="b03", concept="breusch-pagan-sifir-hipotezi", note=_note("2.13", 4),
        prompt="Breusch–Pagan testinin sıfır hipotezi **(1)** ______'dir.",
        answer=FillBlanks((TextBlank(
            ("homoskedastisite", "homoskedastiklik", "homoskedastik", "sabit varyans", "eşit varyans", "eşvaryans", "varyans sabittir"),
            "homoskedastisite (sabit varyans)",
        ),)),
        explanation="Test, artık karelerinin regresörlerle ilişkisiz olduğunu sınar; reddedilmesi heteroskedastisiteye işaret eder (§2.13, Adım 4).",
    ),
    Question(
        key="b04", concept="hc1-kucuk-orneklem-carpani", note=_note("2.6"),
        prompt="HC1, HC0'daki artık karelerini **(1)** ______ ile çarpar.",
        answer=FillBlanks((TextBlank(("n/(n-k)", "n/(n−k)", "n/(n - k)"), "n/(n − k)"),)),
        explanation="Bu çarpan, $k$ katsayı tahmin edilirken artıkların sistematik olarak küçük kalmasını telafi eder (§2.6).",
    ),
    Question(
        key="b05", concept="etkilesimde-grup-egimi", note=_note("2.13", 5),
        prompt="M3'te erkeklerin eğitim eğimi 0,1083, eğitim × kadın etkileşimi 0,0044 ise kadınların eğitim eğimi **(1)** ______'tür.",
        answer=FillBlanks((NumberBlank(0.1127, 0.0001, "0,1127"),)),
        explanation="Etkileşimli modelde ana etki referans grubun eğimidir; diğer grubun eğimi iki katsayının toplamıdır (§2.13, Adım 5).",
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="f01", concept="karesel-donum-noktasi", note=_note("2.3.1"),
        prompt=r"$\mathbb{E}[Y\mid X]=\beta_0+\beta_1X+\beta_2X^2$ modelinde, $\beta_2\neq0$ iken dönüm noktası $X^*$'ı yazın.",
        answer=Equation(
            lhs="X^*",
            symbols=(
                Symbol("b1", r"\beta_1", "doğrusal terim", 0.01, 0.1, aliases=("β1", "beta1", "β_1")),
                Symbol("b2", r"\beta_2", "karesel terim", -0.01, -0.001, aliases=("β2", "beta2", "β_2")),
            ),
            answer="-b1/(2*b2)",
            shown=r"X^*=-\dfrac{\beta_1}{2\beta_2}",
        ),
        explanation=r"Marjinal etki $\beta_1+2\beta_2X$ sıfıra eşitlenir: $X^*=-\beta_1/(2\beta_2)$ (§2.3.1).",
    ),
    Question(
        key="f02", concept="dogrusal-birlesim-varyansi", note=_note("2.13", 5),
        prompt=r"$\hat\beta_1+\hat\beta_3$ birleşiminin varyansını $\operatorname{Var}(\hat\beta_1)$, $\operatorname{Var}(\hat\beta_3)$ ve $\operatorname{Cov}(\hat\beta_1,\hat\beta_3)$ cinsinden yazın.",
        answer=Equation(
            lhs=r"\operatorname{Var}(\hat\beta_1+\hat\beta_3)",
            symbols=(
                Symbol("v1", r"\operatorname{Var}(\hat\beta_1)", "ilk katsayının varyansı", 0.5, 2.0),
                Symbol("v3", r"\operatorname{Var}(\hat\beta_3)", "ikinci katsayının varyansı", 0.5, 2.0),
                Symbol("c13", r"\operatorname{Cov}(\hat\beta_1,\hat\beta_3)", "kovaryans", -0.4, 0.4),
            ),
            answer="v1 + v3 + 2*c13",
            shown=r"\operatorname{Var}(\hat\beta_1)+\operatorname{Var}(\hat\beta_3)+2\operatorname{Cov}(\hat\beta_1,\hat\beta_3)",
        ),
        explanation=(
            r"Genel kural $a'Va$; $a=(1,1)'$ için kovaryans iki kez girer. Kovaryansı unutmak, grup eğiminin "
            "standart hatasını yanlış hesaplamanın en yaygın yoludur (§2.13, Adım 5)."
        ),
    ),
    Question(
        key="f03", concept="delta-yuzde-etki-sh", note=_note("2.8.1"),
        prompt=r"$g(\beta)=100\,(e^{\beta}-1)$ için delta yöntemiyle $\operatorname{SH}(g(\hat\beta))$'yı $\hat\beta$ ve $\operatorname{SH}(\hat\beta)$ cinsinden yazın.",
        answer=Equation(
            lhs=r"\operatorname{SH}(g)",
            symbols=(
                Symbol("b", r"\hat\beta", "katsayı tahmini", -0.5, 0.5, aliases=("β̂", "β", "beta")),
                Symbol("s", r"\operatorname{SH}(\hat\beta)", "katsayının standart hatası", 0.01, 0.2, aliases=("SH", "SE", "se")),
            ),
            answer="100*exp(b)*s",
            shown=r"\operatorname{SH}(g)\approx 100\,e^{\hat\beta}\,\operatorname{SH}(\hat\beta)",
        ),
        explanation=r"Delta yöntemi: $\operatorname{SH}(g)\approx|g'(\hat\beta)|\,\operatorname{SH}(\hat\beta)$ ve $g'(\beta)=100e^{\beta}>0$. Uygulama Adım 6'da M2 için 0,126 puan (§2.8.1).",
    ),
    Question(
        key="f04", concept="t-istatistigi", note=_note("2.9"),
        prompt=r"$H_0:\beta=\beta_0$ için t istatistiğini $\hat\beta$, $\beta_0$ ve $\operatorname{SH}(\hat\beta)$ cinsinden yazın.",
        answer=Equation(
            lhs="t",
            symbols=(
                Symbol("b", r"\hat\beta", "tahmin", -2.0, 2.0, aliases=("β̂", "beta_hat")),
                Symbol("b0", r"\beta_0", "sıfır hipotezindeki değer", -1.0, 1.0, aliases=("β0", "β_0", "beta0")),
                Symbol("s", r"\operatorname{SH}(\hat\beta)", "standart hata", 0.1, 1.0, aliases=("SH", "SE", "se")),
            ),
            answer="(b - b0)/s",
            shown=r"t=\dfrac{\hat\beta-\beta_0}{\operatorname{SH}(\hat\beta)}",
        ),
        explanation="t istatistiği, tahminin hipotezdeki değerden uzaklığını standart hata biriminde ölçer (§2.9).",
    ),
    Question(
        key="f05", concept="moulton-sh-orani", note=_note("2.7.3"),
        prompt=r"Eşit küme büyüklüğü $m$ ve küme içi korelasyon $\rho$ iken küme-dayanıklı SH'nin klasik SH'ye yaklaşık oranını yazın.",
        answer=Equation(
            lhs=r"\operatorname{SH}_{\text{küme}}/\operatorname{SH}_{\text{klasik}}",
            symbols=(
                Symbol("m", "m", "küme büyüklüğü", 2.0, 50.0),
                Symbol("rho", r"\rho", "küme içi korelasyon", 0.0, 0.9, aliases=("ρ",)),
            ),
            answer="sqrt(1 + (m - 1)*rho)",
            shown=r"\sqrt{1+(m-1)\rho}",
        ),
        explanation=(
            r"Tasarım etkisi $\tau=1+(m-1)\rho$ varyans oranıdır; SH oranı $\sqrt\tau$'dur. Sezgi Deney 3'te "
            "bu çarpan tahmin edilen oranla karşılaştırılır (§2.7.3)."
        ),
    ),
)


KONU02_QUIZ = QuestionSet(
    topic_key="konu02",
    title="Konu 2: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set haftanın temel kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir "
        "bölüme bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Bölüm sonu "
        "egzersizlerinin türetmelerini tekrar etmez, onları tamamlar."
    ),
    sections=("2.1.1", "2.2.3", "2.2.4", "2.3.1", "2.3.2", "2.4", "2.6", "2.6.1", "2.7.1", "2.7.3",
              "2.7.4", "2.8.1", "2.9", "2.9.1", "2.9.2", "2.10", "2.11.3", "2.13"),
)
