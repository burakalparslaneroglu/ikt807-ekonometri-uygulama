"""Konu 10 "Kendini sına" soru seti: jackknife, bootstrap standart hatası, güven aralıkları, wild ve küme bootstrap'ı.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Bölüm sonu egzersizlerindeki hesaplar
(dışarıda bırakma artıkları, {2, 5, 9} örneği, dört tahminden bootstrap SH'si, Y ve X'i ayrı yeniden örnekleme,
sıralı tekrarlardan percentile aralığı, sayısal percentile-t aralığı, parametrik ve pairs adımları, tek
gözlemde wild artığı, Tablo 10.1'de yüzde fark, B = 1.000'den 4.000'e Monte Carlo hatası, 50 okul örneği,
içsel OLS'de dar aralık, replikasyon protokolü) tekrar edilmez.
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
        key="k01", concept="plug-in-ilkesi", note=_note("10.1"),
        prompt="Bootstrap'ın çıkış noktası nedir?",
        answer=MultipleChoice((
            "Bilinmeyen anakütle dağılımı $F$ yerine örneklemin ampirik dağılımını $\\hat F_n$ kullanmak",
            "Aynı anakütleden yeni ve bağımsız veri toplamak",
            "Tahmin edicinin normal dağıldığını varsaymak",
            "Gözlemleri sırayla birer birer dışarıda bırakmak",
        ), correct=0),
        explanation=(
            "Plug-in ilkesi: $\\hat F_n$ her gözleme $1/n$ olasılık verir; bootstrap örneği ondan yerine koyarak çekilen $n$ "
            "gözlemdir. Sonuncu seçenek jackknife'tır (§10.1, §10.2)."
        ),
    ),
    Question(
        key="k02", concept="jackknife-duzgun-olmayan", note=_note("10.2"),
        prompt="Jackknife varyans tahmini hangi tür tahmin edicide tutarsız olabilir?",
        answer=MultipleChoice((
            "Örneklem ortalamasında",
            "OLS katsayısında",
            "Medyan ve kantil regresyon gibi düzgün olmayan (türevlenemeyen) tahmin edicilerde",
            "Katsayıların doğrusal birleşimlerinde",
        ), correct=2),
        explanation=(
            "Bir gözlemi çıkarmak medyanı ya hiç değiştirmez ya da sıçratır; bu davranış varyansı doğru temsil etmez. Bu "
            "durumlarda varsayılan tercih bootstrap'tır (§10.2)."
        ),
    ),
    Question(
        key="k03", concept="ayni-tahmin-proseduru", note=_note("10.3"),
        prompt="Ana analizde değişken seçimi veya standardizasyon varsa bootstrap'ta ne yapılmalıdır?",
        answer=MultipleChoice((
            "Bu adımlar bir kez yapılır; bootstrap yalnız son regresyonu tekrarlar",
            "Bu adımlar her tekrarın içinde yeniden yapılır; aksi hâlde yalnız son aşamanın belirsizliği taklit edilir",
            "Bootstrap bu durumda kullanılamaz",
            "Tekrar sayısı iki katına çıkarılır",
        ), correct=1),
        explanation=(
            "Algoritmanın kritik noktası \"aynı tahmin prosedürü\"dür: bootstrap, tahmin edicinin gerçek veri-analiz zincirini "
            "olabildiğince doğru yeniden üretmelidir (§10.3)."
        ),
    ),
    Question(
        key="k04", concept="normal-aralik-asimetri", note=_note("10.6"),
        prompt="Normal yaklaşımlı bootstrap aralığı $\\hat\\theta\\pm1{,}96\\,\\widehat{se}_{boot}$ bootstrap dağılımının hangi bilgisini kullanmaz?",
        answer=MultipleChoice((
            "Yayılımını",
            "Tekrar sayısını",
            "Orijinal tahmini",
            "Çarpıklığını ve asimetrik biçimini",
        ), correct=3),
        explanation=(
            "Bu aralık yalnız yayılımı kullanır. Dağılım belirgin çarpıksa percentile veya percentile-$t$ aralıkları "
            "karşılaştırılmalıdır (§10.6)."
        ),
    ),
    Question(
        key="k05", concept="parametrik-bootstrap-riski", note=_note("10.9"),
        prompt="Parametrik bootstrap'ın temel riski nedir?",
        answer=MultipleChoice((
            "Model yanlışsa yanlış model (normallik, varyans yapısı, fonksiyonel biçim) her yapay örneklemde yeniden üretilir",
            "Her zaman nonparametrik bootstrap'tan daha geniş aralık verir",
            "Seed kullanılamaz",
            "Yalnız kümeli veride çalışır",
        ), correct=0),
        explanation=(
            "Model doğruysa küçük örneklemde daha hassas olabilir; yanlışsa bootstrap da yanlışı tekrarlar. İki bootstrap "
            "türünü karşılaştırmak bir duyarlılık analizidir (§10.9)."
        ),
    ),
    Question(
        key="k06", concept="seed-iyi-uygulama", note=_note("10.13"),
        prompt="Seed ile ilgili iyi uygulama hangisidir?",
        answer=MultipleChoice((
            "En dar güven aralığını veren seed seçilir",
            "Seed hiç kaydedilmez; her çalıştırma farklı olmalıdır",
            "Analiz sırasında sonucun $B$ ve seed'e duyarlılığı görülür; nihai tablo için büyük $B$ seçilir ve seed kodda "
            "kaydedilir",
            "Küçük $B$ ile sabit seed kullanmak yeterlidir",
        ), correct=2),
        explanation=(
            "Küçük $B$ ile sabit seed sonucu yapay biçimde istikrarlı gösterebilir; büyük $B$ ve kayıtlı seed sonucu yeniden "
            "üretilebilir kılar (§10.13)."
        ),
    ),
    Question(
        key="k07", concept="zayif-tanimlama-bootstrap", note=_note("10.15"),
        prompt="Zayıf araç altında standart 2SLS bootstrap'ı için hangisi doğrudur?",
        answer=MultipleChoice((
            "Yeniden örnekleme tanımlama bilgisini yaratır",
            "Zayıf tanımlamayı otomatik olarak düzeltir",
            "Her zaman klasik 2SLS aralığından daha iyi kapsama verir",
            "Beklenmedik davranabilir; zayıf tanımlamaya dayanıklı testler veya özel bootstrap tasarımları gerekir",
        ), correct=3),
        explanation=(
            "Tahmin edicinin dağılımı tanımlama gücüne göre keskin biçimde değişiyorsa standart bootstrap bunu doğru taklit "
            "etmeyebilir; bulanık RDD'de küçük tedavi sıçraması da aynı sınıftadır (§10.15)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="tekrar-sayisi-veri-degil", note=_note("10.4"),
        prompt="$B$'yi 1.000'den 100.000'e çıkarmak orijinal örneklemin örnekleme belirsizliğini azaltır.",
        answer=TrueFalse(False),
        explanation=(
            "Bütün tekrarlar aynı örneklemin bilgisini yeniden düzenler. $B$ büyüdükçe yalnız bootstrap hesabının Monte "
            "Carlo gürültüsü azalır (§10.4)."
        ),
    ),
    Question(
        key="d02", concept="pairs-heteroskedastisite", note=_note("10.5"),
        prompt=(
            "Pairs bootstrap, artıkların $X$'e bağlı büyüklüğünü gözlemle birlikte taşıdığı için heteroskedastisite altında "
            "doğal bir seçenektir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "$(Y_i,X_i)$ çifti birlikte çekilir; belirli $X$ değerlerine bağlı artık büyüklükleri de gözlemle gelir ve koşullu "
            "varyans deseni korunur (§10.5)."
        ),
    ),
    Question(
        key="d03", concept="percentile-t-sh-dogrulugu", note=_note("10.8"),
        prompt="Percentile-$t$ aralığının avantajı ancak her tekrarda hesaplanan standart hata doğruysa ortaya çıkar.",
        answer=TrueFalse(True),
        explanation=(
            "Yanlış kovaryans formülünü binlerce kez tekrarlamak onu doğru yapmaz. Studentize etmek istatistiği yaklaşık "
            "pivotal yapmayı amaçlar; bedeli her tekrarda güvenilir bir SH'dir (§10.8)."
        ),
    ),
    Question(
        key="d04", concept="artik-bootstrap-hetero", note=_note("10.10"),
        prompt=(
            "Regresörleri sabit tutup artıkların yerlerini rastgele değiştiren basit artık bootstrap'ı heteroskedastik "
            "tasarımda koşullu varyans desenini korur."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Artıkları karıştırmak büyük artıkları küçük varyanslı gözlemlere taşır ve deseni bozar. Wild bootstrap her "
            "gözlemin artığını kendi yerinde tutar ve yalnız işaretini rastgele değiştirir (§10.10)."
        ),
    ),
    Question(
        key="d05", concept="kisitli-wild-test", note=_note("10.10"),
        prompt="Hipotez testi için wild bootstrap'ta yapay veriler kısıtlı modelin artıklarıyla, $H_0$ altında üretilmelidir.",
        answer=TrueFalse(True),
        explanation=(
            "Test istatistiğinin sıfır hipotezi altındaki dağılımı taklit edilir. Ana modelin artıkları test edilen kısıtı "
            "yapay örneklemlerde ihlal edebilir (§10.10)."
        ),
    ),
    Question(
        key="d06", concept="aralik-turleri-yakinligi", note=_note("10.11"),
        prompt=(
            "Notlardaki heteroskedastik örnekte ($n=240$) percentile ve percentile-$t$ aralıkları HC1 normal aralığından çok "
            "farklıdır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "HC1 normal [1,741; 2,041], percentile [1,735; 2,036], percentile-$t$ [1,741; 2,049]: aralıklar birbirine yakındır. "
            "İyi kurulmuş bootstrap ile doğru analitik formül aynı belirsizliği hedefler (§10.11)."
        ),
    ),
    Question(
        key="d07", concept="yakinlik-kanit-degil", note=_note("10.16"),
        prompt="Analitik robust SH ile bootstrap SH'nin yakın çıkması modelin doğru belirtildiğini kanıtlar.",
        answer=TrueFalse(False),
        explanation=(
            "İki yöntem de aynı spesifikasyonu ve aynı tanımlama varsayımlarını kullanır. Yakınlık yalnız örnekleme "
            "belirsizliği açısından yöntemsel güveni artırır (§10.16, §10.17)."
        ),
    ),
    # --- Boşluk doldurma ---------------------------------------------------------
    Question(
        key="b01", concept="pairs-adi", note=_note("10.5"),
        prompt=(
            "Regresyonda $(Y_i,X_i)$ gözlem çiftlerini birlikte yeniden örnekleyen yöntem gözlem çiftleri bootstrap'ıdır; "
            "İngilizce adı **(1)** ______ bootstrap."
        ),
        answer=FillBlanks((TextBlank(("pairs", "pair", "pairs bootstrap", "case", "cases"), "pairs"),)),
        explanation=(
            "Bootstrap tasarım matrisi ve bağımlı değişken $(Y^*,X^*)$ birlikte oluşur ve OLS her tekrarda "
            "$\\hat\\beta^*=(X^{*\\prime}X^*)^{-1}X^{*\\prime}Y^*$ olarak yeniden hesaplanır (§10.5)."
        ),
    ),
    Question(
        key="b02", concept="rademacher", note=_note("10.10"),
        prompt="Wild bootstrap'ta $\\pm1$ değerlerini eşit olasılıkla alan çarpana **(1)** ______ değişkeni denir.",
        answer=FillBlanks((TextBlank(("Rademacher", "radamacher", "rademaher"), "Rademacher"),)),
        explanation=(
            "$Y_i^*=X_i'\\hat\\beta+\\hat e_i\\xi_i^*$, $P(\\xi_i^*=1)=P(\\xi_i^*=-1)=1/2$: artığın büyüklüğü korunur, işareti "
            "rastgele değişir (§10.10)."
        ),
    ),
    Question(
        key="b03", concept="hansen-tekrar-onerisi", note=_note("10.12"),
        prompt=(
            "Hansen nihai raporlama için asgari $B=$ **(1)** ______, nihai hassas hesaplar için $B=$ **(2)** ______ önerir."
        ),
        answer=FillBlanks((
            TextBlank(("1000", "1.000", "1 000", "bin"), "1.000"),
            TextBlank(("10000", "10.000", "10 000", "on bin"), "10.000"),
        )),
        explanation=(
            "Hızlı kontrol için $B\\approx100$ yeterli olabilir. Kuyruk kantillerine dayanan aralıklar ve küçük p-değerleri "
            "daha fazla tekrar ister (§10.12)."
        ),
    ),
    Question(
        key="b04", concept="blok-bootstrap", note=_note("10.14"),
        prompt=(
            "Zaman serisinde bağımlılığı korumak için ardışık gözlemleri birlikte yeniden örnekleyen yönteme **(1)** ______ "
            "bootstrap denir."
        ),
        answer=FillBlanks((TextBlank(("blok", "block", "blok bootstrap", "block bootstrap"), "blok"),)),
        explanation=(
            "Genel ilke aynıdır: yeniden örnekleme birimi veri üretim sürecindeki bağımsız bilgi birimine karşılık gelmelidir; "
            "kümeli veride bu birim kümedir (§10.14)."
        ),
    ),
    Question(
        key="b05", concept="cps-bootstrap-sonuclari", note=_note("10.17", 2),
        prompt=(
            "CPS laboratuvarında 1.000 tekrarlı pairs bootstrap standart hatası **(1)** ______ (beş ondalık), percentile "
            "%95 aralığı [**(2)** ______; **(3)** ______]'dir (dört ondalık)."
        ),
        answer=FillBlanks((
            NumberBlank(0.00109, 0.000005, "0,00109"),
            NumberBlank(0.1127, 0.00005, "0,1127"),
            NumberBlank(0.1169, 0.00005, "0,1169"),
        )),
        explanation=(
            "HC1 standart hatası 0,00107'dir: bu örnekte analitik dayanıklı SH ile bootstrap neredeyse aynıdır. Tohum 807 "
            "ile uygulama ve Python kodu bu sayıları birebir verir (§10.17, Adım 2)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="f01", concept="secilmeme-olasiligi", note=_note("10.1"),
        prompt=(
            "$n$ gözlemli örneklemden yerine koyarak $n$ kez çekildiğinde belirli bir gözlemin bootstrap örneğinde hiç "
            "yer almama olasılığını yazın."
        ),
        answer=Equation(
            lhs=r"P(\text{seçilmez})",
            symbols=(Symbol("n", "n", "örneklem büyüklüğü", 2.0, 200.0),),
            answer="(1 - 1/n)^n",
            shown=r"\left(1-\frac1n\right)^n\approx e^{-1}\approx0{,}368",
        ),
        explanation=(
            "Her çekilişte seçilmeme olasılığı $1-1/n$'dir. Büyük $n$'de bu olasılık $e^{-1}$'e yaklaşır: bir bootstrap "
            "örneği ortalama olarak farklı gözlemlerin yaklaşık yüzde 63'ünü içerir (§10.1)."
        ),
    ),
    Question(
        key="f02", concept="jackknife-carpani", note=_note("10.2"),
        prompt=(
            "$S=\\sum_{i=1}^n(\\hat\\theta_{(-i)}-\\bar\\theta_{(\\cdot)})^2$ olsun. Jackknife varyans tahminini $n$ ve $S$ "
            "cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\widehat{\operatorname{Var}}_{jack}",
            symbols=(
                Symbol("n", "n", "gözlem sayısı", 5.0, 500.0),
                Symbol("S", "S", "dışarıda bırakma tahminlerinin kareler toplamı", 0.1, 3.0),
            ),
            answer="(n - 1)/n*S",
            shown=r"\frac{n-1}{n}\,S",
        ),
        explanation=(
            "Bir gözlemi çıkarmak tahmini çok az değiştirir; ham yayılım gerçek değişkenliği küçük gösterir. $(n-1)/n$ "
            "çarpanı ortalama için tam doğru varyansı verecek biçimde seçilmiştir (§10.2)."
        ),
    ),
    Question(
        key="f03", concept="bootstrap-t-istatistigi", note=_note("10.8"),
        prompt=(
            "Bir tekrarın tahmini $\\hat\\theta_b^*$, orijinal tahmin $\\hat\\theta$ ve tekrarın standart hatası "
            "$s(\\hat\\theta_b^*)$ olsun. Percentile-$t$ için bootstrap $t$ istatistiğini $T_b^*$ yazın."
        ),
        answer=Equation(
            lhs="T_b^*",
            symbols=(
                Symbol("tb", r"\hat\theta_b^*", "tekrarın tahmini", -2.0, 2.0, aliases=("θb", "theta_b")),
                Symbol("th", r"\hat\theta", "orijinal tahmin", -2.0, 2.0, aliases=("θ", "theta")),
                Symbol("sb", r"s(\hat\theta_b^*)", "tekrarın standart hatası", 0.2, 2.0),
            ),
            answer="(tb - th)/sb",
            shown=r"\frac{\hat\theta_b^*-\hat\theta}{s(\hat\theta_b^*)}",
        ),
        explanation=(
            "Pay tekrarın orijinal tahminden sapmasıdır; payda o tekrarın kendi standart hatasıdır. Aralık "
            "$[\\hat\\theta-q^*_{1-\\alpha/2}s(\\hat\\theta),\\ \\hat\\theta-q^*_{\\alpha/2}s(\\hat\\theta)]$ olur (§10.8)."
        ),
    ),
    Question(
        key="f04", concept="p-degeri-mc-hatasi", note=_note("10.12"),
        prompt="Bootstrap p-değeri $\\hat p$ ve tekrar sayısı $B$ ise p-değerinin Monte Carlo standart hatasını yazın.",
        answer=Equation(
            lhs=r"se_{MC}(\hat p)",
            symbols=(
                Symbol("p", r"\hat p", "bootstrap p-değeri", 0.01, 0.5),
                Symbol("B", "B", "tekrar sayısı", 100.0, 10000.0),
            ),
            answer="sqrt(p*(1 - p)/B)",
            shown=r"\sqrt{\hat p(1-\hat p)/B}",
        ),
        explanation=(
            "Bootstrap p-değeri yaklaşık bir binom oranıdır. Küçük p-değerlerini hassas raporlamak için $B$'nin büyük "
            "olması gerekir: $B$ seçimi sayısal hassasiyet kararıdır (§10.12)."
        ),
    ),
    Question(
        key="f05", concept="percentile-donusum-ucu", note=_note("10.7"),
        prompt=(
            "$\\beta$ için percentile %95 aralığının alt ucu $a$ ise, yüzde etki $g(\\beta)=100(e^\\beta-1)$ için percentile "
            "aralığının alt ucunu yazın."
        ),
        answer=Equation(
            lhs=r"g_{alt}",
            symbols=(Symbol("a", "a", "β aralığının alt ucu", -1.0, 1.0),),
            answer="100*(exp(a) - 1)",
            shown=r"100\,(e^{a}-1)",
        ),
        explanation=(
            "$g$ monoton artan olduğu için $g(\\hat\\beta^*)$'lerin sıralaması $\\hat\\beta^*$'lerinkiyle aynıdır; uçlara $g$ "
            "uygulamak yeterlidir. Delta yöntemi ise simetrik bir aralık kurar (§10.7)."
        ),
    ),
)


KONU10_QUIZ = QuestionSet(
    topic_key="konu10",
    title="Konu 10: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set jackknife ve bootstrap'ın mantığını, bootstrap standart hatasını, normal, percentile ve percentile-$t$ "
        "aralıklarını, pairs, parametrik ve wild bootstrap'ı, tekrar sayısını ve seed'i, yeniden örnekleme birimini ve "
        "CPS laboratuvarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir bölüme bağlıdır; yanlış "
        "cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Bölüm sonu egzersizlerini tekrar etmez, onları "
        "tamamlar."
    ),
    sections=("10.1", "10.2", "10.3", "10.4", "10.5", "10.6", "10.7", "10.8", "10.9", "10.10", "10.11", "10.12",
              "10.13", "10.14", "10.15", "10.16", "10.17"),
)
