"""Konu 4 "Kendini sına" soru seti: araçsal değişkenler ve 2SLS.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Bölüm sonu egzersizlerindeki
hesaplar (Wald oranı, moment türetmesi, rank, J testi, LATE hesabı) tekrar edilmez; buradaki
sorular aynı konuların farklı ve kısa yönlerini sınar.
"""

from __future__ import annotations

from core.labs.spec import NoteRef
from core.quiz.expression import Symbol
from core.quiz.model import (
    Equation,
    FillBlanks,
    MultipleChoice,
    Question,
    QuestionSet,
    TextBlank,
    TrueFalse,
)


def _note(section: str, step: int = 0, *objects: str) -> NoteRef:
    return NoteRef(section, step, tuple(objects))


PI = Symbol("p", r"\pi", "ilk aşama katsayısı", 0.2, 2.0, aliases=("π", "pi"))
BETA = Symbol("b", r"\beta", "yapısal katsayı", -1.0, 2.0, aliases=("β", "beta"))


QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="kosullu-ilgililik", note=_note("4.2.1"),
        prompt="Dışsal kontroller $W$ modeldeyken ilgililik (relevance) koşulu tam olarak neyi ister?",
        answer=MultipleChoice((
            "$\\operatorname{Cov}(Z,X)\\neq0$ olması yeterlidir",
            "$W$'nin doğrusal etkisi ayrıldıktan sonra da $Z$'nin $X$'i açıklamaya devam etmesini",
            "$Z$'nin $Y$ ile güçlü biçimde ilişkili olmasını",
            "$Z$'nin $W$ ile ilişkisiz olmasını",
        ), correct=1),
        explanation=(
            "İlgililik koşulludur: ilk aşama dışsal kontrolleri içermeli ve dışlanmış aracın katsayısı bu kontroller "
            "varken sıfırdan farklı olmalıdır (§4.2.1)."
        ),
    ),
    Question(
        key="k02", concept="tam-tanimlida-asiri-tanimlama-testi-yok", note=_note("4.10"),
        prompt=(
            "Card uygulamasındaki gibi tek araç–tek endojen regresörlü modelde aşırı tanımlama (Sargan/Hansen $J$) "
            "testi için ne söylenebilir?"
        ),
        answer=MultipleChoice((
            "Her zaman reddeder",
            "Aracın geçerli olduğunu kanıtlar",
            "Hesaplanamaz: sınanacak fazladan moment koşulu yoktur",
            "Yalnız heteroskedastisite varsa hesaplanabilir",
        ), correct=2),
        explanation=(
            "Aşırı tanımlama testi, fazladan moment koşullarının birbiriyle uyumunu sınar. Tam tanımlı modelde fazladan "
            "moment yoktur; araç geçerliliği tamamen kurumsal argümana dayanır (§4.10)."
        ),
    ),
    Question(
        key="k03", concept="kontroller-ilk-asamada", note=_note("4.14"),
        prompt=(
            "2SLS'i elle hesaplarken ilk aşamada dışsal kontrolleri unutup yalnız $X$'i $Z$ üzerine regres etmek "
            "neden hatalıdır?"
        ),
        answer=MultipleChoice((
            "Farklı bir moment koşulu kullanılır; sonuç genel olarak doğru 2SLS değildir",
            "Katsayı değişmez, yalnız standart hata değişir",
            "İlk aşama F istatistiği kendiliğinden 10'u geçer",
            "Kontroller zaten yalnız ikinci aşamada yer almalıdır",
        ), correct=0),
        explanation=(
            "Dışsal kontroller hem ilk hem ikinci aşamada yer alır; yazılım 2SLS'i tek bir sistem olarak çözer. İlk "
            "aşamadan kontrolü çıkarmak farklı bir tahmin edici üretir (§4.14)."
        ),
    ),
    Question(
        key="k04", concept="ilk-asama-f-turu", note=_note("4.14"),
        prompt="Yazılımın otomatik raporladığı klasik ilk aşama $F$ istatistiği hangi durumda yanıltıcı olabilir?",
        answer=MultipleChoice((
            "Örneklem büyük olduğunda",
            "Araç ikili bir değişken olduğunda",
            "Model tam tanımlı olduğunda",
            "Hatalar heteroskedastik veya kümelenmiş olduğunda",
        ), correct=3),
        explanation=(
            "Çıktıdaki \"first stage F\" değerinin hangi tür F olduğu kontrol edilmelidir; heteroskedastik veya kümeli "
            "tasarımda dayanıklı ilk aşama istatistiği gerekir. Card uygulamasında F, HC1 t istatistiğinin karesidir (§4.14)."
        ),
    ),
    Question(
        key="k05", concept="monotonluk-defier-yok", note=_note("4.12"),
        prompt="LATE yorumu için gereken monotonluk varsayımı hangi grubun var olmadığını söyler?",
        answer=MultipleChoice((
            "Her zaman alanların (always-takers)",
            "Hiç almayanların (never-takers)",
            "Araç onları tedaviye ittiğinde tedaviden vazgeçenlerin (defiers)",
            "Uyumluların (compliers)",
        ), correct=2),
        explanation=(
            "Monotonluk $P\\{D(1)-D(0)<0\\}=0$ demektir: araç kimsenin tedavi alma olasılığını düşürmez. Defier olsaydı "
            "Wald oranı uyumluların etkisinden farklı bir karışımı ölçerdi (§4.12)."
        ),
    ),
    Question(
        key="k06", concept="xhat-anlami", note=_note("4.7"),
        prompt="2SLS'de $\\widehat X=P_ZX$ neyi temsil eder?",
        answer=MultipleChoice((
            "$X$'in araçlar ve dışsal kontroller tarafından açıklanan bileşenini",
            "$X$'in yapısal hatayla ilişkili kısmını",
            "$Y$'nin tahmin edilen değerini",
            "İlk aşama artıklarını",
        ), correct=0),
        explanation=(
            "$P_Z$, $Z$'nin sütun uzayına projeksiyondur. 2SLS $X$'in yalnız araç kaynaklı, geçerli araç altında yapısal "
            "hatayla ortogonal bileşenini kullanır; Uygulama Adım 4'te bu bileşen elle oluşturulur (§4.7)."
        ),
    ),
    Question(
        key="k07", concept="hausman-reddetmeme-yorumu", note=_note("4.11"),
        prompt="Bir Durbin–Wu–Hausman içsellik testi sıfır hipotezini reddetmiyor. Doğru yorum hangisidir?",
        answer=MultipleChoice((
            "OLS kesinlikle nedensel etkiyi ölçer",
            "Araç kesinlikle geçersizdir",
            "IV kullanmak artık yanlıştır",
            "OLS ile IV arasında sistematik fark bulunamamıştır; bu dışsallığın kanıtı değildir, testin gücü araç kalitesine bağlıdır",
        ), correct=3),
        explanation=(
            "Test geçerli ve yeterince güçlü bir araca dayanır. Zayıf veya geçersiz araçla OLS–IV farkı güvenilir bir "
            "içsellik tanısı değildir; ekonomik endojenlik argümanının yerine geçmez (§4.11)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="guclu-ilk-asama-dislamayi-kanitlamaz", note=_note("4.2.3"),
        prompt="İlk aşamanın güçlü ve istatistiksel olarak anlamlı olması, aracın dışlama kısıtını sağladığını gösterir.",
        answer=TrueFalse(False),
        explanation=(
            "Güçlü ilk aşama yalnız ilgililiğin kanıtıdır. Dışsallık ve dışlama kısıtı çoğunlukla veriden test edilemez; "
            "ekonomik ve kurumsal argümanla savunulur. Sezgi Deney 1'de güçlü π ile bile κ ≠ 0 yanlılık üretir (§4.2.3)."
        ),
    ),
    Question(
        key="d02", concept="f10-evrensel-degil", note=_note("4.9"),
        prompt="\"İlk aşama $F>10$\" kuralı, her tasarımda aracın yeterince güçlü olduğunu garanti eden evrensel bir eşiktir.",
        answer=TrueFalse(False),
        explanation=(
            "Kaba bir kuraldır. Araç sayısı, endojen regresör sayısı, heteroskedastisite ve hedeflenen testin boyut "
            "bozulması daha dikkatli bir zayıf araç değerlendirmesi gerektirir (§4.9)."
        ),
    ),
    Question(
        key="d03", concept="kucuk-payda-kararsizlik", note=_note("4.4"),
        prompt=(
            "İkili araçla Wald oranının paydası (ilk aşama farkı) sıfıra yaklaştıkça, küçük örnekleme dalgalanmaları "
            "bile tahmini çok değiştirebilir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Araç $X$'i yeterince hareket ettirmiyorsa, $Y$'deki araç kaynaklı değişimi $X$ birimine çevirmek kararsızlaşır: "
            "zayıf araç probleminin en basit sezgisi (§4.4)."
        ),
    ),
    Question(
        key="d04", concept="elle-ikinci-asama-sh", note=_note("4.17", 4),
        prompt=(
            "Card uygulamasında elle ikinci aşamanın katsayısı 2SLS ile aynı (0,1315) olduğu için, o regresyonun HC1 "
            "standart hatası da 2SLS'in doğru standart hatasıdır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Katsayı aynıdır, çıkarım değil: tahmin edilen eğitim aynı örneklemden türetilmiştir ve ikinci aşama artığı "
            "yapısal artık değildir. Uygulama Adım 4'teki tabloda iki standart hata farklıdır (§4.17, Adım 4)."
        ),
    ),
    Question(
        key="d05", concept="farkli-arac-farkli-late", note=_note("4.12"),
        prompt="Tedavi etkileri heterojense, iki farklı geçerli araç farklı IV tahminleri (farklı LATE'ler) üretebilir.",
        answer=TrueFalse(True),
        explanation=(
            "Her araç kendi uyumlularının etkisini tanımlar: üniversite yakınlığı ile mali yardım farklı gruplarda "
            "tedavi kararını değiştirir. \"Etki kimin için?\" sorusu \"araç geçerli mi?\" kadar önemlidir (§4.12)."
        ),
    ),
    Question(
        key="d06", concept="dissallik-test-edilemez", note=_note("4.2.2"),
        prompt=(
            "Dışsallık koşulu $\\mathbb E[Ze]=0$, çoğu gözlemsel IV uygulamasında veriden doğrudan test edilemez; "
            "kurumsal argümanla savunulur."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Yapısal hata gözlenmediği için koşul doğrudan sınanamaz. Araç ile gözlenen kovaryatlar arasındaki denge "
            "destekleyici kanıt sunar ama gözlenmeyenlerle ilişkiyi dışlamaz (§4.2.2)."
        ),
    ),
    Question(
        key="d07", concept="iv-hassasiyet-kaybi", note=_note("4.7"),
        prompt="2SLS, $X$'teki bütün değişkenliği kullandığı için standart hatası genellikle OLS'ninkinden küçüktür.",
        answer=TrueFalse(False),
        explanation=(
            "2SLS yalnız araçların açıkladığı bileşeni kullanır; tanımlama için değişkenliğin bir bölümünden bilinçli "
            "olarak vazgeçer. Bu yüzden standart hatası genellikle daha büyüktür: Card'da 0,0541'e karşı 0,0036 (§4.7)."
        ),
    ),
    # --- Boşluk doldurma ---------------------------------------------------------
    Question(
        key="b01", concept="yakinlik-ilgililik-kanali", note=_note("4.13"),
        prompt="Kolej yakınlığı aracının ilgililik argümanı: koleje yakın yaşamak eğitimin **(1)** ______ düşürür.",
        answer=FillBlanks((TextBlank(
            ("maliyetini", "maliyeti", "maliyet", "masrafını", "masrafi", "maliyetlerini", "fırsat maliyetini"),
            "maliyetini",
        ),)),
        explanation=(
            "Düşük maliyet daha uzun eğitim süresi demektir; ilk aşamanın ekonomik gerekçesi budur. Zor soru, yakınlığın "
            "ücreti eğitim dışındaki kanallardan etkilememesidir (§4.13)."
        ),
    ),
    Question(
        key="b02", concept="uyumlu-tanimi", note=_note("4.12"),
        prompt="Araç 1 olduğunda tedavi alan, araç 0 olduğunda almayan bireylere **(1)** ______ denir.",
        answer=FillBlanks((TextBlank(
            ("uyumlu", "uyumlular", "uyumlu birey", "complier", "compliers", "uyumlu grup"),
            "uyumlular (compliers)",
        ),)),
        explanation=(
            "$D(1)=1$, $D(0)=0$. Monotonluk ve diğer IV koşulları altında Wald oranı bu grubun ortalama etkisini, "
            "LATE'i verir (§4.12)."
        ),
    ),
    Question(
        key="b03", concept="sayma-kosulu", note=_note("4.6"),
        prompt=(
            "Sayma (order) koşuluna göre dışlanmış araç sayısı, endojen regresör sayısından **(1)** ______ olmamalıdır; "
            "bu koşul gereklidir ama yeterli değildir."
        ),
        answer=FillBlanks((TextBlank(("az", "daha az", "küçük", "daha küçük", "az sayıda"), "az"),)),
        explanation=(
            "$\\ell_2\\ge k_2$ gerekli bir sayma koşuludur. Gerçek tanımlama koşulu rank koşuludur: araçlar endojen "
            "regresörleri bağımsız yönlerde hareket ettirmelidir (§4.6)."
        ),
    ),
    Question(
        key="b04", concept="iv-moment-kosulu", note=_note("4.1"),
        prompt="Araçsal değişken yaklaşımının temel moment koşulu $\\mathbb E[\\,$**(1)** ______$\\,]=0$'dır.",
        answer=FillBlanks((TextBlank(
            ("Ze", "Z e", "Z*e", "Z·e", "eZ", "Z'e", "Z(Y-X'β)", "Z(Y-Xβ)", "Z(Y-βX)", "Z(Y-X'b)", "Z(Y-X'beta)",
             "Z*(Y-X'β)", "Z(Y-X′β)"),
            "Ze, yani Z(Y − X'β)",
        ),)),
        explanation=(
            "Araç yapısal hatayla ortogonaldir: $\\mathbb E[Ze]=\\mathbb E[Z(Y-X'\\beta)]=0$. OLS $\\mathbb E[Xe]=0$'a "
            "dayanır; içsellikte bu koşul çöker, IV onun yerine araç momentlerini kullanır (§4.1)."
        ),
    ),
    Question(
        key="b05", concept="zayif-arac-yanlilik-yonu", note=_note("4.9"),
        prompt="Geçerli fakat zayıf bir araçla 2SLS tahmini sonlu örneklemde **(1)** ______ yönünde yanlı olabilir.",
        answer=FillBlanks((TextBlank(
            ("OLS", "OLS'nin", "OLS'ye", "OLS tahmini", "en küçük kareler", "EKK"),
            "OLS",
        ),)),
        explanation=(
            "Araç zayıfladıkça 2SLS, OLS'nin olasılık limitine doğru kayabilir; dağılım normalden sapar ve güven "
            "aralıkları yanıltıcı olabilir. Sezgi Deney 2'de π'yi küçültüp medyanı izleyin (§4.9)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="f01", concept="oran-mantigi", note=_note("4.3"),
        prompt=(
            "Kontrolsüz ve tek araçlı modelde yapısal katsayı $\\beta$'yı indirgenmiş biçim katsayısı $\\lambda$ ve ilk "
            "aşama katsayısı $\\pi$ cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\beta",
            symbols=(
                Symbol("l", r"\lambda", "indirgenmiş biçim katsayısı", -1.0, 1.0, aliases=("λ", "lambda")),
                PI,
            ),
            answer="l/p",
            shown=r"\beta=\dfrac{\lambda}{\pi}",
        ),
        explanation=(
            "$Y=\\beta(\\pi Z+v)+e=(\\beta\\pi)Z+(\\beta v+e)$, yani $\\lambda=\\beta\\pi$. IV'nin oran mantığı buradan "
            "gelir; Card'da $\\hat\\lambda/\\hat\\pi=0{,}04207/0{,}31990\\approx0{,}1315$ (§4.3, §4.17.2)."
        ),
    ),
    Question(
        key="f02", concept="dislama-ihlali-yanliligi", note=_note("4.2.3"),
        prompt=(
            "$X=\\pi Z+v$ ve $Y=\\beta X+\\kappa Z+e$ ($Z$ dışsal, $\\kappa\\neq0$ dışlama ihlali). Tek araçlı IV'nin "
            "olasılık limitini $\\beta$, $\\kappa$ ve $\\pi$ cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\operatorname{plim}\hat\beta_{IV}",
            symbols=(
                BETA,
                Symbol("k", r"\kappa", "aracın sonuca doğrudan etkisi", -0.5, 0.5, aliases=("κ", "kappa")),
                PI,
            ),
            answer="b + k/p",
            shown=r"\beta+\dfrac{\kappa}{\pi}",
        ),
        explanation=(
            "İndirgenmiş biçim katsayısı $\\beta\\pi+\\kappa$ olur; ilk aşamaya bölününce $\\beta+\\kappa/\\pi$. Zayıf "
            "ilk aşama (küçük $\\pi$) küçük bir ihlali büyütür. Sezgi Deney 1'de κ ve π ile oynayın (§4.2.3)."
        ),
    ),
    Question(
        key="f03", concept="tek-arac-f-t-kare", note=_note("4.17", 2),
        prompt=(
            "Tek dışlanmış araçlı ilk aşamada robust $F$ istatistiğini katsayı $\\hat\\pi$ ve standart hatası "
            "$\\operatorname{SH}(\\hat\\pi)$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="F",
            symbols=(
                Symbol("p", r"\hat\pi", "ilk aşama katsayısı", 0.1, 1.0, aliases=("π̂", "π", "pi")),
                Symbol("s", r"\operatorname{SH}(\hat\pi)", "katsayının standart hatası", 0.02, 0.3,
                       aliases=("SH", "SE", "se")),
            ),
            answer="(p/s)^2",
            shown=r"F=\left(\dfrac{\hat\pi}{\operatorname{SH}(\hat\pi)}\right)^2",
        ),
        explanation=(
            "Tek kısıtta Wald $F$ istatistiği $t$'nin karesidir. Card'da $(0{,}3199/0{,}0851)^2\\approx14{,}14$ "
            "(§4.17, Adım 2)."
        ),
    ),
    Question(
        key="f04", concept="ate-tip-karisimi", note=_note("4.12"),
        prompt=(
            "Anakütlede her zaman alan, uyumlu ve hiç almayan payları $p_a$, $p_c$, $p_n$; ortalama etkileri "
            "$\\tau_a$, $\\tau_c$, $\\tau_n$ olsun (defier yok). ATE'yi yazın."
        ),
        answer=Equation(
            lhs="ATE",
            symbols=(
                Symbol("pa", "p_a", "her zaman alan payı", 0.05, 0.4, aliases=("p_a",)),
                Symbol("pc", "p_c", "uyumlu payı", 0.1, 0.6, aliases=("p_c",)),
                Symbol("pn", "p_n", "hiç almayan payı", 0.05, 0.4, aliases=("p_n",)),
                Symbol("ta", r"\tau_a", "her zaman alanların etkisi", -1.0, 2.0, aliases=("τ_a", "τa", "tau_a")),
                Symbol("tc", r"\tau_c", "uyumluların etkisi", -1.0, 2.0, aliases=("τ_c", "τc", "tau_c")),
                Symbol("tn", r"\tau_n", "hiç almayanların etkisi", -1.0, 2.0, aliases=("τ_n", "τn", "tau_n")),
            ),
            answer="pa*ta + pc*tc + pn*tn",
            shown=r"ATE=p_a\tau_a+p_c\tau_c+p_n\tau_n",
        ),
        explanation=(
            "ATE bütün tiplerin ağırlıklı ortalamasıdır; IV ise yalnız $\\tau_c$'yi (LATE) tanımlar. İkisi ancak etkiler "
            "tipler arasında aynıysa çakışır; Sezgi Deney 3'te karşılaştırın (§4.12)."
        ),
    ),
    Question(
        key="f05", concept="indirgenmis-bicim-hatasi", note=_note("4.3"),
        prompt=(
            "$Y=\\beta X+e$ ve $X=\\pi Z+v$ iken indirgenmiş biçim $Y=\\lambda Z+\\eta$'nın hata terimi $\\eta$'yı "
            "$\\beta$, $v$ ve $e$ cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\eta",
            symbols=(
                BETA,
                Symbol("v", "v", "ilk aşama hatası", -2.0, 2.0),
                Symbol("e", "e", "yapısal hata", -2.0, 2.0),
            ),
            answer="b*v + e",
            shown=r"\eta=\beta v+e",
        ),
        explanation=(
            "Yapısal denkleme ilk aşamayı yerleştirin: $Y=(\\beta\\pi)Z+(\\beta v+e)$. Araç dışsalsa $Z$ hem $v$ hem $e$ "
            "ile ilişkisizdir; bu yüzden indirgenmiş biçim OLS ile tutarlı tahmin edilir (§4.3)."
        ),
    ),
)


KONU04_QUIZ = QuestionSet(
    topic_key="konu04",
    title="Konu 4: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set araçsal değişkenler ve 2SLS kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir "
        "bölüme bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Bölüm sonu "
        "egzersizlerini tekrar etmez, onları tamamlar."
    ),
    sections=("4.1", "4.2.1", "4.2.2", "4.2.3", "4.3", "4.4", "4.6", "4.7", "4.9", "4.10", "4.11", "4.12", "4.14",
              "4.13", "4.17"),
)
