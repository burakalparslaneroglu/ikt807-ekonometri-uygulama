"""Konu 9 "Kendini sına" soru seti: keskin ve bulanık RDD, yerel doğrusal tahmin, bant genişliği ve tanılar.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Bölüm sonu egzersizlerindeki konular
(tedavi kuralının yönü, verilen limitlerden sıçrama, verilen katsayılarla iki doğru, h = 3 ile h = 10'un
karşılaştırılması ve p-değerine göre bant seçimi, Head Start'ta h = 4 ile h = 12 farkı, not yuvarlama
manipülasyonu, yaş dengesizliği, sayısal bulanık RDD oranları, zayıf ilk aşama hesabı, plasebo sonuç,
eşikten uzak birimlere genelleme, raporlama paketi) tekrar edilmez.
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
        key="k01", concept="keskin-rdd-estimand", note=_note("9.2"),
        prompt="Keskin (sharp) RDD'de eşikteki sıçrama, süreklilik varsayımı altında hangi büyüklüğü tanımlar?",
        answer=MultipleChoice((
            "Bütün anakütlede ortalama tedavi etkisini $\\mathbb E[Y_1-Y_0]$",
            "Eşikteki birimler için ortalama tedavi etkisini $\\theta(c)=\\mathbb E[Y_1-Y_0\\mid X=c]$",
            "Tedavi edilenlerde ortalama tedavi etkisini",
            "$Y$ ile $X$ arasındaki doğrusal ilişkinin eğimini",
        ), correct=1),
        explanation=(
            "$\\theta(c)=\\lim_{x\\downarrow c}\\mathbb E[Y\\mid X=x]-\\lim_{x\\uparrow c}\\mathbb E[Y\\mid X=x]$: sağ ve sol "
            "limitlerin farkı eşikteki birimler için etkiyi verir; başka $X$ değerlerine taşımak ek varsayım ister (§9.2)."
        ),
    ),
    Question(
        key="k02", concept="sureklilik-varsayimi-icerigi", note=_note("9.3"),
        prompt="RDD'deki süreklilik varsayımı ne söyler?",
        answer=MultipleChoice((
            "$X$ ile $Y$ arasında ilişki olmadığını",
            "Eşiğin iki yanında eşit sayıda gözlem bulunduğunu",
            "Tedavi dışında sonucu tam eşikte sıçratan başka bir mekanizma olmadığını: potansiyel sonuçların koşullu "
            "ortalamaları eşikte süreklidir",
            "Tedavi etkisinin bütün $X$ değerlerinde aynı olduğunu",
        ), correct=2),
        explanation=(
            "$Y$, $X$ ile güçlü biçimde değişebilir; gereken, $m_0$ ve $m_1$'in eşikte sıçramamasıdır. Aynı eşikte başka "
            "bir politika başlıyorsa sıçrama iki politikanın toplam etkisini yansıtır. Varsayım doğrudan test edilemez (§9.3)."
        ),
    ),
    Question(
        key="k03", concept="hansen-bant-olcegi", note=_note("9.4"),
        prompt="Hansen'in birim varyanslı çekirdeklerinde bant genişliği $h$ neyi ifade eder?",
        answer=MultipleChoice((
            "Çekirdeğin standart sapmasını; farklı çekirdeklerde aynı $h$ aynı yerelliği ifade eder",
            "Pencerenin yarı genişliğini",
            "Eşiğin iki yanındaki gözlem sayısını",
            "Yerel polinomun derecesini",
        ), correct=0),
        explanation=(
            "Hansen çekirdekleri varyansı 1 olacak biçimde ölçekler. Üçgen çekirdekte ağırlık eşikten $h\\sqrt6$, dikdörtgende "
            "$h\\sqrt3$ uzaklıkta sıfırlanır; birçok yazılım ise $h$'yi pencerenin yarı genişliği olarak kullanır (§9.4, §9.8)."
        ),
    ),
    Question(
        key="k04", concept="undersmoothing", note=_note("9.6"),
        prompt="Hansen, RDD güven aralığının düzgünleştirme yanlılığı nedeniyle bozulmasına karşı hangi önlemi vurgular?",
        answer=MultipleChoice((
            "Bütün veriye global yüksek dereceli polinom uydurmak",
            "Bant genişliğini en küçük p-değerini verecek biçimde seçmek",
            "Modele çok sayıda kovaryat eklemek",
            "AMSE-optimal banttan daha küçük bir bant seçmek (undersmoothing) ve iki tarafta ortak bant kullanmak",
        ), correct=3),
        explanation=(
            "Daha küçük bant yanlılığı azaltır, bedeli daha büyük standart hatadır. Güncel literatürde yanlılığı açıkça "
            "düzelten (robust bias-corrected) prosedürler de kullanılır (§9.6)."
        ),
    ),
    Question(
        key="k05", concept="yogunluk-testi-yorumu", note=_note("9.10"),
        prompt="Eşik değişkeninin yoğunluğunda eşikte sıçrama bulunmaması neyi gösterir?",
        answer=MultipleChoice((
            "Tasarımın geçerli olduğunu kanıtlar",
            "Manipülasyon olmadığıyla uyumludur; diğer tanımlama tehditlerini otomatik olarak dışlamaz",
            "Tedavi etkisinin sıfır olduğunu gösterir",
            "Bant genişliğinin doğru seçildiğini gösterir",
        ), correct=1),
        explanation=(
            "Yoğunluk testi belirli bir ihlal türüne, eşik çevresinde yığılmaya bakar. Reddetmemesi, aynı eşikte başka "
            "politika veya potansiyel sonuç süreksizliği gibi tehditleri dışlamaz (§9.10)."
        ),
    ),
    Question(
        key="k06", concept="hansen-grafik-nesnesi", note=_note("9.11"),
        prompt="Hansen'in önerdiği temel RDD grafiğinde ana görsel nesne nedir?",
        answer=MultipleChoice((
            "Eşiğin iki yanında ayrı yerel doğrusal koşullu ortalama tahmini ve güven bantları",
            "Kutulanmış ortalamalar (binned means)",
            "Bütün veriye uydurulmuş global dördüncü derece polinom",
            "Yalnız ham gözlemlerin serpilme grafiği",
        ), correct=0),
        explanation=(
            "Kutulanmış ortalamalar ham veri değildir; dikdörtgen çekirdekli kaba bir tahmin edicidir. Ana görsel kanıt "
            "yerel tahmin ve belirsizliğidir; ham gözlemler ikincil katman olabilir (§9.11)."
        ),
    ),
    Question(
        key="k07", concept="olcek-farki-replikasyon", note=_note("9.15", 3),
        prompt="LM2007'de Hansen'in $h=8$'i pencerenin yarı genişliği olarak okunursa (pencere $\\pm8$) ne olur?",
        answer=MultipleChoice((
            "Tahmin değişmez, çünkü çekirdek türü aynıdır",
            "Pencere genişler ve tahmin sıfıra yaklaşır",
            "Pencere daralır; 482 ilçe kalır ve tahmin $-2{,}25$ (SH 1,08) olur",
            "Yalnız standart hata küçülür",
        ), correct=2),
        explanation=(
            "Hansen ölçeğinde $h=8$ üçgen çekirdekle $\\pm19{,}6$'lık pencereye ve $-1{,}51$ (0,71) tahminine karşılık gelir. "
            "Fark yöntem değil ölçek farkıdır; replikasyonda önce bu tür tanımlar karşılaştırılır (§9.15, Adım 3)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="rdd-bilgi-kaynagi", note=_note("9.1"),
        prompt="RDD bilgiyi esas olarak bütün örneklemdeki tedavi edilen ve edilmeyen grupları karşılaştırarak üretir.",
        answer=TrueFalse(False),
        explanation=(
            "RDD esas olarak eşiğe çok yakın gözlemlerden bilgi alır; eşikten uzak grupları karşılaştırmak güçlü seçilim "
            "farkları içerir. Bu yerellik hem tasarımın gücü hem dış geçerlilik sınırıdır (§9.1)."
        ),
    ),
    Question(
        key="d02", concept="tanimlamanin-kaynagi", note=_note("9.2"),
        prompt="RDD'nin nedensel gücü, seçilen regresyon polinomunun doğru olmasından gelir.",
        answer=TrueFalse(False),
        explanation=(
            "Tanımlama limit/süreklilik argümanından gelir ve fonksiyonel biçim varsayımı içermez. Yerel doğrusal regresyon, "
            "spline veya polinom iki limiti tahmin etmek için kullanılan araçlardır (§9.2)."
        ),
    ),
    Question(
        key="d03", concept="ortak-egim-kisiti", note=_note("9.4"),
        prompt="Yerel doğrusal RDD'de eşiğin iki yanında aynı eğimi zorlamak ($\\beta_2=0$) zararsız bir sadeleştirmedir.",
        answer=TrueFalse(False),
        explanation=(
            "$\\beta_1$ sol, $\\beta_1+\\beta_2$ sağ eğimdir. Koşullu ortalamanın türevi iki tarafta farklıysa ortak eğim "
            "kısıtı sıçrama tahminini de bozar (§9.4)."
        ),
    ),
    Question(
        key="d04", concept="cekirdek-ikinci-derece", note=_note("9.7"),
        prompt=(
            "LM2007'de bant genişliği aynı ölçekte tanımlandığında üçgen çekirdek ($-1{,}51$) ve dikdörtgen çekirdek "
            "($-1{,}55$) birbirine yakın tahmin verir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Hansen ölçeğinde $h=8$: üçgende pencere $\\pm19{,}6$, dikdörtgende $\\pm13{,}86$. Yakınlık, çekirdek seçiminin "
            "bant genişliğine göre ikinci derecede kaldığını gösterir (§9.7)."
        ),
    ),
    Question(
        key="d05", concept="kovaryat-rolu", note=_note("9.9"),
        prompt=(
            "Geçerli bir keskin RDD'de tanımlama kovaryat kontrolüne ihtiyaç duymaz; önceden belirlenmiş kovaryatlar esas "
            "olarak hassasiyeti artırır."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Temel tanımlamayı eşikteki süreklilik sağlar. Kovaryat eklemek manipülasyon veya aynı eşikteki başka bir "
            "politika gibi tasarım ihlallerini düzeltmez (§9.9)."
        ),
    ),
    Question(
        key="d06", concept="bulanik-rdd-iv", note=_note("9.12"),
        prompt=(
            "Bulanık RDD'de eşiği geçme göstergesi tedavi için araç rolü görür: sonuç sıçraması indirgenmiş biçim, tedavi "
            "olasılığındaki sıçrama ilk aşamadır."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Yerel Wald oranı $\\theta_{FRD}=\\Delta_Y/\\Delta_D$, Konu 4'teki IV mantığıyla aynıdır. Yorum (koşullu ortalama "
            "etki mi, uyucular için LATE mi) kullanılan varsayımlara bağlıdır (§9.12)."
        ),
    ),
    Question(
        key="d07", concept="kovaryat-sicramasi-kanit-degil", note=_note("9.13"),
        prompt="Tedaviden önce belirlenmiş bir kovaryatta eşikte belirgin sıçrama bulunması tasarımı destekleyen bir kanıttır.",
        answer=TrueFalse(False),
        explanation=(
            "Tersine: önceden belirlenmiş özelliklerde eşikte sıçrama beklenmez. Sıçrama, iki yandaki birimlerin "
            "karşılaştırılabilirliğini sorgulatır (§9.13)."
        ),
    ),
    # --- Boşluk doldurma ---------------------------------------------------------
    Question(
        key="b01", concept="running-variable-adi", note=_note("9.1"),
        prompt=(
            "Tedaviyi belirleyen sürekli değişkene eşik değişkeni, İngilizce literatürde **(1)** ______ denir."
        ),
        answer=FillBlanks((TextBlank(
            ("running variable", "running", "forcing variable", "assignment variable", "running değişkeni"),
            "running variable",
        ),)),
        explanation=(
            "Keskin tasarımda tedavi tamamen eşik tarafından belirlenir: $D=\\mathbf 1\\{X\\ge c\\}$. Eşik (cut-off) $c$ "
            "politika kuralından gelir (§9.1)."
        ),
    ),
    Question(
        key="b02", concept="headstart-esik", note=_note("9.7"),
        prompt=(
            "Head Start uygulamasında eşik değişkeni 1960 yoksulluk oranıdır ve eşik $c=$ **(1)** ______'tür (dört ondalık)."
        ),
        answer=FillBlanks((NumberBlank(59.1984, 0.00005, "59,1984"),)),
        explanation=(
            "Federal hükümet 1965'te en yoksul 300 ilçeye başvuru desteği verdi; eşik 300. ilçenin yoksulluk oranıdır. "
            "LM2007'de 2.783 ilçeden 294'ü eşiğin sağındadır (§9.7, §9.15)."
        ),
    ),
    Question(
        key="b03", concept="h8-tahmin-ve-sh", note=_note("9.8"),
        prompt=(
            "LM2007'de üçgen çekirdek ve $h=8$ (Hansen ölçeği) ile eşikteki sıçrama **(1)** ______, HC1 standart hatası "
            "**(2)** ______'dir (iki ondalık)."
        ),
        answer=FillBlanks((NumberBlank(-1.51, 0.005, "−1,51"), NumberBlank(0.71, 0.005, "0,71"))),
        explanation=(
            "Bu, Hansen'in tercih ettiği yerel tahmindir: 1.041 ilçe pozitif ağırlık alır; eşiğin solunda tahmini ölüm "
            "oranı 3,31, sağında 1,80'dir (§9.7, §9.8)."
        ),
    ),
    Question(
        key="b04", concept="zayif-ilk-asama-adi", note=_note("9.12"),
        prompt=(
            "Bulanık RDD'de paydadaki tedavi olasılığı sıçraması çok küçükse oran çok oynaklaşır; bu, **(1)** ______ "
            "probleminin bulanık RDD karşılığıdır."
        ),
        answer=FillBlanks((TextBlank(
            ("zayıf araç", "zayif arac", "zayıf araçlar", "weak instrument", "weak instruments", "zayıf ilk aşama",
             "zayif ilk asama"),
            "zayıf araç",
        ),)),
        explanation=(
            "Paydanın küçüklüğü delta yöntemi standart hatasını da yanıltıcı kılar. Bu yüzden sonuç grafiğinin yanında "
            "tedavi olasılığı grafiği de raporlanır (§9.12)."
        ),
    ),
    Question(
        key="b05", concept="plasebo-esik", note=_note("9.13"),
        prompt=(
            "Tedavinin gerçekte değişmediği $X$ noktalarında kurulan yapay RDD tahminlerine **(1)** ______ eşik kontrolü denir."
        ),
        answer=FillBlanks((TextBlank(
            ("plasebo", "placebo", "sahte", "plasebo eşik", "sahte eşik"),
            "plasebo (sahte)",
        ),)),
        explanation=(
            "Çok sayıda sahte eşikte benzer sıçramalar bulunması, sonuç fonksiyonunun doğal kırılganlığı veya yanlış "
            "fonksiyonel biçim konusunda uyarıdır (§9.13)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="f01", concept="gozlenen-sonuc-denklemi", note=_note("9.2"),
        prompt="Gözlenen sonucu $Y$, potansiyel sonuçlar $Y_0$, $Y_1$ ve tedavi göstergesi $D$ cinsinden yazın.",
        answer=Equation(
            lhs="Y",
            symbols=(
                Symbol("Y0", "Y_0", "tedavi yokken sonuç", -3.0, 3.0, aliases=("Y_0", "Y₀")),
                Symbol("Y1", "Y_1", "tedavi altında sonuç", -3.0, 3.0, aliases=("Y_1", "Y₁")),
                Symbol("D", "D", "tedavi göstergesi", 0.0, 1.0),
            ),
            answer="Y0 + D*(Y1 - Y0)",
            shown=r"Y_0+D(Y_1-Y_0)=(1-D)Y_0+DY_1",
        ),
        explanation=(
            "Keskin tasarımda $x<c$ için yalnız $Y_0$, $x\\ge c$ için yalnız $Y_1$ gözlenir; bu yüzden gözlenen koşullu "
            "ortalama eşikte $m_0$'dan $m_1$'e geçer (§9.2)."
        ),
    ),
    Question(
        key="f02", concept="ucgen-pencere", note=_note("9.4"),
        prompt=(
            "Hansen'in birim varyanslı üçgen çekirdeğinde ağırlığın sıfıra indiği uzaklığı (pencerenin yarı genişliği) "
            "bant genişliği $h$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="W",
            symbols=(Symbol("h", "h", "bant genişliği (Hansen ölçeği)", 0.5, 12.0),),
            answer="h*sqrt(6)",
            shown=r"h\sqrt6",
        ),
        explanation=(
            "$K(u)=\\frac{1}{\\sqrt6}(1-|u|/\\sqrt6)$, $|u|\\le\\sqrt6$ ve ağırlık $K(R_i/h)$ olduğundan $|R_i|=h\\sqrt6$'da "
            "sıfırlanır. LM2007'de $h=8$ için $8\\sqrt6\\simeq19{,}6$ (§9.4, §9.7)."
        ),
    ),
    Question(
        key="f03", concept="birim-varyans-ucgen-cekirdek", note=_note("9.4"),
        prompt=(
            "$0\\le u\\le\\sqrt6$ için Hansen'in birim varyanslı üçgen çekirdeğini $K(u)$ yazın (varyansı 1, integrali 1)."
        ),
        answer=Equation(
            lhs="K(u)",
            symbols=(Symbol("u", "u", "ölçeklenmiş uzaklık", 0.1, 2.4),),
            answer="(1/sqrt(6))*(1 - u/sqrt(6))",
            shown=r"\frac{1}{\sqrt6}\left(1-\frac{u}{\sqrt6}\right)",
        ),
        explanation=(
            "Destek $[-\\sqrt6,\\sqrt6]$ ve yükseklik $1/\\sqrt6$ ile integral 1, varyans 1 olur. Ölçek çarpanı ağırlıklı "
            "EKK'yi değiştirmez; önemli olan pencere genişliğidir (§9.4)."
        ),
    ),
    Question(
        key="f04", concept="rdd-sh-mertebesi", note=_note("9.6"),
        prompt=(
            "Yerel doğrusal sıçrama tahmininde $\\operatorname{Var}(\\hat\\tau)=O(1/(nh))$ ise standart hatanın mertebesini "
            "$n$ ve $h$ cinsinden yazın (sabitler hariç)."
        ),
        answer=Equation(
            lhs=r"se(\hat\tau)\propto",
            symbols=(
                Symbol("n", "n", "örneklem büyüklüğü", 100.0, 5000.0),
                Symbol("h", "h", "bant genişliği", 0.1, 2.0),
            ),
            answer="1/sqrt(n*h)",
            shown=r"(nh)^{-1/2}",
        ),
        explanation=(
            "Etkin gözlem sayısı kabaca $nh$'dir: bant daraldıkça standart hata büyür. Yanlılık ise $O(h^2)$ mertebesindedir "
            "(§9.6)."
        ),
    ),
    Question(
        key="f05", concept="yanlilik-sh-orani", note=_note("9.6"),
        prompt=(
            "Yanlılık $O(h^2)$, standart hata $O((nh)^{-1/2})$ ise yanlılığın standart hataya oranının mertebesini $n$ ve $h$ "
            "cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\frac{\operatorname{Bias}}{se}\propto",
            symbols=(
                Symbol("n", "n", "örneklem büyüklüğü", 100.0, 5000.0),
                Symbol("h", "h", "bant genişliği", 0.1, 2.0),
            ),
            answer="sqrt(n*h^5)",
            shown=r"\sqrt{nh^5}",
        ),
        explanation=(
            "$h^2\\sqrt{nh}=\\sqrt{nh^5}$. AMSE-optimal $h\\propto n^{-1/5}$'te bu oran sıfıra gitmez: klasik aralık yanlılığı "
            "ihmal eder; daha küçük bant (undersmoothing) oranı küçültür (§9.6)."
        ),
    ),
)


KONU09_QUIZ = QuestionSet(
    topic_key="konu09",
    title="Konu 9: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set keskin ve bulanık RDD'nin tanımlama mantığını, süreklilik varsayımını, yerel doğrusal tahmini, bant "
        "genişliğinin ölçeğini ve çıkarımdaki rolünü, grafik ve tasarım tanılarını ve Head Start laboratuvarını sınar. "
        "Her soru tek bir kavrama odaklanır ve notlardaki bir bölüme bağlıdır; yanlış cevaplarınız için tekrar edilecek "
        "bölümler en üstte listelenir. Bölüm sonu egzersizlerini tekrar etmez, onları tamamlar."
    ),
    sections=("9.1", "9.2", "9.3", "9.4", "9.6", "9.7", "9.8", "9.9", "9.10", "9.11", "9.12", "9.13", "9.15"),
)
