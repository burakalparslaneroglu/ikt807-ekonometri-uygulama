"""Konu 12 "Kendini sına" soru seti: yüksek boyutlu kontroller altında hedef katsayı, double selection,
artıklaştırma, ortogonal moment, çapraz uyarlama, DML standart hatası ve bütünleşik araştırma akışı.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Bölüm sonu egzersizlerindeki hesaplar ve
açıklamalar (0,20X ve 1,5X ile eksik değişken yanlılığı, iki seçim kümesinin birleşimi, dört gözlemli artıklaştırma,
naif momentin duyarlılığının sözel açıklaması, n = 1000 için kat büyüklükleri, üç gözlemli DML2, bütün veride
standartlaştırma, DML ve içsellik, sonuç Lasso'sunda sıfır katsayılı kontrol, robust SH gerekçesi, beş seed'li
duyarlılık raporu, dijitalleşme tez planı) tekrar edilmez.
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


_BETA = Symbol("b", r"\beta", "X'in Y'deki doğrudan etkisi", -2.0, 2.0, aliases=("β", "beta"))
_GAMMA = Symbol("g", r"\gamma", "X'in D'deki etkisi", -2.0, 2.0, aliases=("γ", "gamma"))

QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="yardimci-bilesen", note=_note("12.1"),
        prompt="Kısmen doğrusal modelde $Y=D\\theta+X'\\beta+e$ için yardımcı (nuisance) bileşen hangisidir?",
        answer=MultipleChoice((
            "Hedef parametre $\\theta$",
            "$X$'in etkisini taşıyan $\\beta$ (genel olarak $X$'in fonksiyonları)",
            "Temel ilgi değişkeni $D$",
            "Hata terimi $e$",
        ), correct=1),
        explanation=(
            "Yardımcı bileşen araştırma sorusunun doğrudan hedefi değildir ama $\\theta$'nın güvenilir tahmini için "
            "öğrenilmesi gerekir; DML onu esnek biçimde tahmin ederken $\\theta$ için kök-$n$ çıkarımı korumaya çalışır "
            "(§12.1)."
        ),
    ),
    Question(
        key="k02", concept="yaklasik-seyreklik", note=_note("12.3"),
        prompt="Double selection Lasso'nun asimptotik sonucu hangi yapısal varsayıma dayanır?",
        answer=MultipleChoice((
            "Kontrollerin birbirinden bağımsız olmasına",
            "Hataların normal dağılmasına",
            "Yaklaşık seyrekliğe: büyük kontrol uzayının görece küçük bir aktif değişken kümesiyle iyi yaklaşıklanmasına",
            "Kontrol sayısının gözlem sayısının onda birinden az olmasına",
        ), correct=2),
        explanation=(
            "Bu yapı yoksa Lasso yardımcı modelleri yeterince iyi öğrenemeyebilir; \"çok değişken var\" tek başına Lasso "
            "gerekçesi değildir (§12.3)."
        ),
    ),
    Question(
        key="k03", concept="artiklastirma-kokeni", note=_note("12.4"),
        prompt="Artıklaştırma (partialling-out) tahmin edicisi hangi klasik sonuçların yüksek boyutlu devamıdır?",
        answer=MultipleChoice((
            "Gauss–Markov teoremi",
            "Merkezi limit teoremi",
            "Slutsky teoremi",
            "FWL teoremi ve Robinson'ın kısmen doğrusal tahmini",
        ), correct=3),
        explanation=(
            "$Y$ ve $D$'den $X$'in öngörülebilir kısmı çıkarılır ve $\\theta$ artıklar arasındaki eğimden öğrenilir. "
            "Değişen yalnız yardımcı fonksiyonları tahmin etme aracıdır (§12.4; Konu 2 ve Konu 8)."
        ),
    ),
    Question(
        key="k04", concept="tek-bolme-verimsizlik", note=_note("12.6"),
        prompt=(
            "Yardımcı modelleri A yarısında öğrenip $\\theta$'yı yalnız B yarısında tahmin eden basit örneklem bölmenin "
            "çapraz uyarlamaya göre dezavantajı nedir?"
        ),
        answer=MultipleChoice((
            "Verinin yalnız bir kısmı nihai hedef tahmininde kullanılır ve sonuç hangi gözlemlerin hangi tarafa düştüğüne "
            "duyarlıdır",
            "Aşırı uyumdan doğan bağımlılığı artırır",
            "Hedef katsayının standart hatası hesaplanamaz",
            "Yardımcı modeller hiç öğrenilemez",
        ), correct=0),
        explanation=(
            "Bölme aşırı uyum bağımlılığını kırar; bedeli bilgi kaybıdır. Çapraz uyarlama kat rollerini döndürerek her "
            "gözlemin nihai tahmine katkı vermesini sağlar (§12.6)."
        ),
    ),
    Question(
        key="k05", concept="urun-hizi", note=_note("12.8"),
        prompt="DML teorisindeki ürün-hızı (product-rate) koşulu ne söyler?",
        answer=MultipleChoice((
            "Her yardımcı tahmin hatası ayrı ayrı $n^{-1/2}$'den hızlı küçülmelidir",
            "İki yardımcı tahmin hatasının çarpımı $n^{-1/2}$'den hızlı küçülmelidir",
            "Yardımcı modellerin $R^2$'si 0,9'u aşmalıdır",
            "Kat sayısı örneklemle birlikte büyümelidir",
        ), correct=1),
        explanation=(
            "$\\|\\hat m_Y-m_Y\\|_2\\|\\hat m_D-m_D\\|_2=o_p(n^{-1/2})$: ortogonal momentte yardımcı hatalar ikinci "
            "dereceden, çarpım olarak girer. Her yardımcı model tek başına daha yavaş yakınsayabilir (§12.8)."
        ),
    ),
    Question(
        key="k06", concept="ogrenici-duyarliligi", note=_note("12.12"),
        prompt="Notlardaki pratik protokole göre bir tezde DML'in yardımcı modelleri nasıl raporlanmalıdır?",
        answer=MultipleChoice((
            "Yalnız rassal orman kullanılır; teorik dayanağı en güçlü seçenektir",
            "Test hatası en düşük öğrenici seçilir ve yalnız o raporlanır",
            "Ana sonuç Lasso/Post-Lasso ile üretilir; en az bir esnek alternatifle tekrarlanıp duyarlılık tablosu "
            "raporlanır",
            "Öğrenici seçimi sonucu etkilemediği için raporlanması gerekmez",
        ), correct=2),
        explanation=(
            "Lasso'nun hız teorisi en iyi bilinen durumdur. Sonuçlar yakınsa güven artar; belirgin ayrışıyorsa yardımcı "
            "modellerin uyumu incelenmelidir (§12.12)."
        ),
    ),
    Question(
        key="k07", concept="ddk-tedavi-lassosu", note=_note("12.15", 1),
        prompt="DDK laboratuvarında tracking ($D$) için yardımcı Lasso neden hiçbir katta sıfırdan farklı terim seçmez?",
        answer=MultipleChoice((
            "Ceza ızgarası yanlış ölçekte kurulduğu için",
            "$D$ ikili olduğu için Lasso uygulanamaz",
            "Kovaryatlar standartlaştırılmadığı için",
            "Tracking okul düzeyinde rastgele atandığından kovaryatlar $D$'yi öngörmez; iç CV sabit modeli seçer",
        ), correct=3),
        explanation=(
            "Tasarımın izi yardımcı modellerde görünür: $\\hat V$ tracking'in eğitim katındaki ortalamasından sapmasıdır. "
            "$Y$ için Lasso 27 terimden 17 ile 26 arasını tutar (§12.15, Adım 1)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="kucuk-katsayi-kucuk-yanlilik-degil", note=_note("12.2"),
        prompt=(
            "Bir kontrolün $Y$ denklemindeki doğrudan katsayısı küçükse, onu dışarıda bırakmanın $\\theta$ üzerindeki "
            "yanlılığı da mutlaka küçüktür."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Yanlılık yaklaşık $\\beta_j\\operatorname{Cov}(D,X_j)/\\operatorname{Var}(D)$'dir: $X_j$ $D$ ile güçlü "
            "ilişkiliyse küçük $\\beta_j$ bile belirgin yanlılık doğurur (§12.2)."
        ),
    ),
    Question(
        key="d02", concept="neyman-ortogonalligi", note=_note("12.5"),
        prompt=(
            "Neyman ortogonalliği, gerçek yardımcı değerlerde hedef momentin yardımcı bileşene göre türevinin sıfır olması "
            "demektir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Bu yüzden yardımcı tahmindeki küçük hatalar momenti birinci dereceden sürüklemez; kalan etki daha yüksek "
            "derecedendir. Artıklaştırma momentinde $-\\mathbb{E}[VX']=0$'dır (§12.5)."
        ),
    ),
    Question(
        key="d03", concept="capraz-uyarlama-tum-gozlem", note=_note("12.7"),
        prompt=(
            "Çapraz uyarlamada her gözlemin artığı o gözlemi eğitimde görmemiş bir yardımcı modelle hesaplanır ve bütün "
            "gözlemler son aşamada kullanılır."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Her kat bir kez dışarıda bırakılır; artıklar birleştirilir ve $\\hat\\theta$ bütün gözlemlerin artıklarından "
            "hesaplanır. Tek yönlü bölmeye göre veri daha etkin kullanılır (§12.7)."
        ),
    ),
    Question(
        key="d04", concept="kucuk-v-buyuk-sh", note=_note("12.9"),
        prompt="$\\sum_i\\hat V_i^2$ çok küçükse DML standart hatası küçülür ve $\\theta$ daha kesin tahmin edilir.",
        answer=TrueFalse(False),
        explanation=(
            "$\\sum\\hat V_i^2$ varyansın paydasındadır: tedavi $X$ tarafından neredeyse tamamen açıklanıyorsa efektif "
            "varyasyon azdır ve standart hata büyür. Bu, örtüşme ve tanımlama sorununun DML'deki görünümüdür (§12.9)."
        ),
    ),
    Question(
        key="d05", concept="dml-olcum-secim", note=_note("12.10"),
        prompt="DML, kötü ölçülmüş tedavi değişkeni ve seçici örnekleme sorunlarını otomatik olarak çözmez.",
        answer=TrueFalse(True),
        explanation=(
            "DML yüksek boyutlu gözlenen kontrollerin esnek biçimde işlenmesini kolaylaştıran bir tahmin teknolojisidir; "
            "ölçüm hatası, seçim ve zayıf örtüşme tanımlama tasarımıyla ele alınır (§12.10)."
        ),
    ),
    Question(
        key="d06", concept="ceza-secimi-sizintisi", note=_note("12.13"),
        prompt=(
            "Lasso cezasını bütün örneklemde CV ile seçip sonra çapraz uyarlamada bu sabit cezayı kullanmak, ayrılmış kat "
            "bilgisini yardımcı model eğitimine sızdırmaz."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Tuning de yardımcı tahminin parçasıdır; bütün veriyle seçilen ceza ayrılmış kattaki gözlemleri görmüştür. "
            "Ceza her eğitim katında yalnız o katın verisiyle seçilmelidir (§12.13)."
        ),
    ),
    Question(
        key="d07", concept="akis-geri-donus", note=_note("12.16"),
        prompt="Bütünleşik tez araştırma akışı, geri dönüşlere izin vermeyen doğrusal bir komut listesidir.",
        answer=TrueFalse(False),
        explanation=(
            "Tanılama veya veri incelemesi araştırma sorusunu yeniden düşünmeye yol açabilir; önemli olan geri dönüşlerin "
            "şeffaf olmasıdır (§12.16)."
        ),
    ),
    # --- Boşluk doldurma ---------------------------------------------------------
    Question(
        key="b01", concept="kontrollu-karistirici-secimi", note=_note("12.11"),
        prompt=(
            "Kontrollü örnekte yalnız-sonuç Post-Lasso ilk sekiz karıştırıcının **(1)** ______ tanesini, $D$ denklemi "
            "Lasso'su **(2)** ______ tanesini seçmiştir."
        ),
        answer=FillBlanks((
            NumberBlank(0, 0.1, "0"),
            NumberBlank(8, 0.1, "8"),
        )),
        explanation=(
            "Karıştırıcıların $Y$'deki doğrudan etkileri zayıf ($\\pm0{,}10$), $D$'deki etkileri güçlüdür; double "
            "selection birleşimde 14 kontrol ve sekiz karıştırıcının tümünü korur (§12.11)."
        ),
    ),
    Question(
        key="b02", concept="hansen-kat-onerisi", note=_note("12.7"),
        prompt="Hansen'in aktardığı yaygın kat sayısı $K=$ **(1)** ______, tercih edilen DML biçimi **(2)** ______'dir.",
        answer=FillBlanks((
            NumberBlank(10, 0.1, "10"),
            TextBlank(("DML2", "dml2", "DML 2", "DML-2"), "DML2"),
        )),
        explanation=(
            "DML2 bütün katların artıklarını birleştirip tek bir artık regresyonu kurar. Hesaplama maliyeti yaklaşık "
            "$K$ ile artar (§12.7)."
        ),
    ),
    Question(
        key="b03", concept="ddk-bolme-araligi", note=_note("12.15", 3),
        prompt=(
            "DDK laboratuvarında okulların dış katlara dağılım kuralı değiştirildiğinde (11 kural) DML tahmini "
            "**(1)** ______ ile **(2)** ______ arasında değişir (üç ondalık)."
        ),
        answer=FillBlanks((
            NumberBlank(1.267, 0.0006, "1,267"),
            NumberBlank(1.554, 0.0006, "1,554"),
        )),
        explanation=(
            "Veri, öğrenici ve kat sayısı aynıdır; yalnız okulların katlara dağılımı değişir. On bir bölmenin medyanı "
            "1,464'tür; bölme kuralı önceden kaydedilmeli ve değişim raporlanmalıdır (§12.15, Adım 3)."
        ),
    ),
    Question(
        key="b04", concept="agac-varyansi", note=_note("12.12"),
        prompt=(
            "Tek bir regresyon ağacı DML yardımcı modeli olarak nadiren tercih edilir; zayıf yanı yüksek **(1)** ______'tır."
        ),
        answer=FillBlanks((TextBlank(("varyans", "variance", "değişkenlik"), "varyans"),)),
        explanation=(
            "Verinin küçük bir bölümü değişince ağaç yapısı tamamen değişebilir. Rassal orman çok sayıda ağacın "
            "ortalamasını alarak bu varyansı azaltır (§12.12)."
        ),
    ),
    Question(
        key="b05", concept="dml-sh-duzeltmesi", note=_note("12.9"),
        prompt=(
            "Bu bölümün tablolarında DML varyansı, büyük örneklem formülünün $n/(n-1)$ serbestlik düzeltmeli hâliyle "
            "hesaplanır; bu düzeltmenin kısa adı **(1)** ______'dir."
        ),
        answer=FillBlanks((TextBlank(("HC1", "hc1", "HC-1"), "HC1"),)),
        explanation=(
            "Son aşama tek açıklayıcılı, sabitsiz bir regresyondur; gözlemler kümeler hâlinde bağımlıysa skorlar küme "
            "içinde toplanarak küme-dayanıklı varyans hesaplanır (§12.9)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="f01", concept="indirgenmis-bicim-katsayisi", note=_note("12.4"),
        prompt=(
            "Doğrusal modelde $Y=D\\theta+X'\\beta+e$ ve $D=X'\\gamma+V$ ise $Y$'nin $X$ üzerindeki indirgenmiş biçim "
            "katsayısı $\\eta$'yı yazın (tek kontrol için skaler)."
        ),
        answer=Equation(
            lhs=r"\eta",
            symbols=(_BETA, _GAMMA, Symbol("t", r"\theta", "hedef katsayı", -2.0, 2.0, aliases=("θ", "theta"))),
            answer="b + g*t",
            shown=r"\beta+\gamma\theta",
        ),
        explanation=(
            "$D$'yi $Y$ denkleminde yerine koymak $Y=X'(\\beta+\\gamma\\theta)+\\theta V+e$ verir. Lasso ile $Y$'yi $X$ "
            "üzerinde tahmin etmek $\\eta$'yı hedefler, $\\beta$'yı değil (§12.3, §12.4)."
        ),
    ),
    Question(
        key="f02", concept="naif-moment-yanliligi", note=_note("12.5"),
        prompt=(
            "$D=\\gamma X+V$, $Y=\\theta D+\\beta X+\\varepsilon$, $X$ ve $V$ bağımsız, varyansları 1 olsun. Naif "
            "moment $\\beta$ yerine $(1-\\delta)\\beta$ kullanırsa $\\hat\\theta$'nın olasılık limitindeki yanlılığı yazın."
        ),
        answer=Equation(
            lhs=r"\operatorname{plim}\hat\theta-\theta",
            symbols=(Symbol("d", r"\delta", "yardımcı parametre hatası", 0.0, 1.0, aliases=("δ", "delta")), _BETA,
                     _GAMMA),
            answer="d*b*g/(1 + g^2)",
            shown=r"\frac{\delta\beta\gamma}{1+\gamma^2}",
        ),
        explanation=(
            "$\\mathbb{E}[D(Y-D\\theta-X\\tilde\\beta)]=0$ çözümü $\\theta+\\mathbb{E}[DX]\\delta\\beta/\\mathbb{E}[D^2]$ "
            "verir; $\\mathbb{E}[DX]=\\gamma$, $\\mathbb{E}[D^2]=1+\\gamma^2$. Yanlılık $\\delta$ ile doğrusal büyür; "
            "ortogonal momentte $\\delta^2$ ile büyür (§12.5; Sezgi, Deney 2)."
        ),
    ),
    Question(
        key="f03", concept="dml-varyans-toplamlar", note=_note("12.9"),
        prompt=(
            "$A=\\sum_i\\hat\\psi_i^2$ ve $B=\\sum_i\\hat V_i^2$ olsun. Serbestlik düzeltmesi olmadan "
            "$\\widehat{\\operatorname{Var}}(\\hat\\theta_{DML})$'yi $A$ ve $B$ cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\widehat{\operatorname{Var}}(\hat\theta_{DML})",
            symbols=(
                Symbol("A", r"\sum\hat\psi_i^2", "skor karelerinin toplamı", 0.5, 50.0),
                Symbol("B", r"\sum\hat V_i^2", "tedavi artığı karelerinin toplamı", 1.0, 100.0),
            ),
            answer="A/B^2",
            shown=r"\frac{\sum_i\hat\psi_i^2}{\left(\sum_i\hat V_i^2\right)^2}",
        ),
        explanation=(
            "$\\frac1n\\cdot\\frac{A/n}{(B/n)^2}=A/B^2$: sabitsiz tek regresörlü OLS'nin HC0 sandviçidir. HC1 bu değeri "
            "$n/(n-1)$ ile çarpar (§12.9)."
        ),
    ),
    Question(
        key="f04", concept="medyan-birlestirme-terimi", note=_note("12.14"),
        prompt=(
            "Chernozhukov vd. (2018) medyan birleştirmesinde bir bölmenin tahmini $\\hat\\theta_s$, standart hatası "
            "$\\hat\\sigma_s$ ve bölmelerin medyan tahmini $\\hat\\theta_{med}$ olsun. Bölmenin "
            "$\\hat\\sigma^2_{med}$ medyanına giren terimini yazın."
        ),
        answer=Equation(
            lhs=r"\hat\sigma^2_{(s)}",
            symbols=(
                Symbol("s", r"\hat\sigma_s", "bölmenin standart hatası", 0.1, 2.0, aliases=("σs", "σ_s", "sigma_s")),
                Symbol("t", r"\hat\theta_s", "bölmenin tahmini", -2.0, 2.0, aliases=("θs", "θ_s", "theta_s")),
                Symbol("m", r"\hat\theta_{med}", "medyan tahmin", -2.0, 2.0,
                       aliases=("θmed", "θ_med", "theta_med")),
            ),
            answer="s^2 + (t - m)^2",
            shown=r"\hat\sigma_s^2+(\hat\theta_s-\hat\theta_{med})^2",
        ),
        explanation=(
            "İkinci terim bölmeler arası değişkenliği standart hataya ekler; tek bir \"en iyi\" bölmeyi seçmek yerine "
            "bölme belirsizliği raporlanır (§12.14)."
        ),
    ),
    Question(
        key="f05", concept="gozlenmeyen-karistirici-yanliligi", note=_note("12.10"),
        prompt=(
            "$D=X'\\gamma+\\kappa U+V$, $Y=\\theta D+X'\\beta+\\kappa U+\\varepsilon$; $U$, $V$, $\\varepsilon$ "
            "bağımsız standart normal ve $U$ gözlenmiyor. DML yardımcı fonksiyonları doğru öğrense bile "
            "$\\hat\\theta$'nın olasılık limitindeki sapmayı yazın."
        ),
        answer=Equation(
            lhs=r"\operatorname{plim}\hat\theta-\theta",
            symbols=(Symbol("k", r"\kappa", "gözlenmeyen karıştırıcının gücü", 0.0, 2.0, aliases=("κ", "kappa")),),
            answer="k^2/(1 + k^2)",
            shown=r"\frac{\kappa^2}{1+\kappa^2}",
        ),
        explanation=(
            "$U$ dışarıda kalınca tedavi artığı $D-\\mathbb{E}[D\\mid X]=\\kappa U+V$ olur ve sonuç artığı $\\kappa U$'yu "
            "taşır: sapma $\\operatorname{Cov}(\\kappa U+V,\\kappa U)/\\operatorname{Var}(\\kappa U+V)$'dir ve $n$ "
            "büyüse de küçülmez (§12.10; Sezgi, Deney 3)."
        ),
    ),
)


KONU12_QUIZ = QuestionSet(
    topic_key="konu12",
    title="Konu 12: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set yüksek boyutlu kontroller altında hedef katsayı problemini, yalnız sonuç denklemine dayalı seçimin "
        "riskini, double selection'ı, artıklaştırmayı, ortogonal momenti, örneklem bölme ve çapraz uyarlamayı, DML'in "
        "düzenlilik koşullarını ve standart hatasını, DML'in çözmediği tanımlama sorunlarını ve DDK laboratuvarını sınar. "
        "Her soru tek bir kavrama odaklanır ve notlardaki bir bölüme bağlıdır; yanlış cevaplarınız için tekrar edilecek "
        "bölümler en üstte listelenir. Bölüm sonu egzersizlerini tekrar etmez, onları tamamlar."
    ),
    sections=("12.1", "12.2", "12.3", "12.4", "12.5", "12.6", "12.7", "12.8", "12.9", "12.10", "12.11", "12.12",
              "12.13", "12.14", "12.15", "12.16"),
)
