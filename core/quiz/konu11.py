"""Konu 11 "Kendini sına" soru seti: model seçimi, çapraz doğrulama, Ridge, Lasso, Elastic Net, Post-Lasso ve
model seçimi sonrası çıkarım.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Bölüm sonu egzersizlerindeki hesaplar ve
açıklamalar (iki modelin AIC–BIC karşılaştırması, iki tahmin edicinin MSE'si, üç gözlemli LOOCV, beş katın CV
ortalaması, diag(4, 1) ile Ridge çözümü, TL–milyon TL ölçekleme, yumuşak eşikleme hesabı, Post-Lasso'nun ikinci
aşaması, üç cezalı 1se örneği, test MSE'si ile nedensel soru, 100 kontrol kümesi, n = 1500 protokolü) tekrar edilmez.
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


_LAMBDA_Y = Symbol("ly", r"\lambda_y", "yazılımın ceza parametresi", 0.001, 1.0,
                   aliases=("λ_y", "λy", "lambda_y", "lambday"))
_N = Symbol("n", "n", "eğitim gözlem sayısı", 20.0, 5000.0)

QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="tahmin-edici-secimi", note=_note("11.1"),
        prompt="Hansen'e göre ampirik çalışmada \"model seçimi\" çoğu zaman aslında neyin seçimidir?",
        answer=MultipleChoice((
            "Aynı veri üretim süreci için farklı kısıtlar ve tahmin kuralları arasından bir tahmin edicinin seçimi",
            "Hangi veri setinin kullanılacağının seçimi",
            "Hipotez testlerinde anlamlılık düzeyinin seçimi",
            "Kullanılacak istatistik yazılımının seçimi",
        ), correct=0),
        explanation=(
            "Farklı kontrol kümeleri, fonksiyonel biçimler veya yöntemler aynı anakütle ilişkisini farklı tahmin "
            "edicilerle yaklaşıklar; seçim kuralı da tahmin prosedürünün parçası olur (§11.1)."
        ),
    ),
    Question(
        key="k02", concept="parametre-sayisi-sigma", note=_note("11.3"),
        prompt="Doğrusal regresyonda $k$ regresyon katsayısı varsa AIC ve BIC formüllerindeki $K$ kaçtır?",
        answer=MultipleChoice((
            "$k-1$",
            "$k+1$, çünkü hata varyansı $\\sigma^2$ de tahmin edilen bir parametredir",
            "$k$",
            "$2k$",
        ), correct=1),
        explanation=(
            "$\\sigma^2$'yi saymamak, aday modeller aynı sayıda ek parametre içermediğinde sıralamayı değiştirebilir. "
            "Ayrıca $\\hat\\sigma^2$'nin paydası $n$'dir, $n-k$ değil (§11.3)."
        ),
    ),
    Question(
        key="k03", concept="cv-on-isleme-sizintisi", note=_note("11.5"),
        prompt="$K$-katlı çapraz doğrulamada standartlaştırmanın ortalama ve standart sapması nereden öğrenilmelidir?",
        answer=MultipleChoice((
            "Test örneklemi dahil bütün veriden",
            "Yalnız değerlendirme katından",
            "Her turda o turun eğitim katlarından; değerlendirme katına aynı değerlerle uygulanır",
            "Standartlaştırma çapraz doğrulamada gereksizdir",
        ), correct=2),
        explanation=(
            "Veri-temelli bütün ön işlemler kat içinde öğrenilmelidir; aksi hâlde değerlendirme gözlemleri eğitim "
            "sürecine bilgi sızdırır ve performans yapay biçimde iyimser görünür (§11.5)."
        ),
    ),
    Question(
        key="k04", concept="ridge-kucultme", note=_note("11.8"),
        prompt="Ridge katsayıları hakkında hangisi doğrudur?",
        answer=MultipleChoice((
            "Ceza büyüdükçe bazı katsayılar tam sıfıra oturur",
            "$\\lambda=0$ iken Lasso çözümüne döner",
            "$\\lambda>0$ iken yansızdır",
            "Katsayılar sıfıra doğru küçülür ama genellikle tam sıfır olmaz; Ridge bir küçültme yöntemidir",
        ), correct=3),
        explanation=(
            "$(X'X+\\lambda I)^{-1}X'Y$ çözümü her öz-yönde ölçeği $1/d_j$ yerine $1/(d_j+\\lambda)$ yapar; katsayılar "
            "küçülür, sıfırlanmaz. $\\lambda=0$ OLS'dir ve $\\lambda>0$ iken Ridge genel olarak yanlıdır (§11.8)."
        ),
    ),
    Question(
        key="k05", concept="glmnet-alpha-lasso", note=_note("11.11"),
        prompt=(
            "Hansen'in Elastic Net gösteriminde $\\alpha=0$ Lasso'dur. R `glmnet`'te Lasso'yu hangi `alpha` değeri verir?"
        ),
        answer=MultipleChoice((
            "0",
            "1",
            "0,5",
            "Seçilen $\\lambda$ ile aynı değer",
        ), correct=1),
        explanation=(
            "Yazılımlarda karışım parametresi ters yöndedir: `glmnet` `alpha` ve scikit-learn `l1_ratio` $L_1$ "
            "cezasının ağırlığıdır, $r=1-\\alpha$ (§11.11)."
        ),
    ),
    Question(
        key="k06", concept="test-seti-tuning", note=_note("11.16"),
        prompt="Tez çalışması için model seçimi protokolüne göre hangisi yapılmamalıdır?",
        answer=MultipleChoice((
            "Estimand'ı analizin başında açıkça tanımlamak",
            "Standartlaştırma ve özellik üretimini işlem zinciri içinde öğrenmek",
            "Farklı ceza değerlerini test örnekleminde deneyip en iyisini seçmek",
            "Benzer CV değerine sahip alternatif modelleri raporlamak",
        ), correct=2),
        explanation=(
            "Tuning CV ile yapılır; test örneklemi yalnız son değerlendirmede bir kez kullanılır. Test setinde ceza "
            "denemek onu model seçiminin parçası yapar (§11.16, §11.4)."
        ),
    ),
    Question(
        key="k07", concept="cps-lasso-sifirlar", note=_note("11.15", 1),
        prompt="CPS laboratuvarında Lasso'nun 30 terimden sıfırladığı 14 terim hangileridir?",
        answer=MultipleChoice((
            "Üç bölge göstergesi ve Hispanik göstergesi",
            "Deneyimin ikinci ve üçüncü kuvvetleri",
            "Eğitim ve eğitimin karesi",
            "Eğitim × kadın etkileşimi ile gözlem sayısı küçük 13 ırk göstergesi",
        ), correct=3),
        explanation=(
            "Seyrek modelin test hatası zengin OLS'ten yalnız dördüncü ondalıkta farklıdır; parsimoniklik ile tahmin "
            "performansı arasındaki seçim araştırma amacına bağlıdır (§11.15, Adım 1)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="yansiz-her-zaman-iyi-degil", note=_note("11.2"),
        prompt="Öngörü amacında yansız bir tahmin edici, MSE'si daha düşük yanlı bir tahmin ediciye her zaman tercih edilmelidir.",
        answer=TrueFalse(False),
        explanation=(
            "MSE yanlılığın karesi ile varyansın toplamıdır. Küçük bir yanlılık varyansı yeterince azaltıyorsa yanlı "
            "tahmin edici daha düşük MSE verir; düzenlileştirme bu dengeyi kullanır (§11.2)."
        ),
    ),
    Question(
        key="d02", concept="tek-bolme-oynakligi", note=_note("11.4"),
        prompt=(
            "Tek bir eğitim/test bölmesinden elde edilen model sıralaması, özellikle küçük örneklemde, bölmenin kendisine "
            "duyarlı olabilir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Tek bölmede model daha küçük eğitim örneklemiyle tahmin edilir ve sonuç rastlantısal bölmeye bağlıdır; "
            "farklı seed'ler farklı sıralamalar verebilir. Bu, çapraz doğrulamanın motivasyonudur (§11.4)."
        ),
    ),
    Question(
        key="d03", concept="p-buyuk-n-tekillik", note=_note("11.6"),
        prompt="$p\\ge n$ olduğunda $X'X$ tekildir ve standart OLS çözümü tek değildir.",
        answer=TrueFalse(True),
        explanation=(
            "$p$ büyüdükçe $X'X$ kötü koşullu hâle gelir; $p\\ge n$ iken tersi yoktur. Yüksek boyutlu regresyon bu "
            "nedenle seyreklik ve düzenlileştirme gibi ek yapı ister (§11.6)."
        ),
    ),
    Question(
        key="d04", concept="lasso-korelasyon-secimi", note=_note("11.9"),
        prompt="Lasso, yüksek korelasyonlu regresörler arasında bile gerçek önemli değişkenleri her zaman seçer.",
        answer=TrueFalse(False),
        explanation=(
            "Yüksek korelasyon, zayıf sinyal veya küçük örneklemde Lasso eşdeğer değişkenlerden birini seçip diğerini "
            "dışarıda bırakabilir; seçim kümesi veriye ve cezaya bağlıdır (§11.9)."
        ),
    ),
    Question(
        key="d05", concept="cikarim-icin-ceza", note=_note("11.10"),
        prompt="Çıkarım amaçlı Lasso uygulamalarında da ceza parametresi her zaman öngörü CV'siyle seçilmelidir.",
        answer=TrueFalse(False),
        explanation=(
            "Öngörü için CV yaygındır; çıkarım amaçlı uygulamalarda hata dağılımı ve yüksek boyutlu moment sınırlarına "
            "dayanan teori-temelli ceza seçimleri de vardır (§11.10; Konu 12)."
        ),
    ),
    Question(
        key="d06", concept="post-lasso-secim-sonrasi", note=_note("11.12"),
        prompt=(
            "Post-Lasso seçimi aynı veriyle yaptığı için bir model seçimi sonrası tahmin edicidir; ikinci aşamanın klasik "
            "OLS standart hataları seçim belirsizliğini içermez."
        ),
        answer=TrueFalse(True),
        explanation=(
            "İkinci aşama küçültme yanlılığını azaltır, fakat hangi değişkenlerin seçildiği de veriden gelir. Klasik "
            "standart hatalar modeli önceden sabitlenmiş sayar (§11.12, §11.13)."
        ),
    ),
    Question(
        key="d07", concept="kontrollu-siralama", note=_note("11.14"),
        prompt=(
            "Notlardaki kontrollü örnekte Post-Lasso'nun en düşük test MSE'yi vermesi, Post-Lasso'nun genel olarak en iyi "
            "tahmin edici olduğunu gösterir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Bu tek bir simülasyondur; sinyal gücü, korelasyon yapısı ve örneklem büyüklüğü değişince sıralama değişebilir. "
            "Ders, dış-örneklem kaybını değerlendirmek ve cezayı önceden tanımlı bir prosedürle seçmektir (§11.14)."
        ),
    ),
    # --- Boşluk doldurma ---------------------------------------------------------
    Question(
        key="b01", concept="bic-aic-esigi", note=_note("11.3"),
        prompt=(
            "$n>e^2\\approx$ **(1)** ______ olduğunda BIC'in ek parametre başına cezası AIC'ninkinden büyüktür (iki "
            "ondalık)."
        ),
        answer=FillBlanks((NumberBlank(7.39, 0.006, "7,39"),)),
        explanation=(
            "Ek parametre başına ceza AIC'de 2, BIC'te $\\log n$'dir; $\\log n>2$ ancak $n>e^2$ iken sağlanır. Bu yüzden "
            "BIC tipik örneklemlerde daha parsimonik modeli seçer (§11.3)."
        ),
    ),
    Question(
        key="b02", concept="kontrollu-lasso-sse", note=_note("11.14"),
        prompt=(
            "Kontrollü örnekte eğitim örneklemi **(1)** ______ gözlemdir; Lasso'nun seçtiği $\\hat\\lambda_y=0{,}138$ "
            "Hansen'in SSE ölçeğinde yaklaşık **(2)** ______ cezasına karşılık gelir (tam sayı)."
        ),
        answer=FillBlanks((
            NumberBlank(224, 0.5, "224"),
            NumberBlank(62, 1.0, "62"),
        )),
        explanation=(
            "$n=320$ gözlemin ilk yüzde 70'i eğitim örneklemidir; SSE ölçeğinde ceza $2\\cdot224\\cdot0{,}138\\approx62$'dir. "
            "Ceza değeri raporlanırken ölçeği de belirtilmelidir (§11.14, §11.9)."
        ),
    ),
    Question(
        key="b03", concept="cps-bolme-buyuklukleri", note=_note("11.15"),
        prompt=(
            "CPS laboratuvarında $u_i=\\{i\\varphi\\}<0{,}25$ kuralıyla **(1)** ______ gözlem test, kalan **(2)** ______ "
            "gözlem eğitim örneklemine düşer."
        ),
        answer=FillBlanks((
            TextBlank(("12687", "12.687", "12 687"), "12.687"),
            TextBlank(("38055", "38.055", "38 055"), "38.055"),
        )),
        explanation=(
            "Bölme rastgele sayı üreteciyle değil açık bir kuralla yapılır; $\\{i\\varphi\\}$ dizisi $[0,1)$ aralığına düzgün "
            "yayıldığı için dengeli bir rastgele bölme gibi davranır ve üç dilde aynı sayıları verir (§11.15)."
        ),
    ),
    Question(
        key="b04", concept="loocv-ozel-durum", note=_note("11.5"),
        prompt="Dışarıda bırakma çapraz doğrulaması (LOOCV), $K$-katlı çapraz doğrulamanın $K=$ **(1)** ______ özel durumudur.",
        answer=FillBlanks((TextBlank(("n", "N", "gözlem sayısı"), "n"),)),
        explanation=(
            "Her gözlem kendi başına bir kattır. Uygulamada $K=5$ veya $K=10$ yaygındır; büyük $K$ hesaplamayı, çok küçük "
            "$K$ ise eğitim örneklemini gereğinden fazla küçültür (§11.5)."
        ),
    ),
    Question(
        key="b05", concept="sifir-ceza-ols", note=_note("11.7"),
        prompt="Genel cezalı amaç fonksiyonunda ceza parametresi $\\lambda=$ **(1)** ______ olduğunda çözüm OLS'dir.",
        answer=FillBlanks((NumberBlank(0.0, 1e-9, "0"),)),
        explanation=(
            "$\\|Y-X\\beta\\|_2^2+\\lambda P(\\beta)$ ölçütünde $\\lambda$ büyüdükçe katsayıların serbestliği azalır; ceza "
            "bilinçli bir yanlılık karşılığında varyansı düşürmeyi amaçlar (§11.7)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="f01", concept="bic-aic-farki", note=_note("11.3"),
        prompt="Aynı model için $K$ parametre ve $n$ gözlemle BIC ile AIC arasındaki farkı (BIC − AIC) yazın.",
        answer=Equation(
            lhs=r"BIC-AIC",
            symbols=(Symbol("K", "K", "parametre sayısı (σ² dahil)", 1.0, 30.0), _N),
            answer="K*(log(n) - 2)",
            shown=r"K(\log n-2)",
        ),
        explanation=(
            "$n+n\\log(2\\pi\\hat\\sigma^2)$ terimi iki ölçütte ortaktır; fark yalnız cezalardan gelir: $K\\log n-2K$. "
            "Fark $n>e^2$ iken pozitiftir (§11.3)."
        ),
    ),
    Question(
        key="f02", concept="lasso-olcek-donusumu", note=_note("11.9"),
        prompt=(
            "Yazılımlar Lasso'yu $\\frac{1}{2n}(Y-X\\beta)'(Y-X\\beta)+\\lambda_y\\sum_j|\\beta_j|$ biçiminde çözer. Aynı "
            "çözümü veren Hansen SSE ölçeğindeki cezayı $n$ ve $\\lambda_y$ cinsinden yazın."
        ),
        answer=Equation(lhs=r"\lambda", symbols=(_N, _LAMBDA_Y), answer="2*n*ly", shown=r"2n\lambda_y"),
        explanation=(
            "Yazılımın amacını $2n$ ile çarpmak çözümü değiştirmez ve Hansen'in ölçütünü verir: $(Y-X\\beta)'(Y-X\\beta)"
            "+2n\\lambda_y\\sum_j|\\beta_j|$ (§11.9)."
        ),
    ),
    Question(
        key="f03", concept="ridge-olcek-donusumu", note=_note("11.8"),
        prompt=(
            "R `glmnet` Ridge amacını $\\frac{1}{2n}(Y-X\\beta)'(Y-X\\beta)+\\frac{\\lambda_y}{2}\\beta'\\beta$ "
            "biçiminde yazar. Hansen ölçeğindeki Ridge cezasını $n$ ve $\\lambda_y$ cinsinden yazın."
        ),
        answer=Equation(lhs=r"\lambda", symbols=(_N, _LAMBDA_Y), answer="n*ly", shown=r"n\lambda_y"),
        explanation=(
            "Amacı $2n$ ile çarpmak $(Y-X\\beta)'(Y-X\\beta)+n\\lambda_y\\beta'\\beta$ verir. scikit-learn "
            "`Ridge(alpha)` ise doğrudan Hansen ölçeğini kullanır (§11.8)."
        ),
    ),
    Question(
        key="f04", concept="l1-agirligi-alpha", note=_note("11.11"),
        prompt=(
            "Hansen'in Elastic Net gösteriminde karışım parametresi $\\alpha$ $L_2$ cezasının ağırlığıdır. Yazılımlardaki "
            "$L_1$ ağırlığı $r$'yi $\\alpha$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="r",
            symbols=(Symbol("a", r"\alpha", "Hansen'in karışım parametresi", 0.0, 1.0, aliases=("α", "alpha")),),
            answer="1 - a",
            shown=r"1-\alpha",
        ),
        explanation=(
            "Notlardaki kontrollü örnekte CV $r=0{,}8$ seçmiştir; bu Hansen gösteriminde $\\alpha=0{,}2$'dir. Ceza "
            "değerleriyle birlikte karışım parametresinin tanımı da raporlanmalıdır (§11.11)."
        ),
    ),
    Question(
        key="f05", concept="kisa-model-olasilik-limiti", note=_note("11.13"),
        prompt=(
            "$Y=\\beta_1X_1+\\beta_2X_2+\\varepsilon$, $\\operatorname{Var}(X_1)=\\operatorname{Var}(X_2)=1$ ve "
            "$\\operatorname{Corr}(X_1,X_2)=\\rho$ olsun. Ön test $X_2$'yi dışarıda bıraktığında $\\hat\\beta_1$'in olasılık "
            "limitini yazın."
        ),
        answer=Equation(
            lhs=r"\operatorname{plim}\hat\beta_1",
            symbols=(
                Symbol("b1", r"\beta_1", "hedef katsayı", -2.0, 2.0, aliases=("β₁", "β1", "beta1", "beta_1")),
                Symbol("b2", r"\beta_2", "dışlanan değişkenin katsayısı", -1.0, 1.0,
                       aliases=("β₂", "β2", "beta2", "beta_2")),
                Symbol("r", r"\rho", "iki regresörün korelasyonu", -0.9, 0.9, aliases=("ρ", "rho")),
            ),
            answer="b1 + b2*r",
            shown=r"\beta_1+\beta_2\rho",
        ),
        explanation=(
            "$X_2$ seçilmediğinde kısa model $\\beta_2\\rho$ kadar yanlıdır; seçilen modelin klasik güven aralığı bu "
            "yanlılığı ve seçim olayını bilmediği için kapsaması nominal düzeyin altına düşebilir (§11.13; Sezgi, "
            "Deney 3)."
        ),
    ),
)


KONU11_QUIZ = QuestionSet(
    topic_key="konu11",
    title="Konu 11: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set model seçiminin amaca bağlı anlamını, yanlılık–varyans dengesini, AIC ve BIC'i, ayırma örneklemini ve "
        "çapraz doğrulamayı, Ridge, Lasso, Elastic Net ve Post-Lasso'yu, ceza ölçeklerini, model seçimi sonrası çıkarımı "
        "ve CPS laboratuvarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir bölüme bağlıdır; yanlış "
        "cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Bölüm sonu egzersizlerini tekrar etmez, onları "
        "tamamlar."
    ),
    sections=("11.1", "11.2", "11.3", "11.4", "11.5", "11.6", "11.7", "11.8", "11.9", "11.10", "11.11", "11.12",
              "11.13", "11.14", "11.15", "11.16"),
)
