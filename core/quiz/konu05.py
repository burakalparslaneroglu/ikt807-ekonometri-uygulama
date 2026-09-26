"""Konu 5 "Kendini sına" soru seti: ikili ve ayrık sonuç modelleri.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Bölüm sonu egzersizlerindeki
hesaplar (koşullu varyans, LPM tahmini, Logit olasılığı, Bernoulli olabilirliği, kukla sonlu fark,
Poisson yüzde etkisi) tekrar edilmez; buradaki sorular aynı konuların farklı yönlerini sınar.
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


BETA = Symbol("b", r"\beta_j", "katsayı", -2.0, 2.0, aliases=("β_j", "βj", "β", "beta"))


QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="lpm-yapisal-heteroskedastisite", note=_note("5.3"),
        prompt="Doğrusal olasılık modelinde klasik (homoskedastik) standart hata neden doğal varsayılan değildir?",
        answer=MultipleChoice((
            "LPM katsayıları yanlı olduğu için",
            "$\\operatorname{Var}(e\\mid X)=P(X)\\{1-P(X)\\}$ $X$'e bağlıdır: heteroskedastisite yapısaldır",
            "İkili bağımlı değişkenle OLS hesaplanamadığı için",
            "Yalnız örneklem küçük olduğunda sorun çıktığı için",
        ), correct=1),
        explanation=(
            "İkili $Y$'de koşullu varyans koşullu olasılığa bağlıdır; olasılık $X$ ile değiştikçe varyans da değişir. Bu "
            "yüzden LPM'de dayanıklı (HC) standart hata doğal seçimdir; katsayılar ise bundan etkilenmez (§5.3)."
        ),
    ),
    Question(
        key="k02", concept="olcek-normalizasyonu", note=_note("5.5"),
        prompt=(
            "Aynı CPS verisinde yaşın Logit katsayısı Probit katsayısının yaklaşık 1,65 katı çıkıyor (Uygulama Adım 2). "
            "Bu farkın kaynağı nedir?"
        ),
        answer=MultipleChoice((
            "Logit yaşın etkisini abartır",
            "Probit'in standart hataları daha büyüktür",
            "İki model gizli hatanın ölçeğini farklı normalize eder; katsayılar farklı ölçektedir",
            "İki model farklı örneklemlerle tahmin edilmiştir",
        ), correct=2),
        explanation=(
            "Veri yalnız $\\beta/\\sigma$'yı belirler; Probit $\\operatorname{Var}(u)=1$, Logit standart lojistik ölçeği "
            "kullanır. Olasılık ölçeğinde (AME) iki model neredeyse aynı sonucu verir: 0,0441 ve 0,0445 (§5.5)."
        ),
    ),
    Question(
        key="k03", concept="ame-tanimi", note=_note("5.8.1"),
        prompt="Ortalama marjinal etki (AME) nasıl hesaplanır?",
        answer=MultipleChoice((
            "Her gözlemde marjinal etki hesaplanır, sonra örneklem ortalaması alınır",
            "Kovaryatların örneklem ortalamasında tek bir marjinal etki hesaplanır",
            "Logit katsayısı 4'e bölünür",
            "Tahmin edilen olasılıkların örneklem ortalaması alınır",
        ), correct=0),
        explanation=(
            "$\\widehat{AME}_j=n^{-1}\\sum_i\\hat\\beta_j g(X_i'\\hat\\beta)$: gerçek kovaryat dağılımını korur ve "
            "veride bulunmayabilecek bir \"ortalama birey\" yaratmaz (§5.8.1)."
        ),
    ),
    Question(
        key="k04", concept="ordered-yon-yalniz-uclarda", note=_note("5.11"),
        prompt=(
            "Sıralı Probit'te bir regresörün katsayısı pozitif. Hangi kategori olasılıklarının değişim yönünden emin "
            "olabiliriz?"
        ),
        answer=MultipleChoice((
            "Bütün kategorilerin olasılığı artar",
            "Yalnız ortadaki kategorinin olasılığı artar",
            "Hiçbirinin; katsayı tek başına yön bilgisi vermez",
            "En yüksek kategorinin olasılığı artar, en düşüğünkü azalır",
        ), correct=3),
        explanation=(
            "Gizli indeks sağa kayınca uç kategorilerin olasılıkları belirli yönde değişir; ara kategorilerde değişimin "
            "işareti eşiklere ve başlangıç noktasına bağlıdır. Bu yüzden kategori olasılıkları ayrı ayrı raporlanır (§5.11)."
        ),
    ),
    Question(
        key="k05", concept="poisson-qmle-dayanikli", note=_note("5.12"),
        prompt=(
            "Poisson regresyonunda koşullu ortalama $\\exp(X'\\beta)$ doğru belirtilmiş, fakat koşullu varyans ortalamadan "
            "büyük (aşırı saçılma). Doğru tutum hangisidir?"
        ),
        answer=MultipleChoice((
            "Katsayılar tutarsızdır; model bırakılmalıdır",
            "Katsayılar anlamlı kalır; çıkarım dayanıklı (sandviç) standart hatayla yapılır",
            "Klasik MLE standart hataları zaten doğrudur",
            "Sayım değişkeninin logaritması alınıp OLS yapılmalıdır",
        ), correct=1),
        explanation=(
            "Koşullu ortalama doğruysa Poisson QMLE katsayıları anlamlıdır; eşvaryans bozulduğu için klasik standart "
            "hatalar geçersizdir, dayanıklı kovaryans kullanılır (§5.12)."
        ),
    ),
    Question(
        key="k06", concept="iia", note=_note("5.10"),
        prompt="Multinomial Logit'in ilgisiz alternatiflerden bağımsızlık (IIA) özelliği neyi söyler?",
        answer=MultipleChoice((
            "Alternatifler doğal bir sıraya sahiptir",
            "Her alternatifin seçilme olasılığı eşittir",
            "İki alternatif arasındaki odds oranı diğer alternatiflerin varlığından ve özelliklerinden etkilenmez",
            "Katsayılar bütün alternatiflerde aynıdır",
        ), correct=2),
        explanation=(
            "$P_j/P_k=\\exp\\{x'(\\beta_j-\\beta_k)\\}$ yalnız $j$ ve $k$'ye bağlıdır. Yakın ikame alternatifler (ör. iki "
            "benzer otobüs hattı) varsa bu varsayım gerçekçi olmayabilir (§5.10)."
        ),
    ),
    Question(
        key="k07", concept="ayrisma-mle-yok", note=_note("5.15", 1),
        prompt="Uygulamada ırkın ayrıntılı kodlarıyla (18 kategori) kurulan Logit neden yakınsamaz?",
        answer=MultipleChoice((
            "Örneklem çok büyük olduğu için",
            "Birkaç küçük kategoride herkes evli ya da herkes bekâr; bu katsayıların MLE'si sonsuza gider (ayrışma)",
            "Logit kategorik değişken kabul etmediği için",
            "Eğitim ile ırk birbiriyle ilişkili olduğu için",
        ), correct=1),
        explanation=(
            "1–2 gözlemli dört kategoride sonuç hep 0 veya hep 1'dir: olabilirlik o katsayı sonsuza giderken artmaya devam "
            "eder, sonlu bir MLE yoktur (yarı-tam ayrışma). Çözüm kategorileri anlamlı gruplara toplamaktır (§5.15, Adım 1)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="lpm-sinir-ihlali-yorumu", note=_note("5.3"),
        prompt=(
            "Doğrusal olasılık modelinin tahmin ettiği olasılıkların bir kısmı $[0,1]$ dışında kalıyorsa model hiçbir "
            "amaçla kullanılamaz."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Sınır ihlali bireysel olasılık tahmini için sorundur; ortalama farklar veya kolay yorumlanan etkiler hedefse "
            "LPM yararlı bir yaklaşım olmaya devam eder. Asıl soru hedefin ne olduğudur (§5.3)."
        ),
    ),
    Question(
        key="d02", concept="logit-skor-ortalama-tahmin", note=_note("5.6.1"),
        prompt=(
            "Sabit terim içeren bir Logit modelinde tahmin edilen olasılıkların örneklem ortalaması, $Y=1$ olanların "
            "örneklem payına eşittir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Skor denklemi $\\sum_i X_i(Y_i-\\hat p_i)=0$'ın sabit terime karşılık gelen satırı "
            "$\\sum_i(Y_i-\\hat p_i)=0$'dır. CPS uygulamasında ikisi de 0,5422'dir. Probit'te bu eşitlik tam olarak geçerli "
            "değildir (§5.6.1)."
        ),
    ),
    Question(
        key="d03", concept="bilgi-matrisi-esitligi", note=_note("5.7.2"),
        prompt=(
            "Model doğru belirtilmişse Logit'in klasik ve sandviç (robust) standart hataları büyük örneklemde aynı değere "
            "yakınsar."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Doğru belirtimde $\\mathcal S=\\mathcal H$ (bilgi matrisi eşitliği) ve sandviç $\\mathcal H^{-1}\\mathcal "
            "S\\mathcal H^{-1}$ klasik $\\mathcal H^{-1}$'e indirgenir (§5.7.2)."
        ),
    ),
    Question(
        key="d04", concept="robust-hedefi-duzeltmez", note=_note("5.7.2"),
        prompt="Logit'te robust standart hata kullanmak, bağlantı fonksiyonu yanlışsa katsayıyı doğru hedefe getirir.",
        answer=TrueFalse(False),
        explanation=(
            "Sandviç yalnız belirsizliği yanlış belirtime karşı daha dürüst ölçer; tahmin edicinin olasılık limitini "
            "değiştirmez. Yanlış bağlantıda katsayı sözde-doğru bir değere gider (§5.7.2)."
        ),
    ),
    Question(
        key="d05", concept="ame-nedensel-degil", note=_note("5.8.1"),
        prompt="Ortalama marjinal etki, hangi model ailesiyle hesaplanırsa hesaplansın nedensel bir etkidir.",
        answer=TrueFalse(False),
        explanation=(
            "AME model temelli bir özettir. Açıklayıcı içselse veya kontrol seti nedensel karşılaştırmayı tanımlamıyorsa "
            "yalnız koşullu ilişkiyi gösterir; Konu 3'teki tanımlama mantığı aynen geçerlidir (§5.8.1)."
        ),
    ),
    Question(
        key="d06", concept="logit-probit-ame-yakin", note=_note("5.9"),
        prompt="CPS uygulamasında Logit ve Probit ham katsayıları farklı olsa da ortalama marjinal etkiler neredeyse aynıdır.",
        answer=TrueFalse(True),
        explanation=(
            "Tablo 5.1'de yaş AME'si 0,0441 ve 0,0445, Siyah göstergesininki −0,1617 ve −0,1582. Farklılık katsayı ölçeğinde, "
            "olasılık ölçeğinde değil (§5.9)."
        ),
    ),
    Question(
        key="d07", concept="profil-ortalama-birey-degil", note=_note("5.15", 3),
        prompt=(
            "Uygulamadaki yaş profili, kovaryatların örneklem ortalamasındaki \"ortalama birey\" için hesaplanan "
            "olasılıktır."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Her yaş için herkesin yaşı o değere eşitlenir, diğer kovaryatlar gözlenen değerlerinde kalır ve bireysel "
            "tahminler ortalanır. Doğrusal olmayan modelde $G(\\mathbb E[X]'\\beta)\\neq\\mathbb E[G(X'\\beta)]$ "
            "(§5.15, Adım 3)."
        ),
    ),
    # --- Boşluk doldurma ---------------------------------------------------------
    Question(
        key="b01", concept="sinirli-bagimli-degisken", note=_note("5.1"),
        prompt="Destek kümesi reel sayı doğrusunun tamamı olmayan sonuç değişkenlerine **(1)** ______ bağımlı değişken denir.",
        answer=FillBlanks((TextBlank(("sınırlı", "sinirli", "limited", "kısıtlı"), "sınırlı (limited)"),)),
        explanation=(
            "İkili, çok kategorili, sıralı, sayım ve sansürlü sonuçlar bu sınıftadır; sınırlama koşullu dağılımın nasıl "
            "modelleneceğini belirler (§5.1)."
        ),
    ),
    Question(
        key="b02", concept="probit-yogunluk", note=_note("5.4"),
        prompt=(
            "Probit modelinde sürekli bir $X_j$'nin marjinal etkisi $\\beta_j$ ile standart normal **(1)** ______ "
            "fonksiyonunun $x'\\beta$ noktasındaki değerinin çarpımıdır."
        ),
        answer=FillBlanks((TextBlank(
            ("yoğunluk", "yogunluk", "yoğunluğu", "yoğunluk fonksiyonu", "density", "pdf"),
            "yoğunluk (φ)",
        ),)),
        explanation=(
            "$g=G'$ ve Probit'te $G=\\Phi$ olduğundan $g=\\phi$: marjinal etki $\\beta_j\\phi(x'\\beta)$. İndeks sıfıra "
            "yakınken en büyük, uçlarda küçüktür (§5.4)."
        ),
    ),
    Question(
        key="b03", concept="delta-yontemi-ame", note=_note("5.7.3"),
        prompt=(
            "AME'nin standart hatası, $\\hat\\beta$'nın kovaryans matrisinin **(1)** ______ yöntemiyle AME'ye "
            "aktarılmasıyla elde edilir."
        ),
        answer=FillBlanks((TextBlank(("delta", "delta yöntemi", "delta method"), "delta"),)),
        explanation=(
            "AME katsayıların doğrusal olmayan bir fonksiyonudur: $\\operatorname{SH}=\\sqrt{\\nabla'V\\nabla}$. Hangi "
            "$V$'nin (klasik mi, dayanıklı mı) kullanıldığı raporlanmalıdır (§5.7.3)."
        ),
    ),
    Question(
        key="b04", concept="sonlu-fark-tanimi", note=_note("5.8.2"),
        prompt=(
            "$D\\in\\{0,1\\}$ bir göstergede etki türevle değil, $D=1$ ve $D=0$ senaryolarındaki tahmin edilen olasılıklar "
            "arasındaki **(1)** ______ farkla tanımlanır."
        ),
        answer=FillBlanks((TextBlank(("sonlu", "kesikli", "ayrık"), "sonlu"),)),
        explanation=(
            "$\\Delta_D(x)=G(x'\\beta+\\beta_D)-G(x'\\beta)$. Uygulama Adım 5'te Siyah göstergesi için türev tabanlı "
            "yaklaşım −0,1598, sonlu fark −0,1617 (§5.8.2)."
        ),
    ),
    Question(
        key="b05", concept="yas-profili-35", note=_note("5.15", 3),
        prompt=(
            "Uygulamada 35 yaşında ortalama tahmin edilen evlilik olasılığı yaklaşık **(1)** ______'dir (iki ondalık)."
        ),
        answer=FillBlanks((NumberBlank(0.82, 0.006, "0,82"),)),
        explanation=(
            "Profil 18 yaşında yaklaşık 0,11'den 35 yaşında 0,82'ye çıkar ve S biçimlidir: eğim orta yaşlarda en büyüktür "
            "(§5.15, Adım 3)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="f01", concept="logit-marjinal-etki-p", note=_note("5.4.1"),
        prompt=(
            "Logit'te sürekli $X_j$'nin gözlem düzeyindeki marjinal etkisini, tahmin edilen olasılık $p$ ve katsayı "
            "$\\beta_j$ cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\partial P/\partial x_j",
            symbols=(Symbol("p", "p", "tahmin edilen olasılık", 0.05, 0.95), BETA),
            answer="p*(1-p)*b",
            shown=r"p(1-p)\,\beta_j",
        ),
        explanation=(
            "$\\Lambda'(v)=\\Lambda(v)\\{1-\\Lambda(v)\\}$ olduğundan etki $p(1-p)\\beta_j$; en büyük değeri $p=0{,}5$'te "
            "$\\beta_j/4$'tür (§5.4.1)."
        ),
    ),
    Question(
        key="f02", concept="tanimlanan-katsayi-oran", note=_note("5.5"),
        prompt=(
            "$Y^*=x'\\beta+\\sigma\\varepsilon$, $\\varepsilon$'nin dağılım fonksiyonu simetrik $G$ ve "
            "$Y=\\mathbf 1\\{Y^*>0\\}$ ise $P(Y=1\\mid x)=G(x'c)$ olur. Veriden tanımlanan $c$'yi $\\beta$ ve $\\sigma$ "
            "cinsinden yazın."
        ),
        answer=Equation(
            lhs="c",
            symbols=(
                Symbol("b", r"\beta", "gizli katsayı", -2.0, 2.0, aliases=("β", "beta")),
                Symbol("s", r"\sigma", "hata ölçeği", 0.5, 3.0, aliases=("σ", "sigma")),
            ),
            answer="b/s",
            shown=r"c=\beta/\sigma",
        ),
        explanation=(
            "$\\beta$ ve $\\sigma$'yı aynı oranda büyütmek olasılığı değiştirmez; yalnız oranları tanımlanır. Probit "
            "$\\sigma=1$ normalizasyonu yapar (§5.5; Sezgi Deney 2)."
        ),
    ),
    Question(
        key="f03", concept="multinomial-olasilik", note=_note("5.10"),
        prompt=(
            "Üç alternatifli Multinomial Logit'te üçüncü alternatif referanstır (indeksi 0). Birinci alternatifin "
            "olasılığını indeksler $v_1$ ve $v_2$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="P_1",
            symbols=(
                Symbol("v1", "v_1", "birinci alternatifin indeksi", -2.0, 2.0, aliases=("v_1",)),
                Symbol("v2", "v_2", "ikinci alternatifin indeksi", -2.0, 2.0, aliases=("v_2",)),
            ),
            answer="exp(v1)/(1 + exp(v1) + exp(v2))",
            shown=r"P_1=\dfrac{e^{v_1}}{1+e^{v_1}+e^{v_2}}",
        ),
        explanation=(
            "Referans alternatifin indeksi 0 olduğundan paydadaki terimi $e^0=1$'dir; olasılıklar toplamı 1 olur (§5.10)."
        ),
    ),
    Question(
        key="f04", concept="sandvic-tek-parametre", note=_note("5.7.1"),
        prompt=(
            "Tek parametreli bir MLE'de ortalama eğrilik $\\mathcal H$ ve skor varyansı $\\mathcal S$ ise "
            "$\\sqrt n(\\hat\\beta-\\beta_0)$'ın asimptotik varyansını yazın."
        ),
        answer=Equation(
            lhs=r"\operatorname{avar}",
            symbols=(
                Symbol("H", r"\mathcal H", "ortalama eğrilik", 0.5, 3.0),
                Symbol("S", r"\mathcal S", "skor varyansı", 0.5, 3.0),
            ),
            answer="S/H^2",
            shown=r"\mathcal S/\mathcal H^2",
        ),
        explanation=(
            "Sandviç $\\mathcal H^{-1}\\mathcal S\\mathcal H^{-1}$ tek boyutta $\\mathcal S/\\mathcal H^2$'dir; doğru "
            "belirtimde $\\mathcal S=\\mathcal H$ ve varyans $1/\\mathcal H$ olur (§5.7.1)."
        ),
    ),
    Question(
        key="f05", concept="poisson-marjinal-etki", note=_note("5.12"),
        prompt=(
            "$\\mathbb E[Y\\mid X=x]=\\exp(x'\\beta)$ ise sürekli $X_j$'nin koşullu ortalama üzerindeki marjinal etkisini "
            "$\\beta_j$ ve $\\mu=\\exp(x'\\beta)$ cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\partial\mathbb E[Y\mid x]/\partial x_j",
            symbols=(BETA, Symbol("m", r"\mu", "koşullu ortalama", 0.2, 5.0, aliases=("μ", "mu"))),
            answer="b*m",
            shown=r"\beta_j\,\mu",
        ),
        explanation=(
            "Zincir kuralı: $\\beta_j\\exp(x'\\beta)$. Etki düzeye bağlıdır; göreli etki (yarı esneklik) ise sabittir: "
            "$\\partial\\log\\mathbb E[Y\\mid x]/\\partial x_j=\\beta_j$ (§5.12)."
        ),
    ),
)


KONU05_QUIZ = QuestionSet(
    topic_key="konu05",
    title="Konu 5: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set ikili ve ayrık sonuç modellerinin kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki "
        "bir bölüme bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Bölüm sonu "
        "egzersizlerini tekrar etmez, onları tamamlar."
    ),
    sections=("5.1", "5.3", "5.4", "5.4.1", "5.5", "5.6.1", "5.7.1", "5.7.2", "5.7.3", "5.8.1", "5.8.2", "5.9", "5.10",
              "5.11", "5.12", "5.15"),
)
