"""Konu 1 "Kendini sına" soru seti.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Sorular notların
bölüm sonu egzersizlerini tekrar etmez; onları tamamlar. Sayısal cevaplar notlardaki
basılı değerlerden alınmıştır ve testlerde gerçek veriyle doğrulanır.
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

BETA = Symbol("b", r"\beta", "eğim katsayısı β", -0.5, 0.5, aliases=("β", "beta"))


def _note(section: str, step: int = 0, *objects: str) -> NoteRef:
    return NoteRef(section, step, tuple(objects))


QUESTIONS = (
    # --- Çoktan seçmeli -----------------------------------------------------------
    Question(
        key="k01", concept="tahmin-hedefi-once-gelir", note=_note("1.1.1"),
        prompt=(
            "Araştırma sorusu: *Tam zamanlı çalışanlarda eğitim düzeyi yükseldikçe log saatlik ücretin "
            "ortalaması nasıl değişmektedir?* Bu sorunun **ilk** tahmin hedefi hangisidir?"
        ),
        answer=MultipleChoice(
            (
                r"Eğitim katsayısı $\beta_1$",
                r"$m(x)=\mathbb{E}[\log(\text{ücret})\mid \text{eğitim}=x]$ fonksiyonu",
                "Eğitimin ücret üzerindeki nedensel etkisi",
                r"Log ücretin koşulsuz ortalaması $\mathbb{E}[\log(\text{ücret})]$",
            ),
            correct=1,
        ),
        explanation=(
            "Soru, eğitim düzeylerine göre ortalamanın nasıl değiştiğini soruyor; bu bir anakütle "
            "fonksiyonudur: koşullu beklenti. Katsayı ancak bu fonksiyonu bir modelle özetlediğimizde "
            "ortaya çıkar. Soru \"eğitimi artırırsak\" demediği için nedensel bir hedef de değildir "
            "(§1.1.1; Uygulama, Adım 1)."
        ),
    ),
    Question(
        key="k02", concept="cef-en-iyi-ongorucu", note=_note("1.5"),
        prompt=(
            r"Kareli hata kaybı $\mathbb{E}[(Y-g(X))^2]$ altında, $X$'in **bütün** fonksiyonları $g$ "
            r"arasında $Y$'nin en iyi öngörücüsü hangisidir?"
        ),
        answer=MultipleChoice(
            (
                r"Doğrusal projeksiyon $X'\beta$",
                "Koşullu medyan",
                r"Koşulsuz ortalama $\mathbb{E}[Y]$",
                r"Koşullu beklenti $\mathbb{E}[Y\mid X]$",
            ),
            correct=3,
        ),
        explanation=(
            "Koşullu beklenti, ortalama kareli hatayı $X$'in bütün fonksiyonları arasında en küçükler "
            "(§1.5). Doğrusal projeksiyon yalnız doğrusal fonksiyonlar arasında en iyidir; koşullu "
            "medyan mutlak hata kaybında en iyidir; koşulsuz ortalama $X$ bilgisini hiç kullanmaz."
        ),
    ),
    Question(
        key="k03", concept="doymus-model-hucre-ortalamasi", note=_note("1.13", 4, "Tablo 1.3"),
        prompt=(
            r"Eğitim yılının her değeri için ayrı bir kukla içeren (**doymuş**) regresyonun uyum "
            r"değerleri $\hat Y_i$ neye eşittir?"
        ),
        answer=MultipleChoice(
            (
                "Aynı eğitim düzeyindeki gözlemlerin $Y$ ortalamasına (hücre ortalamasına)",
                "Gerçek koşullu ortalama $m(x)$'e",
                "Yalnız eğitimi içeren OLS doğrusunun değerlerine",
                "Bütün örneklemin $Y$ ortalamasına",
            ),
            correct=0,
        ),
        explanation=(
            "Doymuş modelin uyum değerleri hücre ortalamalarına **tam olarak** eşittir (Sezgi, Deney 1). "
            "Bunlar $m(x)$'in tahminidir, kendisi değil: örneklem büyüdükçe $m(x)$'e yakınsarlar. Doğrusal "
            "OLS doğrusu ise ancak $m(x)$ doğrusalsa hücre ortalamalarıyla aynı yere gider (§1.13, Adım 4)."
        ),
    ),
    Question(
        key="k04", concept="ols-hucreleri-gozlem-sayisiyla-agirliklandirir",
        note=_note("1.13", 5, "Şekil 1.7"),
        prompt=(
            "Bireysel veriyle tahmin edilen basit OLS eğimi, 13 eğitim hücresinin ortalamalarına "
            "aşağıdaki regresyonlardan hangisi uydurulursa **birebir** elde edilir?"
        ),
        answer=MultipleChoice(
            (
                "Hücre ortalamalarının ağırlıksız OLS regresyonu",
                "Hücre standart sapmasıyla ağırlıklı regresyon",
                "Hücredeki gözlem sayısıyla ağırlıklı regresyon",
                "Hiçbiri; bireysel veri hücre ortalamalarında olmayan bilgi taşır",
            ),
            correct=2,
        ),
        explanation=(
            "Hücre içi sapmalar eğitimin her fonksiyonuyla ilişkisizdir; bu yüzden OLS'e yalnız hücre "
            "ortalamaları ve hücre büyüklükleri girer. CPS'te iki yol da 0,1082 eğimini verir; ağırlıksız "
            "regresyon 0,0750 verirdi. Şekil 1.7'de nokta boyutlarının gözlem sayısıyla orantılı "
            "çizilmesinin nedeni budur: kalabalık hücreler (12 ve 16 yıl) doğruyu kendine çeker (§1.13, Adım 5)."
        ),
    ),
    Question(
        key="k05", concept="r2-neyi-olcer", note=_note("1.13", 6, "Tablo 1.4"),
        prompt=r"Tablo 1.4'te Model (3) için $R^2=0{,}2728$. Hangisi doğru bir yorumdur?",
        answer=MultipleChoice(
            (
                "Eğitim katsayısının nedensel olma olasılığı yaklaşık yüzde 27'dir",
                "Modelin fonksiyonel biçimi yaklaşık yüzde 73 oranında yanlıştır",
                "Log ücretin örneklem varyansının yaklaşık yüzde 27'si modelin uyum değerleriyle açıklanır",
                "Model yeni örneklemlerde öngörü hatasını yüzde 27 azaltır",
            ),
            correct=2,
        ),
        explanation=(
            "$R^2$ örneklem içi açıklanan varyans payıdır. Yüksek $R^2$ ne fonksiyonel biçimin "
            "doğruluğunu, ne katsayının nedenselliğini, ne de yeni örneklemde daha iyi öngörüyü tek "
            "başına kanıtlar (§1.13, Adım 6)."
        ),
    ),
    Question(
        key="k06", concept="regresyonun-kullanim-amaci", note=_note("1.12"),
        prompt=(
            "Bir e-ticaret şirketi, müşterilerin geçmiş alışverişlerinden gelecek ay yapacakları "
            "harcamayı tahmin etmek istiyor. Müşterilerin **neden** harcadığını açıklamak amaç değil. "
            "Bu, regresyonun hangi kullanım amacıdır?"
        ),
        answer=MultipleChoice(("Betimleyici", "Öngörüsel", "Nedensel", "Üçü aynı anda"), correct=1),
        explanation=(
            "Hedef yeni gözlemler için iyi tahmin: öngörüsel kullanım (§1.12). Katsayıların nedensel "
            "yorumu gerekmez; başarı ölçütü örneklem dışı öngörü performansıdır."
        ),
    ),
    Question(
        key="k07", concept="kontrol-eklenince-katsayinin-anlami", note=_note("1.10", 0, "Tablo 1.4"),
        prompt=(
            "Tablo 1.4'te Model (2)'den Model (3)'e geçerken kadın göstergesi eklenince eğitim katsayısı "
            "0,1126'dan 0,1148'e çıkıyor. En doğru yorum hangisidir?"
        ),
        answer=MultipleChoice(
            (
                "Kadın göstergesi eklenince eğitimin nedensel etkisi arttı",
                "Model (2) hatalı tahmin edilmişti",
                "Kadınların eğitim getirisi erkeklerinkinden yüksektir",
                "Katsayı artık, kadın göstergesiyle ilişkili kısmı ayrılmış eğitim değişkenliği ile "
                "ücret arasındaki ilişkiyi ölçüyor",
            ),
            correct=3,
        ),
        explanation=(
            "Çoklu regresyon katsayısı, diğer regresörlerin taşıdığı doğrusal bilgi ayrıldıktan sonra "
            "kalan ilişkiyi ölçer (§1.10). Kontrol seti değişince katsayının temsil ettiği nesne değişir. "
            "Model cinsiyete göre ayrı eğim içermediği için üçüncü seçenek bu tablodan söylenemez; "
            "birinci seçenek nedensel bir iddiadır."
        ),
    ),
    Question(
        key="k08", concept="ols-projeksiyonun-orneklem-karsiligi", note=_note("1.8"),
        prompt=(
            r"Anakütle projeksiyon katsayısı $\beta=(\mathbb{E}[XX'])^{-1}\mathbb{E}[XY]$ ise örneklem "
            r"karşılığı olan OLS tahmin edicisi hangisidir? ($\mathbf{X}$: $n\times k$ regresör matrisi, "
            r"$\mathbf{Y}$: $n\times 1$ vektör)"
        ),
        answer=MultipleChoice(
            (
                r"$(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$",
                r"$(\mathbf{X}\mathbf{X}')^{-1}\mathbf{X}\mathbf{Y}$",
                r"$\mathbf{X}'\mathbf{Y}\,(\mathbf{X}'\mathbf{X})$",
                r"$(\mathbf{X}'\mathbf{Y})^{-1}\mathbf{X}'\mathbf{X}$",
            ),
            correct=0,
        ),
        explanation=(
            r"Beklentiler örneklem ortalamalarıyla değiştirilir: $\mathbb{E}[XX']\to n^{-1}\mathbf{X}'\mathbf{X}$, "
            r"$\mathbb{E}[XY]\to n^{-1}\mathbf{X}'\mathbf{Y}$; $n^{-1}$'ler sadeleşir (§1.8). İkinci "
            r"seçenekte $\mathbf{X}\mathbf{X}'$ $n\times n$ boyutludur ve $n>k$ iken terslenemez."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="kosullu-ortalama-varyansi-kisitlamaz", note=_note("1.4"),
        prompt=r"$\mathbb{E}[e\mid X]=0$ ise hatanın koşullu varyansı $\operatorname{Var}(e\mid X)$ da $X$'e bağlı olamaz.",
        answer=TrueFalse(False),
        explanation=(
            "$\\mathbb{E}[e\\mid X]=0$ yalnız koşullu **ortalama** hakkındadır. Hatanın yayılımı $X$ ile "
            "değişebilir; örneğin ücretlerin dağılımı yüksek eğitim düzeylerinde daha geniş olabilir. "
            "Bu duruma heteroskedastisite denir; çıkarıma etkisi Konu 2'de ele alınır (§1.4)."
        ),
    ),
    Question(
        key="d02", concept="artiklarin-toplami-cebirdir", note=_note("1.8.2"),
        prompt="Sabit terim içeren bir OLS regresyonunda artıkların örneklem toplamı, doğrusal model yanlış olsa bile sıfırdır.",
        answer=TrueFalse(True),
        explanation=(
            r"Normal denklemler $\mathbf{X}'\hat e=0$'ın sabit terime karşılık gelen satırı "
            r"$\sum_i \hat e_i=0$'dır. Bu bir cebir sonucudur, modelin doğruluğuna bağlı değildir "
            "(§1.8.2; Sezgi, Deney 2)."
        ),
    ),
    Question(
        key="d03", concept="orneklem-ortogonalligi-kanit-degildir", note=_note("1.8.2"),
        prompt=r"Örneklemde $\mathbf{X}'\hat e=0$ sağlandığına göre anakütlede $\mathbb{E}[e\mid X]=0$ olduğu gösterilmiş olur.",
        answer=TrueFalse(False),
        explanation=(
            r"$\mathbf{X}'\hat e=0$ her örneklemde OLS'nin tanımı gereği sağlanır; hiçbir şey kanıtlamaz. "
            r"$\mathbb{E}[Xe]=0$ projeksiyon özelliğidir; $\mathbb{E}[e\mid X]=0$ ise daha güçlü olan doğrusal "
            "koşullu ortalama varsayımıdır. Sezgi Deney 2'de eğrilik varken normal denklemler sağlandığı "
            "halde artık ortalamalarının U biçimli olduğunu görebilirsiniz (§1.8.2)."
        ),
    ),
    Question(
        key="d04", concept="projeksiyon-dogrusal-olmayan-cef-icin-tanimli", note=_note("1.6"),
        prompt=(
            r"Koşullu beklenti fonksiyonu doğrusal olmasa bile, $\mathbb{E}[XX']$ terslenebilir olduğu sürece "
            r"doğrusal projeksiyon katsayısı $\beta$ tanımlıdır."
        ),
        answer=TrueFalse(True),
        explanation=(
            r"$\beta=(\mathbb{E}[XX'])^{-1}\mathbb{E}[XY]$ tanımı CEF'nin biçimine dayanmaz. CEF doğrusal "
            r"değilse $X'\beta$, CEF'nin en iyi doğrusal yaklaşımıdır; doğrusalsa CEF'nin kendisidir "
            "(§1.6, Durum 1 ve 2)."
        ),
    ),
    Question(
        key="d05", concept="gozlemsel-veride-x-rassaldir", note=_note("1.2"),
        prompt=(
            "Gözlemsel veride açıklayıcı değişkenler de, sonuç değişkeni gibi, veri üretim sürecinden "
            "gelen rassal değişkenler olarak modellenir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Gözlemsel veride araştırmacı $X$'i sabitlemez; $X$ de aynı veri üretim sürecinin sonucudur. "
            "Bu yüzden $(Y,X)$ birlikte rassal bir vektör olarak ele alınır (§1.2). Laboratuvar deneyinde "
            "ise müdahale araştırmacı tarafından atanır."
        ),
    ),
    Question(
        key="d06", concept="ols-dogru-hesaplar-yorum-farklidir", note=_note("1.11", 0, "Şekil 1.5"),
        prompt=(
            "Sezgi Deney 3'te OLS eğimi nedensel etkiden (0,08) farklı çıkıyorsa, OLS projeksiyon "
            "katsayısını yanlış tahmin ediyor demektir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "OLS, eğitim ile ücret arasındaki en iyi doğrusal projeksiyonu doğru tahmin eder; örneklem "
            "büyüdükçe projeksiyon katsayısına yakınsar. Sorun hesapta değil yorumdadır: gözlenmeyen "
            "yetenek hem eğitimi hem ücreti etkilediğinde projeksiyon katsayısı nedensel etkiye eşit "
            "değildir (§1.11)."
        ),
    ),
    Question(
        key="d07", concept="ham-veri-korunur-temizlik-kodla", note=_note("1.14"),
        prompt=(
            "Veri temizliğini elle bir tabloda yapıp yalnız temizlenmiş dosyayı saklamak, yeniden "
            "üretilebilir bir çalışma için yeterlidir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Ham veri değiştirilmeden korunmalı, bütün temizlik ve dönüşüm adımları kodla yapılmalıdır. "
            "Başka biri aynı ham veriden aynı analiz örneklemine ulaşabilmelidir (§1.14; Uygulama, Adım 2)."
        ),
    ),
    # --- Boşluk doldurma ---------------------------------------------------------
    Question(
        key="b01", concept="kosullu-beklentinin-anlami", note=_note("1.3"),
        prompt=r"$X$ kesikli ise $m(x)=\mathbb{E}[Y\mid X=x]$, $X=x$ olan alt anakütlede $Y$'nin **(1)** ______'dır.",
        answer=FillBlanks((TextBlank(("ortalaması", "ortalama", "beklenen değeri", "beklenen değer"), "ortalaması"),)),
        explanation=(
            "Koşullandırmak, anakütleyi $X=x$ olan birimlere daraltmak demektir; $m(x)$ bu alt "
            "anakütledeki ortalamadır (§1.3). Uygulamanın Adım 4'teki hücre ortalamaları bunun örneklem "
            "karşılığıdır."
        ),
    ),
    Question(
        key="b02", concept="yinelenen-beklentiler", note=_note("1.4"),
        prompt=r"Yinelenen beklentiler yasası: $\mathbb{E}[Y]=\mathbb{E}\big[\,\textbf{(1)}\,\big]$.",
        answer=FillBlanks((TextBlank(("E[Y|X]", "E[Y∣X]", "m(X)", "m(x)"), "E[Y | X] (veya m(X))"),)),
        explanation=(
            r"Önce $X$'e koşullu ortalama alınır, sonra bu ortalamaların $X$'in dağılımı üzerinden "
            r"ortalaması: $\mathbb{E}[Y]=\mathbb{E}[m(X)]$ (§1.4). Uygulamada bu, genel ortalamanın hücre "
            "ortalamalarının gözlem sayısıyla ağırlıklı ortalaması olması demektir."
        ),
    ),
    Question(
        key="b03", concept="ucret-dagiliminin-carpikligi", note=_note("1.13", 3, "Tablo 1.2"),
        prompt=(
            "Tablo 1.2'de ortalama saatlik ücret 23,90, medyan 19,23'tür. Ortalama medyandan büyük "
            "olduğuna göre ücret dağılımı **(1)** ______ çarpıktır."
        ),
        answer=FillBlanks((TextBlank(("sağa", "sağ", "saga", "sag", "pozitif", "pozitif yönde", "sağa doğru"), "sağa"),)),
        explanation=(
            "Sağ kuyruktaki yüksek ücretler ortalamayı medyanın üstüne çeker. Log dönüşümü bu "
            "çarpıklığı azaltır ve farkları oransal okumayı sağlar (§1.13, Adım 3)."
        ),
    ),
    Question(
        key="b04", concept="kukla-ortalamasi-paydir", note=_note("1.13", 3, "Tablo 1.2"),
        prompt="Kadın göstergesinin örneklem ortalaması 0,4257'dir. Buna göre örneklemdeki kadınların payı yüzde **(1)** ______'dir.",
        answer=FillBlanks((NumberBlank(42.57, 0.01, "42,57"),)),
        explanation="0/1 kukla değişkenin ortalaması, 1 değerini alan grubun payıdır (§1.13, Adım 3).",
    ),
    Question(
        key="b05", concept="analiz-orneklemi-tanimi", note=_note("1.13", 2),
        prompt=(
            "Hansen'in cps09mar örnekleminde tam zamanlı çalışan: haftada en az **(1)** ____ saat ve "
            "yılda en az **(2)** ____ hafta çalışan kişidir."
        ),
        answer=FillBlanks((NumberBlank(36, 0.001, "36"), NumberBlank(48, 0.001, "48"))),
        explanation=(
            "Örneklem tanımı, sonuçların kimin için geçerli olduğunu belirler ve tezde açıkça "
            "yazılmalıdır. Bu kısıtlar Hansen'in veri açıklamasında yer alır (§1.13, Adım 2)."
        ),
    ),
    Question(
        key="b06", concept="uyum-degeri-hesabi", note=_note("1.13", 5),
        prompt=(
            r"Model (1): $\widehat{\log(\text{ücret})}=1{,}4396+0{,}1082\cdot \text{eğitim}$. Eğitimi 16 yıl "
            r"olan biri için tahmin edilen log saatlik ücret **(1)** ______'dir (dört ondalık)."
        ),
        answer=FillBlanks((NumberBlank(3.1708, 0.002, "3,1708"),)),
        explanation=(
            r"$1{,}4396+0{,}1082\times16=3{,}1708$. Tablo 1.3'te eğitimi 16 yıl olanların **hücre "
            "ortalaması** ise 3,2011'dir. Aradaki fark, doğrusal projeksiyonun koşullu ortalamayla aynı "
            "nesne olmadığını gösterir (§1.13, Adım 4–5)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="f01", concept="egimin-korelasyon-bicimi", note=_note("1.7"),
        prompt=r"Basit regresyonun anakütle eğimi $\beta$'yı korelasyon $\rho$ ve standart sapmalar cinsinden yazın.",
        answer=Equation(
            lhs=r"\beta",
            symbols=(
                Symbol("rho", r"\rho", "Corr(X, Y)", -0.9, 0.9, aliases=("ρ",)),
                Symbol("sy", r"\sigma_Y", "Y'nin standart sapması", 0.5, 3.0, aliases=("σ_Y", "σY", "sigma_Y")),
                Symbol("sx", r"\sigma_X", "X'in standart sapması", 0.5, 3.0, aliases=("σ_X", "σX", "sigma_X")),
            ),
            answer="rho*sy/sx",
            shown=r"\beta=\rho\,\dfrac{\sigma_Y}{\sigma_X}",
        ),
        explanation=(
            r"$\operatorname{Cov}(X,Y)=\rho\,\sigma_X\sigma_Y$ yazılırsa "
            r"$\beta=\operatorname{Cov}(X,Y)/\operatorname{Var}(X)=\rho\,\sigma_Y/\sigma_X$ olur (§1.7). "
            "Korelasyon birimsizdir; eğim ise $Y$ birimi / $X$ birimi taşır."
        ),
    ),
    Question(
        key="f02", concept="sabit-terim", note=_note("1.7"),
        prompt=r"Sabit terimli basit projeksiyon $Y=\alpha+\beta X+e$, $\mathbb{E}[e]=0$ için sabit terim $\alpha$'yı yazın.",
        answer=Equation(
            lhs=r"\alpha",
            symbols=(
                Symbol("EY", r"\mathbb{E}[Y]", "Y'nin beklenen değeri", -2.0, 3.0, aliases=("E[Y]", "E(Y)", "EY")),
                Symbol("EX", r"\mathbb{E}[X]", "X'in beklenen değeri", -2.0, 3.0, aliases=("E[X]", "E(X)", "EX")),
                BETA,
            ),
            answer="EY - b*EX",
            shown=r"\alpha=\mathbb{E}[Y]-\beta\,\mathbb{E}[X]",
        ),
        explanation=(
            r"Denklemin beklentisi alınırsa $\mathbb{E}[Y]=\alpha+\beta\,\mathbb{E}[X]$ olur; buradan "
            r"$\alpha=\mathbb{E}[Y]-\beta\,\mathbb{E}[X]$ (§1.7). Doğru, ortalamalar noktası "
            r"$(\mathbb{E}[X],\mathbb{E}[Y])$'den geçer."
        ),
    ),
    Question(
        key="f03", concept="egimin-olcu-birimi", note=_note("1.7"),
        prompt=(
            r"Eğitim yıl yerine ay cinsinden ölçülürse ($X_{\text{ay}}=12\,X_{\text{yıl}}$), yeni eğim "
            r"$\beta_{\text{ay}}$'ı eski eğim $\beta$ cinsinden yazın."
        ),
        answer=Equation(lhs=r"\beta_{\text{ay}}", symbols=(BETA,), answer="b/12", shown=r"\beta_{\text{ay}}=\beta/12"),
        explanation=(
            "Eğim $Y$ birimi / $X$ birimi taşır. Bir ay, bir yılın on ikide biri olduğundan bir birimlik "
            "(bir aylık) fark 12 kat küçük bir farktır; eğim 12'ye bölünür. Korelasyon ise birimden "
            "bağımsızdır, değişmez (§1.7)."
        ),
    ),
    Question(
        key="f04", concept="log-duzey-kesin-yuzde", note=_note("1.13.10"),
        prompt=(
            "Log-düzey modelde bir değişkenin katsayısı $b$ ise, o değişkenin bir birim yüksek olmasına "
            "karşılık gelen **kesin** yüzde farkı yazın."
        ),
        answer=Equation(
            lhs=r"\%\Delta",
            symbols=(Symbol("b", "b", "katsayı", -0.5, 0.5),),
            answer="100*(exp(b)-1)",
            shown=r"100\,\{e^{b}-1\}",
        ),
        explanation=(
            "Yaklaşık yorum $100\\,b$'dir ve küçük $b$ için iyi çalışır. Kesin dönüşüm $100\\{e^{b}-1\\}$'dir: "
            "$b=0{,}1148$ için %12,16, $b=-0{,}2596$ için %−22,87 (§1.13.10; Uygulama, Adım 6)."
        ),
    ),
)


KONU01_QUIZ = QuestionSet(
    topic_key="konu01",
    title="Konu 1: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set haftanın temel kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir "
        "bölüme bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. "
        "Notların bölüm sonu egzersizlerini tekrar etmez, onları tamamlar."
    ),
    sections=("1.1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.8.2", "1.10", "1.11",
              "1.12", "1.13", "1.13.10", "1.14"),
)
