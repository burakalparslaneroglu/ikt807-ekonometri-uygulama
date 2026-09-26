"""Konu 3 "Kendini sına" soru seti: tanımlama, nedensellik ve içsellik.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Bölüm sonu egzersizleri
sayısal ayrıştırma ve türetme ister; buradaki sorular aynı konuların farklı ve kısa yönlerini
sınar (egzersizlerdeki sorular tekrar edilmez).
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
        key="k01", concept="tanimlama-tanimi", note=_note("3.1"),
        prompt="Hansen'in tanımına göre bir $\\theta$ parametresinin **tanımlı** olması ne demektir?",
        answer=MultipleChoice((
            "Tahmin edicisinin standart hatasının küçük olması",
            "Gözlenen veri dağılımı $F$'nin bir fonksiyonu olarak tekil biçimde yazılabilmesi: $\\theta=g(F)$",
            "Örneklemde istatistiksel olarak anlamlı çıkması",
            "Modelin $R^2$'sinin yüksek olması",
        ), correct=1),
        explanation=(
            "Tanımlama bir örneklem özelliği değil, anakütle dağılımı ve açık varsayımlar düzeyinde bir özelliktir. "
            "Tanımlı olmayan bir parametreyi daha çok veri veya daha karmaşık tahmin edici kurtaramaz (§3.1)."
        ),
    ),
    Question(
        key="k02", concept="ham-fark-bilesenleri", note=_note("3.3"),
        prompt="Ham grup farkı $\\mathbb E[Y\\mid D=1]-\\mathbb E[Y\\mid D=0]$ hangi iki bileşene ayrılır?",
        answer=MultipleChoice((
            "ATE ile örnekleme hatasına",
            "ATE ile ATT'ye",
            "LATE ile seçim farkına",
            "ATT ile seçim farkı $\\mathbb E[Y(0)\\mid D=1]-\\mathbb E[Y(0)\\mid D=0]$'a",
        ), correct=3),
        explanation=(
            "Bir terim ekleyip çıkarınca ham fark = ATT + seçim farkı. Seçim farkı, tedavi hiç uygulanmasaydı bile iki "
            "grubun farklı olup olmayacağını ölçer; Sezgi Deney 1'de ikisi ayrı ayrı görülür (§3.3)."
        ),
    ),
    Question(
        key="k03", concept="araci-degisken-kontrolu", note=_note("3.10"),
        prompt=(
            "Tedavinin sonucu etkilediği kanal üzerindeki bir **aracı** (mediator) değişkeni regresyona kontrol "
            "olarak eklemek ne yapar?"
        ),
        answer=MultipleChoice((
            "Toplam etkinin o kanaldan geçen kısmını bloke eder; tahmin hedefi değişir",
            "Seçim yanlılığını ortadan kaldırır",
            "Yalnız standart hatayı küçültür, katsayıyı etkilemez",
            "Tedaviyi rastgele atanmış hale getirir",
        ), correct=0),
        explanation=(
            "İdeal kontrol tedaviden önce belirlenmiş ortak nedendir. Aracı değişkeni kontrol etmek toplam etkiyi değil, "
            "o kanal dışındaki etkiyi tahmin etmeye yol açar; kontrol sayısını artırmak modeli otomatik iyileştirmez (§3.10)."
        ),
    ),
    Question(
        key="k04", concept="acik-arka-kapi-yolu", note=_note("3.6"),
        prompt=(
            "Gözlenmeyen $U$ hem $D$'yi hem $Y$'yi etkiliyor ($D\\leftarrow U\\rightarrow Y$) ve $U$ gözlenmiyor. "
            "$D$ ile $Y$ arasındaki ilişki neyi taşır?"
        ),
        answer=MultipleChoice((
            "Yalnız $D\\rightarrow Y$ nedensel etkisini",
            "Yalnız $U$'nun $Y$ üzerindeki etkisini",
            "$D\\rightarrow Y$ etkisiyle birlikte $U$ kaynaklı ilişkiyi",
            "Hiçbir ilişki taşımaz; iki yol birbirini götürür",
        ), correct=2),
        explanation=(
            "Açık arka kapı yolu kapanmadıkça $D$–$Y$ ilişkisi nedensel yolu ve karıştırıcı yolu birlikte taşır. $U$ "
            "gözlenirse koşullandırmayla kapatılabilir; gözlenmezse basit kontrol yetersiz kalır (§3.6)."
        ),
    ),
    Question(
        key="k05", concept="kovaryat-orneklem-degisimi", note=_note("3.15", 3),
        prompt=(
            "DDK uygulamasında kovaryat ayarlı modelin $n$'si ham farktan küçüktür (5.135 < 5.795). "
            "Bunun nedeni nedir?"
        ),
        answer=MultipleChoice((
            "Kümelenmiş standart hata bazı okulları dışarıda bırakır",
            "Bütün kontrol değişkenleri her öğrenci için gözlenmediğinden, eksik değerli öğrenciler tahmin örnekleminden çıkar",
            "Randomization başarısız olduğu için bazı öğrenciler çıkarılmıştır",
            "Kovaryat eklemek gözlemleri otomatik olarak ağırlıklandırır",
        ), correct=1),
        explanation=(
            "Sütunlar arası fark yalnız \"kontroller eklendi\" değildir; örneklem kompozisyonu da değişmiştir. Bir tabloyu "
            "okurken her sütunun N'sine bakılmalıdır (§3.15, Adım 3)."
        ),
    ),
    Question(
        key="k06", concept="r2-tanimlama-kaniti-degil", note=_note("3.11"),
        prompt=(
            "Bir tezde \"modelimizin $R^2$'si 0,85; dolayısıyla eğitim katsayısı nedensel etkiyi ölçer\" yazıyor. "
            "Bu çıkarımdaki hata nedir?"
        ),
        answer=MultipleChoice((
            "0,85 düşük bir $R^2$'dir; en az 0,90 gerekir",
            "$R^2$ yalnız kategorik değişkenler için anlamlıdır",
            "Hata yoktur; yüksek $R^2$ içselliği dışlar",
            "Yüksek $R^2$ öngörü başarısını gösterir; tanımlama varsayımı hakkında bilgi vermez",
        ), correct=3),
        explanation=(
            "Güçlü biçimde endojen bir değişken sonucu çok iyi öngörebilir; tersine rastgele bir deneyde etki temiz "
            "tanımlansa bile $R^2$ düşük olabilir. Tahmin sorusu ile nedensel soru farklıdır (§3.11)."
        ),
    ),
    Question(
        key="k07", concept="secim-mekanizmasi-aciklanmali", note=_note("3.8.3"),
        prompt="\"$X$ bireyin kendi seçimi olduğu için endojendir\" ifadesi neden tek başına yeterli değildir?",
        answer=MultipleChoice((
            "Seçim mekanizmasının sonuç denklemindeki gözlenmeyen bileşenle neden bağlantılı olduğu açıklanmalıdır",
            "Bireysel seçimler her zaman dışsaldır",
            "Seçim değişkenleri regresyona hiç giremez",
            "Endojenlik yalnız ölçüm hatasından doğar",
        ), correct=0),
        explanation=(
            "Karar değişkeni, kararı etkileyen gözlenmeyen yetenek veya beklenti sonucu da etkiliyorsa endojendir. İyi bir "
            "çalışma endojenlik iddiasını ekonomik davranış mekanizmasıyla temellendirir (§3.8.3)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="veri-tanimlamayi-cozmez", note=_note("3.1"),
        prompt="Daha fazla veri toplamak, tanımlanmamış bir parametreyi tanımlı hale getirir.",
        answer=TrueFalse(False),
        explanation=(
            "Daha fazla veri gözlenen dağılımı daha hassas öğrenmemizi sağlar; hedefi o dağılıma bağlayan varsayım yoksa "
            "tanımlama eksikliğini tek başına gidermez (§3.1)."
        ),
    ),
    Question(
        key="d02", concept="nedensel-cikarimin-temel-problemi", note=_note("3.2"),
        prompt=(
            "Bireysel nedensel etki $Y_i(1)-Y_i(0)$ doğrudan hesaplanamaz, çünkü aynı birey için iki potansiyel sonuç "
            "aynı anda gözlenemez."
        ),
        answer=TrueFalse(True),
        explanation=(
            "$D_i=1$ ise $Y_i(1)$ gözlenir, $Y_i(0)$ karşı-olgusal kalır; $D_i=0$ için tersi. Nedensel etki bu yüzden "
            "veri setinde bir sütun değildir; ortalama etkiler bir tanımlama argümanıyla elde edilir (§3.2)."
        ),
    ),
    Question(
        key="d03", concept="denge-tablosu-sinav-degil", note=_note("3.4"),
        prompt=(
            "Rastgele atanmış bir deneyin denge tablosunda bir kovaryatın p-değerinin 0,05'in altında çıkması, "
            "randomization'ın başarısız olduğunu kanıtlar."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Rassal atama sonlu örneklemde tam denge garanti etmez; çok sayıda kovaryat incelenince bazı farkların yalnız "
            "şansla belirgin görünmesi olağandır. DDK Tablo 3.1'de beş kovaryattan dördünün p < 0,10 olduğunu görün (§3.4)."
        ),
    ),
    Question(
        key="d04", concept="kosullu-bagimsizlik-test-edilemez", note=_note("3.5"),
        prompt="Koşullu bağımsızlık varsayımı $\\{Y(0),Y(1)\\}\\perp D\\mid X$ gözlenen veriyle doğrudan test edilebilir.",
        answer=TrueFalse(False),
        explanation=(
            "Varsayım gözlenemeyen karşı-olgusal sonuçlar hakkındadır; veri doğrudan test edemez. İkna ediciliği alan "
            "bilgisi, veri toplama süreci ve değişkenlerin zamanlamasıyla savunulur (§3.5)."
        ),
    ),
    Question(
        key="d05", concept="ols-projeksiyon-hedefini-tutarli-tahmin-eder", note=_note("3.7"),
        prompt=(
            "İçsellik altında da OLS kendi hedefi olan doğrusal projeksiyon katsayısını tutarlı tahmin eder; sorun, "
            "bu hedefin yapısal parametreden farklı olmasıdır."
        ),
        answer=TrueFalse(True),
        explanation=(
            "$\\beta^*=\\beta+\\{\\mathbb E[XX']\\}^{-1}\\mathbb E[Xe]$ ve $\\hat\\beta_{OLS}\\to_p\\beta^*$. OLS "
            "\"yanlış hesap\" yapmaz; doğru biçimde farklı bir anakütle nesnesini tahmin eder (§3.7)."
        ),
    ),
    Question(
        key="d06", concept="icsellikte-kapsama-sifira-gider", note=_note("3.9"),
        prompt=(
            "İçsellik varken örneklem büyüdükçe, OLS'nin %95 güven aralığının yapısal parametreyi kapsama olasılığı "
            "sıfıra yaklaşır."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Standart hata $1/\\sqrt n$ hızıyla küçülür ama aralığın merkezi $\\beta^*\\neq\\beta$ etrafında kalır; "
            "sonunda aralık $\\beta$'yı dışarıda bırakır. Sezgi Deney 3'te n'yi artırıp kapsamanın düştüğünü görün (§3.9)."
        ),
    ),
    Question(
        key="d07", concept="collider-kontrolu", note=_note("3.10"),
        prompt=(
            "Tedavinin ve sonucun ortak sonucu olan bir değişkeni (collider) kontrol etmek, daha önce kapalı olan bir "
            "ilişki yolunu açarak yeni yanlılık yaratabilir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Collider'a koşullanmak, iki nedeni arasında veride olmayan bir ilişki yaratır. \"Bütün mevcut değişkenleri "
            "ekledim\" yaklaşımı nedensel hedef için hatalı olabilir (§3.10)."
        ),
    ),
    # --- Boşluk doldurma ---------------------------------------------------------
    Question(
        key="b01", concept="zayiflama-yonu", note=_note("3.8.4"),
        prompt="Açıklayıcı değişkendeki klasik ölçüm hatası OLS eğimini **(1)** ______ doğru çeker.",
        answer=FillBlanks((TextBlank(
            ("sıfıra", "sifira", "sıfır", "0'a", "0", "sıfır değerine"),
            "sıfıra",
        ),)),
        explanation=(
            "plim $\\hat\\beta=\\beta\\cdot\\operatorname{Var}(Z)/\\{\\operatorname{Var}(Z)+\\operatorname{Var}(u)\\}$; "
            "çarpan 0 ile 1 arasındadır (attenuation). Sezgi Deney 2'de Var(u) arttıkça eğim küçülür (§3.8.4)."
        ),
    ),
    Question(
        key="b02", concept="ate-tanimi", note=_note("3.2"),
        prompt="Ortalama nedensel etki $ATE=\\mathbb E[\\,$**(1)** ______$\\,]$ olarak tanımlanır.",
        answer=FillBlanks((TextBlank(
            ("Y(1)-Y(0)", "Y1-Y0", "Y_1-Y_0", "Y_i(1)-Y_i(0)", "Yi(1)-Yi(0)", "Y(1) - Y(0)", "tau", "τ"),
            "Y(1) − Y(0)",
        ),)),
        explanation=(
            "ATE, bireysel etkilerin $\\tau_i=Y_i(1)-Y_i(0)$ anakütle ortalamasıdır. Hansen aynı nesneyi ortalama "
            "nedensel etki (ACE) olarak adlandırır (§3.2)."
        ),
    ),
    Question(
        key="b03", concept="atama-birimi", note=_note("3.15", 1),
        prompt="DDK deneyinde tracking müdahalesi öğrenci düzeyinde değil, **(1)** ______ düzeyinde atanmıştır.",
        answer=FillBlanks((TextBlank(("okul", "okullar", "okul düzeyinde", "school"), "okul"),)),
        explanation=(
            "Atama birimi okul olduğu için aynı okuldaki öğrencilerin hataları bağımsız sayılmaz ve standart hatalar okul "
            "düzeyinde kümelenir; bu seçim tasarımın parçasıdır (§3.15, Adım 1)."
        ),
    ),
    Question(
        key="b04", concept="ortak-destek-adi", note=_note("3.5"),
        prompt=(
            "Koşullu bağımsızlığa ek olarak, ilgili her $x$ bölgesinde hem tedavi hem kontrol gözlemi bulunmasını "
            "isteyen $0<P(D=1\\mid X=x)<1$ koşuluna **(1)** ______ denir."
        ),
        answer=FillBlanks((TextBlank(
            ("ortak destek", "ortakdestek", "örtüşme", "ortusme", "overlap", "common support"),
            "ortak destek (örtüşme)",
        ),)),
        explanation=(
            "Ortak destek yoksa bazı profillerde karşı-olgusalı veriden öğrenemeyiz; esnek modeller yalnız veri desteğinin "
            "dışına ekstrapolasyon yapar (§3.5)."
        ),
    ),
    Question(
        key="b05", concept="standart-puan-yorumu", note=_note("3.15", 4),
        prompt=(
            "Tracking okullarında $std\\_mark$'ın $lowstream$ üzerine regresyonunda $\\hat\\delta=-1{,}574$. "
            "$std\\_mark$ standartlaştırılmış bir puan olduğundan, düşük akıştaki öğrencilerin başlangıç puanı "
            "yaklaşık **(1)** ____ standart sapma daha düşüktür."
        ),
        answer=FillBlanks((NumberBlank(1.574, 0.03, "1,57"),)),
        explanation=(
            "İkili göstergenin katsayısı grup ortalamaları farkıdır; sonuç standart sapma biriminde olduğu için fark "
            "yaklaşık 1,6 standart sapmadır. Bu büyüklük lowstream'in rastgele değil, başlangıç başarısına göre "
            "oluştuğunu gösterir (§3.15, Adım 4)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="f01", concept="gozlenen-sonuc-denklemi", note=_note("3.2"),
        prompt="Gözlenen sonucu $Y$'yi tedavi göstergesi $D$ ve potansiyel sonuçlar $Y(1)$, $Y(0)$ cinsinden yazın.",
        answer=Equation(
            lhs="Y",
            symbols=(
                Symbol("d", "D", "tedavi göstergesi (0 veya 1)", 0.0, 1.0, aliases=("D",)),
                Symbol("y1", "Y(1)", "tedavi altında sonuç", 0.0, 10.0, aliases=("Y(1)", "Y_1", "Y1", "y(1)")),
                Symbol("y0", "Y(0)", "tedavisiz sonuç", 0.0, 10.0, aliases=("Y(0)", "Y_0", "Y0", "y(0)")),
            ),
            answer="d*y1 + (1 - d)*y0",
            shown=r"Y=D\,Y(1)+(1-D)\,Y(0)",
        ),
        explanation=(
            "Her birey için yalnız bir potansiyel sonuç gözlenir: $D=1$ ise $Y(1)$, $D=0$ ise $Y(0)$. Eşdeğer yazım: "
            "$Y=Y(0)+D\\{Y(1)-Y(0)\\}$ (§3.2)."
        ),
    ),
    Question(
        key="f02", concept="denge-fiyati", note=_note("3.8.2"),
        prompt=(
            "Talep $Q=-\\beta_dP+u_d$ ve arz $Q=\\beta_sP+u_s$ iken denge fiyatı $P$'yi $u_d$, $u_s$, $\\beta_d$ ve "
            "$\\beta_s$ cinsinden yazın."
        ),
        answer=Equation(
            lhs="P",
            symbols=(
                Symbol("ud", "u_d", "talep şoku", -2.0, 2.0, aliases=("u_d",)),
                Symbol("us", "u_s", "arz şoku", -2.0, 2.0, aliases=("u_s",)),
                Symbol("bd", r"\beta_d", "talep eğimi (pozitif)", 0.5, 2.0, aliases=("β_d", "βd", "beta_d")),
                Symbol("bs", r"\beta_s", "arz eğimi (pozitif)", 0.5, 2.0, aliases=("β_s", "βs", "beta_s")),
            ),
            answer="(ud - us)/(bd + bs)",
            shown=r"P=\dfrac{u_d-u_s}{\beta_d+\beta_s}",
        ),
        explanation=(
            "İki denklemi eşitleyip $P$'yi çözün. Denge fiyatı $u_d$'yi içerdiği için talep denkleminde $P$ ile $u_d$ "
            "ilişkilidir: eşanlılık bir içsellik kaynağıdır (§3.8.2)."
        ),
    ),
    Question(
        key="f03", concept="zayiflama-olasilik-limiti", note=_note("3.8.4"),
        prompt=(
            "Klasik ölçüm hatasında ($X=Z+u$) basit regresyon eğiminin olasılık limitini $\\beta$, "
            "$\\operatorname{Var}(Z)$ ve $\\operatorname{Var}(u)$ cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\operatorname{plim}\hat\beta_{OLS}",
            symbols=(
                Symbol("b", r"\beta", "yapısal eğim", -2.0, 2.0, aliases=("β", "beta")),
                Symbol("vz", r"\operatorname{Var}(Z)", "gerçek değişkenin varyansı", 0.5, 3.0,
                       aliases=("Var(Z)", "VarZ", "var(z)")),
                Symbol("vu", r"\operatorname{Var}(u)", "ölçüm hatasının varyansı", 0.1, 3.0,
                       aliases=("Var(u)", "Varu", "var(u)")),
            ),
            answer="b*vz/(vz + vu)",
            shown=r"\beta\,\dfrac{\operatorname{Var}(Z)}{\operatorname{Var}(Z)+\operatorname{Var}(u)}",
        ),
        explanation=(
            "Çarpan (güvenilirlik oranı) 0 ile 1 arasındadır; eğim sıfıra doğru çekilir. Sezgi Deney 2'de Var(Z) = 1 "
            "iken çarpan 1/(1 + Var(u))'dur (§3.8.4)."
        ),
    ),
    Question(
        key="f04", concept="olcum-hatasi-kovaryansi", note=_note("3.8.4"),
        prompt=(
            "$X=Z+u$ ve $Y=\\beta X+v$, $v=\\varepsilon-\\beta u$ olsun ($u$; $Z$ ve $\\varepsilon$'dan bağımsız). "
            "$\\operatorname{Cov}(X,v)$'yi $\\beta$ ve $\\operatorname{Var}(u)$ cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\operatorname{Cov}(X,v)",
            symbols=(
                Symbol("b", r"\beta", "yapısal eğim", -2.0, 2.0, aliases=("β", "beta")),
                Symbol("vu", r"\operatorname{Var}(u)", "ölçüm hatasının varyansı", 0.1, 3.0,
                       aliases=("Var(u)", "Varu", "var(u)")),
            ),
            answer="-b*vu",
            shown=r"\operatorname{Cov}(X,v)=-\beta\operatorname{Var}(u)",
        ),
        explanation=(
            "$\\operatorname{Cov}(Z+u,\\varepsilon-\\beta u)=-\\beta\\operatorname{Var}(u)\\neq0$: gözlenen $X$ yeni hata "
            "terimiyle ilişkilidir, yani ölçüm hatası $X$'i endojen yapar (§3.8.4)."
        ),
    ),
    Question(
        key="f05", concept="icsellik-olasilik-limiti", note=_note("3.9"),
        prompt=(
            "$Y=\\alpha+\\beta X+e$ ve $\\operatorname{Cov}(X,e)\\neq0$ iken basit regresyon eğiminin olasılık limitini "
            "$\\beta$, $\\operatorname{Cov}(X,e)$ ve $\\operatorname{Var}(X)$ cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\operatorname{plim}\hat\beta_{OLS}",
            symbols=(
                Symbol("b", r"\beta", "yapısal eğim", -2.0, 2.0, aliases=("β", "beta")),
                Symbol("c", r"\operatorname{Cov}(X,e)", "regresör ile hatanın kovaryansı", -1.0, 1.0,
                       aliases=("Cov(X,e)", "Cov(X, e)", "cov(x,e)")),
                Symbol("v", r"\operatorname{Var}(X)", "regresörün varyansı", 0.5, 3.0,
                       aliases=("Var(X)", "VarX", "var(x)")),
            ),
            answer="b + c/v",
            shown=r"\beta+\dfrac{\operatorname{Cov}(X,e)}{\operatorname{Var}(X)}",
        ),
        explanation=(
            "İkinci terim $n$ ile sıfıra gitmez: içsellik bir tutarlılık ve tanımlama problemidir, örneklem büyütmek onu "
            "düzeltmez (§3.9)."
        ),
    ),
)


KONU03_QUIZ = QuestionSet(
    topic_key="konu03",
    title="Konu 3: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set tanımlama, nedensellik ve içsellik kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki "
        "bir bölüme bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Bölüm sonu "
        "egzersizlerini tekrar etmez, onları tamamlar."
    ),
    sections=("3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8.2", "3.8.3", "3.8.4", "3.9", "3.10", "3.11",
              "3.15"),
)
