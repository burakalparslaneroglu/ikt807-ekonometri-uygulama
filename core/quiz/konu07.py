"""Konu 7 "Kendini sına" soru seti: koşullu kantiller, check-loss ve kantil profilleri.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Bölüm sonu egzersizlerindeki
hesaplar (kantil tanımı, mutlak kayıp ve check-loss değerleri, negatif artık payı, doğrusal kantil
yorumu, paralel kantil fonksiyonları, kesişme noktası, bootstrap tasarımı) tekrar edilmez.
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
        key="k01", concept="kantil-secme-gerekcesi", note=_note("7.1"),
        prompt="Bir araştırmada kantil regresyon kullanmanın en sağlam gerekçesi hangisidir?",
        answer=MultipleChoice((
            "Uç değerlerden kurtulmak için OLS'nin daha dayanıklı bir sürümü olması",
            "Standart hatalarının OLS'ninkinden her zaman küçük olması",
            "Araştırma sorusunun koşullu dağılımın belirli bir noktasıyla (ör. alt yüzde 10 eşiği) ilgili olması",
            "Koşullu ortalamayı OLS'den daha kesin tahmin etmesi",
        ), correct=2),
        explanation=(
            "Kantil regresyon farklı bir tahmin hedefidir: ortalama ile medyan ya da yüzde 90 kantili farklı anakütle "
            "nesneleridir. Yöntem seçimi estimand'dan başlar; \"robust olsun\" tek başına gerekçe değildir (§7.1, §7.14)."
        ),
    ),
    Question(
        key="k02", concept="kantil-moment-kosulu-anlami", note=_note("7.4"),
        prompt=(
            "$\\psi_\\tau(u)=\\tau-\\mathbf 1(u<0)$ olmak üzere $\\mathbb E[\\psi_\\tau\\{Y-q_\\tau(X)\\}\\mid X]=0$ "
            "koşulu neyi düzenler?"
        ),
        answer=MultipleChoice((
            "Her $X$ değerinde gözlemlerin $\\tau$ payının koşullu kantilin altında kalmasını",
            "Hataların koşullu ortalamasının sıfır olmasını",
            "Hataların koşullu varyansının sabit olmasını",
            "Artıkların normal dağılmasını",
        ), correct=0),
        explanation=(
            "Koşul $P\\{Y<q_\\tau(X)\\mid X\\}=\\tau$ demektir: OLS'deki $\\mathbb E[e\\mid X]=0$'ın kantil karşılığıdır, "
            "ama hataların ortalamasını değil işaretlerinin oranını düzenler (§7.4)."
        ),
    ),
    Question(
        key="k03", concept="en-iyi-dogrusal-kantil", note=_note("7.5"),
        prompt=(
            "Gerçek koşullu kantil fonksiyonu doğrusal değilse "
            "$\\beta_\\tau=\\arg\\min_b\\mathbb E[\\rho_\\tau(Y-X'b)]$ neyi tanımlar?"
        ),
        answer=MultipleChoice((
            "Hiçbir şeyi; model yanlış belirtildiği için parametre tanımsızdır",
            "Koşullu ortalamanın doğrusal projeksiyonunu",
            "Gerçek kantil fonksiyonunun $X$'in ortalamasındaki türevini",
            "Check-loss anlamında en iyi doğrusal kantil öngörücüsünü",
        ), correct=3),
        explanation=(
            "OLS'deki koşullu beklenti–en iyi doğrusal projeksiyon ayrımının kantil karşılığıdır: $X'\\beta_\\tau$, "
            "gerçek $q_\\tau(x)$ doğrusal değilse veri dağılımına göre ağırlıklandırılmış bir yaklaşımdır (§7.5)."
        ),
    ),
    Question(
        key="k04", concept="kantil-tahmin-hesabi", note=_note("7.6"),
        prompt="Kantil regresyon tahmin edicisi $\\hat\\beta_\\tau$ nasıl hesaplanır?",
        answer=MultipleChoice((
            "Kapalı form $(X'X)^{-1}X'Y$ ile",
            "Parçalı doğrusal amaç fonksiyonunun doğrusal programlama (veya başka bir sayısal yöntem) ile en "
            "küçüklenmesiyle",
            "Normal dağılım varsayımıyla en çok olabilirlikle",
            "Artık karelerini ağırlıklandıran tek adımlı ağırlıklı en küçük karelerle",
        ), correct=1),
        explanation=(
            "Genel kapalı form yoktur. Kesin çözüm doğrusal programlamadır (R `rq`, Stata `qreg`); yinelemeli yaklaşımlar "
            "kesin çözüme yalnız yaklaşır. CPS'de τ = 0,90 eğitim katsayısı kesin çözümde 0,1226'dır (§7.6, §7.16)."
        ),
    ),
    Question(
        key="k05", concept="yogunluk-ve-varyans", note=_note("7.11"),
        prompt=(
            "Kantil regresyonun asimptotik varyansında hedef kantildeki koşullu yoğunluk $f\\{q_\\tau(x)\\mid x\\}$ "
            "neden yer alır?"
        ),
        answer=MultipleChoice((
            "Yoğunluk arttıkça tahmin daha belirsiz olduğu için",
            "Yalnız heteroskedastisite düzeltmesi için",
            "Hedef kantilin çevresinde az gözlem varsa küçük bir olasılık değişimi büyük bir kantil değişimine karşılık "
            "gelir; varyans büyür",
            "Kantil regresyonu aslında bir yoğunluk tahmin edicisi olduğu için",
        ), correct=2),
        explanation=(
            "Tek bir kantilde varyans yaklaşık $\\tau(1-\\tau)/\\{nf(q_\\tau)^2\\}$'dir: yoğunluk düşükse varyans büyür. "
            "Sezgi Deney 3'te normal hatada τ = 0,90 eğiminin standart sapması medyandakinin yaklaşık 1,36 katıdır (§7.11)."
        ),
    ),
    Question(
        key="k06", concept="kesisme-nerede-belirgin", note=_note("7.12"),
        prompt="Ayrı ayrı tahmin edilen kantil doğrularının kesişmesi en çok nerede görülür?",
        answer=MultipleChoice((
            "Veri desteğinin zayıf olduğu uç bölgelerde ve çok sayıda regresör olduğunda",
            "Yalnız medyan regresyonunda",
            "Örneklem büyüdükçe daha sık",
            "Yalnız hata homoskedastik olduğunda",
        ), correct=0),
        explanation=(
            "Kesişme örnekleme hatası, zayıf veri desteği veya doğrusal spesifikasyonun farklı kantillerde uyumsuz "
            "davranmasından doğar; örneklem dışı bölgelerde ve çok regresörlü modellerde daha belirgindir (§7.12)."
        ),
    ),
    Question(
        key="k07", concept="fark-testi-ortak-kovaryans", note=_note("7.16", 2),
        prompt=(
            "CPS uygulamasında $\\operatorname{Var}(\\hat\\beta_{0,90}-\\hat\\beta_{0,10})$, iki kantil varyansının "
            "toplamından küçüktür. Neden?"
        ),
        answer=MultipleChoice((
            "Üst kantilin standart hatası daha küçük olduğu için",
            "Fark testi normal değil $t$ dağılımı kullandığı için",
            "Kantil tahminleri birbirinden bağımsız olduğu için",
            "İki tahmin aynı veriden geldiği için kovaryansları pozitiftir; varyanstan kovaryansın iki katı çıkarılır",
        ), correct=3),
        explanation=(
            "Ortak kovaryans $\\{\\min(\\tau_1,\\tau_2)-\\tau_1\\tau_2\\}H_1^{-1}X'XH_2^{-1}$'dir; 0,10 ve 0,90 için ağırlık "
            "0,01. Eğitimde farkın SH'si 0,0021; bağımsızlık varsayılsaydı $\\sqrt{0{,}0017^2+0{,}0014^2}\\approx0{,}0023$ "
            "olurdu (§7.16, Adım 2)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="lad-y-yonunde-saglam", note=_note("7.2"),
        prompt="Medyan regresyonu, $Y$ yönündeki uç gözlemlerin tahmine etkisine OLS'den daha az duyarlıdır.",
        answer=TrueFalse(True),
        explanation=(
            "Mutlak kayıp artıkla doğrusal, kareli kayıp karesel büyür. Sezgi Deney 2'de gözlemlerin onda biri büyük "
            "şok aldığında LAD eğiminin standart sapması OLS'ninkinin yaklaşık 0,4 katıdır. Bu, iki yöntemin aynı "
            "estimand'ı hedeflediği anlamına gelmez (§7.2)."
        ),
    ),
    Question(
        key="d02", concept="kantil-katsayisi-bireysel-degil", note=_note("7.7"),
        prompt=(
            "$\\beta_{0{,}9}$ katsayısı, $X_j$ arttığında yüksek ücretli bireylerin ücretinin ne kadar değişeceğini verir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Katsayı koşullu dağılımın 0,9 kantilindeki farktır. Aynı bireyin bu kantilde kaldığını (sıra "
            "değişmezliği) varsaymadıkça bireysel etki çıkarılamaz (§7.7)."
        ),
    ),
    Question(
        key="d03", concept="anlamli-anlamsiz-fark-degil", note=_note("7.8"),
        prompt=(
            "Kantil profilinde 0,25 katsayısı anlamlı, 0,75 katsayısı anlamsızsa iki katsayının birbirinden farklı "
            "olduğu sonucuna varılabilir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Komşu kantil tahminleri aynı veriden gelir ve bağımsız değildir. Eşitlik için ortak kovaryansı kullanan bir "
            "fark testi veya ortak bootstrap gerekir; güven bantlarının örtüşmesi de formal test değildir (§7.8)."
        ),
    ),
    Question(
        key="d04", concept="ols-medyan-hangisi-dogru-sorusu", note=_note("7.9"),
        prompt=(
            "OLS eğimi ile medyan regresyonu eğimi farklı çıktığında sorulacak ilk soru \"hangisi doğru?\" sorusudur."
        ),
        answer=TrueFalse(False),
        explanation=(
            "İki katsayı farklı estimand'lardır: koşullu ortalama ve koşullu medyan. İlk soru hangi koşullu konum "
            "ölçüsünün hedeflendiğidir. Simetrik ve yalnız konum kaymalı yapılarda ikisi yakın olabilir (§7.9)."
        ),
    ),
    Question(
        key="d05", concept="kantil-farki-bireysel-etki-kantili-degil", note=_note("7.13"),
        prompt=(
            "İki potansiyel sonuç dağılımının $\\tau$ kantilleri arasındaki fark, bireysel tedavi etkilerinin $\\tau$ "
            "kantiline eşittir."
        ),
        answer=TrueFalse(False),
        explanation=(
            "Kantil farkı iki marjinal dağılımı karşılaştırır; bireysel etkilerin dağılımı ortak dağılımı gerektirir. "
            "İki kavram tezde ayrı ayrı tanımlanmalıdır (§7.13)."
        ),
    ),
    Question(
        key="d06", concept="kumeli-bootstrap-birimi", note=_note("7.11"),
        prompt="Kümeli veride kantil regresyon için bootstrap, gözlemleri değil kümeleri yeniden örneklemelidir.",
        answer=TrueFalse(True),
        explanation=(
            "Tek tek gözlemleri yeniden örneklemek küme içi bağımlılığı bozar ve belirsizliği olduğundan küçük gösterir; "
            "kümeler blok olarak çekilir. Tekrar sayısı ve kümelenme birimi raporlanır (§7.11)."
        ),
    ),
    Question(
        key="d07", concept="cps-uclarda-sh-buyuk", note=_note("7.16", 1),
        prompt=(
            "CPS uygulamasında eğitim katsayısının Hendricks–Koenker standart hatası τ = 0,10 ve τ = 0,90'da medyandakinden "
            "büyüktür."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Standart hatalar sırasıyla 0,0017 ve 0,0014, medyanda 0,0009'dur: dağılımın kuyruklarında koşullu yoğunluk "
            "düşüktür ve aynı örneklem daha az bilgi taşır (§7.16, Adım 1)."
        ),
    ),
    # --- Boşluk doldurma ---------------------------------------------------------
    Question(
        key="b01", concept="lad-adi", note=_note("7.2"),
        prompt=(
            "Medyan regresyonu mutlak sapmaların toplamını en küçük yaptığı için İngilizce literatürde **(1)** ______ "
            "(LAD) olarak da adlandırılır."
        ),
        answer=FillBlanks((TextBlank(
            ("least absolute deviations", "least absolute deviation", "en küçük mutlak sapmalar",
             "en küçük mutlak sapma", "en kucuk mutlak sapma", "en kucuk mutlak sapmalar"),
            "least absolute deviations (en küçük mutlak sapmalar)",
        ),)),
        explanation=(
            "$\\hat\\beta_{0{,}5}=\\arg\\min_b\\sum_i|Y_i-X_i'b|$; medyan mutlak sapmanın beklenen değerini en küçük yapan "
            "sayıdır (§7.2)."
        ),
    ),
    Question(
        key="b02", concept="check-loss-adi", note=_note("7.3"),
        prompt=(
            "Genel $\\tau$ kantili için kullanılan $\\rho_\\tau(u)=u\\{\\tau-\\mathbf 1(u<0)\\}$ asimetrik mutlak kaybına "
            "**(1)** ______ fonksiyonu denir."
        ),
        answer=FillBlanks((TextBlank(
            ("check-loss", "check loss", "check", "checkloss", "kontrol kaybı", "kontrol kaybi", "pinball",
             "pinball loss"),
            "check-loss",
        ),)),
        explanation=(
            "τ = 0,5'te $\\rho_{0{,}5}(u)=|u|/2$ ve LAD elde edilir; τ büyüdükçe pozitif artıklar daha ağır cezalandırılır "
            "ve tahmin dağılımın üstüne taşınır (§7.3)."
        ),
    ),
    Question(
        key="b03", concept="kantil-kesismesi-adi", note=_note("7.12"),
        prompt=(
            "Ayrı tahmin edilen kantil doğrularının bazı $X$ değerlerinde $\\hat q_{0{,}25}(x)>\\hat q_{0{,}75}(x)$ "
            "olmasına kantil **(1)** ______ denir."
        ),
        answer=FillBlanks((TextBlank(
            ("kesişmesi", "kesişme", "kesismesi", "kesisme", "crossing", "quantile crossing"),
            "kesişmesi (quantile crossing)",
        ),)),
        explanation=(
            "Gerçek koşullu kantiller sıralıdır; kesişme tahmin edilen koşullu dağılımın o bölgede kendi içinde "
            "tutarsız olduğunu gösterir (§7.12)."
        ),
    ),
    Question(
        key="b04", concept="cps-ust-kantil-egitim", note=_note("7.16", 1),
        prompt=(
            "CPS uygulamasında τ = 0,90'da eğitim katsayısı **(1)** ______'dır (dört ondalık); tam dönüşümle bir ek "
            "eğitim yılı yaklaşık yüzde **(2)** ______ daha yüksek ücretle ilişkilidir (bir ondalık)."
        ),
        answer=FillBlanks((NumberBlank(0.1226, 0.00006, "0,1226"), NumberBlank(13.0, 0.06, "13,0"))),
        explanation=(
            "$100\\{\\exp(0{,}1226)-1\\}\\approx13{,}0$. Alt kantilde katsayı 0,1036, yüzde karşılığı 10,9 (§7.16, Adım 1)."
        ),
    ),
    Question(
        key="b05", concept="cps-fark-z", note=_note("7.16", 2),
        prompt=(
            "CPS uygulamasında eğitim katsayısının 0,90 ve 0,10 kantilleri arasındaki farkı için $z$ istatistiği yaklaşık "
            "**(1)** ______'dır (iki ondalık)."
        ),
        answer=FillBlanks((NumberBlank(8.90, 0.006, "8,90"),)),
        explanation=(
            "Fark 0,0189, ortak kovaryansla SH 0,0021: $z\\approx8{,}90$. Kadın katsayısında aynı test $z\\approx-10{,}04$ "
            "verir (§7.16, Adım 2)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="f01", concept="check-loss-turevi", note=_note("7.4"),
        prompt=(
            "Beklenen check-loss $\\mathbb E[\\rho_\\tau(Y-a)]$'nın $a$'ya göre türevini $\\tau$ ve $F=P(Y<a)$ "
            "cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\partial\,\mathbb E[\rho_\tau(Y-a)]/\partial a",
            symbols=(
                Symbol("t", r"\tau", "kantil düzeyi", 0.05, 0.95, aliases=("τ", "tau")),
                Symbol("F", "F", "P(Y<a)", 0.05, 0.95),
            ),
            answer="F - t",
            shown=r"P(Y<a)-\tau",
        ),
        explanation=(
            "$\\rho_\\tau'(u)=\\psi_\\tau(u)$ ve $\\partial(Y-a)/\\partial a=-1$ olduğundan türev "
            "$-\\mathbb E[\\tau-\\mathbf 1(Y<a)]=P(Y<a)-\\tau$'dur. Sıfıra eşitlemek $P(Y<a)=\\tau$ verir (§7.4)."
        ),
    ),
    Question(
        key="f02", concept="orneklem-kantili-varyansi", note=_note("7.11"),
        prompt=(
            "Tek bir örneklem kantilinin asimptotik varyansını $\\tau$, örneklem büyüklüğü $n$ ve hedef kantildeki "
            "yoğunluk $f$ cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\operatorname{Var}(\hat q_\tau)",
            symbols=(
                Symbol("t", r"\tau", "kantil düzeyi", 0.05, 0.95, aliases=("τ", "tau")),
                Symbol("n", "n", "örneklem büyüklüğü", 50.0, 5000.0),
                Symbol("f", "f", "hedef kantildeki yoğunluk f(q_τ)", 0.05, 0.5),
            ),
            answer="t*(1-t)/(n*f^2)",
            shown=r"\dfrac{\tau(1-\tau)}{n\,f(q_\tau)^2}",
        ),
        explanation=(
            "Pay kantilin altında kalma göstergesinin varyansı, payda kantili olasılık ölçeğinden değer ölçeğine "
            "çeviren yoğunluktur. Regresyonda aynı yapı sandviç biçimini alır: $\\tau(1-\\tau)H^{-1}X'XH^{-1}$ (§7.11)."
        ),
    ),
    Question(
        key="f03", concept="fark-varyansi", note=_note("7.16", 2),
        prompt=(
            "$v_1=\\operatorname{Var}(\\hat\\beta_{\\tau_1})$, $v_2=\\operatorname{Var}(\\hat\\beta_{\\tau_2})$ ve "
            "$c=\\operatorname{Cov}(\\hat\\beta_{\\tau_1},\\hat\\beta_{\\tau_2})$ ise farkın varyansını yazın."
        ),
        answer=Equation(
            lhs=r"\operatorname{Var}(\hat\beta_{\tau_2}-\hat\beta_{\tau_1})",
            symbols=(
                Symbol("v1", "v_1", "birinci kantil tahmininin varyansı", 0.5, 3.0, aliases=("v_1", "v₁")),
                Symbol("v2", "v_2", "ikinci kantil tahmininin varyansı", 0.5, 3.0, aliases=("v_2", "v₂")),
                Symbol("c", "c", "iki tahminin kovaryansı", -0.4, 0.4),
            ),
            answer="v1 + v2 - 2*c",
            shown=r"v_1+v_2-2c",
        ),
        explanation=(
            "Aynı veriden gelen iki kantil tahmininde $c>0$ olduğu için farkın varyansı $v_1+v_2$'den küçüktür. "
            "Uygulamada $c$, Hendricks–Koenker parçalarından hesaplanır (§7.16, Adım 2)."
        ),
    ),
    Question(
        key="f04", concept="konum-olcek-kantil-egimi", note=_note("7.8"),
        prompt=(
            "$Y=1+X+(1+\\gamma X)e$, $e\\sim N(0,1)$ ve $X$'ten bağımsız. Koşullu $\\tau$ kantilinin $X$'e göre eğimini "
            "$\\gamma$ ve $z=\\Phi^{-1}(\\tau)$ cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\partial Q_\tau(Y\mid X)/\partial X",
            symbols=(
                Symbol("g", r"\gamma", "ölçek eğimi", 0.0, 1.0, aliases=("γ", "gamma")),
                Symbol("z", "z", "standart normal kantil Φ⁻¹(τ)", -2.0, 2.0),
            ),
            answer="1 + g*z",
            shown=r"1+\gamma\,\Phi^{-1}(\tau)",
        ),
        explanation=(
            "$Q_\\tau(Y\\mid X)=1+z+(1+\\gamma z)X$. $\\gamma=0$ iken bütün eğimler 1'dir (paralel kantiller); $\\gamma>0$ "
            "iken üst kantil eğimi büyür, alt kantil eğimi küçülür. Sezgi Deney 1 bunu gösterir (§7.8)."
        ),
    ),
    Question(
        key="f05", concept="log-kantil-yuzde-donusumu", note=_note("7.10"),
        prompt=(
            "Log ücret kantil regresyonunda eğitim katsayısı $b$ ise bir ek eğitim yılıyla ilişkili kesin yüzde farkı yazın."
        ),
        answer=Equation(
            lhs=r"\%\,\Delta",
            symbols=(Symbol("b", "b", "kantil katsayısı", -0.5, 0.5, aliases=("β", "beta")),),
            answer="100*(exp(b)-1)",
            shown=r"100\,(e^{b}-1)",
        ),
        explanation=(
            "Küçük katsayılarda $100b$ yaklaşık aynı sonucu verir; farklar büyüdükçe tam dönüşüm kullanılır. Koşullu log "
            "ücret kantilindeki fark, monoton dönüşüm nedeniyle ücret kantilindeki orantılı farka karşılık gelir (§7.10)."
        ),
    ),
)


KONU07_QUIZ = QuestionSet(
    topic_key="konu07",
    title="Konu 7: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set koşullu kantil, check-loss, kantil katsayılarının yorumu, çıkarım ve kantil profilleri kavramlarını "
        "sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir bölüme bağlıdır; yanlış cevaplarınız için tekrar "
        "edilecek bölümler en üstte listelenir. Bölüm sonu egzersizlerini tekrar etmez, onları tamamlar."
    ),
    sections=("7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8", "7.9", "7.10", "7.11", "7.12", "7.13", "7.16"),
)
