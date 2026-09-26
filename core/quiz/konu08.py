"""Konu 8 "Kendini sına" soru seti: çekirdek regresyonu, bant genişliği, seri regresyonu ve kısmen doğrusal model.

Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır. Bölüm sonu egzersizlerindeki
hesaplar (belirli noktalarda çekirdek ağırlıkları, Nadaraya–Watson ortalaması, aday bantlarda MSE ve
CV karşılaştırması, spline baz değeri, 0,25^d oranı, artıklaştırma adımları) tekrar edilmez.
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
        key="k01", concept="parametrik-olmayan-anlami", note=_note("8.1"),
        prompt="\"Parametrik olmayan regresyon\" ifadesi aşağıdakilerden hangisini anlatır?",
        answer=MultipleChoice((
            "Hiçbir varsayım yapılmadığını",
            "$m(\\cdot)$'in biçiminin sonlu sayıda katsayıyla önceden sabitlenmediğini; düzgünlük, bant genişliği ve "
            "destek gibi varsayımların sürdüğünü",
            "Tahminin yalnız gözlemlerin sıralarına dayandığını",
            "Standart hatanın hesaplanamayacağını",
        ), correct=1),
        explanation=(
            "Fonksiyonel biçim varsayımı gevşer, başka düzenlilik varsayımlarına geçilir; esnekliğin karşılığı daha çok "
            "veri ve ayar parametresi seçimidir (§8.1)."
        ),
    ),
    Question(
        key="k02", concept="cekirdek-olcek-carpani", note=_note("8.3"),
        prompt=(
            "$K_h(u)=K(u/h)/h$ tanımındaki $1/h$ çarpanı Nadaraya–Watson tahminini nasıl etkiler?"
        ),
        answer=MultipleChoice((
            "Tahmini $h$ ile çarpar",
            "Sınır yanlılığını ortadan kaldırır",
            "Ağırlıkların bir kısmını negatif yapar",
            "Hiç etkilemez: pay ve paydada ortak olduğu için sadeleşir",
        ), correct=3),
        explanation=(
            "NW bir ağırlıklı ortalamadır; ağırlıklar normalize edildiği için ortak çarpanlar düşer. Uygulamada Gauss "
            "ağırlığı $\\exp(-u^2/2)$ olarak yazılır ve $h$ çekirdeğin standart sapmasıdır (§8.3)."
        ),
    ),
    Question(
        key="k03", concept="yerel-egim-turevi", note=_note("8.4"),
        prompt="Yerel doğrusal regresyonda $x$ noktasındaki eğim tahmini $\\hat b(x)$ neyi yaklaşıklar?",
        answer=MultipleChoice((
            "Koşullu ortalamanın o noktadaki türevini $m'(x)$",
            "Koşullu varyansı",
            "Bant genişliğinin en iyi değerini",
            "Kesme teriminin standart hatasını",
        ), correct=0),
        explanation=(
            "$\\hat a(x)$ koşullu ortalamayı, $\\hat b(x)$ aynı noktadaki türevi yaklaşıklar. Türev tahminleri düzey "
            "tahminlerinden daha gürültülü ve bant genişliğine daha duyarlıdır (§8.4)."
        ),
    ),
    Question(
        key="k04", concept="yanlilik-varyans-mertebeleri", note=_note("8.5"),
        prompt="Bir boyutlu yerel doğrusal tahminde iç noktada yanlılık ve varyans kabaca hangi mertebededir?",
        answer=MultipleChoice((
            "Yanlılık $O(h)$, varyans $O(1/n)$",
            "Yanlılık $O(1/(nh))$, varyans $O(h^2)$",
            "Yanlılık $O(h^2)$, varyans $O(1/(nh))$",
            "İkisi de $h$'den bağımsızdır",
        ), correct=2),
        explanation=(
            "$h$ küçüldükçe yanlılık azalır ama etkili gözlem sayısı ($\\approx nh$) düştüğü için varyans artar; yaklaşık "
            "MSE $C_1h^4+C_2/(nh)$ (§8.5)."
        ),
    ),
    Question(
        key="k05", concept="kumeli-cv-sorunu", note=_note("8.6"),
        prompt="Öğrenciler okul içinde kümeliyse birini dışarıda bırakan CV'nin sorunu nedir?",
        answer=MultipleChoice((
            "Hesaplanamaz",
            "Dışarıda bırakılan öğrencinin okul arkadaşları tahminde kaldığı için performans iyimser ölçülebilir",
            "Her zaman en büyük bant genişliğini seçer",
            "Yalnız Nadaraya–Watson için tanımlıdır",
        ), correct=1),
        explanation=(
            "Kümedeki benzer gözlemler dışarıda bırakılan gözlemin tahminine bilgi sızdırır. Küme-silmeli CV bir okulun "
            "bütün öğrencilerini birlikte dışarıda bırakır; DDK'de iki ölçüt farklı $h$ seçer (§8.6, §8.13)."
        ),
    ),
    Question(
        key="k06", concept="noktasal-eszamanli-bant", note=_note("8.11"),
        prompt=(
            "Eğrinin her noktasında ayrı %95 güven aralığı çizmek neden bütün eğri için %95 güven bandı değildir?"
        ),
        answer=MultipleChoice((
            "Çünkü noktasal aralıklar her zaman çok geniştir",
            "Çünkü parametrik olmayan tahminde güven aralığı tanımsızdır",
            "Çünkü her aralığın ayrı ayrı %95 kapsaması, bütün eğrinin aynı anda bant içinde olma olasılığını %95 yapmaz; "
            "eşanlı bant daha geniş olmalıdır",
            "Çünkü bant genişliği CV ile seçilmiştir",
        ), correct=2),
        explanation=(
            "Noktasal kapsama ile eşanlı (uniform) kapsama farklıdır. Ayrıca düzgünleştirme yanlılığı ihmal edilirse "
            "noktasal aralık bile nominal kapsamayı sağlamayabilir (§8.11)."
        ),
    ),
    Question(
        key="k07", concept="yassi-cv-yorumu", note=_note("8.13", 2),
        prompt=(
            "DDK uygulamasında küme-silmeli CV ölçütünün yaklaşık $h\\in[5,11]$ aralığında yassı olması ne anlama gelir?"
        ),
        answer=MultipleChoice((
            "Bu aralıktaki bantlar neredeyse aynı tahmin hatasını verir; seçilen tek $h$ aşırı yorumlanmamalı ve "
            "duyarlılık gösterilmelidir",
            "Hesaplamada hata vardır",
            "En iyi bant genişliği kesin olarak 6,2'dir",
            "Parametrik doğrusal model daha iyidir",
        ), correct=0),
        explanation=(
            "Ölçüt bu aralıkta 67,425 ile 67,432 arasında (yaklaşık yüzde 0,01) oynar: veri bu bantları ayırt etmez. "
            "Tezde ana $h$ ile birlikte makul alternatiflerde sonucun değişip değişmediği raporlanır (§8.13, Adım 2)."
        ),
    ),
    # --- Doğru–yanlış -------------------------------------------------------------
    Question(
        key="d01", concept="nw-sinir-yanliligi", note=_note("8.3"),
        prompt=(
            "Nadaraya–Watson tahmini, desteğin sınırında eğimli bir gerçek fonksiyonun düzeyini sistematik olarak "
            "kaçırabilir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Sınırda pencerenin bir tarafında veri yoktur; yerel ortalama eğimli eğriyi içeri doğru çeker. Sezgi Deney "
            "2'de $m(x)=1+3x$ için sol sınırdaki NW hatası yaklaşık +0,22'dir (§8.3)."
        ),
    ),
    Question(
        key="d02", concept="ll-dogruyu-yeniden-uretir", note=_note("8.4"),
        prompt=(
            "Gerçek koşullu ortalama doğrusal ise yerel doğrusal tahmin, her $h$ için bu doğruyu (beklenen değerde) "
            "yanlılıksız yeniden üretir."
        ),
        answer=TrueFalse(True),
        explanation=(
            "Her noktada çözülen ağırlıklı EKK bir doğru uydurur; gerçek fonksiyon doğruysa yerel yaklaşım tamdır ve "
            "yanlılık sıfırdır. Bu, sınır noktalarında da geçerlidir (§8.4)."
        ),
    ),
    Question(
        key="d03", concept="cekirdek-turu-ikincil", note=_note("8.5"),
        prompt="Çekirdek türünün (Gauss, Epanechnikov, üçgen) seçimi çoğu uygulamada bant genişliği seçiminden daha önemlidir.",
        answer=TrueFalse(False),
        explanation=(
            "Makul çekirdekler arasındaki fark genellikle ikinci derecededir; $h$'deki büyük değişim ise eğrinin görünümünü "
            "kökten değiştirir (§8.5)."
        ),
    ),
    Question(
        key="d04", concept="seri-esneklik-k-artar", note=_note("8.7"),
        prompt="Seri regresyonunda esneklik, örneklem büyüdükçe baz sayısı $K$'nın sabit tutulmasından gelir.",
        answer=TrueFalse(False),
        explanation=(
            "Sabit $K$ için model sıradan OLS'tir; esneklik $K$'nın örneklemle birlikte artmasına izin verilmesinden gelir. "
            "Çok küçük $K$ yetersiz, çok büyük $K$ aşırı uyum yaratır (§8.7)."
        ),
    ),
    Question(
        key="d05", concept="boyut-yakinsama-hizi", note=_note("8.9"),
        prompt="Boyut arttıkça kernel regresyonunun yakınsama hızı değişmez; yalnız hesaplama süresi artar.",
        answer=TrueFalse(False),
        explanation=(
            "Optimal bant genişliği kabaca $n^{-1/(4+d)}$ mertebesine kayar ve yakınsama boyutla yavaşlar: aynı yerel "
            "hassasiyet için çok daha büyük örneklem gerekir (§8.9)."
        ),
    ),
    Question(
        key="d06", concept="kismen-dogrusal-icsellik", note=_note("8.10"),
        prompt="Kısmen doğrusal modelde $g(X)$'i esnek tahmin etmek, $D$'nin içselliğini çözmez.",
        answer=TrueFalse(True),
        explanation=(
            "Esnek $g(X)$ fonksiyonel biçim hatasını azaltır; gözlenmeyen karıştırıcılar varsa $\\theta$ yine nedensel "
            "değildir. Tanımlama sorusu Konu 3'teki gibi ayrıca yanıtlanmalıdır (§8.10)."
        ),
    ),
    Question(
        key="d07", concept="ddk-loo-kucuk-h-secmez", note=_note("8.13", 2),
        prompt="DDK uygulamasında birini dışarıda bırakan CV, küme-silmeli CV'den daha küçük bir bant genişliği seçer.",
        answer=TrueFalse(False),
        explanation=(
            "Tersi: birini dışarıda bırakan CV $h=12{,}3$, küme-silmeli CV $h=6{,}2$ seçer. İki ölçütün farklı sonuç "
            "vermesi, bağımlılığın ayar aşamasını da etkilediğini gösterir (§8.13, Adım 2)."
        ),
    ),
    # --- Boşluk doldurma ---------------------------------------------------------
    Question(
        key="b01", concept="boyutluluk-laneti-adi", note=_note("8.9"),
        prompt=(
            "Çok boyutlu $X$'te bir gözlemin bütün koordinatlarda aynı anda hedef noktaya yakın olmasının giderek "
            "zorlaşmasına **(1)** ______ denir."
        ),
        answer=FillBlanks((TextBlank(
            ("boyutluluk laneti", "boyut laneti", "boyutsallık laneti", "boyutsallik laneti", "boyutluluk lanetı",
             "curse of dimensionality"),
            "boyutluluk laneti",
        ),)),
        explanation=(
            "Yerel pencere hacmi boyutla üstel küçülür; çözüm yolları kısmen doğrusal/toplamsal yapılar, boyut indirgeme "
            "ve düzenlileştirmedir (§8.9)."
        ),
    ),
    Question(
        key="b02", concept="ddk-iki-h", note=_note("8.6"),
        prompt=(
            "DDK uygulamasında birini dışarıda bırakan CV $h=$ **(1)** ______, küme-silmeli CV ise $h=$ **(2)** ______ "
            "seçer (bir ondalık)."
        ),
        answer=FillBlanks((NumberBlank(12.3, 0.05, "12,3"), NumberBlank(6.2, 0.05, "6,2"))),
        explanation=(
            "Uygulama iki ölçütü $h\\in\\{2{,}0;\\,2{,}1;\\ldots;20{,}0\\}$ ızgarasında hesaplar ve Hansen'in raporladığı "
            "değerleri yeniden üretir (§8.6, §8.13)."
        ),
    ),
    Question(
        key="b03", concept="spline-90-tahmini", note=_note("8.13", 3),
        prompt=(
            "DDK uygulamasında düğümleri 25, 50 ve 75. yüzdelikte olan kübik spline, 90. yüzdelikte toplam test puanını "
            "**(1)** ______ olarak tahmin eder (iki ondalık)."
        ),
        answer=FillBlanks((NumberBlank(20.89, 0.006, "20,89"),)),
        explanation=(
            "Aynı noktada doğrusal model 20,31, kübik polinom 21,22, yerel doğrusal ($h=6{,}2$) 21,06 verir: esnek "
            "yöntemler de birbirinin aynısı değildir (§8.13, Adım 3; Tablo 8.1)."
        ),
    ),
    Question(
        key="b04", concept="rdd-sinir-nedeni", note=_note("8.4"),
        prompt=(
            "Regresyon süreksizliği tasarımında yerel doğrusal tahminin tercih edilmesinin temel nedeni, desteğin "
            "**(1)** ______ noktalarındaki davranışıdır."
        ),
        answer=FillBlanks((TextBlank(
            ("sınır", "sinir", "sınır noktaları", "sınırdaki", "boundary", "kenar", "uç"),
            "sınır",
        ),)),
        explanation=(
            "RDD'de tahmin kesme noktasında, yani her iki tarafın sınırında yapılır; yerel doğrusal tahmin yerel eğimi "
            "hesaba katarak sınır yanlılığını düşürür (§8.4, Konu 9)."
        ),
    ),
    Question(
        key="b05", concept="optimal-h-orani", note=_note("8.5"),
        prompt=(
            "Standart düzgünlük koşullarında bir boyutlu yerel doğrusal tahminde optimal bant genişliği $n^{a}$ "
            "mertebesindedir; $a=$ **(1)** ______ (kesir olarak yazın)."
        ),
        answer=FillBlanks((TextBlank(("-1/5", "−1/5", "-0,2", "-0.2", "−0,2", "-1 / 5"), "−1/5"),)),
        explanation=(
            "$C_1h^4+C_2/(nh)$'yi en küçük yapan $h$, $n^{-1/5}$ ile orantılıdır; MSE ise $n^{-4/5}$ hızıyla azalır: "
            "parametrik $n^{-1}$'den yavaş (§8.5)."
        ),
    ),
    # --- Denklem yazma -----------------------------------------------------------
    Question(
        key="f01", concept="optimal-h-turetme", note=_note("8.5"),
        prompt=(
            "Yaklaşık $MSE(h)=Ah^4+C/(nh)$'yi en küçük yapan bant genişliğini $A$, $C$ ve $n$ cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"h^*",
            symbols=(
                Symbol("A", "A", "yanlılık² katsayısı", 0.1, 3.0),
                Symbol("C", "C", "varyans katsayısı", 0.1, 3.0),
                Symbol("n", "n", "örneklem büyüklüğü", 50.0, 5000.0),
            ),
            answer="(C/(4*A*n))^(1/5)",
            shown=r"\left(\dfrac{C}{4An}\right)^{1/5}",
        ),
        explanation=(
            "Birinci derece koşul $4Ah^3-C/(nh^2)=0$, yani $h^5=C/(4An)$. Sonuç $n^{-1/5}$ oranını açıkça gösterir (§8.5)."
        ),
    ),
    Question(
        key="f02", concept="yerel-dogrusal-kapali-form", note=_note("8.4"),
        prompt=(
            "$x$ noktasında $S_k=\\sum_iK_i(X_i-x)^k$ ve $T_k=\\sum_iK_i(X_i-x)^kY_i$ olsun. Yerel doğrusal tahmini "
            "$\\hat m(x)=\\hat a(x)$'i $S_0,S_1,S_2,T_0,T_1$ cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"\hat m_{LL}(x)",
            symbols=(
                Symbol("S0", "S_0", "Σ K_i", 5.0, 10.0, aliases=("S_0", "S₀")),
                Symbol("S1", "S_1", "Σ K_i (X_i − x)", -1.0, 1.0, aliases=("S_1", "S₁")),
                Symbol("S2", "S_2", "Σ K_i (X_i − x)²", 2.0, 4.0, aliases=("S_2", "S₂")),
                Symbol("T0", "T_0", "Σ K_i Y_i", 1.0, 10.0, aliases=("T_0", "T₀")),
                Symbol("T1", "T_1", "Σ K_i (X_i − x) Y_i", -2.0, 2.0, aliases=("T_1", "T₁")),
            ),
            answer="(S2*T0 - S1*T1)/(S0*S2 - S1^2)",
            shown=r"\dfrac{S_2T_0-S_1T_1}{S_0S_2-S_1^2}",
        ),
        explanation=(
            "Ağırlıklı EKK'nin normal denklemleri $S_0a+S_1b=T_0$, $S_1a+S_2b=T_1$'dir; Cramer kuralıyla $\\hat a$ bulunur. "
            "$S_1=0$ (simetrik komşuluk) iken Nadaraya–Watson oranı $T_0/S_0$'a iner (§8.4)."
        ),
    ),
    Question(
        key="f03", concept="robinson-kimligi", note=_note("8.10"),
        prompt=(
            "$Y=D\\theta+g(X)+e$ ve $\\mathbb E[e\\mid D,X]=0$ ise $Y-\\mathbb E[Y\\mid X]$'i $\\theta$, "
            "$v=D-\\mathbb E[D\\mid X]$ ve $e$ cinsinden yazın."
        ),
        answer=Equation(
            lhs=r"Y-\mathbb E[Y\mid X]",
            symbols=(
                Symbol("t", r"\theta", "ilgilenilen katsayı", -2.0, 2.0, aliases=("θ", "theta")),
                Symbol("v", "v", "D − E[D|X]", -2.0, 2.0),
                Symbol("e", "e", "hata", -2.0, 2.0),
            ),
            answer="t*v + e",
            shown=r"\theta\{D-\mathbb E[D\mid X]\}+e",
        ),
        explanation=(
            "$\\mathbb E[Y\\mid X]=\\theta\\,\\mathbb E[D\\mid X]+g(X)$ olduğu için $g(X)$ düşer. Artıklaştırılmış $D$ ile "
            "$e$ ortogonal olduğundan $\\theta$ artık regresyonuyla tanımlanır (Robinson; §8.10)."
        ),
    ),
    Question(
        key="f04", concept="boyutla-bant-orani", note=_note("8.9"),
        prompt="$d$ boyutlu $X$ ile kernel regresyonunda optimal bant genişliği $n^{a}$ ile orantılıdır. $a$'yı $d$ cinsinden yazın.",
        answer=Equation(
            lhs="a",
            symbols=(Symbol("d", "d", "X'in boyutu", 1.0, 10.0),),
            answer="-1/(4+d)",
            shown=r"-\dfrac{1}{4+d}",
        ),
        explanation=(
            "$d=1$'de $-1/5$'e iner. Boyut arttıkça üs sıfıra yaklaşır: bant yavaş daralır ve tahmin yavaş yakınsar "
            "(§8.9)."
        ),
    ),
    Question(
        key="f05", concept="kubik-spline-parametre-sayisi", note=_note("8.7.1"),
        prompt=(
            "Truncated-power gösteriminde $J$ düğümlü kübik spline'ın (sabit dahil) kaç katsayısı vardır? $J$ cinsinden "
            "yazın."
        ),
        answer=Equation(
            lhs="K",
            symbols=(Symbol("J", "J", "düğüm sayısı", 1.0, 10.0),),
            answer="4 + J",
            shown=r"4+J",
        ),
        explanation=(
            "Kübik polinomun dört katsayısı ($\\beta_0,\\ldots,\\beta_3$) ve her düğüm için bir $\\gamma_j$. DDK'deki üç "
            "düğümlü spline yedi katsayılı bir OLS'tir (§8.7.1)."
        ),
    ),
)


KONU08_QUIZ = QuestionSet(
    topic_key="konu08",
    title="Konu 8: Kendini sına",
    questions=QUESTIONS,
    intro=(
        "Bu set çekirdek ve yerel doğrusal regresyon, bant genişliği seçimi, seri regresyonu, boyutluluk laneti ve "
        "kısmen doğrusal model kavramlarını sınar. Her soru tek bir kavrama odaklanır ve notlardaki bir bölüme "
        "bağlıdır; yanlış cevaplarınız için tekrar edilecek bölümler en üstte listelenir. Bölüm sonu egzersizlerini "
        "tekrar etmez, onları tamamlar."
    ),
    sections=("8.1", "8.3", "8.4", "8.5", "8.6", "8.7", "8.7.1", "8.9", "8.10", "8.11", "8.13"),
)
