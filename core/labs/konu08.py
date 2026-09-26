"""Konu 8 uygulama laboratuvarı: DDK verisinde bant genişliği bir model kararıdır.

Ders notları §8.13'ün (Adım 1–4 ve §8.13.5) birebir karşılığıdır; Adım 3 aynı zamanda §8.8'deki
Tablo 8.1'i yeniden üretir. Veri: Hansen'in DDK2011 dosyası (Duflo, Dupas ve Kremer 2011),
tracking uygulanan okullardaki kız öğrenciler; test puanı, başlangıç yüzdeliği ve okul bilgisi
eksiksiz 1.487 öğrenci (60 okul). Her ``Check`` notlarda basılı bir sayıdır.

Yerel doğrusal tahmin Gauss çekirdeğiyle; h çekirdeğin standart sapmasıdır. Çapraz doğrulama
h ∈ {2,0; 2,1; …; 20,0} ızgarasında; küme-silmeli CV bir okulun bütün öğrencilerini birlikte
dışarıda bırakır. Kübik spline düğümleri 25, 50 ve 75. yüzdeliktedir (truncated-power bazı).
"""

from __future__ import annotations

from core.labs import expr as E
from core.labs.spec import (
    OLS,
    BandwidthCV,
    BinMeans,
    Check,
    Curve,
    Derive,
    DropMissing,
    KeepIf,
    LabSpec,
    LabStep,
    LoadHansen,
    LocalCurve,
    LocalLinear,
    ModelLine,
    NoteRef,
    Plot,
    ProfileCurves,
    ReproClass,
    ScalarTarget,
    StatTarget,
    TableTarget,
)

SECTION = "8.13"
FRAME = "ddk"
CLUSTER = "schoolid"
X = "percentile"
Y = "totalscore"
KNOTS = (25, 50, 75)
POINTS = (10.0, 50.0, 90.0)
BINS = 20
H_CLUSTER, H_LOO = 6.2, 12.3
CUBIC = (X, "pct2", "pct3")
SPLINE = (*CUBIC, *(f"s{k}" for k in KNOTS))

# Tablo 8.3 (§8.13.3) ve Tablo 8.1 (§8.8): seçilmiş yüzdeliklerde tahmin edilen toplam test puanı.
PARAMETRIC = {
    "dogrusal": (7.57, 13.94, 20.31),
    "kubik": (8.09, 13.26, 21.22),
    "spline": (8.02, 13.56, 20.89),
}
LOCAL = {"h6_2": (8.02, 13.49, 21.06), "h12_3": (8.04, 13.56, 21.33)}
_LABELS = {"dogrusal": "Doğrusal", "kubik": "Kübik polinom", "spline": "Kübik spline",
           "h6_2": "Yerel doğrusal h = 6,2", "h12_3": "Yerel doğrusal h = 12,3"}


def _basis(x: E.Expr) -> dict[str, E.Expr]:
    """Seri regresyonunun baz fonksiyonları, x'in fonksiyonu olarak (Derive, profil ve eğri için aynı).

    Kare terim 100'e, kübik terimler 10⁴'e bölünür: katsayılar okunur ölçekte olur, tahminler değişmez.
    """

    terms = {X: x, "pct2": E.div(E.power(x, 2), 100), "pct3": E.div(E.power(x, 3), 10000)}
    terms.update({f"s{k}": E.div(E.power(E.maximum(E.sub(x, k), 0), 3), 10000) for k in KNOTS})
    return terms


BASIS = _basis(E.var(X))


def _fit(model: str, terms: tuple[str, ...]) -> E.Expr:
    """Tahmin edilen seri regresyonu m̂(x) = b₀ + Σ bⱼ pⱼ(x), grafik eğrisi için."""

    expression: E.Expr = E.coef(model, E.INTERCEPT)
    for term in terms:
        expression = E.add(expression, E.mul(E.coef(model, term), BASIS[term]))
    return expression


def _prediction_checks() -> tuple[Check, ...]:
    checks: list[Check] = []
    for model, values in PARAMETRIC.items():
        checks += [
            Check(f"{_LABELS[model]}, yüzdelik {int(point)}", TableTarget("parametrik", point, model), value, 2)
            for point, value in zip(POINTS, values)
        ]
    for column, values in LOCAL.items():
        checks += [
            Check(f"{_LABELS[column]}, yüzdelik {int(point)}", TableTarget("yerel", point, column), value, 2)
            for point, value in zip(POINTS, values)
        ]
    return tuple(checks)


STEPS = (
    LabStep(
        number=1,
        title="Araştırma sorusu: ilişki doğrusal mı?",
        note=NoteRef(SECTION, 1),
        explanation=(
            "Duflo–Dupas–Kremer (2011) verisinde tracking uygulanan okullardaki kız öğrencilerin dönem sonu toplam "
            "test puanını, başlangıç yüzdelik diliminin esnek bir fonksiyonu olarak inceliyoruz. Soru: başlangıç "
            "başarısı ile dönem sonu puanı arasındaki ilişki doğrusal mı, yoksa özellikle dağılımın uçlarında eğrilik "
            "var mı?\n\n"
            "Parametrik doğrusal model $\\mathbb E[Y\\mid X=x]=\\beta_0+\\beta_1x$ tek bir eğim dayatır. Aşağıdaki "
            "grafikte 1.487 tekil nokta yerine $x$'in yirmi eşit genişlikli aralığındaki ortalamalar gösterilir; "
            "doğru bütün gözlemlerle tahmin edilmiştir. Öğrenciler okul içinde bağımlı olduğu için standart hatalar "
            "okul düzeyinde kümelenmiştir."
        ),
        operations=(
            LoadHansen("ddk2011", "DDK2011.dta", FRAME),
            KeepIf(FRAME, (("tracking", "==", 1), ("girl", "==", 1)),
                   "Analiz örneklemi: tracking uygulanan okullardaki kız öğrenciler"),
            DropMissing(FRAME, (Y, X, CLUSTER), "Test puanı, başlangıç yüzdeliği veya okul bilgisi eksik gözlemler"),
            OLS("dogrusal", FRAME, Y, (X,), vcov="cluster", cluster=CLUSTER),
            Plot(
                FRAME, X,
                (
                    BinMeans(Y, BINS, "Yirmi aralığın ortalaması"),
                    ModelLine("dogrusal", "Doğrusal OLS"),
                ),
                "Başlangıç yüzdelik dilimi", "Toplam test puanı", "DDK2011: başlangıç başarısı ve test puanı",
            ),
        ),
        checks=(Check("Analiz örneklemi (N)", StatTarget(FRAME, Y, "count"), 1487, decimals=0),),
        reproducibility=ReproClass.EXACT,
        takeaway=(
            "Doğru genel artışı yakalar, ama aralık ortalamaları alt uçta doğrunun üstünde, ortada altında ve üst uçta "
            "yeniden üstünde kalır: tek eğim bu eğriliği ortalamaya indirger. Yerel doğrusal tahmin her $x$ noktasında "
            "yakın gözlemlere daha çok ağırlık vererek koşullu ortalama eğrisini tahmin eder."
        ),
    ),
    LabStep(
        number=2,
        title="İki farklı CV yaklaşımı iki farklı h seçer",
        note=NoteRef(SECTION, 2, ("Şekil 8.4",)),
        explanation=(
            "Yerel doğrusal tahminde Gauss çekirdeği $K(u)=\\phi(u)$, $u=(X_i-x)/h$ kullanılır; $h$ çekirdeğin standart "
            "sapmasıdır. Çapraz doğrulama ölçütü dışarıda bırakılan tahmin hatalarının kareler ortalamasıdır:\n\n"
            "$$CV(h)=\\frac1n\\sum_{i=1}^n\\{Y_i-\\tilde m_{-i}(X_i;h)\\}^2.$$\n\n"
            "**Birini dışarıda bırakan** CV'de $\\tilde m_{-i}$ yalnız $i$. gözlem çıkarılarak, **küme-silmeli** CV'de "
            "$i$'nin okulundaki bütün öğrenciler çıkarılarak hesaplanır. Ölçüt $h\\in\\{2{,}0;\\,2{,}1;\\,\\ldots;\\,20{,}0\\}$ "
            "ızgarasında değerlendirilir. Hansen konvansiyonel CV'nin yaklaşık $h=12{,}3$, küme-silmeli CV'nin yaklaşık "
            "$h=6{,}2$ seçtiğini raporlar."
        ),
        operations=(
            BandwidthCV("cv", FRAME, X, Y, (2.0, 20.0, 0.1), "cv_tablosu", "Bant genişliği h",
                        "DDK2011: çapraz doğrulama ölçütü", cluster=CLUSTER),
            Plot(
                FRAME, X,
                (
                    BinMeans(Y, BINS, "Yirmi aralığın ortalaması"),
                    ModelLine("dogrusal", "Doğrusal OLS"),
                    LocalCurve(Y, H_CLUSTER, "Yerel doğrusal h = 6,2"),
                    LocalCurve(Y, H_LOO, "Yerel doğrusal h = 12,3", dashed=True),
                ),
                "Başlangıç yüzdelik dilimi", "Toplam test puanı", "DDK2011: bant genişliği ve tahmin edilen profil",
            ),
        ),
        checks=(
            Check("Birini dışarıda bırakan CV ile h", ScalarTarget("cv_h"), H_LOO, decimals=1),
            Check("Küme-silmeli CV ile h", ScalarTarget("cv_h_kume"), H_CLUSTER, decimals=1),
        ),
        reproducibility=ReproClass.EXACT,
        takeaway=(
            "Küme-silmeli ölçüt yaklaşık 5–11 aralığında çok yassıdır: bu aralıktaki bütün h değerleri neredeyse aynı "
            "tahmin hatası verir, yani veri tek bir bant genişliğini keskin biçimde ayırt etmez. Yazılımın seçtiği tek "
            "sayı \"doğru bant genişliği\" değildir; ölçütün en küçük değer çevresindeki şekli de raporlanmalıdır."
        ),
        code_note=(
            "Çapraz doğrulama n×n ağırlık matrisiyle vektörel hesaplanır (1.487 öğrenci için yaklaşık 2,2 milyon "
            "öğe); 181 bant genişliği ve iki ölçüt birkaç saniye sürer. Aynı fonksiyonlar Python'da NumPy, R'de "
            "matris işlemleri, Stata'da Mata ile yazılmıştır. Eşit CV değerlerinde küçük h seçilir."
        ),
    ),
    LabStep(
        number=3,
        title="Aynı veri, farklı düzgünlük",
        note=NoteRef(SECTION, 3, ("Tablo 8.3", "Tablo 8.1", "Şekil 8.3")),
        explanation=(
            "Aynı koşullu ortalamayı beş biçimde tahmin ediyoruz: doğrusal model; kübik polinom "
            "$\\beta_0+\\beta_1x+\\beta_2x^2+\\beta_3x^3$; düğümleri 25, 50 ve 75. yüzdelikte olan kübik spline "
            "(kübik polinoma $(x-25)_+^3$, $(x-50)_+^3$, $(x-75)_+^3$ terimleri eklenir); iki bant genişliğinde "
            "yerel doğrusal tahmin ($h=6{,}2$ ve $h=12{,}3$). Polinom ve spline sıradan OLS ile tahmin edilir; esneklik "
            "baz fonksiyonlarından gelir (katsayılar okunur ölçekte olsun diye kare terim 100'e, kübik terimler "
            "$10^4$'e bölünür; tahminler değişmez). Tahminler 10., 50. ve 90. yüzdelikte karşılaştırılır."
        ),
        operations=(
            Derive(FRAME, "pct2", BASIS["pct2"], "Yüzdeliğin karesi / 100"),
            Derive(FRAME, "pct3", BASIS["pct3"], "Yüzdeliğin küpü / 10⁴"),
            *(Derive(FRAME, f"s{k}", BASIS[f"s{k}"], f"Spline terimi (yüzdelik − {k})₊³ / 10⁴") for k in KNOTS),
            OLS("kubik", FRAME, Y, CUBIC, vcov="cluster", cluster=CLUSTER),
            OLS("spline", FRAME, Y, SPLINE, vcov="cluster", cluster=CLUSTER),
            ProfileCurves(
                models=(("dogrusal", "Doğrusal"), ("kubik", "Kübik polinom"), ("spline", "Kübik spline")),
                frame=FRAME,
                variable=X,
                derived=tuple((name, BASIS[name]) for name in SPLINE[1:]),
                values=POINTS,
                result="parametrik",
                plot_grid=(0.0, 100.0, 101),
                x_label="Başlangıç yüzdelik dilimi",
                y_label="Toplam test puanı",
                title="DDK2011: parametrik uyumlar",
            ),
            LocalLinear("ll", FRAME, X, Y, (("h6_2", H_CLUSTER), ("h12_3", H_LOO)), POINTS, "yerel"),
            Plot(
                FRAME, X,
                (
                    BinMeans(Y, BINS, "Yirmi aralığın ortalaması"),
                    ModelLine("dogrusal", "Doğrusal"),
                    Curve(_fit("kubik", CUBIC), "Kübik polinom"),
                    Curve(_fit("spline", SPLINE), "Kübik spline (düğümler 25, 50, 75)", dashed=True),
                    LocalCurve(Y, H_CLUSTER, "Yerel doğrusal h = 6,2"),
                ),
                "Başlangıç yüzdelik dilimi", "Toplam test puanı",
                "DDK2011: doğrusal, polinom, spline ve yerel doğrusal uyumlar",
            ),
        ),
        checks=_prediction_checks(),
        reproducibility=ReproClass.EXACT,
        takeaway=(
            "Orta bölgede yöntemler birbirine yakındır (13,3–13,9). Üst uçta esnek yöntemler doğrusal modelden 0,6–1 "
            "puan yüksek tahmin üretir: 90. yüzdelikte doğrusal 20,31, spline 20,89, yerel doğrusal 21,06–21,33. Büyük "
            "h eğriyi düzleştirir, küçük h yerel değişimleri izler. Farklılıkların hiçbiri tek başına \"doğru model\" "
            "kanıtı değildir: veri desteği, tuning ölçütü ve makul alternatiflerde sonucun değişip değişmediği birlikte "
            "değerlendirilir."
        ),
        code_note=(
            "Kübik polinom ve spline OLS'tir; baz terimleri açıkça türetilir, tahminler 10., 50. ve 90. yüzdelikte "
            "x'β olarak hesaplanır. Standart hatalar okul düzeyinde kümelenmiştir (tahminleri etkilemez)."
        ),
    ),
    LabStep(
        number=4,
        title="Kümeli yapıyı tuning aşamasına da taşımak",
        note=NoteRef(SECTION, 4),
        explanation=(
            "Öğrenciler okul içinde bağımlıdır. Birini dışarıda bırakan CV tek bir öğrenciyi çıkarırken aynı okuldan çok "
            "benzer gözlemleri tahminde tutar; bu nedenle tahmin performansını gereğinden iyimser değerlendirebilir. "
            "Hansen'in önerdiği küme-silmeli yaklaşım bir okulun bütün öğrencilerini "
            "birlikte dışarıda bırakır (Adım 2).\n\n"
            "Bağımlılık yalnız standart hata sorununa indirgenemez. Veri kümeli olduğunda standart hata, çapraz "
            "doğrulama, bootstrap ve eğitim/test bölmesi gibi bütün yeniden örnekleme veya doğrulama adımlarının "
            "kümeyi koruması gerekir."
        ),
        takeaway=(
            "Tez cümlesi örneği: \"Yerel doğrusal tahminde ana bant genişliği küme-silmeli çapraz doğrulama ile "
            "seçilmiş, ayrıca daha geniş bir bantla duyarlılık analizi yapılmıştır. Sonuç fonksiyonunun genel şekli "
            "korunmakla birlikte üst destek bölgesindeki tahminler bant genişliğine daha duyarlıdır.\""
        ),
    ),
    LabStep(
        number=5,
        title="Bir parametrik olmayan makale çıktısını okurken",
        note=NoteRef("8.13.5", 0),
        explanation=(
            "1. Tahmin edilen nesne koşullu ortalama mı, yoğunluk mu, olasılık mı?\n"
            "2. Çekirdek ve yerel polinom derecesi nedir?\n"
            "3. Bant genişliği nasıl seçilmiştir?\n"
            "4. Veri bağımlıysa CV ve standart hata kümeyi koruyor mu?\n"
            "5. Grafikte sınır bölgelerinde güven aralıkları genişliyor mu?\n"
            "6. Farklı makul $h$ değerleri ana ekonomik sonucu değiştiriyor mu?\n"
            "7. Çok boyutlu $X$ kullanılıyorsa boyutluluk laneti nasıl yönetiliyor?"
        ),
        takeaway=(
            "Bu laboratuvarın cevapları: koşullu ortalama; Gauss çekirdeği, yerel doğrusal; küme-silmeli CV "
            "(h = 6,2) ve duyarlılık için h = 12,3; kümeli yapı hem CV'de hem standart hatada korunur; üst uçtaki "
            "tahminler h'ye en duyarlı bölgedir."
        ),
    ),
)


KONU08_LAB = LabSpec(
    topic_key="konu08",
    title="DDK Verisinde Bant Genişliği Bir Model Kararıdır",
    dataset="ddk2011",
    note_section=SECTION,
    steps=STEPS,
    labels=(
        (E.INTERCEPT, "Sabit"),
        (X, "Başlangıç yüzdeliği"),
        (Y, "Toplam test puanı"),
        ("pct2", "Yüzdelik²/100"),
        ("pct3", "Yüzdelik³/10⁴"),
        *((f"s{k}", f"(yüzdelik − {k})₊³/10⁴") for k in KNOTS),
        ("dogrusal", "Doğrusal"),
        ("kubik", "Kübik polinom"),
        ("spline", "Kübik spline"),
    ),
    consistency_notes=(
        "Notların Tablo 8.1'indeki kübik spline sütunu (7,97 / 13,46 / 20,81) düğümleri belirtilmemiş bir betikten "
        "geliyordu ve yeniden üretilemiyordu. Notlar ve laboratuvar düğümleri 25, 50 ve 75. yüzdelikte olan "
        "truncated-power kübik spline'a göre güncellendi: 8,02 / 13,56 / 20,89.",
    ),
)
