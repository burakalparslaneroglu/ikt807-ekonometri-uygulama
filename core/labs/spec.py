"""Ders notu uygulama laboratuvarlarının dilden bağımsız tanım şeması.

Bir laboratuvar tanımı dört şeyi birlikte besler:

1. Uygulama arayüzündeki adım adım anlatım,
2. Uygulamanın kendi hesabı (``core.labs.runner``),
3. Python, R ve Stata kodu (``core.codegen``),
4. Notlardaki sayılarla karşılaştıran testler.

Bir sayı veya işlem yalnız burada değişir; diğer dört çıktı kendiliğinden izler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Union

from core.labs.expr import Expr


class ReproClass(str, Enum):
    """Üç dilde aynı sayının hangi anlamda beklenebileceği."""

    EXACT = "birebir"
    CONVENTION = "ayar"
    DISTRIBUTIONAL = "dagilim"


REPRO_DESCRIPTIONS = {
    ReproClass.EXACT: (
        "Birebir aynı",
        "Deterministik hesap. Python, R ve Stata aynı sayıyı ondalık düzeyinde verir.",
    ),
    ReproClass.CONVENTION: (
        "Ayar sabitlenince aynı",
        "Paketlerin varsayılan ayarları farklı. Kodda ayar açıkça sabitlendiği için sonuç aynıdır; "
        "varsayılan ayarla çalıştırırsanız farklı sayı görebilirsiniz.",
    ),
    ReproClass.DISTRIBUTIONAL: (
        "Yalnız dağılımda aynı",
        "Rastgele çekiliş içerir. Diller farklı rastgele sayı üreteci kullandığı için aynı seed aynı "
        "çekilişi vermez; sonuçlar Monte Carlo hatası kadar farklılaşır.",
    ),
}

STATISTICS = ("count", "sum", "mean", "sd", "median", "min", "max")
VCOV_TYPES = ("classic", "HC1", "cluster")
BINARY_LINKS = ("logit", "probit")
BINARY_VCOV_TYPES = ("classic", "robust")
KEEP_OPERATORS = ("==", "!=", "<", "<=", ">", ">=")
SOURCES = ("notlar", "alternatif", "kendi")
"""Uygulama sekmesinin veri kaynakları (``LabSpec.source``): notlardaki örnek, alternatif örnek (aynı adımlar, Hansen
arşivindeki başka bir veriyle) ve öğrencinin kendi verisi (Excel ya da CSV dosyası)."""
FILE_FORMATS = ("xlsx", "csv")
FILE_COLUMN_KINDS = ("metin", "kod", "sayi", "sayi_metin")
"""Yüklenen dosyadaki bir sütunun koddaki dönüşümü: ``metin`` metin (kategori etiketi; tam sayı hücreler "12"
olur), ``kod`` tam sayı kodlu kategorik değişken (1, 2, 3 → "1", "2", "3"), ``sayi`` sayısal sütun, ``sayi_metin``
metin olarak saklanmış sayı (ondalık virgül noktaya çevrilir). Eksik değerler her türde eksik kalır."""


# --- İşlemler ---------------------------------------------------------------

@dataclass(frozen=True)
class LoadHansen:
    """Hansen'in veri arşivinden bir veri setini yükler.

    Hansen'in ``.txt`` dosyaları başlıksızdır; ``columns`` değişken adlarını veri
    setinin açıklama belgesindeki sırayla verir. ``.dta`` dosyaları adlarını kendi
    taşır; bu durumda ``columns`` boş bırakılır ve adlar küçük harfe çevrilir.
    """

    dataset: str
    member: str
    frame: str
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReadFile:
    """Öğrencinin yüklediği Excel (.xlsx) ya da CSV dosyası ("Kendi verini yükle").

    Kod dosyayı okur, ``columns`` sütunlarını seçip koddaki (ASCII) adlarıyla yeniden adlandırır ve metin hücrelerini
    temizler: bölünmez boşluk boşluğa çevrilir, baştaki ve sondaki boşluklar silinir, boş kalan hücre ve "NA" eksik
    değerdir. Sonra ``required`` sütunlarından birinde eksik değer olan satırlar çıkarılır ve sütun türleri dönüştürülür
    (``FILE_COLUMN_KINDS``). ``rows`` bu işlemlerden sonraki değerlerdir: uygulamanın hesabı bunlardan yapılır, üretilen
    kod aynı değerleri dosyadan elde eder (testle denetlenir). R, CSV dosyasının bütün sütunlarını metin olarak okur ve
    sayıları açıkça dönüştürür; böylece iki dilin tür tahmini birbirinden ayrılamaz. Stata kodu üretilmez.
    """

    frame: str
    file_name: str
    file_format: str
    columns: tuple[tuple[str, str, str], ...]
    """(koddaki ad, dosyadaki sütun adı, dönüşüm türü)."""
    rows: tuple[tuple[object, ...], ...]
    comment: str
    sheet: str | None = None
    separator: str = ","
    decimal: str = "."
    encoding: str = "utf-8-sig"
    dropped: int = 0
    """Boş hücre nedeniyle çıkarılan satır sayısı."""
    required: tuple[str, ...] = ()
    """Eksik değeri satırı çıkaran sütunlar (koddaki adlar); boşsa bütün sütunlar."""
    strip_names: bool = False
    """Dosyadaki sütun adlarının baştaki ve sondaki boşlukları silinir (Excel'de sık görülen bir yazım)."""

    def __post_init__(self) -> None:
        if self.file_format not in FILE_FORMATS:
            raise ValueError(f"Desteklenmeyen dosya biçimi: {self.file_format}")
        if any(kind not in FILE_COLUMN_KINDS for _, _, kind in self.columns):
            raise ValueError("Tanımsız sütun dönüşümü.")


@dataclass(frozen=True)
class CompleteCases:
    """``source`` çerçevesinde ``columns`` sütunlarının hepsinde değeri olan satırlar: ``frame`` adlı yeni çerçeve.

    Satır numaraları 1'den yeniden başlar; kaynak çerçeve değişmez.
    """

    frame: str
    source: str
    columns: tuple[str, ...]
    comment: str


@dataclass(frozen=True)
class Indicator:
    """Kategorik bir değişkenden gösterge: ``source`` = ``level`` ise 1, değilse 0; ``source`` eksikse eksik."""

    frame: str
    name: str
    source: str
    level: str
    comment: str


@dataclass(frozen=True)
class GroupMean:
    """``source`` değişkeninin ``by`` gruplarındaki ortalaması, grubun bütün satırlarına yazılır (ör. her bloğun
    saldırıdan önceki aylık ortalaması). ``condition`` (değişken, işleç, değer) verilirse ortalama yalnız koşulu sağlayan
    satırlarla alınır; işleç ``KEEP_OPERATORS`` içinden seçilir. Koşulu sağlayan dolu değeri olmayan grupta eksik.
    ``by`` ve koşul değişkeni eksiksiz olmalıdır (eksik anahtar dillerde farklı işlenir; uygulama hata verir)."""

    frame: str
    name: str
    source: str
    by: str
    comment: str
    condition: tuple[str, str, float] | None = None

    def __post_init__(self) -> None:
        if self.condition is not None and self.condition[1] not in KEEP_OPERATORS:
            raise ValueError(f"Desteklenmeyen işleç: {self.condition[1]}")


@dataclass(frozen=True)
class Derive:
    """Yeni bir değişken türetir."""

    frame: str
    name: str
    expr: Expr
    comment: str


@dataclass(frozen=True)
class Describe:
    """Betimsel istatistik tablosu."""

    frame: str
    variables: tuple[str, ...]
    result: str
    stats: tuple[str, ...] = ("mean", "sd", "median", "min", "max")


@dataclass(frozen=True)
class GroupSummary:
    """Bir değişkenin düzeylerine göre özet (koşullu ortalama tablosu)."""

    frame: str
    by: str
    columns: tuple[tuple[str, str, str], ...]
    result: str


@dataclass(frozen=True)
class OLS:
    """En küçük kareler tahmini.

    Tahmin örneklemi, modeldeki (ve varsa küme) değişkenlerinde eksik değeri olmayan
    gözlemlerdir; ``where`` verilirse yalnız o alt grup kullanılır.
    """

    name: str
    frame: str
    outcome: str
    regressors: tuple[str, ...]
    vcov: str = "classic"
    cluster: str | None = None
    categorical: tuple[str, ...] = ()
    where: tuple[str, float] | None = None
    constant: bool = True
    """``False``: sabit terimsiz regresyon (ör. artıkların artıklar üzerine regresyonu, Notlar §12.4)."""


@dataclass(frozen=True)
class ShowModel:
    """Tahmin edilmiş bir modelin tam yazılım çıktısını gösterir."""

    model: str


@dataclass(frozen=True)
class RegressionTable:
    """Birden çok modeli yan yana gösteren makale biçimli tablo."""

    models: tuple[str, ...]
    terms: tuple[str, ...]
    result: str
    title: str = ""
    """Uygulamadaki tablo başlığı; boşsa notlardaki başlık (log saatlik ücret için OLS regresyonları)."""


@dataclass(frozen=True)
class Scalar:
    """Katsayılardan türetilen tek sayı (ör. kesin yüzde etki, ilk aşama F, Wald oranı).

    ``percent`` yüzde olarak gösterilip gösterilmeyeceği, ``decimals`` gösterim basamağıdır.
    """

    name: str
    expr: Expr
    comment: str
    percent: bool = True
    decimals: int = 2


@dataclass(frozen=True)
class GroupMeanPlot:
    """Bir değişkenin, ikinci bir değişkenin düzeylerine göre ortalama eğrileri."""

    frame: str
    x: str
    y: str
    group: str
    group_labels: tuple[tuple[int, str], ...]
    x_label: str
    y_label: str
    title: str


@dataclass(frozen=True)
class ProjectionPlot:
    """Grup ortalamaları (nokta boyutu gözlem sayısıyla) ve bir OLS doğrusu."""

    frame: str
    x: str
    y: str
    model: str
    x_label: str
    y_label: str
    title: str
    relative_size: bool = False
    """Nokta boyutu en kalabalık gruba göre ölçeklenir (küçük örneklemde de noktalar görünür); notlarda mutlak ölçek."""


@dataclass(frozen=True)
class NewSample:
    """Simülasyon için boş bir örneklem ve tohumlanmış rastgele sayı üreteci.

    ``seed=None`` yalnız Monte Carlo döngüsü içinde kullanılır: üreteç döngüden önce
    bir kez tohumlanır, her tekrar yeni çekilişler yapar.
    """

    frame: str
    nobs: int
    seed: int | None


@dataclass(frozen=True)
class Draw:
    """Bir rastgele değişken çeker: normal(ortalama, std. sapma) veya uniform(alt, üst)."""

    frame: str
    name: str
    distribution: str
    first: float
    second: float
    comment: str


@dataclass(frozen=True)
class ClusterDraw:
    """Küme düzeyinde rastgele şok: aynı kümedeki bütün gözlemler aynı değeri alır."""

    frame: str
    name: str
    cluster: str
    groups: int
    first: float
    second: float
    comment: str


@dataclass(frozen=True)
class Predict:
    """Bir modelin uyum değerlerini veya artıklarını veri çerçevesine yazar."""

    model: str
    frame: str
    name: str
    kind: str


@dataclass(frozen=True)
class Summaries:
    """Birkaç değişkenin toplam veya ortalamasını tek tabloda gösterir."""

    frame: str
    rows: tuple[tuple[str, str, str], ...]
    result: str


@dataclass(frozen=True)
class MeanPoints:
    """Grafik katmanı: x'in her değerinde y'nin ortalaması; nokta boyutu gözlem sayısı."""

    y: str
    label: str


@dataclass(frozen=True)
class Scatter:
    """Grafik katmanı: bütün gözlemler."""

    y: str
    label: str


@dataclass(frozen=True)
class Curve:
    """Grafik katmanı: x'in bilinen bir fonksiyonu (ör. DGP'deki gerçek koşullu ortalama).

    ``color`` verilirse eğri rengi sıradan değil bu indeksle seçilir (aynı gruba ait gerçek ve
    tahmin eğrileri aynı renkte, biri kesikli çizilsin diye).
    """

    expr: Expr
    label: str
    dashed: bool = False
    color: int | None = None


@dataclass(frozen=True)
class ModelLine:
    """Grafik katmanı: tek regresörlü bir modelin doğrusu."""

    model: str
    label: str
    dashed: bool = False


@dataclass(frozen=True)
class ZeroLine:
    label: str


@dataclass(frozen=True)
class LocalCurve:
    """Grafik katmanı: y'nin x üzerindeki Gauss çekirdekli yerel doğrusal (``degree=1``) veya yerel
    sabit / Nadaraya–Watson (``degree=0``) tahmini, bant genişliği h."""

    y: str
    bandwidth: float
    label: str
    degree: int = 1
    dashed: bool = False
    color: int | None = None


@dataclass(frozen=True)
class BinMeans:
    """Grafik katmanı: x'in eşit genişlikli ``bins`` aralığında x ve y ortalamaları."""

    y: str
    bins: int
    label: str


@dataclass(frozen=True)
class RDDCurve:
    """Grafik katmanı: eşiğin iki yanında ayrı yerel doğrusal tahmin ve noktasal %95 güven bandı.

    Her değerlendirme noktası x₀ için yalnız o taraftaki gözlemlerle, birim varyanslı üçgen çekirdekle
    (Hansen ölçeği: pencere ±h√6) ağırlıklı EKK'nin sabit terimi; SH o yerel regresyonun HC1 sandviçi.
    Noktalar grafik aralığında, her tarafta ``points`` adet (eşik iki tarafta da dahil).
    """

    y: str
    cutoff: float
    bandwidth: float
    label: str
    points: int = 120


@dataclass(frozen=True)
class VLine:
    """Grafik katmanı: dikey çizgi (ör. eşik)."""

    x: float
    label: str


Layer = Union[MeanPoints, Scatter, Curve, ModelLine, ZeroLine, LocalCurve, BinMeans, RDDCurve, VLine]


@dataclass(frozen=True)
class Plot:
    """Katmanlardan oluşan grafik; üç dilde aynı katmanlarla çizilir.

    ``x_range`` verilirse yatay eksen ve eğri ızgaraları bu aralıktadır; verilmezse verinin aralığı. Yalnız
    bilinen fonksiyonlar (``Curve``, ``ZeroLine``, ``VLine``) çizilen grafikte ``frame`` boş bırakılabilir; o zaman
    ``x_range`` gerekir.
    """

    frame: str
    x: str
    layers: tuple[Layer, ...]
    x_label: str
    y_label: str
    title: str
    x_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class StandardErrorTable:
    """Aynı modeller için katsayı, klasik ve HC1 standart hata, R² yan yana."""

    models: tuple[str, ...]
    term: str
    result: str


@dataclass(frozen=True)
class BreuschPagan:
    """Breusch–Pagan (Koenker, n·R²) testi; regresörler modelin kendi regresörleridir."""

    model: str
    name: str


@dataclass(frozen=True)
class LinearCombination:
    """Katsayıların doğrusal birleşimi a'β ve standart hatası √(a'Va)."""

    name: str
    model: str
    weights: tuple[tuple[str, float], ...]
    comment: str


@dataclass(frozen=True)
class DeltaMethod:
    """Katsayıların doğrusal olmayan bir fonksiyonu g(β) ve delta yöntemiyle standart hatası."""

    name: str
    model: str
    expr: Expr
    comment: str


@dataclass(frozen=True)
class DropMissing:
    """Analiz örneklemi: listelenen değişkenlerde eksik değeri olan gözlemleri çıkarır."""

    frame: str
    variables: tuple[str, ...]
    comment: str


@dataclass(frozen=True)
class EffectTable:
    """Farklı modellerdeki tahminleri yan yana gösterir: katsayı, SH, p (normal yaklaşım), N.

    Her satır (etiket, model, terim) üçlüsüdür. SH modelin kendi kovaryans türüyledir.
    """

    rows: tuple[tuple[str, str, str], ...]
    result: str
    title: str = ""
    se_label: str = "SH"


@dataclass(frozen=True)
class IV:
    """İki aşamalı en küçük kareler (2SLS).

    ``vcov="HC1"`` heteroskedastisiteye dayanıklı sandviçtir ve n/(n−k) ile ölçeklenir. ``categorical``: dışsal
    kontrollerden kategorik olanlar (her düzey için bir kukla, ilk düzey referans; ``OLS.categorical`` gibi).
    """

    name: str
    frame: str
    outcome: str
    endogenous: tuple[str, ...]
    instruments: tuple[str, ...]
    exogenous: tuple[str, ...] = ()
    vcov: str = "HC1"
    categorical: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(name not in self.exogenous for name in self.categorical):
            raise ValueError("Kategorik kontroller dışsal değişkenler arasında olmalı.")


@dataclass(frozen=True)
class KeepIf:
    """Analiz örneklemi: bütün koşulları sağlayan gözlemler kalır.

    Her koşul (değişken, işleç, değer) üçlüsüdür; işleç ``KEEP_OPERATORS`` içinden seçilir.
    """

    frame: str
    conditions: tuple[tuple[str, str, float], ...]
    comment: str


@dataclass(frozen=True)
class Recode:
    """Bir değişkenin değerlerini yeni kategorilere toplar.

    ``mapping`` (eski değerler, yeni değer) çiftleridir; listede olmayan değerler ``other`` olur.
    """

    frame: str
    name: str
    source: str
    mapping: tuple[tuple[tuple[float, ...], int], ...]
    other: int
    comment: str


@dataclass(frozen=True)
class BinaryChoice:
    """İkili sonuç için Logit veya Probit, maksimum olabilirlik.

    ``vcov="robust"``: gözlenen Hessian ile sandviç H⁻¹(Σ sᵢsᵢ')H⁻¹, serbestlik düzeltmesi
    olmadan (statsmodels ``HC0``). Stata ``vce(robust)`` ayrıca n/(n−1) ile çarpar; binlerce
    gözlemde fark beşinci anlamlı basamaktadır. ``vcov="classic"``: ters gözlenen bilgi matrisi.
    """

    name: str
    frame: str
    outcome: str
    regressors: tuple[str, ...]
    link: str = "logit"
    categorical: tuple[str, ...] = ()
    vcov: str = "robust"


@dataclass(frozen=True)
class MarginalEffects:
    """Ortalama marjinal etkiler (AME) ve delta yöntemiyle standart hataları.

    Sürekli regresörde türevin (βⱼ·g(Xᵢ'β)), kategorik regresörde her düzey için referans
    düzeyine göre olasılık farkının örneklem ortalaması. ``discrete`` içindeki kategorik
    olmayan 0/1 değişkenlerde de türev yerine 1 − 0 farkı alınır. Doğrusal olasılık modelinde
    (OLS) etkiler katsayıların kendisidir. Sonuç, katsayıları etkiler olan bir "model" gibi
    saklanır; terim adları ``age`` veya ``race4=2`` biçimindedir.
    """

    model: str
    name: str
    terms: tuple[str, ...]
    discrete: tuple[str, ...] = ()


@dataclass(frozen=True)
class AverageProfile:
    """Bir regresör bütün gözlemlerde aynı değere eşitlendiğinde ortalama tahmin edilen olasılık.

    Diğer regresörler gözlenen değerlerinde kalır (``margins, at()`` mantığı). Sonuç tablosu:
    satırlar ``values``, tek sütun ``olasilik``.
    """

    model: str
    name: str
    variable: str
    values: tuple[float, ...]
    x_label: str
    y_label: str
    title: str


@dataclass(frozen=True)
class Tobit:
    """Soldan ``left`` noktasında sansürlü Tobit, maksimum olabilirlik.

    Standart hatalar ters gözlenen bilgi matrisinden (klasik); β bloğu σ'nın nasıl
    parametrelendiğinden etkilenmez.

    ``scale``: 1'den farklıysa R kodunda sonuç bu sayıya bölünerek tahmin edilir, β ve σ aynı
    sayıyla geri ölçeklenir. Tobit ölçekle eşdeğişkendir, sonuç değişmez; ``survival::survreg``
    çok büyük ölçekli sonuçta (ör. standart sapması 10⁶ düzeyinde) katsayıları tekil sayıp NA
    verebiliyor. Uygulamanın Olsen–Newton çözümü ölçekten etkilenmez.
    """

    name: str
    frame: str
    outcome: str
    regressors: tuple[str, ...]
    left: float = 0.0
    scale: float = 1.0


@dataclass(frozen=True)
class QuantileRegression:
    """Kantil regresyon; ``q=0.5`` medyan (en küçük mutlak sapma, LAD) regresyonudur.

    Katsayılar doğrusal programlama probleminin kesin çözümüdür (R ``rq``, Stata ``qreg`` ile
    aynı). ``vcov="nid"``: Hendricks–Koenker sandviç kovaryansı, koşullu yoğunluk her gözlemde
    τ ± h kantil doğrularının farkından (h: Hall–Sheather); R ``summary.rq(se = "nid")``.
    ``vcov="none"``: standart hata hesaplanmaz.
    """

    name: str
    frame: str
    outcome: str
    regressors: tuple[str, ...]
    q: float = 0.5
    vcov: str = "none"


@dataclass(frozen=True)
class QuantileDifference:
    """Aynı spesifikasyonun iki kantil tahmini arasındaki fark β̂(τ₂) − β̂(τ₁) için Wald testi.

    Standart hata iki tahminin ortak asimptotik kovaryansından: Cov = (min(τ₁,τ₂) − τ₁τ₂)
    H₁⁻¹ X'X H₂⁻¹, Hₖ = X'FₖX (Hendricks–Koenker yoğunlukları). Skalerler: ``name``, ``name_se``,
    ``name_z``, ``name_p``.
    """

    name: str
    low: str
    high: str
    term: str
    comment: str


@dataclass(frozen=True)
class CoefficientProfile:
    """Bir katsayının kantiller boyunca profili ve noktasal %95 güven bandı (± 1,96·SH).

    ``models`` (τ, model adı) çiftleridir; ``reference`` verilirse (ör. OLS) yatay çizgi olarak çizilir.
    """

    models: tuple[tuple[float, str], ...]
    term: str
    x_label: str
    y_label: str
    title: str
    reference: str | None = None
    reference_label: str = "OLS"


@dataclass(frozen=True)
class LocalLinear:
    """Gauss çekirdekli yerel doğrusal tahmin, seçilmiş noktalarda ve birkaç bant genişliğinde.

    ``bandwidths`` (sütun adı, h) çiftleridir. Sonuç tablosu: satırlar ``values``, sütunlar bu adlar.
    """

    name: str
    frame: str
    x: str
    y: str
    bandwidths: tuple[tuple[str, float], ...]
    values: tuple[float, ...]
    result: str


@dataclass(frozen=True)
class BandwidthCV:
    """Yerel doğrusal tahminde çapraz doğrulama ölçütü CV(h), bir h ızgarasında.

    ``grid`` (alt, üst, adım). Birini-dışarıda-bırak CV her zaman; ``cluster`` verilirse küme-silmeli
    CV de hesaplanır. Sonuç tablosu: satırlar h, sütunlar ``cv`` ve ``cv_kume``. Skalerler:
    ``name_h`` ve ``name_h_kume`` (ölçütü en küçük yapan h; eşitlikte küçük olan).
    """

    name: str
    frame: str
    x: str
    y: str
    grid: tuple[float, float, float]
    result: str
    x_label: str
    title: str
    cluster: str | None = None


@dataclass(frozen=True)
class LocalResidual:
    """Artıklaştırma: ``name`` = v − m̂(x), m̂ Gauss çekirdekli yerel doğrusal tahmin (örneklem noktalarında)."""

    frame: str
    name: str
    variable: str
    x: str
    bandwidth: float
    comment: str


@dataclass(frozen=True)
class ProfileCurves:
    """Modellerin doğrusal indeksi x'β, bir değişkenin ızgarasında; diğer regresörler örneklem ortalamasında.

    ``derived`` ızgara değişkeninden türetilen regresörlerdir (ör. spline terimleri). OLS ve
    LAD'de x'β tahmin edilen koşullu ortalama/medyan, Tobit'te gizli ortalamadır. Sonuç
    tablosu: satırlar ``values``, sütunlar model adları. ``plot_grid`` (alt, üst, nokta sayısı)
    grafikteki eğriler içindir.
    """

    models: tuple[tuple[str, str], ...]
    frame: str
    variable: str
    derived: tuple[tuple[str, Expr], ...]
    values: tuple[float, ...]
    result: str
    plot_grid: tuple[float, float, int]
    x_label: str
    y_label: str
    title: str


@dataclass(frozen=True)
class TobitTargets:
    """Tobit'in üç hedefi, ``curves`` tablosundaki gizli ortalama x'β üzerinden.

    P(Y>0|x) = Φ(z), m(x) = Φ(z)x'β + σφ(z), m#(x) = x'β + σφ(z)/Φ(z), z = x'β/σ. Sonuç
    tablosunun sütunları: ``gizli``, ``p_poz``, ``gozlenen``, ``poz_ort`` (Stata skaler adları
    32 karakteri aşmasın diye kısa).
    """

    model: str
    curves: str
    result: str


@dataclass(frozen=True)
class TobitFitCheck:
    """Model kontrolü: Tobit'in ima ettiği P(Y>0) ve E[Y] örneklem ortalamaları, verideki karşılıklarıyla.

    Sonuç tablosu: satırlar ``p_poz`` ve ``ortalama``, sütunlar ``model`` ve ``veri``.
    """

    model: str
    result: str


@dataclass(frozen=True)
class MonteCarlo:
    """Bir işlem bloğunu ``reps`` kez yeni çekilişlerle tekrarlar ve seçilen sayıları toplar.

    ``collect`` her tekrarda hesaplanan (sütun adı, ifade) çiftleridir; ifadeler
    katsayı (``Coef``) ve standart hata (``StdErr``) içerebilir. ``coverage`` her
    (tahmin sütunu, SH sütunu, gerçek değer) üçlüsü için %95 güven aralığının
    (tahmin ± 1,96·SH) gerçek değeri kapsama oranını raporlar.
    """

    frame: str
    reps: int
    seed: int
    body: tuple["Operation", ...]
    collect: tuple[tuple[str, Expr], ...]
    result: str
    comment: str
    coverage: tuple[tuple[str, str, float], ...] = ()


@dataclass(frozen=True)
class Histogram:
    """Bir sonuç tablosundaki sütunların dağılımı; [lower, upper] dışı değerler çizilmez."""

    table: str
    columns: tuple[tuple[str, str], ...]
    lower: float
    upper: float
    references: tuple[tuple[float, str], ...]
    x_label: str
    title: str
    bins: int = 40


# --- Regresyon süreksizliği ve bootstrap (Konu 9–10) --------------------------

@dataclass(frozen=True)
class RDD:
    """Keskin RDD: eşikte yerel doğrusal sıçrama tahmini.

    Pencere içindeki gözlemlerde Y'nin D = 1{X ≥ c}, R = X − c ve D·R (``DR``) üzerine çekirdek ağırlıklı
    EKK'si; sıçrama D katsayısıdır. Çekirdek ``triangular`` veya ``rectangular``. ``scale="hansen"``:
    h çekirdeğin standart sapmasıdır (Hansen'in birim varyanslı çekirdekleri; üçgende pencere ±h√6,
    dikdörtgende ±h√3). ``scale="window"``: h pencerenin yarı genişliğidir. Standart hata ağırlıklı
    regresyonun HC1 sandviçidir. Terimler: sabit, ``D``, ``R``, ``DR``.
    """

    name: str
    frame: str
    x: str
    y: str
    cutoff: float
    bandwidth: float
    kernel: str = "triangular"
    scale: str = "hansen"


@dataclass(frozen=True)
class RDDTable:
    """Bant genişliği duyarlılık tablosu ve grafiği.

    ``rows`` (h, RDD modeli) çiftleridir. Sütunlar: ``n`` (pozitif ağırlık alan gözlem), ``tahmin`` (sıçrama),
    ``sh`` (HC1), ``alt`` ve ``ust`` (normal yaklaşımla %95 güven aralığı, tahmin ± 1,96·SH).
    """

    rows: tuple[tuple[float, str], ...]
    result: str
    x_label: str
    y_label: str
    title: str


BOOT = "bootstrap_tekrari"
"""``Bootstrap.collect`` ifadelerinde o tekrarda yeniden tahmin edilen modelin adı."""


@dataclass(frozen=True)
class Bootstrap:
    """Bir OLS modelini ``reps`` kez yeniden örneklenmiş veride yeniden tahmin eder (Notlar §10.5, §10.10, §10.14).

    ``method``: ``pairs`` gözlem satırlarını yerine koyarak çeker; ``wild`` regresörleri sabit tutar ve
    Y* = Xβ̂ + ê·ξ üretir (ξ Rademacher, ±1); ``cluster`` ``cluster`` değişkeninin kümelerini yerine koyarak
    çeker. ``collect`` her tekrarda hesaplanan (sütun adı, ifade) çiftleridir: ``E.coef(BOOT, terim)`` ve
    ``E.se(BOOT, terim)`` o tekrarın katsayısı ve HC1 standart hatasıdır; diğer model adları orijinal tahmini
    gösterir. ``seed=None``: ``frame`` verisinin rastgele sayı üreteci kaldığı yerden devam eder (simülasyonda
    veri çekilişlerinden sonra aynı üreteç). Skalerler: her sütun için ``{result}_{sütun}_se`` (tekrarlar
    arası standart sapma, ddof = 1), ``_lo`` ve ``_hi`` (yüzde 2,5 ve 97,5 yüzdelikleri).
    """

    model: str
    frame: str
    reps: int
    seed: int | None
    collect: tuple[tuple[str, Expr], ...]
    result: str
    comment: str
    method: str = "pairs"
    cluster: str | None = None


@dataclass(frozen=True)
class ScalarTable:
    """Skaler ifadelerden tek sütunlu (``deger``) özet tablo; ifadeler katsayı, SH ve skaler (``E.ref``) içerebilir."""

    rows: tuple[tuple[str, Expr], ...]
    result: str
    decimals: int = 3


# --- Model seçimi, düzenlileştirme ve DML (Konu 11–12) ------------------------------------------

PENALTIES = ("ols", "ridge", "lasso", "enet")
SELECTION_RULES = ("min", "1se")
LEARNERS = ("lasso", "ols")


@dataclass(frozen=True)
class RowNumber:
    """Gözlem sıra numarası 1, 2, …, n (verideki satır sırası); açık kat ve bölme kuralları için."""

    frame: str
    name: str
    comment: str


@dataclass(frozen=True)
class GroupRank:
    """Küme numaralarının küçükten büyüğe sıra numarası 1, 2, …, G (ör. okul numarasından okul sırası)."""

    frame: str
    name: str
    source: str
    comment: str


@dataclass(frozen=True)
class Dictionary:
    """Aday terim sözlüğü: ``base`` değişkenleri, ``powers`` içindeki her değişkenin 2, …, ``degree`` kuvvetleri
    (``{ad}_{k}``) ve ``interactions`` ise ``base`` değişkenlerinin bütün ikili çarpımları (``{a}_x_{b}``)."""

    frame: str
    base: tuple[str, ...]
    powers: tuple[str, ...]
    degree: int
    interactions: bool
    comment: str


def dictionary_terms(op: Dictionary) -> tuple[str, ...]:
    """Sözlüğün terimleri, üç dilde aynı sırayla: taban, kuvvetler, ikili etkileşimler."""

    terms = list(op.base)
    for name in op.powers:
        terms += [f"{name}_{power}" for power in range(2, op.degree + 1)]
    if op.interactions:
        terms += [f"{a}_x_{b}" for index, a in enumerate(op.base) for b in op.base[index + 1:]]
    return tuple(terms)


@dataclass(frozen=True)
class DrawColumns:
    """``count`` adet standart normal sütun ``{prefix}1``, …: x₁ = z₁, xⱼ = ρ·xⱼ₋₁ + √(1 − ρ²)·zⱼ.

    Corr(xⱼ, xₖ) = ρ^|j−k| ve her sütunun varyansı 1; ρ = 0 bağımsız sütunlardır. Üreteçten her sütun için sırayla
    n normal çekiliş yapılır (Python kodu uygulamayla aynı çekilişleri yapar).
    """

    frame: str
    prefix: str
    count: int
    rho: float
    comment: str


GridSpec = tuple[float, float, int]
"""Ceza ızgarası (üst üs, alt üs, nokta sayısı): 10^üst'ten 10^alt'a logaritmik eşit aralıklı, büyükten küçüğe."""


@dataclass(frozen=True)
class Penalized:
    """Doğrusal tahmin: cezasız (``ols``), Ridge, Lasso veya Elastic Net (Notlar §11.8–§11.11).

    Ceza ölçekleri: Ridge ``(Y − Xβ)'(Y − Xβ) + λβ'β`` (Hansen'in SSE ölçeği); Lasso ve Elastic Net yazılım ölçeği
    ``(1/(2n))‖Y − Xβ‖² + λ[r‖β‖₁ + (1 − r)/2·‖β‖²]``, r = L1 ağırlığı (Lasso r = 1). Sabit terim cezasızdır.
    Sürekli değişkenler (``numeric``) ``standardize`` ise eğitim verisinin ortalaması ve (n'e bölünen) standart
    sapmasıyla ölçeklenir; ``categorical`` değişkenlerin eğitim verisinde görülen düzeyleri göstergedir, ilk düzey
    referanstır. ``sample`` eğitim satırlarında 1, test satırlarında 0 olan değişkendir (``None``: bütün örneklem).
    Ceza ``folds`` katlarıyla (eğitim satırlarında 1, …, K) CV'de seçilir; ölçekleme ve göstergeler her katta yeniden
    öğrenilir. ``rule``: ``min`` en küçük CV, ``1se`` en küçüğün bir SH'si içindeki en büyük ceza. ``grid``: ``GridSpec``.
    ``path``: bütün eğitim örnekleminde ızgara boyunca katsayı yolu saklanır.

    Skalerler (``penalized_key``): ``test_mse``, ``egitim_mse``, ``sifirdan`` (sıfırdan farklı katsayı sayısı), ``norm``
    (‖β̂‖₂, sabit hariç), ``lambda`` (seçilen ceza), ``l1`` (seçilen L1 ağırlığı).
    """

    name: str
    frame: str
    outcome: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...] = ()
    penalty: str = "ols"
    grid: GridSpec | None = None
    sample: str | None = None
    folds: str | None = None
    rule: str = "min"
    l1_ratios: tuple[float, ...] = (1.0,)
    standardize: bool = True
    path: bool = False


PENALIZED_SCALARS = ("test_mse", "egitim_mse", "sifirdan", "norm", "lambda", "l1")


def penalized_key(model: str, quantity: str) -> str:
    """Düzenlileştirilmiş model skalerinin adı; üç dilde aynı ad (Stata'da en çok 32 karakter)."""

    if quantity not in PENALIZED_SCALARS:
        raise ValueError(f"Desteklenmeyen nicelik: {quantity}")
    return f"{model}_{quantity}"


@dataclass(frozen=True)
class PostSelection:
    """Post-Lasso (Notlar §11.12): ``source`` modelinin sıfırdan farklı katsayılı terimleri üzerinde cezasız EKK; örneklem,
    ölçekleme ve göstergeler kaynak modelinkilerle aynıdır. Skalerler ``Penalized`` ile aynı adlarla."""

    name: str
    source: str


@dataclass(frozen=True)
class ModelMetrics:
    """Modellerin dış-örneklem karşılaştırması. ``rows`` (model, etiket) çiftleri; tablo satırları model adlarıdır.
    Sütunlar: ``test_mse``, ``sifirdan``, ``norm``, ``lambda`` (cezasız modellerde boş)."""

    rows: tuple[tuple[str, str], ...]
    result: str


@dataclass(frozen=True)
class DotPlot:
    """Yatay nokta grafiği: ``table`` tablosunun ``column`` sütunu, satır etiketleriyle. Farklar küçükken çubuk
    grafiğin sıfırdan başlamayan ekseni yanıltır; nokta grafiği bu sorunu taşımaz."""

    table: str
    column: str
    labels: tuple[tuple[str, str], ...]
    x_label: str
    title: str
    decimals: int = 4


@dataclass(frozen=True)
class CVCurve:
    """CV ölçütü ve ±1 SH bandı ceza ızgarası boyunca (logaritmik eksen); dikey çizgi seçilen ceza. Elastic Net'te
    seçilen L1 ağırlığının eğrisi çizilir."""

    model: str
    x_label: str
    title: str


@dataclass(frozen=True)
class CoefPath:
    """Katsayı yolu: bütün eğitim örnekleminde ızgara boyunca katsayılar (``Penalized(path=True)``). ``highlight``
    terimleri renkli, diğerleri gri; dikey çizgi CV ile seçilen ceza."""

    model: str
    highlight: tuple[str, ...]
    highlight_label: str
    other_label: str
    x_label: str
    title: str


@dataclass(frozen=True)
class CrossFitDML:
    """Kısmen doğrusal model Y = θD + g(X) + ε için DML2 (Notlar §12.4, §12.7).

    Her dış kat k için m_Y(X) = E[Y|X] ve m_D(X) = E[D|X] k dışındaki gözlemlerde öğrenilir, k'deki gözlemlerde artık
    alınır: Û = Y − m̂_Y, V̂ = D − m̂_D. θ̂ = ΣV̂Û/ΣV̂² (sabitsiz artık regresyonu). ``outer`` dış kat değişkeni; ``None``
    ise çapraz uyarlama yoktur (artıklaştırma: yardımcı modeller bütün örneklemde, §12.4). ``learner``: ``lasso``
    (ceza ``inner`` katlarıyla eğitim satırlarında CV'de seçilir; ``grid``, ``rule``) veya ``ols``. Özellikler her
    eğitim katında ölçeklenir. SH: HC1 (n/(n−1)); ``cluster`` verilirse skorlar küme içinde toplanır, G/(G−1).
    ``residuals`` (Û, V̂) sütun adlarıdır. Terim ``theta``. Tablo ``{name}_katlar``: kat, lambda_y, sifirdan_y, lambda_d,
    sifirdan_d. Skalerler ``{name}_sifirdan_y_min``, ``_y_max``, ``_d_min``, ``_d_max``.
    """

    name: str
    frame: str
    outcome: str
    treatment: str
    features: tuple[str, ...]
    outer: str | None
    inner: str | None = None
    grid: GridSpec | None = None
    rule: str = "min"
    cluster: str | None = None
    learner: str = "lasso"
    residuals: tuple[str, str] | None = None


THETA = "theta"
"""DML modellerinde hedef katsayının terim adı."""


def dml_fold_key(model: str, part: str, statistic: str) -> str:
    """DML katlarındaki sıfırdan farklı katsayı sayısının özeti (``y``/``d``, ``min``/``max``)."""

    return f"{model}_sifirdan_{part}_{statistic}"


@dataclass(frozen=True)
class DMLSplits:
    """Bölme duyarlılığı (Notlar §12.14): ``dml`` modelinin hesabı, dış katlar ⌊K·{r·c}⌋ + 1 kuralıyla (r küme sıra
    numarası, {·} kesirli kısım) her çarpan c için tekrarlanır. ``rules`` (anahtar, etiket, c) üçlüleridir. Birleştirme
    (Chernozhukov vd., 2018): θ̂_med = medyan θ̂_s, σ̂²_med = medyan{σ̂²_s + (θ̂_s − θ̂_med)²}. Sonuç tablosu satırları
    kural anahtarları, sütunları ``theta`` ve ``sh``; ``name`` modeli θ̂_med ve SH_med'i taşır (terim ``theta``).
    Skalerler ``{name}_min`` ve ``{name}_max``."""

    name: str
    dml: str
    rank: str
    rules: tuple[tuple[str, str, float], ...]
    folds: int
    result: str


@dataclass(frozen=True)
class DoubleSelection:
    """Yalnız-sonuç Post-Lasso (Notlar §12.2) ve double selection (§12.3).

    S_Y: Y'nin X üzerindeki Lasso'suyla seçilen kontroller; D cezalandırılmasın diye Y ve her X önce [1, D] üzerinde
    artıklaştırılır. S_D: D'nin X üzerindeki Lasso'su. Cezalar ``folds`` katlarıyla CV'de, ``rule`` kuralıyla seçilir.
    Modeller (HC1): ``{name}_sonuc`` Y ~ D + S_Y, ``{name}`` Y ~ D + (S_Y ∪ S_D). Skalerler: ``{name}_ny``, ``{name}_nd``,
    ``{name}_n`` (seçilen kontrol sayıları) ve ``track`` değişkenlerinden seçilenler ``{name}_ny_iz``, ``_nd_iz``, ``_n_iz``.
    """

    name: str
    frame: str
    outcome: str
    treatment: str
    controls: tuple[str, ...]
    grid: GridSpec
    folds: str
    rule: str = "1se"
    track: tuple[str, ...] = ()


@dataclass(frozen=True)
class EstimatePlot:
    """Tahminler ve %95 güven aralıkları (tahmin ± 1,96·SH), yatay. ``rows`` (etiket, model, terim); ``truth`` gerçek
    değer (dikey kesikli çizgi); ``splits`` bir ``DMLSplits`` sonuç tablosu verilirse ``splits_row`` satırının yanında
    bölme tahminleri gri noktalardır. ``result`` tablosu: satırlar etiketler; sütunlar tahmin, sh, alt, ust."""

    rows: tuple[tuple[str, str, str], ...]
    result: str
    x_label: str
    title: str
    truth: float | None = None
    truth_label: str = ""
    splits: str | None = None
    splits_row: int = 0


@dataclass(frozen=True)
class ComplexityCurve:
    """Polinom derecesi boyunca model seçim ölçütleri (Notlar §11.2–§11.5).

    Her d = 1, …, ``max_degree`` için Y'nin [1, X, …, X^d] üzerine OLS'si eğitim satırlarında (``sample`` = 1).
    Sütunlar: ``egitim_mse`` (σ̂² = SSR/n), ``aic`` = n + n·log(2πσ̂²) + 2K, ``bic`` = n + n·log(2πσ̂²) + K·log n
    (K = d + 2: d + 1 katsayı ve σ²), ``loocv`` (kaldıraçla kesin: ortalama [ê/(1 − h)]²), ``test_mse`` (``sample`` = 0).
    Skalerler: her ölçütü en küçük yapan derece ``{name}_d_{sütun}``.
    """

    name: str
    frame: str
    x: str
    y: str
    max_degree: int
    sample: str
    result: str


COMPLEXITY_COLUMNS = ("egitim_mse", "loocv", "test_mse", "aic", "bic")


@dataclass(frozen=True)
class LinePlot:
    """Bir sonuç tablosunun sütunları, tablo satır değerleri (ör. polinom derecesi) boyunca çizgi ve nokta.
    ``relative``: her sütundan kendi en küçük değeri çıkarılır (ör. AIC ve BIC farkları)."""

    table: str
    columns: tuple[tuple[str, str], ...]
    x_label: str
    y_label: str
    title: str
    relative: bool = False


Operation = Union[
    ReadFile,
    CompleteCases,
    Indicator,
    GroupMean,
    RowNumber,
    GroupRank,
    Dictionary,
    DrawColumns,
    Penalized,
    PostSelection,
    ModelMetrics,
    DotPlot,
    CVCurve,
    CoefPath,
    CrossFitDML,
    DMLSplits,
    DoubleSelection,
    EstimatePlot,
    ComplexityCurve,
    LinePlot,
    RDD,
    RDDTable,
    Bootstrap,
    ScalarTable,
    QuantileDifference,
    CoefficientProfile,
    LocalLinear,
    BandwidthCV,
    LocalResidual,
    KeepIf,
    Recode,
    BinaryChoice,
    MarginalEffects,
    AverageProfile,
    Tobit,
    QuantileRegression,
    ProfileCurves,
    TobitTargets,
    TobitFitCheck,
    DropMissing,
    EffectTable,
    IV,
    MonteCarlo,
    Histogram,
    ClusterDraw,
    StandardErrorTable,
    BreuschPagan,
    LinearCombination,
    DeltaMethod,
    NewSample,
    Draw,
    Predict,
    Summaries,
    Plot,
    LoadHansen,
    Derive,
    Describe,
    GroupSummary,
    OLS,
    ShowModel,
    RegressionTable,
    Scalar,
    GroupMeanPlot,
    ProjectionPlot,
]


# --- Notlarla karşılaştırma -------------------------------------------------

@dataclass(frozen=True)
class StatTarget:
    """Bir değişkenin (isteğe bağlı olarak bir alt gruptaki) istatistiği."""

    frame: str
    variable: str
    stat: str
    where: tuple[str, float] | None = None


@dataclass(frozen=True)
class CoefTarget:
    """Katsayı niceliği: ``coef``, ``se`` (modelin kovaryans türüyle), ``se_hc1`` veya
    ``p`` (normal yaklaşımla iki yönlü p-değeri, 2Φ(−|b/SH|))."""

    model: str
    term: str
    quantity: str = "coef"


@dataclass(frozen=True)
class ModelTarget:
    model: str
    quantity: str


@dataclass(frozen=True)
class ScalarTarget:
    name: str


@dataclass(frozen=True)
class TableTarget:
    """Bir sonuç tablosunun hücresi: satır (ör. ızgara değeri) ve sütun adı."""

    table: str
    row: float | str
    column: str


Target = Union[StatTarget, CoefTarget, ModelTarget, ScalarTarget, TableTarget]


@dataclass(frozen=True)
class Check:
    """Notlarda basılı bir sayı ve onu üreten hesap.

    ``mc_tolerance`` yalnız rastgele çekilişe dayanan değerlerde verilir: uygulama ve üretilen Python kodu
    notlardaki çekilişin aynısını yapar ve ondalık toleransıyla denetlenir; R ve Stata farklı rastgele sayı
    üreteci kullandığı için bu Monte Carlo toleransıyla denetlenir.
    """

    label: str
    target: Target
    expected: float
    decimals: int = 4
    mc_tolerance: float | None = None

    @property
    def tolerance(self) -> float:
        if self.decimals <= 0:
            return 0.5
        return 0.5 * 10 ** (-self.decimals) + 1e-12


# --- Adım ve laboratuvar -----------------------------------------------------

@dataclass(frozen=True)
class NoteRef:
    section: str
    step: int
    objects: tuple[str, ...] = ()

    def label(self) -> str:
        parts = [f"Notlar §{self.section}"]
        if self.step:
            parts.append(f"Adım {self.step}")
        parts.extend(self.objects)
        return " · ".join(parts)


@dataclass(frozen=True)
class LabStep:
    number: int
    title: str
    note: NoteRef
    explanation: str
    operations: tuple[Operation, ...] = ()
    checks: tuple[Check, ...] = ()
    reproducibility: ReproClass = ReproClass.EXACT
    takeaway: str = ""
    code_note: str = ""
    note_for: Callable[[object], str] | None = field(default=None, compare=False, repr=False)
    """Ek veri kaynaklarında adımın yorumu sonuçlardan yazılır: ``note_for(state)`` (``LabState``). Hesap yokken ya da
    yorum yazılamazsa ``takeaway`` gösterilir."""

    @property
    def key(self) -> str:
        return f"adim{self.number}"


@dataclass(frozen=True)
class LabSpec:
    topic_key: str
    title: str
    dataset: str
    note_section: str
    steps: tuple[LabStep, ...]
    labels: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    consistency_notes: tuple[str, ...] = field(default_factory=tuple)
    kind: str = "uygulama"
    source: str = "notlar"
    """Veri kaynağı (``SOURCES``). Notlar dışındaki kaynaklarda kontrollerin beklenen değerleri uygulamanın hesabıdır
    (alternatif örnekte Hansen'in tam örneklemiyle, kendi verinde yüklenen dosyayla); üretilen kod bu değerleri
    yeniden üretmelidir."""

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"{self.topic_key}: tanımsız veri kaynağı {self.source!r}.")

    def label(self, name: str) -> str:
        """Değişken veya terim için öğrenciye gösterilecek Türkçe ad."""

        return dict(self.labels).get(name, name)

    def step(self, number: int) -> LabStep:
        for item in self.steps:
            if item.number == number:
                return item
        raise ValueError(f"Adım bulunamadı: {number}")

    def operations_through(self, number: int) -> tuple[Operation, ...]:
        """Bir adımı tek başına çalıştırmak için gereken bütün önceki işlemler."""

        collected: list[Operation] = []
        for item in self.steps:
            if item.number > number:
                break
            collected.extend(item.operations)
        return tuple(collected)
