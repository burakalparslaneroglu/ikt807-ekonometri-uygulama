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
from typing import Union

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


Layer = Union[MeanPoints, Scatter, Curve, ModelLine, ZeroLine]


@dataclass(frozen=True)
class Plot:
    """Katmanlardan oluşan grafik; üç dilde aynı katmanlarla çizilir."""

    frame: str
    x: str
    layers: tuple[Layer, ...]
    x_label: str
    y_label: str
    title: str


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

    ``vcov="HC1"`` heteroskedastisiteye dayanıklı sandviçtir ve n/(n−k) ile ölçeklenir.
    """

    name: str
    frame: str
    outcome: str
    endogenous: tuple[str, ...]
    instruments: tuple[str, ...]
    exogenous: tuple[str, ...] = ()
    vcov: str = "HC1"


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
    """

    name: str
    frame: str
    outcome: str
    regressors: tuple[str, ...]
    left: float = 0.0


@dataclass(frozen=True)
class QuantileRegression:
    """Kantil regresyon; ``q=0.5`` medyan (en küçük mutlak sapma, LAD) regresyonudur."""

    name: str
    frame: str
    outcome: str
    regressors: tuple[str, ...]
    q: float = 0.5


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


Operation = Union[
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
    """Notlarda basılı bir sayı ve onu üreten hesap."""

    label: str
    target: Target
    expected: float
    decimals: int = 4

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
