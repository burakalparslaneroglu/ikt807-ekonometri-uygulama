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


# --- İşlemler ---------------------------------------------------------------

@dataclass(frozen=True)
class LoadHansen:
    """Hansen'in veri arşivinden bir veri setini yükler.

    Hansen'in ``.txt`` dosyaları başlıksızdır; ``columns`` değişken adlarını veri
    setinin açıklama belgesindeki sırayla verir.
    """

    dataset: str
    member: str
    frame: str
    columns: tuple[str, ...]


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
    """En küçük kareler tahmini."""

    name: str
    frame: str
    outcome: str
    regressors: tuple[str, ...]
    vcov: str = "classic"
    cluster: str | None = None
    categorical: tuple[str, ...] = ()


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
    """Katsayılardan türetilen tek sayı (ör. kesin yüzde etki)."""

    name: str
    expr: Expr
    comment: str


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
    """Simülasyon için boş bir örneklem ve tohumlanmış rastgele sayı üreteci."""

    frame: str
    nobs: int
    seed: int


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
    """Grafik katmanı: x'in bilinen bir fonksiyonu (ör. DGP'deki gerçek koşullu ortalama)."""

    expr: Expr
    label: str


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


Operation = Union[
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


Target = Union[StatTarget, CoefTarget, ModelTarget, ScalarTarget]


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
