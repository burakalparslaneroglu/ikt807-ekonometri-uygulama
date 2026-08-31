"""Uygulama genelinde kullanılan değişmez metadata ve sonuç sözleşmeleri."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class VariableMetadata:
    """Öğrenciye gösterilecek değişken tanımı."""

    name: str
    label: str
    description: str
    unit: str


@dataclass(frozen=True)
class DatasetMetadata:
    """Bir öğretim veri setinin kaynak ve kullanım sözleşmesi."""

    dataset_id: str
    title: str
    source: str
    observation_unit: str
    sample_definition: str
    variables: tuple[VariableMetadata, ...]
    expected_columns: tuple[str, ...]
    allowed_topics: tuple[str, ...]
    cluster_variable: str | None = None
    resampling_unit: str = "gözlem"
    redistribution_status: str = "license_review_required"

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id boş olamaz.")
        if not self.expected_columns:
            raise ValueError(f"{self.dataset_id} için beklenen sütunlar tanımlanmalıdır.")
        if set(self.expected_columns) != {item.name for item in self.variables}:
            raise ValueError(
                f"{self.dataset_id} değişken metadata'sı ile beklenen sütunlar eşleşmiyor."
            )
        if not self.allowed_topics:
            raise ValueError(f"{self.dataset_id} en az bir konuya bağlanmalıdır.")


@dataclass(frozen=True)
class TopicMetadata:
    """Bir konu sayfasının tek kaynak pedagojik metadata'sı."""

    key: str
    number: int
    title: str
    short_title: str
    guiding_question: str
    estimand: str
    identification_focus: str
    application_focus: str
    dataset_ids: tuple[str, ...]
    methods: tuple[str, ...]
    questions: tuple[tuple[str, str], ...]

    @property
    def label(self) -> str:
        return f"Konu {self.number:02d} - {self.short_title}"


@dataclass(frozen=True)
class EstimandMetadata:
    """Tahmin hedefinin hem teknik hem yalın dilde tanımı."""

    name: str
    plain_language: str
    target_population: str
    causal_interpretation_condition: str | None = None


@dataclass(frozen=True)
class InferenceSpec:
    """Çıkarım tercihlerinin görünür sözleşmesi."""

    covariance_type: str
    confidence_level: float = 0.95
    finite_sample_correction: bool = False
    cluster_variable: str | None = None

    def __post_init__(self) -> None:
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level 0 ile 1 arasında olmalıdır.")
        if self.covariance_type == "cluster" and not self.cluster_variable:
            raise ValueError("Küme-dayanıklı çıkarımda cluster_variable zorunludur.")


@dataclass(frozen=True)
class TuningSpec:
    """Yeniden üretilebilirlik için yöntem ayarları."""

    seed: int
    bandwidth: float | None = None
    kernel: str | None = None
    penalty: float | None = None
    folds: int | None = None
    split_unit: str | None = None
    optimization_tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed negatif olamaz.")
        if self.bandwidth is not None and self.bandwidth <= 0:
            raise ValueError("bandwidth pozitif olmalıdır.")
        if self.penalty is not None and self.penalty < 0:
            raise ValueError("penalty negatif olamaz.")
        if self.folds is not None and self.folds < 2:
            raise ValueError("folds en az 2 olmalıdır.")
        if self.optimization_tolerance is not None and self.optimization_tolerance <= 0:
            raise ValueError("optimization_tolerance pozitif olmalıdır.")


@dataclass(frozen=True)
class Estimate:
    """Tek bir raporlanabilir tahmin."""

    name: str
    label: str
    value: float
    standard_error: float | None = None
    confidence_interval: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.value):
            raise ValueError("Tahmin sonlu olmalıdır.")
        if self.standard_error is not None and (
            not isfinite(self.standard_error) or self.standard_error < 0
        ):
            raise ValueError("Standart hata sonlu ve negatif olmayan bir sayı olmalıdır.")
        if self.confidence_interval is not None:
            low, high = self.confidence_interval
            if not (isfinite(low) and isfinite(high) and low <= high):
                raise ValueError("Güven aralığı sonlu ve sıralı olmalıdır.")


@dataclass(frozen=True)
class ModelResult:
    """Yöntem dallarının kullanacağı ortak sonuç taşıyıcısı."""

    model_name: str
    dataset_id: str
    sample_definition: str
    estimand: EstimandMetadata
    estimates: tuple[Estimate, ...]
    inference: InferenceSpec
    tuning: TuningSpec
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.estimates:
            raise ValueError("ModelResult en az bir tahmin içermelidir.")
