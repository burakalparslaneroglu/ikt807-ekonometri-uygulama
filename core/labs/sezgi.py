"""Sezgi sekmesindeki kontrollü simülasyon deneylerinin tanım şeması.

Bir deney, kaydırıcı değerlerinden işlem listesi üreten tek bir tanımdır. Aynı
işlem listesi uygulamanın hesabını ve Python, R, Stata kodunu besler. Veri
simülasyonla üretildiği için gerçek koşullu ortalama gibi normalde bilinmeyen
nesneler bilinir; deneylerin amacı tahmini bu bilinen gerçekle karşılaştırmaktır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from core.labs.runner import LabState
from core.labs.spec import LabSpec, LabStep, NoteRef, Operation, ReproClass

Parameters = Mapping[str, float]


@dataclass(frozen=True)
class SimParameter:
    key: str
    label: str
    minimum: float
    maximum: float
    default: float
    step: float
    help: str
    integer: bool = False
    decimals: int = 2


@dataclass(frozen=True)
class SimMetric:
    label: str
    value: str
    help: str


@dataclass(frozen=True)
class SimExperiment:
    topic_key: str
    number: int
    title: str
    question: str
    note: NoteRef
    parameters: tuple[SimParameter, ...]
    dgp: Callable[[Parameters], tuple[str, ...]]
    dgp_note: str
    look_at: tuple[str, ...]
    build: Callable[[Parameters], tuple[Operation, ...]]
    metrics: Callable[[LabState, Parameters], tuple[SimMetric, ...]]
    takeaway: Callable[[LabState, Parameters], str]
    tables: tuple[tuple[str, str], ...] = ()
    labels: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def key(self) -> str:
        return f"{self.topic_key}_sezgi{self.number}"

    def defaults(self) -> dict[str, float]:
        return {item.key: item.default for item in self.parameters}

    def label(self, name: str) -> str:
        return dict(self.labels).get(name, name)

    def spec(self, parameters: Parameters) -> LabSpec:
        """Kod üreticilerinin beklediği biçimde tek adımlık bir tanım."""

        step = LabStep(
            number=self.number,
            title=self.title,
            note=self.note,
            explanation=self.question,
            operations=self.build(parameters),
            reproducibility=ReproClass.DISTRIBUTIONAL,
        )
        return LabSpec(
            topic_key=self.topic_key,
            title=self.title,
            dataset="simülasyon",
            note_section=self.note.section,
            steps=(step,),
            labels=self.labels,
            kind="sezgi",
        )


def number(value: float, decimals: int = 3) -> str:
    """Türkçe ondalık virgülüyle sayı (LaTeX içinde kullanılmak üzere)."""

    return f"{value:.{decimals}f}".replace(".", "{,}")


def plain(value: float, decimals: int = 4) -> str:
    """Türkçe ondalık virgülüyle düz metin sayı; eksi işareti tipografik."""

    return f"{value:.{decimals}f}".replace(".", ",").replace("-", "−")
