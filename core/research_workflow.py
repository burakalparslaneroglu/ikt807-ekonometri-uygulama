"""Bütünleşik ekonometrik araştırma akışının denetlenebilir aşamaları."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowStage:
    key: str
    label: str
    deliverable: str


@dataclass(frozen=True)
class WorkflowAuditItem:
    stage: WorkflowStage
    complete: bool


RESEARCH_WORKFLOW: tuple[WorkflowStage, ...] = (
    WorkflowStage("estimand", "Tahmin hedefi", "Hedef parametre ve anakütle"),
    WorkflowStage("identification", "Tanımlama", "Varsayımlar ve tehditler"),
    WorkflowStage("data", "Veri ve örneklem", "Gözlem birimi, filtre ve veri izi"),
    WorkflowStage("nuisance", "Yardımcı modeller", "Öğrenici, özellikler ve ayar alanı"),
    WorkflowStage("cross_fit", "Çapraz uyarlama", "Kat, bölünme birimi ve rastgelelik tohumu"),
    WorkflowStage("estimate", "Hedef tahmin", "Nokta tahmini ve belirsizlik"),
    WorkflowStage("sensitivity", "Duyarlılık", "Öğrenici, kat ve bölünme tohumu karşılaştırması"),
    WorkflowStage("reproducibility", "Yeniden üretilebilirlik", "Kod, ortam ve çıktı kaydı"),
)


def audit_workflow(completed_keys: set[str]) -> tuple[WorkflowAuditItem, ...]:
    known = {stage.key for stage in RESEARCH_WORKFLOW}
    unknown = completed_keys - known
    if unknown:
        raise ValueError(f"Bilinmeyen araştırma aşaması: {sorted(unknown)}")
    return tuple(
        WorkflowAuditItem(stage=stage, complete=stage.key in completed_keys)
        for stage in RESEARCH_WORKFLOW
    )
