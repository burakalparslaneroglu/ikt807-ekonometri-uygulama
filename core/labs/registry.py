"""Ders notu uygulama laboratuvarlarının kaydı."""

from __future__ import annotations

from core.labs.konu01 import KONU01_LAB
from core.labs.spec import LabSpec

LABS: dict[str, LabSpec] = {
    KONU01_LAB.topic_key: KONU01_LAB,
}


def get_lab(topic_key: str) -> LabSpec | None:
    """Konunun uygulama laboratuvarı; henüz hazırlanmadıysa ``None``."""

    return LABS.get(topic_key)
