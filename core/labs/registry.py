"""Ders notu uygulama laboratuvarlarının kaydı."""

from __future__ import annotations

from core.labs.konu01 import KONU01_LAB
from core.labs.konu02 import KONU02_LAB
from core.labs.konu03 import KONU03_LAB
from core.labs.konu04 import KONU04_LAB
from core.labs.konu05 import KONU05_LAB
from core.labs.konu06 import KONU06_LAB
from core.labs.konu07 import KONU07_LAB
from core.labs.konu08 import KONU08_LAB
from core.labs.spec import LabSpec

LABS: dict[str, LabSpec] = {
    lab.topic_key: lab for lab in (KONU01_LAB, KONU02_LAB, KONU03_LAB, KONU04_LAB, KONU05_LAB, KONU06_LAB, KONU07_LAB, KONU08_LAB)
}


def get_lab(topic_key: str) -> LabSpec | None:
    """Konunun uygulama laboratuvarı; henüz hazırlanmadıysa ``None``."""

    return LABS.get(topic_key)
