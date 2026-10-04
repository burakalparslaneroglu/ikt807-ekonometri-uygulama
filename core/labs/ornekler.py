"""Uygulama sekmesinin ek veri kaynaklarının kaydı (alternatif örnek ve kendi verini yükle).

Bir konu buraya eklendiğinde Uygulama sekmesinin üstünde veri kaynağı seçimi görünür ve ortak testler
(``tests/test_lab_variants.py``) o konuyu kendiliğinden kapsar. Kayıtta olmayan konularda sekme yalnız notlardaki
örneği gösterir.
"""

from __future__ import annotations

from core.labs import ornek_konu01, ornek_konu02, ornek_konu03, ornek_konu04
from core.labs.ornek import TopicVariants

VARIANTS: dict[str, TopicVariants] = {
    "konu01": ornek_konu01.VARIANTS,
    "konu02": ornek_konu02.VARIANTS,
    "konu03": ornek_konu03.VARIANTS,
    "konu04": ornek_konu04.VARIANTS,
}


def get_variants(topic_key: str) -> TopicVariants | None:
    return VARIANTS.get(topic_key)
