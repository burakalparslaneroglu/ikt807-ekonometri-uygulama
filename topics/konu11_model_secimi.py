"""Konu 11: model seçimi, çapraz doğrulama ve düzenlileştirme.

Uygulama sekmesi ders notları §11.15'i Hansen'in CPS verisiyle adım adım yeniden üretir. Sezgi sekmesi bilinen bir
veri üretim süreciyle üç kontrollü deney sunar; ilki notlardaki Tablo 11.1'i birebir verir. Kendini sına sekmesi
haftanın kavramlarını dört soru türüyle sınar.
"""

from __future__ import annotations

import streamlit as st

from core.labs.registry import get_lab
from core.labs.sezgi_konu11 import KONU11_EXPERIMENTS
from core.quiz.registry import get_quiz
from topics.lab_ui import render_lab
from topics.quiz_ui import render_quiz
from topics.shared import render_topic_header, topic_tabs
from topics.sim_ui import render_experiments

TOPIC_KEY = "konu11"


def render() -> None:
    render_topic_header(TOPIC_KEY)
    application, intuition, self_test = topic_tabs(TOPIC_KEY)
    with application:
        render_lab(get_lab(TOPIC_KEY))
    with intuition:
        render_experiments(KONU11_EXPERIMENTS)
    with self_test:
        render_quiz(get_quiz(TOPIC_KEY))
