"""Konu 03: tanımlama, nedensellik ve içsellik.

Uygulama sekmesi ders notları §3.15'i Hansen'in DDK2011 verisiyle adım adım yeniden
üretir. Sezgi sekmesi bilinen bir veri üretim süreciyle üç kontrollü deney sunar.
Kendini sına sekmesi haftanın kavramlarını dört soru türüyle sınar.
"""

from __future__ import annotations

import streamlit as st

from core.labs.registry import get_lab
from core.labs.sezgi_konu03 import KONU03_EXPERIMENTS
from core.quiz.registry import get_quiz
from topics.lab_ui import render_lab
from topics.quiz_ui import render_quiz
from topics.shared import render_topic_header
from topics.sim_ui import render_experiments

TOPIC_KEY = "konu03"


def render() -> None:
    render_topic_header(TOPIC_KEY)
    application, intuition, self_test = st.tabs(("Uygulama", "Sezgi", "Kendini sına"))
    with application:
        render_lab(get_lab(TOPIC_KEY))
    with intuition:
        render_experiments(KONU03_EXPERIMENTS)
    with self_test:
        render_quiz(get_quiz(TOPIC_KEY))
