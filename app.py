"""IKT 807 uygulamasının ortak Streamlit kabuğu ve konu yönlendirmesi."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.app_config import APP_CONFIG
from core.session_utils import synchronize_active_topic
from core.topic_registry import get_topic, list_topics
from core.ui_preferences import (
    DEFAULT_TEXT_SCALE_LABEL,
    TEXT_SCALE_OPTIONS,
    normalize_text_scale,
    text_scale_css,
)
from topics.konu01_ampirik_modelleme import render as render_konu01
from topics.konu02_dogrusal_regresyon import render as render_konu02
from topics.konu03_tanimlama_icsellik import render as render_konu03
from topics.konu04_iv_2sls import render as render_konu04
from topics.konu05_ikili_ayrik import render as render_konu05
from topics.konu06_sansurleme_secim import render as render_konu06
from topics.konu07_kantil_regresyon import render as render_konu07
from topics.konu08_parametrik_olmayan import render as render_konu08
from topics.konu09_rdd import render as render_konu09
from topics.konu10_bootstrap import render as render_konu10
from topics.konu11_model_secimi import render as render_konu11
from topics.konu12_dml import render as render_konu12


TOPIC_RENDERERS = {
    "konu01": render_konu01,
    "konu02": render_konu02,
    "konu03": render_konu03,
    "konu04": render_konu04,
    "konu05": render_konu05,
    "konu06": render_konu06,
    "konu07": render_konu07,
    "konu08": render_konu08,
    "konu09": render_konu09,
    "konu10": render_konu10,
    "konu11": render_konu11,
    "konu12": render_konu12,
}


def load_styles(scale: float) -> None:
    """Ortak CSS'i ve tek metin ölçeği değişkenini yükler."""

    style_path = Path(__file__).parent / "assets" / "styles.css"
    try:
        css = style_path.read_text(encoding="utf-8")
    except OSError as error:
        st.error(f"Görsel stil dosyası yüklenemedi: {error}")
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown(text_scale_css(scale), unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_CONFIG.course_code} | {APP_CONFIG.course_name}",
        layout="wide",
        initial_sidebar_state="auto",
    )

    scale_label = st.session_state.get(
        "text_scale_label", DEFAULT_TEXT_SCALE_LABEL
    )
    if scale_label not in TEXT_SCALE_OPTIONS:
        scale_label = DEFAULT_TEXT_SCALE_LABEL
    st.session_state["text_scale_label"] = scale_label
    scale = normalize_text_scale(TEXT_SCALE_OPTIONS[scale_label])
    st.session_state["text_scale"] = scale
    load_styles(scale)

    topic_keys = tuple(topic.key for topic in list_topics())
    with st.sidebar:
        st.markdown(f"### {APP_CONFIG.course_code}")
        st.caption(APP_CONFIG.course_name)
        selected_topic = st.radio(
            "Konu seçimi",
            options=topic_keys,
            format_func=lambda key: get_topic(key).label,
            key="topic_selector",
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("#### Görünüm")
        selected_scale = st.selectbox(
            "Metin boyutu",
            options=tuple(TEXT_SCALE_OPTIONS),
            key="text_scale_label",
        )
        st.session_state["text_scale"] = normalize_text_scale(
            TEXT_SCALE_OPTIONS[selected_scale]
        )
        st.divider()
        st.caption(APP_CONFIG.program_name)
        st.caption(APP_CONFIG.institution_name)

    synchronize_active_topic(st.session_state, selected_topic)

    st.markdown(
        f"<div class='app-kicker'>{APP_CONFIG.institution_name}</div>",
        unsafe_allow_html=True,
    )
    st.title(f"{APP_CONFIG.course_code} {APP_CONFIG.course_name}")
    st.caption(
        f"{APP_CONFIG.application_subtitle} | {APP_CONFIG.academic_year}"
    )
    TOPIC_RENDERERS[selected_topic]()


if __name__ == "__main__":
    main()
