"""Konu sayfalarının ortak Streamlit yerleşimi."""

from __future__ import annotations

from html import escape

import streamlit as st

from core.code_recipes import build_colab_notebook, build_python_recipe
from core.data_registry import get_dataset_metadata
from core.question_engine import next_question_index, question_at
from core.session_utils import (
    answer_visibility_key,
    question_index_key,
    set_next_question,
    toggle_answer,
)
from core.topic_registry import get_topic


def render_topic_header(topic_key: str) -> None:
    topic = get_topic(topic_key)
    st.markdown(
        (
            "<section class='topic-band'>"
            f"<div class='topic-number'>Konu {topic.number:02d}</div>"
            f"<h2>{escape(topic.title)}</h2>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            "<div class='guiding-question'>"
            "<div class='label'>Araştırma sorusu</div>"
            f"<p>{escape(topic.guiding_question)}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_orientation(topic_key: str) -> None:
    topic = get_topic(topic_key)
    target_tab, identification_tab, application_tab = st.tabs(
        ["Tahmin hedefi", "Tanımlama", "Veri laboratuvarı"]
    )
    with target_tab:
        st.markdown("#### Tahmin hedefi")
        st.write(topic.estimand)
        st.caption("Yöntemler: " + " | ".join(topic.methods))
    with identification_tab:
        st.markdown("#### Yorum sınırı")
        st.write(topic.identification_focus)
    with application_tab:
        st.markdown("#### Uygulama odağı")
        st.write(topic.application_focus)
        for dataset_id in topic.dataset_ids:
            dataset = get_dataset_metadata(dataset_id)
            st.markdown(f"**{dataset.title}**")
            st.write(f"Gözlem birimi: {dataset.observation_unit}")
            st.write(f"Örneklem: {dataset.sample_definition}")
            st.caption(dataset.source)


def render_question(topic_key: str) -> None:
    index = int(st.session_state[question_index_key(topic_key)])
    question = question_at(topic_key, index)
    visible = bool(st.session_state[answer_visibility_key(topic_key)])

    st.divider()
    st.subheader("Uygulama sorusu")
    st.markdown(f"**{question.prompt}**")
    answer_column, next_column, _ = st.columns([1.1, 1, 3])
    with answer_column:
        label = "Cevabı gizle" if visible else "Cevabı göster"
        if st.button(
            label,
            key=f"{topic_key}_toggle_answer",
            type="primary",
            use_container_width=True,
        ):
            toggle_answer(st.session_state, topic_key)
            st.rerun()
    with next_column:
        if st.button(
            "Yeni soru",
            key=f"{topic_key}_new_question",
            use_container_width=True,
        ):
            next_index = next_question_index(topic_key, index)
            set_next_question(st.session_state, topic_key, next_index)
            st.rerun()
    if visible:
        st.success(question.answer)


def render_reproduction_code(topic_key: str, section_key: str) -> None:
    """Bir laboratuvar bölümünün bağımsız Python ve Colab çıktısını sunar."""

    recipe = build_python_recipe(topic_key, section_key)
    with st.expander("Uygulama kodu", icon=":material/code:"):
        st.caption(
            f"{recipe.section_title} için bağımsız çalışan kod ve Colab defteri."
        )
        python_column, colab_column = st.columns(2)
        python_column.download_button(
            "Python kodunu indir",
            data=recipe.python_code,
            file_name=f"{recipe.filename_stem}.py",
            mime="text/x-python",
            key=f"{topic_key}_{section_key}_python_download",
            icon=":material/download:",
            width="stretch",
        )
        colab_column.download_button(
            "Colab defterini indir",
            data=build_colab_notebook(recipe),
            file_name=f"{recipe.filename_stem}.ipynb",
            mime="application/x-ipynb+json",
            key=f"{topic_key}_{section_key}_colab_download",
            icon=":material/download:",
            width="stretch",
        )


def render_topic(topic_key: str) -> None:
    """Registry'deki bir konunun ortak foundation görünümünü oluşturur."""

    render_topic_header(topic_key)
    render_orientation(topic_key)
    render_question(topic_key)
