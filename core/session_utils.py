"""Streamlit session_state ile uyumlu saf state geçişleri."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


def question_index_key(topic_key: str) -> str:
    return f"{topic_key}_question_index"


def answer_visibility_key(topic_key: str) -> str:
    return f"{topic_key}_answer_visible"


def initialize_question_state(state: MutableMapping[str, Any], topic_key: str) -> None:
    state.setdefault(question_index_key(topic_key), 0)
    state.setdefault(answer_visibility_key(topic_key), False)


def synchronize_active_topic(
    state: MutableMapping[str, Any], topic_key: str
) -> bool:
    """Konu değişiminde soru state'ini sıfırlar; değişim olduysa True döndürür."""

    previous = state.get("active_topic")
    changed = previous != topic_key
    state["active_topic"] = topic_key
    initialize_question_state(state, topic_key)
    if changed:
        state[question_index_key(topic_key)] = 0
        state[answer_visibility_key(topic_key)] = False
    return changed


def toggle_answer(state: MutableMapping[str, Any], topic_key: str) -> bool:
    initialize_question_state(state, topic_key)
    key = answer_visibility_key(topic_key)
    state[key] = not bool(state[key])
    return bool(state[key])


def set_next_question(
    state: MutableMapping[str, Any], topic_key: str, next_index: int
) -> None:
    initialize_question_state(state, topic_key)
    state[question_index_key(topic_key)] = next_index
    state[answer_visibility_key(topic_key)] = False
