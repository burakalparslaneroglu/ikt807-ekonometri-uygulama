from __future__ import annotations

from core.question_engine import (
    next_question_index,
    question_at,
    questions_for_topic,
)
from core.session_utils import (
    answer_visibility_key,
    question_index_key,
    set_next_question,
    synchronize_active_topic,
    toggle_answer,
)


def test_question_order_is_deterministic_and_wraps() -> None:
    first = questions_for_topic("konu04")
    second = questions_for_topic("konu04")
    assert first == second
    assert question_at("konu04", 0) == question_at("konu04", len(first))
    assert next_question_index("konu04", len(first) - 1) == 0


def test_topic_change_resets_question_but_view_scale_survives() -> None:
    state: dict[str, object] = {"text_scale_label": "%130"}
    assert synchronize_active_topic(state, "konu01")
    toggle_answer(state, "konu01")
    set_next_question(state, "konu01", 1)
    assert state[question_index_key("konu01")] == 1
    assert state[answer_visibility_key("konu01")] is False

    assert synchronize_active_topic(state, "konu04")
    assert state[question_index_key("konu04")] == 0
    assert state[answer_visibility_key("konu04")] is False
    assert state["text_scale_label"] == "%130"


def test_same_topic_does_not_reset_current_question() -> None:
    state: dict[str, object] = {}
    synchronize_active_topic(state, "konu03")
    set_next_question(state, "konu03", 1)
    assert not synchronize_active_topic(state, "konu03")
    assert state[question_index_key("konu03")] == 1
