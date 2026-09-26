"""Kayıtlı bütün "Kendini sına" setleri için ortak içerik sözleşmesi."""

from __future__ import annotations

from collections import Counter

import pytest

from core.quiz.model import FillBlanks, MultipleChoice, NumberBlank, TrueFalse, correct_answer_text, grade
from core.quiz.registry import QUIZZES

SETS = list(QUIZZES.values())


@pytest.mark.parametrize("quiz", SETS, ids=lambda q: q.topic_key)
def test_rich_and_non_overlapping(quiz) -> None:
    kinds = Counter(q.kind for q in quiz.questions)
    assert set(kinds) == {"coktan", "dogru_yanlis", "bosluk", "denklem"} and min(kinds.values()) >= 4
    for field in ("concept", "key", "prompt"):
        values = [getattr(q, field) for q in quiz.questions]
        assert len(set(values)) == len(values), field
    assert {q.note.section for q in quiz.questions} == set(quiz.sections)


@pytest.mark.parametrize("quiz", SETS, ids=lambda q: q.topic_key)
def test_answer_keys_are_balanced(quiz) -> None:
    choices = [q.answer for q in quiz.questions if isinstance(q.answer, MultipleChoice)]
    assert all(len(set(a.options)) == 4 for a in choices)
    assert len({a.correct for a in choices}) == 4
    truths = [q.answer.statement_is_true for q in quiz.questions if isinstance(q.answer, TrueFalse)]
    assert sum(truths) >= 3 and len(truths) - sum(truths) >= 3


@pytest.mark.parametrize("quiz", SETS, ids=lambda q: q.topic_key)
def test_every_answer_key_grades_itself_and_cites_notes(quiz) -> None:
    for question in quiz.questions:
        answer = question.answer
        if isinstance(answer, MultipleChoice):
            response = answer.correct
        elif isinstance(answer, TrueFalse):
            response = answer.statement_is_true
        elif isinstance(answer, FillBlanks):
            response = tuple(b.shown if isinstance(b, NumberBlank) else b.accepted[0] for b in answer.blanks)
        else:
            response = answer.answer
        assert grade(question, response).correct, (quiz.topic_key, question.key)
        assert correct_answer_text(question)
        assert "§" in question.explanation and len(question.explanation) > 60, question.key
