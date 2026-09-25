"""Haftalık "Kendini sına" soru setlerinin kaydı."""

from __future__ import annotations

from core.quiz.konu01 import KONU01_QUIZ
from core.quiz.model import QuestionSet

QUIZZES: dict[str, QuestionSet] = {KONU01_QUIZ.topic_key: KONU01_QUIZ}


def get_quiz(topic_key: str) -> QuestionSet | None:
    return QUIZZES.get(topic_key)
