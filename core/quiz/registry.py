"""Haftalık "Kendini sına" soru setlerinin kaydı."""

from __future__ import annotations

from core.quiz.konu01 import KONU01_QUIZ
from core.quiz.konu02 import KONU02_QUIZ
from core.quiz.model import QuestionSet

QUIZZES: dict[str, QuestionSet] = {quiz.topic_key: quiz for quiz in (KONU01_QUIZ, KONU02_QUIZ)}


def get_quiz(topic_key: str) -> QuestionSet | None:
    return QUIZZES.get(topic_key)
