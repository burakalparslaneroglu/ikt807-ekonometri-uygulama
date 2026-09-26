"""Haftalık "Kendini sına" soru setlerinin kaydı."""

from __future__ import annotations

from core.quiz.konu01 import KONU01_QUIZ
from core.quiz.konu02 import KONU02_QUIZ
from core.quiz.konu03 import KONU03_QUIZ
from core.quiz.konu04 import KONU04_QUIZ
from core.quiz.konu05 import KONU05_QUIZ
from core.quiz.konu06 import KONU06_QUIZ
from core.quiz.konu07 import KONU07_QUIZ
from core.quiz.konu08 import KONU08_QUIZ
from core.quiz.model import QuestionSet

QUIZZES: dict[str, QuestionSet] = {
    quiz.topic_key: quiz for quiz in (
        KONU01_QUIZ, KONU02_QUIZ, KONU03_QUIZ, KONU04_QUIZ, KONU05_QUIZ, KONU06_QUIZ, KONU07_QUIZ, KONU08_QUIZ,
    )
}


def get_quiz(topic_key: str) -> QuestionSet | None:
    return QUIZZES.get(topic_key)
