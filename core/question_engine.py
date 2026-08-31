"""Güncel konu sonucuna bağlanabilecek deterministik soru sırası."""

from __future__ import annotations

from dataclasses import dataclass

from core.topic_registry import get_topic


@dataclass(frozen=True)
class Question:
    question_id: str
    prompt: str
    answer: str


def questions_for_topic(topic_key: str) -> tuple[Question, ...]:
    """Konu sorularını kararlı kimliklerle üretir."""

    topic = get_topic(topic_key)
    return tuple(
        Question(
            question_id=f"{topic.key}_q{index + 1}",
            prompt=prompt,
            answer=answer,
        )
        for index, (prompt, answer) in enumerate(topic.questions)
    )


def question_at(topic_key: str, index: int) -> Question:
    """İndeksi güvenli biçimde döngüsel soru sırasına eşler."""

    questions = questions_for_topic(topic_key)
    if not questions:
        raise ValueError(f"{topic_key} için soru tanımlanmamış.")
    return questions[index % len(questions)]


def next_question_index(topic_key: str, current_index: int) -> int:
    """Bir sonraki soru indeksini kararlı biçimde döndürür."""

    return (current_index + 1) % len(questions_for_topic(topic_key))
