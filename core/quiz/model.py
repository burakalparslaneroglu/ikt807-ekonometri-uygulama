"""Haftalık "Kendini sına" soru setlerinin veri modeli ve notlandırma kuralları.

Dört soru türü vardır: çoktan seçmeli, doğru–yanlış, boşluk doldurma ve denklem
yazma. Her soru tek bir kavramı sınar (``concept``); aynı set içinde iki soru aynı
kavramı sınamaz. Bu kural testle denetlenir.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Union

from core.labs.spec import NoteRef
from core.quiz.expression import FormulaError, Symbol, equivalent, parse

KIND_LABELS = {
    "coktan": "Çoktan seçmeli",
    "dogru_yanlis": "Doğru–yanlış",
    "bosluk": "Boşluk doldurma",
    "denklem": "Denklem yazma",
}


@dataclass(frozen=True)
class MultipleChoice:
    options: tuple[str, ...]
    correct: int
    kind: str = "coktan"


@dataclass(frozen=True)
class TrueFalse:
    statement_is_true: bool
    kind: str = "dogru_yanlis"


@dataclass(frozen=True)
class TextBlank:
    accepted: tuple[str, ...]
    shown: str


@dataclass(frozen=True)
class NumberBlank:
    value: float
    tolerance: float
    shown: str


Blank = Union[TextBlank, NumberBlank]


@dataclass(frozen=True)
class FillBlanks:
    blanks: tuple[Blank, ...]
    kind: str = "bosluk"


@dataclass(frozen=True)
class Equation:
    lhs: str
    symbols: tuple[Symbol, ...]
    answer: str
    shown: str
    kind: str = "denklem"


Answer = Union[MultipleChoice, TrueFalse, FillBlanks, Equation]


@dataclass(frozen=True)
class Question:
    key: str
    concept: str
    note: NoteRef
    prompt: str
    answer: Answer
    explanation: str

    @property
    def kind(self) -> str:
        return self.answer.kind


@dataclass(frozen=True)
class QuestionSet:
    topic_key: str
    title: str
    questions: tuple[Question, ...]
    intro: str = ""
    sections: tuple[str, ...] = field(default_factory=tuple)

    def question(self, key: str) -> Question:
        for item in self.questions:
            if item.key == key:
                return item
        raise ValueError(f"Soru bulunamadı: {key}")


@dataclass(frozen=True)
class Grade:
    correct: bool
    message: str = ""


# --- Notlandırma ------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Türkçe büyük/küçük harf, boşluk ve noktalama farklarını yok sayar."""

    text = text.strip().replace("İ", "i").replace("I", "ı").lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", "", text)
    text = text.replace("(", "[").replace(")", "]")
    return text.strip(".,;:!?'\"")


def parse_number(text: str) -> float | None:
    cleaned = text.strip().replace("%", "").replace(" ", "").replace("−", "-")
    if cleaned.count(",") == 1 and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def grade_blank(blank: Blank, response: str) -> bool:
    if isinstance(blank, NumberBlank):
        value = parse_number(response)
        return value is not None and abs(value - blank.value) <= blank.tolerance
    return normalize_text(response) in {normalize_text(item) for item in blank.accepted}


def grade(question: Question, response) -> Grade:
    answer = question.answer
    if isinstance(answer, MultipleChoice):
        return Grade(response == answer.correct)
    if isinstance(answer, TrueFalse):
        return Grade(response == answer.statement_is_true)
    if isinstance(answer, FillBlanks):
        responses = list(response or ())
        if len(responses) != len(answer.blanks) or any(not str(item).strip() for item in responses):
            return Grade(False, "Bütün boşlukları doldurun.")
        results = [grade_blank(blank, str(item)) for blank, item in zip(answer.blanks, responses)]
        if all(results):
            return Grade(True)
        wrong = [str(index) for index, ok in enumerate(results, start=1) if not ok]
        return Grade(False, "Tutmayan boşluk: " + ", ".join(wrong) + ".")
    if isinstance(answer, Equation):
        try:
            candidate = parse(str(response or ""), answer.symbols)
        except FormulaError as error:
            return Grade(False, str(error))
        target = parse(answer.answer, answer.symbols)
        return Grade(equivalent(candidate, target, answer.symbols))
    raise TypeError(type(answer).__name__)


def correct_answer_text(question: Question) -> str:
    answer = question.answer
    if isinstance(answer, MultipleChoice):
        return answer.options[answer.correct]
    if isinstance(answer, TrueFalse):
        return "Doğru" if answer.statement_is_true else "Yanlış"
    if isinstance(answer, FillBlanks):
        return " · ".join(f"({i}) {blank.shown}" for i, blank in enumerate(answer.blanks, start=1))
    return answer.shown
