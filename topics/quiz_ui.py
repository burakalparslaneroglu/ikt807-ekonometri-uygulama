"""Haftalık "Kendini sına" sekmesi.

Üstte ilerleme özeti ve yanlış cevaplara göre tekrar edilecek bölümler; ortada
durum işaretli soru haritası; altta tek soru kartı. Denklem sorularında öğrencinin
yazdığı ifade anında LaTeX olarak önizlenir.
"""

from __future__ import annotations

import streamlit as st

from core.quiz.expression import FormulaError, latex, parse
from core.quiz.model import (
    KIND_LABELS,
    Equation,
    FillBlanks,
    MultipleChoice,
    Question,
    QuestionSet,
    TrueFalse,
    correct_answer_text,
    grade,
)


def _results_key(quiz: QuestionSet) -> str:
    return f"{quiz.topic_key}_quiz_results"


def _current_key(quiz: QuestionSet) -> str:
    return f"{quiz.topic_key}_quiz_current"


def _widget_key(quiz: QuestionSet, question: Question, part: int | None = None) -> str:
    base = f"{quiz.topic_key}_quiz_{question.key}"
    return base if part is None else f"{base}_{part}"


def _results(quiz: QuestionSet) -> dict[str, dict]:
    return st.session_state.setdefault(_results_key(quiz), {})


def _response(quiz: QuestionSet, question: Question):
    answer = question.answer
    if isinstance(answer, FillBlanks):
        return tuple(
            st.session_state.get(_widget_key(quiz, question, index), "") or ""
            for index in range(1, len(answer.blanks) + 1)
        )
    return st.session_state.get(_widget_key(quiz, question))


def _reset(quiz: QuestionSet) -> None:
    st.session_state[_results_key(quiz)] = {}
    for question in quiz.questions:
        for key in list(st.session_state.keys()):
            if str(key).startswith(_widget_key(quiz, question)):
                del st.session_state[key]
    st.session_state[_current_key(quiz)] = 1


def _move(quiz: QuestionSet, delta: int) -> None:
    current = st.session_state.get(_current_key(quiz)) or 1
    st.session_state[_current_key(quiz)] = min(max(current + delta, 1), len(quiz.questions))


def _check(quiz: QuestionSet, question: Question) -> None:
    response = _response(quiz, question)
    result = grade(question, response)
    _results(quiz)[question.key] = {"response": response, "correct": result.correct, "message": result.message}


# --- Özet ve harita ----------------------------------------------------------

def _summary(quiz: QuestionSet) -> None:
    results = _results(quiz)
    total = len(quiz.questions)
    answered = len(results)
    correct = sum(item["correct"] for item in results.values())
    with st.container(border=True):
        left, middle, right, reset = st.columns([1, 1, 1, 1], vertical_alignment="center")
        left.metric("Cevaplanan", f"{answered}/{total}")
        middle.metric("Doğru", f"{correct}")
        right.metric("Başarı", f"%{round(100 * correct / answered)}" if answered else "—")
        reset.button("Baştan başla", key=f"{quiz.topic_key}_quiz_reset", on_click=_reset, args=(quiz,),
                     icon=":material/restart_alt:", width="stretch")
        st.progress(answered / total)
        wrong_sections: list[str] = []
        for question in quiz.questions:
            item = results.get(question.key)
            if item and not item["correct"] and question.note.section not in wrong_sections:
                wrong_sections.append(question.note.section)
        if wrong_sections:
            ordered = sorted(wrong_sections, key=lambda text: tuple(int(part) for part in text.split(".")))
            st.markdown("**Tekrar edilecek bölümler:** " + ", ".join(f"§{s}" for s in ordered))
        elif answered == total and total:
            st.markdown("**Bütün sorular doğru.** Bu haftanın temel kavramları yerinde.")


def _navigation(quiz: QuestionSet) -> Question:
    key = _current_key(quiz)
    total = len(quiz.questions)
    if st.session_state.get(key) not in range(1, total + 1):
        st.session_state[key] = 1
    results = _results(quiz)

    def label(number: int) -> str:
        item = results.get(quiz.questions[number - 1].key)
        if item is None:
            return f"{number}"
        return f"✓ {number}" if item["correct"] else f"✗ {number}"

    st.pills("Soru", options=list(range(1, total + 1)), format_func=label, key=key,
             label_visibility="collapsed")
    if st.session_state.get(key) is None:  # seçimi kaldırılırsa ilk soruya dön
        st.session_state[key] = 1
    return quiz.questions[st.session_state[key] - 1]


# --- Soru kartı -------------------------------------------------------------

def _input(quiz: QuestionSet, question: Question) -> None:
    answer = question.answer
    key = _widget_key(quiz, question)
    if isinstance(answer, MultipleChoice):
        st.radio("Cevabınız", options=list(range(len(answer.options))), index=None,
                 format_func=lambda i: answer.options[i], key=key, label_visibility="collapsed")
    elif isinstance(answer, TrueFalse):
        st.radio("Cevabınız", options=[True, False], index=None, horizontal=True,
                 format_func=lambda value: "Doğru" if value else "Yanlış", key=key,
                 label_visibility="collapsed")
    elif isinstance(answer, FillBlanks):
        columns = st.columns(len(answer.blanks))
        for index, column in enumerate(columns, start=1):
            column.text_input(f"Boşluk ({index})", key=_widget_key(quiz, question, index))
    elif isinstance(answer, Equation):
        legend = " · ".join(f"`{s.name}` = ${s.latex}$ ({s.meaning})" for s in answer.symbols)
        st.caption(
            f"Semboller: {legend}. İşlemler: `+ - * / ^ ( )`, `exp()`, `log()`, `sqrt()`. "
            "Çarpma işareti atlanabilir: `100(exp(b)-1)` geçerlidir."
        )
        st.text_input("Formülünüz", key=key, placeholder="ör. 2*b/12")
        text = st.session_state.get(key) or ""
        if text.strip():
            try:
                st.latex(f"{answer.lhs} = {latex(parse(text, answer.symbols), answer.symbols)}")
            except FormulaError as error:
                st.caption(f"Önizleme: {error}")


def _feedback(quiz: QuestionSet, question: Question) -> None:
    item = _results(quiz).get(question.key)
    if item is None:
        return
    if item["response"] != _response(quiz, question):
        st.caption("Cevabınızı değiştirdiniz; yeniden kontrol edin.")
        return
    if item["correct"]:
        st.success("Doğru.", icon=":material/check_circle:")
    else:
        extra = f" {item['message']}" if item.get("message") else ""
        st.error(f"Doğru değil.{extra}", icon=":material/cancel:")
        if isinstance(question.answer, Equation):
            st.markdown("**Doğru cevap:**")
            st.latex(question.answer.shown)
        else:
            st.markdown(f"**Doğru cevap:** {correct_answer_text(question)}")
    st.markdown(question.explanation)


def _card(quiz: QuestionSet, question: Question) -> None:
    number = quiz.questions.index(question) + 1
    with st.container(border=True):
        st.caption(f"Soru {number}/{len(quiz.questions)} · {KIND_LABELS[question.kind]} · {question.note.label()}")
        st.markdown(question.prompt)
        _input(quiz, question)
        left, _, prev, nxt = st.columns([1.3, 2, 1, 1])
        left.button("Kontrol et", key=f"{quiz.topic_key}_quiz_check_{question.key}", type="primary",
                    on_click=_check, args=(quiz, question), width="stretch")
        prev.button("‹ Önceki", key=f"{quiz.topic_key}_quiz_prev", on_click=_move, args=(quiz, -1),
                    width="stretch", disabled=number == 1)
        nxt.button("Sonraki ›", key=f"{quiz.topic_key}_quiz_next", on_click=_move, args=(quiz, 1),
                   width="stretch", disabled=number == len(quiz.questions))
        _feedback(quiz, question)


def render_quiz(quiz: QuestionSet) -> None:
    st.markdown(quiz.intro)
    _summary(quiz)
    question = _navigation(quiz)
    _card(quiz, question)
