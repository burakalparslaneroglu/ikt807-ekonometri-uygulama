"""Konu 1 "Kendini sına": içerik sözleşmesi, notlandırma, güvenlik ve arayüz."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core.quiz.expression import FormulaError, Symbol, parse
from core.quiz.konu01 import KONU01_QUIZ
from core.quiz.model import (
    Equation,
    FillBlanks,
    MultipleChoice,
    NumberBlank,
    TextBlank,
    TrueFalse,
    correct_answer_text,
    grade,
)
from core.quiz.registry import get_quiz

QUESTIONS = {question.key: question for question in KONU01_QUIZ.questions}
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


# --- İçerik sözleşmesi ----------------------------------------------------------

def test_set_is_registered_and_rich() -> None:
    assert get_quiz("konu01") is KONU01_QUIZ
    kinds = Counter(question.kind for question in KONU01_QUIZ.questions)
    assert set(kinds) == {"coktan", "dogru_yanlis", "bosluk", "denklem"}
    assert all(count >= 4 for count in kinds.values())


def test_questions_do_not_overlap() -> None:
    concepts = [question.concept for question in KONU01_QUIZ.questions]
    keys = [question.key for question in KONU01_QUIZ.questions]
    prompts = [question.prompt for question in KONU01_QUIZ.questions]
    assert len(set(concepts)) == len(concepts)
    assert len(set(keys)) == len(keys)
    assert len(set(prompts)) == len(prompts)


def test_every_declared_note_section_is_covered_and_nothing_else() -> None:
    covered = {question.note.section for question in KONU01_QUIZ.questions}
    assert covered == set(KONU01_QUIZ.sections)


def test_multiple_choice_items_are_well_formed() -> None:
    items = [q.answer for q in KONU01_QUIZ.questions if isinstance(q.answer, MultipleChoice)]
    for answer in items:
        assert len(answer.options) == 4 and len(set(answer.options)) == 4
        assert 0 <= answer.correct < 4
    positions = Counter(answer.correct for answer in items)
    assert len(positions) == 4, "Doğru şık hep aynı konumda olmamalı."


def test_true_false_items_are_balanced() -> None:
    values = [q.answer.statement_is_true for q in KONU01_QUIZ.questions if isinstance(q.answer, TrueFalse)]
    assert sum(values) >= 3 and len(values) - sum(values) >= 3


def test_every_question_explains_itself_and_cites_the_notes() -> None:
    for question in KONU01_QUIZ.questions:
        assert len(question.explanation) > 60, question.key
        assert "§" in question.explanation, question.key


# --- Notlandırma --------------------------------------------------------------------

def test_shown_answers_are_graded_correct() -> None:
    for question in KONU01_QUIZ.questions:
        answer = question.answer
        if isinstance(answer, MultipleChoice):
            response = answer.correct
        elif isinstance(answer, TrueFalse):
            response = answer.statement_is_true
        elif isinstance(answer, FillBlanks):
            response = tuple(
                blank.shown if isinstance(blank, NumberBlank) else blank.accepted[0] for blank in answer.blanks
            )
        else:
            response = answer.answer
        assert grade(question, response).correct, question.key
        assert correct_answer_text(question)


@pytest.mark.parametrize(
    ("key", "right", "wrong"),
    [
        ("b01", ["Ortalaması", "beklenen değeri"], ["medyanı", "varyansı"]),
        ("b02", ["E[Y|X]", "E(Y | X)", "m(X)"], ["E[Y]", "E[X|Y]"]),
        ("b03", ["Sağa", "sağ", "pozitif yönde"], ["sola", "negatif"]),
        ("b04", ["42,57", "42.57", "%42,57"], ["0,4257", "57,43"]),
        ("b06", ["3,1708", "3.1707", "3,17"], ["3,2011", "1,7312"]),
    ],
)
def test_fill_blanks_accept_variants_and_reject_neighbours(key, right, wrong) -> None:
    question = QUESTIONS[key]
    for response in right:
        assert grade(question, (response,)).correct, response
    for response in wrong:
        assert not grade(question, (response,)).correct, response


def test_two_blank_question_needs_both_blanks() -> None:
    question = QUESTIONS["b05"]
    assert grade(question, ("36", "48")).correct
    assert not grade(question, ("36", "52")).correct
    assert "Bütün boşlukları" in grade(question, ("36", "")).message


@pytest.mark.parametrize(
    ("key", "right", "wrong"),
    [
        ("f01", ["rho*sy/sx", "ρ σ_Y/σ_X", "sy rho / sx", "rho*(sy/sx)"], ["rho*sx/sy", "rho", "sy/sx"]),
        ("f02", ["E[Y] - b*E[X]", "EY - beta EX", "E(Y)-βE(X)", "-b EX + EY"], ["E[Y] + b*E[X]", "E[Y]/E[X]"]),
        ("f03", ["b/12", "β/12", "(1/12)beta", "b*12^(-1)"], ["12b", "b", "b/365"]),
        ("f04", ["100(exp(b)-1)", "100*exp(b)-100", "(exp(b) - 1) × 100"], ["100b", "exp(b)-1", "100*log(1+b)"]),
    ],
)
def test_equations_accept_equivalent_forms_and_reject_common_mistakes(key, right, wrong) -> None:
    question = QUESTIONS[key]
    for response in right:
        assert grade(question, response).correct, response
    for response in wrong:
        assert not grade(question, response).correct, response


def test_brackets_are_math_not_code_and_huge_powers_are_harmless() -> None:
    question = QUESTIONS["f03"]
    assert grade(question, "[b]/12").correct
    assert not grade(question, "b^9^9^9").correct


def test_unreadable_equation_explains_the_problem() -> None:
    result = grade(QUESTIONS["f03"], "b /")
    assert not result.correct and result.message
    result = grade(QUESTIONS["f03"], "c/12")
    assert "tanımlı bir sembol değil" in result.message


def test_parser_rejects_code_injection() -> None:
    symbols = (Symbol("b", "b", "katsayı"),)
    for text in ("__import__('os').system('x')", "b.real", "lambda: 1", "exec('1')", "b if b else 1",
                 "b; import os", "{b}", "b @ b", "b == b", "not b"):
        with pytest.raises(FormulaError):
            parse(text, symbols)


# --- Sorulardaki sayılar gerçek veriyle ------------------------------------------------

def _data_path() -> Path | None:
    for name in ("IKT807_HANSEN_CPS09MAR_PATH", "IKT807_CPS_PATH"):
        value = os.environ.get(name)
        if value and Path(value).is_file():
            return Path(value)
    return None


@pytest.mark.skipif(_data_path() is None, reason="Gerçek Hansen cps09mar dosyası verilmedi.")
def test_numbers_quoted_in_questions_match_the_data() -> None:
    import statsmodels.api as sm

    from core.hansen_data import load_from_upload

    path = _data_path()
    frame = load_from_upload("cps09mar", path.read_bytes(), path.name).frame
    frame["lwage"] = np.log(frame["earnings"] / (frame["hours"] * frame["week"]))
    fit = sm.OLS(frame["lwage"], sm.add_constant(frame["education"])).fit()
    cells = frame.groupby("education")["lwage"].agg(["mean", "size"])
    design = sm.add_constant(cells.index.to_numpy(dtype=float))
    weighted = sm.WLS(cells["mean"], design, weights=cells["size"]).fit()
    unweighted = sm.OLS(cells["mean"], design).fit()
    assert round(fit.params["education"], 4) == 0.1082  # k04
    assert weighted.params.iloc[1] == pytest.approx(fit.params["education"], abs=1e-10)  # k04
    assert round(unweighted.params.iloc[1], 4) == 0.0750  # k04 açıklaması
    prediction = fit.params["const"] + 16 * fit.params["education"]
    assert grade(QUESTIONS["b06"], (f"{prediction:.4f}",)).correct  # b06
    assert round(cells.loc[16, "mean"], 4) == 3.2011  # b06 açıklaması


# --- Arayüz -------------------------------------------------------------------------

def _quiz_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=60).run()


def test_quiz_tab_renders_with_empty_progress() -> None:
    app = _quiz_app()
    assert not app.exception
    assert [tab.label for tab in app.tabs][:3] == ["Uygulama", "Sezgi", "Kendini sına"]
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Cevaplanan"] == "0/25"


def test_correct_and_wrong_answers_update_progress_and_review_list() -> None:
    app = _quiz_app()
    app.radio(key="konu01_quiz_k01").set_value(1).run()
    app.button(key="konu01_quiz_check_k01").click().run()
    assert not app.exception
    assert any(item.value == "Doğru." for item in app.success)

    app.button(key="konu01_quiz_next").click().run()
    app.radio(key="konu01_quiz_k02").set_value(0).run()
    app.button(key="konu01_quiz_check_k02").click().run()
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Cevaplanan"] == "2/25" and metrics["Doğru"] == "1"
    assert any("Tekrar edilecek bölümler" in item.value and "§1.5" in item.value for item in app.markdown)

    app.button(key="konu01_quiz_reset").click().run()
    assert {metric.label: metric.value for metric in app.metric}["Cevaplanan"] == "0/25"


def test_equation_question_previews_and_grades_typed_formula() -> None:
    app = _quiz_app()
    app.pills(key="konu01_quiz_current").set_value(25).run()
    app.text_input(key="konu01_quiz_f04").input("100(exp(b)-1)").run()
    assert any("e^{b}" in block.value for block in app.latex)
    app.button(key="konu01_quiz_check_f04").click().run()
    assert not app.exception
    assert any(item.value == "Doğru." for item in app.success)
