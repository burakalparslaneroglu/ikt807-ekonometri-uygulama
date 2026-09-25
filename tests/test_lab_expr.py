from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.codegen.python_gen import PythonGenerator
from core.codegen.r_gen import RGenerator
from core.codegen.stata_gen import StataGenerator
from core.labs import expr as E
from core.labs.konu01 import KONU01_LAB


EXPERIENCE = E.maximum(E.sub(E.sub(E.var("age"), E.var("education")), 6), 0)
WAGE = E.div(E.var("earnings"), E.mul(E.var("hours"), E.var("week")))


def _dialects():
    return (
        PythonGenerator(KONU01_LAB).dialect("cps"),
        RGenerator(KONU01_LAB).dialect("cps"),
        StataGenerator(KONU01_LAB).dialect(),
    )


def test_rendering_matches_each_language_idiom() -> None:
    python, r, stata = _dialects()
    assert E.render(WAGE, python) == 'cps["earnings"] / (cps["hours"] * cps["week"])'
    assert E.render(WAGE, r) == "cps$earnings / (cps$hours * cps$week)"
    assert E.render(WAGE, stata) == "earnings / (hours * week)"
    assert E.render(EXPERIENCE, python) == 'np.maximum(cps["age"] - cps["education"] - 6, 0)'
    assert E.render(EXPERIENCE, r) == "pmax(cps$age - cps$education - 6, 0)"
    assert E.render(EXPERIENCE, stata) == "max(age - education - 6, 0)"


def test_parentheses_follow_precedence_and_associativity() -> None:
    _, _, stata = _dialects()
    a, b, c = E.var("a"), E.var("b"), E.var("c")
    assert E.render(E.sub(a, E.sub(b, c)), stata) == "a - (b - c)"
    assert E.render(E.sub(E.sub(a, b), c), stata) == "a - b - c"
    assert E.render(E.div(a, E.mul(b, c)), stata) == "a / (b * c)"
    assert E.render(E.mul(E.add(a, b), c), stata) == "(a + b) * c"
    assert E.render(E.power(E.power(a, 2), 3), stata) == "(a ^ 2) ^ 3"
    assert E.render(E.div(E.power(a, 2), 100), stata) == "a ^ 2 / 100"


def test_power_and_coefficient_syntax_differs_by_language() -> None:
    python, r, stata = _dialects()
    square = E.power(E.var("x"), 2)
    assert E.render(square, python) == 'cps["x"] ** 2'
    assert E.render(square, r) == "cps$x ^ 2"
    effect = E.mul(100, E.sub(E.exp(E.coef("m3", "education")), 1))
    assert E.render(effect, python) == '100 * (np.exp(m3.params["education"]) - 1)'
    assert E.render(effect, r) == '100 * (exp(coef(m3)[["education"]]) - 1)'
    assert E.render(effect, stata) == "100 * (exp(_b[education]) - 1)"
    intercept = E.coef("m1", E.INTERCEPT)
    assert E.render(intercept, python) == 'm1.params["Intercept"]'
    assert E.render(intercept, r) == 'coef(m1)[["(Intercept)"]]'
    assert E.render(intercept, stata) == "_b[_cons]"


def test_evaluation_matches_rendered_formula() -> None:
    frame = pd.DataFrame(
        {"age": [15.0, 30.0, 60.0], "education": [12.0, 16.0, 12.0], "earnings": [1.0, 2.0, 3.0],
         "hours": [40.0, 50.0, 40.0], "week": [50.0, 52.0, 48.0]}
    )
    np.testing.assert_allclose(E.evaluate(EXPERIENCE, frame), [0.0, 8.0, 42.0])
    np.testing.assert_allclose(E.evaluate(WAGE, frame), [1 / 2000, 2 / 2600, 3 / 1920])


def test_validation_rejects_unknown_nodes() -> None:
    E.validate(EXPERIENCE)
    with pytest.raises(ValueError):
        E.validate(E.BinOp("%", E.var("a"), E.var("b")))
    with pytest.raises(ValueError):
        E.validate(E.Call("tanh", (E.var("a"),)))
    with pytest.raises(ValueError):
        E.validate(E.Call("maximum", (E.var("a"),)))
