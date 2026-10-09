import numpy as np
import pytest

from core.codegen.base import LANGUAGES, generator
from core.labs.runner import LabState, execute
from core.labs.sezgi_ek_ders import COLLINEARITY, COLLIDER, collinearity_se, collider_target


def run(experiment, **parameters):
    state = LabState()
    for op in experiment.build(experiment.defaults() | parameters):
        execute(op, state, {})
    return state


@pytest.mark.parametrize("rho", [0, 0.8, 0.95])
def test_fwl_true_se_matches_independent_inverse_design(rho):
    state = run(COLLINEARITY, rho=rho)
    frame = state.frames["sim"]
    design = np.column_stack((np.ones(len(frame)), frame[["x1", "x2"]]))
    benchmark = np.sqrt(np.linalg.inv(design.T @ design)[1, 1])
    assert collinearity_se(state) == pytest.approx(benchmark, rel=1e-12)


def test_collinearity_increases_uncertainty_with_same_true_coefficient():
    low, high = run(COLLINEARITY, rho=0), run(COLLINEARITY, rho=0.95)
    assert collinearity_se(high) / collinearity_se(low) == pytest.approx(1 / np.sqrt(1-0.95**2), rel=1e-12)


@pytest.mark.parametrize("b", [0, 1, 2])
def test_collider_target_matches_population_covariance_system(b):
    # Var(D)=1/4, Var(U)=Var(V)=1; invert the population projection system independently.
    covariance = np.array([[0.25, b*0.25], [b*0.25, b*b*0.25+2]])
    outcome_covariance = np.array([0.25, b*0.25+1])
    assert np.linalg.solve(covariance, outcome_covariance)[0] == pytest.approx(collider_target(b))
    state = run(COLLIDER, n=5000, b=b)
    assert state.models["dogru"].params["d"] == pytest.approx(1, abs=0.08)
    assert state.models["yanlis_kontrol"].params["d"] == pytest.approx(collider_target(b), abs=0.08)


@pytest.mark.parametrize("experiment", [COLLINEARITY, COLLIDER])
def test_generated_languages_and_executed_python_match(experiment, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    for language in LANGUAGES:
        code = generator(experiment.spec(experiment.defaults()), language).script()
        assert code
        if language == "Python":
            namespace = {}
            exec(code, namespace)
            for key, model in run(experiment).models.items():
                np.testing.assert_allclose(namespace[key].params, model.params, rtol=0, atol=1e-12)
