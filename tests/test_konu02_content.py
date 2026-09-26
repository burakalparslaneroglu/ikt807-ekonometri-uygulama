"""Konu 2: soru setindeki yazım çeşitleri ve Sezgi deneylerinin ekonometrik doğruluğu."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from core.codegen.base import generator
from core.labs.runner import LabState, execute
from core.labs.sezgi_konu02 import CLUSTER, FWL, HETERO, KONU02_EXPERIMENTS, _moulton, _true_slope_se
from core.quiz.konu02 import KONU02_QUIZ
from core.quiz.model import grade

Q = {q.key: q for q in KONU02_QUIZ.questions}


@pytest.mark.parametrize(("key", "right", "wrong"), [
    ("f01", ["-b1/(2*b2)", "-β1/(2β2)", "-(beta1)/(2 beta2)"], ["b1/(2*b2)", "-b1/b2", "-2*b2/b1"]),
    ("f02", ["v1 + v3 + 2*c13", "v3+v1+2c13"], ["v1 + v3", "v1 + v3 - 2*c13", "v1*v3"]),
    ("f03", ["100*exp(b)*s", "100 s exp(β)", "SE*100*exp(b)"], ["100*b*s", "exp(b)*s", "100*(exp(b)-1)*s"]),
    ("f04", ["(b - b0)/s", "(β̂-β0)/SE"], ["b/s", "(b0 - b)/s", "(b - b0)*s"]),
    ("f05", ["sqrt(1 + (m - 1)*rho)", "sqrt(1+(m-1)ρ)"], ["1 + (m - 1)*rho", "sqrt(m*rho)"]),
])
def test_equation_variants(key, right, wrong) -> None:
    for text in right:
        assert grade(Q[key], text).correct, text
    for text in wrong:
        assert not grade(Q[key], text).correct, text


@pytest.mark.parametrize(("key", "right", "wrong"), [
    ("b01", ["0,02", "0.02"], ["0,04", "-0,02"]),
    ("b03", ["Homoskedastisite", "sabit varyans"], ["heteroskedastisite", "normallik"]),
    ("b04", ["n/(n-k)", "n / (n − k)"], ["(n-k)/n", "n/k"]),
])
def test_fill_variants(key, right, wrong) -> None:
    for text in right:
        assert grade(Q[key], (text,)).correct, text
    for text in wrong:
        assert not grade(Q[key], (text,)).correct, text


def _run(experiment, **overrides) -> LabState:
    p = experiment.defaults() | overrides
    state = LabState()
    for op in experiment.build(p):
        execute(op, state, {})
    return state


def test_hc1_tracks_the_true_se_and_classic_understates_it() -> None:
    state = _run(HETERO, n=5000, gamma=0.3)
    true = _true_slope_se(state)
    assert state.models["klasik"].params["x"] == state.models["hc1"].params["x"]
    assert state.models["klasik"].bse["x"] < 0.85 * true
    assert abs(state.models["hc1"].bse["x"] / true - 1) < 0.1


def test_fwl_is_exact_and_short_regression_targets_the_ovb_formula() -> None:
    state = _run(FWL, n=5000, rho=0.6, beta2=0.8)
    assert state.models["fwl"].params["x1_art"] == pytest.approx(state.models["uzun"].params["x1"], abs=1e-12)
    assert state.models["kisa"].params["x1"] == pytest.approx(0.5 + 0.8 * 0.6, abs=0.04)


def test_cluster_se_inflation_is_near_the_moulton_factor() -> None:
    p = CLUSTER.defaults() | {"groups": 80, "rho": 0.5}
    state = _run(CLUSTER, groups=80, rho=0.5)
    ratio = state.models["kume_sh"].bse["x"] / state.models["klasik"].bse["x"]
    assert ratio == pytest.approx(_moulton(p), rel=0.3)
    assert abs(state.models["hc1"].bse["x"] / state.models["klasik"].bse["x"] - 1) < 0.15


@pytest.mark.parametrize("experiment", KONU02_EXPERIMENTS, ids=lambda e: e.key)
def test_generated_python_reproduces_the_app_exactly(experiment, monkeypatch) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: plt.close("all"))
    namespace: dict = {}
    exec(generator(experiment.spec(experiment.defaults()), "Python").script(), namespace)
    for name, result in _run(experiment).models.items():
        np.testing.assert_allclose(namespace[name].params.to_numpy(), result.params.to_numpy(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(namespace[name].bse.to_numpy(), result.bse.to_numpy(), rtol=0, atol=1e-12)


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript bulunamadı.")
@pytest.mark.parametrize("experiment", KONU02_EXPERIMENTS, ids=lambda e: e.key)
def test_generated_r_runs(experiment, tmp_path: Path) -> None:
    script = tmp_path / "deney.R"
    script.write_text(generator(experiment.spec(experiment.defaults()), "R").script(), encoding="utf-8")
    result = subprocess.run(["Rscript", str(script)], cwd=tmp_path, capture_output=True, encoding="utf-8",
                            errors="replace", timeout=300, env=dict(os.environ, LANG="C.UTF-8", LC_ALL="C.UTF-8"))
    assert result.returncode == 0, result.stderr[-2000:]
