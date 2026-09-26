"""Konu 1 Sezgi deneyleri: ekonometrik doğruluk ve üç dilde kod.

Deneylerin DGP'si bilindiği için tahminler bilinen gerçekle karşılaştırılabilir.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from core.codegen.base import LANGUAGES, generator
from core.labs.runner import LabState, execute
from core.labs.sezgi_konu01 import CAUSAL_RETURN, CONFOUNDING, KONU01_EXPERIMENTS, PROJECTION, RESIDUALS


def _run(experiment, **overrides) -> LabState:
    parameters = experiment.defaults() | overrides
    state = LabState()
    for op in experiment.build(parameters):
        execute(op, state, {})
    return state


# --- Ekonometrik doğruluk ---------------------------------------------------------

def test_linear_cef_makes_the_target_slope_exactly_the_true_slope() -> None:
    state = _run(PROJECTION, gamma=0.0)
    assert state.models["hedef"].params["x"] == pytest.approx(0.08, abs=1e-10)


def test_saturated_model_reproduces_cell_means_exactly() -> None:
    table = _run(PROJECTION, gamma=0.02).tables["karsilastirma"]
    np.testing.assert_allclose(table["yhat_doymus"], table["ort_y"], atol=1e-10)
    assert (table["yhat_dogrusal"] - table["ort_y"]).abs().max() > 0.2


def test_ols_slope_estimates_the_projection_not_the_cef() -> None:
    state = _run(PROJECTION, n=5000, gamma=0.02)
    ols, target = state.models["dogrusal"].params["x"], state.models["hedef"].params["x"]
    assert ols == pytest.approx(target, abs=0.015)


def test_ols_weights_cells_by_their_size() -> None:
    state = _run(PROJECTION, gamma=0.02)
    table = state.tables["karsilastirma"]
    x = table.index.to_numpy(dtype=float)
    weights = table["n"].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    root = np.sqrt(weights)[:, None]
    wls = np.linalg.lstsq(design * root, table["ort_y"].to_numpy() * root[:, 0], rcond=None)[0]
    assert wls[1] == pytest.approx(state.models["dogrusal"].params["x"], abs=1e-10)


def test_normal_equations_hold_even_when_the_cef_is_curved() -> None:
    values = _run(RESIDUALS, gamma=0.02).tables["ozellikler"]["Değer"]
    assert abs(values.iloc[0]) < 1e-8
    assert abs(values.iloc[1]) < 1e-6
    assert values.iloc[2] == pytest.approx(values.iloc[3], abs=1e-10)


def test_curvature_shows_up_as_u_shaped_residual_means() -> None:
    means = _run(RESIDUALS, n=5000, gamma=0.02).tables["artik_ortalamalari"]
    edges = means.loc[means.index.isin([8, 9, 19, 20]), "ort_ehat"]
    centre = means.loc[means.index.isin([13, 14]), "ort_ehat"]
    assert (edges > 0).all() and (centre < 0).all()


def test_confounding_moves_ols_away_from_the_causal_return() -> None:
    state = _run(CONFOUNDING, n=5000)
    assert state.models["kisa"].params["x"] > CAUSAL_RETURN + 0.02
    assert state.models["uzun"].params["x"] == pytest.approx(CAUSAL_RETURN, abs=0.015)


@pytest.mark.parametrize("override", [{"pi": 0.0}, {"delta": 0.0}])
def test_no_gap_when_ability_is_unrelated_to_schooling_or_wages(override) -> None:
    state = _run(CONFOUNDING, n=5000, **override)
    assert state.models["kisa"].params["x"] == pytest.approx(CAUSAL_RETURN, abs=0.015)


# --- Kod ------------------------------------------------------------------------

def test_every_experiment_renders_in_every_language() -> None:
    for experiment in KONU01_EXPERIMENTS:
        for language in LANGUAGES:
            code = generator(experiment.spec(experiment.defaults()), language).script()
            assert "sezgi deneyi" in code
            assert "kontrol_et" not in code


@pytest.mark.parametrize("experiment", KONU01_EXPERIMENTS, ids=lambda e: e.key)
def test_generated_python_reproduces_the_app_exactly(experiment, monkeypatch) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Test ortamında pencere yok: plt.show() yerine grafikler sessizce kapatılır.
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: plt.close("all"))
    parameters = experiment.defaults()
    namespace: dict = {}
    exec(generator(experiment.spec(parameters), "Python").script(), namespace)
    state = _run(experiment)
    for name, result in state.models.items():
        np.testing.assert_allclose(namespace[name].params.to_numpy(), result.params.to_numpy(), rtol=0, atol=1e-12)


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript bulunamadı.")
@pytest.mark.parametrize("experiment", KONU01_EXPERIMENTS, ids=lambda e: e.key)
def test_generated_r_runs(experiment, tmp_path: Path) -> None:
    script = tmp_path / "deney.R"
    script.write_text(generator(experiment.spec(experiment.defaults()), "R").script(), encoding="utf-8")
    environment = dict(os.environ, LANG="C.UTF-8", LC_ALL="C.UTF-8")
    result = subprocess.run(["Rscript", str(script)], cwd=tmp_path, capture_output=True, encoding="utf-8",
                            errors="replace", timeout=300, env=environment)
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize("experiment", KONU01_EXPERIMENTS, ids=lambda e: e.key)
def test_generated_stata_is_well_formed(experiment) -> None:
    code = generator(experiment.spec(experiment.defaults()), "Stata").script()
    assert "version 14" in code and "set seed 807" in code and "set obs 2000" in code
    generates = re.findall(r"^generate\b.*$", code, flags=re.MULTILINE)
    # Hesaplanan her değer double; tek istisna tam sayı sıra numarası (long).
    computed = [line for line in generates if line != "generate long id = _n"]
    assert computed and all(line.startswith("generate double ") for line in computed)
    assert code.count("{") == code.count("}")
    lines = [line.strip() for line in code.splitlines()]
    assert lines.count("preserve") == lines.count("restore")
    for line in code.splitlines():
        if not line.strip().startswith("*"):
            assert line.count('"') % 2 == 0, line
    for foreign in ("smf.", "<-", "np.", "print("):
        assert foreign not in code
    if experiment is PROJECTION:
        assert "quietly regress y i.x" in code
