"""Konu 7 ve 8: kantil regresyon ve yerel doğrusal hesapların sayısal doğruluğu, Sezgi deneylerinin
ekonometrik doğruluğu, üç dilde kod ve soru setlerindeki yazım çeşitleri.

Kantil regresyon benchmark'ı R ``quantreg`` 5.97 ile hesaplanmıştır: ``rq(y ~ x1 + x2, tau)`` ve
``summary(m, se = "nid", covariance = TRUE)``. Veri deterministik bir formülle üretilir (rastgele sayı
üretecine bağlı değildir); R aynı değerleri 17 anlamlı basamaklı bir CSV'den okumuştur.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core.codegen.base import generator
from core.labs import expr as E
from core.labs import quantreg as Q
from core.labs import smoothing as S
from core.labs.konu07 import KONU07_LAB
from core.labs.konu08 import KNOTS, KONU08_LAB
from core.labs.runner import LabState, execute
from core.labs.sezgi_konu07 import (
    CONTAMINATION,
    KONU07_EXPERIMENTS,
    LOCATION_SCALE,
    TAIL,
    theoretical_ratio,
    true_slope,
)
from core.labs.sezgi_konu08 import (
    BANDWIDTH,
    BOUNDARY,
    CV_GRID,
    KONU08_EXPERIMENTS,
    PARTIALLY_LINEAR,
    boundary_errors,
    fit_error,
    linear_control_bias,
)
from core.labs.spec import QuantileRegression
from core.quiz.konu07 import KONU07_QUIZ
from core.quiz.konu08 import KONU08_QUIZ
from core.quiz.model import grade

EXPERIMENTS = KONU07_EXPERIMENTS + KONU08_EXPERIMENTS
Q7 = {q.key: q for q in KONU07_QUIZ.questions}
Q8 = {q.key: q for q in KONU08_QUIZ.questions}


def _run(experiment, **overrides) -> LabState:
    p = experiment.defaults() | overrides
    state = LabState()
    for op in experiment.build(p):
        execute(op, state, {})
    return state


# --- Kantil regresyon: R benchmark'ı --------------------------------------------------------

def _benchmark_design() -> tuple[pd.DataFrame, np.ndarray]:
    i = np.arange(1, 302, dtype=float)
    x1 = 4.0 * np.mod(i * 0.6180339887498949, 1.0)
    x2 = np.sin(1.7 * i)
    e = stats.t.ppf(np.mod(i * 0.7548776662466927 + 0.5, 1.0), 5)
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + (1.0 + 0.3 * x1) * e
    design = pd.DataFrame({"Intercept": 1.0, "x1": x1, "x2": x2})
    return design, y


R_RQ = {
    0.1: ((-0.5355208032553661, 0.0665410596953909, -0.2157466483647256),
          (0.410584765775006, 0.181266422895111, 0.304375099788845)),
    0.5: ((1.043150772203284, 0.488973154793732, -0.357361391016454),
          (0.211738730983153, 0.103473567646311, 0.167559210260248)),
    0.9: ((2.450155173901218, 0.943184562612616, -0.529044257311275),
          (0.408002060208000, 0.194335485613021, 0.318080032631238)),
}
R_DIFFERENCE_X1 = (0.876643502917225, 0.250631882839579)
R_BANDWIDTH = {(0.1, 301): 0.051627431702868251, (0.5, 50742): 0.026242990945165429,
               (0.9, 1000): 0.034599462547393690, (0.95, 100): 0.045725427231861543}


@pytest.mark.parametrize("tau", sorted(R_RQ))
def test_exact_quantile_regression_and_hendricks_koenker_se_match_r(tau) -> None:
    design, y = _benchmark_design()
    fit = Q.quantile_regression(design, y, tau, "nid")
    coefficients, errors = R_RQ[tau]
    np.testing.assert_allclose(fit.params.to_numpy(), coefficients, rtol=0, atol=1e-10)
    np.testing.assert_allclose(fit.bse.to_numpy(), errors, rtol=1e-9)
    plain = Q.quantile_regression(design, y, tau, "none")
    np.testing.assert_allclose(plain.params.to_numpy(), coefficients, rtol=0, atol=1e-10)
    assert plain.bse.isna().all()


def test_joint_covariance_difference_test_matches_r() -> None:
    design, y = _benchmark_design()
    low = Q.quantile_regression(design, y, 0.1, "nid")
    high = Q.quantile_regression(design, y, 0.9, "nid")
    difference, standard_error = Q.quantile_difference(low, high, "x1")
    assert difference == pytest.approx(R_DIFFERENCE_X1[0], abs=1e-10)
    assert standard_error == pytest.approx(R_DIFFERENCE_X1[1], rel=1e-9)
    independent = np.hypot(low.bse["x1"], high.bse["x1"])
    assert standard_error < independent


def test_hall_sheather_bandwidth_matches_r_and_is_halved_at_the_edges() -> None:
    for (tau, n), expected in R_BANDWIDTH.items():
        assert Q.hall_sheather(tau, n) == pytest.approx(expected, rel=1e-12)
    raw = Q.hall_sheather(0.99, 30)
    halved = Q.hk_bandwidth(0.99, 30)
    assert 0.99 + raw > 1 and 0.99 + halved <= 1
    assert np.log2(raw / halved) == pytest.approx(round(np.log2(raw / halved)))


def test_lp_solution_satisfies_the_quantile_subgradient_condition() -> None:
    design, y = _benchmark_design()
    x = design.to_numpy()
    for tau in (0.1, 0.25, 0.5, 0.75, 0.9):
        beta = Q.rq_fit(x, y, tau)
        residual = y - x @ beta
        below, zero = int(np.sum(residual < -1e-9)), int(np.sum(np.abs(residual) <= 1e-9))
        assert zero == x.shape[1]
        assert below <= len(y) * tau <= below + zero
        rng = np.random.default_rng(3)
        best = Q.check_loss_sum(x, y, beta, tau)
        for _ in range(50):
            assert best <= Q.check_loss_sum(x, y, beta + rng.normal(scale=1e-3, size=3), tau) + 1e-12


def test_highs_calls_are_serialized_by_a_process_wide_lock() -> None:
    """HiGHS aynı süreçte eşzamanlı çağrılmaz: Windows'ta eşzamanlı çağrılar erişim ihlaline yol açabiliyor.
    Kilit başka bir iş parçacığında (ör. başka bir Streamlit oturumu) tutulurken çözüm başlamamalıdır."""

    rng = np.random.default_rng(3)
    x = np.column_stack([np.ones(60), rng.normal(size=60)])
    y = x @ np.array([1.0, 2.0]) + rng.normal(size=60)
    results = []
    worker = threading.Thread(target=lambda: results.append(Q.rq_fit(x, y, 0.5)))
    with Q._HIGHS_LOCK:
        worker.start()
        worker.join(timeout=0.5)
        assert worker.is_alive() and not results
    worker.join(timeout=60)
    assert not worker.is_alive() and len(results) == 1
    np.testing.assert_array_equal(results[0], Q.rq_fit(x, y, 0.5))


def test_quantile_difference_requires_the_same_sample_and_nid() -> None:
    design, y = _benchmark_design()
    low = Q.quantile_regression(design, y, 0.1, "none")
    high = Q.quantile_regression(design, y, 0.9, "nid")
    with pytest.raises(ValueError):
        Q.quantile_difference(low, high, "x1")
    shorter = Q.quantile_regression(design.iloc[:-1], y[:-1], 0.1, "nid")
    with pytest.raises(ValueError):
        Q.quantile_difference(shorter, high, "x1")


# --- Yerel doğrusal ve çapraz doğrulama: kaba kuvvet karşılaştırmaları ------------------------

def _smooth_data(n: int = 200, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    return x, np.sin(x) + rng.normal(scale=0.3, size=n)


def test_local_linear_is_the_intercept_of_kernel_weighted_least_squares() -> None:
    x, y = _smooth_data()
    points, h = np.array([0.0, 2.5, 7.3, 10.0]), 0.8
    linear = S.local_fit(x, y, points, h, degree=1)
    constant = S.local_fit(x, y, points, h, degree=0)
    for point, value, average in zip(points, linear, constant):
        w = np.exp(-0.5 * ((x - point) / h) ** 2)
        z = np.column_stack([np.ones_like(x), x - point])
        beta = np.linalg.solve(z.T @ (w[:, None] * z), z.T @ (w * y))
        assert value == pytest.approx(beta[0], rel=1e-10)
        assert average == pytest.approx(np.average(y, weights=w), rel=1e-12)


def test_local_linear_reproduces_a_line_but_nadaraya_watson_does_not_at_the_boundary() -> None:
    x = np.linspace(0, 1, 101)
    y = 1 + 3 * x
    points = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(S.local_fit(x, y, points, 0.1, degree=1), 1 + 3 * points, atol=1e-10)
    nadaraya = S.local_fit(x, y, points, 0.1, degree=0)
    assert nadaraya[0] - 1 > 0.15 and nadaraya[2] - 4 < -0.15
    assert nadaraya[1] == pytest.approx(2.5, abs=1e-10)


def _brute_force_cv(x, y, h, groups=None) -> float:
    errors = []
    for i in range(len(x)):
        keep = np.ones(len(x), dtype=bool)
        keep[i] = False
        if groups is not None:
            keep &= groups != groups[i]
        errors.append(y[i] - S.local_fit(x[keep], y[keep], [x[i]], h)[0])
    return float(np.mean(np.square(errors)))


def test_cv_matches_brute_force_leave_one_out_and_leave_cluster_out() -> None:
    x, y = _smooth_data(n=60, seed=8)
    groups = np.repeat(np.arange(6), 10)
    table = S.cv_curve(x, y, [0.5, 1.2], groups)
    for h in (0.5, 1.2):
        assert table.loc[h, "cv"] == pytest.approx(_brute_force_cv(x, y, h), rel=1e-10)
        assert table.loc[h, "cv_kume"] == pytest.approx(_brute_force_cv(x, y, h, groups), rel=1e-10)
    assert list(S.cv_curve(x, y, [0.5]).columns) == ["cv"]


def test_bandwidth_grid_is_exact_and_ties_pick_the_smaller_h() -> None:
    grid = S.bandwidth_grid(2.0, 20.0, 0.1)
    assert len(grid) == 181 and grid[0] == 2.0 and grid[-1] == 20.0 and 12.3 in grid and 6.2 in grid
    table = pd.DataFrame({"cv": [2.0, 1.0, 1.0, 3.0]}, index=pd.Index([1.0, 2.0, 3.0, 4.0], name="h"))
    assert table["cv"].idxmin() == 2.0


def test_binned_means_and_local_residuals() -> None:
    x, y = _smooth_data(n=97, seed=2)
    binned = S.binned_means(x, y, 5)
    edges = np.linspace(x.min(), x.max(), 6)
    index = np.minimum(np.searchsorted(edges, x, side="right") - 1, 4)
    expected = pd.DataFrame({"x": x, "y": y, "b": index}).groupby("b").agg(x=("x", "mean"), y=("y", "mean"), n=("y", "size"))
    np.testing.assert_allclose(binned.to_numpy(dtype=float), expected.to_numpy(dtype=float), rtol=1e-12)
    np.testing.assert_allclose(S.local_residuals(x, y, 0.7), y - S.local_fit(x, y, x, 0.7), rtol=1e-12)


def test_sine_and_cosine_in_the_expression_language() -> None:
    frame = pd.DataFrame({"x": [0.0, np.pi / 2]})
    np.testing.assert_allclose(E.evaluate(E.sin(E.var("x")), frame), [0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(E.evaluate(E.cos(E.var("x")), frame), [1.0, 0.0], atol=1e-15)
    spec = BANDWIDTH.spec(BANDWIDTH.defaults())
    for language, text in (("Python", 'np.sin(sim["x"])'), ("R", "sin(sim$x)"), ("Stata", "sin(x)")):
        made = generator(spec, language)
        dialect = made.dialect() if language == "Stata" else made.dialect("sim")
        assert E.render(E.sin(E.var("x")), dialect) == text, language


# --- Laboratuvar tanımları ----------------------------------------------------------------------

def test_konu07_lab_uses_exact_lp_hk_se_and_the_corrected_upper_quantile() -> None:
    fits = [op for step in KONU07_LAB.steps for op in step.operations if isinstance(op, QuantileRegression)]
    assert [op.q for op in fits] == [0.10, 0.25, 0.50, 0.75, 0.90]
    assert all(op.vcov == "nid" for op in fits)
    checks = {check.label: check.expected for step in KONU07_LAB.steps for check in step.checks}
    assert checks["τ = 0,90: eğitim"] == 0.1226
    assert checks["τ = 0,90: eğitim SH"] == 0.0014
    assert checks["Eğitim: z istatistiği"] == 8.90


def test_konu08_lab_spline_knots_and_bandwidths() -> None:
    assert KNOTS == (25, 50, 75)
    checks = {check.label: check.expected for step in KONU08_LAB.steps for check in step.checks}
    assert [checks[f"Kübik spline, yüzdelik {p}"] for p in (10, 50, 90)] == [8.02, 13.56, 20.89]
    assert checks["Birini dışarıda bırakan CV ile h"] == 12.3 and checks["Küme-silmeli CV ile h"] == 6.2


# --- Sezgi: ekonometrik doğruluk --------------------------------------------------------------

def test_quantile_slopes_fan_out_only_under_heteroskedasticity() -> None:
    state = _run(LOCATION_SCALE, gamma=0.5, n=5000)
    for tau, model in ((0.1, "q10"), (0.5, "q50"), (0.9, "q90")):
        assert state.models[model].params["x"] == pytest.approx(true_slope(tau, 0.5), abs=0.12)
    parallel = _run(LOCATION_SCALE, gamma=0.0, n=5000)
    slopes = [parallel.models[m].params["x"] for m in ("q10", "q50", "q90")]
    assert max(slopes) - min(slopes) < 0.15


def test_lad_is_more_precise_than_ols_only_with_heavy_tails() -> None:
    heavy = _run(CONTAMINATION).tables["mc"]
    assert heavy["b_lad"].std() < 0.6 * heavy["b_ols"].std()
    assert heavy["b_ols"].mean() == pytest.approx(2.0, abs=0.05) and heavy["b_lad"].mean() == pytest.approx(2.0, abs=0.05)
    normal = _run(CONTAMINATION, share=0.0).tables["mc"]
    assert normal["b_lad"].std() / normal["b_ols"].std() == pytest.approx(np.sqrt(np.pi / 2), rel=0.15)


def test_tail_quantile_noise_follows_the_asymptotic_ratio() -> None:
    table = _run(TAIL, tau=0.95, n=1000).tables["mc"]
    ratio = table["b_kuyruk"].std() / table["b_medyan"].std()
    assert theoretical_ratio(0.95) == pytest.approx(1.686, abs=1e-3)
    assert ratio == pytest.approx(theoretical_ratio(0.95), rel=0.2)


def test_cross_validated_bandwidth_is_close_to_the_oracle() -> None:
    state = _run(BANDWIDTH)
    grid = S.bandwidth_grid(*CV_GRID)
    oracle = min(fit_error(state, h) for h in grid)
    assert fit_error(state, float(state.scalars["cv_h"])) < 1.5 * oracle
    assert fit_error(state, 3.0) > 5 * oracle


def test_boundary_bias_of_nadaraya_watson_but_not_local_linear() -> None:
    state = _run(BOUNDARY, n=3000)
    errors = boundary_errors(state, 0.1)
    assert errors["nw"][0] > 0.15 and errors["nw"][2] < -0.15
    assert abs(errors["ll"][0]) < 0.1 and abs(errors["ll"][2]) < 0.1


def test_robinson_recovers_theta_and_linear_control_is_biased() -> None:
    table = _run(PARTIALLY_LINEAR).tables["mc"]
    assert table["theta_robinson"].mean() == pytest.approx(1.0, abs=0.03)
    assert table["theta_dogrusal"].mean() == pytest.approx(1.0 + linear_control_bias(1.0), abs=0.03)


@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e.key)
def test_experiment_texts_render_for_default_and_extreme_parameters(experiment) -> None:
    for parameters in (experiment.defaults(), {item.key: item.minimum for item in experiment.parameters},
                       {item.key: item.maximum for item in experiment.parameters}):
        lines = experiment.dgp(parameters)
        assert lines and all(isinstance(line, str) and line.strip() for line in lines)
    assert experiment.look_at and experiment.dgp_note and experiment.question.endswith("?")


# --- Üç dilde kod -----------------------------------------------------------------------------

@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e.key)
def test_generated_python_reproduces_the_app_exactly(experiment, monkeypatch) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: plt.close("all"))
    namespace: dict = {}
    exec(generator(experiment.spec(experiment.defaults()), "Python").script(), namespace)
    state = _run(experiment)
    for name, result in state.models.items():
        other = namespace[name]
        np.testing.assert_allclose(other.params.to_numpy(), result.params.to_numpy(), rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(other.bse.to_numpy(), result.bse.to_numpy(), rtol=1e-9, atol=1e-10)
    for name, table in state.tables.items():
        np.testing.assert_allclose(namespace[name].to_numpy(dtype=float), table.to_numpy(dtype=float),
                                   rtol=1e-9, atol=1e-9)
    for name, value in state.scalars.items():
        assert namespace[name] == pytest.approx(value, rel=1e-12), name


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript bulunamadı.")
@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e.key)
def test_generated_r_runs(experiment, tmp_path: Path) -> None:
    script = tmp_path / "deney.R"
    script.write_text(generator(experiment.spec(experiment.defaults()), "R").script(), encoding="utf-8")
    result = subprocess.run(["Rscript", str(script)], cwd=tmp_path, capture_output=True, encoding="utf-8",
                            errors="replace", timeout=600, env=dict(os.environ, LANG="C.UTF-8", LC_ALL="C.UTF-8"))
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e.key)
def test_generated_stata_is_well_formed(experiment) -> None:
    code = generator(experiment.spec(experiment.defaults()), "Stata").script()
    assert "version 14" in code and code.count("{") == code.count("}")
    lines = [line.strip() for line in code.splitlines()]
    assert lines.count("mata:") == lines.count("end") - code.count("program define")
    for line in code.splitlines():
        if not line.strip().startswith("*"):
            assert line.count('"') % 2 == 0, line


def test_code_conventions_for_quantile_and_nonparametric_labs() -> None:
    python = generator(KONU07_LAB, "Python").script()
    assert "linprog(" in python and "quantreg(" not in python and "QuantReg(" not in python and "def kantil_farki(" in python
    r_code = generator(KONU07_LAB, "R").script()
    assert 'q10_ozet <- summary(q10, se = "nid", covariance = TRUE)' in r_code
    assert "fark_egitim_se <- sqrt(" in r_code and "$Hinv %*%" in r_code
    stata = generator(KONU07_LAB, "Stata").script()
    assert "program define hk_sh, eclass" in stata and "ereturn repost V = V_hk" in stata
    assert "hk_sh lwage education female experience experience2_100, tau(0.9) model(q90)" in stata
    assert "matrix C = `w' * Hinv_q10 * J_q10 * Hinv_q90" in stata
    stata8 = generator(KONU08_LAB, "Stata").script()
    assert 'mata: cv_tablosu = cv_sec("percentile", "totalscore", "schoolid", 2, 0.1, 181, "cv")' in stata8
    assert "matrix score double egri_spline = b_spline, equation(#1)" in stata8
    assert "foreach v of varlist  {" not in stata8
    for code in (stata, stata8):
        for line in code.splitlines():
            match = re.search(r"scalar (\w+) =", line)
            if match and "`" not in match.group(1):
                assert len(match.group(1)) <= 32, line
    r8 = generator(KONU08_LAB, "R").script()
    assert "ddk <- ddk[which(ddk$tracking == 1 & ddk$girl == 1), ]" in r8


# --- Kendini sına: yazım çeşitleri ----------------------------------------------------------

@pytest.mark.parametrize(("questions", "key", "right", "wrong"), [
    (Q7, "f01", ["F - t", "F−τ", "-(t - F)"], ["t - F", "F", "1 - F - t"]),
    (Q7, "f02", ["t*(1-t)/(n*f^2)", "τ(1−τ)/(n f^2)", "(t - t^2)/(n*f*f)"], ["t*(1-t)/(n*f)", "t/(n*f^2)", "(1-t)/f^2"]),
    (Q7, "f03", ["v1 + v2 - 2*c", "v_1+v_2-2c"], ["v1 + v2", "v1 + v2 + 2*c", "v1 - v2"]),
    (Q7, "f04", ["1 + g*z", "1+γz", "z*g + 1"], ["g*z", "1 + z", "1 + g"]),
    (Q7, "f05", ["100*(exp(b)-1)", "100(exp(β)-1)", "100*exp(b) - 100"], ["100*b", "exp(b)-1", "100*exp(b)"]),
    (Q8, "f01", ["(C/(4*A*n))^(1/5)", "(C/(4 A n))^0.2"], ["(C/(A*n))^(1/5)", "(C/(4*A*n))^(1/4)", "C/(4*A*n)"]),
    (Q8, "f02", ["(S2*T0 - S1*T1)/(S0*S2 - S1^2)", "(S_2 T_0 - S_1 T_1)/(S_0 S_2 - S_1^2)"],
     ["T0/S0", "(S2*T0 - S1*T1)/(S0*S2)", "(S2*T0 + S1*T1)/(S0*S2 - S1^2)"]),
    (Q8, "f03", ["t*v + e", "θv + e", "e + v*t"], ["v + e", "t*v", "t*(v+e)"]),
    (Q8, "f04", ["-1/(4+d)", "−1/(d+4)"], ["-1/(4*d)", "1/(4+d)", "-1/5"]),
    (Q8, "f05", ["4 + J", "J+4"], ["3 + J", "4*J", "J"]),
])
def test_equation_answers_accept_equivalent_forms(questions, key, right, wrong) -> None:
    question = questions[key]
    for text in right:
        assert grade(question, text).correct, text
    for text in wrong:
        assert not grade(question, text).correct, text


@pytest.mark.parametrize(("questions", "key", "right", "wrong"), [
    (Q7, "b01", [("en küçük mutlak sapmalar",), ("Least Absolute Deviations",)], [("en küçük kareler",)]),
    (Q7, "b04", [("0,1226", "13"), ("0.1226", "13,0")], [("0,1225", "13,0"), ("0,1226", "12,2")]),
    (Q8, "b02", [("12,3", "6,2"), ("12.3", "6.2")], [("6,2", "12,3")]),
    (Q8, "b05", [("-1/5",), ("−0,2",)], [("1/5",), ("-1/4",)]),
])
def test_fill_blank_variants(questions, key, right, wrong) -> None:
    question = questions[key]
    for response in right:
        assert grade(question, response).correct, response
    for response in wrong:
        assert not grade(question, response).correct, response
