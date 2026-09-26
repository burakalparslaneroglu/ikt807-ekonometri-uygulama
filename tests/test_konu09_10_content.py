"""Konu 9 ve 10: RDD ve bootstrap hesaplarının sayısal doğruluğu, Sezgi deneylerinin ekonometrik doğruluğu,
üç dilde kod ve soru setlerindeki yazım çeşitleri.

RDD tahmini statsmodels ``WLS(...).fit(cov_type="HC1")`` ile ve (Rscript varsa) R ``lm(weights)`` +
``sandwich::vcovHC(type = "HC1")`` ile karşılaştırılır. Bootstrap çekiliş sırası, notlardaki kodun açık
döngüsüyle birebir karşılaştırılır.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from core.codegen.base import generator
from core.hansen_data import load_from_upload
from core.labs import rdd as RD
from core.labs import resample as RS
from core.labs.konu09 import KONU09_LAB, TABLE_91
from core.labs.konu10 import KONU10_LAB
from core.labs.runner import LabState, execute
from core.labs.sezgi_konu09 import BANDWIDTH, FUZZY, GLOBAL, KONU09_EXPERIMENTS, approximate_bias
from core.labs.sezgi_konu10 import CLUSTER, HETERO, KONU10_EXPERIMENTS, TRANSFORM, conditional_sd, percent_effect
from core.quiz.konu09 import KONU09_QUIZ
from core.quiz.konu10 import KONU10_QUIZ
from core.quiz.model import grade

EXPERIMENTS = KONU09_EXPERIMENTS + KONU10_EXPERIMENTS
Q9 = {q.key: q for q in KONU09_QUIZ.questions}
Q10 = {q.key: q for q in KONU10_QUIZ.questions}


def _run(experiment, **overrides) -> LabState:
    p = experiment.defaults() | overrides
    state = LabState()
    for op in experiment.build(p):
        execute(op, state, {})
    return state


def _rdd_data(n: int = 600, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-10, 10, n)
    y = 1 + 0.3 * x + 0.02 * x**2 + 1.5 * (x >= 0) + rng.normal(scale=1 + 0.1 * np.abs(x), size=n)
    return x, y


# --- RDD: statsmodels ve R ile aynı sayılar -------------------------------------------------

@pytest.mark.parametrize(("kernel", "scale", "width"), [
    ("triangular", "hansen", 2.0 * np.sqrt(6)), ("rectangular", "hansen", 2.0 * np.sqrt(3)),
    ("triangular", "window", 2.0), ("rectangular", "window", 2.0),
])
def test_rdd_equals_weighted_least_squares_with_hc1(kernel, scale, width) -> None:
    x, y = _rdd_data()
    fit = RD.rdd_fit(x, y, 0.0, 2.0, kernel, scale)
    assert RD.window(2.0, kernel, scale) == pytest.approx(width)
    r = x
    weight = np.maximum(1 - np.abs(r) / width, 0) if kernel == "triangular" else (np.abs(r) <= width) * 1.0
    keep = weight > 0
    d = (r[keep] >= 0) * 1.0
    design = np.column_stack([np.ones(keep.sum()), d, r[keep], d * r[keep]])
    reference = sm.WLS(y[keep], design, weights=weight[keep]).fit(cov_type="HC1")
    np.testing.assert_allclose(fit.params.to_numpy(), reference.params, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fit.bse.to_numpy(), reference.bse, rtol=1e-10)
    assert fit.nobs == keep.sum() and fit.n_left + fit.n_right == fit.nobs
    assert fit.left == pytest.approx(reference.params[0]) and fit.right == pytest.approx(reference.params[0] + reference.params[1])


def test_rdd_drops_missing_values_and_requires_both_sides() -> None:
    x, y = _rdd_data()
    x_missing, y_missing = x.copy(), y.copy()
    x_missing[:5] = np.nan
    y_missing[5:10] = np.nan
    fit = RD.rdd_fit(x_missing, y_missing, 0.0, 2.0)
    complete = RD.rdd_fit(x[10:], y[10:], 0.0, 2.0)
    np.testing.assert_allclose(fit.params, complete.params, rtol=1e-12)
    with pytest.raises(ValueError):
        RD.rdd_fit(np.abs(x) + 20, y, 0.0, 2.0)


def test_local_linear_curve_matches_pointwise_weighted_regressions_and_the_rdd_limits() -> None:
    x, y = _rdd_data(n=400, seed=3)
    h, points = 1.5, np.array([-6.0, -3.2, -0.5, 0.0])
    estimate, standard_error = RD.local_linear_side(x[x < 0], y[x < 0], points, h)
    for point, value, error in zip(points, estimate, standard_error):
        xs, ys = x[x < 0], y[x < 0]
        w = np.maximum(1 - np.abs(xs - point) / (h * np.sqrt(6)), 0)
        keep = w > 0
        fit = sm.WLS(ys[keep], np.column_stack([np.ones(keep.sum()), xs[keep] - point]), weights=w[keep]).fit(
            cov_type="HC1")
        assert value == pytest.approx(fit.params[0], rel=1e-10)
        assert error == pytest.approx(fit.bse[0], rel=1e-10)
    curve = RD.rdd_curve(x, y, 0.0, h, [-5.0, 0.0], [0.0, 5.0])
    jump = RD.rdd_fit(x, y, 0.0, h)
    at_cutoff = curve[curve["x"] == 0.0].set_index("taraf")["tahmin"]
    assert at_cutoff["sol"] == pytest.approx(jump.left, rel=1e-10)
    assert at_cutoff["sag"] == pytest.approx(jump.right, rel=1e-10)


R_RDD = """
set.seed(1)
veri <- read.csv("rdd.csv")
R <- veri$x
w <- pmax(1 - abs(R) / (2 * sqrt(6)), 0)
d <- data.frame(Y = veri$y, R = R, w = w)[w > 0, ]
d$D <- as.numeric(d$R >= 0); d$DR <- d$D * d$R
m <- lm(Y ~ D + R + DR, data = d, weights = w)
cat(sprintf("%.15e", c(coef(m), sqrt(diag(sandwich::vcovHC(m, type = "HC1"))))), sep = "\\n")
"""


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript bulunamadı.")
def test_rdd_matches_r_lm_with_sandwich_hc1(tmp_path: Path) -> None:
    x, y = _rdd_data()
    pd.DataFrame({"x": x, "y": y}).to_csv(tmp_path / "rdd.csv", index=False, float_format="%.17g")
    (tmp_path / "rdd.R").write_text(R_RDD, encoding="utf-8")
    result = subprocess.run(["Rscript", "rdd.R"], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    values = np.array([float(line) for line in result.stdout.split()])
    fit = RD.rdd_fit(x, y, 0.0, 2.0)
    np.testing.assert_allclose(values[:4], fit.params.to_numpy(), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(values[4:], fit.bse.to_numpy(), rtol=1e-9)


# --- Bootstrap: çekiliş sırası ve formüller -----------------------------------------------------

def _regression(n: int = 120, seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    design = np.column_stack([np.ones(n), x])
    return design, 1 + 2 * x + (0.5 + np.abs(x)) * rng.normal(size=n)


def test_pairs_and_wild_draws_follow_the_explicit_loop_of_the_notes() -> None:
    design, y = _regression()
    names = ["Intercept", "x"]
    got = [r.params["x"] for r in RS.replicates(design, y, names, 50, np.random.default_rng(807))]
    rng = np.random.default_rng(807)
    expected = []
    for _ in range(50):
        i = rng.integers(0, len(y), len(y))
        expected.append(np.linalg.lstsq(design[i], y[i], rcond=None)[0][1])
    np.testing.assert_array_equal(got, expected)

    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted, residual = design @ beta, y - design @ beta
    got = [r.params["x"] for r in RS.replicates(design, y, names, 50, np.random.default_rng(3), "wild")]
    rng = np.random.default_rng(3)
    expected = [np.linalg.lstsq(design, fitted + residual * rng.choice([-1.0, 1.0], len(y)), rcond=None)[0][1]
                for _ in range(50)]
    np.testing.assert_array_equal(got, expected)


def test_cluster_bootstrap_resamples_whole_clusters() -> None:
    design, y = _regression(n=60)
    clusters = np.repeat(np.arange(12), 5)
    rng = np.random.default_rng(9)
    replicate = next(RS.replicates(design, y, ["Intercept", "x"], 1, rng, "cluster", clusters))
    rng = np.random.default_rng(9)
    chosen = rng.integers(0, 12, 12)
    index = np.concatenate([np.flatnonzero(clusters == g) for g in chosen])
    expected = np.linalg.lstsq(design[index], y[index], rcond=None)[0]
    np.testing.assert_allclose(replicate.params.to_numpy(), expected, rtol=1e-12)
    with pytest.raises(ValueError):
        next(RS.replicates(design, y, ["Intercept", "x"], 1, rng, "cluster"))


def test_replicate_hc1_errors_match_statsmodels_and_summary_uses_ddof_one() -> None:
    design, y = _regression()
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose(RS.hc1_errors(design, y, beta), sm.OLS(y, design).fit(cov_type="HC1").bse, rtol=1e-10)
    values = np.arange(10.0)
    summary = RS.summary(values)
    assert summary["se"] == pytest.approx(np.std(values, ddof=1))
    assert (summary["lo"], summary["hi"]) == pytest.approx(tuple(np.quantile(values, [0.025, 0.975])))


# --- Laboratuvar tanımları ------------------------------------------------------------------

def test_konu09_lab_uses_hansen_scale_and_the_notes_table() -> None:
    checks = {check.label: check.expected for step in KONU09_LAB.steps for check in step.checks}
    assert checks["h = 8: τ̂"] == -1.51 and checks["h = 8: SH"] == 0.71 and checks["h = 8: etkin örneklem n_h"] == 1041
    assert checks["Dikdörtgen çekirdek: τ̂"] == -1.55 and checks["Pencere ±8: τ̂"] == -2.25
    assert set(TABLE_91) == {4.0, 6.0, 8.0, 10.0, 12.0}
    assert sum(len(step.checks) for step in KONU09_LAB.steps) == 37


def test_konu10_lab_bootstrap_checks_have_monte_carlo_tolerance_only_for_draws() -> None:
    checks = [check for step in KONU10_LAB.steps for check in step.checks]
    assert [check.mc_tolerance is not None for check in checks] == [False, False, False, True, True, True]
    python = generator(KONU10_LAB, "Python").script()
    assert "rng = np.random.default_rng(807)" in python and "i = rng.integers(0, len(y_b), len(y_b))" in python
    assert "np.linalg.lstsq(X_t, y_t, rcond=None)" in python
    r_code = generator(KONU10_LAB, "R").script()
    assert "set.seed(807)" in r_code and "sample.int(nrow(X_b), nrow(X_b), replace = TRUE)" in r_code
    assert "tolerans = 0.00015" in r_code and "tolerans = 0.0005" in r_code
    stata = generator(KONU10_LAB, "Stata").script()
    assert "set seed 807" in stata and "    bsample" in stata and "postfile `boot_sonuc' egitim" in stata
    assert 'kontrol_et `=scalar(boot_egitim_se)\' 0.00109 5 "Pairs bootstrap SH" 0.00015' in stata


def test_lm2007_upload_accepts_hansen_file_and_source_file() -> None:
    frame = pd.DataFrame({"povrate60": [50.0, 60.0, np.nan, 70.0], "mort_age59_related_postHS": [1.0, 2.0, 3.0, np.nan]})
    loaded = load_from_upload("lm2007", frame.to_csv(index=False).encode(), "LM2007_ikt807.csv")
    assert "mort_age59_related_posths" in loaded.frame.columns and not loaded.matches_hansen


# --- Sezgi: ekonometrik doğruluk ------------------------------------------------------------

def test_local_linear_is_nearly_unbiased_while_global_polynomials_are_not() -> None:
    table = _run(GLOBAL).tables["mc"]
    assert abs(table["tau_yerel"].mean() - 1) < 0.03
    assert table["tau_polinom"].mean() - 1 > 0.15
    linear = _run(GLOBAL, p=1).tables["mc"]
    assert linear["tau_polinom"].mean() - 1 < -0.25
    wide = _run(GLOBAL, h=0.3).tables["mc"]
    assert wide["tau_yerel"].mean() - 1 < -0.15


def test_bandwidth_bias_follows_the_second_order_approximation_and_coverage_collapses() -> None:
    assert approximate_bias(0.3) == pytest.approx(-0.216)
    state = _run(BANDWIDTH, h=0.3)
    assert state.tables["mc"]["tau"].mean() - 1 == pytest.approx(approximate_bias(0.3), abs=0.03)
    assert state.scalars["mc_kapsama_tau"] < 0.5
    narrow = _run(BANDWIDTH, h=0.1)
    assert narrow.scalars["mc_kapsama_tau"] > 0.88


def test_weak_first_stage_makes_the_wald_ratio_erratic() -> None:
    strong, weak = _run(FUZZY, delta=0.6).tables["mc"], _run(FUZZY, delta=0.05).tables["mc"]
    assert strong["wald"].median() == pytest.approx(1.0, abs=0.1)
    spread = lambda table: float(np.subtract(*np.quantile(table["wald"], [0.9, 0.1])))
    assert spread(weak) > 5 * spread(strong)
    assert weak["ilk_asama_f"].median() < 3 < 50 < strong["ilk_asama_f"].median()
    assert strong["naif"].mean() > 1.5


def test_heteroskedastic_experiment_reproduces_table_10_1_exactly() -> None:
    table = _run(HETERO).tables["ozet"]["deger"].round(3)
    assert table.tolist() == [1.891, 0.057, 0.076, 0.077, 0.079, 1.741, 2.041, 1.735, 2.036, 1.741, 2.049]
    homoskedastic = _run(HETERO, gamma=0.0)
    assert homoskedastic.models["ols"].bse["x"] == pytest.approx(homoskedastic.models["ols_hc1"].bse["x"], rel=0.15)


def test_percentile_interval_respects_the_transformation_and_is_asymmetric() -> None:
    state = _run(TRANSFORM)
    scalars = state.scalars
    assert scalars["boot_g_lo"] == pytest.approx(percent_effect(scalars["boot_b_lo"]), rel=1e-3)
    assert scalars["boot_g_hi"] == pytest.approx(percent_effect(scalars["boot_b_hi"]), rel=1e-3)
    assert scalars["donusum_alt"] == pytest.approx(scalars["boot_g_lo"], rel=1e-3)
    upper, lower = scalars["boot_g_hi"] - scalars["yuzde"], scalars["yuzde"] - scalars["boot_g_lo"]
    assert upper > 1.3 * lower
    assert scalars["delta_ust"] - scalars["yuzde"] == pytest.approx(scalars["yuzde"] - scalars["delta_alt"])


def test_pairs_bootstrap_understates_uncertainty_with_clustered_errors() -> None:
    state = _run(CLUSTER)
    pairs, cluster = state.scalars["cift_b_se"], state.scalars["kume_b_se"]
    truth = conditional_sd(state, 0.3)
    assert pairs < 0.6 * cluster
    assert cluster == pytest.approx(float(state.models["kume_sh"].bse["x"]), rel=0.3)
    assert cluster == pytest.approx(truth, rel=0.35)
    independent = _run(CLUSTER, rho=0.0)
    assert independent.scalars["cift_b_se"] == pytest.approx(conditional_sd(independent, 0.0), rel=0.2)


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
        assert namespace[name] == pytest.approx(value, rel=1e-9, abs=1e-12), name


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript bulunamadı.")
@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e.key)
def test_generated_r_runs(experiment, tmp_path: Path) -> None:
    script = tmp_path / "deney.R"
    script.write_text(generator(experiment.spec(experiment.defaults()), "R").script(), encoding="utf-8")
    result = subprocess.run(["Rscript", str(script)], cwd=tmp_path, capture_output=True, encoding="utf-8",
                            errors="replace", timeout=900, env=dict(os.environ, LANG="C.UTF-8", LC_ALL="C.UTF-8"))
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e.key)
def test_generated_stata_is_well_formed(experiment) -> None:
    code = generator(experiment.spec(experiment.defaults()), "Stata").script()
    assert "version 14" in code and code.count("{") == code.count("}")
    lines = [line.strip() for line in code.splitlines()]
    assert lines.count("mata:") == lines.count("end") - code.count("program define")
    assert lines.count("preserve") == lines.count("restore")
    for line in code.splitlines():
        if not line.strip().startswith("*"):
            assert line.count('"') % 2 == 0, line


def test_code_conventions_for_rdd_and_bootstrap() -> None:
    python = generator(KONU09_LAB, "Python").script()
    assert 'sm.WLS(ornek[y].to_numpy(dtype=float)[m], Z, weights=w[m]).fit(cov_type="HC1")' in python
    assert 'rdd_dikdortgen = rdd_yerel_dogrusal(hs, "povrate60", "mort_age59_related_posths", 59.1984, 8, ' \
           'cekirdek="dikdortgen")' in python
    assert "def rdd_egrisi(" in python and "ax.set_xlim(15, 82)" in python
    r_code = generator(KONU09_LAB, "R").script()
    assert "lm(Y ~ D + R + DR, data = ornek, weights = w)" in r_code
    assert 'sqrt(diag(sandwich::vcovHC(rdd_dikdortgen, type = "HC1")))[["D"]]' in r_code
    stata = generator(KONU09_LAB, "Stata").script()
    assert "program define rdd_yd" in stata
    assert "quietly regress `y' D R DR [aweight = rdd_w] if rdd_w > 0 & !missing(rdd_w), vce(robust)" in stata
    assert "rdd_yd mort_age59_related_posths povrate60, esik(59.1984) h(8) olcek(pencere)" in stata
    assert 'mata: rdd_egrisi("povrate60", "mort_age59_related_posths", `rdd_n0\', 59.1984, 8*sqrt(6), 15, 82, 120, ' \
           '"egri1")' in stata
    for spec in (KONU09_LAB, KONU10_LAB, *(e.spec(e.defaults()) for e in EXPERIMENTS)):
        for line in generator(spec, "Stata").script().splitlines():
            match = re.search(r"scalar (\w+) =", line)
            if match and "`" not in match.group(1):
                assert len(match.group(1)) <= 32, line


# --- Kendini sına: yazım çeşitleri ----------------------------------------------------------

@pytest.mark.parametrize(("questions", "key", "right", "wrong"), [
    (Q9, "f01", ["Y0 + D*(Y1 - Y0)", "(1-D)Y0 + D Y1", "D*Y1 + (1 - D)*Y_0"], ["D*Y1", "Y0 + Y1", "Y1 - Y0"]),
    (Q9, "f02", ["h*sqrt(6)", "sqrt(6)h", "sqrt(6*h^2)"], ["h*sqrt(3)", "6h", "h"]),
    (Q9, "f03", ["(1/sqrt(6))*(1 - u/sqrt(6))", "1/sqrt(6) - u/6", "(sqrt(6) - u)/6"], ["1 - u/sqrt(6)", "(1-u)/sqrt(6)"]),
    (Q9, "f04", ["1/sqrt(n*h)", "(n h)^(-1/2)", "sqrt(1/(n*h))"], ["1/(n*h)", "1/sqrt(n)", "sqrt(n*h)"]),
    (Q9, "f05", ["sqrt(n*h^5)", "h^2*sqrt(n*h)", "(n h^5)^(1/2)"], ["sqrt(n)*h^2", "n*h^5", "h^2/sqrt(n*h)"]),
    (Q10, "f01", ["(1 - 1/n)^n", "((n-1)/n)^n"], ["1/n", "(1-1/n)", "exp(-1)"]),
    (Q10, "f02", ["(n - 1)/n*S", "S(n-1)/n", "S - S/n"], ["S/n", "(n-1)*S", "n/(n-1)*S"]),
    (Q10, "f03", ["(tb - th)/sb", "(θb − θ)/sb", "tb/sb - th/sb"], ["(th - tb)/sb", "tb/sb", "(tb - th)*sb"]),
    (Q10, "f04", ["sqrt(p*(1 - p)/B)", "sqrt(p(1-p)/B)", "(p - p^2)^(1/2)/sqrt(B)"], ["p*(1-p)/B", "sqrt(p/B)"]),
    (Q10, "f05", ["100*(exp(a) - 1)", "100 exp(a) - 100", "100(exp(a)-1)"], ["100*a", "exp(a) - 1", "100*exp(a)"]),
])
def test_equation_answers_accept_equivalent_forms(questions, key, right, wrong) -> None:
    question = questions[key]
    for text in right:
        assert grade(question, text).correct, text
    for text in wrong:
        assert not grade(question, text).correct, text


@pytest.mark.parametrize(("questions", "key", "right", "wrong"), [
    (Q9, "b02", [("59,1984",), ("59.1984",)], [("59,2",), ("59,198",)]),
    (Q9, "b03", [("−1,51", "0,71"), ("-1.51", "0.71")], [("1,51", "0,71"), ("-2,25", "1,08")]),
    (Q9, "b04", [("zayıf araç",), ("Weak instrument",)], [("güçlü araç",)]),
    (Q10, "b03", [("1000", "10000"), ("1.000", "10.000")], [("100", "1000"), ("10000", "1000")]),
    (Q10, "b05", [("0,00109", "0,1127", "0,1169"), ("0.00109", "0.1127", "0.1169")],
     [("0,00107", "0,1127", "0,1169"), ("0,00109", "0,1169", "0,1127")]),
])
def test_fill_blank_variants(questions, key, right, wrong) -> None:
    question = questions[key]
    for response in right:
        assert grade(question, response).correct, response
    for response in wrong:
        assert not grade(question, response).correct, response
