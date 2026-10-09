"""Konu 11 ve 12: düzenlileştirme, çapraz doğrulama ve DML hesaplarının sayısal doğruluğu, Sezgi deneylerinin
ekonometrik doğruluğu, üç dilde kod ve soru setlerindeki yazım çeşitleri.

Ridge, Lasso ve Elastic Net scikit-learn'ün ``Ridge``, ``Lasso`` ve ``ElasticNet`` tahmin edicileriyle; ceza seçimi
``Pipeline`` + ``GridSearchCV`` + ``PredefinedSplit`` ile (her katta yeniden öğrenilen ölçekleme); artık regresyonu
statsmodels'in sabitsiz HC1 ve küme-dayanıklı kovaryansıyla; model karmaşıklığı ölçütleri statsmodels log-olabilirliği
ve açık dışarıda bırakma döngüsüyle karşılaştırılır. R ``glmnet`` karşılığı laboratuvar testlerinde notlardaki
sayıları yeniden üretir (``tests/test_all_labs.py``).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.codegen.base import generator
from core.labs import penalized as P
from core.labs.konu11 import KONU11_LAB
from core.labs.konu12 import KONU12_LAB
from core.labs.runner import LabState, _monte_carlo, _batchable, execute
from core.labs.sezgi_konu11 import (
    COMPLEXITY,
    KONU11_EXPERIMENTS,
    POST_SELECTION,
    REGULARIZATION,
    long_sd,
    selection_numbers,
)
from core.labs.sezgi_konu12 import (
    CONFOUNDING,
    DOUBLE,
    KONU12_EXPERIMENTS,
    ORTHOGONAL,
    confounding_bias,
    naive_bias,
    omitted_bias,
    orthogonal_bias,
    orthogonal_numbers,
)
from core.quiz.konu11 import KONU11_QUIZ
from core.quiz.konu12 import KONU12_QUIZ
from core.quiz.model import grade

EXPERIMENTS = KONU11_EXPERIMENTS + KONU12_EXPERIMENTS
Q11 = {q.key: q for q in KONU11_QUIZ.questions}
Q12 = {q.key: q for q in KONU12_QUIZ.questions}


def _run(experiment, **overrides) -> LabState:
    p = experiment.defaults() | overrides
    state = LabState()
    for op in experiment.build(p):
        execute(op, state, {})
    return state


def _data(n: int = 240, p: int = 12, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p)) * np.linspace(0.5, 3.0, p) + np.linspace(-1.0, 2.0, p)
    beta = np.zeros(p)
    beta[[0, 3, 7]] = (1.2, -0.8, 0.5)
    frame = pd.DataFrame(x, columns=[f"x{j}" for j in range(1, p + 1)])
    frame["y"] = 0.7 + x @ beta + rng.normal(size=n)
    frame["kat"] = np.arange(n) % 5 + 1
    return frame


NAMES = [f"x{j}" for j in range(1, 13)]


# --- Ceza yolları: scikit-learn ile aynı çözüm --------------------------------------------------------

def _standardized(frame: pd.DataFrame) -> np.ndarray:
    x = frame[NAMES].to_numpy(dtype=float)
    return (x - x.mean(axis=0)) / x.std(axis=0)


@pytest.mark.parametrize(("penalty", "lam", "ratio"), [("ridge", 7.5, 1.0), ("lasso", 0.05, 1.0), ("enet", 0.08, 0.4)])
def test_penalty_path_equals_scikit_learn_with_unpenalized_intercept(penalty, lam, ratio) -> None:
    frame = _data()
    x, y = _standardized(frame), frame["y"].to_numpy()
    path = P.penalty_path(x, y, x, np.array([lam]), penalty, ratio)
    reference = {
        "ridge": Ridge(alpha=lam),
        "lasso": Lasso(alpha=lam, tol=1e-12, max_iter=1_000_000),
        "enet": ElasticNet(alpha=lam, l1_ratio=ratio, tol=1e-12, max_iter=1_000_000),
    }[penalty].fit(x, y)
    np.testing.assert_allclose(path.coefs[:, 0], reference.coef_, rtol=1e-8, atol=1e-10)
    assert path.intercepts[0] == pytest.approx(reference.intercept_, rel=1e-10)
    np.testing.assert_allclose(path.predictions[:, 0], reference.predict(x), rtol=1e-9, atol=1e-10)


def test_lasso_scale_is_the_software_scale_and_twice_n_lambda_in_hansen_scale() -> None:
    """(1/(2n))SSE + λ‖β‖₁ ile SSE + 2nλ‖β‖₁ aynı çözümü verir (Notlar §11.9)."""

    frame = _data()
    x, y = _standardized(frame), frame["y"].to_numpy()
    lam = 0.07
    beta = P.penalty_path(x, y, x, np.array([lam]), "lasso").coefs[:, 0]
    xc, yc = x - x.mean(axis=0), y - y.mean()
    gradient = -xc.T @ (yc - xc @ beta)  # SSE'nin yarısının türevi
    active = beta != 0
    n = len(y)
    np.testing.assert_allclose(gradient[active], -n * lam * np.sign(beta[active]), rtol=1e-6)
    assert np.all(np.abs(gradient[~active]) <= n * lam * (1 + 1e-9))
    assert 0 < active.sum() < len(beta)


def test_cross_validation_matches_grid_search_with_a_pipeline_refitted_in_each_fold() -> None:
    frame = _data(n=300)
    grid = P.grid_values((0, -3, 25))
    ours = P.cross_validate(frame, "y", NAMES, (), frame["kat"].to_numpy(), grid, "lasso")
    search = GridSearchCV(
        make_pipeline(StandardScaler(), Lasso(tol=1e-12, max_iter=1_000_000)),
        {"lasso__alpha": list(grid)}, cv=PredefinedSplit(frame["kat"].to_numpy() - 1),
        scoring="neg_mean_squared_error",
    ).fit(frame[NAMES], frame["y"])
    np.testing.assert_allclose(-search.cv_results_["mean_test_score"], ours.mean[0], rtol=1e-8)
    assert search.best_params_["lasso__alpha"] == ours.lam
    one_se = P.select_index(ours.mean, ours.se, "1se")[1]
    assert one_se <= ours.index[1]
    assert ours.mean[0, one_se] <= ours.mean[0, ours.index[1]] + ours.se[0, ours.index[1]]


def test_categorical_indicators_are_learned_in_training_data_and_reference_is_the_first_level() -> None:
    train = pd.DataFrame({"z": [1.0, 2.0, 3.0], "g": [2, 5, 7]})
    other = pd.DataFrame({"z": [1.0, 2.0], "g": [5, 9]})
    built = P.design(train, other, ("z",), ("g",))
    assert built.names == ["z", "g=5", "g=7"]
    np.testing.assert_array_equal(built.other[:, 1:], [[1.0, 0.0], [0.0, 0.0]])


def test_post_selection_equals_ols_on_the_selected_columns() -> None:
    frame = _data()
    lasso = P.fit_penalized(frame, "y", NAMES, (), "lasso", P.grid_values((0, -2, 20)), None, frame["kat"])
    post = P.post_selection_ols(frame, "y", lasso, NAMES)
    design = sm.add_constant(_standardized(frame)[:, [NAMES.index(n) for n in lasso.selected]])
    reference = sm.OLS(frame["y"].to_numpy(), design).fit()
    np.testing.assert_allclose(post.params[["Intercept", *lasso.selected]].to_numpy(), reference.params, rtol=1e-9)
    assert post.nonzero == len(lasso.selected) < len(NAMES)


# --- DML ve artık regresyonu ------------------------------------------------------------------------------

def test_residual_regression_equals_statsmodels_without_constant() -> None:
    rng = np.random.default_rng(3)
    v = rng.normal(size=400)
    u = 1.3 * v + rng.normal(size=400) * (1 + np.abs(v))
    groups = np.repeat(np.arange(40), 10)
    theta, se = P.residual_regression(u, v)
    reference = sm.OLS(u, v[:, None]).fit(cov_type="HC1")
    assert theta == pytest.approx(reference.params[0], rel=1e-12)
    assert se == pytest.approx(reference.bse[0], rel=1e-10)
    theta_c, se_c = P.residual_regression(u, v, groups)
    clustered = sm.OLS(u, v[:, None]).fit(cov_type="cluster", cov_kwds={"groups": groups})
    assert theta_c == pytest.approx(clustered.params[0], rel=1e-12)
    assert se_c == pytest.approx(clustered.bse[0], rel=1e-10)


def test_crossfit_uses_only_other_folds_and_partialling_out_uses_the_full_sample() -> None:
    frame = _data(n=200)
    frame["d"] = frame["x1"] + np.random.default_rng(9).normal(size=200)
    outer = np.arange(200) % 4 + 1
    fit = P.crossfit_dml(frame, "y", "d", NAMES[:4], outer, None, None, learner="ols")
    x = frame[NAMES[:4]].to_numpy()
    for key in range(1, 5):
        train, test = outer != key, outer == key
        model = sm.OLS(frame["d"].to_numpy()[train], sm.add_constant(x[train])).fit()
        expected = frame["d"].to_numpy()[test] - model.predict(sm.add_constant(x[test], has_constant="add"))
        np.testing.assert_allclose(fit.residuals["v"].to_numpy()[test], expected, rtol=1e-9, atol=1e-10)
    full = P.crossfit_dml(frame, "y", "d", NAMES[:4], None, None, None, learner="ols")
    ols = sm.OLS(frame["y"].to_numpy(), sm.add_constant(np.column_stack([frame["d"], x]))).fit()
    assert full.theta == pytest.approx(ols.params[1], rel=1e-9)  # FWL: artıklaştırma = uzun regresyon


def test_median_aggregation_follows_chernozhukov_et_al() -> None:
    thetas, errors = np.array([1.0, 1.4, 1.1, 0.9, 1.2]), np.array([0.3, 0.2, 0.25, 0.35, 0.3])
    center, error = P.median_aggregate(thetas, errors)
    assert center == 1.1
    assert error == pytest.approx(np.sqrt(np.median(errors**2 + (thetas - 1.1) ** 2)))


def test_complexity_criteria_match_likelihood_and_explicit_leave_one_out() -> None:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"x": rng.uniform(-1, 1, 70)})
    frame["y"] = np.sin(2 * frame["x"]) + 0.3 * rng.normal(size=70)
    frame["egitim"] = (np.arange(70) < 50).astype(float)
    table = P.complexity_curve(frame, "x", "y", 5, frame["egitim"])
    train = frame[frame["egitim"] == 1]
    for degree in (1, 3, 5):
        design = np.column_stack([train["x"] ** power for power in range(degree + 1)])
        model = sm.OLS(train["y"].to_numpy(), design).fit()
        assert table.loc[degree, "aic"] == pytest.approx(-2 * model.llf + 2 * (degree + 2), rel=1e-10)
        assert table.loc[degree, "bic"] == pytest.approx(-2 * model.llf + np.log(50) * (degree + 2), rel=1e-10)
        errors = []
        for i in range(50):
            keep = np.arange(50) != i
            beta = np.linalg.lstsq(design[keep], train["y"].to_numpy()[keep], rcond=None)[0]
            errors.append(train["y"].to_numpy()[i] - design[i] @ beta)
        assert table.loc[degree, "loocv"] == pytest.approx(np.mean(np.square(errors)), rel=1e-9)


def test_batched_monte_carlo_with_ols_dml_equals_the_loop() -> None:
    operation = replace(CONFOUNDING.build(CONFOUNDING.defaults())[0], reps=12)
    assert _batchable(operation)
    batch, loop = LabState(), LabState()
    _monte_carlo(operation, batch, {}, batch=True)
    _monte_carlo(operation, loop, {}, batch=False)
    np.testing.assert_allclose(batch.tables["karistirici"], loop.tables["karistirici"], rtol=1e-10, atol=1e-12)


def test_draw_columns_have_the_ar1_correlation_structure() -> None:
    state = _run(REGULARIZATION, rho=0.6, n=960)
    frame = state.frames["sim"]
    correlation = frame[["x1", "x2", "x3", "x10"]].corr().to_numpy()
    assert correlation[0, 1] == pytest.approx(0.6, abs=0.06)
    assert correlation[0, 2] == pytest.approx(0.36, abs=0.07)
    assert abs(correlation[0, 3]) < 0.08
    assert frame[["x1", "x40"]].std().to_numpy() == pytest.approx([1.0, 1.0], abs=0.06)


# --- Laboratuvar tanımları ------------------------------------------------------------------------------

def test_lab_checks_cover_the_printed_tables() -> None:
    labels11 = [check.label for step in KONU11_LAB.steps for check in step.checks]
    labels12 = [check.label for step in KONU12_LAB.steps for check in step.checks]
    assert len(labels11) == 12 and len(labels12) == 15
    expected = {check.expected for step in KONU12_LAB.steps for check in step.checks}
    assert {1.134, 0.708, 1.383, 0.699, 1.497, 0.704, 1.464, 0.705, 1.267, 1.554} <= expected


# --- Sezgi deneyleri: notlardaki tablolar ve ekonometrik içerik ----------------------------------------

def test_regularization_experiment_reproduces_table_11_1_exactly() -> None:
    state = _run(REGULARIZATION)
    table = state.tables["karsilastirma"]
    assert table["test_mse"].round(3).tolist() == [2.836, 2.820, 2.584, 2.442, 2.164]
    assert table["norm"].round(3).tolist() == [2.305, 1.922, 1.615, 1.835, 2.093]
    assert table["sifirdan"].tolist() == [40, 40, 8, 24, 8]
    assert round(state.models["ridge"].lam, 1) == 20.1
    assert round(state.models["lasso"].lam, 3) == 0.138
    assert round(state.models["enet"].lam, 3) == 0.084 and state.models["enet"].l1_ratio == 0.8
    assert set(state.models["lasso"].selected) >= {"x1", "x2", "x5", "x10", "x20"}


def test_double_selection_experiment_reproduces_table_12_1_exactly() -> None:
    state = _run(DOUBLE)
    table = state.tables["tahminler"]
    assert table["tahmin"].round(3).tolist() == [1.163, 1.124, 1.006, 1.025, 1.032]
    assert table["sh"].round(3).tolist() == [0.027, 0.015, 0.039, 0.039, 0.039]
    scalars = state.scalars
    assert [scalars[f"secim_{k}"] for k in ("ny", "nd", "n")] == [5, 9, 14]
    assert [scalars[f"secim_{k}_iz"] for k in ("ny", "nd", "n")] == [0, 8, 8]
    assert omitted_bias(0.1, 1.0) == pytest.approx(0.6 / 5.6)


def test_training_error_falls_and_bic_is_more_parsimonious_than_aic() -> None:
    for overrides in ({}, {"n": 200}, {"a": 5.0, "sigma": 0.2}):
        state = _run(COMPLEXITY, **overrides)
        table = state.tables["olcutler"]
        assert np.all(np.diff(table["egitim_mse"]) <= 1e-12)
        assert state.scalars["karmasiklik_d_bic"] <= state.scalars["karmasiklik_d_aic"]
        assert table["test_mse"].min() < table["test_mse"].iloc[-1] + 1e-12


def test_pre_testing_breaks_coverage_only_when_the_dropped_control_matters() -> None:
    values = selection_numbers(_run(POST_SELECTION))
    assert values["long"] == pytest.approx(0.95, abs=0.02)
    assert values["post"] < 0.85 and 0.1 < values["rate"] < 0.6 and values["bias"] > 0.05
    assert selection_numbers(_run(POST_SELECTION, beta2=0.0))["post"] > 0.9
    assert selection_numbers(_run(POST_SELECTION, rho=0.0))["post"] > 0.92
    assert long_sd(100, 0.0) == pytest.approx(0.1)


def test_orthogonal_moment_bias_is_second_order_in_the_nuisance_error() -> None:
    values = orthogonal_numbers(_run(ORTHOGONAL))
    assert values["naive"] == pytest.approx(naive_bias(0.15, 1.0, 1.0), abs=0.004)
    assert values["orthogonal"] == pytest.approx(orthogonal_bias(0.15, 1.0, 1.0), abs=0.004)
    assert values["naive_cover"] < 0.5 < 0.88 < values["orthogonal_cover"]
    assert orthogonal_bias(0.1, 1.0, 1.0) < 0.21 * naive_bias(0.1, 1.0, 1.0)
    unbiased = orthogonal_numbers(_run(ORTHOGONAL, delta=0.0))
    assert abs(unbiased["naive"]) < 0.005 and abs(unbiased["orthogonal"]) < 0.005


def test_dml_does_not_remove_unobserved_confounding() -> None:
    state = _run(CONFOUNDING)
    table = state.tables["karistirici"]
    assert table["dml"].mean() - 1 == pytest.approx(confounding_bias(0.5), abs=0.01)
    assert abs(table["dml_u"].mean() - 1) < 0.01
    assert state.scalars["karistirici_kapsama_dml"] < 0.1
    assert state.scalars["karistirici_kapsama_dml_u"] == pytest.approx(0.95, abs=0.025)


@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e.key)
def test_experiment_texts_render_for_default_and_extreme_parameters(experiment) -> None:
    for parameters in (experiment.defaults(), {item.key: item.minimum for item in experiment.parameters},
                       {item.key: item.maximum for item in experiment.parameters}):
        lines = experiment.dgp(parameters)
        assert lines and all(isinstance(line, str) and line.strip() for line in lines)
    assert experiment.look_at and experiment.dgp_note and experiment.question.endswith("?")


@pytest.mark.parametrize("experiment", [e for e in EXPERIMENTS if e is not DOUBLE], ids=lambda e: e.key)
def test_takeaways_render_at_extreme_parameters(experiment) -> None:
    for parameters in ({item.key: item.minimum for item in experiment.parameters},
                       {item.key: item.maximum for item in experiment.parameters}):
        state = LabState()
        for op in experiment.build(parameters):
            execute(op, state, {})
        assert experiment.takeaway(state, parameters)
        assert len(experiment.metrics(state, parameters)) == 4


# --- Üç dilde kod -----------------------------------------------------------------------------------

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
        np.testing.assert_allclose(np.asarray(other.params, dtype=float), np.asarray(result.params, dtype=float),
                                   rtol=1e-9, atol=1e-10)
        if hasattr(result, "bse"):
            np.testing.assert_allclose(np.asarray(other.bse, dtype=float), np.asarray(result.bse, dtype=float),
                                       rtol=1e-9, atol=1e-10)
    compared = 0
    for name, table in state.tables.items():
        if name not in namespace:  # kodda tablo olarak değil, yazdırılarak gösterilen ara sonuçlar
            continue
        mine = table.select_dtypes("number").to_numpy(dtype=float)
        theirs = namespace[name][list(table.select_dtypes("number").columns)].to_numpy(dtype=float)
        np.testing.assert_allclose(theirs, mine, rtol=1e-9, atol=1e-9)
        compared += 1
    assert compared >= 1
    for name, value in state.scalars.items():
        if name in namespace:
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
        match = re.search(r"scalar (\w+) =", line)
        if match and "`" not in match.group(1):
            assert len(match.group(1)) <= 32, line


def test_code_conventions_for_regularization_and_dml() -> None:
    python = generator(REGULARIZATION.spec(REGULARIZATION.defaults()), "Python").script()
    assert "enet_path(xc, yc, l1_ratio=oran, alphas=izgara, precompute=True, tol=1e-12," in python
    assert 'ridge = ceza_modeli(sim, "y", ols_tum_surekli, (), "ridge", ceza_izgarasi(2, -3, 80),' in python
    r_code = generator(REGULARIZATION.spec(REGULARIZATION.defaults()), "R").script()
    assert "standardize = FALSE, intercept = FALSE" in r_code and "thresh = 1e-16" in r_code
    stata = generator(REGULARIZATION.spec(REGULARIZATION.defaults()), "Stata").script()
    assert 'mata: pen_model("lasso", "y", "`ols_tum_surekli\'", "", "lasso", ceza_izgarasi(1, -3, 100),' in stata
    dml_python = generator(DOUBLE.spec(DOUBLE.defaults()), "Python").script()
    assert "dml = dml_capraz(" in dml_python and 'sim["dis10"]' in dml_python
    dml_stata = generator(KONU12_LAB, "Stata").script()
    assert "mata: dml_capraz(" in dml_stata and "dml_kaydet dml" in dml_stata


# --- Kendini sına: yazım çeşitleri ----------------------------------------------------------------

@pytest.mark.parametrize(("questions", "key", "right", "wrong"), [
    (Q11, "f01", ["K*(log(n) - 2)", "K log(n) - 2K", "K*log(n) - 2*K"], ["K*log(n)", "2*K", "K*(log(n) + 2)"]),
    (Q11, "f02", ["2*n*ly", "2nλy", "2 n lambda_y"], ["n*ly", "ly/(2*n)", "2*ly"]),
    (Q11, "f03", ["n*ly", "nλ_y", "ly*n"], ["2*n*ly", "ly/n", "ly"]),
    (Q11, "f04", ["1 - a", "1-α", "1 - alpha"], ["a", "a - 1", "1/a"]),
    (Q11, "f05", ["b1 + b2*r", "β1 + β2 ρ", "beta1 + rho*beta2"], ["b1", "b1 + b2", "b1*r + b2"]),
    (Q12, "f01", ["b + g*t", "β + γθ", "beta + gamma*theta"], ["b + g", "b*g*t", "b + t"]),
    (Q12, "f02", ["d*b*g/(1 + g^2)", "δβγ/(1+γ^2)", "delta*beta*gamma/(1 + gamma^2)"],
     ["d^2*b*g/(1 + d^2*g^2)", "d*b*g", "b*g/(1 + g^2)"]),
    (Q12, "f03", ["A/B^2", "A/(B*B)", "A*B^-2"], ["A/B", "A^2/B", "sqrt(A)/B"]),
    (Q12, "f04", ["s^2 + (t - m)^2", "SH_s^2 + (θ_s - θ_med)^2",
                  "σ_s^2 + (θ_s - θ_med)^2", "(m - t)^2 + s^2"], ["s + (t - m)^2", "s^2", "(t - m)^2"]),
    (Q12, "f05", ["k^2/(1 + k^2)", "κ^2/(1+κ^2)", "1 - 1/(1 + k^2)"], ["k^2", "k/(1 + k^2)", "k^2/(1 + k)"]),
])
def test_equation_answers_accept_equivalent_forms(questions, key, right, wrong) -> None:
    question = questions[key]
    for text in right:
        assert grade(question, text).correct, text
    for text in wrong:
        assert not grade(question, text).correct, text


@pytest.mark.parametrize(("questions", "key", "right", "wrong"), [
    (Q11, "b01", [("7,39",), ("7.39",), ("7,389",)], [("7,4",), ("2",)]),
    (Q11, "b02", [("224", "62"), ("224", "61,9")], [("320", "62"), ("224", "0,138")]),
    (Q11, "b03", [("12.687", "38.055"), ("12687", "38055")], [("38.055", "12.687"), ("12.690", "38.055")]),
    (Q12, "b01", [("0", "8"), ("0,0", "8")], [("5", "9"), ("8", "0")]),
    (Q12, "b02", [("10", "DML2"), ("10", "dml2")], [("5", "DML2"), ("10", "DML1")]),
    (Q12, "b03", [("1,267", "1,554"), ("1.267", "1.554")], [("1,27", "1,55"), ("1,554", "1,267")]),
])
def test_fill_blank_variants(questions, key, right, wrong) -> None:
    question = questions[key]
    for response in right:
        assert grade(question, response).correct, response
    for response in wrong:
        assert not grade(question, response).correct, response
