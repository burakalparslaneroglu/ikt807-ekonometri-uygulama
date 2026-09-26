"""Konu 5 ve 6: sınırlı bağımlı değişken hesapları, Sezgi deneylerinin doğruluğu, üç dilde kod,
soru setlerindeki yazım çeşitleri ve notlardaki benzetim tablolarıyla tutarlılık."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from streamlit.testing.v1 import AppTest

from core.codegen.base import generator
from core.labs import expr as E
from core.labs import limited as L
from core.labs.runner import LabState, _batchable, _monte_carlo, execute
from core.labs.sezgi_konu05 import DUMMY, KONU05_EXPERIMENTS, LPM, SCALE, true_ame_logit, true_dummy_effect
from core.labs.sezgi_konu06 import CENSORING, EXCLUSION, KONU06_EXPERIMENTS, SELECTION
from core.labs.spec import (
    BinaryChoice,
    MarginalEffects,
    MonteCarlo,
    Tobit,
)
from core.quiz.konu05 import KONU05_QUIZ
from core.quiz.konu06 import KONU06_QUIZ
from core.quiz.model import grade

EXPERIMENTS = KONU05_EXPERIMENTS + KONU06_EXPERIMENTS
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
Q5 = {q.key: q for q in KONU05_QUIZ.questions}
Q6 = {q.key: q for q in KONU06_QUIZ.questions}


def _run(experiment, **overrides) -> LabState:
    p = experiment.defaults() | overrides
    state = LabState()
    for op in experiment.build(p):
        execute(op, state, {})
    return state


def _binary_frame(seed: int = 11, n: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x": rng.normal(size=n), "c": rng.integers(1, 4, n), "d": rng.integers(0, 2, n)})
    index = 0.2 + 0.8 * frame.x + 0.5 * (frame.c == 2) - 0.4 * (frame.c == 3) + 0.7 * frame.d
    frame["y"] = (index + rng.logistic(size=n) > 0).astype(int)
    return frame


# --- Hesaplar ------------------------------------------------------------------------------

@pytest.mark.parametrize("link", ("logit", "probit"))
def test_marginal_effects_match_statsmodels(link) -> None:
    frame = _binary_frame()
    state = LabState(frames={"f": frame})
    execute(BinaryChoice("m", "f", "y", ("x", "c", "d"), link, categorical=("c",)), state, {})
    execute(MarginalEffects("m", "me", ("x", "c", "d"), discrete=("d",)), state, {})
    ours = state.models["me"]
    theirs = state.models["m"].get_margeff(at="overall", dummy=True).summary_frame()
    for term in ("x", "d"):
        assert ours.params[term] == pytest.approx(theirs.loc[term, "dy/dx"], abs=1e-10)
        assert ours.bse[term] == pytest.approx(theirs.loc[term, "Std. Err."], abs=1e-10)
    # Çok düzeyli kategoride düzey, referans düzeyine göre fark: elle hesap
    result = state.models["m"]
    X = L.design_frame(result)
    b = result.params[X.columns].to_numpy()
    G, _, _ = L.link_functions(link)
    x1, x0 = X.copy(), X.copy()
    x1[["C(c)[T.2]", "C(c)[T.3]"]] = 0.0
    x0[["C(c)[T.2]", "C(c)[T.3]"]] = 0.0
    x1["C(c)[T.3]"] = 1.0
    assert ours.params["c=3"] == pytest.approx(np.mean(G(x1.to_numpy() @ b) - G(x0.to_numpy() @ b)), abs=1e-12)


def test_robust_binary_covariance_is_hc0_sandwich_with_observed_hessian() -> None:
    frame = _binary_frame(n=1500)
    probit = smf.probit("y ~ x + d", frame).fit(disp=False, cov_type="HC0")
    X = probit.model.exog
    z = X @ probit.params.to_numpy()
    q = 2 * frame["y"].to_numpy() - 1
    from scipy import stats

    ratio = q * stats.norm.pdf(q * z) / stats.norm.cdf(q * z)
    bread = np.linalg.inv((X * (ratio * (ratio + z))[:, None]).T @ X)
    meat = (X * ratio[:, None]).T @ (X * ratio[:, None])
    np.testing.assert_allclose(bread @ meat @ bread, probit.cov_params().to_numpy(), rtol=1e-8)


def test_tobit_newton_matches_known_r_solution() -> None:
    rng = np.random.default_rng(807)
    x = rng.uniform(-3, 3, 5000)
    y = np.maximum(1 + x + rng.normal(0, 1, 5000), 0)
    state = LabState(frames={"f": pd.DataFrame({"x": x, "y": y})})
    execute(Tobit("t", "f", "y", ("x",)), state, {})
    fit = state.models["t"]
    # Notlardaki Tablo 6.2'nin Tobit satırı (R AER::tobit ile aynı çözüm)
    assert fit.params["Intercept"] == pytest.approx(1.00569649, abs=1e-7)
    assert fit.params["x"] == pytest.approx(1.00249191, abs=1e-7)
    assert fit.sigma == pytest.approx(1.00203342, abs=1e-7)


def test_tobit_targets_and_fit_check_formulas() -> None:
    latent = pd.Series([-1.0, 0.0, 2.5])
    table = L.tobit_targets(latent, 2.0)
    from scipy import stats

    z = latent / 2.0
    assert table["p_poz"].tolist() == pytest.approx(stats.norm.cdf(z).tolist())
    assert table["gozlenen"].tolist() == pytest.approx((stats.norm.cdf(z) * latent + 2 * stats.norm.pdf(z)).tolist())
    assert (table["poz_ort"] > table["gizli"]).all()


def test_heckman_monte_carlo_is_vectorized_and_matches_the_loop() -> None:
    block = next(op for op in EXCLUSION.build(EXCLUSION.defaults() | {"n": 500}) if isinstance(op, MonteCarlo))
    block = MonteCarlo(block.frame, 25, block.seed, block.body, block.collect, block.result, block.comment)
    assert _batchable(block)
    fast, slow = LabState(), LabState()
    _monte_carlo(block, fast, {}, batch=True)
    _monte_carlo(block, slow, {}, batch=False)
    np.testing.assert_allclose(fast.tables["mc"].to_numpy(), slow.tables["mc"].to_numpy(), rtol=1e-8, atol=1e-10)


# --- Sezgi: ekonometrik doğruluk ----------------------------------------------------------

def test_logit_ame_recovers_the_true_average_effect() -> None:
    state = _run(LPM, n=5000)
    x = state.frames["sim"]["x"].to_numpy()
    truth = true_ame_logit(x, 0.0, 1.5)
    assert state.models["ame_logit"].params["x"] == pytest.approx(truth, abs=0.01)
    fitted = state.models["lpm"].fittedvalues
    assert ((fitted < 0) | (fitted > 1)).mean() > 0.1


def test_probit_estimates_beta_over_sigma_and_logit_is_rescaled() -> None:
    state = _run(SCALE, n=10000, sigma=2.0, b1=1.0)
    probit = state.models["probit"].params["x"]
    logit = state.models["logit"].params["x"]
    assert probit == pytest.approx(0.5, abs=0.05)
    assert 1.5 < logit / probit < 1.9
    assert state.models["ame_probit"].params["x"] == pytest.approx(state.models["ame_logit"].params["x"], abs=0.01)


def test_dummy_derivative_differs_from_the_finite_difference_for_large_effects() -> None:
    state = _run(DUMMY, gamma=3.0, alpha=-2.0, n=10000)
    x = state.frames["sim"]["x"].to_numpy()
    difference = state.models["me_fark"].params["d"]
    assert difference == pytest.approx(true_dummy_effect(x, -2.0, 3.0), abs=0.03)
    assert abs(state.models["me_turev"].params["d"] - difference) > 0.05
    small = _run(DUMMY, gamma=0.25, alpha=0.0)
    assert abs(small.models["me_turev"].params["d"] - small.models["me_fark"].params["d"]) < 0.005


def test_censoring_defaults_reproduce_table_62_of_the_notes() -> None:
    state = _run(CENSORING)
    assert 1 - state.frames["sim"]["pozitif"].mean() == pytest.approx(0.335, abs=5e-4)
    rounded = {name: state.models[name].params.round(3).tolist() for name in ("tam", "pozitif_ols", "tobit")}
    assert rounded == {"tam": [1.421, 0.716], "pozitif_ols": [1.484, 0.745], "tobit": [1.006, 1.002]}


def test_selection_defaults_reproduce_table_63_of_the_notes() -> None:
    state = _run(SELECTION)
    assert state.frames["sim"]["s"].mean() == pytest.approx(0.558, abs=5e-4)
    assert state.models["naif"].params.round(3).tolist() == [1.389, 1.849]
    assert state.models["heckman"].params.round(3).tolist() == [0.992, 2.038, 0.632]


def test_heckman_without_exclusion_is_much_more_variable() -> None:
    weak = _run(EXCLUSION, gz=0.0, n=1000).tables["mc"]
    strong = _run(EXCLUSION, gz=0.6, n=1000).tables["mc"]
    assert weak["b_heckman"].mean() == pytest.approx(2.0, abs=0.05)
    assert weak["b_heckman"].std() > 2 * strong["b_heckman"].std()
    assert weak["b_naif"].mean() < 1.85


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
    compared = 0
    for name, result in state.models.items():
        other = namespace[name]
        np.testing.assert_allclose(other.params.to_numpy(), result.params.to_numpy(), rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(other.bse.to_numpy(), result.bse.to_numpy(), rtol=1e-9, atol=1e-10)
        compared += 1
    for name, table in state.tables.items():
        np.testing.assert_allclose(namespace[name].to_numpy(dtype=float), table.to_numpy(dtype=float),
                                   rtol=1e-9, atol=1e-9)
        compared += 1
    assert compared


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript bulunamadı.")
@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e.key)
def test_generated_r_runs(experiment, tmp_path: Path) -> None:
    script = tmp_path / "deney.R"
    script.write_text(generator(experiment.spec(experiment.defaults()), "R").script(), encoding="utf-8")
    result = subprocess.run(["Rscript", str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=600,
                            env=dict(os.environ, LANG="C.UTF-8", LC_ALL="C.UTF-8"))
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda e: e.key)
def test_generated_stata_is_well_formed(experiment) -> None:
    code = generator(experiment.spec(experiment.defaults()), "Stata").script()
    assert "version 14" in code and code.count("{") == code.count("}")
    for line in code.splitlines():
        if not line.strip().startswith("*"):
            assert line.count('"') % 2 == 0, line


def test_code_conventions_for_binary_and_censored_models() -> None:
    from core.labs.konu05 import KONU05_LAB
    from core.labs.konu06 import KONU06_LAB

    python = generator(KONU05_LAB, "Python").script()
    assert 'cov_type="HC0"' in python and "def ortalama_marjinal_etkiler(" in python
    r_code = generator(KONU05_LAB, "R").script()
    assert "dayanikli_vcov <- function(model)" in r_code and 'binomial(link = "probit")' in r_code
    assert not any("sandwich::sandwich(" in line for line in r_code.splitlines() if not line.lstrip().startswith("#"))
    stata = generator(KONU05_LAB, "Stata").script()
    assert "logit married age education i.race4 i.hisp i.region, vce(robust)" in stata
    assert "margins, dydx(age education race4 hisp region) post" in stata
    assert "recode race (1=1) (2=2) (4=3) (else=4), generate(race4)" in stata
    stata6 = generator(KONU06_LAB, "Stata").script().replace("///\n    ", "")
    assert re.search(r"tobit received_k .*, ll\(0\)", stata6)
    assert "capture scalar tobit_sigma = _b[/sigma]" in stata6
    assert re.search(r"qreg received_k .*, quantile\(0\.5\)", stata6)
    for line in generator(KONU06_LAB, "Stata").script().splitlines():
        match = re.search(r"scalar (\w+) =", line)
        if match and "`" not in match.group(1):
            assert len(match.group(1)) <= 32, line
    r6 = generator(KONU06_LAB, "R").script()
    assert "AER::tobit(" in r6 and "quantreg::rq(" in r6 and "left = 0" in r6


# --- Kendini sına: yazım çeşitleri ----------------------------------------------------------

@pytest.mark.parametrize(("questions", "key", "right", "wrong"), [
    (Q5, "f01", ["p*(1-p)*b", "p(1-p)β", "b p - b p^2", "βp(1−p)"], ["p*b", "(1-p)*b", "p*(1-p)"]),
    (Q5, "f02", ["b/s", "β/σ", "beta/sigma"], ["b*s", "s/b", "b"]),
    (Q5, "f03", ["exp(v1)/(1+exp(v1)+exp(v2))", "exp(v_1)/(exp(v_1) + exp(v_2) + 1)"],
     ["exp(v1)/(exp(v1)+exp(v2))", "v1/(1+v1+v2)"]),
    (Q5, "f04", ["S/H^2", "S/(H*H)", "(1/H)*S*(1/H)"], ["S/H", "H/S^2", "1/H"]),
    (Q5, "f05", ["b*m", "βμ", "μ β"], ["b", "m", "b*exp(m)"]),
    (Q6, "f01", ["b*f/s", "β φ/σ", "(b/s)*f"], ["b*f", "b/s", "f/s"]),
    (Q6, "f02", ["m + c*l", "μ + σ_eu λ", "c*l + m"], ["m + l", "m*c*l", "m - c*l"]),
    (Q6, "f03", ["r*s", "ρ σ_e", "rho*sigma"], ["r", "r/s", "s"]),
    (Q6, "f04", ["1 - F", "1-Φ"], ["F", "F - 1", "1/F"]),
    (Q6, "f05", ["a + d + k - g", "a+d+k-g", "(a + d + k) - g"], ["a + d + k", "a + d + k + g", "a - g"]),
])
def test_equation_variants(questions, key, right, wrong) -> None:
    for text in right:
        assert grade(questions[key], text).correct, text
    for text in wrong:
        assert not grade(questions[key], text).correct, text


@pytest.mark.parametrize(("questions", "key", "right", "wrong"), [
    (Q5, "b01", ["sınırlı", "Sınırlı", "limited"], ["sürekli", "ikili"]),
    (Q5, "b02", ["yoğunluk", "Yoğunluk fonksiyonu", "density"], ["dağılım", "Φ", "φ"]),
    (Q5, "b03", ["delta", "Delta yöntemi"], ["bootstrap", "Newton"]),
    (Q5, "b04", ["sonlu", "Sonlu"], ["türev", "marjinal"]),
    (Q5, "b05", ["0,82", "0.82", "0,818"], ["0,11", "82"]),
    (Q6, "b01", ["kesilmiş", "truncated", "Kesilmiş"], ["sansürlü", "seçilmiş"]),
    (Q6, "b02", ["ters Mills", "Ters Mills", "inverse Mills"], ["odds", "Probit"]),
    (Q6, "b03", ["dışlama", "Dışlama", "exclusion"], ["ilgililik", "seçim"]),
    (Q6, "b04", ["3,60", "3.6", "3,604"], ["7,72", "9,81"]),
    (Q6, "b05", ["Probit", "probit"], ["Logit", "OLS"]),
])
def test_fill_variants(questions, key, right, wrong) -> None:
    for text in right:
        assert grade(questions[key], (text,)).correct, text
    for text in wrong:
        assert not grade(questions[key], (text,)).correct, text


# --- Gerçek veriyle uygulama ekranı ------------------------------------------------------------

def _data_path(dataset: str) -> Path | None:
    value = os.environ.get(f"IKT807_HANSEN_{dataset.upper()}_PATH")
    if not value and dataset == "cps09mar":
        value = os.environ.get("IKT807_CPS_PATH")
    return Path(value) if value and Path(value).is_file() else None


@pytest.mark.skipif(_data_path("cps09mar") is None, reason="Gerçek Hansen cps09mar dosyası verilmedi.")
def test_konu05_table_51_step_matches_notes_in_the_app() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.radio(key="topic_selector").set_value("konu05").run()
    app.segmented_control(key="konu05_lab_step").set_value(4).run()
    assert not app.exception
    comparisons = [frame.value for frame in app.dataframe if "Durum" in frame.value.columns]
    assert len(comparisons) == 1 and len(comparisons[0]) == 36
    assert set(comparisons[0]["Durum"]) == {"✓"}


@pytest.mark.skipif(_data_path("chj2004") is None, reason="Gerçek Hansen CHJ2004 dosyası verilmedi.")
def test_konu06_curve_step_matches_notes_in_the_app() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.radio(key="topic_selector").set_value("konu06").run()
    app.segmented_control(key="konu06_lab_step").set_value(3).run()
    assert not app.exception
    comparisons = [frame.value for frame in app.dataframe if "Durum" in frame.value.columns]
    assert len(comparisons) == 1 and set(comparisons[0]["Durum"]) == {"✓"}
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["P(Y>0), Tobit"] == "0,579"
    assert metrics["Pozitif payı, veri"] == "0,824"
