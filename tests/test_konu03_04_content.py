"""Konu 3 ve 4: soru setlerindeki yazım çeşitleri, Sezgi deneylerinin ekonometrik doğruluğu,
üç dilde kod ve .dta veri okuma."""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from linearmodels.iv import IV2SLS
from streamlit.testing.v1 import AppTest

from core.codegen.base import generator
from core.hansen_data import HansenDataError, load_from_archive, load_from_upload
from core.labs.runner import LabState, coverage_key, execute, fit_iv
from core.labs.sezgi_konu03 import (
    KONU03_EXPERIMENTS,
    LARGE_SAMPLE,
    MEASUREMENT,
    SELECTION,
    attenuation,
    expected_selection,
    ols_limit,
)
from core.labs.sezgi_konu04 import KONU04_EXPERIMENTS, LATE, WALD, WEAK, ate, iv_target
from core.labs.spec import IV, Draw, MonteCarlo, NewSample
from core.quiz.konu03 import KONU03_QUIZ
from core.quiz.konu04 import KONU04_QUIZ
from core.quiz.model import grade

EXPERIMENTS = KONU03_EXPERIMENTS + KONU04_EXPERIMENTS
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
Q3 = {q.key: q for q in KONU03_QUIZ.questions}
Q4 = {q.key: q for q in KONU04_QUIZ.questions}


# --- Kendini sına: yazım çeşitleri ----------------------------------------------------

@pytest.mark.parametrize(("questions", "key", "right", "wrong"), [
    (Q3, "f01", ["d*y1 + (1-d)*y0", "D*Y(1) + (1-D)*Y(0)", "Y(0) + D(Y(1) - Y(0))", "DY1 + Y0 - DY0"],
     ["d*y1 + d*y0", "y1 - y0", "d*(y1 + y0)"]),
    (Q3, "f02", ["(ud - us)/(bd + bs)", "(u_d - u_s)/(β_d + β_s)"], ["(ud - us)/(bd - bs)", "(ud + us)/(bd + bs)"]),
    (Q3, "f03", ["b*vz/(vz + vu)", "β Var(Z)/(Var(Z)+Var(u))"], ["b*vu/(vz + vu)", "b/(1 + vu)", "b*vz/vu"]),
    (Q3, "f04", ["-b*vu", "-β Var(u)", "-(b vu)"], ["b*vu", "-vu", "-b*b*vu"]),
    (Q3, "f05", ["b + c/v", "β + Cov(X,e)/Var(X)"], ["b - c/v", "c/v", "b + v/c"]),
    (Q4, "f01", ["l/p", "λ/π", "lambda/pi"], ["p/l", "l*p", "l - p"]),
    (Q4, "f02", ["b + k/p", "β + κ/π", "beta + kappa/pi"], ["b + k*p", "b + k", "(b + k)/p"]),
    (Q4, "f03", ["(p/s)^2", "p^2/s^2", "(π/SE)^2"], ["p/s", "(s/p)^2", "p^2/s"]),
    (Q4, "f04", ["pa*ta + pc*tc + pn*tn", "p_a τ_a + p_c τ_c + p_n τ_n"], ["tc", "pc*tc", "(ta + tc + tn)/3"]),
    (Q4, "f05", ["b*v + e", "βv + e", "e + v*beta"], ["v + e", "b*(v + e)", "b*v"]),
])
def test_equation_variants(questions, key, right, wrong) -> None:
    for text in right:
        assert grade(questions[key], text).correct, text
    for text in wrong:
        assert not grade(questions[key], text).correct, text


@pytest.mark.parametrize(("questions", "key", "right", "wrong"), [
    (Q3, "b01", ["sıfıra", "Sıfıra", "0'a"], ["yukarı", "bire"]),
    (Q3, "b02", ["Y(1)-Y(0)", "Y(1) − Y(0)", "Y1 - Y0"], ["Y(1)", "Y(0)-Y(1)"]),
    (Q3, "b03", ["okul", "Okul"], ["öğrenci", "sınıf"]),
    (Q3, "b04", ["ortak destek", "Örtüşme", "overlap"], ["rastgele atama", "bağımsızlık"]),
    (Q3, "b05", ["1,574", "1.57", "1,6"], ["-1,574", "0,016", "15,7"]),
    (Q4, "b01", ["maliyetini", "Maliyeti"], ["getirisini", "süresini"]),
    (Q4, "b02", ["uyumlu", "Compliers", "uyumlular"], ["her zaman alan", "defier"]),
    (Q4, "b03", ["az", "daha az"], ["fazla", "eşit"]),
    (Q4, "b04", ["Ze", "Z e", "Z(Y − X'β)", "Z·e"], ["Xe", "e", "Z"]),
    (Q4, "b05", ["OLS", "ols", "OLS'nin"], ["sıfır", "IV"]),
])
def test_fill_variants(questions, key, right, wrong) -> None:
    for text in right:
        assert grade(questions[key], (text,)).correct, text
    for text in wrong:
        assert not grade(questions[key], (text,)).correct, text


# --- Sezgi: ekonometrik doğruluk ---------------------------------------------------------

def _run(experiment, **overrides) -> LabState:
    p = experiment.defaults() | overrides
    state = LabState()
    for op in experiment.build(p):
        execute(op, state, {})
    return state


def test_selection_decomposition_is_exact_and_random_assignment_removes_bias() -> None:
    state = _run(SELECTION, n=5000, s=1.0)
    observed = state.models["gozlenen"].params["d"]
    selection = state.models["secim"].params["d"]
    assert observed - selection == pytest.approx(1.0, abs=1e-10)
    assert selection == pytest.approx(expected_selection(1.0), rel=0.1)
    random = _run(SELECTION, n=5000, s=0.0)
    assert abs(random.models["secim"].params["d"]) < 0.15
    assert random.models["gozlenen"].params["d"] == pytest.approx(1.0, abs=0.15)


def test_measurement_error_attenuates_towards_the_reliability_ratio() -> None:
    state = _run(MEASUREMENT, n=5000, var_u=1.0)
    assert state.models["olculen"].params["x"] == pytest.approx(attenuation(1.0), abs=0.05)
    assert state.models["gercek"].params["z"] == pytest.approx(1.0, abs=0.05)


def test_large_sample_concentrates_on_the_wrong_target() -> None:
    state = _run(LARGE_SAMPLE, n=5000, a=0.1)
    table = state.tables["mc"]
    assert table["b_kisa"].mean() == pytest.approx(ols_limit(0.1), abs=0.01)
    assert table["b_uzun"].mean() == pytest.approx(2.0, abs=0.01)
    assert state.scalars[coverage_key("mc", "b_kisa")] < 0.1
    assert 0.9 < state.scalars[coverage_key("mc", "b_uzun")] < 0.99
    small = _run(LARGE_SAMPLE, n=100, a=0.1)
    assert small.scalars[coverage_key("mc", "b_kisa")] > 0.7


def test_wald_ratio_equals_2sls_and_targets_beta_plus_kappa_over_pi() -> None:
    state = _run(WALD, n=5000, pi=0.4, kappa=0.2)
    assert state.scalars["wald"] == pytest.approx(float(state.models["iv"].params["x"]), abs=1e-10)
    assert state.models["iv"].params["x"] == pytest.approx(iv_target(0.4, 0.2), abs=0.3)
    clean = _run(WALD, n=5000, pi=1.0, kappa=0.0)
    assert clean.models["iv"].params["x"] == pytest.approx(1.0, abs=0.15)


def test_weak_instrument_distribution_and_coverage() -> None:
    strong = _run(WEAK, pi=0.5, n=2000, rho=0.9)
    assert strong.tables["mc"]["b_2sls"].median() == pytest.approx(1.0, abs=0.05)
    assert 0.9 < strong.scalars[coverage_key("mc", "b_2sls")] < 0.99
    weak = _run(WEAK, pi=0.02, n=500, rho=0.9)
    assert weak.tables["mc"]["b_2sls"].median() > 1.1
    assert weak.tables["mc"]["F"].median() < 3


def test_late_is_the_complier_effect_not_the_ate() -> None:
    p = {"n": 10000, "p_c": 0.5, "tau_c": 0.5}
    state = _run(LATE, **p)
    iv = state.models["iv"]
    assert iv.params["d"] == pytest.approx(0.5, abs=3 * float(iv.bse["d"]))
    assert state.models["ilk"].params["z"] == pytest.approx(0.5, abs=0.03)
    assert ate(LATE.defaults() | p) == pytest.approx(0.25 * 1.5 + 0.5 * 0.5 + 0.25 * 0.5)


@pytest.mark.parametrize("experiment", (LARGE_SAMPLE, WEAK), ids=lambda e: e.key)
def test_vectorized_monte_carlo_matches_the_loop(experiment) -> None:
    from core.labs.runner import _batchable, _monte_carlo

    block = next(op for op in experiment.build(experiment.defaults()) if isinstance(op, MonteCarlo))
    assert _batchable(block)
    fast, slow = LabState(), LabState()
    _monte_carlo(block, fast, {}, batch=True)
    _monte_carlo(block, slow, {}, batch=False)
    np.testing.assert_allclose(fast.tables["mc"].to_numpy(), slow.tables["mc"].to_numpy(), rtol=1e-8, atol=1e-10)
    assert fast.scalars == slow.scalars


def test_mixed_distributions_use_the_loop() -> None:
    from core.labs.runner import _batchable

    body = (NewSample("s", 10, None), Draw("s", "a", "uniform", 0, 1, ""), Draw("s", "b", "normal", 0, 1, ""))
    assert not _batchable(MonteCarlo("s", 5, 1, body, (), "mc", ""))


def test_numpy_2sls_matches_linearmodels() -> None:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"z": rng.normal(size=800), "w": rng.normal(size=800), "v": rng.normal(size=800)})
    frame["x"] = 0.3 * frame.z + 0.5 * frame.w + frame.v
    frame["y"] = 1 + frame.x - frame.w + 0.6 * frame.v + rng.normal(size=800)
    ours = fit_iv(IV("iv", "f", "y", ("x",), ("z",), ("w",)), frame)
    theirs = IV2SLS.from_formula("y ~ 1 + w + [x ~ z]", frame).fit(cov_type="robust", debiased=True)
    np.testing.assert_allclose(ours.params.to_numpy(), theirs.params[ours.params.index].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(ours.bse.to_numpy(), theirs.std_errors[ours.bse.index].to_numpy(), atol=1e-12)


# --- Üç dilde kod ---------------------------------------------------------------------------

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
        errors = other.std_errors if hasattr(other, "std_errors") else other.bse
        np.testing.assert_allclose(other.params.to_numpy(), result.params.to_numpy(), rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(errors.to_numpy(), result.bse.to_numpy(), rtol=1e-9, atol=1e-10)
        compared += 1
    for name, table in state.tables.items():
        np.testing.assert_allclose(namespace[name].to_numpy(dtype=float), table.to_numpy(dtype=float),
                                   rtol=1e-9, atol=1e-9)
        compared += 1
    for name, value in state.scalars.items():
        assert float(namespace[name]) == pytest.approx(value, rel=1e-9, abs=1e-10), name
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
    if "postfile" in code:
        assert "tempfile mc_dosya" in code and "postclose" in code and "save " not in code
        assert "set seed" in code.split("forvalues")[0]


def test_stata_iv_uses_small_sample_scaling_and_r_uses_hc1() -> None:
    from core.labs.konu04 import KONU04_LAB

    stata = generator(KONU04_LAB, "Stata").script()
    assert re.search(r"ivregress 2sls .*vce\(robust\) small", stata.replace("///\n    ", ""))
    r_code = generator(KONU04_LAB, "R").script()
    assert "AER::ivreg(" in r_code and 'sandwich::vcovHC(iv, type = "HC1")' in r_code
    python = generator(KONU04_LAB, "Python").script()
    assert 'fit(cov_type="robust", debiased=True)' in python


# --- .dta okuma -----------------------------------------------------------------------------

def _dta(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_stata(buffer, write_index=False, version=118)
    return buffer.getvalue()


def _ddk_like(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "schoolid": rng.integers(1, 9, rows).astype(float), "tracking": rng.integers(0, 2, rows).astype(float),
        "totalscore": rng.normal(20, 5, rows), "std_mark": rng.normal(size=rows), "girl": rng.integers(0, 2, rows),
        "agetest": rng.normal(9, 1, rows), "sbm": rng.integers(0, 2, rows), "etpteacher": rng.integers(0, 2, rows),
        "lowstream": rng.integers(0, 2, rows), "SDstream_std_mark": rng.normal(size=rows),
    })


def test_stata_archive_member_is_read_with_lowercase_names_and_missing_values() -> None:
    frame = _ddk_like()
    frame.loc[:4, "std_mark"] = np.nan
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("Econometrics Data/DDK2011/DDK2011.dta", _dta(frame))
        bundle.writestr("Econometrics Data/cps09mar/cps09mar.txt", b"1 2\r\n")
    loaded = load_from_archive("ddk2011", archive.getvalue())
    assert "sdstream_std_mark" in loaded.frame.columns
    assert loaded.frame["std_mark"].isna().sum() == 5
    assert loaded.matches_hansen is False


def test_upload_rejects_a_file_without_required_columns() -> None:
    with pytest.raises(HansenDataError, match="lwage76"):
        load_from_upload("card1995", _dta(pd.DataFrame({"x": [1.0, 2.0]})), "Card1995.dta")


def _data_path(dataset: str) -> Path | None:
    value = os.environ.get(f"IKT807_HANSEN_{dataset.upper()}_PATH")
    return Path(value) if value and Path(value).is_file() else None


@pytest.mark.skipif(_data_path("ddk2011") is None, reason="Gerçek Hansen DDK2011 dosyası verilmedi.")
def test_konu03_balance_step_matches_notes_in_the_app() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.radio(key="topic_selector").set_value("konu03").run()
    app.segmented_control(key="konu03_lab_step").set_value(2).run()
    assert not app.exception
    comparisons = [frame.value for frame in app.dataframe if "Durum" in frame.value.columns]
    assert len(comparisons) == 1 and len(comparisons[0]) == 15
    assert set(comparisons[0]["Durum"]) == {"✓"}


@pytest.mark.skipif(_data_path("card1995") is None, reason="Gerçek Hansen Card1995 dosyası verilmedi.")
def test_konu04_chain_step_matches_notes_in_the_app() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.radio(key="topic_selector").set_value("konu04").run()
    app.segmented_control(key="konu04_lab_step").set_value(2).run()
    assert not app.exception
    comparisons = [frame.value for frame in app.dataframe if "Durum" in frame.value.columns]
    assert len(comparisons) == 1 and set(comparisons[0]["Durum"]) == {"✓"}
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["İlk aşama F (robust t²)"] == "14,14"
    assert metrics["Oran λ̂/π̂ (dolaylı EKK)"] == "0,1315"
    assert metrics["2SLS kesin yüzde etki"] == "%14,1"
