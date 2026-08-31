from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from core.discrete import BinaryDGPConfig, fit_binary_model, simulate_binary_data
from core.limited_outcomes import (
    HeckmanDGPConfig,
    TobitDGPConfig,
    fit_tobit,
    heckman_two_step,
    simulate_selection_data,
    simulate_tobit_data,
    tobit_expectations,
    tobit_marginal_effects,
)
from core.marginal_effects import average_marginal_effect, finite_difference


def _binary_frame():
    return simulate_binary_data(BinaryDGPConfig(nobs=3000, seed=805))


@pytest.mark.parametrize("model_name", ["LPM", "Logit", "Probit"])
def test_binary_model_probabilities_and_convergence(model_name: str) -> None:
    frame = _binary_frame()
    fit = fit_binary_model(
        frame["married"],
        frame[["age", "education", "metro"]],
        ("age", "education", "metro"),
        model_name,
    )
    assert fit.converged
    assert fit.predicted_probabilities.shape == (len(frame),)
    if model_name != "LPM":
        assert np.all((fit.predicted_probabilities >= 0) & (fit.predicted_probabilities <= 1))


def test_logit_average_marginal_effect_matches_statsmodels() -> None:
    frame = _binary_frame()
    fit = fit_binary_model(
        frame["married"],
        frame[["age", "education", "metro"]],
        ("age", "education", "metro"),
        "Logit",
    )
    effect = average_marginal_effect(fit, "age")
    benchmark = sm.Logit(
        frame["married"],
        sm.add_constant(frame[["age", "education", "metro"]]),
    ).fit(disp=False).get_margeff(at="overall", method="dydx")
    assert effect.effect == pytest.approx(benchmark.margeff[0], rel=1e-7)
    assert effect.standard_error > 0


def test_dummy_finite_difference_uses_probability_scale() -> None:
    frame = _binary_frame()
    fit = fit_binary_model(
        frame["married"],
        frame[["age", "education", "metro"]],
        ("age", "education", "metro"),
        "Logit",
    )
    difference = finite_difference(fit, "metro")
    low = fit.design_matrix.copy()
    high = fit.design_matrix.copy()
    index = fit.coefficient_names.index("metro")
    low[:, index] = 0
    high[:, index] = 1
    expected = np.mean(fit.predict(high) - fit.predict(low))
    assert difference.effect == pytest.approx(expected)
    assert difference.standard_error > 0


def test_tobit_mle_recovers_known_parameters_and_reports_convergence() -> None:
    config = TobitDGPConfig(nobs=7000, seed=806, intercept=-0.5, slope=1.5, sigma=1.0)
    frame = simulate_tobit_data(config)
    fit = fit_tobit(frame["observed"], frame[["x"]], ("x",))
    assert fit.converged
    assert fit.coefficient("Sabit") == pytest.approx(config.intercept, abs=0.05)
    assert fit.coefficient("x") == pytest.approx(config.slope, abs=0.05)
    assert fit.sigma == pytest.approx(config.sigma, abs=0.05)
    assert fit.gradient_norm < 0.1


def test_tobit_expectations_and_marginal_effects_are_distinct() -> None:
    frame = simulate_tobit_data(TobitDGPConfig(nobs=3000, seed=816))
    fit = fit_tobit(frame["observed"], frame[["x"]], ("x",))
    design = np.array([[1.0, 0.5]])
    expectations = tobit_expectations(fit, design).iloc[0]
    effects = tobit_marginal_effects(fit, "x", design).iloc[0]
    assert expectations["Gizli ortalama"] != pytest.approx(expectations["Gözlenen ortalama"])
    assert effects["Gizli sonuç etkisi"] > effects["Gözlenen ortalama etkisi"]
    assert 0 < expectations["Sansürlenmeme olasılığı"] < 1


def test_heckman_correction_moves_slope_toward_truth() -> None:
    config = HeckmanDGPConfig(
        nobs=12_000,
        seed=866,
        slope=2.0,
        error_correlation=0.65,
        exclusion_strength=0.9,
    )
    frame = simulate_selection_data(config)
    result = heckman_two_step(frame)
    assert result.probit_converged
    assert abs(result.corrected_slope - config.slope) < abs(result.naive_slope - config.slope)
    assert abs(result.mills_coefficient) > 0.1
    assert 0.35 < result.selection_rate < 0.75
