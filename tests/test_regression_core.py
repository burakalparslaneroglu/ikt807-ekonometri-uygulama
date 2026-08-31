from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from core.diagnostics import influence_diagnostics
from core.functional_forms import (
    exact_percent_change,
    interaction_slope,
    quadratic_marginal_effect,
)
from core.inference import (
    cluster_covariance,
    delta_method,
    hc1_covariance,
)
from core.ols import fit_ols, fwl_coefficient, projection_diagnostics
from core.simulation import WageDGPConfig, add_influential_observation, simulate_wage_data


def _sample_frame():
    return simulate_wage_data(
        WageDGPConfig(
            nobs=600,
            seed=807,
            heteroskedasticity=1.2,
            cluster_correlation=0.45,
            clusters=30,
        )
    )


def test_ols_matches_statsmodels_and_projection_invariants() -> None:
    frame = _sample_frame()
    columns = ["education", "experience", "experience2_100", "female"]
    fit = fit_ols(frame["lwage"], frame[columns], tuple(columns))
    benchmark = sm.OLS(frame["lwage"], sm.add_constant(frame[columns])).fit()

    np.testing.assert_allclose(fit.coefficients, benchmark.params, rtol=1e-10)
    diagnostics = projection_diagnostics(fit)
    assert abs(diagnostics.mean_residual) < 1e-12
    assert abs(diagnostics.max_regressor_residual_product) < 1e-9
    assert abs(diagnostics.fitted_residual_product) < 1e-9


def test_fwl_matches_multiple_regression_coefficient() -> None:
    frame = _sample_frame()
    result = fwl_coefficient(
        frame["lwage"],
        frame["education"],
        frame[["experience", "experience2_100", "female"]],
    )
    assert result.direct_coefficient == pytest.approx(
        result.residual_regression_coefficient, abs=1e-12
    )


def test_hc1_and_cluster_covariance_match_statsmodels() -> None:
    frame = _sample_frame()
    columns = ["education", "experience", "experience2_100", "female"]
    fit = fit_ols(frame["lwage"], frame[columns], tuple(columns))
    benchmark = sm.OLS(frame["lwage"], sm.add_constant(frame[columns])).fit()

    np.testing.assert_allclose(
        hc1_covariance(fit),
        benchmark.get_robustcov_results(cov_type="HC1").cov_params(),
        rtol=1e-9,
    )
    covariance, group_count = cluster_covariance(fit, frame["cluster"])
    cluster_benchmark = benchmark.get_robustcov_results(
        cov_type="cluster", groups=frame["cluster"]
    )
    np.testing.assert_allclose(covariance, cluster_benchmark.cov_params(), rtol=1e-9)
    assert group_count == frame["cluster"].nunique()


def test_delta_method_and_functional_form_helpers() -> None:
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    assert delta_method([1, 2], covariance) == pytest.approx(np.sqrt(0.44))
    assert exact_percent_change(np.log(1.1)) == pytest.approx(10.0)
    assert quadratic_marginal_effect(0.2, -0.01, 5) == pytest.approx(0.1)
    assert interaction_slope(0.08, -0.02, 1) == pytest.approx(0.06)


def test_influential_observation_has_largest_cook_distance() -> None:
    frame = add_influential_observation(_sample_frame())
    fit = fit_ols(
        frame["lwage"],
        frame[["education", "experience", "experience2_100", "female"]],
        ("education", "experience", "experience2_100", "female"),
    )
    diagnostics = influence_diagnostics(fit)
    assert int(np.argmax(diagnostics.cooks_distance)) == len(frame) - 1


def test_simulation_is_deterministic() -> None:
    config = WageDGPConfig(nobs=120, seed=42)
    first = simulate_wage_data(config)
    second = simulate_wage_data(config)
    assert first.equals(second)
