from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

from core.causal import (
    EndogeneityConfig,
    TrialConfig,
    decompose_observed_difference,
    endogeneity_estimates,
    endogeneity_probability_limit,
    simulate_cluster_trial,
    simulate_selection_data,
)
from core.inference import classic_covariance
from core.iv import (
    IVConfig,
    conditional_wald_ratio,
    fit_2sls,
    naive_second_stage_fit,
    simulate_iv_data,
)


def test_selection_decomposition_identity_and_randomization() -> None:
    selected = simulate_selection_data(selection_strength=1.4, randomized=False)
    decomposition = decompose_observed_difference(selected)
    assert decomposition.observed_difference == pytest.approx(
        decomposition.treatment_on_treated + decomposition.selection_bias
    )
    assert abs(decomposition.selection_bias) > 0.5

    randomized = decompose_observed_difference(
        simulate_selection_data(nobs=50_000, randomized=True)
    )
    assert randomized.selection_bias == pytest.approx(0.0, abs=0.03)


def test_endogeneity_converges_to_biased_probability_limit() -> None:
    config = EndogeneityConfig(nobs=80_000, seed=803, confounding_strength=0.8)
    omitted, controlled = endogeneity_estimates(config)
    assert omitted == pytest.approx(endogeneity_probability_limit(config), abs=0.025)
    assert controlled == pytest.approx(config.structural_effect, abs=0.02)


def test_cluster_trial_is_deterministic_and_randomized_by_school() -> None:
    config = TrialConfig(students=1200, schools=60, seed=833)
    first = simulate_cluster_trial(config)
    second = simulate_cluster_trial(config)
    assert first.equals(second)
    assert first.groupby("schoolid")["tracking"].nunique().max() == 1


def _iv_frame():
    return simulate_iv_data(
        IVConfig(nobs=1800, seed=804, instrument_strength=0.45)
    )


def test_2sls_and_robust_covariance_match_linearmodels() -> None:
    frame = _iv_frame()
    fit = fit_2sls(
        frame["lwage"],
        frame["education"],
        frame["instrument"],
        exogenous=frame[["control"]],
        endogenous_name="education",
        exogenous_names=("control",),
    )
    exogenous = sm.add_constant(frame[["control"]])
    benchmark = IV2SLS(
        frame["lwage"],
        exogenous,
        frame[["education"]],
        frame[["instrument"]],
    ).fit(cov_type="robust", debiased=True)

    np.testing.assert_allclose(fit.coefficients, benchmark.params, rtol=1e-9)
    np.testing.assert_allclose(fit.covariance, benchmark.cov, rtol=1e-8)


def test_conditional_wald_equals_just_identified_2sls() -> None:
    frame = _iv_frame()
    first_stage, reduced_form, wald = conditional_wald_ratio(
        frame["lwage"],
        frame["education"],
        frame["instrument"],
        exogenous=frame[["control"]],
    )
    fit = fit_2sls(
        frame["lwage"],
        frame["education"],
        frame["instrument"],
        exogenous=frame[["control"]],
        endogenous_name="education",
        exogenous_names=("control",),
    )
    assert first_stage > 0
    assert reduced_form / first_stage == pytest.approx(wald)
    assert wald == pytest.approx(fit.coefficient("education"), abs=1e-10)


def test_naive_second_stage_standard_error_is_not_iv_standard_error() -> None:
    frame = _iv_frame()
    fit = fit_2sls(
        frame["lwage"],
        frame["education"],
        frame["instrument"],
        exogenous=frame[["control"]],
        endogenous_name="education",
        exogenous_names=("control",),
    )
    naive = naive_second_stage_fit(
        frame["lwage"],
        fit,
        exogenous=frame[["control"]],
        exogenous_names=("control",),
    )
    naive_covariance = classic_covariance(naive)
    naive_se = float(np.sqrt(naive_covariance[-1, -1]))
    assert naive.coefficient("Tahmin edilen endojen") == pytest.approx(
        fit.coefficient("education"), abs=1e-10
    )
    assert naive_se != pytest.approx(fit.standard_error("education"), rel=0.05)
