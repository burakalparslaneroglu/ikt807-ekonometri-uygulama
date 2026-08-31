import numpy as np

from core.bootstrap import bootstrap_ols_coefficient
from core.rdd import (
    RDDDGPConfig,
    density_ratio_near_cutoff,
    fit_fuzzy_rdd,
    fit_sharp_rdd,
    placebo_estimates,
    simulate_rdd_data,
)
from core.resampling import draw_resample_indices, resample_frequencies
from core.simulation import WageDGPConfig, simulate_wage_data


def test_sharp_rdd_recovers_right_minus_left_effect() -> None:
    data = simulate_rdd_data(
        RDDDGPConfig(nobs=8000, seed=11, treatment_effect=-2.2)
    )
    fit = fit_sharp_rdd(data["outcome_sharp"], data["running"], bandwidth=3.5)
    assert abs(fit.estimate - (-2.2)) < 0.14
    assert fit.right_limit - fit.left_limit == fit.estimate
    assert fit.standard_error > 0


def test_rdd_effective_sample_grows_with_bandwidth() -> None:
    data = simulate_rdd_data(RDDDGPConfig(nobs=1800, seed=12))
    narrow = fit_sharp_rdd(data["outcome_sharp"], data["running"], bandwidth=2.0)
    wide = fit_sharp_rdd(data["outcome_sharp"], data["running"], bandwidth=6.0)
    assert wide.n_left > narrow.n_left
    assert wide.n_right > narrow.n_right
    assert wide.effective_n > narrow.effective_n


def test_fuzzy_rdd_local_wald_recovers_treatment_effect() -> None:
    data = simulate_rdd_data(
        RDDDGPConfig(
            nobs=12000,
            seed=13,
            treatment_effect=-1.8,
            first_stage_jump=0.6,
        )
    )
    result = fit_fuzzy_rdd(
        data["outcome_fuzzy"],
        data["treatment"],
        data["running"],
        bandwidth=4.0,
    )
    assert result.first_stage > 0.45
    assert abs(result.local_wald - (-1.8)) < 0.22
    np.testing.assert_allclose(
        result.local_wald, result.reduced_form / result.first_stage
    )


def test_density_ratio_detects_running_variable_bunching() -> None:
    clean = simulate_rdd_data(
        RDDDGPConfig(nobs=10000, seed=14, manipulation_strength=0)
    )
    manipulated = simulate_rdd_data(
        RDDDGPConfig(nobs=10000, seed=14, manipulation_strength=0.8)
    )
    clean_ratio, _, _ = density_ratio_near_cutoff(clean["running"], window=1.5)
    manipulated_ratio, _, _ = density_ratio_near_cutoff(
        manipulated["running"], window=1.5
    )
    assert abs(clean_ratio - 1) < 0.15
    assert manipulated_ratio > clean_ratio + 1.0


def test_placebo_cutoffs_do_not_reproduce_true_jump() -> None:
    data = simulate_rdd_data(RDDDGPConfig(nobs=7000, seed=15, treatment_effect=-2))
    placebos = placebo_estimates(
        data["outcome_sharp"],
        data["running"],
        (-6.0, 6.0),
        bandwidth=2.0,
    )
    assert all(abs(fit.estimate) < 0.35 for fit in placebos)


def test_observation_resampling_is_seeded_and_uses_replacement() -> None:
    first = draw_resample_indices(100, seed=20)
    second = draw_resample_indices(100, seed=20)
    np.testing.assert_array_equal(first, second)
    frequencies = resample_frequencies(first, 100)
    assert frequencies.sum() == 100
    assert (frequencies == 0).any()
    assert (frequencies > 1).any()


def test_cluster_resampling_keeps_each_sampled_cluster_whole() -> None:
    groups = np.repeat(np.arange(8), 5)
    indices = draw_resample_indices(groups.size, seed=21, groups=groups)
    frequencies = resample_frequencies(indices, groups.size)
    for group in np.unique(groups):
        group_frequencies = frequencies[groups == group]
        assert np.unique(group_frequencies).size == 1


def test_pairs_bootstrap_is_deterministic_and_matches_hc1_scale() -> None:
    frame = simulate_wage_data(
        WageDGPConfig(nobs=700, seed=22, heteroskedasticity=1.2)
    )
    features = ("education", "experience", "experience2_100", "female")
    first = bootstrap_ols_coefficient(
        frame["lwage"],
        frame[list(features)],
        features,
        "education",
        repetitions=400,
        seed=23,
        method="Pairs",
    )
    second = bootstrap_ols_coefficient(
        frame["lwage"],
        frame[list(features)],
        features,
        "education",
        repetitions=400,
        seed=23,
        method="Pairs",
    )
    np.testing.assert_allclose(first.draws, second.draws)
    relative_gap = abs(first.bootstrap_standard_error - first.analytic_standard_error)
    assert relative_gap / first.analytic_standard_error < 0.2


def test_wild_bootstrap_intervals_are_finite_and_ordered() -> None:
    frame = simulate_wage_data(
        WageDGPConfig(nobs=500, seed=24, heteroskedasticity=1.5)
    )
    features = ("education", "experience", "experience2_100", "female")
    result = bootstrap_ols_coefficient(
        frame["lwage"],
        frame[list(features)],
        features,
        "education",
        repetitions=300,
        seed=25,
        method="Wild",
    )
    for low, high in (
        result.normal_interval,
        result.percentile_interval,
        result.percentile_t_interval,
    ):
        assert np.isfinite((low, high)).all()
        assert low < high
    assert result.monte_carlo_standard_error > 0
