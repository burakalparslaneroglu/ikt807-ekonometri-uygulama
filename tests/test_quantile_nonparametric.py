import numpy as np

from core.cross_validation import cross_validated_predictions, select_bandwidth
from core.nonparametric import (
    NonparametricDGPConfig,
    effective_sample_size,
    kernel_weights,
    local_polynomial_predict,
    simulate_nonlinear_data,
    spline_series_predict,
)
from core.partialling import partialling_out
from core.quantile import (
    QuantileDGPConfig,
    check_loss,
    fit_quantile,
    fit_quantile_linear_program,
    quantile_profile,
    simulate_quantile_data,
)


def test_check_loss_is_asymmetric_and_median_is_half_absolute_loss() -> None:
    residuals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(check_loss(residuals, 0.5), 0.5 * np.abs(residuals))
    assert check_loss(np.array([-2.0]), 0.8)[0] < check_loss(np.array([2.0]), 0.8)[0]


def test_linear_program_quantile_matches_statsmodels_solution() -> None:
    data = simulate_quantile_data(QuantileDGPConfig(nobs=240, seed=17))
    benchmark = fit_quantile(data["y"], data["x"], ("x",), 0.35)
    transparent = fit_quantile_linear_program(data["y"], data["x"], ("x",), 0.35)
    np.testing.assert_allclose(transparent, benchmark.coefficients, atol=2e-4)


def test_quantile_profile_recovers_scale_driven_slope_pattern() -> None:
    data = simulate_quantile_data(
        QuantileDGPConfig(nobs=1800, seed=21, scale_slope=0.22)
    )
    fits = quantile_profile(data["y"], data["x"], ("x",), (0.1, 0.5, 0.9))
    slopes = [fit.coefficient("x") for fit in fits]
    assert slopes[0] < slopes[1] < slopes[2]
    assert all(fit.converged for fit in fits)


def test_kernel_weights_sum_to_one_and_effective_neighborhood_grows() -> None:
    x = np.linspace(-3, 3, 101)
    narrow = kernel_weights(x, 0.0, 0.35)
    wide = kernel_weights(x, 0.0, 1.2)
    np.testing.assert_allclose(narrow.sum(), 1.0)
    assert (narrow >= 0).all()
    assert effective_sample_size(wide) > effective_sample_size(narrow)


def test_local_linear_removes_linear_boundary_bias() -> None:
    x = np.linspace(0, 10, 101)
    y = 2 + 1.5 * x
    constant = local_polynomial_predict(x, y, [0.0], 1.2, degree=0)
    linear = local_polynomial_predict(x, y, [0.0], 1.2, degree=1)
    assert abs(linear.predictions[0] - 2.0) < 1e-10
    assert abs(constant.predictions[0] - 2.0) > 0.2


def test_spline_series_tracks_known_nonlinear_curve() -> None:
    data = simulate_nonlinear_data(NonparametricDGPConfig(nobs=600, seed=31, noise_scale=0.2))
    prediction = spline_series_predict(data["x"], data["y"], data["x"], n_knots=8)
    rmse = np.sqrt(np.mean((prediction - data["conditional_mean"]) ** 2))
    assert rmse < 0.15


def test_cross_validation_is_deterministic_and_excludes_validation_outcome() -> None:
    data = simulate_nonlinear_data(NonparametricDGPConfig(nobs=180, seed=41))
    first = select_bandwidth(data["x"], data["y"], [0.25, 0.5, 0.9], folds=4, seed=9)
    second = select_bandwidth(data["x"], data["y"], [0.25, 0.5, 0.9], folds=4, seed=9)
    np.testing.assert_allclose(first.mean_squared_errors, second.mean_squared_errors)

    baseline = cross_validated_predictions(data["x"], data["y"], 0.5, folds=4, seed=9)
    changed_y = data["y"].copy()
    changed_y[0] += 1000
    changed = cross_validated_predictions(data["x"], changed_y, 0.5, folds=4, seed=9)
    np.testing.assert_allclose(changed[0], baseline[0])


def test_group_cross_validation_keeps_entire_group_out() -> None:
    rng = np.random.default_rng(51)
    groups = np.repeat(np.arange(12), 12)
    x = rng.uniform(0, 10, groups.size)
    y = np.sin(x) + rng.normal(scale=0.2, size=x.size)
    baseline = cross_validated_predictions(x, y, 0.8, folds=4, groups=groups)
    changed_y = y.copy()
    changed_y[groups == 0] += 1000
    changed = cross_validated_predictions(x, changed_y, 0.8, folds=4, groups=groups)
    np.testing.assert_allclose(changed[groups == 0], baseline[groups == 0])


def test_partialling_out_recovers_partially_linear_target() -> None:
    rng = np.random.default_rng(61)
    z = rng.uniform(-2.5, 2.5, 1800)
    treatment = np.sin(1.4 * z) + rng.normal(scale=0.65, size=z.size)
    outcome = 1.7 * treatment + 1.8 * np.cos(z) + rng.normal(scale=0.7, size=z.size)
    result = partialling_out(outcome, treatment, z, n_knots=7)
    assert abs(result.coefficient - 1.7) < 0.08
    assert result.robust_standard_error > 0
