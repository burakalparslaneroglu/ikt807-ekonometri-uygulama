import numpy as np
import pytest
import statsmodels.api as sm

from core.inference_konu02 import classical_f_test, normal_test, vif_summary


@pytest.mark.parametrize("direction,expected,reject", [
    ("two-sided", .07186063822585158, False),
    ("greater", .03593031911292579, True),
    ("less", .9640696808870742, False),
])
def test_signed_normal_example(direction, expected, reject):
    result = normal_test(1.8, direction)
    assert result.p_value == pytest.approx(expected, abs=1e-14)
    assert result.reject is reject


def test_one_sided_sign_and_threshold_convention():
    right = normal_test(-1.8, "greater")
    left = normal_test(1.8, "less")
    assert right.p_value == pytest.approx(left.p_value)
    assert right.p_value > .5
    p = normal_test(1.8, "greater").p_value
    assert not normal_test(1.8, "greater", alpha=p).reject
    assert normal_test(1.8, "greater", alpha=.05).reject
    assert not normal_test(1.8, "greater", alpha=.01).reject


def test_f_numeric_example():
    result = classical_f_test(96, 108, 100, 4, 2)
    assert result.statistic == 6
    assert (result.numerator_df, result.denominator_df) == (2, 96)
    assert result.p_value == pytest.approx(.003504926560466462, abs=1e-14)
    assert result.critical_value == pytest.approx(3.091191258857293, abs=1e-12)


@pytest.mark.parametrize("q", [1, 2])
def test_sse_f_matches_fitted_nested_models(q):
    rng = np.random.default_rng(20261020)
    x = np.linspace(-1, 1, 80)
    design = np.column_stack([np.ones(80), x, np.sin(4*x), np.cos(3*x)])
    y = design @ np.array([2, .4, .3, .2]) + rng.normal(0, .7, 80)
    unrestricted = sm.OLS(y, design).fit()
    restricted = sm.OLS(y, design[:, :4-q]).fit()
    expected_f, expected_p, expected_q = unrestricted.compare_f_test(restricted)
    actual = classical_f_test(unrestricted.ssr, restricted.ssr, 80, 4, q)
    assert actual.statistic == pytest.approx(expected_f, rel=1e-12)
    assert actual.p_value == pytest.approx(expected_p, abs=1e-13)
    assert actual.numerator_df == expected_q
    if q == 1:
        assert actual.statistic == pytest.approx(unrestricted.tvalues[-1]**2, rel=1e-11)


@pytest.mark.parametrize("r2,vif,se", [(0, 1, 1), (.8, 5, np.sqrt(5)), (.9, 10, np.sqrt(10)),
                                      (.147, 1/.853, np.sqrt(1/.853))])
def test_vif_variance_and_se_factors(r2, vif, se):
    actual = vif_summary(r2)
    assert actual == pytest.approx((vif, se))


@pytest.mark.parametrize("r2", [-.1, 1, np.nan])
def test_vif_rejects_undefined_design(r2):
    with pytest.raises(ValueError):
        vif_summary(r2)


@pytest.mark.parametrize("observed,direction,alpha", [
    (np.nan, "greater", .05), (1, "invalid", .05), (1, "less", 0), (1, "greater", 1),
])
def test_normal_rejects_invalid_inputs(observed, direction, alpha):
    with pytest.raises(ValueError):
        normal_test(observed, direction, alpha)


@pytest.mark.parametrize("sse_u,sse_r,n,k,q", [
    (96, 95, 100, 4, 2), (0, 108, 100, 4, 2), (96, np.inf, 100, 4, 2),
    (96, 108, 4, 4, 2), (96, 108, 100, 4, 5), (96, 108, 100.0, 4, 2),
])
def test_f_rejects_invalid_nested_comparison(sse_u, sse_r, n, k, q):
    with pytest.raises(ValueError):
        classical_f_test(sse_u, sse_r, n, k, q)
