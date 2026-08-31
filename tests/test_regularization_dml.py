import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from core.dml import (
    DMLDGPConfig,
    cross_fit_nuisance,
    double_selection,
    fit_dml,
    simulate_dml_data,
)
from core.pipelines import compare_holdout_models, cross_validate_lasso
from core.regularization import (
    SparseDGPConfig,
    orthonormal_lasso_solution,
    polynomial_complexity_curve,
    regularization_path,
    simulate_sparse_regression,
)
from core.research_workflow import RESEARCH_WORKFLOW, audit_workflow


def test_orthonormal_lasso_closed_form_matches_sklearn() -> None:
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(300, 8))
    q, _ = np.linalg.qr(raw)
    x = q * np.sqrt(300)
    y = x @ np.array([2.0, -1.5, 0.8, 0, 0, 0, 0, 0]) + rng.normal(
        scale=0.2, size=300
    )
    expected = Lasso(alpha=0.15, fit_intercept=False, max_iter=20_000).fit(x, y)
    actual = orthonormal_lasso_solution(y, x, 0.15)
    np.testing.assert_allclose(actual, expected.coef_, atol=1e-8)


def test_ridge_shrinks_and_lasso_sparsifies_along_path() -> None:
    data = simulate_sparse_regression(
        SparseDGPConfig(nobs=500, nfeatures=20, nonzero=5, seed=2)
    )
    path = regularization_path(
        data.features,
        data.outcome,
        (0.01, 0.1, 1.0),
        data.feature_names,
    )
    assert np.linalg.norm(path.ridge_coefficients[-1]) < np.linalg.norm(
        path.ridge_coefficients[0]
    )
    nonzero = (np.abs(path.lasso_coefficients) > 1e-8).sum(axis=1)
    assert nonzero[-1] < nonzero[0]


def test_polynomial_complexity_has_finite_selected_degree() -> None:
    result = polynomial_complexity_curve(nobs=280, max_degree=10, seed=3)
    assert 1 <= result.selected_degree <= 10
    assert np.isfinite(result.train_mse).all()
    assert np.isfinite(result.test_mse).all()


def test_lasso_cv_is_deterministic_and_one_se_is_more_regularized() -> None:
    data = simulate_sparse_regression(
        SparseDGPConfig(nobs=500, nfeatures=18, nonzero=5, seed=4)
    )
    alphas = np.logspace(-2.5, 0, 18)
    first = cross_validate_lasso(
        data.features, data.outcome, alphas, data.feature_names, folds=5, seed=5
    )
    second = cross_validate_lasso(
        data.features, data.outcome, alphas, data.feature_names, folds=5, seed=5
    )
    np.testing.assert_allclose(first.mean_mse, second.mean_mse)
    assert first.lambda_1se >= first.lambda_min
    assert len(first.selected_1se) <= len(first.selected_min)


def test_fold_scaling_uses_training_rows_only() -> None:
    data = simulate_sparse_regression(
        SparseDGPConfig(nobs=300, nfeatures=10, nonzero=3, seed=6)
    )
    result = cross_validate_lasso(
        data.features,
        data.outcome,
        (0.03, 0.1, 0.3),
        data.feature_names,
        folds=3,
        seed=7,
    )
    for fold in range(3):
        train = result.fold_assignments != fold
        np.testing.assert_allclose(
            result.fold_training_means[fold], data.features[train].mean(axis=0)
        )


def test_post_lasso_holdout_comparison_is_finite() -> None:
    data = simulate_sparse_regression(
        SparseDGPConfig(nobs=600, nfeatures=24, nonzero=5, seed=8)
    )
    train_x, test_x, train_y, test_y = train_test_split(
        data.features, data.outcome, test_size=0.3, random_state=9
    )
    comparison = compare_holdout_models(
        train_x,
        train_y,
        test_x,
        test_y,
        data.feature_names,
        lasso_alpha=0.1,
    )
    assert np.isfinite(comparison.test_mse).all()
    assert comparison.nonzero_counts[2] < data.features.shape[1]
    assert comparison.nonzero_counts[2] == comparison.nonzero_counts[3]


def test_cross_fitted_dml_recovers_known_theta() -> None:
    data = simulate_dml_data(
        DMLDGPConfig(nobs=1600, nfeatures=18, theta=1.4, groups=50, seed=10)
    )
    result = fit_dml(
        data.outcome,
        data.treatment,
        data.controls,
        folds=5,
        seed=11,
        learner="Random Forest",
        groups=data.groups,
    )
    assert abs(result.theta - 1.4) < 0.22
    assert result.standard_error > 0
    assert result.cross_fit.split_unit == "grup"


def test_group_cross_fit_keeps_each_group_in_one_validation_fold() -> None:
    data = simulate_dml_data(
        DMLDGPConfig(nobs=600, nfeatures=12, groups=30, seed=12)
    )
    result = cross_fit_nuisance(
        data.outcome,
        data.treatment,
        data.controls,
        folds=5,
        seed=13,
        learner="Ridge",
        groups=data.groups,
    )
    for group in np.unique(data.groups):
        assert np.unique(result.fold_assignments[data.groups == group]).size == 1


def test_group_oof_prediction_does_not_use_own_group_outcome() -> None:
    data = simulate_dml_data(
        DMLDGPConfig(nobs=600, nfeatures=12, groups=30, seed=14)
    )
    baseline = cross_fit_nuisance(
        data.outcome,
        data.treatment,
        data.controls,
        folds=5,
        seed=15,
        learner="Ridge",
        groups=data.groups,
    )
    changed_y = data.outcome.copy()
    target = data.groups == 0
    changed_y[target] += 1000
    changed = cross_fit_nuisance(
        changed_y,
        data.treatment,
        data.controls,
        folds=5,
        seed=15,
        learner="Ridge",
        groups=data.groups,
    )
    np.testing.assert_allclose(
        baseline.outcome_predictions[target], changed.outcome_predictions[target]
    )


def test_double_selection_uses_union_of_outcome_and_treatment_sets() -> None:
    data = simulate_dml_data(
        DMLDGPConfig(nobs=900, nfeatures=16, groups=30, seed=16)
    )
    result = double_selection(
        data.outcome,
        data.treatment,
        data.controls,
        data.feature_names,
        folds=4,
        seed=17,
    )
    assert set(result.outcome_selected).issubset(result.union_selected)
    assert set(result.treatment_selected).issubset(result.union_selected)
    assert np.isfinite(result.double_selection_theta)


def test_research_workflow_order_and_audit() -> None:
    assert RESEARCH_WORKFLOW[0].key == "estimand"
    assert RESEARCH_WORKFLOW[-1].key == "reproducibility"
    audit = audit_workflow({"estimand", "identification"})
    assert [item.complete for item in audit[:3]] == [True, True, False]
