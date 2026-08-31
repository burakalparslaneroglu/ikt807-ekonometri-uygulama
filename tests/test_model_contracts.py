from __future__ import annotations

import pytest

from core.types import (
    Estimate,
    EstimandMetadata,
    InferenceSpec,
    ModelResult,
    TuningSpec,
)


def test_cluster_inference_requires_cluster_variable() -> None:
    with pytest.raises(ValueError, match="cluster_variable"):
        InferenceSpec(covariance_type="cluster")


def test_tuning_contract_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="seed"):
        TuningSpec(seed=-1)
    with pytest.raises(ValueError, match="bandwidth"):
        TuningSpec(seed=807, bandwidth=0)
    with pytest.raises(ValueError, match="folds"):
        TuningSpec(seed=807, folds=1)


def test_model_result_keeps_estimand_inference_and_tuning_together() -> None:
    result = ModelResult(
        model_name="Öğretim benchmarkı",
        dataset_id="ddk2011",
        sample_definition="Eksiksiz kovaryat örneklemi",
        estimand=EstimandMetadata(
            name="Ortalama tracking etkisi",
            plain_language="Tracking programının ortalama test puanı etkisi.",
            target_population="DDK deney örneklemi",
            causal_interpretation_condition="Okul düzeyinde rastgele atama.",
        ),
        estimates=(
            Estimate(
                name="theta",
                label="Tracking etkisi",
                value=1.383,
                standard_error=0.699,
                confidence_interval=(0.013, 2.753),
            ),
        ),
        inference=InferenceSpec(
            covariance_type="cluster",
            cluster_variable="schoolid",
            finite_sample_correction=True,
        ),
        tuning=TuningSpec(seed=807, folds=5, split_unit="okul"),
    )
    assert result.estimand.causal_interpretation_condition
    assert result.inference.cluster_variable == "schoolid"
    assert result.tuning.split_unit == "okul"
