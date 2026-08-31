from __future__ import annotations

import pytest

from core.data_registry import (
    get_dataset_metadata,
    list_datasets,
    validate_dataset_columns,
)
from core.topic_registry import list_topics


def test_registry_contains_five_hansen_dataset_families() -> None:
    datasets = list_datasets()
    assert [dataset.dataset_id for dataset in datasets] == [
        "cps09mar",
        "ddk2011",
        "card1995",
        "chj2004",
        "lm2007",
    ]
    assert all(
        dataset.redistribution_status == "license_review_required"
        for dataset in datasets
    )


def test_dataset_topic_links_are_valid() -> None:
    topic_keys = {topic.key for topic in list_topics()}
    for dataset in list_datasets():
        assert set(dataset.allowed_topics) <= topic_keys


def test_ddk_metadata_preserves_cluster_and_resampling_unit() -> None:
    dataset = get_dataset_metadata("ddk2011")
    assert dataset.cluster_variable == "schoolid"
    assert dataset.resampling_unit == "okul"


def test_column_validation_accepts_superset_and_rejects_missing() -> None:
    dataset = get_dataset_metadata("lm2007")
    validate_dataset_columns(
        "lm2007", (*dataset.expected_columns, "extra_teaching_column")
    )
    with pytest.raises(ValueError, match="beklenen sütunları içermiyor"):
        validate_dataset_columns("lm2007", ("povrate60",))
