from __future__ import annotations

from io import StringIO

import pandas as pd
import pytest

from core.data_registry import get_dataset_metadata
from core.datasets import load_cps_csv, load_registered_csv


def _valid_cps_csv() -> StringIO:
    metadata = get_dataset_metadata("cps09mar")
    row = {column: 1 for column in metadata.expected_columns}
    row.update(
        {
            "age": 30,
            "education": 16,
            "earnings": 50000,
            "hours": 40,
            "week": 50,
            "hrwage": 25,
            "experience": 8,
            "experience2_100": 0.64,
            "lwage": 3.2189,
        }
    )
    buffer = StringIO()
    pd.DataFrame([row]).to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


def test_cps_adapter_validates_and_loads_prepared_csv() -> None:
    frame = load_cps_csv(_valid_cps_csv())
    assert len(frame) == 1
    assert frame.loc[0, "education"] == 16


def test_cps_adapter_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="beklenen sütunları"):
        load_cps_csv(StringIO("education,lwage\n16,3.2\n"))


def test_cps_adapter_rejects_nonpositive_wage() -> None:
    source = _valid_cps_csv()
    frame = pd.read_csv(source)
    frame["hrwage"] = 0
    buffer = StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    with pytest.raises(ValueError, match="pozitif"):
        load_cps_csv(buffer)


def test_registered_adapter_validates_ddk_schema_without_copying_data() -> None:
    metadata = get_dataset_metadata("ddk2011")
    buffer = StringIO()
    pd.DataFrame([{column: 1 for column in metadata.expected_columns}]).to_csv(
        buffer, index=False
    )
    buffer.seek(0)
    frame = load_registered_csv("ddk2011", buffer)
    assert tuple(frame.columns) == metadata.expected_columns
