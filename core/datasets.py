"""Kullanıcı tarafından sağlanan hazırlanmış CPS dosyası için güvenli adaptör."""

from __future__ import annotations

from io import BytesIO
from os import environ
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from core.data_registry import validate_dataset_columns


CPS_PATH_ENV = "IKT807_CPS_PATH"


def configured_cps_path() -> Path | None:
    value = environ.get(CPS_PATH_ENV)
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_file() else None


def load_cps_csv(source: str | Path | bytes | BinaryIO) -> pd.DataFrame:
    """Hazırlanmış CPS CSV'sini okur, şemayı ve temel değer alanlarını doğrular."""

    csv_source: str | Path | BinaryIO
    if isinstance(source, bytes):
        csv_source = BytesIO(source)
    else:
        csv_source = source
    frame = pd.read_csv(csv_source)
    validate_dataset_columns("cps09mar", frame.columns)
    numeric_columns = (
        "age",
        "female",
        "hisp",
        "education",
        "earnings",
        "hours",
        "week",
        "hrwage",
        "experience",
        "experience2_100",
        "lwage",
    )
    frame = frame.copy()
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.empty:
        raise ValueError("CPS dosyası boş olamaz.")
    if frame[list(numeric_columns)].isna().any().any():
        raise ValueError("CPS sayısal sütunlarında eksik değer bulunmamalıdır.")
    if (frame["hrwage"] <= 0).any():
        raise ValueError("Saatlik ücret pozitif olmalıdır.")
    return frame


def load_registered_csv(
    dataset_id: str,
    source: str | Path | bytes | BinaryIO,
) -> pd.DataFrame:
    """Hazırlanmış lisans-kapılı bir CSV'yi depoya kopyalamadan doğrular."""

    csv_source: str | Path | BinaryIO
    if isinstance(source, bytes):
        csv_source = BytesIO(source)
    else:
        csv_source = source
    frame = pd.read_csv(csv_source)
    validate_dataset_columns(dataset_id, frame.columns)
    if frame.empty:
        raise ValueError(f"{dataset_id} dosyası boş olamaz.")
    return frame
