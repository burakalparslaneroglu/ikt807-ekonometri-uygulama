from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from core.hansen_data import (
    CPS09MAR_COLUMNS,
    HansenDataError,
    find_member,
    load_from_archive,
    load_from_upload,
)


def _hansen_like_cps(rows: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(807)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 65, rows),
            "female": rng.integers(0, 2, rows),
            "hisp": rng.integers(0, 2, rows),
            "education": rng.choice([12, 14, 16], rows),
            "earnings": rng.integers(20000, 90000, rows),
            "hours": rng.integers(36, 60, rows),
            "week": rng.integers(48, 53, rows),
            "union": 0,
            "uncov": 0,
            "region": rng.integers(1, 5, rows),
            "race": 1,
            "marital": rng.integers(1, 8, rows),
        }
    )[list(CPS09MAR_COLUMNS)]


def _crlf_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\r\n").encode("utf-8")


def _hansen_text(frame: pd.DataFrame, separator: str = " ", scientific: bool = False) -> bytes:
    """Hansen'in .txt biçimi: başlık yok, boşlukla ayrılmış, Windows satır sonu."""

    def cell(value) -> str:
        return f"{float(value):.7e}" if scientific else str(value)

    rows = (separator.join(cell(v) for v in row) for row in frame.itertuples(index=False))
    return ("\r\n".join(rows) + "\r\n").encode("ascii")


def _archive(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


def test_member_is_found_regardless_of_folder_structure_and_case() -> None:
    data = _crlf_csv(_hansen_like_cps())
    for path in ("cps09mar.txt", "Econometrics Data/cps09mar/cps09mar.txt", "DATA/CPS09MAR.TXT"):
        archive = _archive({path: data, "other/DDK2011.txt": b"x\n1\n"})
        assert find_member(archive, "cps09mar.txt") == path


def test_missing_member_raises_clear_error() -> None:
    archive = _archive({"other/DDK2011.txt": b"x\n1\n"})
    with pytest.raises(HansenDataError, match="bulunamadı"):
        find_member(archive, "cps09mar.txt")


def test_archive_load_reads_hansen_text_format_with_crlf() -> None:
    frame = _hansen_like_cps()
    loaded = load_from_archive("cps09mar", _archive({"x/cps09mar.txt": _crlf_csv(frame)}))
    assert list(loaded.frame.columns) == list(CPS09MAR_COLUMNS)
    assert len(loaded.frame) == len(frame)
    assert loaded.matches_hansen is False


def test_upload_accepts_teaching_csv_and_stata_file() -> None:
    frame = _hansen_like_cps()
    teaching = frame.drop(columns=["union", "uncov"]).assign(lwage=1.0, hrwage=2.0)
    assert len(load_from_upload("cps09mar", teaching.to_csv(index=False).encode(), "egitim.csv").frame) == 30

    buffer = io.BytesIO()
    frame.to_stata(buffer, write_index=False)
    loaded = load_from_upload("cps09mar", buffer.getvalue(), "cps09mar.dta")
    assert loaded.frame["education"].tolist() == frame["education"].tolist()


@pytest.mark.parametrize(
    ("separator", "scientific"),
    [(" ", False), ("   ", False), ("\t", False), (" ", True)],
)
def test_hansen_headerless_text_gets_documented_column_names(separator: str, scientific: bool) -> None:
    frame = _hansen_like_cps()
    loaded = load_from_upload("cps09mar", _hansen_text(frame, separator, scientific), "cps09mar.txt")
    assert list(loaded.frame.columns) == list(CPS09MAR_COLUMNS)
    np.testing.assert_allclose(loaded.frame.to_numpy(dtype=float), frame.to_numpy(dtype=float))


def test_archive_with_hansen_folder_layout_and_format() -> None:
    frame = _hansen_like_cps()
    archive = _archive(
        {
            "Econometrics Data/cps09mar/cps09mar.txt": _hansen_text(frame),
            "Econometrics Data/cps09mar/cps09mar_description.pdf": b"%PDF",
            "Econometrics Data/DDK2011/DDK2011.txt": b"1 2 3\r\n",
        }
    )
    loaded = load_from_archive("cps09mar", archive)
    assert loaded.frame["education"].tolist() == frame["education"].tolist()


def test_headerless_text_with_wrong_column_count_is_rejected() -> None:
    with pytest.raises(HansenDataError, match="12 sütun"):
        load_from_upload("cps09mar", b"1 2 3\r\n4 5 6\r\n", "cps09mar.txt")


def test_upload_with_wrong_schema_is_rejected() -> None:
    wrong = pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_csv(index=False).encode()
    with pytest.raises(HansenDataError, match="beklenen sütunlar"):
        load_from_upload("cps09mar", wrong, "yanlis.csv")
    with pytest.raises(HansenDataError, match="Excel"):
        load_from_upload("cps09mar", b"", "cps09mar.xlsx")
