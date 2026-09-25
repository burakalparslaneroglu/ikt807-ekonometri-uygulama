from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _data_path() -> str | None:
    for name in ("IKT807_HANSEN_CPS09MAR_PATH", "IKT807_CPS_PATH"):
        value = os.environ.get(name)
        if value and Path(value).is_file():
            return value
    return None


def _run(timeout: int = 30) -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=timeout).run()


def test_application_tab_renders_without_data_or_network(monkeypatch) -> None:
    monkeypatch.delenv("IKT807_HANSEN_CPS09MAR_PATH", raising=False)
    monkeypatch.delenv("IKT807_CPS_PATH", raising=False)
    app = _run()
    assert not app.exception
    assert [tab.label for tab in app.tabs][:2] == ["Uygulama", "Sezgi"]
    assert app.segmented_control(key="konu01_lab_step").value == 1
    assert any("henüz yüklenmedi" in item.value for item in app.markdown)


def test_language_selector_switches_generated_code(monkeypatch) -> None:
    monkeypatch.delenv("IKT807_HANSEN_CPS09MAR_PATH", raising=False)
    monkeypatch.delenv("IKT807_CPS_PATH", raising=False)
    app = _run()
    app.segmented_control(key="konu01_lab_step").set_value(5).run()
    python_code = "\n".join(block.value for block in app.code)
    assert 'smf.ols("lwage ~ education", data=cps)' in python_code

    app.segmented_control(key="code_language").set_value("R").run()
    r_code = "\n".join(block.value for block in app.code)
    assert "lm(lwage ~ education, data = cps)" in r_code

    app.segmented_control(key="code_language").set_value("Stata").run()
    stata_code = "\n".join(block.value for block in app.code)
    assert "quietly regress lwage education" in stata_code
    assert not app.exception


def test_next_and_previous_buttons_move_between_steps(monkeypatch) -> None:
    monkeypatch.delenv("IKT807_CPS_PATH", raising=False)
    app = _run()
    app.button(key="konu01_lab_next").click().run()
    assert app.segmented_control(key="konu01_lab_step").value == 2
    app.button(key="konu01_lab_prev").click().run()
    app.button(key="konu01_lab_prev").click().run()
    assert app.segmented_control(key="konu01_lab_step").value == 1


@pytest.mark.skipif(_data_path() is None, reason="Gerçek Hansen cps09mar dosyası verilmedi.")
def test_regression_step_matches_notes_with_real_data() -> None:
    app = _run(timeout=60)
    app.segmented_control(key="konu01_lab_step").set_value(6).run()
    assert not app.exception
    assert any("50.742 gözlem" in item.value for item in app.markdown)
    comparisons = [frame.value for frame in app.dataframe if "Durum" in frame.value.columns]
    assert len(comparisons) == 1
    assert len(comparisons[0]) == 13
    assert set(comparisons[0]["Durum"]) == {"✓"}
    percentages = {metric.label: metric.value for metric in app.metric}
    assert percentages["Model (3) eğitim katsayısının kesin yüzde karşılığı"] == "%12,16"
    assert percentages["Model (3) kadın katsayısının kesin yüzde karşılığı"] == "%−22,87"
