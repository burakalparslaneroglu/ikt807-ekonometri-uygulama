from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest
from streamlit.testing.v1 import AppTest

from core.navigation import apply_query, get_experiments, lecture_url, parse_target
from core.labs.registry import get_lab


def test_all_lecture_targets_round_trip():
    for i in range(1, 13):
        topic = f"konu{i:02}"
        for tab, items in (("sezgi", get_experiments(topic)), ("uygulama", get_lab(topic).steps)):
            for item in items:
                url = lecture_url(topic, tab, item.number)
                target, errors = parse_target(dict(parse_qsl(urlsplit(url).query)))
                assert not errors and target.topic == topic and target.number == item.number


@pytest.mark.parametrize("query", [
    {"konu": "99"}, {"konu": "07", "sekme": "other"},
    {"konu": "07", "sekme": "sezgi", "deney": "99"},
    {"konu": "07", "sekme": "sezgi", "deney": "1", "p_n": "nan"},
    {"konu": "02", "sekme": "sezgi", "deney": "4", "p_rho": "1"},
    {"konu": "02", "sekme": "sezgi", "deney": "4", "p_rho": "0.83"},
])
def test_invalid_link_does_not_mutate_user_selection(query):
    state = {"topic_selector": "konu01"}
    assert apply_query(state, query)
    assert state["topic_selector"] == "konu01"


def test_query_applied_once_and_new_url_resets_experiment_defaults():
    state = {}
    query = {"konu": "02", "sekme": "sezgi", "deney": "4", "p_rho": "0.95"}
    assert not apply_query(state, query)
    assert state["konu02_sezgi4_rho"] == 0.95
    state["konu02_sezgi4_rho"] = 0
    apply_query(state, query)
    assert state["konu02_sezgi4_rho"] == 0
    apply_query(state, query | {"p_rho": "0.8"})
    assert state["konu02_sezgi4_rho"] == 0.8


def test_lab_link_overrides_stale_alternative_source_and_step_shadow():
    state = {"_kalici_konu04_lab_kaynak": "kendi", "_kalici_konu04_lab_step": 1}
    apply_query(state, {"konu": "04", "sekme": "uygulama", "adim": "4"})
    assert state["konu04_lab_kaynak"] == state["_kalici_konu04_lab_kaynak"] == "notlar"
    assert state["konu04_lab_step"] == state["_kalici_konu04_lab_step"] == 4


def test_real_app_deep_link_and_manual_interaction():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60)
    app.query_params.update(konu="03", sekme="sezgi", deney="4", p_b="2")
    app.run()
    assert not app.exception
    assert app.radio(key="topic_selector").value == "konu03"
    assert app.session_state["konu03_sekme"] == "Sezgi"
    assert app.segmented_control(key="konu03_sezgi_deney").value == 4
    assert app.slider(key="konu03_sezgi4_b").value == 2
    app.slider(key="konu03_sezgi4_b").set_value(0).run()
    assert not app.exception and app.slider(key="konu03_sezgi4_b").value == 0
    app.radio(key="topic_selector").set_value("konu04").run()
    assert not app.exception and app.radio(key="topic_selector").value == "konu04"
    app.query_params.update(konu="04", sekme="uygulama", adim="4")
    app.run()
    assert not app.exception and app.segmented_control(key="konu04_lab_step").value == 4
    assert app.session_state["konu04_sekme"] == "Uygulama"


@pytest.mark.parametrize("tab,number", [("sezgi", 4), ("uygulama", 7)])
def test_konu02_properties_panel_preserves_deep_link_and_three_tabs(tab, number):
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60)
    selector = "deney" if tab == "sezgi" else "adim"
    app.query_params.update({"konu": "02", "sekme": tab, selector: str(number)})
    app.run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Uygulama", "Sezgi", "Kendini sına"]
    panel = next(e for e in app.expander if "örnekleme özellikleri" in e.label)
    assert len(panel.latex) == 3
    assert not panel.code  # Indented prose must not become Markdown code blocks.
    assert app.session_state["konu02_sekme"] == ("Sezgi" if tab == "sezgi" else "Uygulama")
    widget = "konu02_sezgi_deney" if tab == "sezgi" else "konu02_lab_step"
    assert app.segmented_control(key=widget).value == number
    app.run()
    assert not app.exception and app.segmented_control(key=widget).value == number


def test_konu02_gauss_markov_panel_and_existing_navigation():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60)
    app.query_params.update(konu="02", sekme="sezgi", deney="4")
    app.run()
    assert not app.exception
    panel = next(e for e in app.expander if "Gauss–Markov" in e.label)
    assert len(panel.latex) == 2
    assert not panel.code
    prose = "\n".join(m.value for m in panel.markdown)
    assert "Normallik gerekmez" in prose
    assert "yeniden BLUE yapmaz" in prose
    assert "koşullu sapmasız" in prose
    assert [t.label for t in app.tabs] == ["Uygulama", "Sezgi", "Kendini sına"]
    assert app.segmented_control(key="konu02_sezgi_deney").value == 4
