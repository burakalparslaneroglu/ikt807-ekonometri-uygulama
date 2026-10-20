import pytest
from streamlit.testing.v1 import AppTest


def _panel():
    app = AppTest.from_string(
        "from topics.konu02_inference_ui import render_inference_refresher\nrender_inference_refresher()",
        default_timeout=60,
    )
    app.run()
    assert not app.exception
    return app


def test_direction_control_changes_the_p_event():
    app = _panel()
    assert app.metric[0].value == "0,0719"
    app.radio(key="konu02_test_yonu").set_value("Sağ taraflı").run()
    assert not app.exception and app.metric[0].value == "0,0359"
    assert app.metric[1].value == "H₀ reddedilir"
    app.radio(key="konu02_test_yonu").set_value("Sol taraflı").run()
    assert not app.exception and app.metric[0].value == "0,9641"
    assert app.metric[1].value == "H₀ reddedilemez"


@pytest.mark.parametrize("view", ["F testi", "Wald / LM / LR", "Heteroskedastisite testleri", "VIF eşikleri"])
def test_each_refresher_view_renders_and_controls_work(view):
    app = _panel()
    app.selectbox(key="konu02_cikarim_baslik").set_value(view).run()
    assert not app.exception
    if view == "F testi":
        assert app.metric[0].value == "6,00"
        assert app.metric[1].value == "0,00350"
        app.slider(key="konu02_f_sse_r").set_value(96.0).run()
        assert not app.exception and app.metric[0].value == "0,00"
        assert app.metric[1].value == "1,00000"
    elif view == "VIF eşikleri":
        assert app.metric[0].value == "1,17"
        app.slider(key="konu02_vif_r2").set_value(.8).run()
        assert not app.exception and app.metric[0].value == "5,00"
        assert app.metric[1].value == "2,24"
