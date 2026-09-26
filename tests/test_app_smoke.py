from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _run_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=10).run()


def test_app_loads_with_course_shell_and_topic_01() -> None:
    app = _run_app()
    assert not app.exception
    assert app.title[0].value == "IKT 807 Ekonometrik Modelleme ve Uygulamaları"
    assert app.radio(key="topic_selector").value == "konu01"
    assert any(
        "Ampirik Ekonometrik Modelleme" in item.value for item in app.markdown
    )


def test_topic_switch_and_text_scale_are_independent() -> None:
    app = _run_app()
    app.selectbox(key="text_scale_label").set_value("%130").run()
    app.radio(key="topic_selector").set_value("konu04").run()
    assert not app.exception
    assert app.session_state["text_scale_label"] == "%130"
    assert app.session_state["active_topic"] == "konu04"
    assert any("Araçsal Değişkenler" in item.value for item in app.markdown)


def test_question_answer_and_new_question_state() -> None:
    """Henüz yeni yapıya taşınmamış konularda eski soru-cevap kutusu."""

    app = _run_app()
    app.radio(key="topic_selector").set_value("konu09").run()
    successes = len(app.success)
    app.button(key="konu09_toggle_answer").click().run()
    assert not app.exception
    assert app.session_state["konu09_answer_visible"] is True
    assert len(app.success) == successes + 1

    app.button(key="konu09_new_question").click().run()
    assert app.session_state["konu09_answer_visible"] is False
    assert app.session_state["konu09_question_index"] == 1
    assert len(app.success) == successes


def test_topic_01_intuition_experiments_are_interactive() -> None:
    app = _run_app()
    assert not app.exception
    assert app.segmented_control(key="konu01_sezgi_deney").value == 1
    assert app.slider(key="konu01_sezgi1_gamma").value == 0.01
    assert any(metric.label == "OLS eğimi β̂₁" for metric in app.metric)

    app.slider(key="konu01_sezgi1_gamma").set_value(0.0).run()
    assert not app.exception
    target = next(metric for metric in app.metric if metric.label == "Hedef eğim β₁")
    assert target.value == "0,0800"

    app.segmented_control(key="konu01_sezgi_deney").set_value(3).run()
    assert not app.exception
    assert app.slider(key="konu01_sezgi3_pi").value == 1.2
    assert any(metric.label == "Nedensel etki (DGP)" for metric in app.metric)


def test_topic_02_has_three_tabs_and_interactive_experiments() -> None:
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu02").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs][:3] == ["Uygulama", "Sezgi", "Kendini sına"]
    assert app.segmented_control(key="konu02_lab_step").value == 1
    app.segmented_control(key="konu02_sezgi_deney").set_value(2).run()
    assert not app.exception
    assert any(metric.label == "FWL eğimi" for metric in app.metric)

def test_topic_03_has_three_tabs_lab_and_experiments() -> None:
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu03").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs][:3] == ["Uygulama", "Sezgi", "Kendini sına"]
    assert app.segmented_control(key="konu03_lab_step").value == 1
    assert any("DDK2011.dta" in item.value for item in app.markdown)
    assert any(metric.label == "Gözlenen fark" for metric in app.metric)

    app.slider(key="konu03_sezgi1_s").set_value(0.0).run()
    assert not app.exception
    assert any("rastgele atanıyor" in item.value for item in app.info)

    app.segmented_control(key="konu03_sezgi_deney").set_value(2).run()
    assert not app.exception
    assert any(metric.label == "Zayıflama çarpanı λ (DGP)" for metric in app.metric)


def test_topic_04_has_three_tabs_lab_and_experiments() -> None:
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu04").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs][:3] == ["Uygulama", "Sezgi", "Kendini sına"]
    assert app.segmented_control(key="konu04_lab_step").value == 1
    assert any("Card1995.dta" in item.value for item in app.markdown)
    assert any(metric.label == "Wald = 2SLS" for metric in app.metric)

    app.slider(key="konu04_sezgi1_kappa").set_value(0.1).run()
    assert not app.exception
    assert any("dışlama kısıtı veriden test edilemez" in item.value for item in app.info)

    app.segmented_control(key="konu04_sezgi_deney").set_value(3).run()
    assert not app.exception
    assert any(metric.label == "LATE (DGP)" for metric in app.metric)


def test_topic_05_has_three_tabs_lab_and_experiments() -> None:
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu05").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs][:3] == ["Uygulama", "Sezgi", "Kendini sına"]
    assert app.segmented_control(key="konu05_lab_step").value == 1
    assert any("cps09mar.txt" in item.value for item in app.markdown)
    assert any(metric.label == "LPM sınır dışı tahmin" for metric in app.metric)

    app.slider(key="konu05_sezgi1_beta").set_value(0.5).run()
    assert not app.exception
    assert any("Seçim araştırma hedefine bağlıdır" in item.value for item in app.info)

    app.segmented_control(key="konu05_sezgi_deney").set_value(3).run()
    assert not app.exception
    assert any(metric.label == "Sonlu fark (AME)" for metric in app.metric)


def test_topic_06_has_three_tabs_lab_and_experiments() -> None:
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu06").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs][:3] == ["Uygulama", "Sezgi", "Kendini sına"]
    assert app.segmented_control(key="konu06_lab_step").value == 1
    assert any("CHJ2004.dta" in item.value for item in app.markdown)
    tobit = next(metric for metric in app.metric if metric.label == "Tobit eğimi")
    assert tobit.value == "1,002"

    app.segmented_control(key="konu06_sezgi_deney").set_value(2).run()
    assert not app.exception
    assert any(metric.label == "Heckman eğimi" for metric in app.metric)
    app.slider(key="konu06_sezgi2_gz").set_value(0.0).run()
    assert not app.exception
    assert any("dışlama değişkeni seçimi etkilemiyor" in item.value for item in app.info)


def test_topic_07_has_three_tabs_lab_and_experiments(monkeypatch) -> None:
    # Veri yolu tanımlıysa laboratuvar açılışta 15 kantil regresyonu çözer; bu test yalnız arayüzü sınar.
    for name in ("IKT807_HANSEN_CPS09MAR_PATH", "IKT807_CPS_PATH"):
        monkeypatch.delenv(name, raising=False)
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu07").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs][:3] == ["Uygulama", "Sezgi", "Kendini sına"]
    assert app.segmented_control(key="konu07_lab_step").value == 1
    assert any("cps09mar.txt" in item.value for item in app.markdown)
    assert any(metric.label == "τ = 0,90 eğimi" for metric in app.metric)

    app.slider(key="konu07_sezgi1_gamma").set_value(0.0).run()
    assert not app.exception
    assert any("paraleldir" in item.value for item in app.info)

    app.segmented_control(key="konu07_sezgi_deney").set_value(3).run()
    assert not app.exception
    assert any(metric.label == "Oran (kuram)" for metric in app.metric)


def test_topic_08_has_three_tabs_lab_and_experiments(monkeypatch) -> None:
    monkeypatch.delenv("IKT807_HANSEN_DDK2011_PATH", raising=False)
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu08").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs][:3] == ["Uygulama", "Sezgi", "Kendini sına"]
    assert app.segmented_control(key="konu08_lab_step").value == 1
    assert any("DDK2011.dta" in item.value for item in app.markdown)
    assert any(metric.label == "CV ile seçilen h" for metric in app.metric)

    app.segmented_control(key="konu08_sezgi_deney").set_value(2).run()
    assert not app.exception
    assert any(metric.label == "NW hatası, x = 0" for metric in app.metric)
    app.slider(key="konu08_sezgi2_h").set_value(0.2).run()
    assert not app.exception
    assert any("Nadaraya–Watson" in item.value for item in app.info)


def test_topic_09_rdd_direction_diagnostics_and_lm_gate() -> None:
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu09").run()
    assert not app.exception
    assert app.slider(key="konu09_bandwidth").value == 4.0
    assert any(metric.label == "RDD sıçraması" for metric in app.metric)
    assert any(metric.label == "Eşik solu n" for metric in app.metric)
    assert any(metric.label == "Yerel Wald oranı" for metric in app.metric)
    assert any("LM2007 verisi lisans teyidi" in item.value for item in app.info)

    app.slider(key="konu09_manipulation").set_value(0.8).run()
    assert not app.exception
    assert any("belirgin yığılma" in item.value for item in app.error)

    app.slider(key="konu09_first_stage_jump").set_value(0.10).run()
    assert not app.exception
    assert any("İlk aşama zayıf" in item.value for item in app.error)


def test_topic_10_bootstrap_methods_units_and_cps_gate() -> None:
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu10").run()
    assert not app.exception
    assert app.select_slider(key="konu10_repetitions").value == 500
    assert any(metric.label == "Bootstrap standart hata" for metric in app.metric)
    assert any(metric.label == "SH Monte Carlo hatası" for metric in app.metric)

    app.segmented_control(key="konu10_method").set_value("Wild").run()
    assert not app.exception
    assert app.segmented_control(key="konu10_method").value == "Wild"

    app.segmented_control(key="konu10_resampling_unit").set_value("Küme").run()
    assert not app.exception
    assert any("örnekleme birimi gözlem değil kümedir" in item.value for item in app.info)

    app.segmented_control(key="konu10_data_source").set_value(
        "Hazırlanmış CPS CSV"
    ).run()
    assert any("CPS verisi lisans teyidi" in item.value for item in app.info)


def test_topic_11_regularization_cv_and_cps_gate() -> None:
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu11").run()
    assert not app.exception
    assert app.slider(key="konu11_nfeatures").value == 30
    assert app.segmented_control(key="konu11_lambda_rule").value == "lambda_1se"
    assert any(metric.label == "lambda_min" for metric in app.metric)
    assert any(metric.label == "lambda_1se" for metric in app.metric)
    assert any("katın yalnız eğitim parçasında" in item.value for item in app.success)

    app.segmented_control(key="konu11_lambda_rule").set_value("lambda_min").run()
    assert not app.exception
    assert app.segmented_control(key="konu11_lambda_rule").value == "lambda_min"

    app.segmented_control(key="konu11_data_source").set_value(
        "Hazırlanmış CPS CSV"
    ).run()
    assert any("CPS verisi lisans teyidi" in item.value for item in app.info)


def test_topic_12_dml_cross_fitting_workflow_and_ddk_gate() -> None:
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu12").run()
    assert not app.exception
    assert app.select_slider(key="konu12_folds").value == 5
    assert app.selectbox(key="konu12_learner").value == "Ridge"
    assert any(metric.label == "DML hedef katsayısı" for metric in app.metric)
    assert any(metric.label == "Küme standart hata" for metric in app.metric)
    assert any("yalnız kat-dışı artıkları" in item.value for item in app.success)
    assert any("DDK verisi lisans teyidi" in item.value for item in app.info)
    assert app.multiselect(key="konu12_workflow").value == [
        "estimand", "identification", "data", "cross_fit", "estimate"
    ]

    app.number_input(key="konu12_seed").set_value(919).run()
    assert not app.exception
    assert app.number_input(key="konu12_seed").value == 919
