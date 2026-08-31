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
    app = _run_app()
    app.button(key="konu01_toggle_answer").click().run()
    assert not app.exception
    assert app.session_state["konu01_answer_visible"] is True
    assert len(app.success) == 1

    app.button(key="konu01_new_question").click().run()
    assert app.session_state["konu01_answer_visible"] is False
    assert app.session_state["konu01_question_index"] == 1
    assert len(app.success) == 0


def test_topic_01_mechanism_and_private_data_mode() -> None:
    app = _run_app()
    assert not app.exception
    assert app.slider(key="konu01_nonlinear").value == 0.6
    assert any(metric.label == "OLS eğitim katsayısı" for metric in app.metric)

    app.slider(key="konu01_confounding").set_value(1.0).run()
    assert not app.exception
    assert any("nedensel getiri" in item.value for item in app.warning)

    app.segmented_control(key="konu01_data_source").set_value(
        "Hazırlanmış CPS CSV"
    ).run()
    assert not app.exception
    assert any("Lisansı doğrulanmamış CPS" in item.value for item in app.info)


def test_topic_02_regression_labs_are_interactive() -> None:
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu02").run()
    assert not app.exception
    assert app.slider(key="konu02_heteroskedasticity").value == 1.0
    assert any(metric.label == "Çoklu OLS" for metric in app.metric)

    app.segmented_control(key="konu02_covariance").set_value("Küme").run()
    assert not app.exception
    assert app.segmented_control(key="konu02_covariance").value == "Küme"

    app.selectbox(key="konu02_functional_form").set_value(
        "Eğitim × kadın etkileşimi"
    ).run()
    assert any(
        metric.label == "Kadınlar için eğitim eğimi" for metric in app.metric
    )

    app.toggle(key="konu02_add_influential").set_value(True).run()
    assert not app.exception
    assert any("en etkili gözlem" in item.value for item in app.warning)


def test_topic_03_identification_labs_and_data_gate() -> None:
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu03").run()
    assert not app.exception
    assert app.segmented_control(key="konu03_assignment").value == (
        "Seçime dayalı atama"
    )
    assert any(metric.label == "Seçim bileşeni" for metric in app.metric)

    app.segmented_control(key="konu03_assignment").set_value(
        "Rastgele atama"
    ).run()
    assert any("Rastgele atama" in item.value for item in app.success)

    app.segmented_control(key="konu03_trial_source").set_value(
        "Hazırlanmış DDK CSV"
    ).run()
    assert not app.exception
    assert any("DDK verisi lisans teyidi" in item.value for item in app.info)


def test_topic_04_iv_validity_and_card_data_gate() -> None:
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu04").run()
    assert not app.exception
    assert app.slider(key="konu04_strength").value == 0.35
    assert any("üç koşul birlikte" in item.value for item in app.success)
    assert any(metric.label == "İlk aşama" for metric in app.metric)

    app.slider(key="konu04_exclusion_violation").set_value(0.10).run()
    assert not app.exception
    assert any("dışlama kısıtı ihlal" in item.value for item in app.warning)
    assert any("koşullar birlikte sağlanmıyor" in item.value for item in app.error)

    app.segmented_control(key="konu04_data_source").set_value(
        "Hazırlanmış Card CSV"
    ).run()
    assert any("Card verisi lisans teyidi" in item.value for item in app.info)


def test_topic_05_binary_models_effects_and_cps_gate() -> None:
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu05").run()
    assert not app.exception
    assert app.slider(key="konu05_age_effect").value == 0.20
    assert any(metric.label == "LPM sınır dışı tahmin" for metric in app.metric)

    app.segmented_control(key="konu05_effect_target").set_value(
        "Metropol: 0 → 1"
    ).run()
    assert not app.exception
    assert any("kukla değişkendir" in item.value for item in app.warning)

    app.segmented_control(key="konu05_data_source").set_value(
        "Hazırlanmış CPS CSV"
    ).run()
    assert any("CPS verisi lisans teyidi" in item.value for item in app.info)


def test_topic_06_tobit_selection_and_chj_gate() -> None:
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu06").run()
    assert not app.exception
    assert app.slider(key="konu06_sigma").value == 1.0
    assert any(metric.label == "Sansürlenme oranı" for metric in app.metric)
    assert any("Tobit MLE yakınsama" in item.value for item in app.success)

    app.segmented_control(key="konu06_problem_type").set_value(
        "Örneklem seçimi"
    ).run()
    assert any("sonuç yalnız seçilen" in item.value for item in app.info)

    app.slider(key="konu06_exclusion_strength").set_value(0.0).run()
    assert not app.exception
    assert any("dışlanan değişken yok" in item.value for item in app.error)

    app.segmented_control(key="konu06_data_source").set_value(
        "Hazırlanmış CHJ CSV"
    ).run()
    assert any("CHJ verisi lisans teyidi" in item.value for item in app.info)


def test_topic_07_quantile_targets_and_cps_gate() -> None:
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu07").run()
    assert not app.exception
    assert app.slider(key="konu07_tau").value == 0.50
    assert any(metric.label == "Pozitif artık eğimi" for metric in app.metric)
    assert any(metric.label == "OLS x eğimi" for metric in app.metric)

    app.slider(key="konu07_tau").set_value(0.80).run()
    assert not app.exception
    assert app.slider(key="konu07_tau").value == 0.80

    app.segmented_control(key="konu07_data_source").set_value(
        "Hazırlanmış CPS CSV"
    ).run()
    assert any("CPS verisi lisans teyidi" in item.value for item in app.info)


def test_topic_08_smoothing_controls_and_ddk_gate() -> None:
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu08").run()
    assert not app.exception
    assert app.slider(key="konu08_bandwidth").value == 0.70
    assert any(metric.label == "Ağırlık toplamı" for metric in app.metric)
    assert any(
        metric.label == "Çapraz doğrulama bant genişliği"
        for metric in app.metric
    )
    assert any("DDK verisi lisans teyidi" in item.value for item in app.info)

    app.selectbox(key="konu08_kernel").set_value("Epanechnikov").run()
    assert not app.exception
    assert app.selectbox(key="konu08_kernel").value == "Epanechnikov"


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
