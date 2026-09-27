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


def test_topic_09_has_three_tabs_lab_and_experiments(monkeypatch) -> None:
    monkeypatch.delenv("IKT807_HANSEN_LM2007_PATH", raising=False)
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu09").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs][:3] == ["Uygulama", "Sezgi", "Kendini sına"]
    assert app.segmented_control(key="konu09_lab_step").value == 1
    assert any("LM2007.dta" in item.value for item in app.markdown)
    assert any(metric.label == "Global polinom: yanlılık" for metric in app.metric)

    app.slider(key="konu09_sezgi1_h").set_value(0.3).run()
    assert not app.exception
    assert any("yükselişe ulaşıyor" in item.value for item in app.info)

    app.segmented_control(key="konu09_sezgi_deney").set_value(3).run()
    assert not app.exception
    assert any(metric.label == "Medyan ilk aşama F" for metric in app.metric)
    app.slider(key="konu09_sezgi3_delta").set_value(0.1).run()
    assert not app.exception
    assert any("zayıf araç" in item.value for item in app.info)


def test_topic_10_has_three_tabs_lab_and_experiments(monkeypatch) -> None:
    # Veri yolu tanımlıysa laboratuvar açılışta 1.000 tekrarlı bootstrap yapar; bu test yalnız arayüzü sınar.
    for name in ("IKT807_HANSEN_CPS09MAR_PATH", "IKT807_CPS_PATH"):
        monkeypatch.delenv(name, raising=False)
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu10").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs][:3] == ["Uygulama", "Sezgi", "Kendini sına"]
    assert app.segmented_control(key="konu10_lab_step").value == 1
    assert any("cps09mar.txt" in item.value for item in app.markdown)
    pairs = next(metric for metric in app.metric if metric.label == "Pairs bootstrap SH")
    assert pairs.value == "0,077"

    app.slider(key="konu10_sezgi1_gamma").set_value(0.0).run()
    assert not app.exception
    assert any("homoskedastik" in item.value for item in app.info)

    app.segmented_control(key="konu10_sezgi_deney").set_value(3).run()
    assert not app.exception
    assert any(metric.label == "Küme bootstrap SH" for metric in app.metric)


def test_topic_11_has_three_tabs_lab_and_experiments(monkeypatch) -> None:
    # Veri yolu tanımlıysa laboratuvar açılışta CPS'de Ridge ve Lasso CV'si çalıştırır; bu test yalnız arayüzü sınar.
    for name in ("IKT807_HANSEN_CPS09MAR_PATH", "IKT807_CPS_PATH"):
        monkeypatch.delenv(name, raising=False)
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu11").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs][:3] == ["Uygulama", "Sezgi", "Kendini sına"]
    assert app.segmented_control(key="konu11_lab_step").value == 1
    assert any("cps09mar.txt" in item.value for item in app.markdown)
    post = next(metric for metric in app.metric if metric.label == "Post-Lasso test MSE")
    assert post.value == "2,164"

    app.segmented_control(key="konu11_sezgi_deney").set_value(2).run()
    assert not app.exception
    assert any(metric.label == "BIC'in seçtiği d" for metric in app.metric)

    app.segmented_control(key="konu11_sezgi_deney").set_value(3).run()
    assert not app.exception
    assert any(metric.label == "Kapsama: seçim sonrası" for metric in app.metric)
    app.slider(key="konu11_sezgi3_beta2").set_value(0.0).run()
    assert not app.exception
    assert any("kısa model doğrudur" in item.value for item in app.info)


def test_topic_12_has_three_tabs_lab_and_experiments(monkeypatch) -> None:
    # Veri yolu tanımlıysa laboratuvar açılışta 11 bölmeli DML hesabı yapar; bu test yalnız arayüzü sınar.
    monkeypatch.delenv("IKT807_HANSEN_DDK2011_PATH", raising=False)
    app = _run_app()
    app.radio(key="topic_selector").set_value("konu12").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs][:3] == ["Uygulama", "Sezgi", "Kendini sına"]
    assert app.segmented_control(key="konu12_lab_step").value == 1
    assert any("DDK2011.dta" in item.value for item in app.markdown)
    dml = next(metric for metric in app.metric if metric.label == "DML θ̂")
    assert dml.value == "1,032"

    app.segmented_control(key="konu12_sezgi_deney").set_value(2).run()
    assert not app.exception
    assert any(metric.label == "Ortogonal yanlılık" for metric in app.metric)
    app.slider(key="konu12_sezgi2_delta").set_value(0.0).run()
    assert not app.exception
    assert any("iki moment de yansızdır" in item.value for item in app.info)

    app.segmented_control(key="konu12_sezgi_deney").set_value(3).run()
    assert not app.exception
    assert any(metric.label == "DML kapsama" for metric in app.metric)
