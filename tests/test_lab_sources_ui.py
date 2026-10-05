"""Uygulama sekmesindeki veri kaynağı seçimi (Streamlit AppTest): notlardaki örnek, alternatif örnek, kendi verin.

Gerçek Hansen verisi gerektiren test, veri yolu verilmezse atlanır (``IKT807_HANSEN_<VERİ>_PATH``, ör.
``IKT807_HANSEN_CARD1995_PATH``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.hansen_data import DATASET_MEMBERS
from core.labs import kendi_veri as K
from core.labs.ornekler import VARIANTS

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
START = "Başlamak için bir dosya yükleyin."
STEPS = {"konu01": 9, "konu02": 7, "konu03": 5, "konu04": 5, "konu05": 6, "konu06": 5, "konu07": 5, "konu08": 5}
STATA_STEP = {"konu01": 5, "konu02": 5, "konu03": 2, "konu04": 2, "konu05": 1, "konu06": 2, "konu07": 1, "konu08": 2}
"""Kendi verinde Stata seçilince bilgi kutusunun denetlendiği adım (kodu olan bir adım)."""
SAMPLE_ROLES = {
    "konu01": {"rol_aciklayici": "Eğitim yılı", "kategori_grup": "Kadın"},
    "konu02": {"rol_aciklayici": "Eğitim yılı", "kategori_grup": "Kadın"},
    "konu03": {"rol_sonuc": "Son sınav puanı", "rol_tedavi": "Program", "rol_atama": "Okul",
               "rol_secilmis": "Düşük başlangıç grubu", "rol_secim": "Başlangıç puanı", "kategori_tedavi": "Var",
               "kategori_secilmis": "Evet"},
    "konu04": {"rol_sonuc": "Saatlik ücret (TL)", "rol_icsel": "Eğitim yılı",
               "rol_arac": "Üniversiteye yakınlık (1/0)"},
    "konu05": {"rol_sonuc": "İşe yerleşti", "rol_profil": "Yaş", "rol_gosterge": "Meslek kursu",
               "kategori_sonuc": "Evet", "kategori_gosterge": "Var"},
    "konu06": {"rol_sonuc": "Aylık bağış (TL)", "rol_aciklayici": "Hane geliri (bin TL)"},
    "konu07": {"rol_sonuc": "Saatlik ücret (TL)", "rol_aciklayici": "Eğitim yılı", "rol_kontrol": "Deneyim (yıl)",
               "rol_grup": "Cinsiyet", "kategori_grup": "Kadın"},
    "konu08": {"rol_sonuc": "Sınav puanı", "rol_aciklayici": "Haftalık çalışma saati", "rol_kume": "Okul"},
}
"""Örnek dosya yüklenince önerilen roller (``_guess`` ve konunun ``suggest`` önerileri)."""


def _app(monkeypatch, keep: tuple[str, ...] = ()) -> AppTest:
    for dataset in DATASET_MEMBERS:
        if dataset not in keep:
            monkeypatch.delenv(f"IKT807_HANSEN_{dataset.upper()}_PATH", raising=False)
    monkeypatch.delenv("IKT807_CPS_PATH", raising=False)
    return AppTest.from_file(APP_PATH, default_timeout=300).run()


def _data_path(dataset: str) -> str | None:
    value = os.environ.get(f"IKT807_HANSEN_{dataset.upper()}_PATH")
    return value if value and Path(value).is_file() else None


def _topic(app: AppTest, topic: str) -> None:
    app.radio(key="topic_selector").set_value(topic).run()


def _source(app: AppTest, topic: str, source: str) -> None:
    app.segmented_control(key=f"{topic}_lab_kaynak").set_value(source).run()


def _upload(app: AppTest, topic: str) -> None:
    data = K.sample_excel(VARIANTS[topic].custom.sample())
    app.file_uploader(key=f"{topic}_kendi_dosya").set_value(("ornek.xlsx", data, XLSX_MIME)).run()


def _markdown(app: AppTest) -> str:
    return "\n".join(item.value for item in app.markdown)


def _all_steps(app: AppTest, topic: str, languages: tuple[str, ...], label: str) -> None:
    for language in languages:
        app.segmented_control(key="code_language").set_value(language).run()
        for number in range(1, STEPS[topic] + 1):
            app.segmented_control(key=f"{topic}_lab_step").set_value(number).run()
            assert not app.exception, (label, language, number)
            assert not app.error, (label, language, number, [item.value for item in app.error])
            assert any(item.value.startswith(f"Adım {number}:") for item in app.subheader), (label, number)


def test_source_selector_opens_on_the_notes(monkeypatch) -> None:
    app = _app(monkeypatch)
    for topic in STEPS:
        _topic(app, topic)
        selector = app.segmented_control(key=f"{topic}_lab_kaynak")
        assert selector.value == "notlar"
        assert len(selector.options) == 3
        assert "ders notlarındaki" in _markdown(app) and "henüz yüklenmedi" in _markdown(app)
        assert not app.exception
    _topic(app, "konu09")
    assert "konu09_lab_kaynak" not in {item.key for item in app.segmented_control}


def test_alternative_shows_steps_and_code_before_its_data_is_loaded(monkeypatch) -> None:
    app = _app(monkeypatch)
    _source(app, "konu01", "alternatif")
    assert "başka bir gerçek veriyle yeniden yapar" in _markdown(app)
    assert any("`Card1995.dta` dosyası henüz yüklenmedi" in item.value for item in app.markdown)
    app.segmented_control(key="konu01_lab_step").set_value(5).run()
    assert 'smf.ols("lwage ~ ed76", data=card)' in "\n".join(block.value for block in app.code)
    app.segmented_control(key="konu01_lab_step").set_value(9).run()
    keys = {item.key for item in app.download_button}
    assert {f"konu01_lab_download_alternatif_{language}" for language in ("Python", "R", "Stata")} <= keys
    assert not app.exception
    _topic(app, "konu04")
    _source(app, "konu04", "alternatif")
    assert any("`AK1991.dta` dosyası henüz yüklenmedi" in item.value for item in app.markdown)
    app.segmented_control(key="konu04_lab_step").set_value(2).run()
    assert "[edu ~ q1]" in "\n".join(block.value for block in app.code)
    assert not app.exception


@pytest.mark.parametrize("topic", sorted(STEPS))
def test_every_alternative_step_renders_with_the_real_data(monkeypatch, topic: str) -> None:
    dataset = VARIANTS[topic].alternative().dataset
    if _data_path(dataset) is None:
        pytest.skip(f"Gerçek Hansen {DATASET_MEMBERS[dataset]} dosyası verilmedi.")
    app = _app(monkeypatch, keep=(dataset,))
    _topic(app, topic)
    _source(app, topic, "alternatif")
    _all_steps(app, topic, ("Python", "R", "Stata"), f"{topic} alternatif")
    assert not [frame for frame in app.dataframe if "Durum" in frame.value.columns]  # notlarla karşılaştırma yok
    if topic == "konu01":
        app.segmented_control(key="konu01_lab_step").set_value(6).run()
        metrics = {metric.label: metric.value for metric in app.metric}
        assert metrics["Model (3) eğitim katsayısının kesin yüzde karşılığı"] == "%8,52"
        assert metrics["Model (3) siyahi göstergesinin kesin yüzde karşılığı"] == "%−20,70"


def test_the_sample_file_runs_every_step_and_offers_python_and_r(monkeypatch) -> None:
    app = _app(monkeypatch)
    for topic in STEPS:
        _topic(app, topic)
        _source(app, topic, "kendi")
        assert any(START in item.value for item in app.info)
        _upload(app, topic)
        assert not app.exception and not app.error, [item.value for item in app.error]
        for key, value in SAMPLE_ROLES[topic].items():
            assert app.selectbox(key=f"{topic}_kendi_{key}").value == value, (topic, key)
        _all_steps(app, topic, ("Python", "R"), f"{topic} kendi")
        keys = {item.key for item in app.download_button}
        assert {f"{topic}_lab_download_kendi_Python", f"{topic}_lab_download_kendi_R"} <= keys
        assert f"{topic}_lab_download_kendi_Stata" not in keys
        app.segmented_control(key=f"{topic}_lab_step").set_value(STATA_STEP[topic]).run()
        app.segmented_control(key="code_language").set_value("Stata").run()
        assert any("Python ve R'da üretilir" in item.value for item in app.info)
        assert not app.exception and not app.error
        app.segmented_control(key="code_language").set_value("Python").run()


def test_the_konu03_sample_offers_its_baseline_variables_as_balance_columns(monkeypatch) -> None:
    """Seçilmiş grubun dayandığı başlangıç puanı aynı zamanda denge tablosunun bir değişkenidir (``Role.shared``)."""

    app = _app(monkeypatch)
    _topic(app, "konu03")
    _source(app, "konu03", "kendi")
    _upload(app, "konu03")
    extras = app.multiselect(key="konu03_kendi_ek")
    assert list(extras.value) == ["Başlangıç puanı", "Kız öğrenci (1/0)", "Yaş"]
    assert "Okul" not in extras.options and "Program" not in extras.options
    app.segmented_control(key="konu03_lab_step").set_value(3).run()
    assert not app.exception and not app.error
    assert any("1.030 → 969" in item.value for item in app.info)  # adımın sonuç notu


def test_source_file_and_choices_survive_switches_and_removing_the_file_clears_them(monkeypatch) -> None:
    app = _app(monkeypatch)
    _topic(app, "konu02")
    _source(app, "konu02", "kendi")
    _upload(app, "konu02")
    app.selectbox(key="konu02_kendi_kategori_grup").set_value("Erkek").run()
    app.segmented_control(key="konu02_lab_step").set_value(5).run()
    assert not app.exception and not app.error
    _source(app, "konu02", "alternatif")
    _source(app, "konu02", "kendi")
    assert app.selectbox(key="konu02_kendi_kategori_grup").value == "Erkek"
    assert app.segmented_control(key="konu02_lab_step").value == 5
    _topic(app, "konu01")
    _topic(app, "konu02")
    assert app.segmented_control(key="konu02_lab_kaynak").value == "kendi"
    assert app.selectbox(key="konu02_kendi_kategori_grup").value == "Erkek"
    assert any("Kullanılan dosya: ornek.xlsx" in item.value for item in app.caption)
    app.button(key="konu02_kendi_kaldir").click().run()
    assert not app.exception
    assert any(START in item.value for item in app.info)
    gone = ("konu02_kendi_yuklenen", "konu02_kendi_tablo", "konu02_kendi_uygulama", "konu02_kendi_hesap",
            "konu02_kendi_kategori_grup", "_kalici_konu02_kendi_kategori_grup")
    assert not [key for key in gone if key in app.session_state]


def test_the_logarithm_option_is_off_and_locked_for_a_nonpositive_outcome(monkeypatch) -> None:
    frame = VARIANTS["konu01"].custom.sample()
    frame["Saatlik ücret (TL)"] = frame["Saatlik ücret (TL)"] - 200
    app = _app(monkeypatch)
    _source(app, "konu01", "kendi")
    app.file_uploader(key="konu01_kendi_dosya").set_value(("negatif.xlsx", K.sample_excel(frame), XLSX_MIME)).run()
    assert not app.exception and not app.error
    toggle = app.toggle(key="konu01_kendi_secenek_log_pasif")
    assert toggle.value is False and toggle.disabled
    app.segmented_control(key="konu01_lab_step").set_value(6).run()
    assert not app.exception and not app.error
    assert not [metric for metric in app.metric if "kesin yüzde" in metric.label]  # yüzde dönüşüm yok


def test_the_step_survives_an_invalid_choice_and_a_topic_switch(monkeypatch) -> None:
    """Kendi verinde geçersiz bir seçim adımları çizdirmez; Streamlit adım seçimini siler. Gölge anahtar onu korur."""

    app = _app(monkeypatch)
    _source(app, "konu01", "kendi")
    _upload(app, "konu01")
    app.segmented_control(key="konu01_lab_step").set_value(6).run()
    app.selectbox(key="konu01_kendi_rol_kontrol").set_value("Eğitim yılı").run()
    assert app.error and not app.exception  # açıklayıcı ve kontrol aynı sütun
    app.selectbox(key="konu01_kendi_rol_kontrol").set_value("Deneyim (yıl)").run()
    assert not app.error and app.segmented_control(key="konu01_lab_step").value == 6
    _topic(app, "konu02")
    _topic(app, "konu01")
    assert app.segmented_control(key="konu01_lab_step").value == 6


def test_the_logarithm_option_looks_only_at_the_analysis_rows(monkeypatch) -> None:
    """Sonucu 0 olan satırın grup hücresi boşsa satır analizden çıkar; logaritma seçeneği kilitlenmez."""

    frame = VARIANTS["konu01"].custom.sample().astype({"Cinsiyet": object})
    frame.loc[0, "Saatlik ücret (TL)"] = 0.0
    frame.loc[0, "Cinsiyet"] = None
    app = _app(monkeypatch)
    _source(app, "konu01", "kendi")
    app.file_uploader(key="konu01_kendi_dosya").set_value(("sifir.xlsx", K.sample_excel(frame), XLSX_MIME)).run()
    assert not app.exception and not app.error
    assert app.toggle(key="konu01_kendi_secenek_log").value is True


def test_an_explanatory_column_named_like_a_table_column_renders(monkeypatch) -> None:
    """Kendi verinde açıklayıcı değişkenin adı koşullu ortalama tablosunun "N" sütunuyla aynı olabilir."""

    frame = VARIANTS["konu01"].custom.sample().rename(columns={"Eğitim yılı": "N"})
    app = _app(monkeypatch)
    _source(app, "konu01", "kendi")
    app.file_uploader(key="konu01_kendi_dosya").set_value(("n.xlsx", K.sample_excel(frame), XLSX_MIME)).run()
    if app.selectbox(key="konu01_kendi_rol_aciklayici").value != "N":
        app.selectbox(key="konu01_kendi_rol_aciklayici").set_value("N").run()
    app.segmented_control(key="konu01_lab_step").set_value(4).run()
    assert not app.exception and not app.error
    assert any("N (grup)" in frame.value.columns for frame in app.dataframe)
