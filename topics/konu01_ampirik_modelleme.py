"""Konu 01: koşullu ortalama, doğrusal projeksiyon ve OLS geometrisi."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.datasets import configured_cps_path, load_cps_csv
from core.inference import coefficient_inference
from core.ols import fit_ols, projection_diagnostics
from core.simulation import WageDGPConfig, simulate_wage_data
from topics.regression_ui import render_model_context, show_figure, style_figure
from topics.shared import render_question, render_reproduction_code, render_topic_header


TOPIC_KEY = "konu01"
SEED = 807


@st.cache_data(show_spinner=False)
def _simulation(
    nobs: int,
    nonlinear_strength: float,
    ability_confounding: float,
) -> pd.DataFrame:
    return simulate_wage_data(
        WageDGPConfig(
            nobs=nobs,
            seed=SEED,
            nonlinear_strength=nonlinear_strength,
            ability_confounding=ability_confounding,
        )
    )


def _specification_fits(frame: pd.DataFrame):
    specifications = (
        ("(1) Yalnız eğitim", ("education",)),
        ("(2) + deneyim profili", ("education", "experience", "experience2_100")),
        (
            "(3) + cinsiyet",
            ("education", "experience", "experience2_100", "female"),
        ),
    )
    return tuple(
        (label, fit_ols(frame["lwage"], frame[list(columns)], columns))
        for label, columns in specifications
    )


def _render_mechanism(frame: pd.DataFrame, ability_confounding: float) -> None:
    simple_fit = fit_ols(frame["lwage"], frame[["education"]], ("Eğitim",))
    grouped = frame.groupby("education", as_index=False).agg(
        ortalama_log_ucret=("lwage", "mean"),
        gozlem=("lwage", "size"),
    )
    grid = np.linspace(grouped["education"].min(), grouped["education"].max(), 100)
    line = simple_fit.coefficient("Sabit") + simple_fit.coefficient("Eğitim") * grid

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=grouped["education"],
            y=grouped["ortalama_log_ucret"],
            mode="markers+lines",
            name="Örneklem koşullu ortalaması",
            marker={"size": np.sqrt(grouped["gozlem"]) * 2.2, "symbol": "circle"},
            customdata=grouped[["gozlem"]],
            hovertemplate=(
                "Eğitim: %{x:.0f} yıl<br>Ortalama: %{y:.3f}<br>"
                "n: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=grid,
            y=line,
            mode="lines",
            name="OLS doğrusal projeksiyonu",
            line={"width": 3, "dash": "dash", "color": "#B3392F"},
        )
    )
    style_figure(
        figure,
        title="Kontrollü DGP: koşullu ortalama ve doğrusal projeksiyon",
        x_title="Tamamlanan eğitim (yıl)",
        y_title="Ortalama log saatlik ücret (log birim)",
    )
    show_figure(figure)

    first, second = st.columns(2)
    first.metric("OLS eğitim katsayısı", f"{simple_fit.coefficient('Eğitim'):.3f}")
    first.caption(
        "Tahmin hedefi: log ücretin eğitim üzerindeki en iyi doğrusal projeksiyonu."
    )
    second.metric("DGP doğrusal bileşeni", "0,080")
    second.caption(
        "Bu sayı veri üretim sürecinde sabittir; OLS ile aynı nesne olmak zorunda değildir."
    )
    if ability_confounding > 0:
        st.warning(
            "Yetenek hem eğitimle ilişkili hem ücret denkleminde etkilidir. OLS ilişkisi "
            "görünür, fakat eğitim katsayısına nedensel getiri denemez."
        )
    render_model_context(
        data_label="Kontrollü ücret DGP'si",
        sample_label=f"n = {len(frame):,}",
        model_label="log ücret ~ eğitim",
        inference_label="Bu görünümde nokta tahmini",
        seed=SEED,
    )


def _render_geometry(frame: pd.DataFrame) -> None:
    columns = ("education", "experience", "experience2_100", "female")
    fit = fit_ols(frame["lwage"], frame[list(columns)], columns)
    diagnostics = projection_diagnostics(fit)

    figure = go.Figure(
        go.Scattergl(
            x=fit.fitted_values,
            y=fit.residuals,
            mode="markers",
            name="Gözlemler",
            marker={"size": 6, "opacity": 0.55, "color": "#107C89"},
            hovertemplate=(
                "Uydurulan: %{x:.3f}<br>Artık: %{y:.3f}<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0, line_dash="dash", line_color="#B3392F")
    style_figure(
        figure,
        title="OLS geometrisi: uydurulan değer ile artık",
        x_title="Uydurulan log saatlik ücret",
        y_title="OLS artığı (log birim)",
    )
    show_figure(figure)

    columns_ui = st.columns(3)
    columns_ui[0].metric("Ortalama artık", f"{diagnostics.mean_residual:.2e}")
    columns_ui[1].metric(
        "En büyük |X'e|", f"{diagnostics.max_regressor_residual_product:.2e}"
    )
    columns_ui[2].metric(
        "|ŷ'e|", f"{abs(diagnostics.fitted_residual_product):.2e}"
    )
    st.info(
        "Sabit terimli OLS'de artıkların ortalaması sıfırdır; artıklar modeldeki "
        "açıklayıcı değişkenlere ve uydurulan değerlere örneklem içinde ortogonaldir."
    )


def _load_lab_frame(default_frame: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    source = st.segmented_control(
        "Veri kaynağı",
        options=("Kontrollü DGP", "Hazırlanmış CPS CSV"),
        default="Kontrollü DGP",
        key="konu01_data_source",
    )
    if source == "Kontrollü DGP":
        return default_frame, "Kontrollü ücret DGP'si"

    configured = configured_cps_path()
    uploaded = st.file_uploader(
        "Hazırlanmış CPS CSV",
        type=("csv",),
        key="konu01_cps_upload",
    )
    try:
        if uploaded is not None:
            return load_cps_csv(uploaded.getvalue()), "CPS 2009, kullanıcı dosyası"
        if configured is not None:
            return load_cps_csv(configured), "CPS 2009, ortam değişkeni"
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"CPS dosyası doğrulanamadı: {error}")
        return None, "CPS 2009"

    st.info(
        "Lisansı doğrulanmamış CPS verisi depoya eklenmez. Hazırlanmış dosya bu "
        "oturumda yüklenebilir veya IKT807_CPS_PATH ortam değişkeniyle gösterilebilir."
    )
    return None, "CPS 2009"


def _render_data_lab(default_frame: pd.DataFrame) -> None:
    frame, data_label = _load_lab_frame(default_frame)
    if frame is None:
        return

    rows = []
    for label, fit in _specification_fits(frame):
        inference = coefficient_inference(fit, "education", "HC1")
        rows.append(
            {
                "Spesifikasyon": label,
                "Eğitim katsayısı": inference.estimate,
                "HC1 standart hata": inference.standard_error,
                "%95 GA alt": inference.confidence_interval[0],
                "%95 GA üst": inference.confidence_interval[1],
                "R²": fit.r_squared,
                "Gözlem": fit.nobs,
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(
        table.style.format(
            {
                "Eğitim katsayısı": "{:.4f}",
                "HC1 standart hata": "{:.4f}",
                "%95 GA alt": "{:.4f}",
                "%95 GA üst": "{:.4f}",
                "R²": "{:.3f}",
                "Gözlem": "{:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    figure = go.Figure(
        go.Scatter(
            x=table["Eğitim katsayısı"],
            y=table["Spesifikasyon"],
            mode="markers",
            name="Eğitim katsayısı",
            marker={"size": 10, "color": "#107C89"},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": table["%95 GA üst"] - table["Eğitim katsayısı"],
                "arrayminus": table["Eğitim katsayısı"] - table["%95 GA alt"],
                "color": "#07373D",
            },
        )
    )
    style_figure(
        figure,
        title=f"{data_label}: eğitim katsayısının spesifikasyon duyarlılığı",
        x_title="Eğitim katsayısı ve %95 HC1 güven aralığı",
        y_title="Model spesifikasyonu",
    )
    show_figure(figure)
    render_model_context(
        data_label=data_label,
        sample_label=f"n = {len(frame):,}; eksiksiz hazırlanmış örneklem",
        model_label="log saatlik ücret; üç iç içe kontrol seti",
        inference_label="HC1 heteroskedastisite-dayanıklı",
        seed=SEED if data_label.startswith("Kontrollü") else None,
    )


def _render_output_reading(frame: pd.DataFrame) -> None:
    fits = dict(_specification_fits(frame))
    selected = st.selectbox(
        "İncelenen sütun",
        options=tuple(fits),
        index=2,
        key="konu01_output_model",
    )
    fit = fits[selected]
    inference = coefficient_inference(fit, "education", "HC1")
    st.markdown("#### Bir regresyon sütununu okuma sırası")
    st.write("1. Sonuç değişkeni ve ölçeği: log saatlik ücret.")
    st.write(f"2. Örneklem: {fit.nobs:,} gözlem; aynı DGP ve eksiksiz değişkenler.")
    st.write(
        f"3. Eğitim katsayısı: {inference.estimate:.4f}; tahmin hedefi seçilen kontrol "
        "setine bağlı doğrusal projeksiyon."
    )
    st.write(
        f"4. Belirsizlik: HC1 standart hata {inference.standard_error:.4f}; "
        f"%95 GA [{inference.confidence_interval[0]:.4f}, "
        f"{inference.confidence_interval[1]:.4f}]."
    )
    st.write("5. Yorum sınırı: kontrol eklemek tek başına nedensel tanımlama sağlamaz.")


def render() -> None:
    render_topic_header(TOPIC_KEY)
    controls = st.columns(3)
    nobs = controls[0].slider(
        "Gözlem sayısı",
        min_value=400,
        max_value=2000,
        value=800,
        step=200,
        key="konu01_nobs",
    )
    nonlinear = controls[1].slider(
        "Koşullu ortalama eğriliği",
        min_value=0.0,
        max_value=2.0,
        value=0.6,
        step=0.2,
        key="konu01_nonlinear",
    )
    confounding = controls[2].slider(
        "Gözlenmeyen yetenek etkisi",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.2,
        key="konu01_confounding",
    )
    frame = _simulation(nobs, nonlinear, confounding)

    mechanism, geometry, data_lab, output_reading = st.tabs(
        ("Mekanizma", "OLS geometrisi", "Veri laboratuvarı", "Çıktı okuma")
    )
    with mechanism:
        _render_mechanism(frame, confounding)
        render_reproduction_code(TOPIC_KEY, "mekanizma")
    with geometry:
        _render_geometry(frame)
        render_reproduction_code(TOPIC_KEY, "geometri")
    with data_lab:
        _render_data_lab(frame)
        render_reproduction_code(TOPIC_KEY, "veri")
    with output_reading:
        _render_output_reading(frame)
        render_reproduction_code(TOPIC_KEY, "cikti")
    render_question(TOPIC_KEY)
