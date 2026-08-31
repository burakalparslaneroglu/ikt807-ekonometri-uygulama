"""Konu 02: FWL, güvenilir çıkarım, fonksiyonel biçim ve etki tanıları."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.diagnostics import influence_diagnostics
from core.functional_forms import (
    exact_percent_change,
    interaction_slope,
    quadratic_marginal_effect,
)
from core.inference import coefficient_inference
from core.ols import fit_ols, fwl_coefficient
from core.simulation import (
    WageDGPConfig,
    add_influential_observation,
    simulate_wage_data,
)
from topics.regression_ui import render_model_context, show_figure, style_figure
from topics.shared import render_question, render_reproduction_code, render_topic_header


TOPIC_KEY = "konu02"
SEED = 808
FEATURES = ("education", "experience", "experience2_100", "female")


@st.cache_data(show_spinner=False)
def _simulation(
    nobs: int,
    heteroskedasticity: float,
    cluster_correlation: float,
) -> pd.DataFrame:
    return simulate_wage_data(
        WageDGPConfig(
            nobs=nobs,
            seed=SEED,
            heteroskedasticity=heteroskedasticity,
            cluster_correlation=cluster_correlation,
            clusters=40,
        )
    )


def _render_fwl(frame: pd.DataFrame) -> None:
    result = fwl_coefficient(
        frame["lwage"],
        frame["education"],
        frame[["experience", "experience2_100", "female"]],
    )
    grid = np.linspace(
        result.residualized_treatment.min(),
        result.residualized_treatment.max(),
        100,
    )
    figure = go.Figure(
        go.Scattergl(
            x=result.residualized_treatment,
            y=result.residualized_outcome,
            mode="markers",
            name="Kontrollerden arındırılmış gözlemler",
            marker={"size": 6, "opacity": 0.45, "color": "#107C89"},
            hovertemplate=(
                "Artık eğitim: %{x:.3f}<br>Artık log ücret: %{y:.3f}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=grid,
            y=result.residual_regression_coefficient * grid,
            mode="lines",
            name="Artık üzerine artık OLS",
            line={"width": 3, "color": "#B3392F"},
        )
    )
    style_figure(
        figure,
        title="FWL: aynı eğitim katsayısına iki yol",
        x_title="Kontrollerden arındırılmış eğitim (yıl)",
        y_title="Kontrollerden arındırılmış log ücret",
    )
    show_figure(figure)
    direct, residual, difference = st.columns(3)
    direct.metric("Çoklu OLS", f"{result.direct_coefficient:.6f}")
    residual.metric(
        "FWL artık regresyonu", f"{result.residual_regression_coefficient:.6f}"
    )
    difference.metric(
        "Mutlak fark",
        f"{abs(result.direct_coefficient - result.residual_regression_coefficient):.2e}",
    )
    st.info(
        "FWL açıklayıcı değişkeni modelden çıkarmaz. Eğitim ve sonucu aynı kontrol "
        "setine göre artıklaştırır; artıklar arasındaki eğim çoklu OLS eğitim "
        "katsayısıyla aynıdır."
    )


def _inference_table(frame: pd.DataFrame):
    fit = fit_ols(frame["lwage"], frame[list(FEATURES)], FEATURES)
    rows = []
    for covariance_type in ("Klasik", "HC1", "Küme"):
        inference = coefficient_inference(
            fit,
            "education",
            covariance_type,
            groups=frame["cluster"] if covariance_type == "Küme" else None,
        )
        rows.append(
            {
                "Kovaryans": covariance_type,
                "Eğitim katsayısı": inference.estimate,
                "Standart hata": inference.standard_error,
                "%95 GA alt": inference.confidence_interval[0],
                "%95 GA üst": inference.confidence_interval[1],
                "p-değeri": inference.p_value,
            }
        )
    return pd.DataFrame(rows), fit


def _render_inference(frame: pd.DataFrame) -> None:
    table, fit = _inference_table(frame)
    selected = st.segmented_control(
        "Raporlanan çıkarım",
        options=("Klasik", "HC1", "Küme"),
        default="HC1",
        key="konu02_covariance",
    )
    st.dataframe(
        table.style.format(
            {
                "Eğitim katsayısı": "{:.5f}",
                "Standart hata": "{:.5f}",
                "%95 GA alt": "{:.5f}",
                "%95 GA üst": "{:.5f}",
                "p-değeri": lambda value: (
                    "< 0,001"
                    if value < 0.001
                    else f"{value:.3f}".replace(".", ",")
                ),
            }
        ),
        width="stretch",
        hide_index=True,
    )

    figure = go.Figure(
        go.Scatter(
            x=table["Eğitim katsayısı"],
            y=table["Kovaryans"],
            mode="markers",
            name="Eğitim katsayısı",
            marker={
                "size": 10,
                "color": ["#51696C", "#107C89", "#2F9E6B"],
            },
            error_x={
                "type": "data",
                "symmetric": False,
                "array": table["%95 GA üst"] - table["Eğitim katsayısı"],
                "arrayminus": table["Eğitim katsayısı"] - table["%95 GA alt"],
            },
        )
    )
    style_figure(
        figure,
        title="Aynı nokta tahmini, farklı belirsizlik ölçüleri",
        x_title="Eğitim katsayısı ve %95 güven aralığı",
        y_title="Kovaryans tahmincisi",
    )
    show_figure(figure)
    chosen = table.loc[table["Kovaryans"].eq(selected)].iloc[0]
    st.metric("Raporlanan standart hata", f"{chosen['Standart hata']:.5f}")
    st.warning(
        "Kovaryans tercihi OLS katsayısını değiştirmez ve gözlenmeyen içselliği "
        "çözmez. Küme seçimi, veri üretimindeki bağımlılık birimine dayanmalıdır."
    )
    render_model_context(
        data_label="Heteroskedastik ve kümeli kontrollü DGP",
        sample_label=f"n = {fit.nobs:,}; 40 küme",
        model_label="log ücret ~ eğitim + deneyim + deneyim² + kadın",
        inference_label=str(selected),
        seed=SEED,
    )


def _render_functional_forms(frame: pd.DataFrame) -> None:
    form = st.selectbox(
        "Fonksiyonel biçim",
        options=("Log sonuç", "Karesel deneyim", "Eğitim × kadın etkileşimi"),
        key="konu02_functional_form",
    )
    if form == "Log sonuç":
        fit = fit_ols(frame["lwage"], frame[list(FEATURES)], FEATURES)
        coefficient = fit.coefficient("education")
        exact = exact_percent_change(coefficient)
        approximation = 100 * coefficient
        first, second = st.columns(2)
        first.metric("Tam yüzde değişim", f"%{exact:.2f}")
        second.metric("100 × katsayı yaklaşımı", f"%{approximation:.2f}")
        st.write(
            "Bir yıllık eğitim artışı için diğer değişkenler sabitken log ücret "
            "değişimi katsayıdır; ücret düzeyindeki tam dönüşüm "
            "100 × [exp(β) - 1] ile hesaplanır."
        )
    elif form == "Karesel deneyim":
        point = st.slider(
            "Değerlendirme deneyimi (yıl)",
            min_value=0,
            max_value=35,
            value=15,
            key="konu02_experience_point",
        )
        fit = fit_ols(frame["lwage"], frame[list(FEATURES)], FEATURES)
        effect = quadratic_marginal_effect(
            fit.coefficient("experience"),
            fit.coefficient("experience2_100") / 100,
            point,
        )
        st.metric(f"{point} yılda marjinal log ücret eğimi", f"{effect:.4f}")
        st.write(
            "Karesel modelde deneyim katsayısı tek başına sabit bir etki değildir; "
            "eğim değerlendirme noktasına bağlıdır."
        )
    else:
        working = frame.assign(
            education_female=frame["education"] * frame["female"]
        )
        columns = FEATURES + ("education_female",)
        fit = fit_ols(working["lwage"], working[list(columns)], columns)
        male = interaction_slope(
            fit.coefficient("education"),
            fit.coefficient("education_female"),
            0,
        )
        female = interaction_slope(
            fit.coefficient("education"),
            fit.coefficient("education_female"),
            1,
        )
        first, second = st.columns(2)
        first.metric("Erkekler için eğitim eğimi", f"{male:.4f}")
        second.metric("Kadınlar için eğitim eğimi", f"{female:.4f}")
        st.write(
            "Etkileşim katsayısı kadın grubundaki ek eğimdir; kadın grubunun "
            "toplam eğimi ana etki ile etkileşimin toplamıdır."
        )


def _render_influence(frame: pd.DataFrame) -> None:
    add_extreme = st.toggle(
        "Kontrollü etkili gözlemi ekle",
        value=False,
        key="konu02_add_influential",
    )
    working = add_influential_observation(frame) if add_extreme else frame
    fit = fit_ols(working["lwage"], working[list(FEATURES)], FEATURES)
    diagnostics = influence_diagnostics(fit)
    labels = np.arange(len(working))
    if add_extreme:
        labels[-1] = -1
    figure = go.Figure(
        go.Scatter(
            x=diagnostics.leverage,
            y=diagnostics.internally_studentized_residuals,
            mode="markers",
            name="Gözlemler",
            marker={
                "size": np.clip(7 + 28 * diagnostics.cooks_distance, 7, 26),
                "color": diagnostics.cooks_distance,
                "colorscale": [[0, "#107C89"], [1, "#B3392F"]],
                "showscale": True,
                "colorbar": {"title": "Cook"},
                "opacity": 0.7,
            },
            customdata=np.column_stack((labels, diagnostics.cooks_distance)),
            hovertemplate=(
                "Gözlem: %{customdata[0]}<br>Kaldıraç: %{x:.3f}<br>"
                "Studentized artık: %{y:.2f}<br>Cook: %{customdata[1]:.3f}"
                "<extra></extra>"
            ),
        )
    )
    style_figure(
        figure,
        title="Kaldıraç ve sonuç üzerindeki etki",
        x_title="Kaldıraç (hat matrisi köşegeni)",
        y_title="İçsel studentized artık",
    )
    show_figure(figure)
    maximum = int(np.argmax(diagnostics.cooks_distance))
    baseline_fit = fit_ols(frame["lwage"], frame[list(FEATURES)], FEATURES)
    columns = st.columns(3)
    columns[0].metric("En yüksek kaldıraç", f"{diagnostics.leverage.max():.3f}")
    columns[1].metric(
        "En yüksek Cook uzaklığı", f"{diagnostics.cooks_distance.max():.3f}"
    )
    columns[2].metric(
        "Eğitim katsayısı değişimi",
        f"{fit.coefficient('education') - baseline_fit.coefficient('education'):+.4f}",
    )
    if add_extreme:
        st.warning(
            f"Eklenen gözlem en etkili gözlem olarak belirlendi (satır {maximum + 1}). "
            "Sonuç onun silinmesine göre ayrıca raporlanmalıdır."
        )


def render() -> None:
    render_topic_header(TOPIC_KEY)
    controls = st.columns(3)
    nobs = controls[0].slider(
        "Gözlem sayısı",
        min_value=400,
        max_value=2000,
        value=1000,
        step=200,
        key="konu02_nobs",
    )
    heteroskedasticity = controls[1].slider(
        "Heteroskedastisite şiddeti",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.2,
        key="konu02_heteroskedasticity",
    )
    cluster_correlation = controls[2].slider(
        "Küme içi bağımlılık",
        min_value=0.0,
        max_value=0.8,
        value=0.3,
        step=0.1,
        key="konu02_cluster_correlation",
    )
    frame = _simulation(nobs, heteroskedasticity, cluster_correlation)

    fwl_tab, inference_tab, forms_tab, influence_tab = st.tabs(
        ("FWL", "Güvenilir çıkarım", "Fonksiyonel biçim", "Etkili gözlem")
    )
    with fwl_tab:
        _render_fwl(frame)
        render_reproduction_code(TOPIC_KEY, "fwl")
    with inference_tab:
        _render_inference(frame)
        render_reproduction_code(TOPIC_KEY, "cikarim")
    with forms_tab:
        _render_functional_forms(frame)
        render_reproduction_code(TOPIC_KEY, "bicim")
    with influence_tab:
        _render_influence(frame)
        render_reproduction_code(TOPIC_KEY, "etkili")
    render_question(TOPIC_KEY)
