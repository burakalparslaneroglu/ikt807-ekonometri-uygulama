"""Konu 07: check-loss, kantil doğruları ve katsayı profilleri."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.datasets import configured_cps_path, load_cps_csv
from core.inference import coefficient_inference
from core.ols import fit_ols
from core.quantile import (
    QuantileDGPConfig,
    QuantileFit,
    check_loss,
    fit_quantile,
    quantile_profile,
    simulate_quantile_data,
)
from topics.regression_ui import render_model_context, show_figure, style_figure
from topics.shared import render_question, render_reproduction_code, render_topic_header


TOPIC_KEY = "konu07"
SEED = 807


@st.cache_data(show_spinner=False)
def _simulation(nobs: int, scale_slope: float) -> pd.DataFrame:
    values = simulate_quantile_data(
        QuantileDGPConfig(nobs=nobs, seed=SEED, scale_slope=scale_slope)
    )
    return pd.DataFrame(values)


@st.cache_data(show_spinner=False)
def _profile(
    y: tuple[float, ...],
    x: tuple[float, ...],
    taus: tuple[float, ...],
) -> tuple[QuantileFit, ...]:
    return quantile_profile(np.asarray(y), np.asarray(x), ("x",), taus)


def _render_check_loss(tau: float) -> None:
    residuals = np.linspace(-3, 3, 241)
    losses = check_loss(residuals, tau)
    figure = go.Figure(
        go.Scatter(
            x=residuals,
            y=losses,
            mode="lines",
            line={"color": "#107C89", "width": 4},
            name=f"τ = {tau:.2f}",
        )
    )
    figure.add_vline(x=0, line_color="#51696C", line_dash="dot")
    style_figure(
        figure,
        title="Asimetrik check-loss",
        x_title="Artık u = y - xβ",
        y_title="ρτ(u)",
    )
    show_figure(figure)
    columns = st.columns(3)
    columns[0].metric("Pozitif artık eğimi", f"{tau:.2f}")
    columns[1].metric("Negatif artık eğimi", f"{1 - tau:.2f}")
    columns[2].metric("Hedef", f"Koşullu {tau:.0%} kantil")
    st.info(
        "Pozitif artık, gerçekleşenin tahminden yüksek olduğu gözlemdir. τ büyüdükçe "
        "bu eksik tahminler daha ağır cezalandırılır ve hedef dağılımın üstüne taşınır."
    )


def _fit_selected(frame: pd.DataFrame, tau: float) -> QuantileFit:
    return fit_quantile(frame["y"], frame["x"], ("x",), tau)


def _render_lines(frame: pd.DataFrame, tau: float) -> None:
    ols = fit_ols(frame["y"], frame[["x"]], ("x",))
    selected = _fit_selected(frame, tau)
    comparison_tau = 0.25 if tau >= 0.5 else 0.75
    comparison = _fit_selected(frame, comparison_tau)
    grid = np.linspace(frame["x"].min(), frame["x"].max(), 160)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["x"],
            y=frame["y"],
            mode="markers",
            name="Gözlem",
            marker={"size": 5, "opacity": 0.24, "color": "#51696C"},
        )
    )
    lines = (
        (
            "OLS: koşullu ortalama",
            ols.coefficient("Sabit"),
            ols.coefficient("x"),
            "#B3392F",
            "dash",
        ),
        (
            f"QR τ={tau:.2f}",
            selected.coefficient("Sabit"),
            selected.coefficient("x"),
            "#107C89",
            "solid",
        ),
        (
            f"QR τ={comparison_tau:.2f}",
            comparison.coefficient("Sabit"),
            comparison.coefficient("x"),
            "#2F9E6B",
            "dot",
        ),
    )
    for name, intercept, slope, color, dash in lines:
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=intercept + slope * grid,
                mode="lines",
                name=name,
                line={"color": color, "dash": dash, "width": 3},
            )
        )
    style_figure(
        figure,
        title="Aynı veri, farklı dağılımsal hedefler",
        x_title="X",
        y_title="Sonuç Y",
    )
    show_figure(figure)
    metrics = st.columns(3)
    metrics[0].metric("OLS x eğimi", f"{ols.coefficient('x'):.3f}")
    metrics[1].metric(f"τ={tau:.2f} x eğimi", f"{selected.coefficient('x'):.3f}")
    metrics[2].metric("QR yakınsama", "Evet" if selected.converged else "Hayır")
    st.warning(
        "OLS ve kantil regresyonun farklı sonuç vermesi tek başına model hatası değildir. "
        "Biri koşullu ortalamayı, diğeri seçilen koşullu kantili hedefler."
    )


def _render_profile(frame: pd.DataFrame) -> None:
    taus = tuple(np.round(np.linspace(0.1, 0.9, 9), 2))
    fits = _profile(tuple(frame["y"]), tuple(frame["x"]), taus)
    slopes = np.array([fit.coefficient("x") for fit in fits])
    errors = np.array([fit.standard_error("x") for fit in fits])
    ols = fit_ols(frame["y"], frame[["x"]], ("x",))
    ols_inference = coefficient_inference(ols, "x", "HC1")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=taus,
            y=slopes,
            mode="lines+markers",
            name="Kantil x katsayısı",
            line={"color": "#107C89", "width": 3},
            error_y={"type": "data", "array": 1.96 * errors, "visible": True},
        )
    )
    figure.add_hline(
        y=ols.coefficient("x"),
        line_color="#B3392F",
        line_dash="dash",
        annotation_text="OLS ortalama eğimi",
    )
    style_figure(
        figure,
        title="τ boyunca x katsayısı ve yaklaşık %95 aralık",
        x_title="Kantil düzeyi τ",
        y_title="x katsayısı",
    )
    show_figure(figure)
    st.caption(
        f"OLS HC1 standart hatası: {ols_inference.standard_error:.3f}. Kantil aralıkları "
        "Statsmodels'in robust yoğunluk tahminine dayanır; uç kantillerde veri desteği azalır."
    )
    render_model_context(
        data_label="Kontrollü heteroskedastik kantil DGP'si",
        sample_label=f"n = {len(frame):,}; x ∈ [0, 10]",
        model_label="Y'nin doğrusal koşullu kantili ~ X",
        inference_label="QR robust SH; OLS HC1 karşılaştırması",
        seed=SEED,
    )


def _load_cps(
    default_frame: pd.DataFrame,
) -> tuple[pd.DataFrame | None, str, tuple[str, ...]]:
    source = st.segmented_control(
        "Veri kaynağı",
        options=("Kontrollü DGP", "Hazırlanmış CPS CSV"),
        default="Kontrollü DGP",
        key="konu07_data_source",
    )
    if source == "Kontrollü DGP":
        frame = default_frame.rename(columns={"x": "education", "y": "lwage"}).copy()
        return frame, "Kontrollü heteroskedastik ücret DGP'si", ("education",)

    uploaded = st.file_uploader(
        "Hazırlanmış CPS CSV", type=("csv",), key="konu07_cps_upload"
    )
    configured = configured_cps_path()
    try:
        if uploaded is not None:
            cps = load_cps_csv(uploaded.getvalue())
            label = "CPS 2009 kullanıcı dosyası"
        elif configured is not None:
            cps = load_cps_csv(configured)
            label = "CPS 2009 ortam değişkeni"
        else:
            st.info(
                "CPS verisi lisans teyidi olmadan depoya eklenmez. Hazırlanmış dosya "
                "bu oturumda yüklenebilir veya IKT807_CPS_PATH ile gösterilebilir."
            )
            return None, "CPS 2009", ("education", "experience", "experience2_100")
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"CPS dosyası doğrulanamadı: {error}")
        return None, "CPS 2009", ("education", "experience", "experience2_100")

    columns = ("education", "experience", "experience2_100")
    cps = cps.dropna(subset=["lwage", *columns])
    if len(cps) > 5000:
        cps = cps.sample(5000, random_state=SEED).sort_index()
        label += " (sabit 5.000 gözlemlik hesaplama alt örneklemi)"
    return cps, label, columns


def _render_cps_lab(default_frame: pd.DataFrame) -> None:
    frame, label, features = _load_cps(default_frame)
    if frame is None:
        return
    rows = []
    for tau in (0.25, 0.5, 0.75):
        fit = fit_quantile(frame["lwage"], frame[list(features)], features, tau)
        rows.append(
            {
                "Hedef": f"Koşullu {tau:.0%} kantil",
                "Eğitim katsayısı": fit.coefficient("education"),
                "Robust SH": fit.standard_error("education"),
                "Gözlem": fit.nobs,
            }
        )
    ols = fit_ols(frame["lwage"], frame[list(features)], features)
    ols_ci = coefficient_inference(ols, "education", "HC1")
    rows.append(
        {
            "Hedef": "Koşullu ortalama (OLS)",
            "Eğitim katsayısı": ols.coefficient("education"),
            "Robust SH": ols_ci.standard_error,
            "Gözlem": ols.nobs,
        }
    )
    table = pd.DataFrame(rows)
    st.dataframe(
        table.style.format(
            {
                "Eğitim katsayısı": "{:.4f}",
                "Robust SH": "{:.4f}",
                "Gözlem": "{:,.0f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.info(
        "Tablodaki sütunlar aynı katsayının alternatif standart hataları değildir; "
        "ücret dağılımının farklı koşullu konumlarını izleyen ayrı tahmin hedefleridir."
    )
    render_model_context(
        data_label=label,
        sample_label=f"n = {len(frame):,}",
        model_label="log ücret ~ eğitim + deneyim + deneyim² / DGP karşılığı",
        inference_label="QR robust SH; OLS HC1",
        seed=SEED if label.startswith("Kontrollü") else None,
    )


def render() -> None:
    render_topic_header(TOPIC_KEY)
    controls = st.columns(3)
    nobs = controls[0].slider(
        "Gözlem sayısı", 300, 1600, 900, 100, key="konu07_nobs"
    )
    tau = controls[1].slider(
        "Kantil düzeyi τ", 0.10, 0.90, 0.50, 0.05, key="konu07_tau"
    )
    scale_slope = controls[2].slider(
        "Dağılımsal heterojenlik",
        0.0,
        0.30,
        0.16,
        0.02,
        key="konu07_scale_slope",
    )
    frame = _simulation(nobs, scale_slope)

    loss_tab, lines_tab, profile_tab, cps_tab = st.tabs(
        ("Kantil kaybı", "Aynı veri, farklı hedef", "Katsayı profili", "CPS laboratuvarı")
    )
    with loss_tab:
        _render_check_loss(tau)
        render_reproduction_code(TOPIC_KEY, "kayip")
    with lines_tab:
        _render_lines(frame, tau)
        render_reproduction_code(TOPIC_KEY, "hedef")
    with profile_tab:
        _render_profile(frame)
        render_reproduction_code(TOPIC_KEY, "profil")
    with cps_tab:
        _render_cps_lab(frame)
        render_reproduction_code(TOPIC_KEY, "veri")
    render_question(TOPIC_KEY)
