"""Konu 08: kernel, yerel regresyon, CV, seri ve partialling-out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.cross_validation import BandwidthCVResult, select_bandwidth
from core.datasets import load_registered_csv
from core.nonparametric import (
    LocalPolynomialResult,
    NonparametricDGPConfig,
    kernel_weights,
    local_polynomial_predict,
    simulate_nonlinear_data,
    spline_series_predict,
)
from core.partialling import partialling_out
from topics.regression_ui import render_model_context, show_figure, style_figure
from topics.shared import render_question, render_reproduction_code, render_topic_header


TOPIC_KEY = "konu08"
SEED = 808


@st.cache_data(show_spinner=False)
def _simulation(nobs: int, noise_scale: float) -> pd.DataFrame:
    return pd.DataFrame(
        simulate_nonlinear_data(
            NonparametricDGPConfig(nobs=nobs, seed=SEED, noise_scale=noise_scale)
        )
    )


@st.cache_data(show_spinner=False)
def _local_fit(
    x: tuple[float, ...],
    y: tuple[float, ...],
    grid: tuple[float, ...],
    bandwidth: float,
    degree: int,
    kernel: str,
) -> LocalPolynomialResult:
    return local_polynomial_predict(
        np.asarray(x),
        np.asarray(y),
        np.asarray(grid),
        bandwidth,
        degree=degree,
        kernel=kernel,
    )


@st.cache_data(show_spinner=False)
def _cv_fit(
    x: tuple[float, ...],
    y: tuple[float, ...],
    bandwidths: tuple[float, ...],
    kernel: str,
) -> BandwidthCVResult:
    return select_bandwidth(
        np.asarray(x),
        np.asarray(y),
        np.asarray(bandwidths),
        folds=5,
        seed=SEED,
        kernel=kernel,
    )


def _render_kernel_weights(
    frame: pd.DataFrame, bandwidth: float, kernel: str
) -> None:
    point = st.slider(
        "Değerlendirme noktası x₀",
        0.0,
        10.0,
        5.0,
        0.25,
        key="konu08_point",
    )
    weights = kernel_weights(frame["x"], point, bandwidth, kernel)
    order = np.argsort(np.abs(frame["x"].to_numpy() - point))[:100]
    figure = go.Figure(
        go.Scatter(
            x=frame["x"].to_numpy()[order],
            y=weights[order],
            mode="markers",
            marker={"color": "#107C89", "size": 7},
            name="Çekirdek ağırlığı",
        )
    )
    figure.add_vline(x=point, line_color="#B3392F", line_dash="dash")
    style_figure(
        figure,
        title="x₀ çevresindeki en yakın 100 gözlemin ağırlıkları",
        x_title="X",
        y_title="Normalize çekirdek ağırlığı",
    )
    show_figure(figure)
    effective = 1 / float(weights @ weights)
    columns = st.columns(3)
    columns[0].metric("Ağırlık toplamı", f"{weights.sum():.6f}")
    columns[1].metric("Etkin komşuluk", f"{effective:.1f}")
    columns[2].metric("En yüksek ağırlık", f"{weights.max():.4f}")
    st.info(
        "Bant genişliği yalnız çizginin pürüzsüzlüğünü değil, her x₀ için hangi gözlemlerin "
        "ne kadar bilgi taşıdığını belirleyen bir ayar kararıdır."
    )


def _render_local_regression(
    frame: pd.DataFrame, bandwidth: float, kernel: str
) -> None:
    grid = tuple(np.linspace(0, 10, 150))
    x_values = tuple(frame["x"])
    y_values = tuple(frame["y"])
    constant = _local_fit(x_values, y_values, grid, bandwidth, 0, kernel)
    linear = _local_fit(x_values, y_values, grid, bandwidth, 1, kernel)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["x"],
            y=frame["y"],
            mode="markers",
            name="Gözlem",
            marker={"size": 5, "opacity": 0.22, "color": "#51696C"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=grid,
            y=constant.predictions,
            mode="lines",
            name="Yerel sabit",
            line={"color": "#B3392F", "dash": "dash", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=grid,
            y=linear.predictions,
            mode="lines",
            name="Yerel doğrusal",
            line={"color": "#107C89", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["x"],
            y=frame["conditional_mean"],
            mode="lines",
            name="Gerçek E[Y|X]",
            line={"color": "#2F9E6B", "dash": "dot", "width": 2},
        )
    )
    style_figure(
        figure,
        title="Yerel sabit ve yerel doğrusal sınır davranışı",
        x_title="X",
        y_title="Sonuç Y",
    )
    show_figure(figure)
    left_truth = float(frame["conditional_mean"].iloc[0])
    metrics = st.columns(3)
    metrics[0].metric("Sol sınır gerçeği", f"{left_truth:.3f}")
    metrics[1].metric(
        "Yerel sabit sınır hatası",
        f"{abs(constant.predictions[0] - left_truth):.3f}",
    )
    metrics[2].metric(
        "Yerel doğrusal sınır hatası",
        f"{abs(linear.predictions[0] - left_truth):.3f}",
    )
    st.caption(
        "Yerel doğrusal tahmin sınırda eksik komşuluğun ilk dereceden etkisini düzeltebilir; "
        "bu, her örneklemde otomatik olarak daha düşük varyans anlamına gelmez."
    )


def _render_cv_series(
    frame: pd.DataFrame, bandwidth: float, kernel: str
) -> None:
    candidates = tuple(np.round(np.linspace(0.2, 1.8, 9), 2))
    result = _cv_fit(tuple(frame["x"]), tuple(frame["y"]), candidates, kernel)
    figure = go.Figure(
        go.Scatter(
            x=result.bandwidths,
            y=result.mean_squared_errors,
            mode="lines+markers",
            line={"color": "#107C89", "width": 3},
            name="5-katlı CV MSE",
        )
    )
    figure.add_vline(
        x=result.selected_bandwidth,
        line_color="#B3392F",
        line_dash="dash",
    )
    style_figure(
        figure,
        title="Bant genişliği için dış-kat tahmin hatası",
        x_title="Bant genişliği h",
        y_title="CV ortalama karesel hata",
    )
    show_figure(figure)
    knots = st.slider(
        "Spline düğüm sayısı", 3, 12, 7, key="konu08_knots"
    )
    grid = np.linspace(0, 10, 150)
    spline = spline_series_predict(frame["x"], frame["y"], grid, knots)
    local = _local_fit(
        tuple(frame["x"]),
        tuple(frame["y"]),
        tuple(grid),
        bandwidth,
        1,
        kernel,
    )
    comparison = go.Figure()
    comparison.add_trace(
        go.Scatter(
            x=grid,
            y=spline,
            mode="lines",
            name=f"Spline ({knots} düğüm)",
            line={"color": "#2F9E6B", "width": 3},
        )
    )
    comparison.add_trace(
        go.Scatter(
            x=grid,
            y=local.predictions,
            mode="lines",
            name=f"Yerel doğrusal (h={bandwidth:.2f})",
            line={"color": "#107C89", "width": 3},
        )
    )
    style_figure(
        comparison,
        title="İki esneklik kuralı",
        x_title="X",
        y_title="Tahmin edilen E[Y|X]",
    )
    show_figure(comparison)
    st.metric("Çapraz doğrulama bant genişliği", f"{result.selected_bandwidth:.2f}")

    rng = np.random.default_rng(SEED + 1)
    z = rng.uniform(-2.5, 2.5, 1200)
    nonlinear_control = 0.55 * z**2 + 0.35 * np.sin(1.4 * z)
    treatment = nonlinear_control + rng.normal(scale=0.65, size=z.size)
    outcome = (
        1.5 * treatment
        + 1.8 * nonlinear_control
        + rng.normal(scale=0.7, size=z.size)
    )
    partial = partialling_out(outcome, treatment, z, n_knots=knots)
    st.markdown("#### Kısmen doğrusal modele köprü")
    columns = st.columns(3)
    columns[0].metric("Gerçek hedef katsayı", "1.500")
    columns[1].metric(
        "Naif doğrusal kontrol", f"{partial.naive_linear_coefficient:.3f}"
    )
    columns[2].metric("Esnek artıklaştırma", f"{partial.coefficient:.3f}")
    st.info(
        "Burada sonuç ve hedef değişken aynı spline bazına göre örneklem içinde artıklaştırılır. "
        "Çapraz uyarlama ve DML çıkarımı Konu 12'nin ayrı hedefidir."
    )
    render_model_context(
        data_label="Kontrollü doğrusal olmayan DGP",
        sample_label=f"n = {len(frame):,}; çapraz doğrulama bölme birimi = {result.split_unit}",
        model_label="Yerel doğrusal / eğri bazlı seri / örneklem-içi artıklaştırma",
        inference_label="Çapraz doğrulama MSE'si; artıklaştırma HC1 nokta SH",
        seed=SEED,
    )


def _render_ddk_lab() -> None:
    uploaded = st.file_uploader(
        "Hazırlanmış DDK2011 CSV",
        type=("csv",),
        key="konu08_ddk_upload",
    )
    if uploaded is None:
        st.info(
            "DDK verisi lisans teyidi olmadan depoya eklenmez. Hazırlanmış CSV yalnız "
            "bu oturumda şema doğrulamasından sonra kullanılabilir."
        )
        return
    try:
        frame = load_registered_csv("ddk2011", uploaded.getvalue())
        numeric = ("tracking", "girl", "percentile", "totalscore", "schoolid")
        working = frame.copy()
        for column in numeric:
            working[column] = pd.to_numeric(working[column], errors="coerce")
        working = working.loc[
            working["tracking"].eq(1) & working["girl"].eq(1)
        ]
        working = working.dropna(subset=list(numeric))
        if len(working) < 80:
            raise ValueError(
                "Düzeyleme okullarındaki kız öğrenciler için en az 80 tam gözlem gerekir."
            )
    except (ValueError, pd.errors.ParserError) as error:
        st.error(f"DDK dosyası doğrulanamadı: {error}")
        return

    bandwidth = st.segmented_control(
        "DDK referans bant genişliği",
        options=(6.2, 12.3),
        default=6.2,
        format_func=lambda value: (
            "Okul-kümeli CV: h = 6.2"
            if value == 6.2
            else "Gözlem CV: h = 12.3"
        ),
        key="konu08_ddk_bandwidth",
    )
    grid = np.linspace(
        working["percentile"].quantile(0.02),
        working["percentile"].quantile(0.98),
        100,
    )
    local = local_polynomial_predict(
        working["percentile"],
        working["totalscore"],
        grid,
        float(bandwidth),
        degree=1,
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=working["percentile"],
            y=working["totalscore"],
            mode="markers",
            name="Öğrenci",
            marker={"size": 5, "opacity": 0.2, "color": "#51696C"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=grid,
            y=local.predictions,
            mode="lines",
            name=f"Yerel doğrusal h={bandwidth:.1f}",
            line={"color": "#107C89", "width": 3},
        )
    )
    style_figure(
        figure,
        title="Düzeyleme okullarındaki kız öğrenciler",
        x_title="Başlangıç başarı yüzdeliği",
        y_title="Toplam test puanı",
    )
    show_figure(figure)
    st.caption(
        f"n = {len(working):,}; okul = {working['schoolid'].nunique():,}. "
        "Bant genişliği karşılaştırması betimseldir; rastgele atama tek başına "
        "başlangıç başarısı-puan eğrisini nedensel yapmaz."
    )


def render() -> None:
    render_topic_header(TOPIC_KEY)
    controls = st.columns(3)
    nobs = controls[0].slider(
        "Gözlem sayısı", 200, 1000, 500, 100, key="konu08_nobs"
    )
    bandwidth = controls[1].slider(
        "Bant genişliği h",
        0.20,
        2.00,
        0.70,
        0.10,
        key="konu08_bandwidth",
    )
    kernel = controls[2].selectbox(
        "Çekirdek",
        ("Gaussian", "Epanechnikov", "Üçgensel"),
        format_func=lambda value: "Gauss" if value == "Gaussian" else value,
        key="konu08_kernel",
    )
    noise = st.slider(
        "Gürültü ölçeği",
        0.15,
        0.90,
        0.45,
        0.05,
        key="konu08_noise",
    )
    frame = _simulation(nobs, noise)

    weights_tab, local_tab, cv_tab, ddk_tab = st.tabs(
        (
            "Çekirdek ağırlıkları",
            "Yerel regresyon",
            "Çapraz doğrulama, seri ve artıklaştırma",
            "DDK laboratuvarı",
        )
    )
    with weights_tab:
        _render_kernel_weights(frame, bandwidth, kernel)
        render_reproduction_code(TOPIC_KEY, "agirlik")
    with local_tab:
        _render_local_regression(frame, bandwidth, kernel)
        render_reproduction_code(TOPIC_KEY, "yerel")
    with cv_tab:
        _render_cv_series(frame, bandwidth, kernel)
        render_reproduction_code(TOPIC_KEY, "dogrulama")
    with ddk_tab:
        _render_ddk_lab()
        render_reproduction_code(TOPIC_KEY, "veri")
    render_question(TOPIC_KEY)
