"""Konu 06: sansürleme, Tobit estimandları ve örneklem seçimi."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

from core.datasets import load_registered_csv
from core.limited_outcomes import (
    HeckmanDGPConfig,
    TobitDGPConfig,
    TobitFit,
    fit_tobit,
    heckman_two_step,
    simulate_selection_data,
    simulate_tobit_data,
    tobit_expectations,
    tobit_marginal_effects,
)
from core.ols import fit_ols
from topics.regression_ui import render_model_context, show_figure, style_figure
from topics.shared import render_question, render_reproduction_code, render_topic_header


TOPIC_KEY = "konu06"
TOBIT_SEED = 806
SELECTION_SEED = 866


@st.cache_data(show_spinner=False)
def _tobit_data(nobs: int, intercept: float, sigma: float) -> pd.DataFrame:
    return simulate_tobit_data(
        TobitDGPConfig(
            nobs=nobs,
            seed=TOBIT_SEED,
            intercept=intercept,
            sigma=sigma,
        )
    )


@st.cache_data(show_spinner=False)
def _selection_data(rho: float, exclusion_strength: float) -> pd.DataFrame:
    return simulate_selection_data(
        HeckmanDGPConfig(
            nobs=5000,
            seed=SELECTION_SEED,
            error_correlation=rho,
            exclusion_strength=exclusion_strength,
        )
    )


def _render_mechanism(frame: pd.DataFrame) -> None:
    problem = st.segmented_control(
        "Veri problemi",
        options=("Sansürleme", "Kesilme", "Örneklem seçimi"),
        default="Sansürleme",
        key="konu06_problem_type",
    )
    descriptions = {
        "Sansürleme": "Bütün birimler görünür; eşik altındaki gizli sonuçlar sınır değerinde kaydedilir.",
        "Kesilme": "Eşik dışındaki birimler veri setine hiç girmez; gözlem birimi de kaybolur.",
        "Örneklem seçimi": "Birim ve seçim göstergesi görünür; sonuç yalnız seçilen birimlerde gözlenir.",
    }
    st.info(descriptions[problem])
    censor_rate = float(frame["observed"].eq(0).mean())
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=frame["latent"],
            nbinsx=45,
            name="Gizli sonuç Y*",
            opacity=0.55,
            marker_color="#107C89",
        )
    )
    figure.add_trace(
        go.Histogram(
            x=frame["observed"],
            nbinsx=45,
            name="Gözlenen sonuç Y=max(0,Y*)",
            opacity=0.55,
            marker_color="#B3392F",
        )
    )
    figure.update_layout(barmode="overlay")
    figure.add_vline(x=0, line_dash="dash", line_color="#07373D")
    style_figure(
        figure,
        title="Gizli sonuçtan gözlenen sansürlü sonuca",
        x_title="Sonuç değeri",
        y_title="Gözlem sayısı",
    )
    show_figure(figure)
    metrics = st.columns(3)
    metrics[0].metric("Sansürlenme oranı", f"%{100 * censor_rate:.1f}")
    metrics[1].metric("Gözlenen sıfır", f"{frame['observed'].eq(0).sum():,}")
    metrics[2].metric("Gizli Y* ortalaması", f"{frame['latent'].mean():.3f}")
    st.warning(
        "Çok sayıda sıfır tek başına Tobit gerekçesi değildir. Sıfırların gerçek ekonomik "
        "sıfır mı, ölçüm sınırı mı veya seçim kodu mu olduğu veri üretiminden belirlenir."
    )


def _render_estimands(fit: TobitFit) -> None:
    evaluation = st.slider(
        "Değerlendirme x değeri",
        min_value=-2.0,
        max_value=2.0,
        value=0.5,
        step=0.25,
        key="konu06_evaluation_x",
    )
    grid = np.linspace(-2.5, 2.5, 180)
    design = np.column_stack((np.ones(grid.size), grid))
    expectations = tobit_expectations(fit, design)
    figure = go.Figure()
    styles = (
        ("Gizli ortalama", "#107C89", "solid"),
        ("Gözlenen ortalama", "#B3392F", "dash"),
        ("Pozitif koşullu ortalama", "#2F9E6B", "dot"),
    )
    for column, color, dash in styles:
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=expectations[column],
                mode="lines",
                name=column,
                line={"color": color, "dash": dash, "width": 3},
            )
        )
    style_figure(
        figure,
        title="Tobit'te üç farklı koşullu ortalama",
        x_title="Açıklayıcı değişken x",
        y_title="Tahmin edilen sonuç",
    )
    show_figure(figure)
    point_design = np.array([[1.0, evaluation]])
    point_expectations = tobit_expectations(fit, point_design).iloc[0]
    effects = tobit_marginal_effects(fit, "x", point_design).iloc[0]
    metrics = st.columns(4)
    metrics[0].metric("Tobit x katsayısı", f"{fit.coefficient('x'):.3f}")
    metrics[1].metric(
        "Gözlenen ortalama etkisi", f"{effects['Gözlenen ortalama etkisi']:.3f}"
    )
    metrics[2].metric(
        "Sansürlenmeme olasılığı", f"%{100 * point_expectations['Sansürlenmeme olasılığı']:.1f}"
    )
    metrics[3].metric(
        "Olasılık marjinal etkisi", f"{effects['Sansürlenmeme olasılığı etkisi']:.3f}"
    )
    status = st.success if fit.converged else st.error
    status(
        f"Tobit MLE yakınsama: {'başarılı' if fit.converged else 'başarısız'} | "
        f"İterasyon: {fit.iterations} | σ: {fit.sigma:.3f} | "
        f"Maksimum skor normu: {fit.gradient_norm:.2e}"
    )
    st.caption(
        "Tobit katsayısı gizli sonuç denklemine aittir. Gözlenen ortalama ve "
        "sansürlenmeme olasılığı için ayrı marjinal etkiler gerekir."
    )


def _load_lab_frame(
    default_frame: pd.DataFrame,
) -> tuple[pd.DataFrame | None, str, str, str]:
    source = st.segmented_control(
        "Veri kaynağı",
        options=("Kontrollü Tobit DGP", "Hazırlanmış CHJ CSV"),
        default="Kontrollü Tobit DGP",
        key="konu06_data_source",
    )
    if source == "Kontrollü Tobit DGP":
        return default_frame, "Kontrollü Tobit DGP", "observed", "x"
    uploaded = st.file_uploader(
        "Hazırlanmış CHJ CSV",
        type=("csv",),
        key="konu06_chj_upload",
    )
    if uploaded is None:
        st.info(
            "CHJ verisi lisans teyidi olmadan depoya eklenmez. Hazırlanmış CSV yalnız "
            "bu oturumda şema doğrulamasından sonra kullanılabilir."
        )
        return None, "CHJ2004", "received", "income_adj"
    try:
        frame = load_registered_csv("chj2004", uploaded.getvalue())
    except (ValueError, pd.errors.ParserError) as error:
        st.error(f"CHJ dosyası doğrulanamadı: {error}")
        return None, "CHJ2004", "received", "income_adj"
    working = frame.copy()
    working["received_k"] = pd.to_numeric(working["received"], errors="coerce") / 1000
    working["income_k"] = pd.to_numeric(working["income_adj"], errors="coerce") / 1000
    return working, "CHJ2004 kullanıcı dosyası", "received_k", "income_k"


def _render_data_lab(default_frame: pd.DataFrame) -> None:
    frame, label, outcome, regressor = _load_lab_frame(default_frame)
    if frame is None:
        return
    working = frame.dropna(subset=[outcome, regressor]).copy()
    ols = fit_ols(working[outcome], working[[regressor]], (regressor,))
    tobit = fit_tobit(
        working[outcome],
        working[[regressor]],
        (regressor,),
    )
    quantile = sm.QuantReg(
        working[outcome], sm.add_constant(working[[regressor]])
    ).fit(q=0.5, max_iter=5000)
    table = pd.DataFrame(
        (
            {
                "Yöntem": "OLS",
                "Hedef": "Gözlenen doğrusal ortalama",
                "Eğim": ols.coefficient(regressor),
                "Yakınsama": "Kapalı biçim",
            },
            {
                "Yöntem": "Tobit",
                "Hedef": "Gizli normal sonuç ortalaması",
                "Eğim": tobit.coefficient(regressor),
                "Yakınsama": "Başarılı" if tobit.converged else "Başarısız",
            },
            {
                "Yöntem": "LAD",
                "Hedef": "Gözlenen koşullu medyan",
                "Eğim": float(quantile.params[regressor]),
                "Yakınsama": "Başarılı",
            },
        )
    )
    st.dataframe(
        table.style.format({"Eğim": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )
    grid = np.linspace(
        float(working[regressor].quantile(0.02)),
        float(working[regressor].quantile(0.98)),
        160,
    )
    design = np.column_stack((np.ones(grid.size), grid))
    tobit_means = tobit_expectations(tobit, design)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=grid,
            y=design @ ols.coefficients,
            mode="lines",
            name="OLS gözlenen ortalama",
            line={"color": "#51696C", "dash": "dash"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=grid,
            y=tobit_means["Gizli ortalama"],
            mode="lines",
            name="Tobit gizli ortalama",
            line={"color": "#107C89"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=grid,
            y=tobit_means["Gözlenen ortalama"],
            mode="lines",
            name="Tobit gözlenen ortalama",
            line={"color": "#B3392F"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=grid,
            y=quantile.params["const"] + quantile.params[regressor] * grid,
            mode="lines",
            name="LAD gözlenen medyan",
            line={"color": "#2F9E6B", "dash": "dot"},
        )
    )
    unit = "bin peso" if label.startswith("CHJ") else "DGP birimi"
    style_figure(
        figure,
        title=f"{label}: farklı tahmin hedeflerinin profilleri",
        x_title=f"Açıklayıcı değişken ({unit})",
        y_title=f"Tahmin edilen sonuç ({unit})",
    )
    show_figure(figure)
    st.warning(
        "Yöntem eğimlerini yalnız büyüklük veya anlamlılık yarışına sokmayın. OLS, "
        "Tobit ve LAD aynı koşullu dağılım nesnesini hedeflemeyebilir."
    )
    render_model_context(
        data_label=label,
        sample_label=f"n = {len(working):,}; sansür noktası 0",
        model_label=f"{outcome} ~ {regressor}",
        inference_label="Bu görünümde hedef büyüklük ve nokta tahmini karşılaştırması",
        seed=TOBIT_SEED if label.startswith("Kontrollü") else None,
    )


def _render_selection() -> None:
    controls = st.columns(2)
    rho = controls[0].slider(
        "Sonuç-seçim hata korelasyonu ρ",
        min_value=-0.8,
        max_value=0.8,
        value=0.6,
        step=0.1,
        key="konu06_selection_rho",
    )
    exclusion = controls[1].slider(
        "Dışlama değişkeni gücü",
        min_value=0.0,
        max_value=1.5,
        value=0.9,
        step=0.1,
        key="konu06_exclusion_strength",
    )
    frame = _selection_data(rho, exclusion)
    result = heckman_two_step(frame)
    metrics = st.columns(4)
    metrics[0].metric("Gerçek sonuç eğimi", "2,000")
    metrics[1].metric("Seçilmiş örneklem OLS", f"{result.naive_slope:.3f}")
    metrics[2].metric("Heckman iki aşama", f"{result.corrected_slope:.3f}")
    metrics[3].metric("Ters Mills katsayısı", f"{result.mills_coefficient:.3f}")

    grouped = frame.assign(x_bin=pd.qcut(frame["x"], 10, duplicates="drop")).groupby(
        "x_bin", observed=True
    ).agg(
        ortalama_x=("x", "mean"),
        secilme_orani=("selected", "mean"),
    )
    figure = go.Figure(
        go.Scatter(
            x=grouped["ortalama_x"],
            y=grouped["secilme_orani"],
            mode="lines+markers",
            name="Gözlenen seçilme oranı",
            marker={"size": 9, "color": "#107C89"},
        )
    )
    style_figure(
        figure,
        title="Sonuç regresörü boyunca örnekleme seçilme olasılığı",
        x_title="Sonuç denklemindeki x",
        y_title="Seçilme oranı",
    )
    show_figure(figure)
    if exclusion == 0:
        st.error(
            "Seçim denkleminde dışlanan değişken yok. Model yalnız Probit'in doğrusal "
            "olmayan biçimine dayanıyor; pratik tanımlama kırılgandır."
        )
    else:
        st.success(
            "Dışlama değişkeni seçimi etkiliyor ve sonuç denkleminden dışlanıyor. Bu "
            "geçerlilik yine ekonomik ve kurumsal olarak savunulmalıdır."
        )
    st.warning(
        "Bu öğretim sürümü Heckman iki aşama nokta tahminini gösterir. Ters Mills oranı "
        "üretilmiş regresör olduğu için sıradan ikinci-aşama OLS standart hatası raporlanmaz."
    )
    render_model_context(
        data_label="Kontrollü endojen seçim DGP'si",
        sample_label=(
            f"n = {len(frame):,}; seçilen = {result.selected_observations:,} "
            f"(%{100 * result.selection_rate:.1f})"
        ),
        model_label="Seçim Probit'i + seçilmiş örneklemde ters Mills düzeltmesi",
        inference_label="İki aşamalı nokta tahmini; özel SH gerekli",
        seed=SELECTION_SEED,
    )


def render() -> None:
    render_topic_header(TOPIC_KEY)
    controls = st.columns(3)
    nobs = controls[0].slider(
        "Gözlem sayısı",
        min_value=600,
        max_value=3000,
        value=1600,
        step=200,
        key="konu06_nobs",
    )
    intercept = controls[1].slider(
        "Gizli sonuç sabiti",
        min_value=-1.5,
        max_value=0.5,
        value=-0.5,
        step=0.25,
        key="konu06_intercept",
    )
    sigma = controls[2].slider(
        "Gizli hata σ",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.25,
        key="konu06_sigma",
    )
    frame = _tobit_data(nobs, intercept, sigma)
    fit = fit_tobit(frame["observed"], frame[["x"]], ("x",))

    mechanism, estimands, data_lab, selection = st.tabs(
        ("Veri mekanizması", "Tobit tahmin hedefleri", "Veri laboratuvarı", "Örneklem seçimi")
    )
    with mechanism:
        _render_mechanism(frame)
        render_reproduction_code(TOPIC_KEY, "mekanizma")
    with estimands:
        _render_estimands(fit)
        render_reproduction_code(TOPIC_KEY, "hedefler")
    with data_lab:
        _render_data_lab(frame)
        render_reproduction_code(TOPIC_KEY, "veri")
    with selection:
        _render_selection()
        render_reproduction_code(TOPIC_KEY, "secim")
    render_question(TOPIC_KEY)
