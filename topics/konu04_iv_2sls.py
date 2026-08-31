"""Konu 04: araç geçerliliği, Wald oranı, 2SLS ve zayıf araçlar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.datasets import load_registered_csv
from core.inference import coefficient_inference, hc1_covariance
from core.iv import (
    IVConfig,
    conditional_wald_ratio,
    fit_2sls,
    naive_second_stage_fit,
    simulate_iv_data,
    weak_instrument_monte_carlo,
)
from core.ols import fit_ols
from topics.regression_ui import render_model_context, show_figure, style_figure
from topics.shared import render_question, render_reproduction_code, render_topic_header


TOPIC_KEY = "konu04"
SEED = 830


@st.cache_data(show_spinner=False)
def _simulation(
    nobs: int,
    instrument_strength: float,
    exclusion_violation: float,
) -> pd.DataFrame:
    return simulate_iv_data(
        IVConfig(
            nobs=nobs,
            seed=SEED,
            instrument_strength=instrument_strength,
            exclusion_violation=exclusion_violation,
        )
    )


@st.cache_data(show_spinner=False)
def _weak_instrument_draws() -> pd.DataFrame:
    return weak_instrument_monte_carlo(
        (0.05, 0.10, 0.20, 0.40, 0.80),
        nobs=400,
        repetitions=80,
        seed=844,
    )


def _render_validity(
    instrument_strength: float,
    exclusion_violation: float,
) -> None:
    independent = st.toggle(
        "Araç yapısal hatadan bağımsız",
        value=True,
        key="konu04_independence",
    )
    conditions = (
        (
            "Uygunluk",
            instrument_strength >= 0.10,
            f"İlk aşama DGP katsayısı π = {instrument_strength:.2f}.",
        ),
        (
            "Dışsallık / ortogonallik",
            independent,
            "Z ile yapısal hata arasındaki ilişki kurumsal tasarıma dayanır.",
        ),
        (
            "Dışlama kısıtı",
            exclusion_violation == 0,
            f"Z'nin Y'ye doğrudan DGP yolu = {exclusion_violation:.2f}.",
        ),
    )
    columns = st.columns(3)
    for column, (title, valid, description) in zip(columns, conditions, strict=True):
        symbol = "Geçiyor" if valid else "İhlal / zayıf"
        column.markdown(f"#### {title}")
        column.metric("Durum", symbol)
        column.caption(description)
    if all(condition[1] for condition in conditions):
        st.success(
            "DGP'de üç koşul birlikte sağlanıyor. Sabit etki modelinde IV katsayısı "
            "yapısal eğitim etkisini hedefleyebilir."
        )
    else:
        st.error(
            "Araç için gerekli koşullar birlikte sağlanmıyor. Güçlü bir ilk aşama "
            "dışsallık veya dışlama ihlalini telafi etmez."
        )
    st.caption(
        "Bu ekran koşulları görünür kılar; gerçek araştırmada dışsallık ve dışlama "
        "veri tablosundan otomatik doğrulanamaz."
    )


def _render_wald(frame: pd.DataFrame) -> None:
    first_stage, reduced_form, wald = conditional_wald_ratio(
        frame["lwage"],
        frame["education"],
        frame["instrument"],
        exogenous=frame[["control"]],
    )
    metrics = st.columns(3)
    metrics[0].metric("İlk aşama", f"{first_stage:.3f} yıl")
    metrics[1].metric("İndirgenmiş biçim", f"{reduced_form:.3f} log birim")
    metrics[2].metric("Wald oranı", f"{wald:.3f}")

    standardized = frame[["education", "lwage"]].apply(
        lambda series: (series - series.mean()) / series.std()
    )
    plot_frame = pd.DataFrame(
        {
            "instrument": frame["instrument"],
            "Standart eğitim": standardized["education"],
            "Standart log ücret": standardized["lwage"],
        }
    )
    means = plot_frame.groupby("instrument", as_index=False).mean()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=means["instrument"],
            y=means["Standart eğitim"],
            mode="lines+markers",
            name="Eğitim",
            marker={"symbol": "circle", "size": 10},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=means["instrument"],
            y=means["Standart log ücret"],
            mode="lines+markers",
            name="Log ücret",
            marker={"symbol": "square", "size": 10},
        )
    )
    style_figure(
        figure,
        title="Araç gruplarında eğitim ve ücret ortalamaları",
        x_title="İkili araç Z (0: uzak, 1: yakın)",
        y_title="Grup ortalaması (standart sapma birimi)",
    )
    figure.update_xaxes(tickmode="array", tickvals=(0, 1))
    show_figure(figure)
    st.info(
        "Tam tanımlı tek araç-tek endojen değişken modelinde koşullu Wald oranı, "
        "2SLS eğitim katsayısıyla aynıdır. Payda sıfıra yaklaştığında oran kararsızlaşır."
    )


def _card_frame(uploaded) -> tuple[pd.DataFrame | None, str]:
    if uploaded is None:
        st.info(
            "Card verisi lisans teyidi olmadan depoya eklenmez. Hazırlanmış CSV yalnız "
            "bu oturumda şema doğrulamasından sonra kullanılabilir."
        )
        return None, "Card1995"
    try:
        frame = load_registered_csv("card1995", uploaded.getvalue())
    except (ValueError, pd.errors.ParserError) as error:
        st.error(f"Card dosyası doğrulanamadı: {error}")
        return None, "Card1995"
    return frame, "Card1995 kullanıcı dosyası"


def _model_inputs(
    default_frame: pd.DataFrame,
) -> tuple[pd.DataFrame | None, str, str, str, list[str]]:
    source = st.segmented_control(
        "Veri kaynağı",
        options=("Kontrollü IV DGP", "Hazırlanmış Card CSV"),
        default="Kontrollü IV DGP",
        key="konu04_data_source",
    )
    if source == "Kontrollü IV DGP":
        return default_frame, "Kontrollü IV DGP", "lwage", "education", ["control"]
    uploaded = st.file_uploader(
        "Hazırlanmış Card CSV",
        type=("csv",),
        key="konu04_card_upload",
    )
    frame, label = _card_frame(uploaded)
    return frame, label, "lwage76", "ed76", [
        "exp76",
        "exp762_100",
        "black",
        "smsa76r",
        "reg76r",
        "smsa66r",
        "reg662",
        "reg663",
        "reg664",
        "reg665",
        "reg666",
        "reg667",
        "reg668",
        "reg669",
    ]


def _render_2sls(default_frame: pd.DataFrame, exclusion_violation: float) -> None:
    frame, data_label, outcome, endogenous, controls = _model_inputs(default_frame)
    if frame is None:
        return
    instrument = "instrument" if data_label.startswith("Kontrollü") else "nearc4"
    working = frame.dropna(subset=[outcome, endogenous, instrument, *controls]).copy()
    iv_fit = fit_2sls(
        working[outcome],
        working[endogenous],
        working[instrument],
        exogenous=working[controls],
        endogenous_name="Eğitim",
        exogenous_names=tuple(controls),
    )
    ols = fit_ols(
        working[outcome],
        working[[endogenous, *controls]],
        ("Eğitim", *controls),
    )
    ols_inference = coefficient_inference(ols, "Eğitim", "HC1")
    naive = naive_second_stage_fit(
        working[outcome],
        iv_fit,
        exogenous=working[controls],
        exogenous_names=tuple(controls),
    )
    naive_covariance = hc1_covariance(naive)
    naive_se = float(np.sqrt(naive_covariance[-1, -1]))

    table = pd.DataFrame(
        (
            {
                "Yöntem": "OLS",
                "Eğitim katsayısı": ols_inference.estimate,
                "Uygun standart hata": ols_inference.standard_error,
                "Kovaryans": "HC1 OLS",
            },
            {
                "Yöntem": "2SLS",
                "Eğitim katsayısı": iv_fit.coefficient("Eğitim"),
                "Uygun standart hata": iv_fit.standard_error("Eğitim"),
                "Kovaryans": "Robust IV, debiased",
            },
        )
    )
    st.dataframe(
        table.style.format(
            {
                "Eğitim katsayısı": "{:.4f}",
                "Uygun standart hata": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    metrics = st.columns(3)
    metrics[0].metric("İlk aşama robust F", f"{iv_fit.first_stage_f:.2f}")
    metrics[1].metric("2SLS robust standart hata", f"{iv_fit.standard_error('Eğitim'):.4f}")
    metrics[2].metric("Kullanılmayan ikinci-aşama SH", f"{naive_se:.4f}")

    sampled = working.iloc[: min(1200, len(working))]
    figure = go.Figure(
        go.Scattergl(
            x=sampled[endogenous],
            y=iv_fit.fitted_endogenous[: len(sampled)],
            mode="markers",
            name="Gözlemler",
            marker={"size": 6, "opacity": 0.45, "color": "#107C89"},
            hovertemplate=(
                "Gözlenen eğitim: %{x:.2f}<br>Araçla tahmin edilen eğitim: "
                "%{y:.2f}<extra></extra>"
            ),
        )
    )
    minimum = float(min(sampled[endogenous].min(), iv_fit.fitted_endogenous.min()))
    maximum = float(max(sampled[endogenous].max(), iv_fit.fitted_endogenous.max()))
    figure.add_trace(
        go.Scatter(
            x=(minimum, maximum),
            y=(minimum, maximum),
            mode="lines",
            name="45° referans",
            line={"dash": "dash", "color": "#51696C"},
        )
    )
    style_figure(
        figure,
        title="İlk aşama: gözlenen ve araçla açıklanan eğitim",
        x_title="Gözlenen eğitim (yıl)",
        y_title="İlk aşama uydurulan eğitim (yıl)",
    )
    show_figure(figure)
    st.error(
        "İkinci aşamayı sıradan OLS gibi çalıştırmak katsayıyı yeniden üretebilir; "
        "fakat o regresyonun standart hatası 2SLS standart hatası değildir."
    )
    if exclusion_violation > 0 and data_label.startswith("Kontrollü"):
        st.warning(
            "DGP'de aracın ücrete doğrudan yolu açık. 2SLS hesaplanabilir, ancak "
            "yapısal eğitim etkisi dışlama kısıtı ihlal edildiği için tanımlanmaz."
        )
    render_model_context(
        data_label=data_label,
        sample_label=f"n = {len(working):,}; eksiksiz model örneklemi",
        model_label=f"{outcome} ~ [{endogenous} ~ {instrument}] + dışsal kontroller",
        inference_label="Robust 2SLS, HC1 ölçek düzeltmesi",
        seed=SEED if data_label.startswith("Kontrollü") else None,
    )


def _render_weak_instruments() -> None:
    draws = _weak_instrument_draws()
    figure = go.Figure()
    for strength in sorted(draws["Araç gücü"].unique()):
        values = draws.loc[draws["Araç gücü"].eq(strength), "2SLS"]
        figure.add_trace(
            go.Box(
                y=values,
                name=f"π={strength:.2f}",
                boxpoints=False,
                marker_color="#107C89" if strength >= 0.2 else "#B3392F",
            )
        )
    figure.add_hline(
        y=0.12,
        line_dash="dash",
        line_color="#2F9E6B",
        annotation_text="Yapısal etki = 0,12",
    )
    style_figure(
        figure,
        title="Araç zayıfladıkça 2SLS örnekleme dağılımı",
        x_title="DGP ilk aşama katsayısı",
        y_title="2SLS eğitim katsayısı",
    )
    show_figure(figure)
    summary = draws.groupby("Araç gücü").agg(
        Medyan_F=("İlk aşama F", "median"),
        Tahmin_SD=("2SLS", "std"),
    )
    st.dataframe(
        summary.style.format({"Medyan_F": "{:.2f}", "Tahmin_SD": "{:.3f}"}),
        width="stretch",
    )
    st.warning(
        "F > 10 evrensel bir geçerlilik sınırı değildir. Zayıf araç problemi yalnız "
        "geniş standart hata değil, yanlı ve normal olmayan sonlu örneklem dağılımı yaratabilir."
    )
    render_model_context(
        data_label="Geçerli araç kontrollü Monte Carlo DGP'si",
        sample_label="Her π için n = 400, 80 tekrar",
        model_label="Tek araç, tek endojen eğitim değişkeni",
        inference_label="Robust 2SLS dağılım tanısı",
        seed=844,
    )


def render() -> None:
    render_topic_header(TOPIC_KEY)
    controls = st.columns(3)
    nobs = controls[0].slider(
        "Gözlem sayısı",
        min_value=400,
        max_value=2400,
        value=1200,
        step=200,
        key="konu04_nobs",
    )
    strength = controls[1].slider(
        "Araç gücü π",
        min_value=0.05,
        max_value=1.00,
        value=0.35,
        step=0.05,
        key="konu04_strength",
    )
    violation = controls[2].slider(
        "Doğrudan Z → Y yolu",
        min_value=0.00,
        max_value=0.20,
        value=0.00,
        step=0.02,
        key="konu04_exclusion_violation",
    )
    frame = _simulation(nobs, strength, violation)

    validity, wald, stages, weak = st.tabs(
        ("Geçerlilik", "Wald oranı", "2SLS zinciri", "Zayıf araç")
    )
    with validity:
        _render_validity(strength, violation)
        render_reproduction_code(TOPIC_KEY, "gecerlilik")
    with wald:
        _render_wald(frame)
        render_reproduction_code(TOPIC_KEY, "wald")
    with stages:
        _render_2sls(frame, violation)
        render_reproduction_code(TOPIC_KEY, "iki_asama")
    with weak:
        _render_weak_instruments()
        render_reproduction_code(TOPIC_KEY, "zayif")
    render_question(TOPIC_KEY)
