"""Konu 03: tanımlama, seçim, içsellik ve rastgele atama."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.causal import (
    EndogeneityConfig,
    TrialConfig,
    decompose_observed_difference,
    endogeneity_estimates,
    endogeneity_monte_carlo,
    endogeneity_probability_limit,
    simulate_cluster_trial,
    simulate_endogeneity_data,
    simulate_selection_data,
)
from core.datasets import load_registered_csv
from core.inference import coefficient_inference
from core.ols import fit_ols
from topics.regression_ui import render_model_context, show_figure, style_figure
from topics.shared import render_question, render_reproduction_code, render_topic_header


TOPIC_KEY = "konu03"
ENDOGENEITY_SEED = 803
TRIAL_SEED = 833


@st.cache_data(show_spinner=False)
def _endogeneity_mc(confounding: float) -> pd.DataFrame:
    return endogeneity_monte_carlo(
        (200, 1000, 5000),
        repetitions=60,
        seed=ENDOGENEITY_SEED,
        confounding_strength=confounding,
    )


@st.cache_data(show_spinner=False)
def _trial_data(students: int) -> pd.DataFrame:
    return simulate_cluster_trial(
        TrialConfig(students=students, schools=60, seed=TRIAL_SEED)
    )


def _render_concept_chain() -> None:
    columns = st.columns(4)
    items = (
        ("1. Tahmin hedefi", "Anakütlede öğrenilmek istenen ortalama tedavi etkisi."),
        ("2. Tanımlama", "Gözlenen dağılımı ATE'ye bağlayan varsayım."),
        ("3. Tahmin edici", "Örneklem farkı veya regresyon kuralı."),
        ("4. Tahmin", "Bu örneklemde elde edilen sayısal değer."),
    )
    for column, (title, description) in zip(columns, items, strict=True):
        column.markdown(f"#### {title}")
        column.caption(description)

    assignment = st.segmented_control(
        "Atama mekanizması",
        options=("Rastgele atama", "Seçime dayalı atama"),
        default="Seçime dayalı atama",
        key="konu03_assignment",
    )
    randomized = assignment == "Rastgele atama"
    frame = simulate_selection_data(
        randomized=randomized,
        selection_strength=1.3,
    )
    result = decompose_observed_difference(frame)
    metrics = st.columns(3)
    metrics[0].metric("Gözlenen grup farkı", f"{result.observed_difference:.3f}")
    metrics[1].metric("Tedavi edilenlerde etki", f"{result.treatment_on_treated:.3f}")
    metrics[2].metric("Seçim bileşeni", f"{result.selection_bias:+.3f}")

    figure = go.Figure(
        go.Waterfall(
            x=("Tedavi edilenlerde etki", "Seçim farkı", "Gözlenen fark"),
            y=(
                result.treatment_on_treated,
                result.selection_bias,
                result.observed_difference,
            ),
            measure=("relative", "relative", "total"),
            connector={"line": {"color": "#51696C"}},
            increasing={"marker": {"color": "#2F9E6B"}},
            decreasing={"marker": {"color": "#B3392F"}},
            totals={"marker": {"color": "#107C89"}},
        )
    )
    style_figure(
        figure,
        title="Gözlenen farkın tedavi ve seçim bileşenleri",
        x_title="Ayrıştırma bileşeni",
        y_title="Sonuç farkı (puan)",
    )
    show_figure(figure)
    if randomized:
        st.success(
            "Rastgele atama seçim bileşenini beklentide sıfırlar; basit grup farkı "
            "ATE için geçerli bir tahmin edici olur."
        )
    else:
        st.warning(
            "Tedavi kararı başlangıç potansiyeliyle ilişkili. Gözlenen fark, tedavi "
            "etkisi ile tedavi olmasaydı da var olacak seçim farkını birlikte taşır."
        )


def _render_endogeneity() -> None:
    controls = st.columns(2)
    nobs = controls[0].select_slider(
        "Örneklem büyüklüğü",
        options=(200, 1000, 5000, 20000),
        value=1000,
        key="konu03_endogeneity_n",
    )
    confounding = controls[1].slider(
        "İçsellik şiddeti",
        min_value=0.0,
        max_value=2.0,
        value=0.8,
        step=0.2,
        key="konu03_endogeneity_strength",
    )
    config = EndogeneityConfig(
        nobs=nobs,
        seed=ENDOGENEITY_SEED,
        confounding_strength=confounding,
    )
    omitted, controlled = endogeneity_estimates(config)
    frame = simulate_endogeneity_data(config)
    omitted_fit = fit_ols(frame["y"], frame[["x"]], ("x",))
    omitted_inference = coefficient_inference(omitted_fit, "x", "HC1")
    probability_limit = endogeneity_probability_limit(config)

    metrics = st.columns(4)
    metrics[0].metric("Yapısal β", f"{config.structural_effect:.3f}")
    metrics[1].metric("U dışarıda OLS", f"{omitted:.3f}")
    metrics[2].metric("U kontrol OLS", f"{controlled:.3f}")
    metrics[3].metric("İçsel olasılık limiti", f"{probability_limit:.3f}")
    st.caption(
        f"U dışarıda HC1 standart hata: {omitted_inference.standard_error:.4f}. "
        "Standart hatanın küçülmesi yapısal hedefe yakınsama anlamına gelmez."
    )

    draws = _endogeneity_mc(confounding)
    summary = (
        draws.groupby(["n", "Model"])["Tahmin"]
        .agg(
            Ortalama="mean",
            Alt=lambda values: values.quantile(0.10),
            Üst=lambda values: values.quantile(0.90),
        )
        .reset_index()
    )
    figure = go.Figure()
    for model, color, symbol in (
        ("U dışarıda", "#B3392F", "circle"),
        ("U kontrol", "#107C89", "square"),
    ):
        group = summary.loc[summary["Model"].eq(model)]
        figure.add_trace(
            go.Scatter(
                x=group["n"],
                y=group["Ortalama"],
                mode="lines+markers",
                name=model,
                line={"color": color},
                marker={"symbol": symbol, "size": 9},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": group["Üst"] - group["Ortalama"],
                    "arrayminus": group["Ortalama"] - group["Alt"],
                },
            )
        )
    figure.add_hline(
        y=config.structural_effect,
        line_dash="dash",
        line_color="#2F9E6B",
        annotation_text="Yapısal β",
    )
    figure.add_hline(
        y=probability_limit,
        line_dash="dot",
        line_color="#51696C",
        annotation_text="İçsel olasılık limiti",
    )
    figure.update_xaxes(type="log")
    style_figure(
        figure,
        title="Büyük örneklem belirsizliği azaltır, içsellik hedefini değiştirmez",
        x_title="Örneklem büyüklüğü (log ölçek)",
        y_title="OLS x katsayısı",
    )
    show_figure(figure)
    render_model_context(
        data_label="Eksik değişken kontrollü DGP",
        sample_label=f"n = {nobs:,}; Monte Carlo 60 tekrar",
        model_label="Y = 2X + U + hata",
        inference_label="HC1; U gözlenirse karşılaştırmalı kontrol",
        seed=ENDOGENEITY_SEED,
    )


def _trial_results(frame: pd.DataFrame) -> pd.DataFrame:
    required = ("totalscore", "tracking", "schoolid")
    working = frame.dropna(subset=list(required)).copy()
    raw = fit_ols(working["totalscore"], working[["tracking"]], ("tracking",))
    raw_inference = coefficient_inference(
        raw,
        "tracking",
        "Küme",
        groups=working["schoolid"],
    )
    controls = [column for column in ("std_mark", "girl", "agetest") if column in working]
    adjusted_data = working.dropna(subset=controls)
    adjusted = fit_ols(
        adjusted_data["totalscore"],
        adjusted_data[["tracking", *controls]],
        ("tracking", *controls),
    )
    adjusted_inference = coefficient_inference(
        adjusted,
        "tracking",
        "Küme",
        groups=adjusted_data["schoolid"],
    )
    return pd.DataFrame(
        (
            {
                "Tahmin": "Ham fark",
                "Düzeyleme katsayısı": raw_inference.estimate,
                "Küme standart hata": raw_inference.standard_error,
                "%95 GA alt": raw_inference.confidence_interval[0],
                "%95 GA üst": raw_inference.confidence_interval[1],
                "Gözlem": raw.nobs,
            },
            {
                "Tahmin": "Kovaryat ayarlı",
                "Düzeyleme katsayısı": adjusted_inference.estimate,
                "Küme standart hata": adjusted_inference.standard_error,
                "%95 GA alt": adjusted_inference.confidence_interval[0],
                "%95 GA üst": adjusted_inference.confidence_interval[1],
                "Gözlem": adjusted.nobs,
            },
        )
    )


def _load_trial_frame() -> tuple[pd.DataFrame | None, str]:
    source = st.segmented_control(
        "Veri kaynağı",
        options=("Kontrollü okul deneyi", "Hazırlanmış DDK CSV"),
        default="Kontrollü okul deneyi",
        key="konu03_trial_source",
    )
    if source == "Kontrollü okul deneyi":
        students = st.slider(
            "Öğrenci sayısı",
            min_value=600,
            max_value=3000,
            value=1200,
            step=300,
            key="konu03_trial_students",
        )
        return _trial_data(students), "Kontrollü okul-kümeli deney"

    uploaded = st.file_uploader(
        "Hazırlanmış DDK CSV",
        type=("csv",),
        key="konu03_ddk_upload",
    )
    if uploaded is None:
        st.info(
            "DDK verisi lisans teyidi olmadan depoya eklenmez. Hazırlanmış CSV yalnız "
            "bu oturumda doğrulanarak kullanılabilir."
        )
        return None, "DDK2011"
    try:
        return load_registered_csv("ddk2011", uploaded.getvalue()), "DDK2011 kullanıcı dosyası"
    except (ValueError, pd.errors.ParserError) as error:
        st.error(f"DDK dosyası doğrulanamadı: {error}")
        return None, "DDK2011"


def _render_random_assignment() -> None:
    frame, data_label = _load_trial_frame()
    if frame is None:
        return
    table = _trial_results(frame)
    st.dataframe(
        table.style.format(
            {
                "Düzeyleme katsayısı": "{:.3f}",
                "Küme standart hata": "{:.3f}",
                "%95 GA alt": "{:.3f}",
                "%95 GA üst": "{:.3f}",
                "Gözlem": "{:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    figure = go.Figure(
        go.Scatter(
            x=table["Düzeyleme katsayısı"],
            y=table["Tahmin"],
            mode="markers",
            name="Düzeyleme etkisi",
            marker={"size": 10, "color": "#107C89"},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": table["%95 GA üst"] - table["Düzeyleme katsayısı"],
                "arrayminus": table["Düzeyleme katsayısı"] - table["%95 GA alt"],
            },
        )
    )
    figure.add_vline(x=0, line_dash="dash", line_color="#51696C")
    style_figure(
        figure,
        title=f"{data_label}: düzeyleme etkisi ve okul-kümeli çıkarım",
        x_title="Toplam test puanı etkisi ve %95 güven aralığı",
        y_title="Tahmin spesifikasyonu",
    )
    show_figure(figure)

    if {"lowstream", "std_mark"}.issubset(frame.columns):
        selected = frame.loc[frame["tracking"].eq(1)].dropna(
            subset=["lowstream", "std_mark"]
        )
        selection_fit = fit_ols(
            selected["std_mark"], selected[["lowstream"]], ("lowstream",)
        )
        selection = coefficient_inference(
            selection_fit,
            "lowstream",
            "Küme",
            groups=selected["schoolid"],
        )
        st.warning(
            f"Düzeyleme okullarında alt başarı grubunun başlangıç farkı "
            f"{selection.estimate:.3f} puandır. Bu grup rastgele atanmadığı için alt-üst "
            "grup sonuç farkı tedavi etkisi değildir."
        )
    render_model_context(
        data_label=data_label,
        sample_label=f"n = {len(frame):,}; atama ve çıkarım birimi okul",
        model_label="toplam puan ~ düzeyleme (+ başlangıç kovaryatları)",
        inference_label="Okul-kümeli, küçük örneklem düzeltmeli",
        seed=TRIAL_SEED if data_label.startswith("Kontrollü") else None,
    )


def _render_output_reading() -> None:
    st.markdown("#### Nedensel sonuç tablosunu okuma sırası")
    st.write("1. Tahmin hedefi: hangi anakütle ve hangi ortalama etki?")
    st.write("2. Atama veya seçim mekanizması: karşı-olgusalı hangi grup temsil ediyor?")
    st.write("3. Tanımlama varsayımı: rastgele atama, koşullu bağımsızlık veya başka bir tasarım?")
    st.write("4. Çıkarım birimi: müdahale okulda ise standart hata okul düzeyinde kümelenmiş mi?")
    st.write("5. Duyarlılık: alternatif kontrol seti aynı hedefi mi, farklı bir karşılaştırmayı mı izliyor?")
    st.error("Yüksek R², küçük p-değeri veya büyük örneklem tek başına nedensel tanımlama kanıtı değildir.")


def render() -> None:
    render_topic_header(TOPIC_KEY)
    concept, endogeneity, assignment, output = st.tabs(
        ("Tanımlama zinciri", "İçsellik DGP", "Rastgele atama", "Çıktı okuma")
    )
    with concept:
        _render_concept_chain()
        render_reproduction_code(TOPIC_KEY, "zincir")
    with endogeneity:
        _render_endogeneity()
        render_reproduction_code(TOPIC_KEY, "icsellik")
    with assignment:
        _render_random_assignment()
        render_reproduction_code(TOPIC_KEY, "atama")
    with output:
        _render_output_reading()
        render_reproduction_code(TOPIC_KEY, "cikti")
    render_question(TOPIC_KEY)
