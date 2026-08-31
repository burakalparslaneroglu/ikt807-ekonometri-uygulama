"""Konu 09: sharp/fuzzy RDD, bandwidth duyarlılığı ve tasarım tanıları."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.datasets import load_registered_csv
from core.rdd import (
    RDDDGPConfig,
    RDDEstimate,
    density_ratio_near_cutoff,
    fit_fuzzy_rdd,
    fit_sharp_rdd,
    fitted_rdd_lines,
    placebo_estimates,
    simulate_rdd_data,
)
from topics.regression_ui import render_model_context, show_figure, style_figure
from topics.shared import render_question, render_reproduction_code, render_topic_header


TOPIC_KEY = "konu09"
SEED = 809
LM_CUTOFF = 59.1984


@st.cache_data(show_spinner=False)
def _simulation(
    nobs: int,
    treatment_effect: float,
    first_stage_jump: float,
    manipulation_strength: float = 0.0,
) -> pd.DataFrame:
    return simulate_rdd_data(
        RDDDGPConfig(
            nobs=nobs,
            seed=SEED,
            treatment_effect=treatment_effect,
            first_stage_jump=first_stage_jump,
            manipulation_strength=manipulation_strength,
        )
    )


def _binned_means(
    frame: pd.DataFrame,
    outcome: str,
    *,
    minimum: float = -10,
    maximum: float = 10,
    bins: int = 32,
) -> pd.DataFrame:
    edges = np.linspace(minimum, maximum, bins + 1)
    working = frame.loc[
        frame["running"].between(minimum, maximum), ["running", outcome]
    ].copy()
    working["bin"] = pd.cut(working["running"], edges, include_lowest=True)
    summary = (
        working.groupby("bin", observed=True)
        .agg(running=("running", "mean"), outcome=(outcome, "mean"), n=(outcome, "size"))
        .dropna()
    )
    return summary


def _rdd_figure(
    frame: pd.DataFrame,
    outcome: str,
    fit: RDDEstimate,
    title: str,
) -> go.Figure:
    summary = _binned_means(frame, outcome)
    left_x, left_y, right_x, right_y = fitted_rdd_lines(fit)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=summary["running"],
            y=summary["outcome"],
            mode="markers",
            name="Bin ortalaması",
            marker={
                "size": np.clip(np.sqrt(summary["n"]) * 2.2, 6, 15),
                "color": "#51696C",
                "opacity": 0.7,
            },
        )
    )
    figure.add_trace(
        go.Scatter(
            x=left_x,
            y=left_y,
            mode="lines",
            name="Eşik solu",
            line={"color": "#107C89", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=right_x,
            y=right_y,
            mode="lines",
            name="Eşik sağı",
            line={"color": "#2F9E6B", "width": 3},
        )
    )
    figure.add_vline(x=fit.cutoff, line_color="#B3392F", line_dash="dash")
    style_figure(
        figure,
        title=title,
        x_title="Merkezlenmiş eşik değişkeni",
        y_title="Sonuç",
    )
    return figure


def _render_sharp(
    frame: pd.DataFrame,
    bandwidth: float,
    kernel: str,
    degree: int,
) -> None:
    fit = fit_sharp_rdd(
        frame["outcome_sharp"],
        frame["running"],
        bandwidth=bandwidth,
        degree=degree,
        kernel=kernel,
    )
    show_figure(
        _rdd_figure(
            frame,
            "outcome_sharp",
            fit,
            "Kesin RDD: eşik sağındaki limit eksi solundaki limit",
        )
    )
    metrics = st.columns(4)
    metrics[0].metric("RDD sıçraması", f"{fit.estimate:.3f}")
    metrics[1].metric("HC1 standart hata", f"{fit.standard_error:.3f}")
    metrics[2].metric("Eşik solu n", f"{fit.n_left:,}")
    metrics[3].metric("Eşik sağı n", f"{fit.n_right:,}")
    st.info(
        "İşaret yönü sağ limit eksi sol limittir. Kesin tasarımda tedavi durumu eşikte "
        "0'dan 1'e deterministik geçtiği için sonuç sıçraması yerel tedavi etkisini hedefler."
    )
    render_model_context(
        data_label="Kontrollü kesin RDD veri üretim süreci",
        sample_label=(
            f"Toplam n = {len(frame):,}; yerel n = {fit.effective_n:,} "
            f"({fit.n_left:,} sol + {fit.n_right:,} sağ)"
        ),
        model_label=f"Ayrı yerel polinomlar; derece = {degree}; h = {bandwidth:.1f}",
        inference_label=f"HC1; {kernel} çekirdeği",
        seed=SEED,
    )


def _render_sensitivity(
    frame: pd.DataFrame,
    kernel: str,
    degree: int,
) -> None:
    bandwidths = np.arange(2.0, 8.1, 1.0)
    fits = [
        fit_sharp_rdd(
            frame["outcome_sharp"],
            frame["running"],
            bandwidth=float(value),
            degree=degree,
            kernel=kernel,
        )
        for value in bandwidths
    ]
    estimates = np.array([fit.estimate for fit in fits])
    errors = np.array([fit.standard_error for fit in fits])
    effective = np.array([fit.effective_n for fit in fits])
    figure = go.Figure(
        go.Scatter(
            x=bandwidths,
            y=estimates,
            mode="lines+markers",
            name="RDD tahmini",
            line={"color": "#107C89", "width": 3},
            error_y={"type": "data", "array": 1.96 * errors, "visible": True},
            customdata=effective,
            hovertemplate="h=%{x:.1f}<br>τ=%{y:.3f}<br>yerel n=%{customdata}<extra></extra>",
        )
    )
    style_figure(
        figure,
        title="Bant genişliği boyunca yerel tahmin ve yaklaşık %95 aralık",
        x_title="Bant genişliği h",
        y_title="Eşik sıçraması",
    )
    show_figure(figure)
    table = pd.DataFrame(
        {
            "Bant genişliği": bandwidths,
            "Tahmin": estimates,
            "HC1 SH": errors,
            "Alt %95": estimates - 1.96 * errors,
            "Üst %95": estimates + 1.96 * errors,
            "Yerel n": effective,
        }
    )
    st.dataframe(
        table.style.format(
            {
                "Bant genişliği": "{:.1f}",
                "Tahmin": "{:.3f}",
                "HC1 SH": "{:.3f}",
                "Alt %95": "{:.3f}",
                "Üst %95": "{:.3f}",
                "Yerel n": "{:,.0f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.warning(
        "En büyük veya en anlamlı sıçramayı veren bant genişliği seçilmez. Duyarlılık tablosu "
        "sonuç avcılığı için değil, yerellik ile örnekleme belirsizliği dengesini okumak içindir."
    )


def _render_fuzzy(
    nobs: int,
    treatment_effect: float,
    bandwidth: float,
    kernel: str,
    degree: int,
) -> None:
    first_stage_jump = st.slider(
        "Tedavi olasılığı sıçraması",
        0.10,
        0.70,
        0.55,
        0.05,
        key="konu09_first_stage_jump",
    )
    frame = _simulation(nobs, treatment_effect, first_stage_jump)
    result = fit_fuzzy_rdd(
        frame["outcome_fuzzy"],
        frame["treatment"],
        frame["running"],
        bandwidth=bandwidth,
        degree=degree,
        kernel=kernel,
    )
    metrics = st.columns(3)
    metrics[0].metric("İndirgenmiş biçim", f"{result.reduced_form:.3f}")
    metrics[1].metric("İlk aşama", f"{result.first_stage:.3f}")
    metrics[2].metric("Yerel Wald oranı", f"{result.local_wald:.3f}")
    show_figure(
        _rdd_figure(
            frame,
            "outcome_fuzzy",
            result.outcome_fit,
            "Bulanık RDD: sonuç denklemindeki eşik sıçraması",
        )
    )
    st.info(
        "Bulanık tasarımda tedavi olasılığı eşikte sıçrar fakat 0'dan 1'e zorunlu geçmez. "
        "Yerel Wald oranı, sonuç sıçramasını tedavi olasılığı sıçramasına böler."
    )
    if first_stage_jump <= 0.15 or abs(result.first_stage) < 0.20:
        st.error(
            "İlk aşama zayıf: tasarlanan olasılık sıçraması küçük ve yerel Wald oranı "
            "paydadaki örnekleme oynaklığına çok duyarlı."
        )


def _render_diagnostics(
    nobs: int,
    treatment_effect: float,
    kernel: str,
) -> None:
    manipulation = st.slider(
        "Eşik çevresi manipülasyon gücü",
        0.0,
        1.0,
        0.0,
        0.1,
        key="konu09_manipulation",
    )
    frame = _simulation(nobs, treatment_effect, 0.55, manipulation)
    ratio, left_count, right_count = density_ratio_near_cutoff(
        frame["running"], window=1.5
    )
    figure = go.Figure(
        go.Histogram(
            x=frame["running"],
            xbins={"start": -10, "end": 10, "size": 0.5},
            marker={"color": "#107C89"},
            name="Eşik değişkeni",
        )
    )
    figure.add_vline(x=0, line_color="#B3392F", line_dash="dash")
    style_figure(
        figure,
        title="Eşik değişkeninin yoğunluk görünümü",
        x_title="Merkezlenmiş eşik değişkeni",
        y_title="Gözlem sayısı",
    )
    show_figure(figure)
    metrics = st.columns(3)
    metrics[0].metric("Yakın sol gözlem", f"{left_count:,}")
    metrics[1].metric("Yakın sağ gözlem", f"{right_count:,}")
    metrics[2].metric("Sağ / sol yoğunluk oranı", f"{ratio:.2f}")
    placebos = placebo_estimates(
        frame["outcome_sharp"],
        frame["running"],
        (-6.0, 6.0),
        bandwidth=2.0,
        kernel=kernel,
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Placebo eşik": (-6.0, 6.0),
                "Tahmin": [fit.estimate for fit in placebos],
                "HC1 SH": [fit.standard_error for fit in placebos],
            }
        ).style.format({"Tahmin": "{:.3f}", "HC1 SH": "{:.3f}"}),
        hide_index=True,
        width="stretch",
    )
    if ratio > 1.5 or ratio < 1 / 1.5:
        st.error(
            "Eşik çevresinde belirgin yığılma var. Bu görünüm tek başına resmi yoğunluk "
            "testi değildir, fakat kesin manipülasyon yokluğu varsayımını sorgulatır."
        )
    else:
        st.success(
            "Bu kontrollü örnekte eşik çevresinde belirgin tek taraflı yığılma görünmüyor."
        )

    st.markdown("#### LM2007 Head Start veri laboratuvarı")
    uploaded = st.file_uploader(
        "Hazırlanmış LM2007 CSV",
        type=("csv",),
        key="konu09_lm_upload",
    )
    if uploaded is None:
        st.info(
            "LM2007 verisi lisans teyidi olmadan depoya eklenmez. Hazırlanmış CSV yalnız "
            "bu oturumda şema doğrulamasından sonra kullanılabilir."
        )
        return
    try:
        lm = load_registered_csv("lm2007", uploaded.getvalue())
        x = pd.to_numeric(lm["povrate60"], errors="coerce")
        y = pd.to_numeric(lm["mort_age59_related_postHS"], errors="coerce")
        valid = x.notna() & y.notna()
        x = x.loc[valid]
        y = y.loc[valid]
        rows = []
        for bandwidth in (4.0, 6.0, 8.0, 10.0, 12.0):
            fit = fit_sharp_rdd(
                y,
                x,
                cutoff=LM_CUTOFF,
                bandwidth=bandwidth,
                kernel=kernel,
            )
            rows.append(
                {
                    "h": bandwidth,
                    "Yerel n": fit.effective_n,
                    "Sağ - sol": fit.estimate,
                    "HC1 SH": fit.standard_error,
                }
            )
    except (ValueError, pd.errors.ParserError) as error:
        st.error(f"LM2007 dosyası doğrulanamadı: {error}")
        return
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {"h": "{:.1f}", "Yerel n": "{:,.0f}", "Sağ - sol": "{:.3f}", "HC1 SH": "{:.3f}"}
        ),
        hide_index=True,
        width="stretch",
    )
    render_model_context(
        data_label="LM2007 kullanıcı dosyası",
        sample_label=f"Tam gözlem n = {len(x):,}; eşik = {LM_CUTOFF:.4f}",
        model_label="Head Start ilişkili ölüm oranı; ayrı yerel doğrusal eğriler",
        inference_label=f"HC1; {kernel} çekirdeği",
    )


def render() -> None:
    render_topic_header(TOPIC_KEY)
    controls = st.columns(3)
    nobs = controls[0].slider(
        "Gözlem sayısı", 600, 4000, 1600, 200, key="konu09_nobs"
    )
    bandwidth = controls[1].slider(
        "Bant genişliği h", 2.0, 8.0, 4.0, 0.5, key="konu09_bandwidth"
    )
    treatment_effect = controls[2].slider(
        "Gerçek eşik etkisi", -4.0, 1.0, -2.0, 0.25, key="konu09_effect"
    )
    method_controls = st.columns(2)
    kernel = method_controls[0].selectbox(
        "Çekirdek", ("Üçgensel", "Uniform", "Epanechnikov"),
        format_func=lambda value: "Düzgün" if value == "Uniform" else value,
        key="konu09_kernel",
    )
    degree_label = method_controls[1].segmented_control(
        "Yerel polinom",
        options=("Doğrusal", "Karesel"),
        default="Doğrusal",
        key="konu09_degree",
    )
    degree = 1 if degree_label == "Doğrusal" else 2
    frame = _simulation(nobs, treatment_effect, 0.55)

    sharp_tab, sensitivity_tab, fuzzy_tab, diagnostics_tab = st.tabs(
        ("Kesin RDD", "Bant genişliği duyarlılığı", "Kesin ve bulanık RDD", "Tanılar ve LM2007")
    )
    with sharp_tab:
        _render_sharp(frame, bandwidth, kernel, degree)
        render_reproduction_code(TOPIC_KEY, "kesin")
    with sensitivity_tab:
        _render_sensitivity(frame, kernel, degree)
        render_reproduction_code(TOPIC_KEY, "duyarlilik")
    with fuzzy_tab:
        _render_fuzzy(nobs, treatment_effect, bandwidth, kernel, degree)
        render_reproduction_code(TOPIC_KEY, "bulanik")
    with diagnostics_tab:
        _render_diagnostics(nobs, treatment_effect, kernel)
        render_reproduction_code(TOPIC_KEY, "tanilar")
    render_question(TOPIC_KEY)
