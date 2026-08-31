"""Konu 10: yeniden örnekleme, bootstrap dağılımları ve güven aralıkları."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.bootstrap import BootstrapResult, bootstrap_ols_coefficient
from core.datasets import configured_cps_path, load_cps_csv
from core.resampling import draw_resample_indices, resample_frequencies
from core.simulation import WageDGPConfig, simulate_wage_data
from topics.regression_ui import render_model_context, show_figure, style_figure
from topics.shared import render_question, render_reproduction_code, render_topic_header


TOPIC_KEY = "konu10"
FEATURES = ("education", "experience", "experience2_100", "female")


@st.cache_data(show_spinner=False)
def _simulation(nobs: int, heteroskedasticity: float, seed: int) -> pd.DataFrame:
    return simulate_wage_data(
        WageDGPConfig(
            nobs=nobs,
            seed=seed,
            heteroskedasticity=heteroskedasticity,
            clusters=max(20, nobs // 20),
        )
    )


@st.cache_data(show_spinner="Bootstrap dağılımı hesaplanıyor...")
def _bootstrap(
    frame: pd.DataFrame,
    repetitions: int,
    seed: int,
    method: str,
) -> BootstrapResult:
    return bootstrap_ols_coefficient(
        frame["lwage"],
        frame[list(FEATURES)],
        FEATURES,
        "education",
        repetitions=repetitions,
        seed=seed,
        method=method,
    )


def _render_resampling_mechanism(frame: pd.DataFrame, seed: int) -> None:
    unit = st.segmented_control(
        "Yeniden örnekleme birimi",
        options=("Gözlem", "Küme"),
        default="Gözlem",
        key="konu10_resampling_unit",
    )
    step = st.slider(
        "Örnek çekim adımı",
        1,
        20,
        1,
        key="konu10_resample_step",
    )
    groups = frame["cluster"] if unit == "Küme" else None
    indices = draw_resample_indices(len(frame), seed=seed + step, groups=groups)
    frequencies = resample_frequencies(indices, len(frame))
    visible = np.arange(min(60, len(frame)))
    figure = go.Figure(
        go.Bar(
            x=visible,
            y=frequencies[visible],
            marker={"color": "#107C89"},
            name="Seçilme sayısı",
        )
    )
    style_figure(
        figure,
        title="İlk 60 gözlemin bootstrap örneğine seçilme sayısı",
        x_title="Orijinal gözlem indeksi",
        y_title="Seçilme sayısı",
    )
    show_figure(figure)
    metrics = st.columns(3)
    metrics[0].metric("Bootstrap satır sayısı", f"{len(indices):,}")
    metrics[1].metric("En az bir kez seçilen", f"{(frequencies > 0).sum():,}")
    metrics[2].metric("Birden çok seçilen", f"{(frequencies > 1).sum():,}")
    if unit == "Küme":
        st.info(
            "Küme seçildiğinde aynı kümedeki satırlar birlikte gelir; örnekleme birimi "
            "gözlem değil kümedir. Küme büyüklükleri farklıysa bootstrap satır sayısı değişebilir."
        )
    else:
        st.info(
            "Çiftler yeniden örneklemesi her çekimde n gözlem indeksini yerine koyarak seçer. Bazı "
            "gözlemler hiç gelmez, bazıları birden çok kez görünür."
        )


def _render_distribution(
    result: BootstrapResult,
) -> None:
    figure = go.Figure(
        go.Histogram(
            x=result.draws,
            nbinsx=35,
            marker={"color": "#107C89"},
            name=f"{result.method} bootstrap",
        )
    )
    figure.add_vline(
        x=result.original_estimate,
        line_color="#B3392F",
        line_dash="dash",
        annotation_text="Özgün tahmin",
    )
    style_figure(
        figure,
        title=f"Eğitim katsayısının {result.method.lower()} bootstrap dağılımı",
        x_title="Bootstrap eğitim katsayısı",
        y_title="Tekrar sayısı",
    )
    show_figure(figure)
    metrics = st.columns(4)
    metrics[0].metric("Eğitim tahmini", f"{result.original_estimate:.4f}")
    metrics[1].metric("HC1 standart hata", f"{result.analytic_standard_error:.4f}")
    metrics[2].metric("Bootstrap standart hata", f"{result.bootstrap_standard_error:.4f}")
    metrics[3].metric("SH Monte Carlo hatası", f"{result.monte_carlo_standard_error:.5f}")

    checkpoints = sorted(
        {value for value in (50, 100, 250, 500, result.repetitions) if value <= result.repetitions}
    )
    path = pd.DataFrame(
        {
            "B": checkpoints,
            "Bootstrap SH": [
                result.draws[:value].std(ddof=1) for value in checkpoints
            ],
        }
    )
    path["Yaklaşık MC hatası"] = path["Bootstrap SH"] / np.sqrt(
        2 * (path["B"] - 1)
    )
    st.dataframe(
        path.style.format(
            {"B": "{:,.0f}", "Bootstrap SH": "{:.5f}", "Yaklaşık MC hatası": "{:.5f}"}
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "B arttıkça azalan şey veri üretimindeki belirsizlik değil, sonlu bootstrap "
        "tekrarlarından kaynaklanan Monte Carlo oynaklığıdır."
    )


def _render_intervals(result: BootstrapResult) -> None:
    table = pd.DataFrame(
        (
            {
                "Aralık": "Normal bootstrap",
                "Alt": result.normal_interval[0],
                "Üst": result.normal_interval[1],
                "Kural": "β̂ ± 1.96 × bootstrap SH",
            },
            {
                "Aralık": "Yüzdelik",
                "Alt": result.percentile_interval[0],
                "Üst": result.percentile_interval[1],
                "Kural": "Bootstrap tahminlerinin %2.5 ve %97.5 kantilleri",
            },
            {
                "Aralık": "Öğrencileştirilmiş yüzdelik",
                "Alt": result.percentile_t_interval[0],
                "Üst": result.percentile_t_interval[1],
                "Kural": "Studentize bootstrap dağılımının ters kantilleri",
            },
        )
    )
    st.dataframe(
        table.style.format({"Alt": "{:.4f}", "Üst": "{:.4f}"}),
        hide_index=True,
        width="stretch",
    )
    figure = go.Figure()
    for index, row in table.iterrows():
        center = (row["Alt"] + row["Üst"]) / 2
        figure.add_trace(
            go.Scatter(
                x=[center],
                y=[row["Aralık"]],
                mode="markers",
                marker={"size": 10, "color": ("#107C89", "#2F9E6B", "#B3392F")[index]},
                error_x={
                    "type": "data",
                    "array": [row["Üst"] - center],
                    "arrayminus": [center - row["Alt"]],
                    "visible": True,
                },
                showlegend=False,
            )
        )
    figure.add_vline(x=result.original_estimate, line_color="#51696C", line_dash="dot")
    style_figure(
        figure,
        title="Aynı bootstrap dağılımından üç %95 aralık",
        x_title="Eğitim katsayısı",
        y_title="Aralık kuralı",
    )
    show_figure(figure)
    st.warning(
        "Bootstrap aralığı modelin tanımlama varsayımlarını yeniden örneklemez. İçsellik, "
        "yanlış tahmin hedefi veya veri toplama hatası yeniden örneklemeyle otomatik düzelmez."
    )


def _load_cps(default_frame: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    source = st.segmented_control(
        "Veri kaynağı",
        options=("Kontrollü DGP", "Hazırlanmış CPS CSV"),
        default="Kontrollü DGP",
        key="konu10_data_source",
    )
    if source == "Kontrollü DGP":
        return default_frame, "Kontrollü ücret DGP'si"
    uploaded = st.file_uploader(
        "Hazırlanmış CPS CSV",
        type=("csv",),
        key="konu10_cps_upload",
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
            return None, "CPS 2009"
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"CPS dosyası doğrulanamadı: {error}")
        return None, "CPS 2009"
    cps = cps.dropna(subset=["lwage", *FEATURES])
    sample_size = min(5000, len(cps))
    if sample_size < 100:
        st.error("CPS bootstrap laboratuvarı için en az 100 tam gözlem gerekir.")
        return None, label
    return cps.sample(sample_size, random_state=807).sort_index(), label


def _render_cps_lab(
    default_frame: pd.DataFrame,
    repetitions: int,
    seed: int,
    method: str,
) -> None:
    frame, label = _load_cps(default_frame)
    if frame is None:
        return
    effective_b = min(repetitions, 500) if label.startswith("CPS") else repetitions
    result = _bootstrap(frame, effective_b, seed, method)
    rows = pd.DataFrame(
        (
            {
                "Belirsizlik ölçüsü": "HC1 analitik SH",
                "Değer": result.analytic_standard_error,
            },
            {
                "Belirsizlik ölçüsü": f"{method} bootstrap SH",
                "Değer": result.bootstrap_standard_error,
            },
            {
                "Belirsizlik ölçüsü": "Yüzdelik alt sınır",
                "Değer": result.percentile_interval[0],
            },
            {
                "Belirsizlik ölçüsü": "Yüzdelik üst sınır",
                "Değer": result.percentile_interval[1],
            },
        )
    )
    st.dataframe(
        rows.style.format({"Değer": "{:.5f}"}),
        hide_index=True,
        width="stretch",
    )
    if effective_b < repetitions:
        st.info(
            f"Canlı CPS hesabı tepki süresi için B={effective_b:,} ile sınırlandı. "
            "Ders üretim benchmark'ı B=1.000 kullanır."
        )
    render_model_context(
        data_label=label,
        sample_label=f"n = {len(frame):,}; yeniden örnekleme birimi = gözlem",
        model_label="log ücret ~ eğitim + deneyim + deneyim² + kadın",
        inference_label=f"HC1 ve {method} bootstrap; B = {effective_b:,}",
        seed=seed,
    )


def render() -> None:
    render_topic_header(TOPIC_KEY)
    controls = st.columns(3)
    nobs = controls[0].slider(
        "Gözlem sayısı", 300, 1500, 700, 100, key="konu10_nobs"
    )
    repetitions = controls[1].select_slider(
        "Bootstrap tekrar sayısı B",
        options=(100, 250, 500, 1000),
        value=500,
        key="konu10_repetitions",
    )
    seed = controls[2].number_input(
        "Rastgelelik tohumu",
        min_value=0,
        max_value=9999,
        value=810,
        step=1,
        key="konu10_seed",
    )
    method_controls = st.columns(2)
    method = method_controls[0].segmented_control(
        "Bootstrap yöntemi",
        options=("Pairs", "Wild"),
        default="Pairs",
        format_func=lambda value: {
            "Pairs": "Çiftler",
            "Wild": "Çarpanlı",
        }[value],
        key="konu10_method",
    )
    heteroskedasticity = method_controls[1].slider(
        "Heteroskedastisite",
        0.0,
        2.0,
        1.2,
        0.2,
        key="konu10_heteroskedasticity",
    )
    frame = _simulation(nobs, heteroskedasticity, int(seed))
    result = _bootstrap(frame, int(repetitions), int(seed), method)

    mechanism_tab, distribution_tab, intervals_tab, cps_tab = st.tabs(
        ("Yeniden örnekleme", "Bootstrap dağılımı", "Güven aralıkları", "CPS laboratuvarı")
    )
    with mechanism_tab:
        _render_resampling_mechanism(frame, int(seed))
        render_reproduction_code(TOPIC_KEY, "ornekleme")
    with distribution_tab:
        _render_distribution(result)
        render_reproduction_code(TOPIC_KEY, "dagilim")
    with intervals_tab:
        _render_intervals(result)
        render_reproduction_code(TOPIC_KEY, "aralik")
    with cps_tab:
        _render_cps_lab(frame, int(repetitions), int(seed), method)
        render_reproduction_code(TOPIC_KEY, "veri")
    render_question(TOPIC_KEY)
