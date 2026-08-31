"""Konu 05: LPM, Logit, Probit, olasılıklar ve marjinal etkiler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.datasets import configured_cps_path, load_cps_csv
from core.discrete import (
    BinaryDGPConfig,
    BinaryModelFit,
    fit_binary_model,
    simulate_binary_data,
)
from core.marginal_effects import (
    average_marginal_effect,
    finite_difference,
    observation_marginal_effects,
)
from topics.regression_ui import render_model_context, show_figure, style_figure
from topics.shared import render_question, render_reproduction_code, render_topic_header


TOPIC_KEY = "konu05"
SEED = 805
MODELS = ("LPM", "Logit", "Probit")


@st.cache_data(show_spinner=False)
def _simulation(nobs: int, age_effect: float, intercept: float) -> pd.DataFrame:
    return simulate_binary_data(
        BinaryDGPConfig(
            nobs=nobs,
            seed=SEED,
            age_effect=age_effect,
            intercept=intercept,
        )
    )


def _fit_models(
    frame: pd.DataFrame,
    *,
    outcome: str = "married",
    features: tuple[str, ...] = ("age", "education", "metro"),
) -> dict[str, BinaryModelFit]:
    return {
        model: fit_binary_model(
            frame[outcome],
            frame[list(features)],
            features,
            model,
        )
        for model in MODELS
    }


def _render_probability_curves(
    frame: pd.DataFrame,
    fits: dict[str, BinaryModelFit],
) -> None:
    age_grid = np.linspace(18, 35, 120)
    education = float(frame["education"].mean())
    metro = float(frame["metro"].mean())
    design = np.column_stack(
        (
            np.ones(age_grid.size),
            age_grid,
            np.full(age_grid.size, education),
            np.full(age_grid.size, metro),
        )
    )
    figure = go.Figure()
    styles = {
        "LPM": ("#51696C", "dash", "circle"),
        "Logit": ("#107C89", "solid", "square"),
        "Probit": ("#2F9E6B", "dot", "diamond"),
    }
    for model, fit in fits.items():
        color, dash, symbol = styles[model]
        figure.add_trace(
            go.Scatter(
                x=age_grid,
                y=fit.predict(design),
                mode="lines",
                name=model,
                line={"color": color, "dash": dash, "width": 3},
                marker={"symbol": symbol},
            )
        )
    figure.add_hline(y=0, line_color="#B3392F", line_width=1)
    figure.add_hline(y=1, line_color="#B3392F", line_width=1)
    style_figure(
        figure,
        title="Aynı ikili sonuç için üç koşullu olasılık modeli",
        x_title="Yaş (yıl)",
        y_title="Tahmin edilen evli olma olasılığı",
    )
    show_figure(figure)
    lpm = fits["LPM"].predicted_probabilities
    outside = int(((lpm < 0) | (lpm > 1)).sum())
    metrics = st.columns(3)
    metrics[0].metric("Örneklem olay oranı", f"%{100 * frame['married'].mean():.1f}")
    metrics[1].metric("LPM sınır dışı tahmin", f"{outside:,}")
    metrics[2].metric("Logit yakınsama", "Evet" if fits["Logit"].converged else "Hayır")
    st.info(
        "LPM katsayısı doğrudan olasılık eğimidir. Logit ve Probit katsayıları tek "
        "indeks ölçeğindedir; olasılık etkisi bağlantı fonksiyonuyla dönüştürülür."
    )
    render_model_context(
        data_label="Kontrollü ikili sonuç DGP'si",
        sample_label=f"n = {len(frame):,}; yaş 18-35",
        model_label="evli ~ yaş + eğitim + metropol",
        inference_label="HC1 / robust GLM",
        seed=SEED,
    )


def _render_marginal_effects(
    frame: pd.DataFrame,
    fits: dict[str, BinaryModelFit],
) -> None:
    target = st.segmented_control(
        "Etki hedefi",
        options=("Yaş: sürekli değişim", "Metropol: 0 → 1"),
        default="Yaş: sürekli değişim",
        key="konu05_effect_target",
    )
    continuous = target.startswith("Yaş")
    rows = []
    for model, fit in fits.items():
        result = (
            average_marginal_effect(fit, "age")
            if continuous
            else finite_difference(fit, "metro")
        )
        coefficient_name = "age" if continuous else "metro"
        rows.append(
            {
                "Model": model,
                "Ham katsayı": fit.coefficient(coefficient_name),
                "Olasılık ölçeğinde etki": result.effect,
                "Delta standart hata": result.standard_error,
                "Etki tanımı": result.effect_type,
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(
        table.style.format(
            {
                "Ham katsayı": "{:.4f}",
                "Olasılık ölçeğinde etki": "{:.4f}",
                "Delta standart hata": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    if continuous:
        figure = go.Figure()
        for model, color in (
            ("LPM", "#51696C"),
            ("Logit", "#107C89"),
            ("Probit", "#2F9E6B"),
        ):
            effects = observation_marginal_effects(fits[model], "age")
            order = np.argsort(frame["age"].to_numpy())
            figure.add_trace(
                go.Scatter(
                    x=frame["age"].to_numpy()[order],
                    y=effects[order],
                    mode="markers",
                    name=model,
                    marker={"size": 5, "opacity": 0.35, "color": color},
                )
            )
        style_figure(
            figure,
            title="Yaşın bireysel olasılık marjinal etkileri",
            x_title="Yaş (yıl)",
            y_title="Bir yıllık yaş artışının olasılık etkisi",
        )
        show_figure(figure)
    else:
        st.warning(
            "Metropol bir kukla değişkendir. Türev yerine her gözlemde metropol=0 ve "
            "metropol=1 tahminleri arasındaki sonlu fark hesaplanıp ortalanır."
        )
    st.caption(
        "AME gerçek örneklem kovaryat dağılımında hesaplanır. Marjinal etki, "
        "tanımlama varsayımı olmadan otomatik olarak nedensel etki değildir."
    )


def _load_cps_lab(default_frame: pd.DataFrame) -> tuple[pd.DataFrame | None, str, tuple[str, ...]]:
    source = st.segmented_control(
        "Veri kaynağı",
        options=("Kontrollü DGP", "Hazırlanmış CPS CSV"),
        default="Kontrollü DGP",
        key="konu05_data_source",
    )
    if source == "Kontrollü DGP":
        return default_frame, "Kontrollü ikili sonuç DGP'si", ("age", "education", "metro")

    uploaded = st.file_uploader(
        "Hazırlanmış CPS CSV",
        type=("csv",),
        key="konu05_cps_upload",
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
            return None, "CPS 2009", ("age", "education", "hisp")
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"CPS dosyası doğrulanamadı: {error}")
        return None, "CPS 2009", ("age", "education", "hisp")

    cps = cps.loc[(cps["age"] <= 35) & cps["female"].eq(0)].copy()
    marital = pd.to_numeric(cps["marital"], errors="coerce")
    cps["married"] = marital.le(3).astype(int)
    cps = cps.dropna(subset=["age", "education", "hisp", "marital"])
    return cps, label, ("age", "education", "hisp")


def _render_data_lab(default_frame: pd.DataFrame) -> None:
    frame, label, features = _load_cps_lab(default_frame)
    if frame is None:
        return
    fits = _fit_models(frame, features=features)
    rows = []
    for model, fit in fits.items():
        age_effect = average_marginal_effect(fit, "age")
        education_effect = average_marginal_effect(fit, "education")
        rows.append(
            {
                "Model": model,
                "Yaş AME": age_effect.effect,
                "Yaş AME SH": age_effect.standard_error,
                "Eğitim AME": education_effect.effect,
                "Eğitim AME SH": education_effect.standard_error,
                "Gözlem": fit.nobs,
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(
        table.style.format(
            {
                "Yaş AME": "{:.4f}",
                "Yaş AME SH": "{:.4f}",
                "Eğitim AME": "{:.4f}",
                "Eğitim AME SH": "{:.4f}",
                "Gözlem": "{:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    render_model_context(
        data_label=label,
        sample_label=f"n = {len(frame):,}; 35 yaş ve altı çalışan erkekler / DGP karşılığı",
        model_label="evli ~ yaş + eğitim + grup göstergesi",
        inference_label="Ortalama marjinal etki, delta yöntemi",
        seed=SEED if label.startswith("Kontrollü") else None,
    )


def _render_model_family() -> None:
    support = st.selectbox(
        "Sonuç değişkeninin desteği",
        options=(
            "0/1 ikili sonuç",
            "Sırasız çok kategori",
            "Sıralı kategori",
            "Negatif olmayan sayım",
        ),
        key="konu05_support",
    )
    mapping = {
        "0/1 ikili sonuç": ("LPM / Logit / Probit", "Koşullu gerçekleşme olasılığı"),
        "Sırasız çok kategori": ("Multinomial model", "Her kategori için koşullu olasılık"),
        "Sıralı kategori": ("Ordered Logit / Probit", "Eşiklerle belirlenen kategori olasılıkları"),
        "Negatif olmayan sayım": ("Poisson / Negatif Binom", "Koşullu olay sayısı veya oranı"),
    }
    family, target = mapping[support]
    first, second = st.columns(2)
    first.metric("Aday model ailesi", family)
    second.metric("Doğal tahmin hedefi", target)
    st.info(
        "Model seçimi yalnız yazılımdaki komut adına değil, sonuç desteğine, araştırma "
        "tahmin hedefine ve koşullu dağılım varsayımına dayanır."
    )


def render() -> None:
    render_topic_header(TOPIC_KEY)
    controls = st.columns(3)
    nobs = controls[0].slider(
        "Gözlem sayısı",
        min_value=600,
        max_value=3000,
        value=1400,
        step=200,
        key="konu05_nobs",
    )
    age_effect = controls[1].slider(
        "Yaş indeks katsayısı",
        min_value=0.05,
        max_value=0.35,
        value=0.20,
        step=0.05,
        key="konu05_age_effect",
    )
    intercept = controls[2].slider(
        "Temel olay eğilimi",
        min_value=-7.0,
        max_value=-4.0,
        value=-6.0,
        step=0.5,
        key="konu05_intercept",
    )
    frame = _simulation(nobs, age_effect, intercept)
    fits = _fit_models(frame)

    curves, effects, data_lab, family = st.tabs(
        ("Olasılık eğrileri", "Marjinal etkiler", "CPS laboratuvarı", "Model seçimi")
    )
    with curves:
        _render_probability_curves(frame, fits)
        render_reproduction_code(TOPIC_KEY, "egriler")
    with effects:
        _render_marginal_effects(frame, fits)
        render_reproduction_code(TOPIC_KEY, "etkiler")
    with data_lab:
        _render_data_lab(frame)
        render_reproduction_code(TOPIC_KEY, "veri")
    with family:
        _render_model_family()
        render_reproduction_code(TOPIC_KEY, "secim")
    render_question(TOPIC_KEY)
