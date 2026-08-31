"""Konu 12: double selection, DML ve bütünleşik araştırma akışı."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.datasets import load_registered_csv
from core.dml import DMLDGPConfig, double_selection, fit_dml, simulate_dml_data
from core.inference import coefficient_inference
from core.ols import fit_ols
from core.research_workflow import RESEARCH_WORKFLOW, audit_workflow
from topics.regression_ui import render_model_context, show_figure, style_figure
from topics.shared import render_question, render_reproduction_code, render_topic_header


TOPIC_KEY = "konu12"
SEED = 812


@st.cache_data(show_spinner=False)
def _simulation(nobs: int, nfeatures: int, groups: int):
    return simulate_dml_data(DMLDGPConfig(
        nobs=nobs, nfeatures=nfeatures, groups=groups, seed=SEED
    ))


@st.cache_data(show_spinner=False)
def _selection(outcome, treatment, controls, names, folds: int):
    return double_selection(outcome, treatment, controls, names, folds=folds, seed=SEED)


@st.cache_data(show_spinner=False)
def _dml(outcome, treatment, controls, groups, folds: int, seed: int, learner: str):
    return fit_dml(
        outcome, treatment, controls, folds=folds, seed=seed,
        learner=learner, groups=groups,
    )


def _raw_theta(outcome, treatment) -> float:
    return fit_ols(
        outcome, np.asarray(treatment)[:, None], ("Hedef",)
    ).coefficient("Hedef")


def _render_selection(data, folds: int) -> None:
    selection = _selection(
        data.outcome, data.treatment, data.controls, data.feature_names, folds
    )
    table = pd.DataFrame({
        "Yöntem": ("Kontrolsüz fark", "Yalnız sonuç seçimi", "Çift seçim"),
        "Hedef katsayı": (
            _raw_theta(data.outcome, data.treatment),
            selection.outcome_only_theta,
            selection.double_selection_theta,
        ),
        "Seçili kontrol": (0, len(selection.outcome_selected), len(selection.union_selected)),
    })
    st.dataframe(
        table.style.format({"Hedef katsayı": "{:.3f}", "Seçili kontrol": "{:,.0f}"}),
        hide_index=True, width="stretch",
    )
    metrics = st.columns(3)
    metrics[0].metric("Gerçek theta", f"{data.theta:.2f}")
    metrics[1].metric("Sonuç modeli seçimi", str(len(selection.outcome_selected)))
    metrics[2].metric("Birleşim kümesi", str(len(selection.union_selected)))
    st.markdown("**Sonuç için seçilen:** " + (", ".join(selection.outcome_selected) or "Yok"))
    st.markdown("**Tedavi için seçilen:** " + (", ".join(selection.treatment_selected) or "Yok"))
    st.info(
        "Çift seçim, sonuç modelinin atlayabileceği fakat tedaviyi güçlü açıklayan "
        "kontrolleri ikinci seçim denkleminden birleşim kümesine geri alır."
    )
    st.warning(
        "Lasso ile seçilen modelde sıradan OLS güven aralığı seçim belirsizliğini otomatik "
        "hesaba katmaz. Bu sekme değişken seçimi mekanizmasını karşılaştırır."
    )


def _render_cross_fitting(data, folds: int, seed: int, learner: str) -> None:
    result = _dml(
        data.outcome, data.treatment, data.controls, data.groups,
        folds, seed, learner,
    )
    figure = go.Figure(go.Scatter(
        x=result.residualized_treatment,
        y=result.residualized_outcome,
        mode="markers",
        marker={
            "color": result.cross_fit.fold_assignments, "colorscale": "Viridis",
            "size": 6, "opacity": 0.55, "showscale": True,
            "colorbar": {"title": "Kat"},
        },
        name="Kat-dışı artıklar",
    ))
    xline = np.linspace(
        result.residualized_treatment.min(), result.residualized_treatment.max(), 50
    )
    figure.add_trace(go.Scatter(
        x=xline, y=result.theta * xline, mode="lines",
        line={"color": "#B3392F", "width": 3}, name="Ortogonal hedef eğimi",
    ))
    style_figure(
        figure, title="Kat-dışı yardımcı tahminlerden oluşturulan artıklar",
        x_title="Tedavi artığı D - m_hat(X)", y_title="Sonuç artığı Y - g_hat(X)",
    )
    show_figure(figure)
    metrics = st.columns(4)
    metrics[0].metric("DML hedef katsayısı", f"{result.theta:.3f}")
    metrics[1].metric("Küme standart hata", f"{result.standard_error:.3f}")
    metrics[2].metric("%95 GA alt", f"{result.confidence_interval[0]:.3f}")
    metrics[3].metric("%95 GA üst", f"{result.confidence_interval[1]:.3f}")
    fold_table = pd.DataFrame({
        "Kat": np.arange(result.cross_fit.fold_count),
        "Gözlem": [
            int((result.cross_fit.fold_assignments == fold).sum())
            for fold in range(result.cross_fit.fold_count)
        ],
        "Okul": [
            int(np.unique(data.groups[result.cross_fit.fold_assignments == fold]).size)
            for fold in range(result.cross_fit.fold_count)
        ],
    })
    st.dataframe(fold_table, hide_index=True, width="stretch")
    st.success(
        "Her okul tek bir doğrulama katında kaldı. Her yardımcı tahmin o gözlemin "
        "okulunu içermeyen eğitim katlarından geldi; hedef aşaması yalnız kat-dışı artıkları kullandı."
    )
    render_model_context(
        data_label="Kontrollü kısmen doğrusal DGP",
        sample_label=f"n = {len(data.outcome):,}; grup = {np.unique(data.groups).size}",
        model_label=f"{learner} yardımcı model; {folds}-katlı grup bölmeli artıklaştırma",
        inference_label="Okul-kümeli hedef denklemi", seed=seed,
    )


def _render_sensitivity(data, folds: int, learner: str) -> None:
    rows = []
    for seed in (812, 919, 1207):
        result = _dml(
            data.outcome, data.treatment, data.controls, data.groups,
            folds, seed, learner,
        )
        rows.append({
            "Bölünme tohumu": seed, "Hedef katsayısı": result.theta,
            "Küme SH": result.standard_error,
            "Alt %95": result.confidence_interval[0],
            "Üst %95": result.confidence_interval[1],
        })
    table = pd.DataFrame(rows)
    figure = go.Figure(go.Scatter(
        x=table["Hedef katsayısı"], y=table["Bölünme tohumu"].astype(str), mode="markers",
        marker={"size": 11, "color": "#107C89"},
        error_x={
            "type": "data", "symmetric": False,
            "array": table["Üst %95"] - table["Hedef katsayısı"],
            "arrayminus": table["Hedef katsayısı"] - table["Alt %95"],
        },
    ))
    figure.add_vline(x=data.theta, line_dash="dash", line_color="#B3392F")
    style_figure(
        figure, title="Aynı veri üzerinde çapraz uyarlama bölünmesi duyarlılığı",
        x_title="Hedef katsayısı ve %95 güven aralığı", y_title="Bölünme tohumu",
    )
    show_figure(figure)
    st.dataframe(
        table.style.format({
            "Hedef katsayısı": "{:.3f}", "Küme SH": "{:.3f}",
            "Alt %95": "{:.3f}", "Üst %95": "{:.3f}",
        }),
        hide_index=True, width="stretch",
    )
    st.warning(
        "Tek bir elverişli bölünme raporlanmaz. Bölünme tohumu oynaklığı, kat sayısı ve yardımcı model "
        "seçimi önceden tanımlanan duyarlılık setinin parçasıdır."
    )


def _ddk_arrays(frame: pd.DataFrame):
    controls = ("std_mark", "girl", "agetest", "sbm", "etpteacher", "percentile")
    working = frame.loc[:, ("totalscore", "tracking", "schoolid", *controls)].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    return (
        working,
        working["totalscore"].to_numpy(float),
        working["tracking"].to_numpy(float),
        working.loc[:, controls].to_numpy(float),
        working["schoolid"].to_numpy(),
    )


def _render_ddk(folds: int, seed: int, learner: str) -> None:
    st.markdown("#### DDK2011 okul düzeyleme laboratuvarı")
    uploaded = st.file_uploader(
        "Hazırlanmış DDK CSV", type=("csv",), key="konu12_ddk_upload"
    )
    if uploaded is None:
        st.info(
            "DDK verisi lisans teyidi olmadan depoya eklenmez. Hazırlanmış CSV yalnız "
            "bu oturumda doğrulanır; aynı okulun öğrencileri grup-katlı bölmeyle birlikte tutulur."
        )
        return
    try:
        frame = load_registered_csv("ddk2011", uploaded.getvalue())
        working, y, d, x, groups = _ddk_arrays(frame)
        unique_groups = np.unique(groups).size
        if len(working) < 300 or unique_groups < 5:
            raise ValueError("DML için en az 300 tam gözlem ve 5 okul gerekir.")
        effective_folds = min(folds, unique_groups)
        raw = fit_ols(y, d[:, None], ("tracking",))
        raw_inference = coefficient_inference(raw, "tracking", "Küme", groups=groups)
        result = fit_dml(
            y, d, x, folds=effective_folds, seed=seed,
            learner=learner, groups=groups,
        )
    except (ValueError, pd.errors.ParserError) as error:
        st.error(f"DDK dosyası doğrulanamadı: {error}")
        return
    table = pd.DataFrame({
        "Tahmin": ("Kontrolsüz OLS", f"{learner} DML"),
        "Düzeyleme etkisi": (raw_inference.estimate, result.theta),
        "Okul-kümeli SH": (raw_inference.standard_error, result.standard_error),
        "Alt %95": (raw_inference.confidence_interval[0], result.confidence_interval[0]),
        "Üst %95": (raw_inference.confidence_interval[1], result.confidence_interval[1]),
    })
    st.dataframe(
        table.style.format({
            "Düzeyleme etkisi": "{:.3f}", "Okul-kümeli SH": "{:.3f}",
            "Alt %95": "{:.3f}", "Üst %95": "{:.3f}",
        }),
        hide_index=True, width="stretch",
    )
    render_model_context(
        data_label="DDK2011 kullanıcı dosyası",
        sample_label=f"tam gözlem n = {len(working):,}; okul = {unique_groups:,}",
        model_label=f"{learner} DML; {effective_folds}-katlı grup bölmesi",
        inference_label="Okul-kümeli", seed=seed,
    )


def _render_workflow() -> None:
    st.markdown("#### Araştırma akışı denetimi")
    completed = st.multiselect(
        "Belgelenen aşamalar", options=tuple(stage.key for stage in RESEARCH_WORKFLOW),
        default=("estimand", "identification", "data", "cross_fit", "estimate"),
        format_func=lambda key: next(stage.label for stage in RESEARCH_WORKFLOW if stage.key == key),
        key="konu12_workflow",
    )
    audit = audit_workflow(set(completed))
    table = pd.DataFrame({
        "Durum": ["Tamam" if item.complete else "Eksik" for item in audit],
        "Aşama": [item.stage.label for item in audit],
        "Asgari çıktı": [item.stage.deliverable for item in audit],
    })
    st.dataframe(table, hide_index=True, width="stretch")
    remaining = sum(not item.complete for item in audit)
    if remaining:
        st.warning(f"Araştırma kaydında {remaining} aşama henüz belgelenmedi.")
    else:
        st.success("Tahmin hedefinden yeniden üretilebilirliğe kadar bütün aşamalar belgeli.")


def render() -> None:
    render_topic_header(TOPIC_KEY)
    controls = st.columns(4)
    nobs = controls[0].slider("Gözlem sayısı", 400, 1600, 800, 200, key="konu12_nobs")
    nfeatures = controls[1].slider("Kontrol sayısı", 10, 40, 24, 2, key="konu12_nfeatures")
    folds = controls[2].select_slider("Çapraz uyarlama katı", (3, 5, 8), value=5, key="konu12_folds")
    learner = controls[3].selectbox(
        "Yardımcı model", ("Ridge", "Random Forest"), key="konu12_learner",
        format_func=lambda value: "Rastgele orman" if value == "Random Forest" else value,
    )
    seed = st.number_input("Bölünme tohumu", 0, 9999, SEED, 1, key="konu12_seed")
    data = _simulation(nobs, nfeatures, max(20, nobs // 20))
    tabs = st.tabs((
        "Seçim stratejileri", "Çapraz uyarlama", "Bölünme duyarlılığı",
        "DDK ve araştırma akışı",
    ))
    with tabs[0]:
        _render_selection(data, folds)
        render_reproduction_code(TOPIC_KEY, "secim")
    with tabs[1]:
        _render_cross_fitting(data, folds, int(seed), learner)
        render_reproduction_code(TOPIC_KEY, "capraz")
    with tabs[2]:
        _render_sensitivity(data, folds, learner)
        render_reproduction_code(TOPIC_KEY, "duyarlilik")
    with tabs[3]:
        _render_ddk(folds, int(seed), learner)
        st.divider()
        _render_workflow()
        render_reproduction_code(TOPIC_KEY, "veri")
    render_question(TOPIC_KEY)
