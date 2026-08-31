"""Konu 11: model seçimi, çapraz doğrulama ve düzenlileştirme."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

from core.datasets import configured_cps_path, load_cps_csv
from core.pipelines import compare_holdout_models, cross_validate_lasso
from core.regularization import SparseDGPConfig, polynomial_complexity_curve, regularization_path, simulate_sparse_regression
from topics.regression_ui import render_model_context, show_figure, style_figure
from topics.shared import render_question, render_reproduction_code, render_topic_header


TOPIC_KEY = "konu11"
SEED = 811


@st.cache_data(show_spinner=False)
def _sparse_data(nobs: int, nfeatures: int, noise: float):
    return simulate_sparse_regression(SparseDGPConfig(
        nobs=nobs, nfeatures=nfeatures, nonzero=min(6, nfeatures - 1),
        noise_scale=noise, seed=SEED,
    ))


@st.cache_data(show_spinner=False)
def _selection_results(features, outcome, names, folds: int, seed: int):
    train_x, test_x, train_y, test_y = train_test_split(
        features, outcome, test_size=0.25, random_state=seed
    )
    cv = cross_validate_lasso(
        train_x, train_y, np.logspace(-3, 0.45, 36), names, folds=folds, seed=seed
    )
    comparisons = {
        rule: compare_holdout_models(
            train_x, train_y, test_x, test_y, names,
            lasso_alpha=cv.lambda_min if rule == "lambda_min" else cv.lambda_1se,
        )
        for rule in ("lambda_min", "lambda_1se")
    }
    return cv, comparisons, len(train_y), len(test_y)


def _render_bias_variance(nobs: int, noise: float) -> None:
    curve = polynomial_complexity_curve(
        nobs=max(300, nobs // 2), max_degree=12, noise_scale=noise, seed=SEED
    )
    figure = go.Figure()
    for values, name, color in (
        (curve.train_mse, "Eğitim MSE", "#107C89"),
        (curve.test_mse, "Sınama MSE", "#B3392F"),
    ):
        figure.add_trace(go.Scatter(
            x=curve.degrees, y=values, mode="lines+markers", name=name,
            line={"color": color, "width": 3},
        ))
    figure.add_vline(
        x=curve.selected_degree, line_dash="dash", line_color="#51696C",
        annotation_text="En düşük sınama MSE",
    )
    style_figure(
        figure, title="Model karmaşıklığı boyunca eğitim ve dış örneklem kaybı",
        x_title="Polinom derecesi", y_title="Ortalama karesel hata",
    )
    show_figure(figure)
    metrics = st.columns(3)
    metrics[0].metric("Seçili derece", str(curve.selected_degree))
    metrics[1].metric("En düşük sınama MSE", f"{curve.test_mse.min():.3f}")
    metrics[2].metric("Derece 12 eğitim MSE", f"{curve.train_mse[-1]:.3f}")
    st.warning(
        "Eğitim hatasının düşmesi tek başına model seçimi değildir. Bu mekanizma "
        "grafiğinden farklı olarak gerçek iş akışında ayar seçimi doğrulama katlarında, nihai "
        "performans yalnız bir kez ayrılmış sınama setinde değerlendirilir."
    )


def _render_paths(features, outcome, names) -> None:
    path = regularization_path(features, outcome, np.logspace(-3, 1.2, 42), names)
    family = st.segmented_control(
        "Ceza ailesi", ("Lasso", "Ridge"), default="Lasso",
        key="konu11_penalty_family",
    )
    coefficients = path.lasso_coefficients if family == "Lasso" else path.ridge_coefficients
    figure = go.Figure()
    for index, name in enumerate(path.feature_names):
        figure.add_trace(go.Scatter(
            x=path.alphas, y=coefficients[:, index], mode="lines", name=name,
            showlegend=index < 8, line={"width": 2 if index < 6 else 1},
            opacity=1.0 if index < 6 else 0.35,
        ))
    figure.update_xaxes(type="log", autorange="reversed")
    style_figure(
        figure, title=f"{family} katsayı yolu: ceza arttıkça modelin hareketi",
        x_title="Ceza alpha (soldan sağa azalır)", y_title="Standartlaştırılmış katsayı",
    )
    show_figure(figure)
    probe = st.select_slider(
        "Yolu inceleme alpha değeri", options=tuple(float(x) for x in path.alphas),
        value=float(path.alphas[len(path.alphas) // 2]),
        format_func=lambda value: f"{value:.4f}", key="konu11_path_alpha",
    )
    index = int(np.argmin(np.abs(path.alphas - probe)))
    nonzero = int((np.abs(coefficients[index]) > 1e-8).sum())
    st.metric("Sıfır olmayan katsayı", f"{nonzero} / {len(names)}")
    st.info(
        "Lasso bazı katsayıları tam sıfıra iter; Ridge ilişkili değişkenlerin "
        "katsayılarını birlikte ve sürekli küçültür."
    )


def _comparison_table(comparison) -> pd.io.formats.style.Styler:
    table = pd.DataFrame({
        "Model": comparison.model_names,
        "Sınama MSE": comparison.test_mse,
        "Etkin değişken": comparison.nonzero_counts,
    })
    return table.style.format({"Sınama MSE": "{:.4f}", "Etkin değişken": "{:,.0f}"})


def _render_cv(features, outcome, names, folds: int) -> None:
    cv, comparisons, train_n, test_n = _selection_results(features, outcome, names, folds, SEED)
    rule = st.segmented_control(
        "Ceza seçim kuralı", ("lambda_min", "lambda_1se"),
        default="lambda_1se", key="konu11_lambda_rule",
    )
    figure = go.Figure(go.Scatter(
        x=cv.alphas, y=cv.mean_mse, mode="lines+markers", name="Kat ortalaması",
        line={"color": "#107C89", "width": 3},
        error_y={"type": "data", "array": cv.standard_error_mse, "visible": True},
    ))
    figure.add_vline(x=cv.lambda_min, line_color="#2F9E6B", annotation_text="min")
    figure.add_vline(x=cv.lambda_1se, line_color="#B3392F", line_dash="dash", annotation_text="1se")
    figure.update_xaxes(type="log")
    style_figure(
        figure, title="Yalnız eğitim örneklemindeki K-katlı doğrulama kaybı",
        x_title="Lasso alpha", y_title="Doğrulama MSE",
    )
    show_figure(figure)
    selected = cv.selected_min if rule == "lambda_min" else cv.selected_1se
    metrics = st.columns(4)
    metrics[0].metric("lambda_min", f"{cv.lambda_min:.4f}")
    metrics[1].metric("lambda_1se", f"{cv.lambda_1se:.4f}")
    metrics[2].metric("Seçili değişken", str(len(selected)))
    metrics[3].metric("Sınama gözlemi", f"{test_n:,}")
    st.dataframe(_comparison_table(comparisons[rule]), hide_index=True, width="stretch")
    st.caption("Seçili küme: " + (", ".join(selected) if selected else "boş model"))
    st.success(
        "Ölçekleyici her katın yalnız eğitim parçasında fit edildi. Lambda doğrulama "
        "kaybından geldi; ayrılmış sınama seti yalnız son tabloda kullanıldı."
    )
    render_model_context(
        data_label="Kontrollü seyrek DGP",
        sample_label=f"eğitim n = {train_n:,}; sınama n = {test_n:,}",
        model_label=f"{folds}-katlı Lasso çapraz doğrulaması; kural = {rule}; Post-Lasso",
        inference_label="Dış örneklem kaybı; klasik seçim-sonrası GA yok", seed=SEED,
    )


def _cps_design(frame: pd.DataFrame):
    numeric = ("education", "experience", "experience2_100", "female", "hisp", "age")
    design = pd.concat((
        frame.loc[:, numeric].astype(float),
        pd.get_dummies(frame.loc[:, ["region", "race", "marital"]].astype(str), drop_first=True, dtype=float),
    ), axis=1)
    outcome = pd.to_numeric(frame["lwage"], errors="coerce")
    valid = design.notna().all(axis=1) & outcome.notna()
    return design.loc[valid].to_numpy(float), outcome.loc[valid].to_numpy(float), tuple(map(str, design.columns))


def _render_cps(default_data, folds: int) -> None:
    source = st.segmented_control(
        "Veri kaynağı", ("Kontrollü DGP", "Hazırlanmış CPS CSV"),
        default="Kontrollü DGP", key="konu11_data_source",
    )
    if source == "Kontrollü DGP":
        features, outcome, names = default_data.features, default_data.outcome, default_data.feature_names
        label = "Kontrollü seyrek DGP"
    else:
        uploaded = st.file_uploader("Hazırlanmış CPS CSV", type=("csv",), key="konu11_cps_upload")
        configured = configured_cps_path()
        try:
            if uploaded is not None:
                frame, label = load_cps_csv(uploaded.getvalue()), "CPS 2009 kullanıcı dosyası"
            elif configured is not None:
                frame, label = load_cps_csv(configured), "CPS 2009 ortam değişkeni"
            else:
                st.info(
                    "CPS verisi lisans teyidi olmadan depoya eklenmez. Hazırlanmış dosya "
                    "bu oturumda yüklenebilir veya IKT807_CPS_PATH ile gösterilebilir."
                )
                return
        except (OSError, ValueError, pd.errors.ParserError) as error:
            st.error(f"CPS dosyası doğrulanamadı: {error}")
            return
        if len(frame) > 20_000:
            frame = frame.sample(20_000, random_state=SEED).sort_index()
        features, outcome, names = _cps_design(frame)
        if len(outcome) < 200:
            st.error("CPS laboratuvarı için en az 200 tam gözlem gerekir.")
            return
    cv, comparisons, train_n, test_n = _selection_results(features, outcome, names, folds, SEED)
    st.dataframe(_comparison_table(comparisons["lambda_1se"]), hide_index=True, width="stretch")
    st.caption(
        f"{label} | eğitim n = {train_n:,} | sınama n = {test_n:,} | "
        f"lambda_1se = {cv.lambda_1se:.4f} | kat-içi ölçekleme"
    )
    st.warning(
        "Daha düşük ücret tahmin hatası, eğitim katsayısının nedensel eğitim getirisi "
        "olduğu anlamına gelmez. Öngörü hedefi ile nedensel tahmin hedefi ayrı tutulur."
    )


def render() -> None:
    render_topic_header(TOPIC_KEY)
    controls = st.columns(4)
    nobs = controls[0].slider("Gözlem sayısı", 400, 1600, 800, 200, key="konu11_nobs")
    nfeatures = controls[1].slider("Aday değişken sayısı", 10, 50, 30, 5, key="konu11_nfeatures")
    noise = controls[2].slider("Gürültü ölçeği", 0.5, 2.0, 1.0, 0.1, key="konu11_noise")
    folds = controls[3].select_slider("Çapraz doğrulama katı", (3, 5, 10), value=5, key="konu11_folds")
    data = _sparse_data(nobs, nfeatures, noise)
    tabs = st.tabs(("Yanlılık-varyans", "Ridge ve Lasso yolu", "Çapraz doğrulama ve Post-Lasso", "CPS laboratuvarı"))
    with tabs[0]:
        _render_bias_variance(nobs, noise)
        render_reproduction_code(TOPIC_KEY, "yanlilik")
    with tabs[1]:
        _render_paths(data.features, data.outcome, data.feature_names)
        render_reproduction_code(TOPIC_KEY, "ceza")
    with tabs[2]:
        _render_cv(data.features, data.outcome, data.feature_names, folds)
        render_reproduction_code(TOPIC_KEY, "dogrulama")
    with tabs[3]:
        _render_cps(data, folds)
        render_reproduction_code(TOPIC_KEY, "veri")
    render_question(TOPIC_KEY)
