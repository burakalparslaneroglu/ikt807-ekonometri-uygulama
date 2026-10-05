"""Konu 11–12 işlemlerinin arayüzü: düzenlileştirilmiş modeller, CV eğrileri, katsayı yolları, DML ve tahmin grafikleri.

Hem Uygulama (``topics.lab_ui``) hem Sezgi (``topics.sim_ui``) sekmesi bu işlevleri kullanır. Grafikler üretilen
koddakiyle aynı renkleri ve katmanları taşır (``core.codegen.base.SERIES_COLORS``, ``GRAY``, ``DARK``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.codegen.base import DARK, GRAY, SERIES_COLORS
from core.labs.runner import CI_MULTIPLIER, LabState
from core.labs.spec import (
    CoefPath,
    ComplexityCurve,
    CrossFitDML,
    CVCurve,
    Dictionary,
    DMLSplits,
    DotPlot,
    DoubleSelection,
    EstimatePlot,
    LinePlot,
    ModelMetrics,
    Penalized,
    PostSelection,
    dictionary_terms,
)
from topics.regression_ui import show_figure, style_figure

PENALTY_LABELS = {"ols": "Cezasız EKK", "ridge": "Ridge", "lasso": "Lasso", "enet": "Elastic Net",
                  "postlasso": "Post-Lasso"}
SCALE_LABELS = {"ridge": "SSE ölçeği", "lasso": "yazılım ölçeği", "enet": "yazılım ölçeği"}
SCALE_NOTE = (
    "Ceza ölçekleri: Ridge λ SSE ölçeğindedir, (Y − Xβ)'(Y − Xβ) + λβ'β (Hansen; scikit-learn Ridge alpha). Lasso ve "
    "Elastic Net λ yazılım ölçeğindedir, (1/(2n))‖Y − Xβ‖² + λ[r‖β‖₁ + (1 − r)/2·‖β‖²] (scikit-learn, glmnet, Stata "
    "lasso); Hansen'in SSE ölçeğinde Lasso cezası 2nλ'dır."
)
_BAND = "rgba(16, 124, 137, 0.18)"


# --- Sayı biçimleri ----------------------------------------------------------------------------------

def number(value: float, decimals: int = 4) -> str:
    """Türkçe ondalık virgülüyle sayı; eksi işareti tipografik."""

    return f"{round(float(value), decimals) + 0.0:.{decimals}f}".replace(".", ",").replace("-", "−")


def count(value: float) -> str:
    return f"{int(value):,}".replace(",", ".")


def significant(value: float, digits: int = 4) -> str:
    """Anlamlı basamaklı gösterim, bilimsel gösterim olmadan: 20,13 · 0,1385 · 0,0001425."""

    value = float(value)
    if not np.isfinite(value):
        return "—"
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    decimals = max(0, digits - 1 - exponent)
    return number(value, decimals)


# --- Grafikler ---------------------------------------------------------------------------------------

def _vertical(figure: go.Figure, x: float, label: str, color: str, dash: str = "dash") -> None:
    """Legend'de görünen dikey çizgi (yardımcı eksende 0–1 arası)."""

    figure.add_trace(
        go.Scatter(x=[x, x], y=[0, 1], mode="lines", name=label, yaxis="y2", hoverinfo="skip",
                   line={"color": color, "width": 2, "dash": dash})
    )
    figure.update_layout(yaxis2={"overlaying": "y", "range": [0, 1], "visible": False})


def dot_figure(op: DotPlot, data: pd.DataFrame) -> go.Figure:
    values = data["deger"].to_numpy(dtype=float)
    labels = data["etiket"].tolist()
    figure = go.Figure(
        go.Scatter(
            x=values, y=labels, mode="markers+text", name=op.x_label,
            marker={"size": 12, "color": SERIES_COLORS[0][0]},
            text=[number(value, op.decimals) for value in values], textposition="middle right",
            hovertemplate="%{y}: %{x:." + str(op.decimals) + "f}<extra></extra>",
        )
    )
    span = float(values.max() - values.min()) or abs(float(values.max())) or 1.0
    figure.update_xaxes(range=[values.min() - 0.15 * span, values.max() + 0.35 * span])
    figure.update_yaxes(autorange="reversed")
    style_figure(figure, title=op.title, x_title=op.x_label, y_title="", legend_title="")
    figure.update_layout(showlegend=False)
    return figure


def cv_figure(op: CVCurve, data: pd.DataFrame, fit) -> go.Figure:
    grid = data["lambda"].to_numpy(dtype=float)
    mean, se = data["cv_ort"].to_numpy(dtype=float), data["cv_sh"].to_numpy(dtype=float)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=np.concatenate([grid, grid[::-1]]), y=np.concatenate([mean + se, (mean - se)[::-1]]),
                   fill="toself", fillcolor=_BAND, line={"width": 0}, name="±1 SH (katlar arası)", hoverinfo="skip")
    )
    figure.add_trace(
        go.Scatter(x=grid, y=mean, mode="lines", name="CV ortalama karesel hatası",
                   line={"color": SERIES_COLORS[0][0], "width": 3},
                   hovertemplate="λ = %{x:.4g}<br>CV = %{y:.5f}<extra></extra>")
    )
    _vertical(figure, fit.lam, f"Seçilen λ = {significant(fit.lam)}", DARK[0])
    figure.update_xaxes(type="log", exponentformat="power")
    style_figure(figure, title=op.title, x_title=op.x_label, y_title="CV ortalama karesel hatası", legend_title="")
    return figure


def path_figure(op: CoefPath, path: pd.DataFrame, fit) -> go.Figure:
    grid = path["lambda"].to_numpy(dtype=float)
    names = [name for name in path.columns if name != "lambda"]
    others = [name for name in names if name not in op.highlight]
    figure = go.Figure()
    for index, name in enumerate(others):
        figure.add_trace(
            go.Scatter(x=grid, y=path[name], mode="lines", name=op.other_label, legendgroup="diger",
                       showlegend=index == 0, line={"color": GRAY[0], "width": 1},
                       hovertemplate=f"{name}<br>λ = %{{x:.4g}}<br>katsayı = %{{y:.4f}}<extra></extra>")
        )
    for index, name in enumerate(op.highlight):
        figure.add_trace(
            go.Scatter(x=grid, y=path[name], mode="lines", name=op.highlight_label, legendgroup="vurgu",
                       showlegend=index == 0, line={"color": SERIES_COLORS[0][0], "width": 2.5},
                       hovertemplate=f"{name}<br>λ = %{{x:.4g}}<br>katsayı = %{{y:.4f}}<extra></extra>")
        )
    _vertical(figure, fit.lam, f"CV ile seçilen λ = {significant(fit.lam, 3)}", DARK[0])
    figure.update_xaxes(type="log", autorange="reversed", exponentformat="power")
    style_figure(figure, title=op.title, x_title=op.x_label, y_title="Katsayı", legend_title="")
    return figure


def estimate_figure(op: EstimatePlot, table: pd.DataFrame, splits) -> go.Figure:
    labels = table.index.tolist()
    position = np.arange(len(labels))[::-1].astype(float)
    estimate, error = table["tahmin"].to_numpy(dtype=float), table["sh"].to_numpy(dtype=float)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=estimate, y=position, mode="markers", name="Tahmin ve %95 güven aralığı",
            marker={"size": 11, "color": SERIES_COLORS[0][0]},
            error_x={"type": "data", "array": CI_MULTIPLIER * error, "color": SERIES_COLORS[0][0], "thickness": 2,
                     "width": 6},
            customdata=np.column_stack([error, estimate - CI_MULTIPLIER * error, estimate + CI_MULTIPLIER * error]),
            text=labels,
            hovertemplate="%{text}<br>tahmin = %{x:.4f}<br>SH = %{customdata[0]:.4f}<br>"
                          "%95 GA [%{customdata[1]:.4f}; %{customdata[2]:.4f}]<extra></extra>",
        )
    )
    if splits is not None:
        figure.add_trace(
            go.Scatter(
                x=splits, y=np.full(len(splits), position[op.splits_row] - 0.25), mode="markers",
                name="Farklı kat kurallarıyla DML tahminleri", marker={"size": 7, "color": GRAY[0]},
                hovertemplate="θ̂ = %{x:.4f}<extra></extra>",
            )
        )
    if op.truth is not None:
        _vertical(figure, op.truth, op.truth_label, SERIES_COLORS[1][0])
    style_figure(figure, title=op.title, x_title=op.x_label, y_title="", legend_title="")
    # Yalnız ana eksen: gerçek değer çizgisi yardımcı eksende (0–1) bütün yüksekliği kaplar.
    figure.update_layout(yaxis={"tickvals": position, "ticktext": labels,
                                "range": [position.min() - 0.7, position.max() + 0.7]})
    return figure


def line_figure(op: LinePlot, data: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    x = data.index.to_numpy()
    for (column, label), (color, _) in zip(op.columns, SERIES_COLORS):
        figure.add_trace(
            go.Scatter(x=x, y=data[column], mode="lines+markers", name=label, line={"color": color, "width": 2.5},
                       marker={"size": 7}, hovertemplate=f"{label}<br>%{{x}}: %{{y:.4f}}<extra></extra>")
        )
    figure.update_xaxes(tickmode="linear", dtick=1)
    style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label, legend_title="")
    return figure


def render_plot(op, state: LabState) -> bool:
    """Grafik işlemi ise çizer ve ``True`` döndürür."""

    if isinstance(op, DotPlot):
        show_figure(dot_figure(op, state.plots[f"nokta:{op.table}:{op.column}"]))
    elif isinstance(op, CVCurve):
        show_figure(cv_figure(op, state.plots[f"cv_egrisi:{op.model}"], state.models[op.model]))
    elif isinstance(op, CoefPath):
        show_figure(path_figure(op, state.plots[f"yol:{op.model}"], state.models[op.model]))
    elif isinstance(op, EstimatePlot):
        table, splits = state.plots[f"tahminler:{op.result}"]
        show_figure(estimate_figure(op, table, splits))
    elif isinstance(op, LinePlot):
        show_figure(line_figure(op, state.plots[f"cizgi:{op.table}:{op.title}"]))
    else:
        return False
    return True


# --- Tablolar ----------------------------------------------------------------------------------------

def metrics_table(op: ModelMetrics, table: pd.DataFrame, ops: dict | None = None) -> pd.DataFrame:
    """Notlardaki sütun sırası: test MSE, ‖β̂‖₂, sıfırdan farklı katsayı; cezalı modellerde seçilen λ ve ölçeği."""

    ops = ops or {}
    rows = []
    for model, label in op.rows:
        row = table.loc[model]
        penalty = getattr(ops.get(model), "penalty", None)
        lam = float(row["lambda"])
        scale = f" ({SCALE_LABELS[penalty]})" if penalty in SCALE_LABELS else ""
        rows.append({
            "Model": label,
            "Test MSE": number(row["test_mse"]),
            "‖β̂‖₂": number(row["norm"], 3),
            "Sıfırdan farklı katsayı": count(row["sifirdan"]),
            "Seçilen λ": "—" if not np.isfinite(lam) else significant(lam) + scale,
        })
    return pd.DataFrame(rows)


def estimates_table(table: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "Yöntem": table.index,
        "θ̂": [number(value, decimals) for value in table["tahmin"]],
        "SH": [number(value, decimals) for value in table["sh"]],
        "Yaklaşık %95 GA": [f"[{number(low, decimals)}; {number(high, decimals)}]"
                            for low, high in zip(table["alt"], table["ust"])],
    })


def complexity_table(table: pd.DataFrame) -> pd.DataFrame:
    shown = pd.DataFrame({"Derece d": table.index})
    for column, label, decimals in (("egitim_mse", "Eğitim MSE", 4), ("loocv", "LOOCV", 4), ("test_mse", "Test MSE", 4),
                                    ("aic", "AIC", 2), ("bic", "BIC", 2)):
        values = table[column].to_numpy(dtype=float)
        best = int(np.argmin(values))
        shown[label] = [number(value, decimals) + (" ◀" if index == best else "") for index, value in enumerate(values)]
    return shown


def folds_table(fit) -> pd.DataFrame:
    folds = fit.folds
    return pd.DataFrame({
        "Dış kat": folds["kat"].astype(int),
        "λ (Y denklemi)": [significant(value) for value in folds["lambda_y"]],
        "Y: sıfırdan farklı": folds["sifirdan_y"].astype(int),
        "λ (D denklemi)": [significant(value) for value in folds["lambda_d"]],
        "D: sıfırdan farklı": folds["sifirdan_d"].astype(int),
    })


def splits_table(table: pd.DataFrame, decimals: int = 3, se_label: str = "Küme SH") -> pd.DataFrame:
    return pd.DataFrame({
        "Dış kat kuralı: ⌊5{r·c}⌋ + 1, c =": table["etiket"],
        "θ̂": [number(value, decimals) for value in table["theta"]],
        se_label: [number(value, decimals) for value in table["sh"]],
    })


def experiment_table(op, state: LabState, ops: dict | None = None) -> pd.DataFrame | None:
    """Sezgi deneylerinde ``tables`` listesindeki Konu 11–12 sonuç tabloları için gösterim; tanımazsa ``None``."""

    if isinstance(op, ModelMetrics):
        return metrics_table(op, state.tables[op.result], ops)
    if isinstance(op, EstimatePlot):
        return estimates_table(state.tables[op.result])
    if isinstance(op, ComplexityCurve):
        return complexity_table(state.tables[op.result])
    return None


# --- Uygulama sekmesi: işlem özetleri ----------------------------------------------------------------

def _penalized_line(label: str, penalty: str, fit) -> str:
    parts = [f"**{label}**"]
    if penalty in SCALE_LABELS:
        parts.append(f"λ̂ = {significant(fit.lam)} ({SCALE_LABELS[penalty]})")
        if penalty == "enet":
            parts.append(f"L1 ağırlığı r = {number(fit.l1_ratio, 1)}")
    parts.append(f"{count(fit.nonzero)} katsayı sıfırdan farklı")
    if np.isfinite(fit.test_mse):
        parts.append(f"test MSE {number(fit.test_mse)}")
    parts.append(f"eğitim n = {count(fit.nobs)}")
    return " · ".join(parts)


def render_lab_op(spec, op, state: LabState, operations) -> bool:
    """Uygulama adımındaki Konu 11–12 işlemi; tanımazsa ``False``."""

    has_metrics = any(isinstance(item, ModelMetrics) for item in operations)
    if render_plot(op, state):
        return True
    if isinstance(op, Dictionary):
        st.markdown(f"**{op.comment}** — toplam {len(dictionary_terms(op))} aday terim.")
    elif isinstance(op, Penalized):
        if not has_metrics:
            st.markdown(_penalized_line(spec.label(op.name), op.penalty, state.models[op.name]))
    elif isinstance(op, PostSelection):
        if not has_metrics:
            st.markdown(_penalized_line(spec.label(op.name), "postlasso", state.models[op.name]))
    elif isinstance(op, ModelMetrics):
        ops = {model: state.ops.get(model) for model, _ in op.rows}
        st.markdown("**Dış-örneklem karşılaştırması**")
        st.dataframe(metrics_table(op, state.tables[op.result], ops), hide_index=True, width="stretch")
        st.caption(SCALE_NOTE)
    elif isinstance(op, CrossFitDML):
        fit = state.models[op.name]
        learner = "Lasso yardımcı modeller" if op.learner == "lasso" else "EKK yardımcı modeller"
        folds = f"{len(fit.folds)} dış kat" if op.outer else "çapraz uyarlama yok"
        se = "küme SH" if op.cluster else "HC1 SH"
        st.markdown(
            f"**{spec.label(op.name)}** ({folds}, {learner}): θ̂ = {number(fit.theta)} · {se} {number(fit.se)} · "
            f"n = {count(fit.nobs)}"
        )
        if op.outer and op.learner == "lasso":
            st.markdown("**Katlara göre yardımcı Lasso modelleri** (λ yazılım ölçeğinde)")
            st.dataframe(folds_table(fit), hide_index=True, width="stretch")
    elif isinstance(op, DMLSplits):
        fit = state.models[op.name]
        source = state.ops.get(op.dml)
        clustered = source is None or bool(getattr(source, "cluster", None))
        unit, se_label = ("kümelerin", "Küme SH") if clustered else ("gözlemlerin", "HC1 SH")
        d = op.decimals
        st.markdown(f"**Bölme duyarlılığı:** aynı veri, öğrenici ve kat sayısı; yalnız {unit} katlara dağılımı değişir")
        table = splits_table(state.tables[op.result], d, se_label)
        st.dataframe(table, hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3)
        st.caption(
            f"Medyan birleştirme (Chernozhukov vd., 2018): θ̂_med = {number(fit.theta, d)}, SH_med = {number(fit.se, d)}; "
            f"aralık {number(state.scalars[f'{op.name}_min'], d)} – {number(state.scalars[f'{op.name}_max'], d)}."
        )
    elif isinstance(op, DoubleSelection):
        scalars = state.scalars
        st.markdown(
            f"**Seçilen kontroller:** S_Y {count(scalars[f'{op.name}_ny'])} · S_D {count(scalars[f'{op.name}_nd'])} · "
            f"birleşim {count(scalars[f'{op.name}_n'])}"
        )
    elif isinstance(op, ComplexityCurve):
        st.dataframe(complexity_table(state.tables[op.result]), hide_index=True, width="stretch")
        st.caption("◀: ölçütü en küçük yapan derece.")
    else:
        return False
    return True
