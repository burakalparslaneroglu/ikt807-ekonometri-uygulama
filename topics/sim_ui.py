"""Sezgi sekmesi: bilinen bir veri üretim süreciyle kontrollü deneyler.

Her deney aynı sırayı izler: soru → DGP ve parametreleri → neye bakıyoruz →
sonuç → ne gördük → kod. Kod, uygulamanın hesabıyla aynı tanımdan üretilir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.codegen.base import LANGUAGE_INFO, LANGUAGES, generator, histogram_styles, layer_styles
from core.labs.runner import LabState, PlotLayerData, execute
from core.labs.sezgi import SimExperiment
from core.labs.spec import (
    REPRO_DESCRIPTIONS,
    BandwidthCV,
    BinMeans,
    Curve,
    Histogram,
    LocalCurve,
    MeanPoints,
    ModelLine,
    Plot,
    RDDCurve,
    ReproClass,
    Scatter,
    ScalarTable,
    VLine,
)
from core.labs.sezgi import plain
from topics.lab_ui import CODE_LANGUAGE_KEY, render_bandwidth_cv
from topics.regression_ui import show_figure, style_figure


@st.cache_resource(show_spinner=False, max_entries=48)
def _run(key: str, parameters: tuple[tuple[str, float], ...], _experiment: SimExperiment) -> LabState:
    state = LabState()
    for op in _experiment.build(dict(parameters)):
        execute(op, state, {})
    return state


def _parameter_key(experiment: SimExperiment, name: str) -> str:
    return f"{experiment.key}_{name}"


def _current_parameters(experiment: SimExperiment) -> dict[str, float]:
    values = {}
    for item in experiment.parameters:
        key = _parameter_key(experiment, item.key)
        if key not in st.session_state:
            st.session_state[key] = int(item.default) if item.integer else float(item.default)
        values[item.key] = st.session_state[key]
    return values


def _render_sliders(experiment: SimExperiment) -> None:
    columns = st.columns(len(experiment.parameters))
    for column, item in zip(columns, experiment.parameters):
        if item.integer:
            column.slider(
                item.label, min_value=int(item.minimum), max_value=int(item.maximum),
                step=int(item.step), key=_parameter_key(experiment, item.key), help=item.help,
            )
        else:
            column.slider(
                item.label, min_value=float(item.minimum), max_value=float(item.maximum),
                step=float(item.step), key=_parameter_key(experiment, item.key), help=item.help,
                format=f"%.{item.decimals}f",
            )


def _rgba(rgb: str, alpha: float) -> str:
    red, green, blue = rgb.split()
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _rdd_band(figure: go.Figure, op: Plot, layer: RDDCurve, data: pd.DataFrame, style) -> None:
    """Eşiğin iki yanında ayrı eğri ve noktasal %95 bant; iki taraf tek legend öğesi."""

    for position, (_, part) in enumerate(data.groupby("taraf", sort=False)):
        part = part.dropna(subset=["tahmin"])
        x = part[op.x].to_numpy(dtype=float)
        low, high = part["alt"].to_numpy(dtype=float), part["ust"].to_numpy(dtype=float)
        figure.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]), y=np.concatenate([high, low[::-1]]), fill="toself",
                fillcolor=_rgba(style.rgb, 0.18), line={"width": 0}, hoverinfo="skip", showlegend=False,
                legendgroup=layer.label,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x, y=part["tahmin"], mode="lines", name=layer.label, legendgroup=layer.label,
                showlegend=position == 0, line={"color": style.color, "width": 3},
                customdata=np.column_stack([low, high]),
                hovertemplate="x = %{x:.2f}<br>tahmin = %{y:.3f}<br>%95 bant [%{customdata[0]:.3f}; "
                              "%{customdata[1]:.3f}]<extra></extra>",
            )
        )


def layered_figure(op: Plot, layers: list[PlotLayerData]) -> go.Figure:
    """Katmanlı grafik (Sezgi deneyleri ve Uygulama adımları); üretilen kodla aynı katmanlar ve renkler."""

    figure = go.Figure()
    vertical = False
    for item, style in zip(layers, layer_styles(op.layers)):
        layer, data = item.layer, item.data
        if isinstance(layer, RDDCurve):
            _rdd_band(figure, op, layer, data, style)
        elif isinstance(layer, VLine):
            vertical = True
            figure.add_trace(
                go.Scatter(
                    x=[layer.x, layer.x], y=[0, 1], mode="lines", name=layer.label, yaxis="y2",
                    line={"color": style.color, "width": 2, "dash": "dashdot"}, hoverinfo="skip",
                )
            )
        elif isinstance(layer, (MeanPoints, BinMeans)):
            hover = "x ortalaması = %{x:.2f}" if isinstance(layer, BinMeans) else "x = %{x}"
            figure.add_trace(
                go.Scatter(
                    x=data[op.x], y=data["ortalama"], mode="markers", name=layer.label,
                    marker={"size": np.sqrt(data["n"]) * 1.1 + 4, "color": style.color, "opacity": 0.9},
                    customdata=data[["n"]],
                    hovertemplate=hover + "<br>ortalama = %{y:.4f}<br>n = %{customdata[0]}<extra></extra>",
                )
            )
        elif isinstance(layer, Scatter):
            figure.add_trace(
                go.Scattergl(
                    x=data[op.x], y=data["deger"], mode="markers", name=layer.label,
                    marker={"size": 4, "color": style.color, "opacity": 0.35},
                    hoverinfo="skip",
                )
            )
        else:
            dash = "dash" if style.dashed else ("dot" if style.dotted else "solid")
            width = 3 if isinstance(layer, (Curve, ModelLine, LocalCurve)) else 1.5
            figure.add_trace(
                go.Scatter(
                    x=data[op.x], y=data["deger"], mode="lines", name=layer.label,
                    line={"color": style.color, "width": width, "dash": dash},
                    hovertemplate="x = %{x:.2f}<br>%{y:.4f}<extra>" + layer.label + "</extra>",
                )
            )
    if vertical:
        figure.update_layout(yaxis2={"overlaying": "y", "range": [0, 1], "visible": False})
    if op.x_range is not None:
        figure.update_xaxes(range=list(op.x_range))
    style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label, legend_title="")
    return figure


_REFERENCE_STYLES = (("#07373D", "dash"), ("#6B4C9A", "dot"))


def histogram_figure(op: Histogram, data: pd.DataFrame) -> tuple[go.Figure, str]:
    return _histogram(op, data)


def _histogram(op: Histogram, data: pd.DataFrame) -> tuple[go.Figure, str]:
    """Üretilen kodla aynı kutular: [lower, upper] aralığında ``bins`` eşit genişlikte kutu."""

    edges = np.linspace(op.lower, op.upper, op.bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    figure = go.Figure()
    outside: list[str] = []
    for (column, label), style in zip(op.columns, histogram_styles(len(op.columns))):
        values = data[column].to_numpy(dtype=float)
        counts, _ = np.histogram(values, bins=edges)
        figure.add_trace(
            go.Bar(
                x=centers, y=counts, width=np.diff(edges), name=label,
                marker={"color": style.color, "opacity": 0.55, "line": {"width": 0}},
                hovertemplate=f"{label}<br>%{{x:.3f}} civarı: %{{y}} tekrar<extra></extra>",
            )
        )
        missing = int(((values < op.lower) | (values > op.upper)).sum())
        outside.append(f"{label}: {missing}")
    for index, (value, label) in enumerate(op.references):
        color, dash = _REFERENCE_STYLES[index % len(_REFERENCE_STYLES)]
        figure.add_trace(
            go.Scatter(
                x=[value, value], y=[0, 1], mode="lines", name=label, yaxis="y2",
                line={"color": color, "width": 2.5, "dash": dash}, hoverinfo="skip",
            )
        )
    figure.update_layout(
        barmode="overlay",
        yaxis2={"overlaying": "y", "range": [0, 1], "visible": False},
    )
    style_figure(figure, title=op.title, x_title=op.x_label, y_title="Tekrar sayısı", legend_title="")
    def plain(value: float) -> str:
        return f"{value:g}".replace(".", ",").replace("-", "−")

    caption = (
        f"Kutular: [{plain(op.lower)}; {plain(op.upper)}] aralığında {op.bins} eşit genişlik. "
        "Aralık dışında kalan tekrar sayısı — " + ", ".join(outside) + "."
    )
    return figure, caption


def _table(experiment: SimExperiment, name: str, table: pd.DataFrame) -> pd.DataFrame:
    if "Değer" in table.columns:
        shown = table.copy()
        shown["Değer"] = shown["Değer"].round(10) + 0.0
        shown.index.name = "Nicelik"
        return shown.reset_index()
    shown = table.reset_index()
    shown = shown.rename(columns={column: experiment.label(column) for column in shown.columns})
    return shown


def _format(table: pd.DataFrame) -> dict[str, str]:
    formats: dict[str, str] = {}
    for column in table.columns:
        if column == "Değer":
            formats[column] = "{:.10f}"
        elif table[column].dtype.kind == "f":
            integral = np.allclose(table[column], np.round(table[column]))
            formats[column] = "{:.0f}" if integral else "{:.4f}"
    return formats


def _render_code(experiment: SimExperiment, parameters: dict[str, float]) -> None:
    language = st.session_state.get(CODE_LANGUAGE_KEY, LANGUAGES[0])
    info = LANGUAGE_INFO[language]
    code = generator(experiment.spec(parameters), language).script()
    title, description = REPRO_DESCRIPTIONS[ReproClass.DISTRIBUTIONAL]
    st.markdown(f"**{language} kodu** (şu anki kaydırıcı değerleriyle)")
    st.caption(
        f"Üç dilde sonuç: **{title}**. {description} Python sürümü bu sayfadaki sayıların aynısını verir."
    )
    with st.expander("Kodu göster", icon=":material/code:"):
        st.code(code, language=info.highlight, line_numbers=True)
    st.download_button(
        f"{language} kodunu indir (.{info.extension})",
        data=code,
        file_name=f"ikt807_{experiment.key}.{info.extension}",
        mime=info.mime,
        key=f"{experiment.key}_download_{language}",
        icon=":material/download:",
    )


def render_experiments(experiments: tuple[SimExperiment, ...]) -> None:
    topic_key = experiments[0].topic_key
    st.markdown(
        "Bu sekmedeki veriler **simülasyondur**: veri üretim süreci (DGP) bilinir. Böylece gerçek "
        "veride hiçbir zaman göremediğimiz nesneleri — örneğin gerçek koşullu ortalamayı — "
        "tahminlerle yan yana görebiliriz. Gerçek veri uygulaması **Uygulama** sekmesindedir."
    )
    selector = f"{topic_key}_sezgi_deney"
    numbers = [item.number for item in experiments]
    if st.session_state.get(selector) not in numbers:
        st.session_state[selector] = numbers[0]
    titles = {item.number: item.title for item in experiments}
    st.segmented_control(
        "Deney", options=numbers, format_func=lambda n: f"Deney {n}: {titles[n]}",
        key=selector, label_visibility="collapsed",
    )
    experiment = next(item for item in experiments if item.number == st.session_state[selector])

    st.subheader(f"Deney {experiment.number}: {experiment.title}")
    st.caption(experiment.note.label())
    st.markdown(f"**Soru.** {experiment.question}")

    parameters = _current_parameters(experiment)
    with st.container(border=True):
        st.markdown("**Veri üretim süreci (DGP)**")
        for line in experiment.dgp(parameters):
            st.latex(line)
        st.caption(experiment.dgp_note)
        _render_sliders(experiment)

    state = _run(experiment.key, tuple(sorted(parameters.items())), experiment)

    st.markdown("**Neye bakıyoruz?**")
    st.markdown("\n".join(f"- {line}" for line in experiment.look_at))
    for op in experiment.build(parameters):
        if isinstance(op, Plot):
            show_figure(layered_figure(op, state.plots[f"grafik:{op.title}"]))
        elif isinstance(op, Histogram):
            figure, caption = _histogram(op, state.plots[f"histogram:{op.title}"])
            show_figure(figure)
            st.caption(caption)
        elif isinstance(op, BandwidthCV):
            render_bandwidth_cv(op, state, metrics=False)

    metrics = experiment.metrics(state, parameters)
    columns = st.columns(len(metrics))
    for column, metric in zip(columns, metrics):
        column.metric(metric.label, metric.value, help=metric.help)

    summaries = {op.result: op.decimals for op in experiment.build(parameters) if isinstance(op, ScalarTable)}
    for name, title in experiment.tables:
        st.markdown(f"**{title}**")
        if name in summaries:
            table = state.tables[name]
            shown = pd.DataFrame({"Büyüklük": table.index,
                                  "Değer": [plain(value, summaries[name]) for value in table["deger"]]})
            st.dataframe(shown, hide_index=True, width="stretch", height=35 * (len(shown) + 1) + 3)
            continue
        table = _table(experiment, name, state.tables[name])
        st.dataframe(table.style.format(_format(table)), hide_index=True, width="stretch")

    st.info(experiment.takeaway(state, parameters), icon=":material/lightbulb:")
    _render_code(experiment, parameters)
