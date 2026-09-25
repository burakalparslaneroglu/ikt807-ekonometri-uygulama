"""Sezgi sekmesi: bilinen bir veri üretim süreciyle kontrollü deneyler.

Her deney aynı sırayı izler: soru → DGP ve parametreleri → neye bakıyoruz →
sonuç → ne gördük → kod. Kod, uygulamanın hesabıyla aynı tanımdan üretilir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.codegen.base import LANGUAGE_INFO, LANGUAGES, generator, layer_styles
from core.labs.runner import LabState, PlotLayerData, execute
from core.labs.sezgi import SimExperiment
from core.labs.spec import REPRO_DESCRIPTIONS, Curve, MeanPoints, ModelLine, Plot, ReproClass, Scatter
from topics.lab_ui import CODE_LANGUAGE_KEY
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


def _figure(op: Plot, layers: list[PlotLayerData]) -> go.Figure:
    figure = go.Figure()
    for item, style in zip(layers, layer_styles(op.layers)):
        layer, data = item.layer, item.data
        if isinstance(layer, MeanPoints):
            figure.add_trace(
                go.Scatter(
                    x=data[op.x], y=data["ortalama"], mode="markers", name=layer.label,
                    marker={"size": np.sqrt(data["n"]) * 1.1 + 4, "color": style.color, "opacity": 0.9},
                    customdata=data[["n"]],
                    hovertemplate="x = %{x}<br>ortalama = %{y:.4f}<br>n = %{customdata[0]}<extra></extra>",
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
            width = 3 if isinstance(layer, (Curve, ModelLine)) else 1.5
            figure.add_trace(
                go.Scatter(
                    x=data[op.x], y=data["deger"], mode="lines", name=layer.label,
                    line={"color": style.color, "width": width, "dash": dash},
                    hovertemplate="x = %{x:.2f}<br>%{y:.4f}<extra>" + layer.label + "</extra>",
                )
            )
    style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label, legend_title="")
    return figure


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
            show_figure(_figure(op, state.plots[f"grafik:{op.title}"]))

    metrics = experiment.metrics(state, parameters)
    columns = st.columns(len(metrics))
    for column, metric in zip(columns, metrics):
        column.metric(metric.label, metric.value, help=metric.help)

    for name, title in experiment.tables:
        st.markdown(f"**{title}**")
        table = _table(experiment, name, state.tables[name])
        st.dataframe(table.style.format(_format(table)), hide_index=True, width="stretch")

    st.info(experiment.takeaway(state, parameters), icon=":material/lightbulb:")
    _render_code(experiment, parameters)
