"""Konu 01-02 grafik ve tablo sunumu için ortak yardımcılar."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from core.ui_preferences import plotly_font_size


CHART_COLORS = ("#107C89", "#2F9E6B", "#B3392F", "#51696C")


def style_figure(
    figure: go.Figure,
    *,
    title: str,
    x_title: str,
    y_title: str,
    legend_title: str | None = None,
) -> go.Figure:
    scale = float(st.session_state.get("text_scale", 1.0))
    figure.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        colorway=CHART_COLORS,
        font={"size": plotly_font_size(13, scale), "color": "#07373D"},
        title_font={"size": plotly_font_size(16, scale)},
        legend_title_text=legend_title,
        margin={"l": 20, "r": 20, "t": 64, "b": 20},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hovermode="closest",
    )
    figure.update_xaxes(showgrid=True, gridcolor="#E2EBEB", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor="#E2EBEB", zeroline=False)
    return figure


def show_figure(figure: go.Figure) -> None:
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )


def render_model_context(
    *,
    data_label: str,
    sample_label: str,
    model_label: str,
    inference_label: str,
    seed: int | None = None,
) -> None:
    seed_text = f" | Rastgelelik tohumu: {seed}" if seed is not None else ""
    st.caption(
        f"Veri: {data_label} | Örneklem: {sample_label} | "
        f"Model: {model_label} | Çıkarım: {inference_label}{seed_text}"
    )
