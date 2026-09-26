"""Ders notu uygulama laboratuvarlarının ortak "Uygulama" sekmesi.

Her laboratuvar ``core.labs`` altındaki tek bir tanımdan beslenir: adım metni,
uygulamanın hesabı, üç dildeki kod ve notlarla karşılaştırma aynı kaynaktan gelir.
"""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.codegen.base import (
    HANSEN_PAGE_URL,
    LANGUAGE_INFO,
    LANGUAGES,
    render_script,
    render_step,
    script_filename,
)
from core.hansen_data import (
    HansenDataError,
    LoadedData,
    download_archive,
    extract_member,
    load_from_upload,
    DATASET_MEMBERS,
)
from core.labs.runner import LabRun, run_lab, statsmodels_term
from core.labs.spec import (
    IV,
    OLS,
    BreuschPagan,
    DeltaMethod,
    DropMissing,
    EffectTable,
    LinearCombination,
    StandardErrorTable,
    REPRO_DESCRIPTIONS,
    Describe,
    GroupMeanPlot,
    GroupSummary,
    LabSpec,
    LabStep,
    ProjectionPlot,
    RegressionTable,
    Scalar,
    ShowModel,
)
from topics.regression_ui import show_figure, style_figure

CODE_LANGUAGE_KEY = "code_language"
DATA_PATH_ENV = "IKT807_HANSEN_{dataset}_PATH"
LEGACY_CPS_ENV = "IKT807_CPS_PATH"
_STAT_LABELS = {
    "count": "N",
    "mean": "Ortalama",
    "sd": "Std. sapma",
    "median": "Medyan",
    "min": "En küçük",
    "max": "En büyük",
}
_COLORS = ("#107C89", "#B3392F", "#2F9E6B", "#07373D")


# --- Veri ------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _hansen_archive() -> bytes:
    """Arşivi bir kez indirir; sunucudaki bütün oturumlar ve konular aynı kopyayı kullanır."""

    return download_archive()


@st.cache_data(show_spinner=False, max_entries=8)
def _hansen_member(dataset: str) -> bytes:
    return extract_member(_hansen_archive(), DATASET_MEMBERS[dataset])


def _data_key(topic_key: str) -> str:
    return f"{topic_key}_lab_data"


def _local_path(dataset: str) -> Path | None:
    value = os.environ.get(DATA_PATH_ENV.format(dataset=dataset.upper()))
    if not value and dataset == "cps09mar":
        value = os.environ.get(LEGACY_CPS_ENV)
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_file() else None


def _render_data_panel(spec: LabSpec) -> LoadedData | None:
    key = _data_key(spec.topic_key)
    member = DATASET_MEMBERS[spec.dataset]

    if key not in st.session_state:
        path = _local_path(spec.dataset)
        if path is not None:
            try:
                st.session_state[key] = load_from_upload(spec.dataset, path.read_bytes(), path.name)
            except HansenDataError as error:
                st.error(str(error))

    with st.container(border=True):
        loaded: LoadedData | None = st.session_state.get(key)
        if loaded is not None:
            left, right = st.columns([4, 1])
            status = "Hansen'in tam örneklemi" if loaded.matches_hansen else "Farklı örneklem"
            left.markdown(
                f"**Veri:** `{member}` · {loaded.source} · "
                f"{len(loaded.frame):,} gözlem · {status}".replace(",", ".")
            )
            if not loaded.matches_hansen:
                left.warning(
                    "Yüklenen dosya Hansen'in tam örnekleminden farklı. Sonuçlar notlardaki "
                    "sayılarla uyuşmayabilir."
                )
            if right.button("Veriyi değiştir", key=f"{spec.topic_key}_lab_reset", width="stretch"):
                del st.session_state[key]
                st.rerun()
            return loaded

        st.markdown(f"**Veri:** Hansen'in `{member}` dosyası henüz yüklenmedi.")
        st.caption(
            f"Veri depoda tutulmaz. Hansen'in sayfasından ({HANSEN_PAGE_URL}) çalışma anında "
            "indirilir veya kendi kopyanızı yükleyebilirsiniz."
        )
        left, right = st.columns(2)
        if left.button(
            "Hansen'in sayfasından indir",
            key=f"{spec.topic_key}_lab_download",
            type="primary",
            icon=":material/cloud_download:",
            width="stretch",
        ):
            with st.spinner("Hansen veri arşivi indiriliyor…"):
                try:
                    data = _hansen_member(spec.dataset)
                    loaded = load_from_upload(spec.dataset, data, member)
                    st.session_state[key] = LoadedData(
                        loaded.frame, "Hansen veri arşivi", loaded.matches_hansen
                    )
                    st.rerun()
                except (HansenDataError, OSError, zipfile.BadZipFile) as error:
                    st.error(
                        f"İndirme başarısız: {error}\n\nDosyayı kendiniz indirip yandan yükleyebilirsiniz."
                    )
        uploaded = right.file_uploader(
            f"veya dosya yükleyin ({member} ya da ders notlarının öğretim CSV'si)",
            type=("txt", "csv", "dta"),
            key=f"{spec.topic_key}_lab_upload",
        )
        if uploaded is not None:
            try:
                st.session_state[key] = load_from_upload(spec.dataset, uploaded.getvalue(), uploaded.name)
                st.rerun()
            except (HansenDataError, ValueError) as error:
                st.error(str(error))
    return None


@st.cache_resource(show_spinner=False, max_entries=8)
def _run(topic_key: str, fingerprint: int, _spec: LabSpec, _frame: pd.DataFrame) -> LabRun:
    return run_lab(_spec, {_spec.dataset: _frame})


def _lab_run(spec: LabSpec, loaded: LoadedData) -> LabRun | None:
    fingerprint = int(pd.util.hash_pandas_object(loaded.frame, index=False).sum())
    try:
        return _run(spec.topic_key, fingerprint, spec, loaded.frame)
    except Exception as error:  # veri yapısı sorunları öğrenciye açıkça gösterilir
        st.error(f"Laboratuvar bu veriyle çalıştırılamadı: {error}")
        return None


# --- Adım gezinimi ---------------------------------------------------------

def _step_key(spec: LabSpec) -> str:
    return f"{spec.topic_key}_lab_step"


def _shift(spec: LabSpec, delta: int) -> None:
    key = _step_key(spec)
    numbers = [step.number for step in spec.steps]
    current = st.session_state.get(key) or numbers[0]
    index = min(max(numbers.index(current) + delta, 0), len(numbers) - 1)
    st.session_state[key] = numbers[index]


def _render_navigation(spec: LabSpec) -> LabStep:
    key = _step_key(spec)
    numbers = [step.number for step in spec.steps]
    if st.session_state.get(key) not in numbers:
        st.session_state[key] = numbers[0]
    left, middle, right = st.columns([1, 6, 1], vertical_alignment="bottom")
    left.button("‹ Önceki", key=f"{spec.topic_key}_lab_prev", on_click=_shift, args=(spec, -1), width="stretch")
    middle.segmented_control(
        "Adım",
        options=numbers,
        format_func=lambda number: f"Adım {number}",
        key=key,
        label_visibility="collapsed",
        width="stretch",
    )
    right.button("Sonraki ›", key=f"{spec.topic_key}_lab_next", on_click=_shift, args=(spec, 1), width="stretch")
    return spec.step(st.session_state.get(key) or numbers[0])


# --- Sonuç gösterimi -------------------------------------------------------

def _describe_table(spec: LabSpec, table: pd.DataFrame) -> pd.DataFrame:
    shown = table.rename(columns=_STAT_LABELS)
    shown.index = [spec.label(name) for name in shown.index]
    shown.index.name = "Değişken"
    return shown


def _group_table(spec: LabSpec, op: GroupSummary, table: pd.DataFrame) -> pd.DataFrame:
    shown = table.rename(columns={name: spec.label(name) for name, _, _ in op.columns}).reset_index()
    shown = shown.rename(columns={op.by: spec.label(op.by)})
    count_columns = [spec.label(name) for name, _, stat in op.columns if stat == "count"]
    for column in count_columns:
        shown[column] = shown[column].astype(int)
    return shown


def _term_label(spec: LabSpec, term: str) -> str:
    if term == "Intercept":
        return spec.label("(sabit)")
    match = re.fullmatch(r"C\((\w+)\)\[T\.([^\]]+)\]", term)
    if match:
        level = match.group(2)
        level = level[:-2] if level.endswith(".0") else level
        return f"{spec.label(match.group(1))} = {level}"
    return spec.label(term)


def _ci_text(estimate: float, se: float, decimals: int = 4) -> str:
    low, high = estimate - 1.96 * se, estimate + 1.96 * se
    return f"SH {se:.{decimals}f} · %95 GA [{low:.{decimals}f}; {high:.{decimals}f}]".replace(".", ",")


def _p_text(value: float) -> str:
    if value >= 1e-4:
        return f"{value:.4f}".replace(".", ",")
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10**exponent
    return f"{mantissa:.1f}×10^{exponent}".replace(".", ",")


def _model_output(spec: LabSpec, result) -> pd.DataFrame:
    interval = result.conf_int()
    rows = []
    for term in result.params.index:
        label = _term_label(spec, term)
        rows.append(
            {
                "Terim": label,
                "Katsayı": result.params[term],
                "Std. hata": result.bse[term],
                "t": result.tvalues[term],
                "p-değeri": result.pvalues[term],
                "%95 GA alt": interval.loc[term, 0],
                "%95 GA üst": interval.loc[term, 1],
            }
        )
    return pd.DataFrame(rows)


def _number(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}".replace(".", ",").replace("-", "−")


def _count(value: float) -> str:
    return f"{int(value):,}".replace(",", ".")


def _effect_table(spec: LabSpec, op: EffectTable, table: pd.DataFrame) -> None:
    if op.title:
        st.markdown(f"**{op.title}**")
    shown = pd.DataFrame(
        {
            "": table.index,
            "Tahmin": [_number(v) for v in table["tahmin"]],
            op.se_label: [_number(v) for v in table["sh"]],
            "p-değeri": [_number(v, 3) for v in table["p"]],
            "N": [_count(v) for v in table["n"]],
        }
    )
    st.dataframe(shown, hide_index=True, width="stretch")


def _compact_iv(spec: LabSpec, op: IV, result) -> None:
    endogenous = ", ".join(spec.label(name) for name in op.endogenous)
    instruments = ", ".join(spec.label(name) for name in op.instruments)
    controls = f" + {len(op.exogenous)} dışsal kontrol" if op.exogenous else ""
    term = op.endogenous[0]
    st.markdown(
        f"**2SLS:** {spec.label(op.outcome)} ~ [{endogenous} ← {instruments}]{controls} · "
        f"{spec.label(term)} katsayısı {_number(float(result.params[term]))} "
        f"(HC1 SH {_number(float(result.bse[term]))}) · N = {_count(result.nobs)}"
    )


def _compact_model(spec: LabSpec, op: OLS, result) -> None:
    if op.categorical or len(op.regressors) > 3:
        shown = ", ".join(spec.label(r) for r in op.regressors)
        nobs = f"{int(result.nobs):,}".replace(",", ".")
        st.markdown(
            f"**{spec.label(op.name)}:** {spec.label(op.outcome)} ~ {shown} · {len(result.params)} katsayı · "
            f"N = {nobs} · R² = {result.rsquared:.4f}"
        )
        return
    parts = [f"{result.params['Intercept']:.4f}"]
    for name in op.regressors:
        value = float(result.params[name])
        sign = "−" if value < 0 else "+"
        parts.append(f"{sign} {abs(value):.4f}·{spec.label(name).lower()}")
    st.markdown(f"**Tahmin edilen denklem:** {spec.label(op.outcome)} = " + " ".join(parts))
    st.caption(f"N = {int(result.nobs):,} · R² = {result.rsquared:.4f}".replace(",", "."))


def _group_plot(spec: LabSpec, op: GroupMeanPlot, data: pd.DataFrame) -> None:
    figure = go.Figure()
    for index, (value, label) in enumerate(sorted(op.group_labels)):
        subset = data[data[op.group] == value]
        figure.add_trace(
            go.Scatter(
                x=subset[op.x],
                y=subset["ortalama"],
                mode="lines+markers",
                name=label,
                line={"color": _COLORS[index % len(_COLORS)]},
                hovertemplate=f"{op.x_label}: %{{x}}<br>{op.y_label}: %{{y:.4f}}<extra>{label}</extra>",
            )
        )
    style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label, legend_title="Grup")
    show_figure(figure)


def _projection_plot(op: ProjectionPlot, data: pd.DataFrame, result) -> None:
    grid = np.linspace(data[op.x].min(), data[op.x].max(), 100)
    line = result.params["Intercept"] + result.params[op.x] * grid
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=data[op.x],
            y=data["ortalama"],
            mode="markers",
            name="Koşullu ortalama",
            marker={"size": np.sqrt(data["n"]) / 2.5, "color": _COLORS[0], "opacity": 0.75},
            customdata=data[["n"]],
            hovertemplate=f"{op.x_label}: %{{x}}<br>Ortalama: %{{y:.4f}}<br>n: %{{customdata[0]}}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(x=grid, y=line, mode="lines", name="OLS doğrusal projeksiyonu", line={"color": _COLORS[3], "width": 3})
    )
    style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label, legend_title="Seri")
    show_figure(figure)


def _render_results(spec: LabSpec, step: LabStep, run: LabRun) -> None:
    state = run.state
    has_table = any(isinstance(op, (RegressionTable, EffectTable)) for op in step.operations)
    for op in step.operations:
        if isinstance(op, DropMissing):
            before, after = state.samples[op.frame]
            st.markdown(
                f"**Analiz örneklemi:** N = {_count(after)} "
                f"(yüklenen veride {_count(before)} gözlem; eksik değeri olan {_count(before - after)} gözlem çıkarıldı)"
            )
        elif isinstance(op, EffectTable):
            _effect_table(spec, op, state.tables[op.result])
        elif isinstance(op, IV) and not has_table:
            _compact_iv(spec, op, state.models[op.name])
        elif isinstance(op, Describe):
            st.markdown("**Betimsel istatistikler**")
            st.dataframe(_describe_table(spec, state.tables[op.result]).style.format("{:.4f}"), width="stretch")
        elif isinstance(op, GroupSummary):
            st.markdown(f"**{spec.label(op.by)} düzeyine göre koşullu ortalamalar**")
            table = _group_table(spec, op, state.tables[op.result])
            numeric = [c for c in table.columns if table[c].dtype.kind == "f"]
            st.dataframe(table.style.format({c: "{:.4f}" for c in numeric}), hide_index=True, width="stretch")
        elif isinstance(op, GroupMeanPlot):
            _group_plot(spec, op, state.plots[f"grup:{op.y}:{op.group}"])
        elif isinstance(op, ProjectionPlot):
            _projection_plot(op, state.plots[f"projeksiyon:{op.model}"], state.models[op.model])
        elif isinstance(op, OLS) and not has_table:
            _compact_model(spec, op, state.models[op.name])
        elif isinstance(op, RegressionTable):
            st.markdown("**Log saatlik ücret için OLS regresyonları** (parantez içinde klasik standart hatalar)")
            table = state.tables[op.result].reset_index()
            table["Terim"] = [spec.label(name) if name else "" for name in table["Terim"]]
            st.dataframe(table, hide_index=True, width="stretch")
        elif isinstance(op, ShowModel):
            result = state.models[op.model]
            st.markdown("**Model (3) — yazılım çıktısı**")
            st.dataframe(
                _model_output(spec, result).style.format(
                    {c: "{:.4f}" for c in ("Katsayı", "Std. hata", "%95 GA alt", "%95 GA üst")}
                    | {"t": "{:.2f}", "p-değeri": "{:.3f}"}
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(f"N = {int(result.nobs):,} · R² = {result.rsquared:.4f}".replace(",", "."))
    for op in step.operations:
        if isinstance(op, StandardErrorTable):
            table = state.tables[op.result].reset_index()
            table["model"] = [spec.label(name) for name in table["model"]]
            table.columns = ["Model", f"{spec.label(op.term)} katsayısı", "Klasik SH", "HC1 SH", "R²"]
            st.markdown("**Aynı katsayı, iki belirsizlik ölçüsü**")
            st.dataframe(table.style.format({c: "{:.4f}" for c in table.columns[1:]}), hide_index=True, width="stretch")
        elif isinstance(op, BreuschPagan):
            left, middle, right = st.columns(3)
            left.metric("Breusch–Pagan LM", f"{state.scalars[f'{op.name}_lm']:.2f}".replace(".", ","))
            middle.metric("Serbestlik derecesi", f"{state.scalars[f'{op.name}_df']:.0f}")
            right.metric("p-değeri", _p_text(state.scalars[f"{op.name}_p"]))
        elif isinstance(op, LinearCombination):
            value, se = state.scalars[op.name], state.scalars[f"{op.name}_se"]
            st.metric(op.comment, f"{value:.4f}".replace(".", ","))
            st.caption(_ci_text(value, se) + " (HC1)")
        elif isinstance(op, DeltaMethod):
            value, se = state.scalars[op.name], state.scalars[f"{op.name}_se"]
            st.metric(op.comment + " (%)", f"%{value:.2f}".replace(".", ","))
            st.caption(_ci_text(value, se, 2) + " · delta yöntemi")
    scalars = [op for op in step.operations if isinstance(op, Scalar)]
    if scalars:
        columns = st.columns(len(scalars))
        for column, op in zip(columns, scalars):
            value = _number(state.scalars[op.name], op.decimals)
            column.metric(op.comment, f"%{value}" if op.percent else value)


def _render_checks(step: LabStep, run: LabRun) -> None:
    results = run.step_checks(step.number)
    if not results:
        return
    passed = sum(item.passed for item in results)
    label = f"Notlarla karşılaştırma: {passed}/{len(results)} değer aynı"
    with st.expander(label, icon=":material/fact_check:" if passed == len(results) else ":material/error:"):
        rows = [
            {
                "Değer": item.check.label,
                "Uygulama": f"{item.value:.{item.check.decimals}f}",
                "Notlar": f"{item.check.expected:.{item.check.decimals}f}",
                "Durum": "✓" if item.passed else "✗",
            }
            for item in results
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_code(spec: LabSpec, step: LabStep) -> None:
    if not step.operations:
        return
    language = st.session_state.get(CODE_LANGUAGE_KEY, LANGUAGES[0])
    info = LANGUAGE_INFO[language]
    title, description = REPRO_DESCRIPTIONS[step.reproducibility]
    st.markdown(f"**{language} kodu**")
    st.caption(f"Üç dilde sonuç: **{title}**. {description}")
    st.code(render_step(spec, step.number, language), language=info.highlight, line_numbers=True)
    if step.code_note:
        st.caption(step.code_note)


def _render_downloads(spec: LabSpec) -> None:
    st.markdown("**Bütün laboratuvarı indirin**")
    st.caption(
        "Her dosya veriyi indirir, bütün adımları çalıştırır ve sonunda sonuçları ders notlarındaki "
        "sayılarla karşılaştırır. Bir sayı tutmazsa hangisinin tutmadığını söyleyerek durur."
    )
    columns = st.columns(len(LANGUAGES))
    for column, language in zip(columns, LANGUAGES):
        info = LANGUAGE_INFO[language]
        column.download_button(
            f"{language} (.{info.extension})",
            data=render_script(spec, language),
            file_name=script_filename(spec, language),
            mime=info.mime,
            key=f"{spec.topic_key}_lab_download_{language}",
            icon=":material/download:",
            width="stretch",
        )


# --- Ana giriş ---------------------------------------------------------------

def render_lab(spec: LabSpec) -> None:
    st.markdown(
        f"Bu sekme ders notlarındaki **§{spec.note_section} {spec.title}** laboratuvarını adım adım "
        "yeniden üretir. Tablolar notlardakiyle aynı sayıları verir; kod dilini kenar çubuğundan seçin."
    )
    loaded = _render_data_panel(spec)
    run = _lab_run(spec, loaded) if loaded is not None else None

    step = _render_navigation(spec)
    title, _ = REPRO_DESCRIPTIONS[step.reproducibility]
    st.subheader(f"Adım {step.number}: {step.title}")
    st.caption(step.note.label())
    st.markdown(step.explanation)

    if step.operations:
        if run is not None:
            _render_results(spec, step, run)
            _render_checks(step, run)
        else:
            st.info("Sonuçları görmek için yukarıdan veriyi yükleyin. Kod aşağıda her durumda görünür.")
    if step.takeaway:
        st.info(step.takeaway, icon=":material/lightbulb:")
    _render_code(spec, step)
    if step.number == spec.steps[-1].number:
        _render_downloads(spec)
