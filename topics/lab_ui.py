"""Ders notu uygulama laboratuvarlarının ortak "Uygulama" sekmesi.

Her laboratuvar ``core.labs`` altındaki tek bir tanımdan beslenir: adım metni,
uygulamanın hesabı, üç dildeki kod ve notlarla karşılaştırma aynı kaynaktan gelir.

Ek kaynakları olan konularda (``core.labs.ornekler``) sekmenin en üstünde veri kaynağı seçilir: notlardaki örnek
(varsayılan), alternatif örnek (aynı adımlar, Hansen arşivindeki başka bir gerçek veriyle) ya da öğrencinin kendi verisi
("Kendi verini yükle", ``topics.kendi_veri_ui``). Üç kaynak aynı adımları ve aynı kod üreticisini kullanır; notlar
dışındaki kaynaklarda kontroller ekranda gösterilmez, indirilen kod uygulamanın sayılarıyla karşılaştırır. Kendi
verinde kod Python ve R'da üretilir; öğrencinin verisi ve ondan kurulan hesap ortak önbelleğe girmez.
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
    languages_for,
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
from core.labs.ornek import SOURCE_LABELS, md
from core.labs.ornekler import get_variants
from core.labs.runner import LabRun, bootstrap_key, run_lab, statsmodels_term
from core.labs.spec import (
    SOURCES,
    Indicator,
    ReadFile,
    IV,
    OLS,
    RDD,
    Bootstrap,
    Histogram,
    RDDTable,
    ScalarTable,
    BandwidthCV,
    CoefficientProfile,
    LocalLinear,
    Plot,
    QuantileDifference,
    AverageProfile,
    BinaryChoice,
    KeepIf,
    ProfileCurves,
    QuantileRegression,
    Recode,
    Summaries,
    Tobit,
    TobitFitCheck,
    TobitTargets,
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
from topics.kendi_veri_ui import remember, render_custom, restore
from topics.regression_ui import show_figure, style_figure
from topics.selection_ui import render_lab_op

CODE_LANGUAGE_KEY = "code_language"
TEACHING_CSV_DATASETS = ("cps09mar", "ddk2011", "lm2007")
"""Öğretim CSV'si laboratuvarın bütün ham değişkenlerini taşıyan veri setleri (Card1995 ve CHJ2004'te türetilen
değişkenlerin kaynakları eksiktir; bu veri setlerinde Hansen'in .dta dosyası gerekir)."""
_KERNEL_LABELS = {"triangular": "üçgen", "rectangular": "dikdörtgen"}
_METHOD_LABELS = {"pairs": "pairs (gözlem çiftleri)", "wild": "wild (Rademacher)", "cluster": "küme"}
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


def _data_key(prefix: str) -> str:
    return f"{prefix}_lab_data"


def _local_path(dataset: str) -> Path | None:
    value = os.environ.get(DATA_PATH_ENV.format(dataset=dataset.upper()))
    if not value and dataset == "cps09mar":
        value = os.environ.get(LEGACY_CPS_ENV)
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_file() else None


def _render_data_panel(spec: LabSpec, prefix: str | None = None) -> LoadedData | None:
    """Hansen verisinin yükleme paneli. ``prefix``: oturum ve widget anahtarlarının öneki; notlarda konu anahtarı,
    alternatif örnekte ``konuNN_alternatif`` (iki kaynağın verisi birbirine karışmaz)."""

    prefix = prefix or spec.topic_key
    key = _data_key(prefix)
    member = DATASET_MEMBERS[spec.dataset]
    reference = "notlardaki sayılarla" if spec.source == "notlar" else "bu sayfadaki metinlerdeki sayılarla"

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
                    "Yüklenen dosya Hansen'in tam örnekleminden farklı. Sonuçlar "
                    f"{reference} uyuşmayabilir."
                )
            if right.button("Veriyi değiştir", key=f"{prefix}_lab_reset", width="stretch"):
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
            key=f"{prefix}_lab_download",
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
        csv_note = " ya da ders notlarının öğretim CSV'si" if spec.dataset in TEACHING_CSV_DATASETS else ""
        uploaded = right.file_uploader(
            f"veya dosya yükleyin ({member}{csv_note})",
            type=("txt", "csv", "dta"),
            key=f"{prefix}_lab_upload",
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
    name = spec.topic_key if spec.source == "notlar" else f"{spec.topic_key}_{spec.source}"
    try:
        return _run(name, fingerprint, spec, loaded.frame)
    except Exception as error:  # veri yapısı sorunları öğrenciye açıkça gösterilir
        st.error(f"Laboratuvar bu veriyle çalıştırılamadı: {error}")
        return None


def _own_run(spec: LabSpec) -> LabRun | None:
    """Kendi verinin hesabı; yalnız bu oturumda, aynı tanım için saklanır (ortak önbelleğe girmez)."""

    key = f"{spec.topic_key}_kendi_hesap"
    stored = st.session_state.get(key)
    if isinstance(stored, tuple) and len(stored) == 2 and stored[0] is spec:
        return stored[1]
    try:
        run = run_lab(spec)
    except Exception as error:  # beklenmeyen veri: anlaşılır ileti, ayrıntı türüyle
        st.error(f"Uygulama bu veriyle çalıştırılamadı ({type(error).__name__}: {md(str(error))}).",
                 icon=":material/error:")
        return None
    st.session_state[key] = (spec, run)
    return run


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
    restore(key)  # adımlar bir çalıştırmada çizilmezse (ör. kendi verinde hata) Streamlit seçimi siler
    if st.session_state.get(key) not in numbers:
        st.session_state[key] = numbers[0]
    left, middle, right = st.columns([1, 6, 1], vertical_alignment="bottom")
    left.button("‹ Önceki", key=f"{spec.topic_key}_lab_prev", on_click=_shift, args=(spec, -1), width="stretch")
    middle.segmented_control(
        "Adım",
        options=numbers,
        format_func=lambda number: f"Adım {number}",
        key=key,
        required=True,
        label_visibility="collapsed",
        width="stretch",
    )
    right.button("Sonraki ›", key=f"{spec.topic_key}_lab_next", on_click=_shift, args=(spec, 1), width="stretch")
    remember(key)
    return spec.step(st.session_state.get(key) or numbers[0])


# --- Sonuç gösterimi -------------------------------------------------------

def _describe_table(spec: LabSpec, table: pd.DataFrame) -> pd.DataFrame:
    shown = table.rename(columns=_STAT_LABELS)
    shown.index = [spec.label(name) for name in shown.index]
    shown.index.name = "Değişken"
    return shown


def _group_table(spec: LabSpec, op: GroupSummary, table: pd.DataFrame) -> pd.DataFrame:
    shown = table.rename(columns={name: spec.label(name) for name, _, _ in op.columns}).reset_index()
    groups = shown[op.by]
    if groups.dtype.kind == "f" and groups.notna().all() and (groups == groups.round()).all():
        shown[op.by] = groups.astype("int64")  # tam sayı değerli grup (ör. .dta'da ondalıklı saklanan eğitim yılı)
    by_label = spec.label(op.by)
    if by_label in [spec.label(name) for name, _, _ in op.columns]:  # ör. kendi verinde "N" adlı sütun
        by_label = f"{by_label} (grup)"
    shown = shown.rename(columns={op.by: by_label})
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
    # Satır adı sütunu içeriğe göre otomatik boyutlanırken uzun adları kesiyor; genişlik en uzun ada göre verilir.
    widest = max((len(str(label)) for label in table.index), default=0)
    label_width = int(min(480, max(120, 7.5 * widest + 24)))
    st.dataframe(shown, hide_index=True, width="stretch",
                 column_config={"": st.column_config.TextColumn(width=label_width)})


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


def _escape(spec: LabSpec):
    """Markdown metnine girecek etiketler: kendi verinde öğrencinin sütun adları kaçırılır (ör. "Fiyat ($)"), notlarda
    ve alternatif örnekte etiketler olduğu gibi yazılır."""

    return md if spec.source == "kendi" else (lambda text: text)


def _compact_model(spec: LabSpec, op: OLS, result) -> None:
    escape = _escape(spec)
    if op.categorical or len(op.regressors) > 3:
        shown = ", ".join(escape(spec.label(r)) for r in op.regressors)
        st.markdown(
            f"**{escape(spec.label(op.name))}:** {escape(spec.label(op.outcome))} ~ {shown} · {len(result.params)} "
            f"katsayı · N = {_count(result.nobs)} · R² = {_number(result.rsquared)}"
        )
        return
    parts = [_number(float(result.params["Intercept"]))]
    for name in op.regressors:
        value = float(result.params[name])
        sign = "−" if value < 0 else "+"
        label = spec.label(name) if spec.source == "kendi" else spec.label(name).lower()
        parts.append(f"{sign} {_number(abs(value))}·{escape(label)}")
    st.markdown(f"**Tahmin edilen denklem:** {escape(spec.label(op.outcome))} = " + " ".join(parts))
    st.caption(f"N = {_count(result.nobs)} · R² = {_number(result.rsquared)}")


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
    # Notlarda nokta çapı √n ile mutlak ölçeklenir (CPS'te gruplar binlerce gözlemdir); ek kaynaklarda en kalabalık
    # gruba göre ölçeklenir, böylece küçük örneklemde de noktalar görünür.
    size = 6 + 22 * np.sqrt(data["n"] / data["n"].max()) if op.relative_size else np.sqrt(data["n"]) / 2.5
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=data[op.x],
            y=data["ortalama"],
            mode="markers",
            name="Koşullu ortalama",
            marker={"size": size, "color": _COLORS[0], "opacity": 0.75},
            customdata=data[["n"]],
            hovertemplate=f"{op.x_label}: %{{x}}<br>Ortalama: %{{y:.4f}}<br>n: %{{customdata[0]}}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(x=grid, y=line, mode="lines", name="OLS doğrusal projeksiyonu", line={"color": _COLORS[3], "width": 3})
    )
    style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label, legend_title="Seri")
    show_figure(figure)


def _show_model_title(spec: LabSpec, model: str) -> str:
    """Tanımda etiket yoksa ``m3`` → "Model (3)", ``m2_hc1`` → "Model (2), HC1"."""

    label = spec.label(model)
    if label != model:
        return label
    match = re.fullmatch(r"m(\d+)(?:_(\w+))?", model)
    if match is None:
        return model
    suffix = f", {match.group(2).upper()}" if match.group(2) else ""
    return f"Model ({match.group(1)}){suffix}"


def _fit_caption(result) -> str:
    if hasattr(result, "rsquared"):
        return f"N = {_count(result.nobs)} · R² = {_number(result.rsquared)}"
    if hasattr(result, "prsquared"):
        return (f"N = {int(result.nobs):,} · log-olabilirlik = {result.llf:.2f} · "
                f"McFadden sözde R² = {result.prsquared:.4f}").replace(",", "X").replace(".", ",").replace("X", ".")
    return f"N = {int(result.nobs):,}".replace(",", ".")


def _summary_number(value: float) -> str:
    if float(value).is_integer():
        return _count(value)
    return f"{value:,.3f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _compact_binary(spec: LabSpec, op: BinaryChoice, result) -> None:
    shown = ", ".join(spec.label(r) for r in op.regressors)
    st.markdown(
        f"**{spec.label(op.name)}:** {spec.label(op.outcome)} ~ {shown} · N = {_count(result.nobs)} · "
        f"log-olabilirlik = {_number(result.llf, 2)}"
    )


def _compact_tobit(spec: LabSpec, op: Tobit, result) -> None:
    censored = int(np.sum(result.endog <= op.left))
    st.markdown(
        f"**{spec.label(op.name)}:** {spec.label(op.outcome)} ~ {len(op.regressors)} regresör, soldan "
        f"{_number(op.left, 0)}'da sansürlü · N = {_count(result.nobs)} ({_count(censored)} sansürlü) · "
        f"σ̂ = {_number(result.sigma, 2)} · log-olabilirlik = {_number(result.llf, 2)}"
    )


def _profile_plot(op: AverageProfile, data: pd.DataFrame) -> None:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=data.index, y=data["olasilik"], mode="lines+markers", name="Ortalama tahmin edilen olasılık",
            line={"color": _COLORS[0], "width": 3},
            hovertemplate=f"{op.x_label}: %{{x}}<br>Olasılık: %{{y:.4f}}<extra></extra>",
        )
    )
    style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label, legend_title="")
    show_figure(figure)


def _curves(spec: LabSpec, op: ProfileCurves, state, plot: bool = True) -> None:
    table = state.tables[op.result].reset_index()
    labels = dict(op.models)
    table.columns = [op.x_label] + [labels[column] for column in table.columns[1:]]
    profile_terms = {"Intercept", op.variable, *(name for name, _ in op.derived)}
    has_controls = any(set(state.models[model].params.index) - profile_terms for model, _ in op.models)
    note = " (kontroller örneklem ortalamasında)" if has_controls else ""
    st.markdown(f"**Seçilmiş düzeylerde tahmin**{note}")
    st.dataframe(
        table.style.format({column: "{:.2f}" for column in table.columns[1:]} | {op.x_label: "{:.0f}"}),
        hide_index=True, width="stretch",
    )
    if not plot:
        return
    data = state.plots[f"egriler:{op.result}"]
    figure = go.Figure()
    for index, (model, label) in enumerate(op.models):
        figure.add_trace(
            go.Scatter(
                x=data[op.variable], y=data[model], mode="lines", name=label,
                line={"color": _COLORS[index % len(_COLORS)], "width": 3},
                hovertemplate=f"{op.x_label}: %{{x:.0f}}<br>%{{y:.2f}}<extra>{label}</extra>",
            )
        )
    style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label, legend_title="Tahmin edici")
    show_figure(figure)


def _decimal(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def _quantile_profile(spec: LabSpec, op: CoefficientProfile, state) -> None:
    """Katsayının kantil profili, noktasal %95 güven bandı ve (varsa) OLS referans çizgisi."""

    data = state.plots[f"kantil_profili:{op.term}"]
    tau = data["tau"].to_numpy()
    low = (data["katsayi"] - 1.96 * data["sh"]).to_numpy()
    high = (data["katsayi"] + 1.96 * data["sh"]).to_numpy()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=np.concatenate([tau, tau[::-1]]), y=np.concatenate([high, low[::-1]]), fill="toself",
            fillcolor="rgba(16, 124, 137, 0.18)", line={"width": 0}, name="%95 güven bandı", hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=tau, y=data["katsayi"], mode="lines+markers", name="Kantil regresyon",
            line={"color": _COLORS[0], "width": 3}, customdata=data[["sh"]],
            hovertemplate="τ = %{x:.2f}<br>katsayı = %{y:.4f}<br>SH = %{customdata[0]:.4f}<extra></extra>",
        )
    )
    if op.reference:
        value = float(state.models[op.reference].params[statsmodels_term(op.term)])
        figure.add_trace(
            go.Scatter(
                x=[tau.min(), tau.max()], y=[value, value], mode="lines", name=op.reference_label,
                line={"color": _COLORS[1], "width": 2, "dash": "dash"},
                hovertemplate=f"{op.reference_label}: %{{y:.4f}}<extra></extra>",
            )
        )
    style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label, legend_title="")
    show_figure(figure)


def _quantile_difference(op: QuantileDifference, state) -> None:
    value, se = state.scalars[op.name], state.scalars[f"{op.name}_se"]
    st.markdown(f"**{op.comment}**")
    first, second, third, fourth = st.columns(4)
    first.metric("Fark", _number(value))
    second.metric("SH (ortak kovaryans)", _number(se))
    third.metric("z", _number(state.scalars[f"{op.name}_z"], 2))
    fourth.metric("p-değeri", _p_text(state.scalars[f"{op.name}_p"]))


def _local_linear(spec: LabSpec, op: LocalLinear, state) -> None:
    table = state.tables[op.result].reset_index()
    labels = {column: f"Yerel doğrusal, h = {_decimal(h)}" for column, h in op.bandwidths}
    x_label = spec.label(op.x)
    table.columns = [x_label] + [labels[column] for column in table.columns[1:]]
    st.markdown("**Yerel doğrusal tahmin, seçilmiş noktalarda** (Gauss çekirdeği; h çekirdeğin standart sapması)")
    st.dataframe(
        table.style.format({column: "{:.2f}" for column in table.columns[1:]} | {x_label: "{:g}"}),
        hide_index=True, width="stretch",
    )


def render_bandwidth_cv(op: BandwidthCV, state, metrics: bool = True) -> None:
    """CV(h) − en küçük CV eğrileri ve seçilen h; ``metrics=False`` iken yalnız grafik (Sezgi'de ölçüler ayrı)."""

    table = state.tables[op.result]
    step = op.grid[2]
    decimals = 1 if float(round(step * 10, 9)).is_integer() else 2
    chosen = state.scalars[f"{op.name}_h"]
    series = [("cv", "Birini dışarıda bırak", _COLORS[0], chosen)]
    if op.cluster:
        series.append(("cv_kume", "Küme-silmeli", _COLORS[1], state.scalars[f"{op.name}_h_kume"]))
    if metrics:
        columns = st.columns(len(series))
        titles = {"cv": "Birini dışarıda bırakan CV ile h", "cv_kume": "Küme-silmeli CV ile h"}
        for column, (name, _, _, selected) in zip(columns, series):
            column.metric(titles[name], _decimal(selected, decimals))
    figure = go.Figure()
    for column, label, color, selected in series:
        values = table[column] - table[column].min()
        figure.add_trace(
            go.Scatter(
                x=table.index, y=values, mode="lines", name=label, line={"color": color, "width": 3},
                hovertemplate="h = %{x:.1f}<br>CV − en küçük = %{y:.4f}<extra>" + label + "</extra>",
            )
        )
        figure.add_vline(x=selected, line={"color": color, "dash": "dot", "width": 1.5})
    style_figure(figure, title=op.title, x_title=op.x_label, y_title="CV(h) − en küçük CV", legend_title="")
    show_figure(figure)


def _rdd_compact(op: RDD, result) -> None:
    scale = "pencere ±h" if op.scale == "window" else f"Hansen ölçeği, pencere ±{_number(result.window, 2)}"
    st.markdown(
        f"**Keskin RDD** ({_KERNEL_LABELS[op.kernel]} çekirdek, h = {_number(op.bandwidth, 0)}; {scale}): "
        f"τ̂ = {_number(result.jump)} (HC1 SH {_number(float(result.bse['D']))}) · n = {_count(result.nobs)}"
    )


def _rdd_table(op: RDDTable, table: pd.DataFrame) -> None:
    shown = pd.DataFrame(
        {
            "h": [_number(h, 0) for h in table.index],
            "n_h": [_count(v) for v in table["n"]],
            "τ̂": [_number(v, 2) for v in table["tahmin"]],
            "SH (HC1)": [_number(v, 2) for v in table["sh"]],
            "Alt %95": [_number(v, 2) for v in table["alt"]],
            "Üst %95": [_number(v, 2) for v in table["ust"]],
        }
    )
    st.markdown("**Bant genişliği duyarlılığı** (üçgen çekirdek, Hansen ölçeği; güven aralığı τ̂ ± 1,96·SH)")
    st.dataframe(shown, hide_index=True, width="stretch")
    h = table.index.to_numpy(dtype=float)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=h, y=table["tahmin"], mode="markers", name="Tahmin ve %95 güven aralığı", showlegend=True,
            marker={"size": 10, "color": _COLORS[0]},
            error_y={"type": "data", "symmetric": False, "array": table["ust"] - table["tahmin"],
                     "arrayminus": table["tahmin"] - table["alt"], "color": _COLORS[0], "thickness": 2, "width": 6},
            customdata=table[["sh", "n"]],
            hovertemplate="h = %{x:.0f}<br>τ̂ = %{y:.3f}<br>SH = %{customdata[0]:.3f}<br>n = %{customdata[1]:.0f}"
                          "<extra></extra>",
        )
    )
    figure.add_hline(y=0, line={"color": _COLORS[3], "width": 1, "dash": "dot"})
    style_figure(figure, title=op.title, x_title=op.x_label, y_title=op.y_label, legend_title="")
    show_figure(figure)


def _bootstrap(spec: LabSpec, op: Bootstrap, state) -> None:
    st.markdown(
        f"**Bootstrap:** {_METHOD_LABELS[op.method]}, B = {_count(op.reps)} tekrar"
        + (f", tohum {op.seed}" if op.seed is not None else "")
    )
    for name, _ in op.collect:
        se, low, high = (state.scalars[bootstrap_key(op.result, name, key)] for key in ("se", "lo", "hi"))
        first, second, third = st.columns(3)
        first.metric(f"Bootstrap SH ({spec.label(name)})" if len(op.collect) > 1 else "Bootstrap SH", _number(se, 5))
        second.metric("Percentile %95 alt sınır", _number(low))
        third.metric("Percentile %95 üst sınır", _number(high))


def _render_results(spec: LabSpec, step: LabStep, run: LabRun) -> None:
    state = run.state
    escape = _escape(spec)
    has_table = any(isinstance(op, (RegressionTable, EffectTable, RDDTable)) for op in step.operations)
    has_plot = any(isinstance(op, Plot) for op in step.operations)
    for op in step.operations:
        if render_lab_op(spec, op, state, step.operations):
            continue
        if isinstance(op, RDD) and not has_table:
            _rdd_compact(op, state.models[op.name])
        elif isinstance(op, RDDTable):
            _rdd_table(op, state.tables[op.result])
        elif isinstance(op, Bootstrap):
            _bootstrap(spec, op, state)
        elif isinstance(op, Histogram):
            from topics.sim_ui import histogram_figure

            figure, caption = histogram_figure(op, state.plots[f"histogram:{op.title}"])
            show_figure(figure)
            st.caption(caption)
        elif isinstance(op, ScalarTable):
            table = state.tables[op.result]
            shown = pd.DataFrame({"": table.index, "Değer": [_number(v, op.decimals) for v in table["deger"]]})
            st.dataframe(shown, hide_index=True, width="stretch")
        elif isinstance(op, ReadFile):
            count = len(state.frames[op.frame])
            dropped = (f"; seçilen sütunlarda boş hücresi olan {_count(op.dropped)} satır çıkarıldı" if op.dropped
                       else "; seçilen sütunlarda boş hücre yok")
            st.markdown(f"**Analiz örneklemi:** N = {_count(count)} (dosyada {_count(count + op.dropped)} satır"
                        f"{dropped})")
        elif isinstance(op, Indicator):
            values = state.frames[op.frame][op.name]
            st.caption(f"{escape(spec.label(op.name))}: 1 → {_count((values == 1).sum())} gözlem, "
                       f"0 → {_count((values == 0).sum())} gözlem")
        elif isinstance(op, DropMissing):
            before, after = state.samples[op]
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
            st.markdown(f"**{escape(spec.label(op.by))} düzeyine göre koşullu ortalamalar**")
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
            title = escape(op.title) if op.title else "Log saatlik ücret için OLS regresyonları"
            st.markdown(f"**{title}** (parantez içinde klasik standart hatalar)")
            table = state.tables[op.result].reset_index()
            table["Terim"] = [spec.label(name) if name else "" for name in table["Terim"]]
            st.dataframe(table, hide_index=True, width="stretch")
        elif isinstance(op, ShowModel):
            result = state.models[op.model]
            st.markdown(f"**{_show_model_title(spec, op.model)} — yazılım çıktısı**")
            st.dataframe(
                _model_output(spec, result).style.format(
                    {c: "{:.4f}" for c in ("Katsayı", "Std. hata", "%95 GA alt", "%95 GA üst")}
                    | {"t": "{:.2f}", "p-değeri": "{:.3f}"}
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(_fit_caption(result))
        elif isinstance(op, KeepIf):
            before, after = state.samples[op]
            st.markdown(
                f"**Analiz örneklemi:** N = {_count(after)} "
                f"(yüklenen veride {_count(before)} gözlem; koşulu sağlamayan {_count(before - after)} gözlem çıkarıldı)"
            )
        elif isinstance(op, Recode):
            counts = state.frames[op.frame][op.name].value_counts().sort_index()
            st.caption(
                f"{spec.label(op.name)} dağılımı: " + " · ".join(f"{level}: {_count(n)}" for level, n in counts.items())
            )
        elif isinstance(op, BinaryChoice) and not has_table:
            _compact_binary(spec, op, state.models[op.name])
        elif isinstance(op, Tobit):
            _compact_tobit(spec, op, state.models[op.name])
        elif isinstance(op, QuantileRegression) and not has_table:
            result = state.models[op.name]
            st.markdown(
                f"**{spec.label(op.name)}:** {spec.label(op.outcome)} ~ {len(op.regressors)} regresör · "
                f"τ = {_number(op.q, 2)} · N = {_count(result.nobs)}"
            )
        elif isinstance(op, QuantileDifference):
            _quantile_difference(op, state)
        elif isinstance(op, CoefficientProfile):
            _quantile_profile(spec, op, state)
        elif isinstance(op, LocalLinear):
            _local_linear(spec, op, state)
        elif isinstance(op, BandwidthCV):
            render_bandwidth_cv(op, state)
        elif isinstance(op, Plot):
            from topics.sim_ui import layered_figure

            show_figure(layered_figure(op, state.plots[f"grafik:{op.title}"]))
        elif isinstance(op, Summaries):
            table = state.tables[op.result]
            shown = pd.DataFrame({"": table.index, "Değer": [_summary_number(v) for v in table["Değer"]]})
            st.dataframe(shown, hide_index=True, width="stretch")
        elif isinstance(op, AverageProfile):
            _profile_plot(op, state.plots[f"profil:{op.name}"])
        elif isinstance(op, ProfileCurves):
            _curves(spec, op, state, plot=not has_plot)
        elif isinstance(op, TobitTargets):
            table = state.tables[op.result].reset_index()
            table.columns = [spec.label(table.columns[0]), "Gizli ortalama m*(x)", "P(Y>0|x)",
                             "Gözlenen ortalama m(x)", "Pozitiflerde ortalama m#(x)"]
            st.markdown("**Tobit'in üç hedefi (kontroller örneklem ortalamasında)**")
            st.dataframe(
                table.style.format({column: "{:.3f}" if column == "P(Y>0|x)" else "{:.2f}" for column in table.columns[1:]}
                                   | {table.columns[0]: "{:.0f}"}),
                hide_index=True, width="stretch",
            )
        elif isinstance(op, TobitFitCheck):
            table = state.tables[op.result]
            st.markdown("**Model kontrolü: Tobit'in ima ettiği ile verideki**")
            left, middle, right, last = st.columns(4)
            left.metric("P(Y>0), Tobit", _number(table.loc["p_poz", "model"], 3))
            middle.metric("Pozitif payı, veri", _number(table.loc["p_poz", "veri"], 3))
            right.metric("E[Y], Tobit", _number(table.loc["ortalama", "model"], 2))
            last.metric("Ortalama, veri", _number(table.loc["ortalama", "veri"], 2))
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
    language = st.session_state.get(CODE_LANGUAGE_KEY) or LANGUAGES[0]
    if language not in languages_for(spec):
        st.info(
            "Kendi verinizle kod Python ve R'da üretilir: dosyayı okuma ve temizleme kuralları iki dilde birebir "
            "aynıdır. Kenar çubuğundan Python ya da R seçin.",
            icon=":material/code:",
        )
        return
    info = LANGUAGE_INFO[language]
    title, description = REPRO_DESCRIPTIONS[step.reproducibility]
    st.markdown(f"**{language} kodu**")
    if spec.source == "kendi":
        description = description.replace("Python, R ve Stata", "Python ve R")
        st.caption(f"İki dilde sonuç: **{title}**. {description}")
    else:
        st.caption(f"Üç dilde sonuç: **{title}**. {description}")
    st.code(render_step(spec, step.number, language), language=info.highlight, line_numbers=True)
    if step.code_note:
        st.caption(step.code_note)


_DOWNLOAD_NOTES = {
    "notlar": (
        "Her dosya veriyi indirir, bütün adımları çalıştırır ve sonunda sonuçları ders notlarındaki "
        "sayılarla karşılaştırır. Bir sayı tutmazsa hangisinin tutmadığını söyleyerek durur."
    ),
    "alternatif": (
        "Her dosya veriyi Hansen'in arşivinden indirir, bütün adımları çalıştırır ve sonunda sonuçları uygulamanın "
        "aynı veriyle (Hansen'in tam örneklemi) verdiği sayılarla karşılaştırır. Bir sayı tutmazsa hangisinin "
        "tutmadığını söyleyerek durur."
    ),
    "kendi": (
        "Her dosya veri dosyanızı okur, bütün adımları çalıştırır ve sonunda sonuçları uygulamanın aynı veriyle "
        "verdiği sayılarla karşılaştırır. Bir sayı tutmazsa hangisinin tutmadığını söyleyerek durur. Veri dosyanızı "
        "betikle aynı klasöre koyun."
    ),
}


def _render_downloads(spec: LabSpec) -> None:
    st.markdown("**Bütün laboratuvarı indirin**")
    st.caption(_DOWNLOAD_NOTES[spec.source])
    middle = "" if spec.source == "notlar" else f"{spec.source}_"
    languages = languages_for(spec)
    columns = st.columns(len(languages))
    for column, language in zip(columns, languages):
        info = LANGUAGE_INFO[language]
        column.download_button(
            f"{language} (.{info.extension})",
            data=render_script(spec, language),
            file_name=script_filename(spec, language),
            mime=info.mime,
            key=f"{spec.topic_key}_lab_download_{middle}{language}",
            icon=":material/download:",
            width="stretch",
        )


# --- Veri kaynağı ------------------------------------------------------------

_SOURCE_ICONS = {
    "notlar": ":material/menu_book:",
    "alternatif": ":material/shuffle:",
    "kendi": ":material/upload_file:",
}


def _source_key(topic_key: str) -> str:
    return f"{topic_key}_lab_kaynak"


def _render_source(topic_key: str) -> str:
    """Sekmenin en üstünde veri kaynağı seçimi; varsayılan notlardaki örnektir. Seçim başka konuya geçip dönünce de
    korunur (gölge anahtar)."""

    key = _source_key(topic_key)
    restore(key)
    if st.session_state.get(key) not in SOURCES:
        st.session_state[key] = SOURCES[0]
    st.segmented_control(
        "Veri kaynağı",
        options=list(SOURCES),
        key=key,
        required=True,
        width="stretch",
        format_func=lambda source: f"{_SOURCE_ICONS[source]} {SOURCE_LABELS[source]}",
        help="Notlardaki örnek: ders notlarındaki laboratuvar. Alternatif örnek: aynı adımlar, Hansen'in arşivindeki "
             "başka bir gerçek veriyle. Kendi verini yükle: aynı adımlar, sizin Excel ya da CSV dosyanızla.",
    )
    remember(key)
    return st.session_state[key]


def _note(step: LabStep, run: LabRun | None) -> str:
    """Adımın yorumu: ek kaynaklarda sonuçlardan yazılır (``note_for``); hesap yoksa ya da yazılamazsa sabit metin."""

    if step.note_for is not None and run is not None:
        try:
            return step.note_for(run.state)
        except Exception:  # yorum bu veriyle yazılamadı; sonuçlar yukarıda gösterildi
            return step.takeaway
    return step.takeaway


def _render_steps(spec: LabSpec, run: LabRun | None) -> None:
    """Adım gezinimi, sonuçlar, kontroller (yalnız notlarda), yorum, kod ve indirme."""

    step = _render_navigation(spec)
    st.subheader(f"Adım {step.number}: {step.title}")
    st.caption(step.note.label())
    st.markdown(step.explanation)

    if step.operations:
        if run is not None:
            _render_results(spec, step, run)
            if spec.source == "notlar":
                _render_checks(step, run)
        else:
            st.info("Sonuçları görmek için yukarıdan veriyi yükleyin. Kod aşağıda her durumda görünür.")
    note = _note(step, run)
    if note:
        st.info(note, icon=":material/lightbulb:")
    _render_code(spec, step)
    if step.number == spec.steps[-1].number:
        _render_downloads(spec)


# --- Ana giriş ---------------------------------------------------------------

def render_lab(spec: LabSpec) -> None:
    variants = get_variants(spec.topic_key)
    source = _render_source(spec.topic_key) if variants is not None else "notlar"
    if source == "notlar":
        st.markdown(
            f"Bu sekme ders notlarındaki **§{spec.note_section} {spec.title}** laboratuvarını adım adım "
            "yeniden üretir. Tablolar notlardakiyle aynı sayıları verir; kod dilini kenar çubuğundan seçin."
        )
        loaded = _render_data_panel(spec)
        run = _lab_run(spec, loaded) if loaded is not None else None
        _render_steps(spec, run)
        return
    if source == "alternatif":
        base = variants.alternative()
        st.markdown(
            f"Bu sekme ders notlarındaki **§{spec.note_section} {spec.title}** laboratuvarının adımlarını başka bir "
            f"gerçek veriyle yeniden yapar. {variants.story} Sayılar notlardakinden farklıdır; yöntem, adımlar ve kod "
            "aynıdır. Kod dilini kenar çubuğundan seçin."
        )
        loaded = _render_data_panel(base, prefix=f"{spec.topic_key}_alternatif")
        run = _lab_run(base, loaded) if loaded is not None else None
        _render_steps(base, run)
        return
    base = render_custom(spec.topic_key, variants.custom)
    if base is None:
        return
    _render_steps(base, _own_run(base))
