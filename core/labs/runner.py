"""Laboratuvar tanımını gerçek veri üzerinde çalıştırır ve notlarla karşılaştırır.

Hesap, üretilen Python koduyla aynı kütüphanelerle (pandas + statsmodels formül
arayüzü) yapılır. Böylece uygulamanın gösterdiği sayı ile öğrencinin indirdiği
Python betiğinin ürettiği sayı aynı koddan gelir; R ve Stata testlerle eşlenir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from core.labs import expr as E
from core.labs.spec import (
    OLS,
    BreuschPagan,
    ClusterDraw,
    DeltaMethod,
    LinearCombination,
    StandardErrorTable,
    Check,
    CoefTarget,
    Curve,
    Derive,
    Describe,
    Draw,
    MeanPoints,
    ModelLine,
    NewSample,
    Plot,
    Predict,
    Scatter,
    Summaries,
    ZeroLine,
    GroupMeanPlot,
    GroupSummary,
    LabSpec,
    LoadHansen,
    ModelTarget,
    Operation,
    ProjectionPlot,
    RegressionTable,
    Scalar,
    ScalarTarget,
    ShowModel,
    StatTarget,
)

STATSMODELS_INTERCEPT = "Intercept"


def statsmodels_term(term: str) -> str:
    return STATSMODELS_INTERCEPT if term == E.INTERCEPT else term


@dataclass
class CheckResult:
    check: Check
    value: float
    passed: bool

    @property
    def difference(self) -> float:
        return self.value - self.check.expected


@dataclass
class LabState:
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    models: dict[str, object] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    plots: dict[str, object] = field(default_factory=dict)
    rngs: dict[str, np.random.Generator] = field(default_factory=dict)


@dataclass
class LabRun:
    state: LabState
    checks: dict[int, list[CheckResult]]

    @property
    def all_passed(self) -> bool:
        return all(result.passed for items in self.checks.values() for result in items)

    def step_checks(self, number: int) -> list[CheckResult]:
        return self.checks.get(number, [])


def _formula(outcome: str, regressors: tuple[str, ...], categorical: tuple[str, ...] = ()) -> str:
    terms = [f"C({name})" if name in categorical else name for name in regressors]
    return f"{outcome} ~ " + " + ".join(terms)


def fit_ols(op: OLS, frame: pd.DataFrame):
    model = smf.ols(_formula(op.outcome, op.regressors, op.categorical), data=frame)
    if op.vcov == "classic":
        return model.fit()
    if op.vcov == "HC1":
        return model.fit(cov_type="HC1")
    if op.vcov == "cluster":
        if not op.cluster:
            raise ValueError("Küme-dayanıklı kovaryans için küme değişkeni gerekir.")
        return model.fit(cov_type="cluster", cov_kwds={"groups": frame[op.cluster]})
    raise ValueError(f"Desteklenmeyen kovaryans türü: {op.vcov}")


def _statistic(series: pd.Series, stat: str) -> float:
    if stat == "count":
        return float(series.size)
    if stat == "sum":
        return float(series.sum())
    if stat == "mean":
        return float(series.mean())
    if stat == "sd":
        return float(series.std(ddof=1))
    if stat == "median":
        return float(series.median())
    if stat == "min":
        return float(series.min())
    if stat == "max":
        return float(series.max())
    raise ValueError(f"Desteklenmeyen istatistik: {stat}")


def _coefficient(state: LabState):
    def lookup(model: str, term: str) -> float:
        return float(state.models[model].params[statsmodels_term(term)])

    return lookup


@dataclass
class PlotLayerData:
    layer: object
    data: pd.DataFrame


def _plot_data(op: Plot, state: LabState) -> list[PlotLayerData]:
    frame = state.frames[op.frame]
    x = frame[op.x].astype(float)
    grid = pd.DataFrame({op.x: np.linspace(x.min(), x.max(), 200)})
    layers: list[PlotLayerData] = []
    for layer in op.layers:
        if isinstance(layer, MeanPoints):
            data = frame.groupby(op.x)[layer.y].agg(ortalama="mean", n="size").reset_index()
        elif isinstance(layer, Scatter):
            data = frame[[op.x, layer.y]].rename(columns={layer.y: "deger"})
        elif isinstance(layer, Curve):
            data = grid.assign(deger=np.asarray(E.evaluate(layer.expr, grid), dtype=float))
        elif isinstance(layer, ModelLine):
            params = state.models[layer.model].params
            data = grid.assign(deger=params["Intercept"] + params[op.x] * grid[op.x])
        elif isinstance(layer, ZeroLine):
            data = grid.assign(deger=0.0)
        else:
            raise TypeError(f"Tanınmayan grafik katmanı: {type(layer).__name__}")
        layers.append(PlotLayerData(layer, data))
    return layers


def execute(op: Operation, state: LabState, sources: dict[str, pd.DataFrame]) -> None:
    if isinstance(op, NewSample):
        state.frames[op.frame] = pd.DataFrame({"id": np.arange(1, op.nobs + 1)})
        state.rngs[op.frame] = np.random.default_rng(op.seed)
    elif isinstance(op, ClusterDraw):
        frame, rng = state.frames[op.frame], state.rngs[op.frame]
        shocks = rng.normal(op.first, op.second, size=op.groups)
        frame[op.name] = shocks[frame[op.cluster].astype(int).to_numpy() - 1]
    elif isinstance(op, Draw):
        frame, rng = state.frames[op.frame], state.rngs[op.frame]
        if op.distribution == "normal":
            frame[op.name] = rng.normal(op.first, op.second, size=len(frame))
        elif op.distribution == "uniform":
            frame[op.name] = rng.uniform(op.first, op.second, size=len(frame))
        else:
            raise ValueError(f"Desteklenmeyen dağılım: {op.distribution}")
    elif isinstance(op, Predict):
        result = state.models[op.model]
        if op.kind == "fitted":
            state.frames[op.frame][op.name] = result.fittedvalues
        elif op.kind == "residual":
            state.frames[op.frame][op.name] = result.resid
        else:
            raise ValueError(f"Desteklenmeyen tahmin türü: {op.kind}")
    elif isinstance(op, Summaries):
        frame = state.frames[op.frame]
        state.tables[op.result] = pd.DataFrame(
            {"Değer": [_statistic(frame[variable], stat) for _, variable, stat in op.rows]},
            index=[label for label, _, _ in op.rows],
        )
    elif isinstance(op, Plot):
        state.plots[f"grafik:{op.title}"] = _plot_data(op, state)
    elif isinstance(op, StandardErrorTable):
        rows = []
        for name in op.models:
            result = state.models[name]
            rows.append(
                {
                    "model": name,
                    "katsayi": float(result.params[op.term]),
                    "klasik_sh": float(result.bse[op.term]),
                    "hc1_sh": float(result.HC1_se[op.term]),
                    "r2": float(result.rsquared),
                }
            )
        state.tables[op.result] = pd.DataFrame(rows).set_index("model")
    elif isinstance(op, BreuschPagan):
        from statsmodels.stats.diagnostic import het_breuschpagan

        result = state.models[op.model]
        lm, p_value, _, _ = het_breuschpagan(result.resid, result.model.exog)
        state.scalars[f"{op.name}_lm"] = float(lm)
        state.scalars[f"{op.name}_p"] = float(p_value)
        state.scalars[f"{op.name}_df"] = float(result.model.exog.shape[1] - 1)
    elif isinstance(op, LinearCombination):
        result = state.models[op.model]
        terms = [statsmodels_term(term) for term, _ in op.weights]
        weights = np.array([weight for _, weight in op.weights], dtype=float)
        covariance = result.cov_params().loc[terms, terms].to_numpy()
        state.scalars[op.name] = float(weights @ result.params[terms].to_numpy())
        state.scalars[f"{op.name}_se"] = float(np.sqrt(weights @ covariance @ weights))
    elif isinstance(op, DeltaMethod):
        result = state.models[op.model]
        lookup = _coefficient(state)
        coefs = E.coefficients(op.expr)
        gradient = np.array([float(E.evaluate(E.derivative(op.expr, c), coefficient=lookup)) for c in coefs])
        terms = [statsmodels_term(c.term) for c in coefs]
        covariance = result.cov_params().loc[terms, terms].to_numpy()
        state.scalars[op.name] = float(E.evaluate(op.expr, coefficient=lookup))
        state.scalars[f"{op.name}_se"] = float(np.sqrt(gradient @ covariance @ gradient))
    elif isinstance(op, LoadHansen):
        if op.dataset not in sources:
            raise ValueError(f"'{op.dataset}' verisi yüklenmemiş.")
        state.frames[op.frame] = sources[op.dataset].copy()
    elif isinstance(op, Derive):
        frame = state.frames[op.frame]
        frame[op.name] = E.evaluate(op.expr, frame)
    elif isinstance(op, Describe):
        frame = state.frames[op.frame]
        rows = {
            variable: {stat: _statistic(frame[variable], stat) for stat in op.stats}
            for variable in op.variables
        }
        state.tables[op.result] = pd.DataFrame.from_dict(rows, orient="index")
    elif isinstance(op, GroupSummary):
        frame = state.frames[op.frame]
        grouped = frame.groupby(op.by)
        table = pd.DataFrame(
            {
                name: grouped[variable].agg(
                    lambda values, stat=stat: _statistic(values, stat)
                )
                for name, variable, stat in op.columns
            }
        )
        state.tables[op.result] = table
    elif isinstance(op, OLS):
        state.models[op.name] = fit_ols(op, state.frames[op.frame])
    elif isinstance(op, ShowModel):
        if op.model not in state.models:
            raise ValueError(f"'{op.model}' modeli henüz tahmin edilmedi.")
    elif isinstance(op, RegressionTable):
        state.tables[op.result] = regression_table(state, op)
    elif isinstance(op, Scalar):
        state.scalars[op.name] = float(E.evaluate(op.expr, coefficient=_coefficient(state)))
    elif isinstance(op, GroupMeanPlot):
        frame = state.frames[op.frame]
        state.plots[f"grup:{op.y}:{op.group}"] = (
            frame.groupby([op.group, op.x])[op.y].mean().rename("ortalama").reset_index()
        )
    elif isinstance(op, ProjectionPlot):
        frame = state.frames[op.frame]
        state.plots[f"projeksiyon:{op.model}"] = (
            frame.groupby(op.x)[op.y].agg(ortalama="mean", n="size").reset_index()
        )
    else:
        raise TypeError(f"Tanınmayan işlem: {type(op).__name__}")


def regression_table(state: LabState, op: RegressionTable) -> pd.DataFrame:
    """Makale biçimli tablo: katsayı satırı ve altında parantez içinde standart hata."""

    columns = {}
    for number, name in enumerate(op.models, start=1):
        result = state.models[name]
        cells: list[str] = []
        for term in op.terms:
            key = statsmodels_term(term)
            if key in result.params.index:
                cells.append(f"{result.params[key]:.4f}")
                cells.append(f"({result.bse[key]:.4f})")
            else:
                cells.extend(("", ""))
        cells.append(f"{int(result.nobs):,}".replace(",", "."))
        cells.append(f"{result.rsquared:.4f}")
        columns[f"({number})"] = cells
    index: list[str] = []
    for term in op.terms:
        index.extend((term, ""))
    index.extend(("N", "R²"))
    table = pd.DataFrame(columns, index=index)
    table.index.name = "Terim"
    return table


def evaluate_target(target, state: LabState) -> float:
    if isinstance(target, StatTarget):
        series = state.frames[target.frame][target.variable]
        if target.where is not None:
            variable, value = target.where
            series = series[state.frames[target.frame][variable] == value]
        return _statistic(series, target.stat)
    if isinstance(target, CoefTarget):
        result = state.models[target.model]
        term = statsmodels_term(target.term)
        if target.quantity == "coef":
            return float(result.params[term])
        if target.quantity == "se":
            return float(result.bse[term])
        if target.quantity == "se_hc1":
            return float(result.HC1_se[term])
        raise ValueError(f"Desteklenmeyen katsayı niceliği: {target.quantity}")
    if isinstance(target, ModelTarget):
        result = state.models[target.model]
        if target.quantity == "r2":
            return float(result.rsquared)
        if target.quantity == "nobs":
            return float(result.nobs)
        raise ValueError(f"Desteklenmeyen model niceliği: {target.quantity}")
    if isinstance(target, ScalarTarget):
        return state.scalars[target.name]
    raise TypeError(f"Tanınmayan hedef: {type(target).__name__}")


def run_lab(spec: LabSpec, sources: dict[str, pd.DataFrame]) -> LabRun:
    state = LabState()
    checks: dict[int, list[CheckResult]] = {}
    for step in spec.steps:
        for op in step.operations:
            execute(op, state, sources)
        results = []
        for check in step.checks:
            value = evaluate_target(check.target, state)
            passed = bool(np.isfinite(value)) and abs(value - check.expected) <= check.tolerance
            results.append(CheckResult(check=check, value=value, passed=passed))
        checks[step.number] = results
    return LabRun(state=state, checks=checks)
