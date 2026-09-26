"""Laboratuvar tanımını gerçek veri üzerinde çalıştırır ve notlarla karşılaştırır.

OLS, üretilen Python koduyla aynı kütüphaneyle (statsmodels) hesaplanır. 2SLS,
``linearmodels`` paketinin ``IV2SLS(...).fit(cov_type="robust", debiased=True)``
sonucunu veren formüllerle doğrudan numpy üzerinde hesaplanır: Monte Carlo
deneylerinde yüzlerce tekrar etkileşimli hızda çalışsın diye. İki hesabın eşitliği,
üretilen Python betiğinin uygulamanın sayılarını yeniden ürettiğini sınayan testlerle
denetlenir. R ve Stata sonuçları da testlerle eşlenir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from core.labs import expr as E
from core.labs import limited as L
from core.labs import quantreg as Q
from core.labs import smoothing as S
from core.labs.spec import (
    IV,
    OLS,
    VCOV_TYPES,
    BandwidthCV,
    BinMeans,
    CoefficientProfile,
    LocalCurve,
    LocalLinear,
    LocalResidual,
    QuantileDifference,
    AverageProfile,
    BinaryChoice,
    KeepIf,
    MarginalEffects,
    ProfileCurves,
    QuantileRegression,
    Recode,
    TableTarget,
    Tobit,
    TobitFitCheck,
    TobitTargets,
    BreuschPagan,
    Check,
    ClusterDraw,
    CoefTarget,
    Curve,
    DeltaMethod,
    Derive,
    Describe,
    Draw,
    DropMissing,
    EffectTable,
    GroupMeanPlot,
    GroupSummary,
    Histogram,
    LabSpec,
    LinearCombination,
    LoadHansen,
    MeanPoints,
    ModelLine,
    ModelTarget,
    MonteCarlo,
    NewSample,
    Operation,
    Plot,
    Predict,
    ProjectionPlot,
    RegressionTable,
    Scalar,
    ScalarTarget,
    Scatter,
    ShowModel,
    StandardErrorTable,
    StatTarget,
    Summaries,
    ZeroLine,
)

STATSMODELS_INTERCEPT = "Intercept"
CI_MULTIPLIER = 1.96
"""%95 güven aralığı çarpanı; üç dilde de aynı sabit kullanılır."""


def statsmodels_term(term: str) -> str:
    """Dilden bağımsız terim adının statsmodels karşılığı (sabit ve ``değişken=düzey`` kuklaları)."""

    return L.statsmodels_name(term)


def normal_p_value(estimate: float, standard_error: float) -> float:
    """İki yönlü p-değeri, normal yaklaşımla: 2Φ(−|b/SH|)."""

    return float(2.0 * stats.norm.sf(abs(estimate / standard_error)))


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
    samples: dict[object, tuple[int, int]] = field(default_factory=dict)
    """``DropMissing`` / ``KeepIf`` öncesi ve sonrası gözlem sayısı. Anahtar işlemin kendisidir: aynı veri
    çerçevesinde art arda ``KeepIf`` ve ``DropMissing`` olabilir."""
    links: dict[str, str] = field(default_factory=dict)
    """Olasılık modellerinin bağlantısı (logit, probit; OLS için linear), model adına göre."""


@dataclass
class LabRun:
    state: LabState
    checks: dict[int, list[CheckResult]]

    @property
    def all_passed(self) -> bool:
        return all(result.passed for items in self.checks.values() for result in items)

    def step_checks(self, number: int) -> list[CheckResult]:
        return self.checks.get(number, [])


# --- OLS ------------------------------------------------------------------------

def _formula(outcome: str, regressors: tuple[str, ...], categorical: tuple[str, ...] = ()) -> str:
    terms = [f"C({name})" if name in categorical else name for name in regressors]
    return f"{outcome} ~ " + " + ".join(terms)


def _complete_rows(data: pd.DataFrame, used: list[str]) -> pd.DataFrame:
    """``used`` değişkenlerinde eksik değeri olmayan satırlar (eksik yoksa kopyasız)."""

    try:
        block = data[used].to_numpy(dtype=float)
    except (TypeError, ValueError):  # sayısal olmayan sütun: pandas'ın genel yolu
        return data.dropna(subset=used)
    keep = ~np.isnan(block).any(axis=1)
    return data if keep.all() else data[keep]


def estimation_sample(op: OLS, frame: pd.DataFrame) -> pd.DataFrame:
    """Tahmin örneklemi: ``where`` alt grubu ve modeldeki (ve küme) değişkenleri eksiksiz gözlemler."""

    data = frame
    if op.where is not None:
        variable, value = op.where
        data = data[data[variable] == value]
    used = list(dict.fromkeys([op.outcome, *op.regressors] + ([op.cluster] if op.cluster else [])))
    return _complete_rows(data, used)


def fit_ols(op: OLS, frame: pd.DataFrame):
    if op.vcov not in VCOV_TYPES:
        raise ValueError(f"Desteklenmeyen kovaryans türü: {op.vcov}")
    if op.vcov == "cluster" and not op.cluster:
        raise ValueError("Küme-dayanıklı kovaryans için küme değişkeni gerekir.")
    data = estimation_sample(op, frame)
    if op.categorical:
        model = smf.ols(_formula(op.outcome, op.regressors, op.categorical), data=data)
    else:
        # Formül ayrıştırmadan aynı tasarım matrisi: Monte Carlo tekrarlarında hız için.
        columns = {STATSMODELS_INTERCEPT: np.ones(len(data))}
        columns.update({name: data[name].to_numpy(dtype=float) for name in op.regressors})
        design = pd.DataFrame(columns, index=data.index)
        model = sm.OLS(pd.Series(data[op.outcome].to_numpy(dtype=float), index=data.index, name=op.outcome), design)
    if op.vcov == "classic":
        return model.fit()
    if op.vcov == "HC1":
        return model.fit(cov_type="HC1")
    return model.fit(cov_type="cluster", cov_kwds={"groups": data[op.cluster].to_numpy()})


# --- 2SLS -----------------------------------------------------------------------

@dataclass
class IVFit:
    """2SLS sonucu; statsmodels sonuçlarıyla aynı adlı alanlar (``params``, ``bse`` ...)."""

    params: pd.Series
    bse: pd.Series
    covariance: pd.DataFrame
    nobs: int
    rsquared: float
    resid: pd.Series
    fittedvalues: pd.Series

    def cov_params(self) -> pd.DataFrame:
        return self.covariance

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.bse

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series(2.0 * stats.norm.sf(np.abs(self.tvalues)), index=self.params.index)

    def conf_int(self) -> pd.DataFrame:
        return pd.DataFrame(
            {0: self.params - CI_MULTIPLIER * self.bse, 1: self.params + CI_MULTIPLIER * self.bse}
        )


def fit_iv(op: IV, frame: pd.DataFrame) -> IVFit:
    """2SLS ve heteroskedastisiteye dayanıklı kovaryans, n/(n−k) ölçekli (HC1).

    β = (X̂'X̂)⁻¹X̂'Y, X̂ = P_Z X; V = (X̂'X̂)⁻¹(Σ x̂ᵢx̂ᵢ'êᵢ²)(X̂'X̂)⁻¹ · n/(n−k),
    êᵢ = Yᵢ − Xᵢ'β yapısal artıktır (ikinci aşama artığı değil). Bu,
    linearmodels ``IV2SLS(...).fit(cov_type="robust", debiased=True)`` ile aynıdır.
    """

    if op.vcov != "HC1":
        raise ValueError(f"2SLS için desteklenmeyen kovaryans türü: {op.vcov}")
    if len(op.instruments) < len(op.endogenous):
        raise ValueError("Tanımlama için en az endojen değişken sayısı kadar dışlanmış araç gerekir.")
    used = list(dict.fromkeys([op.outcome, *op.endogenous, *op.instruments, *op.exogenous]))
    data = _complete_rows(frame, used)
    nobs = len(data)
    exogenous = [np.ones(nobs)] + [data[name].to_numpy(dtype=float) for name in op.exogenous]
    x = np.column_stack(exogenous + [data[name].to_numpy(dtype=float) for name in op.endogenous])
    z = np.column_stack(exogenous + [data[name].to_numpy(dtype=float) for name in op.instruments])
    y = data[op.outcome].to_numpy(dtype=float)
    names = [STATSMODELS_INTERCEPT, *op.exogenous, *op.endogenous]

    # İşlem sırası linearmodels ile aynıdır; sonuçlar makine duyarlığında örtüşür.
    pinv_z = np.linalg.pinv(z)
    projection = pinv_z @ x
    beta = np.linalg.inv((x.T @ z) @ projection) @ ((x.T @ z) @ (pinv_z @ y))
    residual = y - x @ beta
    scores = (z @ projection) * residual[:, None]
    meat = scores.T @ scores / nobs * nobs / (nobs - x.shape[1])
    bread = np.linalg.inv((x.T @ z) @ projection / nobs)
    covariance = bread @ meat @ bread / nobs
    covariance = (covariance + covariance.T) / 2
    centered = y - y.mean()
    return IVFit(
        params=pd.Series(beta, index=names),
        bse=pd.Series(np.sqrt(np.diag(covariance)), index=names),
        covariance=pd.DataFrame(covariance, index=names, columns=names),
        nobs=nobs,
        rsquared=float(1.0 - residual @ residual / (centered @ centered)),
        resid=pd.Series(residual, index=data.index),
        fittedvalues=pd.Series(x @ beta, index=data.index),
    )


# --- Yardımcılar ----------------------------------------------------------------

def _statistic(series: pd.Series, stat: str) -> float:
    if stat == "count":
        return float(series.count())
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


def model_key(result, term: str) -> str:
    """Terimin modeldeki adı: marjinal etki sonuçlarında olduğu gibi (``race4=2``), diğerlerinde statsmodels adı."""

    return term if isinstance(result, L.EffectsFit) else statsmodels_term(term)


def _coefficient(state: LabState):
    def lookup(model: str, term: str) -> float:
        result = state.models[model]
        return float(result.params[model_key(result, term)])

    return lookup


def _standard_error(state: LabState):
    def lookup(model: str, term: str) -> float:
        result = state.models[model]
        return float(result.bse[model_key(result, term)])

    return lookup


def _evaluate_scalar(expression: E.Expr, state: LabState) -> float:
    return float(E.evaluate(expression, coefficient=_coefficient(state), standard_error=_standard_error(state)))


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
            values = E.evaluate(layer.expr, grid, coefficient=_coefficient(state))
            data = grid.assign(deger=np.asarray(values, dtype=float) * np.ones(len(grid)))
        elif isinstance(layer, ModelLine):
            params = state.models[layer.model].params
            data = grid.assign(deger=params[STATSMODELS_INTERCEPT] + params[op.x] * grid[op.x])
        elif isinstance(layer, ZeroLine):
            data = grid.assign(deger=0.0)
        elif isinstance(layer, LocalCurve):
            complete = frame[[op.x, layer.y]].dropna()
            values = S.local_fit(complete[op.x], complete[layer.y], grid[op.x], layer.bandwidth, layer.degree)
            data = grid.assign(deger=values)
        elif isinstance(layer, BinMeans):
            complete = frame[[op.x, layer.y]].dropna()
            binned = S.binned_means(complete[op.x], complete[layer.y], layer.bins)
            data = pd.DataFrame({op.x: binned["x"], "ortalama": binned["y"], "n": binned["n"]})
        else:
            raise TypeError(f"Tanınmayan grafik katmanı: {type(layer).__name__}")
        layers.append(PlotLayerData(layer, data))
    return layers


def coverage_key(result: str, estimate: str) -> str:
    """Monte Carlo kapsama oranının skaler adı; üç dilde aynı ad kullanılır."""

    return f"{result}_kapsama_{estimate}"


class _BatchFit:
    """Bir tekrar yığınındaki bütün tahminler: ``params``/``bse`` sözlükleri (terim → dizi).

    ``index`` ikili modellerde doğrusal indekstir (tekrar × gözlem); ``Predict`` onu kullanır.
    """

    def __init__(self, names: list[str], params: np.ndarray, errors: np.ndarray, index=None) -> None:
        self.params = {name: params[:, position] for position, name in enumerate(names)}
        self.bse = {name: errors[:, position] for position, name in enumerate(names)}
        self.index = index


def _batch_ols(op: OLS, data: dict[str, np.ndarray], reps: int, nobs: int) -> _BatchFit:
    """Yığın OLS; ``where`` verilirse alt örneklem 0/1 ağırlıkla seçilir (her tekrarda farklı boyut)."""

    x = np.stack([np.ones((reps, nobs))] + [np.broadcast_to(data[r], (reps, nobs)) for r in op.regressors], axis=2)
    y = np.broadcast_to(data[op.outcome], (reps, nobs))
    if op.where is None:
        weight = np.ones((reps, nobs))
    else:
        variable, value = op.where
        weight = (np.broadcast_to(data[variable], (reps, nobs)) == value).astype(float)
    size = weight.sum(axis=1)
    xtx = np.einsum("rni,rn,rnj->rij", x, weight, x)
    inverse = np.linalg.inv(xtx)
    beta = np.einsum("rij,rj->ri", inverse, np.einsum("rni,rn,rn->ri", x, weight, y))
    residual = y - np.einsum("rni,ri->rn", x, beta)
    k = x.shape[2]
    if op.vcov == "classic":
        sigma2 = np.einsum("rn,rn,rn->r", weight, residual, residual) / (size - k)
        covariance = inverse * sigma2[:, None, None]
    else:
        meat = np.einsum("rni,rn,rnj->rij", x, weight * residual**2, x)
        covariance = inverse @ meat @ inverse * (size / (size - k))[:, None, None]
    names = [STATSMODELS_INTERCEPT, *op.regressors]
    return _BatchFit(names, beta, np.sqrt(np.einsum("rii->ri", covariance)))


def _batch_binary(op: BinaryChoice, data: dict[str, np.ndarray], reps: int, nobs: int) -> _BatchFit:
    x = np.stack([np.ones((reps, nobs))] + [np.broadcast_to(data[r], (reps, nobs)) for r in op.regressors], axis=2)
    y = np.broadcast_to(data[op.outcome], (reps, nobs)).astype(float)
    beta, errors, index = L.binary_newton(x, y, op.link, op.vcov == "robust")
    return _BatchFit([STATSMODELS_INTERCEPT, *op.regressors], beta, errors, index)


def _batch_iv(op: IV, data: dict[str, np.ndarray], reps: int, nobs: int) -> _BatchFit:
    def columns(names) -> list[np.ndarray]:
        return [np.broadcast_to(data[name], (reps, nobs)) for name in names]

    exogenous = [np.ones((reps, nobs))] + columns(op.exogenous)
    x = np.stack(exogenous + columns(op.endogenous), axis=2)
    z = np.stack(exogenous + columns(op.instruments), axis=2)
    y = np.broadcast_to(data[op.outcome], (reps, nobs))
    ztz_inverse = np.linalg.inv(np.einsum("rni,rnj->rij", z, z))
    xtz = np.einsum("rni,rnj->rij", x, z)
    projection = ztz_inverse @ np.einsum("rni,rnj->rij", z, x)
    p1 = xtz @ projection
    p2 = np.einsum("rij,rj->ri", xtz, np.einsum("rij,rj->ri", ztz_inverse, np.einsum("rni,rn->ri", z, y)))
    beta = np.linalg.solve(p1, p2[..., None])[..., 0]
    residual = y - np.einsum("rni,ri->rn", x, beta)
    x_hat = z @ projection
    k = x.shape[2]
    meat = np.einsum("rni,rn,rnj->rij", x_hat, residual**2, x_hat) / nobs * nobs / (nobs - k)
    bread = np.linalg.inv(p1 / nobs)
    covariance = bread @ meat @ bread / nobs
    names = [STATSMODELS_INTERCEPT, *op.exogenous, *op.endogenous]
    return _BatchFit(names, beta, np.sqrt(np.einsum("rii->ri", covariance)))


def _batchable(op: MonteCarlo) -> bool:
    """Vektörleştirilmiş yol yalnız bu işlemlerle ve yalnız normal çekilişlerle kullanılır.

    Normal çekilişler aynı bit akışından sırayla üretildiği için (tekrar × çekiliş × gözlem)
    boyutlu tek bir çekiliş, döngüdeki ardışık çekilişlerin aynısıdır.
    """

    for inner in op.body:
        if isinstance(inner, NewSample):
            if inner.seed is not None or inner.frame != op.frame:
                return False
        elif isinstance(inner, Draw):
            if inner.distribution != "normal" or inner.frame != op.frame:
                return False
        elif isinstance(inner, Derive):
            if inner.frame != op.frame:
                return False
        elif isinstance(inner, OLS):
            if inner.categorical or inner.vcov not in ("classic", "HC1"):
                return False
        elif isinstance(inner, BinaryChoice):
            if inner.categorical or inner.frame != op.frame:
                return False
        elif isinstance(inner, Predict):
            if inner.kind != "index" or inner.frame != op.frame:
                return False
        elif isinstance(inner, IV):
            continue
        else:
            return False
    return sum(isinstance(inner, NewSample) for inner in op.body) == 1 and isinstance(op.body[0], NewSample)


def _monte_carlo_batch(op: MonteCarlo, rng: np.random.Generator) -> pd.DataFrame:
    nobs = op.body[0].nobs
    draws = [inner for inner in op.body if isinstance(inner, Draw)]
    chunk = max(1, min(op.reps, 250_000 // max(nobs, 1)))
    collected: dict[str, list[np.ndarray]] = {name: [] for name, _ in op.collect}
    done = 0
    while done < op.reps:
        reps = min(chunk, op.reps - done)
        normals = rng.standard_normal(size=(reps, len(draws), nobs))
        data: dict[str, np.ndarray] = {"id": np.arange(1, nobs + 1, dtype=float)}
        fits: dict[str, _BatchFit] = {}
        index = 0
        for inner in op.body[1:]:
            if isinstance(inner, Draw):
                data[inner.name] = inner.first + inner.second * normals[:, index, :]
                index += 1
            elif isinstance(inner, Derive):
                data[inner.name] = np.broadcast_to(np.asarray(E.evaluate(inner.expr, data), dtype=float), (reps, nobs))
            elif isinstance(inner, OLS):
                fits[inner.name] = _batch_ols(inner, data, reps, nobs)
            elif isinstance(inner, BinaryChoice):
                fits[inner.name] = _batch_binary(inner, data, reps, nobs)
            elif isinstance(inner, Predict):
                data[inner.name] = fits[inner.model].index
            else:
                fits[inner.name] = _batch_iv(inner, data, reps, nobs)
        for name, expression in op.collect:
            values = E.evaluate(
                expression,
                coefficient=lambda model, term: fits[model].params[statsmodels_term(term)],
                standard_error=lambda model, term: fits[model].bse[statsmodels_term(term)],
            )
            collected[name].append(np.broadcast_to(np.asarray(values, dtype=float), (reps,)))
        done += reps
    return pd.DataFrame({name: np.concatenate(parts) for name, parts in collected.items()})


def _monte_carlo(op: MonteCarlo, state: LabState, sources: dict[str, pd.DataFrame], batch: bool = True) -> None:
    rng = np.random.default_rng(op.seed)
    names = [name for name, _ in op.collect]
    if batch and _batchable(op):
        table = _monte_carlo_batch(op, rng)
    else:
        rows: list[list[float]] = []
        for _ in range(op.reps):
            local = LabState(rngs={op.frame: rng})
            for inner in op.body:
                execute(inner, local, sources)
            rows.append([_evaluate_scalar(expression, local) for _, expression in op.collect])
        table = pd.DataFrame(rows, columns=names)
    state.tables[op.result] = table
    for estimate, standard_error, truth in op.coverage:
        covered = (table[estimate] - truth).abs() <= CI_MULTIPLIER * table[standard_error]
        state.scalars[coverage_key(op.result, estimate)] = float(covered.mean())


# --- İşlemler -------------------------------------------------------------------

def _compare(values: np.ndarray, operator: str, value: float) -> np.ndarray:
    if operator == "==":
        return values == value
    if operator == "!=":
        return values != value
    if operator == "<":
        return values < value
    if operator == "<=":
        return values <= value
    if operator == ">":
        return values > value
    if operator == ">=":
        return values >= value
    raise ValueError(f"Desteklenmeyen karşılaştırma: {operator}")


def execute(op: Operation, state: LabState, sources: dict[str, pd.DataFrame]) -> None:
    if isinstance(op, NewSample):
        state.frames[op.frame] = pd.DataFrame({"id": np.arange(1, op.nobs + 1)})
        if op.seed is not None:
            state.rngs[op.frame] = np.random.default_rng(op.seed)
        elif op.frame not in state.rngs:
            raise ValueError("Tohumsuz örneklem yalnız Monte Carlo döngüsü içinde kullanılabilir.")
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
        if op.kind in ("fitted", "index"):
            # İkili modellerde ve Tobit'te fittedvalues doğrusal indekstir (x'β).
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
    elif isinstance(op, DropMissing):
        frame = state.frames[op.frame]
        missing = [name for name in op.variables if name not in frame.columns]
        if missing:
            raise ValueError("Veride beklenen değişkenler yok: " + ", ".join(missing))
        kept = frame.dropna(subset=list(op.variables)).copy()
        state.samples[op] = (len(frame), len(kept))
        state.frames[op.frame] = kept
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
    elif isinstance(op, KeepIf):
        frame = state.frames[op.frame]
        keep = np.ones(len(frame), dtype=bool)
        for variable, operator, value in op.conditions:
            keep &= _compare(frame[variable].to_numpy(dtype=float), operator, value)
        state.samples[op] = (len(frame), int(keep.sum()))
        state.frames[op.frame] = frame[keep].copy()
    elif isinstance(op, Recode):
        frame = state.frames[op.frame]
        source = frame[op.source].to_numpy(dtype=float)
        conditions = [np.isin(source, np.asarray(values, dtype=float)) for values, _ in op.mapping]
        frame[op.name] = np.select(conditions, [code for _, code in op.mapping], default=op.other).astype(int)
    elif isinstance(op, BinaryChoice):
        state.models[op.name] = L.fit_binary(op, state.frames[op.frame])
        state.links[op.name] = op.link
    elif isinstance(op, MarginalEffects):
        state.models[op.name] = L.marginal_effects(
            state.models[op.model], state.links.get(op.model, "linear"), op.terms, op.discrete
        )
    elif isinstance(op, AverageProfile):
        table = L.average_profile(state.models[op.model], state.links.get(op.model, "linear"), op.variable, op.values)
        state.tables[op.name] = table
        state.plots[f"profil:{op.name}"] = table
    elif isinstance(op, Tobit):
        state.models[op.name] = L.fit_tobit(op, state.frames[op.frame])
    elif isinstance(op, QuantileRegression):
        state.models[op.name] = Q.fit_quantile(op, state.frames[op.frame])
    elif isinstance(op, QuantileDifference):
        difference, standard_error = Q.quantile_difference(
            state.models[op.low], state.models[op.high], statsmodels_term(op.term)
        )
        state.scalars[op.name] = difference
        state.scalars[f"{op.name}_se"] = standard_error
        state.scalars[f"{op.name}_z"] = difference / standard_error
        state.scalars[f"{op.name}_p"] = normal_p_value(difference, standard_error)
    elif isinstance(op, CoefficientProfile):
        key = statsmodels_term(op.term)
        rows = [
            {"tau": tau, "katsayi": float(state.models[model].params[key]),
             "sh": float(state.models[model].bse[key])}
            for tau, model in op.models
        ]
        state.plots[f"kantil_profili:{op.term}"] = pd.DataFrame(rows)
    elif isinstance(op, LocalLinear):
        frame = state.frames[op.frame][[op.x, op.y]].dropna()
        state.tables[op.result] = pd.DataFrame(
            {column: S.local_linear(frame[op.x], frame[op.y], op.values, h) for column, h in op.bandwidths},
            index=pd.Index(op.values, name=op.x),
        )
    elif isinstance(op, BandwidthCV):
        used = [op.x, op.y] + ([op.cluster] if op.cluster else [])
        frame = state.frames[op.frame][used].dropna()
        grid = S.bandwidth_grid(*op.grid)
        cluster = frame[op.cluster].to_numpy() if op.cluster else None
        table = S.cv_curve(frame[op.x], frame[op.y], grid, cluster)
        state.tables[op.result] = table
        state.scalars[f"{op.name}_h"] = float(table["cv"].idxmin())
        if op.cluster:
            state.scalars[f"{op.name}_h_kume"] = float(table["cv_kume"].idxmin())
        state.plots[f"cv:{op.result}"] = table
    elif isinstance(op, LocalResidual):
        frame = state.frames[op.frame]
        frame[op.name] = S.local_residuals(frame[op.x], frame[op.variable], op.bandwidth)
    elif isinstance(op, ProfileCurves):
        frame = state.frames[op.frame]
        needed = list(dict.fromkeys(name for model, _ in op.models for name in state.models[model].params.index))
        table = L.profile_design(frame, op.variable, op.derived, op.values, needed)
        low, high, count = op.plot_grid
        grid = L.profile_design(frame, op.variable, op.derived, np.linspace(low, high, int(count)), needed)
        state.tables[op.result] = pd.DataFrame(
            {model: L.linear_index(state.models[model], table) for model, _ in op.models},
            index=pd.Index(op.values, name=op.variable),
        )
        state.plots[f"egriler:{op.result}"] = pd.DataFrame(
            {op.variable: grid[op.variable]}
            | {model: L.linear_index(state.models[model], grid) for model, _ in op.models}
        )
    elif isinstance(op, TobitTargets):
        state.tables[op.result] = L.tobit_targets(state.tables[op.curves][op.model], state.models[op.model].sigma)
    elif isinstance(op, TobitFitCheck):
        state.tables[op.result] = L.tobit_fit_check(state.models[op.model])
    elif isinstance(op, OLS):
        state.models[op.name] = fit_ols(op, state.frames[op.frame])
        state.links[op.name] = "linear"
    elif isinstance(op, IV):
        state.models[op.name] = fit_iv(op, state.frames[op.frame])
    elif isinstance(op, EffectTable):
        rows = []
        for label, model, term in op.rows:
            result = state.models[model]
            key = model_key(result, term)
            estimate, standard_error = float(result.params[key]), float(result.bse[key])
            rows.append(
                {
                    "etiket": label,
                    "tahmin": estimate,
                    "sh": standard_error,
                    "p": normal_p_value(estimate, standard_error),
                    "n": int(result.nobs),
                }
            )
        state.tables[op.result] = pd.DataFrame(rows).set_index("etiket")
    elif isinstance(op, MonteCarlo):
        _monte_carlo(op, state, sources)
    elif isinstance(op, Histogram):
        table = state.tables[op.table]
        state.plots[f"histogram:{op.title}"] = table[[column for column, _ in op.columns]].copy()
    elif isinstance(op, ShowModel):
        if op.model not in state.models:
            raise ValueError(f"'{op.model}' modeli henüz tahmin edilmedi.")
    elif isinstance(op, RegressionTable):
        state.tables[op.result] = regression_table(state, op)
    elif isinstance(op, Scalar):
        state.scalars[op.name] = _evaluate_scalar(op.expr, state)
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


# --- Notlarla karşılaştırma ---------------------------------------------------------

def evaluate_target(target, state: LabState) -> float:
    if isinstance(target, StatTarget):
        series = state.frames[target.frame][target.variable]
        if target.where is not None:
            variable, value = target.where
            series = series[state.frames[target.frame][variable] == value]
        return _statistic(series, target.stat)
    if isinstance(target, CoefTarget):
        result = state.models[target.model]
        term = model_key(result, target.term)
        if target.quantity == "coef":
            return float(result.params[term])
        if target.quantity == "se":
            return float(result.bse[term])
        if target.quantity == "se_hc1":
            return float(result.HC1_se[term])
        if target.quantity == "p":
            return normal_p_value(float(result.params[term]), float(result.bse[term]))
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
    if isinstance(target, TableTarget):
        return float(state.tables[target.table].loc[target.row, target.column])
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
