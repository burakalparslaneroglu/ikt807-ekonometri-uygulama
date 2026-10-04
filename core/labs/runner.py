"""Laboratuvar tanımını gerçek veri üzerinde çalıştırır ve notlarla karşılaştırır.

OLS, üretilen Python koduyla aynı kütüphaneyle (statsmodels) hesaplanır. 2SLS,
``linearmodels`` paketinin ``IV2SLS(...).fit(cov_type="robust", debiased=True)``
sonucunu veren formüllerle doğrudan numpy üzerinde hesaplanır: Monte Carlo
deneylerinde yüzlerce tekrar etkileşimli hızda çalışsın diye. İki hesabın eşitliği,
üretilen Python betiğinin uygulamanın sayılarını yeniden ürettiğini sınayan testlerle
denetlenir. R ve Stata sonuçları da testlerle eşlenir.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from core.labs import expr as E
from core.labs import limited as L
from core.labs import penalized as P
from core.labs import quantreg as Q
from core.labs import rdd as RD
from core.labs import resample as RS
from core.labs import smoothing as S
from core.labs.spec import (
    BOOT,
    CompleteCases,
    GroupMean,
    Indicator,
    ReadFile,
    IV,
    OLS,
    RDD,
    THETA,
    VCOV_TYPES,
    CoefPath,
    ComplexityCurve,
    CrossFitDML,
    CVCurve,
    Dictionary,
    DMLSplits,
    DotPlot,
    DoubleSelection,
    DrawColumns,
    EstimatePlot,
    GroupRank,
    LinePlot,
    ModelMetrics,
    Penalized,
    PostSelection,
    RowNumber,
    dml_fold_key,
    penalized_key,
    Bootstrap,
    RDDCurve,
    RDDTable,
    ScalarTable,
    VLine,
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
    ops: dict[str, object] = field(default_factory=dict)
    """Düzenlileştirilmiş ve DML modellerini üreten işlemler, model adına göre (Post-Lasso ve bölme duyarlılığı
    kaynak modelin ayarlarını buradan okur)."""


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

def _formula(outcome: str, regressors: tuple[str, ...], categorical: tuple[str, ...] = (), constant: bool = True) -> str:
    terms = [f"C({name})" if name in categorical else name for name in regressors]
    return f"{outcome} ~ " + " + ".join(terms) + ("" if constant else " - 1")


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


LARGE_DESIGN = 3_000_000
"""Tasarım matrisinin hücre sayısı bu eşiği aşarsa (ör. AK1991: 329.509 gözlem × 21 sütun) OLS sonucunda yalnız sonraki
işlemlerin okuduğu alanlar tutulur. statsmodels sonucu tasarım matrisini ve sözde tersini saklar: bu boyutta model
başına yaklaşık 100 MB. Laboratuvar birkaç modeli birlikte tuttuğu için bellek sınırlı sunucularda (Streamlit Community
Cloud) uygulama düşebilir."""


def _slim(result):
    """Büyük tasarımda katsayı, standart hata (HC1 dahil), uyum değerleri ve artıklar önbelleğe alınır; tasarım
    matrisi ve sözde tersi bırakılır. Tasarım matrisine yeniden ihtiyaç duyan işlemler (Breusch–Pagan, bootstrap) böyle
    bir modelde açık bir hatayla durur."""

    model = result.model
    if getattr(model, "exog", None) is None or model.exog.size < LARGE_DESIGN:
        return result
    _ = (result.params, result.bse, result.tvalues, result.pvalues, result.rsquared, result.rsquared_adj,
         result.resid, result.fittedvalues, result.nobs, result.df_resid, result.ssr, result.centered_tss,
         result.fvalue, result.f_pvalue, result.HC1_se)
    _ = (model.exog_names, model.data.param_names, model.data.xnames)
    for name in ("pinv_wexog", "wexog", "exog"):
        model.__dict__.pop(name, None)
    model.data.exog = None
    model.data.orig_exog = None
    model.slim = True
    return result


def fit_ols(op: OLS, frame: pd.DataFrame):
    if op.vcov not in VCOV_TYPES:
        raise ValueError(f"Desteklenmeyen kovaryans türü: {op.vcov}")
    if op.vcov == "cluster" and not op.cluster:
        raise ValueError("Küme-dayanıklı kovaryans için küme değişkeni gerekir.")
    data = estimation_sample(op, frame)
    if op.categorical:
        model = smf.ols(_formula(op.outcome, op.regressors, op.categorical, op.constant), data=data)
    else:
        # Formül ayrıştırmadan aynı tasarım matrisi: Monte Carlo tekrarlarında hız için.
        columns = {STATSMODELS_INTERCEPT: np.ones(len(data))} if op.constant else {}
        columns.update({name: data[name].to_numpy(dtype=float) for name in op.regressors})
        design = pd.DataFrame(columns, index=data.index)
        model = sm.OLS(pd.Series(data[op.outcome].to_numpy(dtype=float), index=data.index, name=op.outcome), design)
    if op.vcov == "classic":
        return _slim(model.fit())
    if op.vcov == "HC1":
        return _slim(model.fit(cov_type="HC1"))
    return _slim(model.fit(cov_type="cluster", cov_kwds={"groups": data[op.cluster].to_numpy()}))


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
    exogenous = [np.ones(nobs)]
    exogenous_names: list[str] = []
    for name in op.exogenous:
        if name in op.categorical:
            # Her düzey için bir kukla, ilk (en küçük) düzey referans: patsy C(), R factor() ve Stata i. ile aynı.
            values = data[name]
            for level in sorted(values.unique())[1:]:
                exogenous.append((values == level).to_numpy(dtype=float))
                exogenous_names.append(f"C({name})[T.{level}]")
        else:
            exogenous.append(data[name].to_numpy(dtype=float))
            exogenous_names.append(name)
    x = np.column_stack(exogenous + [data[name].to_numpy(dtype=float) for name in op.endogenous])
    z = np.column_stack(exogenous + [data[name].to_numpy(dtype=float) for name in op.instruments])
    y = data[op.outcome].to_numpy(dtype=float)
    names = [STATSMODELS_INTERCEPT, *exogenous_names, *op.endogenous]

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


def _scalar(state: LabState):
    def lookup(name: str) -> float:
        return float(state.scalars[name])

    return lookup


def _evaluate_scalar(expression: E.Expr, state: LabState) -> float:
    return float(E.evaluate(expression, coefficient=_coefficient(state), standard_error=_standard_error(state),
                            scalar=_scalar(state)))


@dataclass
class PlotLayerData:
    layer: object
    data: pd.DataFrame


def plot_range(op: Plot, frame: pd.DataFrame) -> tuple[float, float]:
    """Grafiğin yatay aralığı: tanımda verildiyse o, yoksa verinin en küçük ve en büyük değeri."""

    if op.x_range is not None:
        return float(op.x_range[0]), float(op.x_range[1])
    x = frame[op.x].astype(float)
    return float(x.min()), float(x.max())


def _plot_data(op: Plot, state: LabState) -> list[PlotLayerData]:
    if not op.frame and op.x_range is None:
        raise ValueError("Veri çerçevesiz grafikte yatay aralık (x_range) gerekir.")
    frame = state.frames[op.frame] if op.frame else pd.DataFrame({op.x: [float(op.x_range[0]), float(op.x_range[1])]})
    low, high = plot_range(op, frame)
    grid = pd.DataFrame({op.x: np.linspace(low, high, 200)})
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
        elif isinstance(layer, RDDCurve):
            curve = RD.rdd_curve(
                frame[op.x], frame[layer.y], layer.cutoff, layer.bandwidth,
                np.linspace(low, layer.cutoff, layer.points), np.linspace(layer.cutoff, high, layer.points),
            )
            data = curve.rename(columns={"x": op.x})
        elif isinstance(layer, VLine):
            data = pd.DataFrame({op.x: [float(layer.x)]})
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

    constant = [np.ones((reps, nobs))] if op.constant else []
    x = np.stack(constant + [np.broadcast_to(data[r], (reps, nobs)) for r in op.regressors], axis=2)
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
    names = ([STATSMODELS_INTERCEPT] if op.constant else []) + list(op.regressors)
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
        elif isinstance(inner, DrawColumns):
            if inner.frame != op.frame:
                return False
        elif isinstance(inner, CrossFitDML):
            # Yalnız EKK öğrenicili, kümesiz DML; dış katlar sıra numarasından türetilmiş olmalıdır (her tekrarda aynı).
            if (inner.learner != "ols" or inner.cluster is not None or inner.residuals is not None
                    or inner.outer is None or inner.frame != op.frame):
                return False
        else:
            return False
    return sum(isinstance(inner, NewSample) for inner in op.body) == 1 and isinstance(op.body[0], NewSample)


class _NotBatchable(Exception):
    """Yığın hesabı bu tekrar bloğunda kullanılamaz (ör. dış katlar tekrardan tekrara değişiyor)."""


def _batch_crossfit_ols(op: CrossFitDML, data: dict[str, np.ndarray], reps: int, nobs: int) -> _BatchFit:
    """EKK öğrenicili DML2, bütün tekrarlar için birlikte: her dış kat k için yardımcı regresyonlar (sabit + özellikler)
    k dışındaki gözlemlerde çözülür, k'de tahmin edilir; θ̂ = ΣV̂Û/ΣV̂², SH HC1 (``penalized.crossfit_dml`` ile aynı
    formüller; EKK tahminleri ölçeklemeden bağımsız olduğu için özellikler ölçeklenmez, yalnız merkezlenir)."""

    folds = np.broadcast_to(np.asarray(data[op.outer], dtype=float), (reps, nobs))
    if not np.all(folds == folds[:1]):
        raise _NotBatchable(op.name)
    folds = folds[0]
    features = np.stack([np.broadcast_to(data[name], (reps, nobs)) for name in op.features], axis=2)
    targets = {name: np.broadcast_to(data[name], (reps, nobs)) for name in (op.outcome, op.treatment)}
    fitted = {name: np.zeros((reps, nobs)) for name in targets}
    for key in np.unique(folds):
        evaluate = folds == key
        train = ~evaluate
        xa = features[:, train, :]
        center = xa.mean(axis=1, keepdims=True)
        xc, xb = xa - center, features[:, evaluate, :] - center
        gram = np.einsum("rni,rnj->rij", xc, xc)
        for name, target in targets.items():
            ya = target[:, train]
            mean = ya.mean(axis=1, keepdims=True)
            beta = np.linalg.solve(gram, np.einsum("rni,rn->ri", xc, ya - mean)[..., None])[..., 0]
            fitted[name][:, evaluate] = mean + np.einsum("rni,ri->rn", xb, beta)
    u = targets[op.outcome] - fitted[op.outcome]
    v = targets[op.treatment] - fitted[op.treatment]
    denominator = np.einsum("rn,rn->r", v, v)
    theta = np.einsum("rn,rn->r", v, u) / denominator
    score = v * (u - theta[:, None] * v)
    error = np.sqrt(np.einsum("rn,rn->r", score, score) * nobs / (nobs - 1)) / denominator
    return _BatchFit([THETA], theta[:, None], error[:, None])


def _monte_carlo_batch(op: MonteCarlo, rng: np.random.Generator) -> pd.DataFrame:
    nobs = op.body[0].nobs
    # Her Draw bir, her DrawColumns ``count`` normal çekiliş bloğu (n gözlem) tüketir; döngüdeki sırayla aynı.
    draws = sum(1 if isinstance(inner, Draw) else inner.count for inner in op.body
                if isinstance(inner, (Draw, DrawColumns)))
    width = max(nobs, sum(len(inner.features) for inner in op.body if isinstance(inner, CrossFitDML)) * nobs)
    chunk = max(1, min(op.reps, 250_000 // max(width, 1)))
    collected: dict[str, list[np.ndarray]] = {name: [] for name, _ in op.collect}
    done = 0
    while done < op.reps:
        reps = min(chunk, op.reps - done)
        normals = rng.standard_normal(size=(reps, draws, nobs))
        data: dict[str, np.ndarray] = {"id": np.arange(1, nobs + 1, dtype=float)}
        fits: dict[str, _BatchFit] = {}
        index = 0
        for inner in op.body[1:]:
            if isinstance(inner, Draw):
                data[inner.name] = inner.first + inner.second * normals[:, index, :]
                index += 1
            elif isinstance(inner, DrawColumns):
                previous = None
                for column in range(1, inner.count + 1):
                    draw = normals[:, index, :]
                    index += 1
                    if previous is None or inner.rho == 0:
                        value = draw
                    else:
                        value = inner.rho * previous + np.sqrt(1 - inner.rho**2) * draw
                    data[f"{inner.prefix}{column}"] = value
                    previous = value
            elif isinstance(inner, CrossFitDML):
                fits[inner.name] = _batch_crossfit_ols(inner, data, reps, nobs)
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
    table = None
    if batch and _batchable(op):
        try:
            table = _monte_carlo_batch(op, rng)
        except _NotBatchable:
            rng = np.random.default_rng(op.seed)  # döngü aynı çekilişlerle baştan başlar
    if table is None:
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


# --- Bootstrap -------------------------------------------------------------------

def uses_replicate_se(expression: E.Expr) -> bool:
    """İfade bootstrap tekrarının standart hatasını kullanıyor mu (percentile-t için tekrar başına HC1)?"""

    if isinstance(expression, E.StdErr):
        return expression.model == BOOT
    if isinstance(expression, E.BinOp):
        return uses_replicate_se(expression.left) or uses_replicate_se(expression.right)
    if isinstance(expression, E.Call):
        return any(uses_replicate_se(argument) for argument in expression.args)
    return False


def bootstrap_key(result: str, column: str, statistic: str) -> str:
    """Bootstrap özet skalerinin adı (``se``, ``lo``, ``hi``); üç dilde aynı ad kullanılır."""

    return f"{result}_{column}_{statistic}"


def _require_design(result, name: str) -> None:
    if getattr(result.model, "slim", False):
        raise ValueError(f"'{name}' modelinin tasarım matrisi bellekte tutulmadı (büyük veri); bu işlem yapılamaz.")


def _bootstrap(op: Bootstrap, state: LabState) -> None:
    model = state.models[op.model]
    _require_design(model, op.model)
    design = np.asarray(model.model.exog, dtype=float)
    y = np.asarray(model.model.endog, dtype=float)
    names = list(model.model.exog_names)
    if op.method == "cluster":
        if not op.cluster:
            raise ValueError("Küme bootstrap'ı için küme değişkeni gerekir.")
        clusters = state.frames[op.frame].loc[model.model.data.row_labels, op.cluster].to_numpy()
    else:
        clusters = None
    if op.seed is None:
        if op.frame not in state.rngs:
            raise ValueError("Tohumsuz bootstrap, verisini üreten örneklemin üretecini kullanır.")
        rng = state.rngs[op.frame]
    else:
        rng = np.random.default_rng(op.seed)
    need_se = any(uses_replicate_se(expression) for _, expression in op.collect)
    fitted = np.asarray(model.fittedvalues, dtype=float) if op.method == "wild" else None
    original_coefficient, original_error, scalar = _coefficient(state), _standard_error(state), _scalar(state)
    rows: list[list[float]] = []
    for replicate in RS.replicates(design, y, names, op.reps, rng, op.method, clusters, need_se, fitted):
        def coefficient(name: str, term: str, replicate=replicate) -> float:
            if name == BOOT:
                return float(replicate.params[statsmodels_term(term)])
            return original_coefficient(name, term)

        def standard_error(name: str, term: str, replicate=replicate) -> float:
            if name == BOOT:
                return float(replicate.bse[statsmodels_term(term)])
            return original_error(name, term)

        rows.append([
            float(E.evaluate(expression, coefficient=coefficient, standard_error=standard_error, scalar=scalar))
            for _, expression in op.collect
        ])
    table = pd.DataFrame(rows, columns=[name for name, _ in op.collect])
    state.tables[op.result] = table
    for column in table.columns:
        for statistic, value in RS.summary(table[column]).items():
            state.scalars[bootstrap_key(op.result, column, statistic)] = value


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


# --- Model seçimi, düzenlileştirme ve DML (Konu 11–12) -----------------------------------------------

def _penalized_scalars(state: LabState, name: str, fit: P.PenalizedFit) -> None:
    values = {
        "test_mse": fit.test_mse, "egitim_mse": fit.train_mse, "sifirdan": float(fit.nonzero), "norm": fit.norm,
        "lambda": fit.lam, "l1": fit.l1_ratio,
    }
    for quantity, value in values.items():
        state.scalars[penalized_key(name, quantity)] = float(value)


def _fit_penalized(op: Penalized, frame: pd.DataFrame) -> P.PenalizedFit:
    if op.penalty != "ols" and (op.grid is None or op.folds is None):
        raise ValueError("Cezalı model için ceza ızgarası ve CV katları gerekir.")
    return P.fit_penalized(
        frame, op.outcome, op.numeric, op.categorical, op.penalty,
        P.grid_values(op.grid) if op.grid is not None else (0.0,),
        sample=frame[op.sample].to_numpy() if op.sample else None,
        folds=frame[op.folds].to_numpy() if op.folds else None,
        rule=op.rule, l1_ratios=op.l1_ratios, standardize=op.standardize, keep_path=op.path,
    )


def _crossfit(op: CrossFitDML, frame: pd.DataFrame, outer=None) -> P.DMLFit:
    folds = outer if outer is not None else (frame[op.outer].to_numpy() if op.outer else None)
    return P.crossfit_dml(
        frame, op.outcome, op.treatment, op.features, folds,
        frame[op.inner].to_numpy() if op.inner else None,
        P.grid_values(op.grid) if op.grid is not None else None,
        op.rule, op.cluster, learner=op.learner,
    )


def _hc1_ols(frame: pd.DataFrame, outcome: str, regressors: list[str]):
    design = pd.DataFrame({STATSMODELS_INTERCEPT: np.ones(len(frame))}, index=frame.index)
    for name in regressors:
        design[name] = frame[name].to_numpy(dtype=float)
    target = pd.Series(frame[outcome].to_numpy(dtype=float), index=frame.index, name=outcome)
    return sm.OLS(target, design).fit(cov_type="HC1")


def _execute_selection(op: Operation, state: LabState) -> bool:
    """Konu 11–12 işlemleri; işlemi tanımazsa ``False``."""

    if isinstance(op, RowNumber):
        frame = state.frames[op.frame]
        frame[op.name] = np.arange(1, len(frame) + 1, dtype=float)
    elif isinstance(op, GroupRank):
        frame = state.frames[op.frame]
        frame[op.name] = P.dense_rank(frame[op.source].to_numpy()).astype(float)
    elif isinstance(op, Dictionary):
        frame = state.frames[op.frame]
        created: dict[str, np.ndarray] = {}
        for name in op.powers:
            values = frame[name].to_numpy(dtype=float)
            for power in range(2, op.degree + 1):
                created[f"{name}_{power}"] = values**power
        if op.interactions:
            for index, first in enumerate(op.base):
                for second in op.base[index + 1:]:
                    created[f"{first}_x_{second}"] = frame[first].to_numpy(dtype=float) * frame[second].to_numpy(dtype=float)
        state.frames[op.frame] = pd.concat([frame, pd.DataFrame(created, index=frame.index)], axis=1)
    elif isinstance(op, DrawColumns):
        frame, rng = state.frames[op.frame], state.rngs[op.frame]
        created = {}
        previous = None
        for index in range(1, op.count + 1):
            draw = rng.normal(0, 1, size=len(frame))
            if previous is None or op.rho == 0:
                value = draw
            else:
                value = op.rho * previous + np.sqrt(1 - op.rho**2) * draw
            created[f"{op.prefix}{index}"] = value
            previous = value
        state.frames[op.frame] = pd.concat([frame, pd.DataFrame(created, index=frame.index)], axis=1)
    elif isinstance(op, Penalized):
        fit = _fit_penalized(op, state.frames[op.frame])
        state.models[op.name] = fit
        state.ops[op.name] = op
        _penalized_scalars(state, op.name, fit)
    elif isinstance(op, PostSelection):
        source_op = state.ops[op.source]
        frame = state.frames[source_op.frame]
        fit = P.post_selection_ols(
            frame, source_op.outcome, state.models[op.source], source_op.numeric, source_op.categorical,
            sample=frame[source_op.sample].to_numpy() if source_op.sample else None, standardize=source_op.standardize,
        )
        state.models[op.name] = fit
        state.ops[op.name] = source_op
        _penalized_scalars(state, op.name, fit)
    elif isinstance(op, ModelMetrics):
        rows = []
        for model, label in op.rows:
            fit = state.models[model]
            penalized = fit.penalty in ("ridge", "lasso", "enet")
            rows.append({
                "model": model, "etiket": label, "test_mse": fit.test_mse, "sifirdan": float(fit.nonzero),
                "norm": fit.norm, "lambda": fit.lam if penalized else float("nan"),
            })
        state.tables[op.result] = pd.DataFrame(rows).set_index("model")
    elif isinstance(op, DotPlot):
        table = state.tables[op.table]
        labels = dict(op.labels)
        state.plots[f"nokta:{op.table}:{op.column}"] = pd.DataFrame(
            {"etiket": [labels.get(row, row) for row in table.index], "deger": table[op.column].to_numpy(dtype=float)}
        )
    elif isinstance(op, CVCurve):
        fit = state.models[op.model]
        cv = fit.cv.table()
        state.plots[f"cv_egrisi:{op.model}"] = cv[cv["l1_orani"] == fit.cv.l1_ratio].reset_index(drop=True)
    elif isinstance(op, CoefPath):
        fit = state.models[op.model]
        if fit.path is None:
            raise ValueError("Katsayı yolu için model path=True ile tahmin edilmelidir.")
        state.plots[f"yol:{op.model}"] = fit.path
    elif isinstance(op, CrossFitDML):
        frame = state.frames[op.frame]
        fit = _crossfit(op, frame)
        state.models[op.name] = fit
        state.ops[op.name] = op
        folds = fit.folds
        state.tables[f"{op.name}_katlar"] = folds.set_index("kat")
        for part in ("y", "d"):
            counts = folds[f"sifirdan_{part}"].to_numpy(dtype=float)
            state.scalars[dml_fold_key(op.name, part, "min")] = float(counts.min())
            state.scalars[dml_fold_key(op.name, part, "max")] = float(counts.max())
        if op.residuals is not None:
            frame[op.residuals[0]] = fit.residuals["u"].to_numpy()
            frame[op.residuals[1]] = fit.residuals["v"].to_numpy()
    elif isinstance(op, DMLSplits):
        source: CrossFitDML = state.ops[op.dml]
        frame = state.frames[source.frame]
        rank = frame[op.rank].to_numpy(dtype=float)
        reference = frame[source.outer].to_numpy(dtype=float) if source.outer else None
        partitions = [P.multiplier_folds(rank, multiplier, op.folds) for _, _, multiplier in op.rules]
        pending = [index for index, folds in enumerate(partitions)
                   if reference is None or not np.array_equal(folds, reference)]
        # Bölmeler birbirinden bağımsızdır; iş parçacıkları sonucu değiştirmez, yalnız süreyi kısaltır.
        with ThreadPoolExecutor(max_workers=max(1, min(len(pending), os.cpu_count() or 1))) as pool:
            computed = dict(zip(pending, pool.map(lambda index: _crossfit(source, frame, partitions[index]), pending)))
        rows = []
        for index, (key, label, _) in enumerate(op.rules):
            fit = computed.get(index, state.models[op.dml])  # aynı katlar: ana DML hesabı yeniden kullanılır
            rows.append({"kural": key, "etiket": label, "theta": fit.theta, "sh": fit.se})
        table = pd.DataFrame(rows).set_index("kural")
        center, error = P.median_aggregate(table["theta"], table["sh"])
        state.tables[op.result] = table
        state.models[op.name] = P.DMLFit(center, error, pd.DataFrame(), pd.DataFrame(), len(frame))
        state.scalars[f"{op.name}_min"] = float(table["theta"].min())
        state.scalars[f"{op.name}_max"] = float(table["theta"].max())
    elif isinstance(op, DoubleSelection):
        frame = state.frames[op.frame]
        chosen = P.double_selection_sets(
            frame, op.outcome, op.treatment, op.controls, P.grid_values(op.grid), frame[op.folds].to_numpy(), op.rule
        )
        state.models[f"{op.name}_sonuc"] = _hc1_ols(frame, op.outcome, [op.treatment, *chosen.outcome])
        state.models[op.name] = _hc1_ols(frame, op.outcome, [op.treatment, *chosen.union])
        state.links[op.name] = state.links[f"{op.name}_sonuc"] = "linear"
        tracked = set(op.track)
        for suffix, names in (("ny", chosen.outcome), ("nd", chosen.treatment), ("n", chosen.union)):
            state.scalars[f"{op.name}_{suffix}"] = float(len(names))
            state.scalars[f"{op.name}_{suffix}_iz"] = float(len(tracked & set(names)))
        state.tables[f"{op.name}_secim"] = pd.DataFrame(
            {"secilen": [", ".join(chosen.outcome), ", ".join(chosen.treatment), ", ".join(chosen.union)]},
            index=pd.Index(["sonuc", "tedavi", "birlesim"], name="denklem"),
        )
    elif isinstance(op, EstimatePlot):
        rows = []
        for label, model, term in op.rows:
            result = state.models[model]
            key = model_key(result, term)
            estimate, error = float(result.params[key]), float(result.bse[key])
            rows.append({"etiket": label, "tahmin": estimate, "sh": error, "alt": estimate - CI_MULTIPLIER * error,
                         "ust": estimate + CI_MULTIPLIER * error})
        table = pd.DataFrame(rows).set_index("etiket")
        state.tables[op.result] = table
        splits = state.tables[op.splits]["theta"].to_numpy(dtype=float) if op.splits else None
        state.plots[f"tahminler:{op.result}"] = (table, splits)
    elif isinstance(op, ComplexityCurve):
        frame = state.frames[op.frame]
        table = P.complexity_curve(frame, op.x, op.y, op.max_degree, frame[op.sample].to_numpy())
        state.tables[op.result] = table
        for column in table.columns:
            state.scalars[f"{op.name}_d_{column}"] = float(table[column].idxmin())
    elif isinstance(op, LinePlot):
        table = state.tables[op.table][[column for column, _ in op.columns]].copy()
        if op.relative:
            table = table - table.min()
        state.plots[f"cizgi:{op.table}:{op.title}"] = table
    else:
        return False
    return True


def read_file_frame(op: ReadFile) -> pd.DataFrame:
    """``ReadFile`` tanımındaki temizlenmiş değerlerden veri çerçevesi: sayısal sütunlar ondalıklı sayı, diğerleri metin
    (eksik değer ``None``)."""

    widths = {len(row) for row in op.rows}
    if widths - {len(op.columns)}:
        raise ValueError("Her satırda sütun sayısı kadar değer olmalıdır.")
    data: dict[str, object] = {}
    for position, (name, _, kind) in enumerate(op.columns):
        values = [row[position] for row in op.rows]
        if kind in ("sayi", "sayi_metin"):
            data[name] = np.array([np.nan if value is None else float(value) for value in values], dtype=float)
        else:
            data[name] = pd.Series(values, dtype=object)
    return pd.DataFrame(data)


def execute(op: Operation, state: LabState, sources: dict[str, pd.DataFrame]) -> None:
    if _execute_selection(op, state):
        return
    if isinstance(op, RDD):
        frame = state.frames[op.frame]
        state.models[op.name] = RD.rdd_fit(frame[op.x], frame[op.y], op.cutoff, op.bandwidth, op.kernel, op.scale)
    elif isinstance(op, RDDTable):
        table = RD.rdd_table({h: state.models[model] for h, model in op.rows})
        state.tables[op.result] = table
        state.plots[f"rdd_tablosu:{op.result}"] = table
    elif isinstance(op, Bootstrap):
        _bootstrap(op, state)
    elif isinstance(op, ScalarTable):
        state.tables[op.result] = pd.DataFrame(
            {"deger": [_evaluate_scalar(expression, state) for _, expression in op.rows]},
            index=pd.Index([label for label, _ in op.rows], name="nicelik"),
        )
    elif isinstance(op, NewSample):
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
        _require_design(result, op.model)
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
    elif isinstance(op, ReadFile):
        # Dosya uygulamada yüklenirken okunup temizlendi; tanım temizlenmiş değerleri taşır.
        state.frames[op.frame] = read_file_frame(op)
    elif isinstance(op, CompleteCases):
        source = state.frames[op.source]
        state.frames[op.frame] = source.dropna(subset=list(op.columns)).reset_index(drop=True)
    elif isinstance(op, Indicator):
        frame = state.frames[op.frame]
        values = frame[op.source]
        frame[op.name] = np.where(values.isna(), np.nan, (values == op.level).to_numpy(dtype=float))
    elif isinstance(op, GroupMean):
        frame = state.frames[op.frame]
        keys = [op.by] + ([op.condition[0]] if op.condition is not None else [])
        incomplete = [name for name in keys if frame[name].isna().any()]
        if incomplete:
            # Eksik grup ya da koşul değeri dillerde farklı işlenir (R ave satırın kendi değerini tutar, Stata egen eksik
            # anahtarı ayrı grup sayar): bu sütunlar eksiksiz olmalı.
            raise ValueError("Grup ortalaması için şu sütunlarda eksik değer olmamalı: " + ", ".join(incomplete))
        values = frame[op.source].astype(float)
        if op.condition is not None:
            variable, operator, value = op.condition
            values = values.where(_compare(frame[variable].to_numpy(dtype=float), operator, value))
        frame[op.name] = values.groupby(frame[op.by]).transform("mean")
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


def run_lab(spec: LabSpec, sources: dict[str, pd.DataFrame] | None = None) -> LabRun:
    """Bütün adımları çalıştırır ve kontrolleri değerlendirir. ``sources``: Hansen verileri (veri seti → çerçeve);
    kendi verinde boş kalır (veri ``ReadFile`` tanımındadır)."""

    sources = {} if sources is None else sources
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
