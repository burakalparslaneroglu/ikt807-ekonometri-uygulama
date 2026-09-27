"""Düzenlileştirilmiş regresyon, çapraz doğrulama ve çift/yanlılıktan arındırılmış makine öğrenmesi (DML).

Ceza ölçekleri (Notlar §11.8–§11.11):

* Ridge: ``min (Y − Xβ)'(Y − Xβ) + λ β'β``. Hansen'in ölçeğidir; scikit-learn ``Ridge(alpha=λ)`` aynı
  ölçeği kullanır. R ve Stata'da kapalı biçim ``(X'X + λI)⁻¹X'Y`` yazılır.
* Lasso ve Elastic Net: yazılım ölçeği ``min (1/(2n))‖Y − Xβ‖² + λ[r‖β‖₁ + (1 − r)/2 ‖β‖²]``; r L1
  ağırlığıdır (Lasso r = 1). scikit-learn ``Lasso``/``ElasticNet``, R ``glmnet`` ve Stata ``lasso`` bu ölçeği
  kullanır. Hansen'in SSE ölçeğindeki Lasso cezası ``2nλ``'dır.

Sabit terim cezalandırılmaz: veriler eğitim ortalamasıyla merkezlenir, sabit terim ȳ − x̄'β̂'dır. Sürekli
açıklayıcılar eğitim verisinin ortalama ve standart sapmasıyla (n'e bölünen) ölçeklenir; kategori
göstergeleri ölçeklenmez. Kategori göstergeleri eğitim verisinde görülen düzeylerden kurulur, ilk düzey
referanstır; eğitimde görülmeyen bir düzey değerlendirme verisinde bütün göstergeleri sıfır alır
(scikit-learn ``OneHotEncoder(drop="first", handle_unknown="ignore")`` ile aynı).

Çapraz doğrulamada ölçekleme ve gösterge kurma her eğitim katında yeniden öğrenilir (§11.5); CV ölçütü kat
ortalama karesel hatalarının ağırlıksız ortalamasıdır, standart hatası katlar arası standart sapma/√K'dır.
Izgara büyükten küçüğe sıralıdır; eşitlikte ilk (en büyük) ceza seçilir. Lasso yolu scikit-learn
``enet_path`` ile sıcak başlangıçla, ``TOL`` toleransıyla çözülür; R ``glmnet`` ve Stata Mata koordinat
inişi aynı optimuma yakınsar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import enet_path

PENALTIES = ("ols", "ridge", "lasso", "enet")
RULES = ("min", "1se")
TOL = 1e-12
MAX_ITER = 1_000_000
GOLDEN = 0.6180339887498949
"""(√5 − 1)/2: gözlem ve küme katları için kesirli kısım kuralının çarpanı."""


# --- Katlar ----------------------------------------------------------------------------------------

def fractional(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values - np.floor(values)


def golden_key(index) -> np.ndarray:
    """u = {i·φ}: i = 1, 2, … sıra numarası için [0, 1) aralığına düzgün yayılan deterministik dizi."""

    return fractional(np.asarray(index, dtype=float) * GOLDEN)


def interval_fold(key, folds: int) -> np.ndarray:
    """Anahtarı [0, 1) aralığında K eşit parçaya bölen kat numarası: ⌊K·u⌋ + 1."""

    return (np.floor(folds * np.asarray(key, dtype=float)) + 1).astype(int)


def dense_rank(values) -> np.ndarray:
    """Küme numaralarının küçükten büyüğe sıra numarası (1, 2, …)."""

    _, codes = np.unique(np.asarray(values), return_inverse=True)
    return codes + 1


def multiplier_folds(rank, multiplier: float, folds: int) -> np.ndarray:
    """Küme katı ⌊K·{r·c}⌋ + 1: sıra numarası r, çarpan c (ör. φ, √2, √3)."""

    return interval_fold(fractional(np.asarray(rank, dtype=float) * multiplier), folds)


def grid_values(spec) -> np.ndarray:
    """``GridSpec`` (üst üs, alt üs, nokta): 10^üst'ten 10^alt'a logaritmik ızgara, büyükten küçüğe."""

    high, low, count = spec
    return np.logspace(float(high), float(low), int(count))


# --- Tasarım matrisi -------------------------------------------------------------------------------

@dataclass
class Design:
    train: np.ndarray
    other: np.ndarray
    names: list[str]


def design(train: pd.DataFrame, other: pd.DataFrame, numeric, categorical=(), standardize: bool = True) -> Design:
    """Eğitim verisinden öğrenilen ölçek ve kategori düzeyleriyle iki tasarım matrisi."""

    numeric, categorical = list(numeric), list(categorical)
    blocks_a, blocks_b, names = [], [], []
    if numeric:
        a = train[numeric].to_numpy(dtype=float)
        b = other[numeric].to_numpy(dtype=float)
        if standardize:
            mean, sd = a.mean(axis=0), a.std(axis=0)
            a, b = (a - mean) / sd, (b - mean) / sd
        blocks_a.append(a)
        blocks_b.append(b)
        names += numeric
    for column in categorical:
        levels = np.sort(train[column].unique())[1:]
        blocks_a.append((train[column].to_numpy()[:, None] == levels[None, :]).astype(float))
        blocks_b.append((other[column].to_numpy()[:, None] == levels[None, :]).astype(float))
        names += [f"{column}={format_level(level)}" for level in levels]
    return Design(np.column_stack(blocks_a), np.column_stack(blocks_b), names)


def format_level(level) -> str:
    value = float(level)
    return str(int(value)) if value.is_integer() else f"{value:g}"


# --- Ceza yolları ----------------------------------------------------------------------------------

@dataclass
class Path:
    """Izgaranın her noktasında sabit terim, katsayılar ve değerlendirme verisi tahminleri."""

    intercepts: np.ndarray  # (G,)
    coefs: np.ndarray       # (p, G)
    predictions: np.ndarray  # (m, G)


def penalty_path(xa: np.ndarray, ya: np.ndarray, xb: np.ndarray, grid, penalty: str, l1_ratio: float = 1.0) -> Path:
    """Tek bir eğitim/değerlendirme bölmesinde ceza ızgarası boyunca çözüm."""

    if penalty not in PENALTIES:
        raise ValueError(f"Desteklenmeyen ceza: {penalty}")
    grid = np.asarray(grid, dtype=float)
    x_mean, y_mean = xa.mean(axis=0), float(ya.mean())
    xc, yc = xa - x_mean, ya - y_mean
    if penalty == "ols":
        beta = np.linalg.lstsq(xc, yc, rcond=None)[0]
        coefs = np.repeat(beta[:, None], max(len(grid), 1), axis=1)
    elif penalty == "ridge":
        values, vectors = np.linalg.eigh(xc.T @ xc)
        rotated = vectors.T @ (xc.T @ yc)
        coefs = vectors @ (rotated[:, None] / (values[:, None] + grid[None, :]))
    else:
        if np.any(np.diff(grid) >= 0):
            raise ValueError("Lasso/Elastic Net ızgarası büyükten küçüğe sıralı olmalıdır.")
        ratio = 1.0 if penalty == "lasso" else float(l1_ratio)
        alphas, coefs, _ = enet_path(xc, yc, l1_ratio=ratio, alphas=grid, precompute=True, tol=TOL, max_iter=MAX_ITER)
        if not np.allclose(alphas, grid):
            raise RuntimeError("Lasso yolu beklenen ızgarayı döndürmedi.")
    intercepts = y_mean - x_mean @ coefs
    predictions = intercepts[None, :] + xb @ coefs
    return Path(intercepts, coefs, predictions)


# --- Çapraz doğrulama ------------------------------------------------------------------------------

@dataclass
class CVResult:
    grid: np.ndarray
    l1_ratios: tuple[float, ...]
    errors: np.ndarray  # (R, K, G): karışım × kat × ceza
    mean: np.ndarray    # (R, G)
    se: np.ndarray      # (R, G)
    index: tuple[int, int]  # seçilen (karışım, ceza)

    @property
    def lam(self) -> float:
        return float(self.grid[self.index[1]])

    @property
    def l1_ratio(self) -> float:
        return float(self.l1_ratios[self.index[0]])

    def table(self) -> pd.DataFrame:
        rows = []
        for r, ratio in enumerate(self.l1_ratios):
            for g, lam in enumerate(self.grid):
                rows.append({"l1_orani": ratio, "lambda": lam, "cv_ort": self.mean[r, g], "cv_sh": self.se[r, g]})
        return pd.DataFrame(rows)


def select_index(mean: np.ndarray, se: np.ndarray, rule: str) -> tuple[int, int]:
    """En küçük CV (``min``) ya da en küçüğün bir standart hatası içindeki en büyük ceza (``1se``)."""

    if rule not in RULES:
        raise ValueError(f"Desteklenmeyen seçim kuralı: {rule}")
    r, g = np.unravel_index(int(np.argmin(mean)), mean.shape)
    if rule == "1se":
        eligible = np.flatnonzero(mean[r] <= mean[r, g] + se[r, g])
        g = int(eligible[0])
    return int(r), int(g)


def cross_validate(frame: pd.DataFrame, outcome: str, numeric, categorical, folds, grid, penalty: str,
                   rule: str = "min", l1_ratios=(1.0,), standardize: bool = True) -> CVResult:
    """Kat içinde ölçeklenen K-katlı CV; ``folds`` her satırın kat numarası (1..K)."""

    folds = np.asarray(folds, dtype=int)
    grid = np.asarray(grid, dtype=float)
    ratios = tuple(float(r) for r in l1_ratios) if penalty == "enet" else (1.0,)
    keys = np.unique(folds)
    y = frame[outcome].to_numpy(dtype=float)
    errors = np.empty((len(ratios), len(keys), len(grid)))
    for k, key in enumerate(keys):
        train, valid = folds != key, folds == key
        built = design(frame[train], frame[valid], numeric, categorical, standardize)
        for r, ratio in enumerate(ratios):
            path = penalty_path(built.train, y[train], built.other, grid, penalty, ratio)
            errors[r, k] = ((y[valid][:, None] - path.predictions) ** 2).mean(axis=0)
    mean = errors.mean(axis=1)
    se = errors.std(axis=1, ddof=1) / np.sqrt(len(keys))
    return CVResult(grid, ratios, errors, mean, se, select_index(mean, se, rule))


# --- Tek model: seçilen cezayla yeniden tahmin -----------------------------------------------------

@dataclass
class PenalizedFit:
    """Eğitim örnekleminde seçilen cezayla tahmin; ``params`` statsmodels adlandırmasıyla (sabit dahil)."""

    penalty: str
    lam: float
    l1_ratio: float
    params: pd.Series
    names: list[str]
    train_mse: float
    test_mse: float
    nobs: int
    ntest: int
    fitted: pd.Series
    cv: CVResult | None = None
    path: pd.DataFrame | None = None
    extra: dict = field(default_factory=dict)

    @property
    def coef(self) -> pd.Series:
        return self.params.drop("Intercept")

    @property
    def nonzero(self) -> int:
        return int((self.coef != 0).sum())

    @property
    def norm(self) -> float:
        return float(np.sqrt((self.coef ** 2).sum()))

    @property
    def selected(self) -> list[str]:
        return [name for name, value in self.coef.items() if value != 0]


def fit_penalized(frame: pd.DataFrame, outcome: str, numeric, categorical=(), penalty: str = "lasso", grid=(0.0,),
                  sample=None, folds=None, rule: str = "min", l1_ratios=(1.0,), standardize: bool = True,
                  keep_path: bool = False) -> PenalizedFit:
    """``sample`` eğitim satırlarını (1) ve test satırlarını (0) gösterir; ``folds`` eğitim satırlarının katı."""

    n = len(frame)
    train = np.ones(n, dtype=bool) if sample is None else np.asarray(sample).astype(bool)
    training = frame[train]
    grid = np.asarray(grid, dtype=float)
    cv = None
    lam, ratio = (float(grid[0]) if len(grid) else 0.0), float(l1_ratios[0]) if penalty == "enet" else 1.0
    if penalty != "ols" and folds is not None:
        cv = cross_validate(training, outcome, numeric, categorical, np.asarray(folds)[train], grid, penalty, rule,
                            l1_ratios, standardize)
        lam, ratio = cv.lam, cv.l1_ratio
    built = design(training, frame, numeric, categorical, standardize)
    y = frame[outcome].to_numpy(dtype=float)
    path = penalty_path(built.train, y[train], built.other, np.array([lam]), penalty, ratio)
    prediction = path.predictions[:, 0]
    params = pd.Series(np.r_[path.intercepts[0], path.coefs[:, 0]], index=["Intercept"] + built.names)
    residual = y - prediction
    test = ~train
    path_table = None
    if keep_path and penalty in ("lasso", "enet", "ridge"):
        full = penalty_path(built.train, y[train], built.train[:1], grid, penalty, ratio)
        path_table = pd.DataFrame(full.coefs.T, columns=built.names).assign(**{"lambda": grid})
    return PenalizedFit(
        penalty=penalty, lam=float(lam) if penalty != "ols" else 0.0, l1_ratio=ratio, params=params, names=built.names,
        train_mse=float(np.mean(residual[train] ** 2)),
        test_mse=float(np.mean(residual[test] ** 2)) if test.any() else float("nan"),
        nobs=int(train.sum()), ntest=int(test.sum()), fitted=pd.Series(prediction, index=frame.index), cv=cv,
        path=path_table,
    )


def post_selection_ols(frame: pd.DataFrame, outcome: str, source: PenalizedFit, numeric, categorical=(), sample=None,
                       standardize: bool = True) -> PenalizedFit:
    """Post-Lasso: kaynak modelin sıfırdan farklı katsayılı sütunları üzerinde cezasız EKK (aynı ölçekleme)."""

    n = len(frame)
    train = np.ones(n, dtype=bool) if sample is None else np.asarray(sample).astype(bool)
    built = design(frame[train], frame, numeric, categorical, standardize)
    keep = [built.names.index(name) for name in source.selected]
    y = frame[outcome].to_numpy(dtype=float)
    xa, xb = built.train[:, keep], built.other[:, keep]
    path = penalty_path(xa, y[train], xb, np.array([0.0]), "ols")
    coefs = np.zeros(len(built.names))
    coefs[keep] = path.coefs[:, 0]
    params = pd.Series(np.r_[path.intercepts[0], coefs], index=["Intercept"] + built.names)
    residual = y - path.predictions[:, 0]
    test = ~train
    return PenalizedFit(
        penalty="postlasso", lam=0.0, l1_ratio=1.0, params=params, names=built.names,
        train_mse=float(np.mean(residual[train] ** 2)),
        test_mse=float(np.mean(residual[test] ** 2)) if test.any() else float("nan"),
        nobs=int(train.sum()), ntest=int(test.sum()), fitted=pd.Series(path.predictions[:, 0], index=frame.index),
    )


# --- DML ---------------------------------------------------------------------------------------------

@dataclass
class DMLFit:
    """Kısmen doğrusal modelde artık regresyonu: Û = θ V̂ + hata, sabitsiz."""

    theta: float
    se: float
    residuals: pd.DataFrame  # u, v sütunları
    folds: pd.DataFrame      # kat, lambda_y, sifirdan_y, lambda_d, sifirdan_d
    nobs: int

    @property
    def params(self) -> pd.Series:
        return pd.Series({"theta": self.theta})

    @property
    def bse(self) -> pd.Series:
        return pd.Series({"theta": self.se})


def residual_regression(u: np.ndarray, v: np.ndarray, cluster=None) -> tuple[float, float]:
    """θ̂ = Σv̂û / Σv̂² ve standart hatası: HC1 (küme yoksa) ya da CR1 (G/(G−1)·(n−1)/(n−1)) küme-dayanıklı."""

    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    n = len(u)
    denominator = float(v @ v)
    theta = float(v @ u) / denominator
    score = v * (u - theta * v)
    if cluster is None:
        meat = float(score @ score) * n / (n - 1)
    else:
        codes = np.unique(np.asarray(cluster), return_inverse=True)[1]
        sums = np.bincount(codes, weights=score)
        groups = len(sums)
        meat = float(sums @ sums) * groups / (groups - 1)
    return theta, float(np.sqrt(meat) / denominator)


def crossfit_dml(frame: pd.DataFrame, outcome: str, treatment: str, features, outer, inner, grid, rule: str = "min",
                 cluster: str | None = None, standardize: bool = True, learner: str = "lasso") -> DMLFit:
    """DML2 (Hansen §29.22): her dış kat k için yardımcı modeller k dışındaki gözlemlerde öğrenilir.

    ``outer`` dış kat numaraları (``None`` ise çapraz uyarlama yok: artıklaştırma bütün örneklemde).
    ``inner`` Lasso cezasını seçen iç CV katları (her satır için; eğitim satırlarına kısıtlanarak kullanılır).
    ``learner="ols"``: yardımcı modeller cezasız EKK'dir; ``inner`` ve ``grid`` kullanılmaz.
    """

    if learner not in ("lasso", "ols"):
        raise ValueError(f"Desteklenmeyen öğrenici: {learner}")
    features = list(features)
    y = frame[outcome].to_numpy(dtype=float)
    d = frame[treatment].to_numpy(dtype=float)
    n = len(frame)
    outer_folds = np.ones(n, dtype=int) if outer is None else np.asarray(outer, dtype=int)
    inner = None if learner == "ols" else np.asarray(inner, dtype=int)
    y_hat, d_hat = np.zeros(n), np.zeros(n)
    rows = []
    for key in np.unique(outer_folds):
        evaluate = outer_folds == key
        train = ~evaluate if outer is not None else np.ones(n, dtype=bool)
        part = frame[train]
        record = {"kat": int(key)}
        for target, store, tag in ((outcome, y_hat, "y"), (treatment, d_hat, "d")):
            if learner == "ols":
                fit_lam, fit_rule = 0.0, "ols"
            else:
                cv = cross_validate(part, target, features, (), inner[train], grid, "lasso", rule, standardize=standardize)
                fit_lam, fit_rule = cv.lam, "lasso"
            built = design(part, frame[evaluate], features, (), standardize)
            path = penalty_path(built.train, part[target].to_numpy(dtype=float), built.other, np.array([fit_lam]), fit_rule)
            store[evaluate] = path.predictions[:, 0]
            record[f"lambda_{tag}"] = fit_lam
            record[f"sifirdan_{tag}"] = int((path.coefs[:, 0] != 0).sum())
        rows.append(record)
    u, v = y - y_hat, d - d_hat
    theta, se = residual_regression(u, v, None if cluster is None else frame[cluster].to_numpy())
    return DMLFit(theta, se, pd.DataFrame({"u": u, "v": v}, index=frame.index), pd.DataFrame(rows), n)


def median_aggregate(thetas, errors) -> tuple[float, float]:
    """Bölmelerin medyan birleştirmesi (Chernozhukov vd., 2018, Tanım 3.3):
    θ̂_med = medyan θ̂_s, σ̂_med = √medyan{σ̂²_s + (θ̂_s − θ̂_med)²}."""

    thetas, errors = np.asarray(thetas, dtype=float), np.asarray(errors, dtype=float)
    center = float(np.median(thetas))
    return center, float(np.sqrt(np.median(errors**2 + (thetas - center) ** 2)))


# --- Double selection ------------------------------------------------------------------------------

def partial_out(frame: pd.DataFrame, columns, treatment: str) -> pd.DataFrame:
    """Sütunların [1, D] üzerine EKK artıkları (D'nin cezalandırılmaması için, Notlar §12.2)."""

    columns = list(columns)
    z = np.column_stack([np.ones(len(frame)), frame[treatment].to_numpy(dtype=float)])
    block = frame[columns].to_numpy(dtype=float)
    residual = block - z @ np.linalg.lstsq(z, block, rcond=None)[0]
    return pd.DataFrame(residual, columns=columns, index=frame.index)


@dataclass
class Selection:
    outcome: list[str]
    treatment: list[str]
    union: list[str]


def double_selection_sets(frame: pd.DataFrame, outcome: str, treatment: str, controls, grid, folds,
                          rule: str = "1se") -> Selection:
    """S_Y (Y ve X, D üzerinde artıklaştırıldıktan sonra Lasso), S_D (D'nin X üzerine Lasso'su) ve birleşim."""

    controls = list(controls)
    folds = np.asarray(folds, dtype=int)
    residualized = partial_out(frame, controls + [outcome], treatment)
    s_y = fit_penalized(residualized, outcome, controls, (), "lasso", grid, None, folds, rule).selected
    s_d = fit_penalized(frame, treatment, controls, (), "lasso", grid, None, folds, rule).selected
    union = [name for name in controls if name in s_y or name in s_d]
    return Selection(s_y, s_d, union)


# --- Model karmaşıklığı ---------------------------------------------------------------------------

def complexity_curve(frame: pd.DataFrame, x: str, y: str, max_degree: int, sample) -> pd.DataFrame:
    """Polinom derecesi 1, …, D için eğitim MSE, AIC, BIC, LOOCV ve test MSE (Notlar §11.3–§11.5).

    AIC = n + n·log(2πσ̂²) + 2K, BIC = n + n·log(2πσ̂²) + K·log n; σ̂² = SSR/n, K = d + 2 (katsayılar ve σ²).
    LOOCV kaldıraç değerleriyle kesin: (1/n)Σ[êᵢ/(1 − hᵢᵢ)]².
    """

    train = np.asarray(sample).astype(bool)
    values = frame[x].to_numpy(dtype=float)
    target = frame[y].to_numpy(dtype=float)
    rows = []
    for degree in range(1, int(max_degree) + 1):
        design = np.column_stack([values**power for power in range(degree + 1)])
        xa, ya = design[train], target[train]
        q, r = np.linalg.qr(xa)
        beta = np.linalg.solve(r, q.T @ ya)
        residual = ya - xa @ beta
        leverage = (q**2).sum(axis=1)
        n = len(ya)
        sigma2 = float(residual @ residual) / n
        k = degree + 2
        base = n + n * np.log(2 * np.pi * sigma2)
        test = ~train
        rows.append({
            "derece": degree,
            "egitim_mse": sigma2,
            "loocv": float(np.mean((residual / (1 - leverage)) ** 2)),
            "test_mse": float(np.mean((target[test] - design[test] @ beta) ** 2)) if test.any() else float("nan"),
            "aic": float(base + 2 * k),
            "bic": float(base + k * np.log(n)),
        })
    return pd.DataFrame(rows).set_index("derece")
