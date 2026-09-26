"""Sınırlı bağımlı değişken tahmin edicileri: Logit/Probit, marjinal etkiler, Tobit, LAD ve profiller.

Laboratuvar koşucusunun (``core.labs.runner``) kullandığı hesaplar. Logit/Probit ve LAD,
üretilen Python koduyla aynı kütüphaneyle (statsmodels) tahmin edilir; marjinal etkiler,
Tobit ve profil eğrileri burada numpy ile hesaplanır ve üretilen Python/R kodu aynı
formülleri kullanır. Eşitlikler testlerle denetlenir.

Terim adları dilden bağımsızdır: kategorik bir değişkenin düzeyi ``race4=2`` biçiminde
yazılır; statsmodels'te ``C(race4)[T.2]``, R'de ``factor(race4)2``, Stata'da ``2.race4``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import special, stats

from core.labs import expr as E

INTERCEPT = "Intercept"
_FACTOR_TERM = re.compile(r"C\((\w+)\)\[T\.([^\]]+)\]")


# --- Terim adları ---------------------------------------------------------------

def level_text(level) -> str:
    """Düzey adı: tam sayı değerli ondalıklar ``.0`` olmadan (2.0 → "2")."""

    text = str(level)
    return text[:-2] if text.endswith(".0") else text


def factor_key(variable: str, level) -> str:
    return f"{variable}={level_text(level)}"


def split_factor_key(term: str) -> tuple[str, str] | None:
    if "=" not in term:
        return None
    variable, level = term.split("=", 1)
    return variable, level


def statsmodels_name(term: str) -> str:
    """Dilden bağımsız terim adının statsmodels karşılığı."""

    if term == E.INTERCEPT:
        return INTERCEPT
    parts = split_factor_key(term)
    if parts is not None:
        return f"C({parts[0]})[T.{parts[1]}]"
    return term


def parse_statsmodels_name(name: str) -> tuple[str, str] | None:
    match = _FACTOR_TERM.fullmatch(name)
    if match is None:
        return None
    return match.group(1), level_text(match.group(2))


# --- Bağlantı fonksiyonları ------------------------------------------------------

def link_functions(link: str):
    """G (olasılık), g = G' ve g' — marjinal etki ve delta yöntemi için."""

    if link == "logit":
        def G(v):
            return special.expit(v)

        def g(v):
            p = special.expit(v)
            return p * (1.0 - p)

        def g1(v):
            p = special.expit(v)
            return p * (1.0 - p) * (1.0 - 2.0 * p)

        return G, g, g1
    if link == "probit":
        return stats.norm.cdf, stats.norm.pdf, (lambda v: -v * stats.norm.pdf(v))
    if link == "linear":
        return (lambda v: v), (lambda v: np.ones_like(v)), (lambda v: np.zeros_like(v))
    raise ValueError(f"Desteklenmeyen bağlantı: {link}")


# --- Logit / Probit ---------------------------------------------------------------

def _complete(frame: pd.DataFrame, used: list[str]) -> pd.DataFrame:
    block = frame[used]
    keep = ~block.isna().any(axis=1)
    return frame if bool(keep.all()) else frame[keep]


def fit_binary(op, frame: pd.DataFrame):
    """Logit veya Probit (statsmodels); kovaryans ``HC0`` (dayanıklı) veya klasik."""

    from core.labs.spec import BINARY_LINKS, BINARY_VCOV_TYPES

    if op.link not in BINARY_LINKS:
        raise ValueError(f"Desteklenmeyen bağlantı: {op.link}")
    if op.vcov not in BINARY_VCOV_TYPES:
        raise ValueError(f"Desteklenmeyen kovaryans türü: {op.vcov}")
    data = _complete(frame, list(dict.fromkeys([op.outcome, *op.regressors])))
    cov_type = "HC0" if op.vcov == "robust" else "nonrobust"
    if op.categorical:
        terms = [f"C({name})" if name in op.categorical else name for name in op.regressors]
        formula = f"{op.outcome} ~ " + " + ".join(terms)
        builder = smf.logit if op.link == "logit" else smf.probit
        return builder(formula, data=data).fit(disp=False, cov_type=cov_type)
    columns = {INTERCEPT: np.ones(len(data))}
    columns.update({name: data[name].to_numpy(dtype=float) for name in op.regressors})
    design = pd.DataFrame(columns, index=data.index)
    outcome = pd.Series(data[op.outcome].to_numpy(dtype=float), index=data.index, name=op.outcome)
    model = sm.Logit(outcome, design) if op.link == "logit" else sm.Probit(outcome, design)
    return model.fit(disp=False, cov_type=cov_type)


def binary_newton(x: np.ndarray, y: np.ndarray, link: str, robust: bool, max_iter: int = 100):
    """Vektörleştirilmiş Newton–Raphson: ``x`` (tekrar × n × k), ``y`` (tekrar × n).

    Monte Carlo döngüsünde yüzlerce Logit/Probit'i birlikte tahmin eder. Katsayılar ve
    standart hatalar statsmodels ile makine duyarlığında aynıdır (testle denetlenir).
    Dönüş: katsayılar, standart hatalar ve doğrusal indeks x'β.
    """

    reps, _, k = x.shape
    beta = np.zeros((reps, k))
    for _ in range(max_iter):
        z = np.einsum("rni,ri->rn", x, beta)
        score, weight = _binary_pieces(z, y, link)
        gradient = np.einsum("rni,rn->ri", x, score)
        information = np.einsum("rni,rn,rnj->rij", x, weight, x)
        step = np.linalg.solve(information, gradient[..., None])[..., 0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-11:
            break
    z = np.einsum("rni,ri->rn", x, beta)
    score, weight = _binary_pieces(z, y, link)
    inverse = np.linalg.inv(np.einsum("rni,rn,rnj->rij", x, weight, x))
    if robust:
        meat = np.einsum("rni,rn,rnj->rij", x, score**2, x)
        covariance = inverse @ meat @ inverse
    else:
        covariance = inverse
    return beta, np.sqrt(np.einsum("rii->ri", covariance)), z


def _binary_pieces(z: np.ndarray, y: np.ndarray, link: str) -> tuple[np.ndarray, np.ndarray]:
    """Gözlem başına skor çarpanı ve gözlenen bilgi ağırlığı (−Hessian = X'diag(w)X)."""

    if link == "logit":
        p = special.expit(z)
        return y - p, p * (1.0 - p)
    q = 2.0 * y - 1.0
    ratio = q * np.exp(stats.norm.logpdf(q * z) - stats.norm.logcdf(q * z))
    return ratio, ratio * (ratio + z)


# --- Marjinal etkiler ---------------------------------------------------------------

@dataclass
class EffectsFit:
    """Ortalama marjinal etkiler; statsmodels sonuçlarıyla aynı adlı alanlar."""

    params: pd.Series
    bse: pd.Series
    nobs: int

    def cov_params(self) -> pd.DataFrame:
        index = self.params.index
        return pd.DataFrame(np.diag(self.bse.to_numpy() ** 2), index=index, columns=index)

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.bse

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series(2.0 * stats.norm.sf(np.abs(self.tvalues)), index=self.params.index)


def design_frame(result) -> pd.DataFrame:
    """Tahmin edilmiş modelin tasarım matrisi, sütun adlarıyla."""

    return pd.DataFrame(np.asarray(result.model.exog, dtype=float), columns=list(result.model.exog_names))


def marginal_effects(result, link: str, terms: tuple[str, ...], discrete: tuple[str, ...] = ()) -> EffectsFit:
    """AME: sürekli terimde türevin, kategorik/kesikli terimde 1 − 0 (referans) farkının ortalaması.

    SH delta yöntemiyle, modelin kendi kovaryansıyla: √(∇'V∇), ∇ = ∂AME/∂β. Açıklayıcılar
    sabit kabul edilir (Stata ``margins`` ve statsmodels ``get_margeff`` ile aynı).
    """

    X = design_frame(result)
    columns = list(X.columns)
    b = result.params[columns].to_numpy(dtype=float)
    V = result.cov_params().loc[columns, columns].to_numpy(dtype=float)
    G, g, g1 = link_functions(link)
    matrix = X.to_numpy()
    estimates: dict[str, float] = {}
    errors: dict[str, float] = {}

    def contrast(key: str, ones: list[int], zeros: list[int]) -> None:
        x1, x0 = matrix.copy(), matrix.copy()
        x1[:, zeros] = 0.0
        x0[:, zeros] = 0.0
        x1[:, ones] = 1.0
        z1, z0 = x1 @ b, x0 @ b
        estimates[key] = float(np.mean(G(z1) - G(z0)))
        gradient = (g(z1)[:, None] * x1 - g(z0)[:, None] * x0).mean(axis=0)
        errors[key] = float(np.sqrt(gradient @ V @ gradient))

    for term in terms:
        levels = [
            (index, parsed[1]) for index, name in enumerate(columns)
            if (parsed := parse_statsmodels_name(name)) is not None and parsed[0] == term
        ]
        if levels:
            group = [index for index, _ in levels]
            for index, level in levels:
                contrast(factor_key(term, level), [index], group)
        elif term in discrete:
            index = columns.index(term)
            contrast(term, [index], [index])
        else:
            index = columns.index(term)
            z = matrix @ b
            scale = float(np.mean(g(z)))
            estimates[term] = scale * b[index]
            gradient = scale * np.eye(len(b))[index] + b[index] * (g1(z)[:, None] * matrix).mean(axis=0)
            errors[term] = float(np.sqrt(gradient @ V @ gradient))
    return EffectsFit(pd.Series(estimates, dtype=float), pd.Series(errors, dtype=float), int(result.nobs))


def average_profile(result, link: str, variable: str, values: tuple[float, ...]) -> pd.DataFrame:
    """Değişken her gözlemde aynı değere eşitlendiğinde ortalama tahmin edilen olasılık."""

    X = design_frame(result)
    b = result.params[list(X.columns)].to_numpy(dtype=float)
    G, _, _ = link_functions(link)
    probabilities = []
    for value in values:
        changed = X.copy()
        changed[variable] = float(value)
        probabilities.append(float(np.mean(G(changed.to_numpy() @ b))))
    return pd.DataFrame({"olasilik": probabilities}, index=pd.Index(values, name=variable))


# --- Tobit ---------------------------------------------------------------------------------

@dataclass
class TobitFit:
    """Tobit sonucu; statsmodels sonuçlarıyla aynı adlı alanlar. ``params`` yalnız β'dır."""

    params: pd.Series
    bse: pd.Series
    sigma: float
    sigma_se: float
    nobs: int
    llf: float
    exog: pd.DataFrame
    endog: np.ndarray
    left: float
    covariance: pd.DataFrame

    def cov_params(self) -> pd.DataFrame:
        return self.covariance

    @property
    def fittedvalues(self) -> pd.Series:
        return pd.Series(self.exog.to_numpy() @ self.params.to_numpy(), index=self.exog.index)

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.bse

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series(2.0 * stats.norm.sf(np.abs(self.tvalues)), index=self.params.index)

    def conf_int(self) -> pd.DataFrame:
        return pd.DataFrame({0: self.params - 1.96 * self.bse, 1: self.params + 1.96 * self.bse})


def tobit_newton(y: np.ndarray, x: np.ndarray, left: float = 0.0, tol: float = 1e-10, max_iter: int = 200):
    """Tobit MLE, Olsen (1978) parametrelemesiyle Newton–Raphson: γ = β/σ, θ = 1/σ.

    Bu parametrelemede log-olabilirlik içbükeydir; adım yarılama ile her adımda artar.
    Dönüş: β, σ, (β, σ) kovaryansı (ters gözlenen bilgi) ve log-olabilirlik.
    """

    shifted = y - left
    censored = shifted <= 0
    positive = ~censored
    xp, xc, yp = x[positive], x[censored], shifted[positive]
    start = np.linalg.lstsq(x, shifted, rcond=None)[0]
    sigma = float(np.std(shifted - x @ start))
    gamma, theta = start / sigma, 1.0 / sigma

    def loglik(g: np.ndarray, t: float) -> float:
        return float(
            np.sum(np.log(t) + stats.norm.logpdf(t * yp - xp @ g)) + np.sum(stats.norm.logcdf(-(xc @ g)))
        )

    def pieces(g: np.ndarray, t: float):
        residual = t * yp - xp @ g
        zc = xc @ g
        ratio = np.exp(stats.norm.logpdf(zc) - stats.norm.logcdf(-zc))
        gradient = np.concatenate([xp.T @ residual - xc.T @ ratio, [positive.sum() / t - residual @ yp]])
        h_gg = -(xp.T @ xp) - (xc * (ratio * (ratio - zc))[:, None]).T @ xc
        h_gt = xp.T @ yp
        h_tt = -positive.sum() / t**2 - yp @ yp
        hessian = np.block([[h_gg, h_gt[:, None]], [h_gt[None, :], np.array([[h_tt]])]])
        return gradient, hessian

    current = loglik(gamma, theta)
    for _ in range(max_iter):
        gradient, hessian = pieces(gamma, theta)
        step = np.linalg.solve(hessian, -gradient)
        size = 1.0
        while True:
            candidate_g, candidate_t = gamma + size * step[:-1], theta + size * step[-1]
            if candidate_t > 0:
                value = loglik(candidate_g, candidate_t)
                if value >= current - 1e-12:
                    break
            size /= 2.0
            if size < 1e-12:
                raise RuntimeError("Tobit: Newton adımı log-olabilirliği artıramadı.")
        gamma, theta, current = candidate_g, candidate_t, value
        if np.max(np.abs(size * step)) < tol:
            break
    _, hessian = pieces(gamma, theta)
    k = x.shape[1]
    covariance_olsen = np.linalg.inv(-hessian)
    jacobian = np.zeros((k + 1, k + 1))
    jacobian[:k, :k] = np.eye(k) / theta
    jacobian[:k, k] = -gamma / theta**2
    jacobian[k, k] = -1.0 / theta**2
    covariance = jacobian @ covariance_olsen @ jacobian.T
    beta = gamma / theta
    beta[0] += left
    return beta, 1.0 / theta, covariance, current


def fit_tobit(op, frame: pd.DataFrame) -> TobitFit:
    data = _complete(frame, list(dict.fromkeys([op.outcome, *op.regressors])))
    columns = {INTERCEPT: np.ones(len(data))}
    columns.update({name: data[name].to_numpy(dtype=float) for name in op.regressors})
    design = pd.DataFrame(columns, index=data.index)
    y = data[op.outcome].to_numpy(dtype=float)
    beta, sigma, covariance, loglik = tobit_newton(y, design.to_numpy(), op.left)
    names = list(design.columns)
    k = len(names)
    errors = np.sqrt(np.diag(covariance))
    return TobitFit(
        params=pd.Series(beta, index=names),
        bse=pd.Series(errors[:k], index=names),
        sigma=float(sigma),
        sigma_se=float(errors[k]),
        nobs=len(y),
        llf=loglik,
        exog=design,
        endog=y,
        left=op.left,
        covariance=pd.DataFrame(covariance[:k, :k], index=names, columns=names),
    )


def fit_quantile(op, frame: pd.DataFrame):
    """Kantil regresyon (statsmodels ``QuantReg``); üretilen Python koduyla aynı hesap."""

    data = _complete(frame, list(dict.fromkeys([op.outcome, *op.regressors])))
    formula = f"{op.outcome} ~ " + " + ".join(op.regressors)
    return smf.quantreg(formula, data=data).fit(q=op.q, max_iter=5000)


# --- Profiller ---------------------------------------------------------------------------------

def profile_design(frame: pd.DataFrame, variable: str, derived, values, needed: list[str]) -> pd.DataFrame:
    """Izgara tasarımı: ızgara değişkeni ve ondan türetilenler; diğer regresörler örneklem ortalamasında."""

    design = pd.DataFrame({variable: np.asarray(values, dtype=float)})
    for name, expression in derived:
        design[name] = np.asarray(E.evaluate(expression, design), dtype=float) * np.ones(len(design))
    for name in needed:
        if name in (INTERCEPT, variable) or name in design.columns:
            continue
        design[name] = float(frame[name].mean())
    design.insert(0, INTERCEPT, 1.0)
    return design


def linear_index(result, design: pd.DataFrame) -> np.ndarray:
    params = result.params
    return design[list(params.index)].to_numpy() @ params.to_numpy(dtype=float)


def tobit_targets(latent: pd.Series, sigma: float) -> pd.DataFrame:
    z = latent.to_numpy(dtype=float) / sigma
    cdf, pdf = stats.norm.cdf(z), stats.norm.pdf(z)
    return pd.DataFrame(
        {
            "gizli": latent.to_numpy(dtype=float),
            "p_poz": cdf,
            "gozlenen": cdf * latent.to_numpy(dtype=float) + sigma * pdf,
            "poz_ort": latent.to_numpy(dtype=float) + sigma * pdf / cdf,
        },
        index=latent.index,
    )


def tobit_fit_check(result: TobitFit) -> pd.DataFrame:
    latent = result.exog.to_numpy() @ result.params.to_numpy()
    z = (latent - result.left) / result.sigma
    cdf, pdf = stats.norm.cdf(z), stats.norm.pdf(z)
    implied_mean = result.left + cdf * (latent - result.left) + result.sigma * pdf
    observed_positive = result.endog > result.left
    return pd.DataFrame(
        {
            "model": [float(np.mean(cdf)), float(np.mean(implied_mean))],
            "veri": [float(np.mean(observed_positive)), float(np.mean(result.endog))],
        },
        index=["p_poz", "ortalama"],
    )
