"""Konu 5–8 ek kaynakları: alternatif örneklerin bağımsız hesabı, yeni hesap işlemleri ve kendi veri kuralları.

* CHJ2004 (Konu 5) ve CK1994 (Konu 6) alternatiflerinin bütün kontrol değerleri ve metinlerdeki sayılar, uygulamanın
  kodundan ayrı bir hesapla doğrulanır: Logit/Probit statsmodels ile, ortalama marjinal etkiler ve delta yöntemi
  standart hataları elle (sayısal türev), LAD primal doğrusal programla, Tobit genel amaçlı bir optimizasyonla. Dosya yolu
  verilmezse atlanır: ``IKT807_HANSEN_CHJ2004_PATH``, ``IKT807_HANSEN_CK1994_PATH``.
* İki alternatifin üretilen Python ve R kodu gerçek veri olmadan, sentetik bir ``.dta`` dosyasıyla da çalıştırılır.
* Kendi verinde Konu 5 (ayrışma, az gözlemli kategori, gösterge 0/1 sayısal ve AME'de sonlu fark) ve Konu 6 (negatif
  değer, sıfır payı sınırları, düğümler, tek olmayan LAD çözümünde LAD eğrisinin karşılaştırılmaması) kuralları; üretilen
  Python ve R kodu aynı sayıları verir (R yoksa R adımları atlanır).
* Konu 7 (Card1995) ve Konu 8 (AL1999) alternatifleri primal doğrusal programla, elle yazılmış Hendricks–Koenker
  hesabıyla ve döngüyle ağırlıklı en küçük kareler CV'siyle doğrulanır. Kendi verinde Konu 7'de tek olmayan kantil
  çözümleri, az değerli sonuç ve terim ölçekleri; Konu 8'de iki CV'nin aynı h'yi seçmesi, seyrek uç değerler (sayısal
  olarak güvenilir ızgara), çarpık açıklayıcı, tam doğrusal sonuç ve ölçek kuralları sınanır.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import optimize, stats

from core.codegen.base import render_script, script_filename
from core.codegen.r_gen import r_row_name
from core.hansen_data import DATASET_MEMBERS, load_from_upload
from core.labs import kendi_veri as K
from core.labs import ornek_konu05 as K5
from core.labs import ornek_konu06 as K6
from core.labs import quantreg as Q
from core.labs.ornek import CustomChoices, custom_case, with_app_values
from core.labs.registry import LABS
from core.labs.runner import run_lab
from core.labs.spec import BinaryChoice, MarginalEffects, OLS

RSCRIPT = shutil.which("Rscript")
CSV = "veri.csv"
KONU05 = CustomChoices(
    roles={"sonuc": "İşe yerleşti", "profil": "Yaş", "gosterge": "Meslek kursu"},
    extra=("Eğitim yılı", "Kadın (1/0)", "İl merkezinde yaşıyor (1/0)"), picks={"sonuc": "Evet", "gosterge": "Var"})
KONU06 = CustomChoices(roles={"sonuc": "Aylık bağış (TL)", "aciklayici": "Hane geliri (bin TL)"},
                       extra=("Yaş", "Hane büyüklüğü", "Kentte yaşıyor (1/0)"))


def _texts(module) -> str:
    """Alternatifin bütün metinleri: açıklamalar, çıkarımlar ve kod notları."""

    spec = module.alternative()
    return " ".join(part for step in spec.steps for part in (step.explanation, step.takeaway, step.code_note or ""))


def _hansen(dataset: str) -> pd.DataFrame:
    value = os.environ.get(f"IKT807_HANSEN_{dataset.upper()}_PATH")
    if not value or not Path(value).is_file():
        pytest.skip(f"Gerçek Hansen {DATASET_MEMBERS[dataset]} dosyası verilmedi.")
    loaded = load_from_upload(dataset, Path(value).read_bytes(), Path(value).name)
    assert loaded.matches_hansen
    return loaded.frame


def _matches(value: float, expected: float, decimals: int) -> bool:
    return abs(value - expected) <= 0.5 * 10 ** (-decimals) + 1e-12


def _expected(module, step: int, label: str) -> tuple[float, int]:
    check = next(check for item in module.alternative().steps if item.number == step for check in item.checks
                 if check.label == label)
    return check.expected, check.decimals


@pytest.fixture(scope="module")
def chj() -> pd.DataFrame:
    return _hansen("chj2004")


@pytest.fixture(scope="module")
def ck() -> pd.DataFrame:
    return _hansen("ck1994")


@pytest.fixture(scope="module")
def card() -> pd.DataFrame:
    return _hansen("card1995")


# --- Konu 5 alternatifi: Cox, Hansen ve Jimenez (2004) -------------------------------------------------------

def _chj_design(chj: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int], pd.DataFrame]:
    """Sabit, yaş, eğitim kuklaları (1–5; referans ilkokulu bitirmemiş), kadın, evli, hane büyüklüğü, gelir (10 bin
    peso) ve bölge kuklaları (1–13; referans 0). Uygulamanın kodundan ayrı kurulur."""

    data = chj.astype("float64").copy()
    data["egitim"] = (data["primary"] + 2 * data["somesecondary"] + 3 * data["secondary"]
                      + 4 * data["someuniversity"] + 5 * data["university"])
    columns = {"sabit": np.ones(len(data)), "age": data["age"].to_numpy()}
    columns.update({f"egitim={k}": (data["egitim"] == k).to_numpy(float) for k in range(1, 6)})
    columns.update({"female=1": data["female"].to_numpy(), "married=1": data["married"].to_numpy(),
                    "size": data["size"].to_numpy(), "gelir": (data["income"] / 10000).to_numpy()})
    columns.update({f"region={r}": (data["region"] == r).to_numpy(float) for r in range(1, 14)})
    names = list(columns)
    design = np.column_stack([columns[name] for name in names])
    y = (data["tabroad"] > 0).to_numpy(float)
    return design, y, {name: i for i, name in enumerate(names)}, data


def _links(link: str):
    if link == "logit":
        return stats.logistic.cdf, stats.logistic.pdf
    return stats.norm.cdf, stats.norm.pdf


def _ame(beta: np.ndarray, design: np.ndarray, index: dict[str, int], term: str, link: str) -> float:
    """AME: sürekli terimde türevin, kukla grubunda referansa göre olasılık farkının örneklem ortalaması."""

    G, g = _links(link)
    if "=" not in term:
        return float(np.mean(g(design @ beta)) * beta[index[term]])
    variable = term.split("=")[0]
    group = [i for name, i in index.items() if name.startswith(variable + "=")]
    base = design.copy()
    base[:, group] = 0.0
    treated = base.copy()
    treated[:, index[term]] = 1.0
    return float(np.mean(G(treated @ beta) - G(base @ beta)))


def _ame_se(beta: np.ndarray, covariance: np.ndarray, design: np.ndarray, index: dict[str, int], term: str,
            link: str) -> float:
    """Delta yöntemi, merkezi sayısal türevle (uygulamanın analitik türevinden bağımsız)."""

    jacobian = np.empty(len(beta))
    for j in range(len(beta)):
        step = 1e-6 * max(1.0, abs(beta[j]))
        up, down = beta.copy(), beta.copy()
        up[j] += step
        down[j] -= step
        jacobian[j] = (_ame(up, design, index, term, link) - _ame(down, design, index, term, link)) / (2 * step)
    return float(np.sqrt(jacobian @ covariance @ jacobian))


def _binary(design: np.ndarray, y: np.ndarray, link: str):
    model = (sm.Logit if link == "logit" else sm.Probit)(y, design)
    return model.fit(disp=False, method="newton", maxiter=100, tol=1e-12, cov_type="HC0")


def test_konu05_alternative_numbers_follow_chj2004(chj: pd.DataFrame) -> None:
    design, y, index, data = _chj_design(chj)
    assert len(y) == 8684 and round(y.mean(), 3) == 0.204
    values: dict[tuple[int, str], float] = {(1, "Analiz örneklemi (N)"): float(len(y))}
    # LPM, HC1
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    bread = np.linalg.inv(design.T @ design)
    scores = design * residual[:, None]
    n, k = design.shape
    hc1 = np.sqrt(np.diag(bread @ scores.T @ scores @ bread * n / (n - k)))
    values.update({(1, "LPM yaş"): beta[index["age"]], (1, "LPM yaş HC1 SH"): hc1[index["age"]],
                   (1, "LPM kadın hane reisi"): beta[index["female=1"]],
                   (1, "LPM kadın hane reisi HC1 SH"): hc1[index["female=1"]]})
    fits = {link: _binary(design, y, link) for link in ("logit", "probit")}
    for link, title in (("logit", "Logit"), ("probit", "Probit")):
        b, v = np.asarray(fits[link].params), np.asarray(fits[link].cov_params())
        for term, label in (("age", "yaş"), ("female=1", "kadın hane reisi")):
            values[(1, f"{title} AME {label}")] = _ame(b, design, index, term, link)
            values[(1, f"{title} AME {label} SH")] = _ame_se(b, v, design, index, term, link)
        for term, label in K5.ALT_TERMS:
            values[(4, f"{title} AME: {label}")] = _ame(b, design, index, term, link)
            values[(4, f"{title} SH: {label}")] = _ame_se(b, v, design, index, term, link)
    logit, probit = (np.asarray(fits[link].params) for link in ("logit", "probit"))
    scale_logit = float(np.mean(stats.logistic.pdf(design @ logit)))
    scale_probit = float(np.mean(stats.norm.pdf(design @ probit)))
    values.update({(2, "Logit yaş katsayısı"): logit[index["age"]], (2, "Ortalama Λ(1−Λ)"): scale_logit,
                   (2, "Probit yaş katsayısı"): probit[index["age"]], (2, "Ortalama φ"): scale_probit,
                   (2, "Logit/Probit katsayı oranı"): logit[index["age"]] / probit[index["age"]]})
    profile = {}
    for age in K5.ALT_AGES:
        moved = design.copy()
        moved[:, index["age"]] = age
        profile[age] = float(np.mean(stats.logistic.cdf(moved @ logit)))
    values.update({(3, "Yaş = 20 için ortalama olasılık"): profile[20.0],
                   (3, "Yaş = 80 için ortalama olasılık"): profile[80.0]})
    derivative = logit[index["female=1"]] * scale_logit
    finite = _ame(logit, design, index, "female=1", "logit")
    values.update({(5, "Kadın hane reisi, türev tabanlı"): derivative, (5, "Kadın hane reisi, sonlu fark"): finite})
    spec = K5.alternative()
    labels = {(step.number, check.label) for step in spec.steps for check in step.checks}
    assert labels == set(values)
    for (step, label), value in values.items():
        expected, decimals = _expected(K5, step, label)
        assert _matches(value, expected, decimals), (label, value, expected)
    # Stata vce(robust): Logit/Probit kovaryansı ayrıca n/(n−1) ile çarpılır; standart hata kontrolleri yine geçmeli.
    factor = np.sqrt(n / (n - 1))
    for (step, label), value in values.items():
        if label.endswith(" SH") and "LPM" not in label or " SH: " in label:
            expected, decimals = _expected(K5, step, label)
            assert _matches(value * factor, expected, decimals), label
    # Metinlerdeki sayılar
    married_female = data[(data["female"] == 1) & (data["married"] == 1)]
    single_female = data[(data["female"] == 1) & (data["married"] == 0)]
    married_male = data[(data["female"] == 0) & (data["married"] == 1)]
    rates = [round(100 * float((group["tabroad"] > 0).mean()), 1) for group in (married_female, single_female,
                                                                              married_male)]
    assert rates == [72.0, 24.4, 16.4]
    assert (round(float(np.diff([profile[20.0], profile[30.0]])[0]), 3),
            round(profile[80.0] - profile[70.0], 3)) == (0.026, 0.044)
    assert all(np.diff([profile[a] for a in K5.ALT_AGES], 2) > 0)  # dışbükey
    fitted = stats.logistic.cdf(design @ logit)
    assert round(float(np.mean(fitted < 0.5)), 2) == 0.95 and round(float(fitted.mean()), 2) == 0.20
    assert round(logit[index["female=1"]], 2) == 1.68 and round(finite - derivative, 4) == 0.0618
    assert round(stats.norm.pdf(0) / 0.25, 2) == 1.60
    assert round(stats.norm.pdf(stats.norm.ppf(0.2)) / 0.16, 2) == 1.75
    assert sorted(data["region"].unique()) == list(range(14))
    single_male = data[(data["female"] == 0) & (data["married"] == 0)]
    assert round(100 * float((single_male["tabroad"] > 0).mean()), 1) == 22.4
    assert round(1 - float(data["female"].mean()), 2) == 0.83
    assert (data["marriedf"] == data["female"] * data["married"]).all()
    interacted = _binary(np.column_stack([design, data["marriedf"].to_numpy()]), y, "logit")
    assert (round(float(interacted.params[index["female=1"]]), 2), round(float(interacted.params[-1]), 2)) == (0.11, 2.34)
    assert round(scale_probit / scale_logit, 2) == 1.77
    notes = {(step.number, check.label): check.expected for step in LABS["konu05"].steps for check in step.checks}
    assert (notes[(5, "Siyah, türev tabanlı")], notes[(5, "Siyah, sonlu fark")]) == (-0.1598, -0.1617)
    assert notes[(2, "Logit/Probit katsayı oranı")] == 1.65
    assert (round(notes[(3, "18 yaşında ortalama olasılık")], 2), round(notes[(3, "35 yaşında ortalama olasılık")], 2)) \
        == (0.11, 0.82)
    texts = _texts(K5)
    for number in ("8.684", "%20,4", "0,00336", "0,34 yüzde puan", "0,00325", "0,00340", "0,2991", "0,2925", "0,2824",
                   "%72,0", "%24,4", "%16,4", "0,02370", "0,1416", "0,01357", "0,2501", "1,75", "1,65", "1,60",
                   "0,130", "0,341", "0,026", "0,044", "0,11'den 0,82'ye", "0,0288", "0,2062", "−0,0097", "0,0013",
                   "0,2373", "0,0618", "0,0019", "−0,1598", "−0,1617", "1,68", "%95", "30 yüzde puan",
                   "28 yüzde puan", "14 bölgenin 13", "age=(20(1)80)", "%22,4", "1,68'den 0,11'e", "2,34",
                   "0,2501/0,1416 ≈ 1,77", "%83", "üst %2"):
        assert number in texts, number


# --- Konu 6 alternatifi: Card ve Krueger (1994) --------------------------------------------------------------

def _ck_sample(ck: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    data = ck.astype("float64").dropna(subset=["empft", "hoursopen"]).reset_index(drop=True)
    hours = data["hoursopen"]
    columns = {"sabit": np.ones(len(data)), "hoursopen": hours, "saat12": np.maximum(hours - 12, 0),
               "saat16": np.maximum(hours - 16, 0), "kfc": data["chain"] == 2, "roys": data["chain"] == 3,
               "wendys": data["chain"] == 4, "co_owned": data["co_owned"], "state": data["state"],
               "time": data["time"]}
    names = list(columns)
    design = np.column_stack([np.asarray(columns[name], dtype=float) for name in names])
    return data, design, data["empft"].to_numpy(), names


def _grid_design(design: np.ndarray) -> np.ndarray:
    hours = np.asarray(K6.ALT_GRID)
    rows = np.tile(design.mean(axis=0), (len(hours), 1))
    rows[:, 1], rows[:, 2], rows[:, 3] = hours, np.maximum(hours - 12, 0), np.maximum(hours - 16, 0)
    return rows


def _lad_primal(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LAD, primal doğrusal program (uygulama dual problemi çözer): min Σ(u + v)/2, Xβ + u − v = y."""

    n, k = design.shape
    from scipy import sparse

    equality = sparse.hstack([sparse.csr_matrix(design), sparse.identity(n), -sparse.identity(n)], format="csr")
    result = optimize.linprog(np.concatenate([np.zeros(k), np.full(2 * n, 0.5)]), A_eq=equality, b_eq=y,
                              bounds=[(None, None)] * k + [(0, None)] * (2 * n), method="highs")
    assert result.status == 0
    return result.x[:k]


def _tobit(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Tobit (soldan 0'da sansür) MLE, genel amaçlı optimizasyonla; parametreler (β, log σ)."""

    positive = y > 0

    def negative_loglik(theta: np.ndarray) -> float:
        beta, sigma = theta[:-1], np.exp(theta[-1])
        index = design @ beta
        return -float(np.sum(stats.norm.logpdf((y[positive] - index[positive]) / sigma) - np.log(sigma))
                      + np.sum(stats.norm.logcdf(-index[~positive] / sigma)))

    start = np.linalg.lstsq(design, y, rcond=None)[0]
    theta = np.concatenate([start, [np.log(np.std(y - design @ start))]])
    for _ in range(3):
        theta = optimize.minimize(negative_loglik, theta, method="BFGS", options={"gtol": 1e-9, "maxiter": 10000}).x
    return theta[:-1], float(np.exp(theta[-1]))


def test_konu06_alternative_numbers_follow_ck1994(ck: pd.DataFrame) -> None:
    assert len(ck) == 820
    closed = ck[(ck["time"] == 1) & (ck["empft"] == 0) & (ck["emppt"] == 0) & (ck["nmgrs"] == 0)]
    assert len(closed) == 6 and closed["hoursopen"].isna().all()
    data, design, y, names = _ck_sample(ck)
    assert len(data) == 795 and int((y == 0).sum()) == 144
    assert list(np.quantile(data["hoursopen"], (0.25, 0.75))) == [12.0, 16.0]
    values: dict[tuple[int, str], float] = {
        (1, "Gözlem sayısı (restoran × tur)"): len(y), (1, "Sıfır payı (%)"): 100 * float(np.mean(y == 0)),
        (1, "Ortalama tam zamanlı çalışan"): float(y.mean()), (1, "Medyan tam zamanlı çalışan"): float(np.median(y)),
    }
    grid = _grid_design(design)
    ols = np.linalg.lstsq(design, y, rcond=None)[0]
    lad = _lad_primal(design, y)
    tobit, sigma = _tobit(design, y)
    for label, beta in (("OLS", ols), ("Tobit gizli ortalama", tobit), ("LAD medyan", lad)):
        for hour, value in zip(K6.ALT_GRID, grid @ beta):
            values[(2, f"{label}, saat {int(hour)}")] = value
    latent = grid @ tobit
    z = latent / sigma
    targets = {"P(Y>0|x)": stats.norm.cdf(z), "m(x)": stats.norm.cdf(z) * latent + sigma * stats.norm.pdf(z),
               "m#(x)": latent + sigma * stats.norm.pdf(z) / stats.norm.cdf(z)}
    for label, column in targets.items():
        for hour, value in zip(K6.ALT_GRID, column):
            values[(3, f"{label}, saat {int(hour)}")] = value
    everyone = design @ tobit / sigma
    values.update({
        (3, "Model: ortalama P(Y>0)"): float(np.mean(stats.norm.cdf(everyone))),
        (3, "Veri: tam zamanlı çalışanı olanların payı"): float(np.mean(y > 0)),
        (3, "Model: ortalama E[Y]"): float(np.mean(stats.norm.cdf(everyone) * (design @ tobit)
                                                   + sigma * stats.norm.pdf(everyone))),
        (3, "Veri: ortalama tam zamanlı çalışan"): float(y.mean()),
    })
    spec = K6.alternative()
    assert {(step.number, check.label) for step in spec.steps for check in step.checks} == set(values)
    for (step, label), value in values.items():
        expected, decimals = _expected(K6, step, label)
        assert _matches(value, expected, decimals), (label, value, expected)
    # LAD çözümü tektir: dual problemin optimal köşe çözümü dejenere değil (tam k tane dual değişken sınırların
    # içinde), dolayısıyla tümleyici gevşeklik β'yı tek belirler (diller aynı köşe çözümünü verir).
    n, k = design.shape
    dual = optimize.linprog(-y, A_eq=design.T, b_eq=0.5 * design.sum(axis=0), bounds=(0, 1), method="highs-ds")
    inside = (dual.x > 1e-7) & (dual.x < 1 - 1e-7)
    assert dual.status == 0 and int(inside.sum()) == k
    assert np.allclose(np.linalg.solve(design[inside], y[inside]), lad, atol=1e-8)
    # Metinlerdeki sayılar
    slopes = (ols[1], ols[1] + ols[2], ols[1] + ols[2] + ols[3])
    assert [round(s, 1) for s in slopes] == [1.5, 0.7, 1.8]
    for beta in (ols, tobit, lad):
        assert beta[1] + beta[2] + beta[3] > beta[1] + beta[2] > 0  # 16 saatten sonra dikleşir
    fine = np.tile(design.mean(axis=0), (141, 1))
    hours = np.linspace(10, 24, 141)
    fine[:, 1], fine[:, 2], fine[:, 3] = hours, np.maximum(hours - 12, 0), np.maximum(hours - 16, 0)
    assert all(fine @ lad < fine @ ols)  # 10–24 saat aralığının her noktasında
    assert K6.alternative_plan().plot_grid == (10.0, 24.0, 141)
    assert int((data["hoursopen"] == 24).sum()) == 11 and not ((data["hoursopen"] > 19) & (data["hoursopen"] < 24)).any()
    zero = y == 0
    part_time = data["emppt"]
    assert (round(float(part_time[zero].mean()), 1), round(float(part_time[~zero].mean()), 1)) == (25.1, 17.5)
    notes = {(step.number, check.label): check.expected for step in LABS["konu06"].steps for check in step.checks}
    assert (notes[(3, "Model: ortalama P(Y>0)")], notes[(3, "Veri: pozitif transfer payı")],
            notes[(3, "Model: ortalama E[Y] (bin peso)")], notes[(3, "Veri: ortalama transfer (bin peso)")]) == \
        (0.579, 0.824, 11.46, 7.71)
    texts = _texts(K6)
    for number in ("820", "795", "6 restoran", "`status2` = 3", "%18,1", "8,29", "medyan 6", "12 ve 16 saatte",
                   "yaklaşık 1,5 çalışanla", "12–16 saat arasında yaklaşık 0,7", "yaklaşık 1,8", "11 restorana",
                   "3,45", "6,38", "9,19", "23,38", "2,80", "21,21", "8,32",
                   "9,20", "11,21", "0,591", "0,994", "0,778", "0,819", "8,56", "0,579'a karşı 0,824",
                   "11,46'ya karşı 7,71", "25,1", "17,5", "4,25 dolardan 5,05 dolara"):
        assert number in texts, number


# --- Alternatiflerin kodu sentetik veriyle (gerçek veri gerekmez) ------------------------------------------------

def _synthetic_chj2004() -> pd.DataFrame:
    """CHJ2004 düzeninde kurgusal hane verisi: 1.500 hane; eğitim kuklaları birbirini dışlar, 14 bölge."""

    rng = np.random.default_rng(6)
    n = 1500
    level = rng.integers(0, 6, n)
    frame = pd.DataFrame({name: (level == k).astype(int) for k, name in
                          enumerate(("primary", "somesecondary", "secondary", "someuniversity", "university"), start=1)})
    age = rng.integers(20, 81, n)
    female = (rng.random(n) < 0.2).astype(int)
    married = (rng.random(n) < 0.8).astype(int)
    size = rng.integers(1, 10, n)
    income = np.round(np.exp(rng.normal(10.5, 0.7, n)))
    region = rng.integers(0, 14, n)
    index = -3.0 + 0.02 * age + 0.15 * level + 1.2 * female + 0.4 * married + 0.03 * size - 0.000002 * income
    tabroad = np.where(rng.random(n) < 1 / (1 + np.exp(-index)), np.round(rng.exponential(20000, n)), 0.0)
    frame = frame.assign(age=age, female=female, married=married, size=size, income=income, region=region,
                         tabroad=tabroad, tdomestic=0.0, tinkind=0.0, tgifts=0.0, transfers=tabroad, marriedf=0,
                         child1=0, child7=0, child15=0, bothwork=0, notemployed=0)
    return frame


def _synthetic_ck1994() -> pd.DataFrame:
    """CK1994 düzeninde kurgusal restoran × tur verisi: 300 restoran, iki tur; birkaç eksik kayıt ve kapanan restoran."""

    rng = np.random.default_rng(7)
    stores = 300
    chain = rng.integers(1, 5, stores)
    owned = rng.integers(0, 2, stores)
    hours = 8 + 16 * rng.beta(2.5, 2.5, stores)  # sürekli: LAD çözümü tek olsun (bağlı tasarım satırı yok)
    rows = []
    for time in (0, 1):
        latent = -8 + 1.1 * hours - 2 * (chain == 2) + rng.normal(0, 6, stores)
        empft = np.maximum(latent, 0)
        rows.append(pd.DataFrame({"store": np.arange(1, stores + 1), "chain": chain, "co_owned": owned,
                                  "state": (np.arange(stores) % 5 != 0).astype(int), "empft": empft,
                                  "emppt": rng.integers(5, 30, stores).astype(float), "nmgrs": 3.0,
                                  "wage_st": 4.5, "hoursopen": hours.copy(), "nregisters": 3.0, "time": time}))
    frame = pd.concat(rows, ignore_index=True)
    frame.loc[[5, 310], "empft"] = np.nan
    frame.loc[[320, 321], ["empft", "emppt", "nmgrs"]] = 0.0
    frame.loc[[320, 321], "hoursopen"] = np.nan  # kapanan restoranlar
    return frame


@pytest.mark.parametrize("module, dataset, build", [(K5, "chj2004", _synthetic_chj2004),
                                                    (K6, "ck1994", _synthetic_ck1994)], ids=["konu05", "konu06"])
def test_alternative_code_reproduces_the_app_on_synthetic_data(module, dataset: str, build, tmp_path: Path) -> None:
    frame = build()
    spec = with_app_values(module.alternative(), {dataset: frame})
    path = tmp_path / DATASET_MEMBERS[dataset]
    frame.to_stata(path, write_index=False, version=118)
    for language in ("Python", "R") if RSCRIPT else ("Python",):
        before, after = {"Python": ("YEREL_DOSYA = None", f'YEREL_DOSYA = "{path.as_posix()}"'),
                         "R": ("yerel_dosya <- NULL", f'yerel_dosya <- "{path.as_posix()}"')}[language]
        script = render_script(spec, language)
        assert before in script
        _check_script(spec, language, script.replace(before, after, 1), tmp_path)


# --- Kendi verin: ortak araçlar ---------------------------------------------------------------------------

def _csv(frame: pd.DataFrame) -> bytes:
    """Türkçe Excel'in "CSV (noktalı virgülle ayrılmış)" kaydı: R betiği readxl istemeden okur."""

    return frame.to_csv(sep=";", decimal=",", index=False).encode("cp1254")


def _case(module, frame: pd.DataFrame, choices: CustomChoices):
    return custom_case(module.CUSTOM, K.read_upload(CSV, _csv(frame)), choices)[0]


def _own(module, frame: pd.DataFrame, choices: CustomChoices):
    data = _csv(frame)
    case, _ = custom_case(module.CUSTOM, K.read_upload(CSV, data), choices)
    return module.CUSTOM.build(case), (CSV, data)


def _check_script(spec, language: str, script: str, folder: Path) -> None:
    path = folder / script_filename(spec, language)
    path.write_text(script, encoding="utf-8")
    command = [sys.executable] if language == "Python" else [RSCRIPT]
    result = subprocess.run(command + [str(path)], cwd=folder, capture_output=True, encoding="utf-8",
                            errors="replace", timeout=900,
                            env=dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="cp1254", LANG="C.UTF-8",
                                     LC_ALL="C.UTF-8"))
    assert result.returncode == 0, language + result.stdout[-1500:] + result.stderr[-1500:]
    assert result.stdout.count("  OK   ") == sum(len(step.checks) for step in spec.steps), language
    assert "Bütün değerler uygulamadaki sonuçlarla uyuşuyor." in result.stdout, language


def _reproduces_in_both_languages(spec, data: tuple[str, bytes], folder: Path) -> None:
    (folder / data[0]).write_bytes(data[1])
    for language in ("Python", "R") if RSCRIPT else ("Python",):
        _check_script(spec, language, render_script(spec, language), folder)


def _notes(spec, run=None) -> dict[int, str]:
    run = run_lab(spec) if run is None else run
    assert all(item.passed for items in run.checks.values() for item in items)
    return {step.number: step.note_for(run.state) for step in spec.steps if step.note_for is not None}


# --- Kendi verin: Konu 5 ----------------------------------------------------------------------------------

def test_konu05_indicator_is_numeric_with_a_finite_difference_in_every_language() -> None:
    spec, _ = _own(K5, K5.sample(), KONU05)
    operations = [op for step in spec.steps for op in step.operations]
    models = [op for op in operations if isinstance(op, (OLS, BinaryChoice))]
    assert models and all(K5.D01 in op.regressors and not op.categorical for op in models)
    effects = [op for op in operations if isinstance(op, MarginalEffects)]
    binary = (K5.D01, "kadin_1_0", "il_merkezinde_yasiyor_1")  # gösterge ve yalnız 0/1 değerli kontroller
    assert effects and all(op.discrete == binary for op in effects)
    python, r = render_script(spec, "Python"), render_script(spec, "R")
    assert f'kesikli=["{K5.D01}", "kadin_1_0", "il_merkezinde_yasiyor_1"]' in python and f"C({K5.D01})" not in python
    assert f'kesikli = c("{K5.D01}", "kadin_1_0", "il_merkezinde_yasiyor_1")' in r and f"factor({K5.D01})" not in r
    run = run_lab(spec)
    frame = run.state.frames["veri"]
    logit = run.state.models["logit"]
    columns = ["Intercept", *[name for name in logit.params.index if name != "Intercept"]]
    data = np.column_stack([np.ones(len(frame)), frame[columns[1:]].to_numpy(dtype=float)])
    beta = logit.params[columns].to_numpy()
    for name in binary:  # AME = sonlu fark (türev değil)
        one, zero = data.copy(), data.copy()
        one[:, columns.index(name)], zero[:, columns.index(name)] = 1.0, 0.0
        finite = float(np.mean(stats.logistic.cdf(one @ beta) - stats.logistic.cdf(zero @ beta)))
        assert abs(run.state.models["ame_logit"].params[name] - finite) < 1e-10, name
    labels = [check.label for check in spec.steps[4].checks]
    assert labels == ["“Meslek kursu” = “Var”, türev tabanlı", "“Meslek kursu” = “Var”, sonlu fark"]
    assert K5._cap("ilkokul") == "İlkokul" and K5._cap("ılık") == "Ilık" and K5._cap("“a” = “B”") == "“a” = “B”"
    notes = _notes(spec)
    assert "1.178 gözlem" in notes[1] and "“İşe yerleşti” = “Evet”" in notes[1]


def test_konu05_role_rules_are_enforced() -> None:
    frame = K5.sample()
    with pytest.raises(K.UploadError, match="üç farklı sütun"):
        _case(K5, frame, CustomChoices(roles={**KONU05.roles, "gosterge": "İşe yerleşti"}, extra=KONU05.extra,
                                       picks=KONU05.picks))
    rare = frame.assign(**{"İşe yerleşti": np.where(np.arange(len(frame)) < 8, "Evet", "Hayır")})
    with pytest.raises(K.UploadError, match="en az 10 gözlem"):
        _case(K5, rare, KONU05)
    coarse = frame.assign(Yaş=frame["Yaş"] // 15)
    with pytest.raises(K.UploadError, match="en az 5 farklı değer"):
        _case(K5, coarse, KONU05)
    few = frame.assign(**{"Meslek kursu": np.where(np.arange(len(frame)) < 3, "Var", "Yok")})
    with pytest.raises(K.UploadError, match="en az 5 gözlem"):
        _case(K5, few, KONU05)
    separated = frame.assign(Sinyal=np.where(frame["İşe yerleşti"] == "Evet", 1.0, 0.0))
    with pytest.raises(K.UploadError, match="ayrışma"):
        _case(K5, separated, CustomChoices(roles=KONU05.roles, extra=("Sinyal",), picks=KONU05.picks))
    collinear = frame.assign(Kopya=2 * frame["Kadın (1/0)"] + 1)
    with pytest.raises(K.UploadError, match="tam doğrusal bağlantı"):
        _case(K5, collinear, CustomChoices(roles=KONU05.roles, extra=("Kadın (1/0)", "Kopya"), picks=KONU05.picks))


def test_konu05_without_controls_and_with_the_other_category_reproduces(tmp_path: Path) -> None:
    choices = CustomChoices(roles=KONU05.roles, picks={"sonuc": "Hayır", "gosterge": "Yok"})
    spec, data = _own(K5, K5.sample(), choices)
    notes = _notes(spec)
    assert "Ek kontrol seçilmedi" in spec.steps[0].explanation and "“İşe yerleşti” = “Hayır”" in notes[1]
    _reproduces_in_both_languages(spec, data, tmp_path)


# --- Kendi verin: Konu 6 ----------------------------------------------------------------------------------

def test_konu06_sample_keeps_the_lad_checks_and_quartile_knots() -> None:
    frame = K6.sample()
    case = _case(K6, frame, KONU06)
    plan = K6.own_plan(case)
    incomes = case.data[case.roles["aciklayici"]]
    assert plan.knots == tuple(float(v) for v in np.round(np.quantile(incomes, (0.25, 0.5, 0.75)), 2))
    assert plan.grid[0] < plan.knots[0] and plan.grid[-1] > plan.knots[-1]
    spread, checked = K6.lad_guard(case.data, plan)
    assert spread < 1e-6 and checked
    spec = K6.build(case)
    labels = [check.label for check in spec.steps[1].checks]
    assert sum(label.startswith("LAD medyan") for label in labels) == len(plan.grid)
    notes = _notes(spec)
    assert "%24,1" in notes[1] and "LAD çözümü tek değil" not in notes[2]


def _tied_counts(seed: int = 4) -> pd.DataFrame:
    """Küçük, bağlı değerli sayım verisi: LAD çözümü tek değildir (R "nonunique" uyarısı verir)."""

    rng = np.random.default_rng(seed)
    n = 120
    x = rng.integers(1, 13, n)
    group = (rng.random(n) < 0.5).astype(int)
    y = np.maximum(0, np.round(0.3 * x + group + rng.normal(0, 1.0, n) - 1))
    return pd.DataFrame({"Sayı": y, "Puan": x, "Grup (1/0)": group})


def test_konu06_a_nonunique_lad_solution_is_shown_but_not_compared(tmp_path: Path) -> None:
    frame = _tied_counts()
    choices = CustomChoices(roles={"sonuc": "Sayı", "aciklayici": "Puan"}, extra=("Grup (1/0)",))
    case = _case(K6, frame, choices)
    plan = K6.own_plan(case)
    spread, checked = K6.lad_guard(case.data, plan)
    assert spread > 0.5 and not checked
    spec, data = _own(K6, frame, choices)
    labels = [check.label for check in spec.steps[1].checks]
    assert labels and not any(label.startswith("LAD medyan") for label in labels)
    run = run_lab(spec)
    assert "lad" in run.state.tables["egriler"].columns  # eğri yine gösterilir
    assert "LAD çözümü tek değil" in _notes(spec, run)[2]
    _reproduces_in_both_languages(spec, data, tmp_path)


def test_fitted_value_range_finds_the_set_of_medians() -> None:
    design = np.ones((4, 1))
    assert np.allclose(Q.fitted_value_range(design, np.array([1.0, 2.0, 3.0, 4.0]), 0.5, design[:1]), [[2.0, 3.0]])
    unique = Q.fitted_value_range(design[:3], np.array([1.0, 2.0, 7.0]), 0.5, design[:1])
    assert np.allclose(unique, [[2.0, 2.0]], atol=1e-7)


def test_konu06_role_rules_are_enforced() -> None:
    frame = K6.sample()
    outcome = "Aylık bağış (TL)"
    with pytest.raises(K.UploadError, match="negatif değer"):
        _case(K6, frame.assign(**{outcome: frame[outcome] - 50}), KONU06)
    few = frame.assign(**{outcome: np.where(frame[outcome] == 0, 10.0, frame[outcome])})
    few.loc[:9, outcome] = 0.0
    with pytest.raises(K.UploadError, match="en az %5"):
        _case(K6, few, KONU06)
    many = frame.assign(**{outcome: np.where(np.arange(len(frame)) % 3 == 0, frame[outcome], 0.0)})
    with pytest.raises(K.UploadError, match="en çok %50"):
        _case(K6, many, KONU06)
    coarse = frame.assign(**{"Hane geliri (bin TL)": frame["Hane geliri (bin TL)"] // 20})
    with pytest.raises(K.UploadError, match="en az 10 farklı değer"):
        _case(K6, coarse, KONU06)
    with pytest.raises(K.UploadError, match="iki farklı sütun|hem bir rol için hem ek sütun"):
        _case(K6, frame, CustomChoices(roles={"sonuc": outcome, "aciklayici": outcome}))
    collinear = frame.assign(Kopya=frame["Yaş"] * 2 + 3)
    with pytest.raises(K.UploadError, match="tam doğrusal bağlantı"):
        _case(K6, collinear, CustomChoices(roles=KONU06.roles, extra=("Yaş", "Kopya")))


# --- Ölçek ve sınır durumları -----------------------------------------------------------------------------

def test_konu05_a_large_scale_profile_gets_more_decimals_and_r_finds_scientific_row_names(tmp_path: Path) -> None:
    """TL cinsinden gelir (10.000–100.000): katsayılar küçük olduğu için ondalık artar; R 100000 satırını "1e+05" adıyla
    tutar, betik o adla arar."""

    frame = K5.sample()
    frame["Gelir (TL)"] = np.random.default_rng(9).integers(1, 11, len(frame)) * 10000.0
    choices = CustomChoices(roles={**KONU05.roles, "profil": "Gelir (TL)"}, extra=KONU05.extra, picks=KONU05.picks)
    spec, data = _own(K5, frame, choices)
    assert K5.own_plan(_case(K5, frame, choices)).decimals == 8
    assert spec.steps[2].checks[-1].label == "“Gelir (TL)” = 100.000 için ortalama olasılık"
    assert 'profil["1e+05", "olasilik"]' in render_script(spec, "R")
    huge = frame.assign(**{"Gelir (TL)": frame["Gelir (TL)"] * 10})
    with pytest.raises(K.UploadError, match="birimini değiştirin"):
        _case(K5, huge, choices)
    _reproduces_in_both_languages(spec, data, tmp_path)


def test_konu05_a_birth_year_profile_is_not_mistaken_for_separation(tmp_path: Path) -> None:
    frame = K5.sample()
    frame["Doğum yılı"] = 2024 - frame["Yaş"]
    choices = CustomChoices(roles={**KONU05.roles, "profil": "Doğum yılı"}, extra=KONU05.extra, picks=KONU05.picks)
    spec, data = _own(K5, frame, choices)
    _reproduces_in_both_languages(spec, data, tmp_path)


def _heaped_incomes() -> pd.DataFrame:
    """Gelir 50.000'in katlarında yığılı (anket verisinde sık): çeyrekler 100.000, 150.000, 200.000 gibi değerlerdir."""

    frame = K6.sample()
    rng = np.random.default_rng(12)
    income = np.round(frame["Hane geliri (bin TL)"] * 6) * 1000.0
    income = np.where(rng.random(len(frame)) < 0.6, np.round(income / 50000) * 50000, income)
    return frame.assign(**{"Hane geliri (bin TL)": np.maximum(income, 10000.0)}).rename(
        columns={"Hane geliri (bin TL)": "Hane geliri (TL)"})


def test_konu06_round_income_knots_are_found_by_name_in_r(tmp_path: Path) -> None:
    frame = _heaped_incomes()
    choices = CustomChoices(roles={"sonuc": "Aylık bağış (TL)", "aciklayici": "Hane geliri (TL)"},
                            extra=KONU06.extra)
    plan = K6.own_plan(_case(K6, frame, choices))
    assert any(r_row_name(value).endswith("e+05") for value in plan.grid)
    spec, data = _own(K6, frame, choices)
    _reproduces_in_both_languages(spec, data, tmp_path)


def test_konu06_a_large_outcome_is_scaled_for_survreg(tmp_path: Path) -> None:
    """Standart sapması 10⁶ düzeyindeki sonuçta R ``survreg`` katsayıları tekil sayabilir; kod sonucu 10'un kuvvetine
    bölüp β, σ ve doğrusal indeksi geri ölçekler (Tobit ölçekle eşdeğişkendir)."""

    frame = K6.sample()
    frame["Aylık bağış (TL)"] = frame["Aylık bağış (TL)"] * 10000
    spec, data = _own(K6, frame, KONU06)
    tobit = next(op for step in spec.steps for op in step.operations if type(op).__name__ == "Tobit")
    assert tobit.scale == 1_000_000
    code = render_script(spec, "R")
    assert "I(aylik_bagis_tl / 1000000)" in code and "tobit$linear.predictors <- tobit$linear.predictors * 1000000" in code
    small, _ = _own(K6, K6.sample(), KONU06)
    assert "I(aylik_bagis_tl" not in render_script(small, "R")
    _reproduces_in_both_languages(spec, data, tmp_path)


def test_konu06_a_near_deterministic_outcome_is_rejected() -> None:
    frame = K6.sample()
    income = frame["Hane geliri (bin TL)"]
    frame["Aylık bağış (TL)"] = np.maximum(0.0, 20.0 * (income - income.quantile(0.2)))
    with pytest.raises(K.UploadError, match="Tobit"):
        _case(K6, frame, CustomChoices(roles=KONU06.roles))


def test_konu06_knots_of_a_small_scale_variable_stay_distinct() -> None:
    values = pd.Series(np.random.default_rng(2).uniform(0, 0.01, 500))
    knots = K6._quantile_knots(values)
    assert len(knots) == 3 and all(0.001 < knot < 0.009 for knot in knots)
    assert K6._value(0.0025) == "0,0025" and K6._value(150000.0) == "150.000"


def test_fitted_value_range_is_fast_and_scale_free() -> None:
    import time

    rng = np.random.default_rng(3)
    n = 10_000
    x = rng.integers(1, 30, n).astype(float)
    group = (rng.random(n) < 0.5).astype(float)
    y = np.maximum(0, np.round(0.3 * x + group + rng.normal(0, 2, n) - 1))
    design = np.column_stack([np.ones(n), x, np.maximum(x - 10, 0), np.maximum(x - 20, 0), group])
    points = design[:5]
    start = time.perf_counter()
    for scale in (1.0, 1e7):
        ranges = Q.fitted_value_range(design, y * scale, 0.5, points)
        fitted = points @ Q.rq_fit(design, y * scale, 0.5)
        assert np.all(ranges[:, 0] <= fitted + 1e-6 * scale) and np.all(ranges[:, 1] >= fitted - 1e-6 * scale)
        assert float((ranges[:, 1] - ranges[:, 0]).max()) <= 1e-6 * scale
    assert time.perf_counter() - start < 20


# --- Konu 7 alternatifi: Card (1995), kantil regresyon ----------------------------------------------------------

def _quantile_primal(design: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Kantil regresyon, primal doğrusal program (uygulama dual problemi çözer)."""

    from scipy import sparse

    n, k = design.shape
    equality = sparse.hstack([sparse.csr_matrix(design), sparse.identity(n), -sparse.identity(n)], format="csr")
    cost = np.concatenate([np.zeros(k), np.full(n, tau), np.full(n, 1 - tau)])
    result = optimize.linprog(cost, A_eq=equality, b_eq=y, bounds=[(None, None)] * k + [(0, None)] * (2 * n),
                              method="highs-ds")
    assert result.status == 0
    return result.x[:k]


def _hall_sheather(tau: float, n: int) -> float:
    """Hall–Sheather bant genişliği (R ``bandwidth.rq``, hs = TRUE, α = 0,05)."""

    x0 = stats.norm.ppf(tau)
    f0 = stats.norm.pdf(x0)
    return float(n ** (-1 / 3) * stats.norm.ppf(0.975) ** (2 / 3)
                 * ((1.5 * f0 ** 2) / (2 * x0 ** 2 + 1)) ** (1 / 3))


def _hk(design: np.ndarray, y: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """β̂(τ), H⁻¹ ve X'X; Hendricks–Koenker yoğunlukları τ ± h çözümlerinden (R ``summary.rq(se = "nid")``)."""

    n = len(y)
    h = _hall_sheather(tau, n)
    while tau + h > 1 or tau - h < 0:
        h /= 2
    beta = _quantile_primal(design, y, tau)
    spread = design @ (_quantile_primal(design, y, tau + h) - _quantile_primal(design, y, tau - h))
    density = np.maximum(0.0, 2 * h / (spread - np.finfo(float).eps ** 0.5))
    hinv = np.linalg.inv(design.T @ (density[:, None] * design))
    return beta, hinv, design.T @ design


def test_konu07_alternative_numbers_follow_card1995(card: pd.DataFrame) -> None:
    from core.labs import ornek_konu07 as K7

    data = card.astype("float64").dropna(subset=["lwage76"]).reset_index(drop=True)
    y = np.log(data["wage76"].to_numpy() / 100)
    experience = data["age76"] - data["ed76"] - 6
    design = np.column_stack([np.ones(len(y)), data["ed76"], data["black"], experience, experience ** 2 / 100])
    assert len(y) == 3010
    values: dict[tuple[int, str], float] = {(1, "Analiz örneklemi (N)"): float(len(y))}
    fits = {}
    for tau, name in K7.QUANTILES:
        beta, hinv, xtx = _hk(design, y, tau)
        covariance = tau * (1 - tau) * hinv @ xtx @ hinv
        fits[name] = (tau, beta, hinv, xtx)
        se = np.sqrt(np.diag(covariance))
        title = K7.MODEL_LABELS[name]
        values.update({(1, f"{title}: eğitim"): beta[1], (1, f"{title}: eğitim SH"): se[1],
                       (1, f"{title}: siyahi"): beta[2], (1, f"{title}: siyahi SH"): se[2],
                       (1, f"{title}: deneyim (Tablo 7.1)"): beta[3],
                       (1, f"{title}: deneyim²/100 (Tablo 7.1)"): beta[4]})
    ols = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ ols
    bread = np.linalg.inv(design.T @ design)
    n, k = design.shape
    hc1 = np.sqrt(np.diag(bread @ (design * residual[:, None]).T @ (design * residual[:, None]) @ bread * n / (n - k)))
    values.update({(1, "OLS: eğitim"): ols[1], (1, "OLS: eğitim SH"): hc1[1], (1, "OLS: siyahi"): ols[2],
                   (1, "OLS: siyahi SH"): hc1[2], (1, "OLS: deneyim (Tablo 7.1)"): ols[3],
                   (1, "OLS: deneyim²/100 (Tablo 7.1)"): ols[4],
                   (1, "τ = 0,10 kesin yüzde etki"): 100 * (np.exp(fits["q10"][1][1]) - 1),
                   (1, "τ = 0,90 kesin yüzde etki"): 100 * (np.exp(fits["q90"][1][1]) - 1)})
    (t1, b1, h1, j1), (t2, b2, h2, j2) = fits["q10"], fits["q90"]
    v1, v2 = t1 * (1 - t1) * h1 @ j1 @ h1, t2 * (1 - t2) * h2 @ j2 @ h2
    cross = (min(t1, t2) - t1 * t2) * h1 @ j1 @ h2
    tests = {}
    for index, word in ((1, "Eğitim"), (2, "Siyahi")):
        difference = b2[index] - b1[index]
        error = float(np.sqrt(v1[index, index] + v2[index, index] - 2 * cross[index, index]))
        tests[word] = (difference, error)
        values.update({(2, f"{word}: kantiller arası fark"): difference, (2, f"{word}: farkın SH'si"): error,
                       (2, f"{word}: z istatistiği"): difference / error})
    spec = K7.alternative()
    assert {(step.number, check.label) for step in spec.steps for check in step.checks} == set(values)
    for (step, label), value in values.items():
        expected, decimals = _expected(K7, step, label)
        assert _matches(value, expected, decimals), (label, value, expected)
    # Metinlerdeki sayılar
    p_black = 2 * stats.norm.sf(abs(tests["Siyahi"][0] / tests["Siyahi"][1]))
    assert 0.05 < p_black < 0.10
    assert fits["q50"][1][1] > max(fits["q10"][1][1], fits["q90"][1][1])  # ters U
    assert fits["q25"][1][2] < fits["q10"][1][2] and fits["q25"][1][2] < fits["q50"][1][2] < fits["q75"][1][2] \
        < fits["q90"][1][2] < 0
    assert (round(tests["Eğitim"][0], 3), round(tests["Eğitim"][1], 3)) == (-0.002, 0.008)
    assert [round(fits[name][1][1], 3) for name in ("q10", "q50", "q90")] == [0.077, 0.085, 0.075]
    texts = _texts(K7)
    for number in ("3.010", "0,0774", "0,0853", "0,0749", "%8,0", "%7,8", "−0,2757", "−0,1915", "0,0818", "−0,2319",
                   "0,0042", "0,0060", "0,0064", "−0,0025", "0,0083", "z = −0,30", "0,0688", "0,0367", "z = 1,88",
                   "%10 düzeyinde anlamlıdır", "50.742", "0,077", "0,085", "0,075", "−0,002", "(SH 0,008)", "NLSYM",
                   "Boylamsal", "ed76_i", "7 gözlem"):
        assert number in texts, number
    assert "fark yoktur" not in texts  # anlamlı olmayan fark "fark yok" diye okunmaz


# --- Konu 8 alternatifi: Angrist ve Lavy (1999), yerel doğrusal ve CV -----------------------------------------

@pytest.fixture(scope="module")
def al() -> pd.DataFrame:
    return _hansen("al1999")


def _cv(x: np.ndarray, y: np.ndarray, h: float, groups: np.ndarray | None) -> float:
    """Yerel doğrusal CV ölçütü (Gauss çekirdeği): her gözlem için kendisi (ya da kümesi) dışarıda, ağırlıklı en küçük
    kareler sabit terimi."""

    distance = x[None, :] - x[:, None]
    weight = np.exp(-0.5 * (distance / h) ** 2)
    if groups is None:
        np.fill_diagonal(weight, 0.0)
    else:
        weight[groups[:, None] == groups[None, :]] = 0.0
    errors = np.empty(len(x))
    for i in range(len(x)):
        w = weight[i]
        design = np.column_stack([np.ones(len(x)), distance[i]])
        coef = np.linalg.solve(design.T @ (w[:, None] * design), design.T @ (w * y))
        errors[i] = y[i] - coef[0]
    return float(np.mean(errors ** 2))


def _local(x: np.ndarray, y: np.ndarray, point: float, h: float) -> float:
    w = np.exp(-0.5 * ((x - point) / h) ** 2)
    design = np.column_stack([np.ones(len(x)), x - point])
    return float(np.linalg.solve(design.T @ (w[:, None] * design), design.T @ (w * y))[0])


def test_konu08_alternative_numbers_follow_al1999(al: pd.DataFrame) -> None:
    from core.labs import ornek_konu08 as K8

    data = al.astype("float64")
    assert len(data) == 4067
    data = data[data["grade"] == 4].dropna(subset=["avgmath", "disadvantaged", "schlcode"]).reset_index(drop=True)
    x, y = data["disadvantaged"].to_numpy(), data["avgmath"].to_numpy()
    schools = pd.factorize(data["schlcode"])[0]
    assert (len(y), data["schlcode"].nunique()) == (2049, 1013)
    assert (data.groupby("schlcode")["disadvantaged"].nunique() == 1).all()  # okul düzeyinde
    assert list(np.quantile(x, (0.10, 0.25, 0.5, 0.75, 0.90))) == [2.0, 4.0, 9.0, 19.0, 35.0]
    grid = np.round(np.arange(2.0, 20.0001, 0.5), 1)
    # Bağımsız CV yalnız seçilen h'lerin çevresinde (her h bir n×n hesap): en küçük değerler komşularından küçük.
    loo = {h: _cv(x, y, h, None) for h in (3.0, 3.5, 4.0)}
    clustered = {h: _cv(x, y, h, schools) for h in (5.0, 6.0, 6.5, 7.0, 8.5, 9.0, 4.5)}
    assert min(loo, key=loo.get) == 3.5 and min((6.0, 6.5, 7.0), key=clustered.get) == 6.5
    best = clustered[6.5]
    assert clustered[5.0] - best < 0.01 and clustered[8.5] - best < 0.01
    assert clustered[9.0] - best > 0.01 and clustered[4.5] - best > 0.01
    from core.labs import smoothing as S

    table = S.cv_curve(x, y, grid, schools)  # uygulamanın bütün ızgarası: en küçük değerler aynı yerde
    assert float(table["cv"].idxmin()) == 3.5 and float(table["cv_kume"].idxmin()) == 6.5
    assert abs(table.loc[3.5, "cv"] - loo[3.5]) < 1e-9 and abs(table.loc[6.5, "cv_kume"] - best) < 1e-9
    points = np.array(K8.ALT_POINTS)

    def basis(v: np.ndarray, spline: bool) -> np.ndarray:
        columns = [np.ones(len(v)), v, v ** 2 / 100, v ** 3 / 1e4]
        if spline:
            columns += [np.maximum(v - k, 0) ** 3 / 1e4 for k in K8.ALT_KNOTS]
        return np.column_stack(columns)

    linear = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    cubic = np.linalg.lstsq(basis(x, False), y, rcond=None)[0]
    spline = np.linalg.lstsq(basis(x, True), y, rcond=None)[0]
    predictions = {"Doğrusal": linear[0] + linear[1] * points, "Kübik polinom": basis(points, False) @ cubic,
                   "Kübik spline": basis(points, True) @ spline,
                   "Yerel doğrusal h = 6,5": np.array([_local(x, y, p, 6.5) for p in points]),
                   "Yerel doğrusal h = 3,5": np.array([_local(x, y, p, 3.5) for p in points])}
    values = {(1, "Analiz örneklemi (N)"): float(len(y)), (2, "Birini dışarıda bırakan CV ile h"): 3.5,
              (2, "Küme-silmeli CV ile h"): 6.5}
    for label, column in predictions.items():
        for point, value in zip(points, column):
            values[(3, f"{label}, dezavantajlı %{int(point)}")] = value
    spec = K8.alternative()
    assert {(step.number, check.label) for step in spec.steps for check in step.checks} == set(values)
    for (step, label), value in values.items():
        expected, decimals = _expected(K8, step, label)
        assert _matches(value, expected, decimals), (label, value, expected)
    # Metinlerdeki sayılar: eğim ve aralık ortalamaları (yirmi eşit genişlikli aralık)
    assert round(linear[1], 3) == -0.296
    edges = np.linspace(x.min(), x.max(), 21)
    index = np.minimum(np.searchsorted(edges, x, side="right") - 1, 19)
    bins = pd.DataFrame({"i": index, "x": x, "y": y}).groupby("i").agg(x=("x", "mean"), y=("y", "mean"),
                                                                       n=("y", "size"))
    line = linear[0] + linear[1] * bins["x"]
    above = bins["y"] > line
    low = pd.Series(edges[bins.index], index=bins.index)
    assert len(bins) == 20 and np.allclose(edges[[2, 7, 8, 10, 13, 16]], [7.6, 26.6, 30.4, 38.0, 49.4, 60.8])
    assert above[low < 7.5].all()  # %7,6'nın altında üstte
    assert list(bins.index[(low > 7.5) & (low < 37.9) & above]) == [7]  # %7,6–38'de yalnız %26,6–30,4 üstte
    assert above[(low > 37.9) & (low < 60)].all()  # %38–60,8 üstte
    tail = bins[low > 49]
    assert len(tail) == 7 and tail["n"].between(1, 22).all() and tail["n"].max() == 22
    assert above[low > 49].any() and not above[low > 49].all()  # doğrunun iki yanında
    mid = [column[1] for column in predictions.values()]
    assert (round(min(mid), 2), round(max(mid), 2)) == (69.89, 70.29)
    gaps = [abs(a - b) for a, b in zip(predictions["Yerel doğrusal h = 6,5"], predictions["Yerel doğrusal h = 3,5"])]
    assert round(max(gaps), 2) == 0.23 and int(np.argmax(gaps)) == 2  # üç noktada en çok, %35'te
    ends = {point: abs(_local(x, y, point, 6.5) - _local(x, y, point, 3.5)) for point in (0.0, 76.0)}
    assert (round(ends[0.0], 2), round(ends[76.0], 2)) == (0.84, 2.36)
    assert (int((x == 0).sum()), int((x == 76).sum()), x.max()) == (87, 1, 76)

    def spread(point: float) -> float:
        v = np.array([point])
        values = [linear[0] + linear[1] * point, (basis(v, False) @ cubic)[0], (basis(v, True) @ spline)[0],
                  _local(x, y, point, 6.5), _local(x, y, point, 3.5)]
        return max(values) - min(values)

    three = [spread(point) for point in points]
    assert round(three[0], 2) == 1.25 and three[0] == max(three)  # üç nokta içinde en çok %2'de (72,36–73,61)
    assert [round(spread(point), 2) for point in (0.0, 55.0, 76.0)] == [3.07, 2.75, 8.78]
    texts = _texts(K8)
    for number in ("4.067", "2.049", "1.013", "−0,296", "%7,6'nın altında", "%26,6–30,4", "%38–60,8",
                   "%49,4'ün üstündeki yedi aralığın", "1–22 sınıf", "5–8,5", "0,01'in", "12,3", "6,2",
                   "(%4, %9 ve %19)", "(%2, %9 ve %35)", "69,89–70,29", "72,87–73,02", "73,18", "73,61", "62,58–62,81",
                   "en çok 0,23 puandır (%35'te)", "%0'da (87 sınıf) 0,84", "%76'da (tek sınıf) 2,36", "72,36–73,61",
                   "%0'da 3,07, %55'te 2,75, %76'da 8,78", "Maimonides", "4,2 milyon", "37 bant"):
        assert number in texts, number
    assert len(grid) == 37


# --- Alternatiflerin kodu sentetik veriyle: Konu 7–8 ----------------------------------------------------------

def _synthetic_card1995() -> pd.DataFrame:
    rng = np.random.default_rng(8)
    n = 800
    education = rng.integers(8, 19, n)
    age = rng.integers(24, 35, n)
    # 201 siyahi, 597 diğer (iki eksik ücret ilk satırlardadır): grup büyüklüğü × τ hiçbir kantilde tam sayı değil.
    # Tam sayı olsaydı grubun τ kantili iki sıra istatistiği arasında herhangi bir değer olabilir ve gösterge katsayısı
    # tek olmazdı (dillerin seçtiği köşe çözümü farklı çıkar).
    black = (np.arange(n) < 203).astype(int)
    log_wage = 0.6 + 0.08 * education + 0.03 * (age - education - 6) - 0.2 * black + rng.normal(0, 0.4, n) * (
        1 + 0.03 * education)
    wage = np.exp(log_wage) * 100  # sürekli: kantil çözümleri tek olsun (bağlı sonuç değeri yok)
    frame = pd.DataFrame({"ed76": education, "age76": age, "black": black, "wage76": wage,
                          "lwage76": np.log(wage), "nearc4": rng.integers(0, 2, n), "smsa76r": rng.integers(0, 2, n),
                          "reg76r": rng.integers(0, 2, n), "smsa66r": rng.integers(0, 2, n)})
    for i in range(2, 10):
        frame[f"reg66{i}"] = 0
    frame.loc[[3, 50], ["lwage76", "wage76"]] = np.nan
    return frame


def _synthetic_al1999() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    schools = 300
    sizes = rng.integers(1, 4, schools)
    school = np.repeat(np.arange(1, schools + 1), sizes)
    share = np.repeat(np.round(rng.gamma(1.5, 9, schools)).clip(0, 80), sizes)
    effect = np.repeat(rng.normal(0, 4, schools), sizes)
    score = 74 - 0.45 * share + 0.004 * share ** 2 + effect + rng.normal(0, 5, len(school))
    grade = np.where(rng.random(len(school)) < 0.8, 4, 5)
    return pd.DataFrame({"schlcode": school, "grade": grade, "classid": 1, "classize": 30, "enrollment": 80,
                         "avgmath": score, "avgverb": score + 3, "disadvantaged": share})


@pytest.mark.parametrize("module_name, dataset, build", [("ornek_konu07", "card1995", _synthetic_card1995),
                                                         ("ornek_konu08", "al1999", _synthetic_al1999)],
                         ids=["konu07", "konu08"])
def test_alternative_code_reproduces_the_app_on_synthetic_data_7_8(module_name: str, dataset: str, build,
                                                                    tmp_path: Path) -> None:
    import importlib

    module = importlib.import_module(f"core.labs.{module_name}")
    frame = build()
    spec = with_app_values(module.alternative(), {dataset: frame})
    path = tmp_path / DATASET_MEMBERS[dataset]
    frame.to_stata(path, write_index=False, version=118)
    for language in ("Python", "R") if RSCRIPT else ("Python",):
        before, after = {"Python": ("YEREL_DOSYA = None", f'YEREL_DOSYA = "{path.as_posix()}"'),
                         "R": ("yerel_dosya <- NULL", f'yerel_dosya <- "{path.as_posix()}"')}[language]
        script = render_script(spec, language)
        assert before in script
        _check_script(spec, language, script.replace(before, after, 1), tmp_path)


# --- Kendi verin: Konu 7 ----------------------------------------------------------------------------------

KONU07 = CustomChoices(roles={"sonuc": "Saatlik ücret (TL)", "aciklayici": "Eğitim yılı", "kontrol": "Deneyim (yıl)",
                              "grup": "Cinsiyet"}, extra=("Evli (1/0)", "Kıdem (yıl)"), picks={"grup": "Kadın"})


def _even_groups() -> pd.DataFrame:
    """Grup büyüklüğü × τ tam sayı (200 × 0,25 = 50): 0,25 ve 0,75 kantillerinde gösterge katsayısı tek değildir."""

    rng = np.random.default_rng(8)
    n = 800
    education = rng.integers(8, 19, n)
    experience = rng.integers(0, 15, n)
    group = np.where(np.arange(n) < 200, "B", "A")
    log_wage = (0.6 + 0.08 * education + 0.03 * experience - 0.2 * (group == "B")
                + rng.normal(0, 0.4, n) * (1 + 0.03 * education))
    return pd.DataFrame({"Ücret": np.exp(log_wage) * 10, "Eğitim": education, "Deneyim": experience, "Grup": group})


def test_konu07_nonunique_quantile_solutions_drop_only_their_checks(tmp_path: Path) -> None:
    from core.labs import ornek_konu07 as K7

    choices = CustomChoices(roles={"sonuc": "Ücret", "aciklayici": "Eğitim", "kontrol": "Deneyim", "grup": "Grup"},
                            picks={"grup": "B"})
    frame = _even_groups()
    case = _case(K7, frame, choices)
    plan = K7.own_plan(case)
    result = K7.stability(K7._derived(case), plan)
    unstable, unstable_se = result.unstable, result.unstable_se
    assert unstable | unstable_se  # bu veride en az bir kantilde çözüm (ya da τ ± h çözümü) tek değil
    spec, data = _own(K7, frame, choices)
    labels = [check.label for step in spec.steps for check in step.checks]
    for name in unstable:
        assert not any(label.startswith(K7.MODEL_LABELS[name] + ":") for label in labels), name
    for name in unstable_se:
        assert not any(label.startswith(K7.MODEL_LABELS[name] + ":") and "SH" in label for label in labels), name
    stable = [name for _, name in K7.QUANTILES if name not in unstable | unstable_se]
    assert stable and all(any(label.startswith(K7.MODEL_LABELS[name] + ":") for label in labels) for name in stable)
    run = run_lab(spec)
    note = _notes(spec, run)[1]
    assert ("kantil çözümü gösterim basamağında tek değil" in note) == bool(unstable)
    assert ("τ ± h kantillerinde çözüm tek değil" in note) == bool(unstable_se - unstable)
    _reproduces_in_both_languages(spec, data, tmp_path)


def test_konu07_role_rules_and_scale() -> None:
    from core.labs import ornek_konu07 as K7

    frame = K7.sample()
    with pytest.raises(K.UploadError, match="dört farklı sütun"):
        _case(K7, frame, CustomChoices(roles={**KONU07.roles, "kontrol": "Eğitim yılı"}, picks=KONU07.picks))
    rare = frame.assign(Cinsiyet=np.where(np.arange(len(frame)) < 8, "Kadın", "Erkek"))
    with pytest.raises(K.UploadError, match="en az 10 gözlem"):
        _case(K7, rare, KONU07)
    with pytest.raises(K.UploadError, match="7 katsayı var; .* en az 140 gözlem gerekir, dosyada 110 var"):
        _case(K7, frame.iloc[:110], KONU07)  # sabit + 4 rol terimi + 2 ek kontrol: en az 20 × 7 gözlem
    negative = frame.assign(**{"Saatlik ücret (TL)": frame["Saatlik ücret (TL)"] - 400})
    spec, _ = _own(K7, negative, CustomChoices(roles=KONU07.roles, extra=KONU07.extra, picks=KONU07.picks,
                                               options={"log": False}))
    labels = [check.label for step in spec.steps for check in step.checks]
    assert not any("yüzde" in label for label in labels)
    wide = frame.assign(**{"Eğitim yılı": frame["Eğitim yılı"] * 1000})
    assert K7.own_plan(_case(K7, wide, KONU07)).decimals >= 7


# --- Kendi verin: Konu 8 ----------------------------------------------------------------------------------

KONU08 = CustomChoices(roles={"sonuc": "Sınav puanı", "aciklayici": "Haftalık çalışma saati", "kume": "Okul"})


def test_konu08_without_a_cluster_compares_h_with_twice_h(tmp_path: Path) -> None:
    from core.labs import ornek_konu08 as K8

    choices = CustomChoices(roles={"sonuc": "Sınav puanı", "aciklayici": "Haftalık çalışma saati"})
    spec, data = _own(K8, K8.sample(), choices)
    labels = [check.label for step in spec.steps for check in step.checks]
    assert "Küme-silmeli CV ile h" not in labels
    plan = K8.own_plan(_case(K8, K8.sample(), choices))
    assert plan.cluster is None and plan.h_other == round(2 * plan.h_main, plan.h_decimals)
    assert "HC1" in spec.steps[0].explanation and "küme seçilmedi" in spec.steps[0].explanation.lower()
    _reproduces_in_both_languages(spec, data, tmp_path)


def test_konu08_role_rules_suggestion_and_cv_cache(monkeypatch) -> None:
    from core.labs import ornek_konu08 as K8
    from core.labs import smoothing as S

    frame = K8.sample()
    table = K.read_upload(CSV, _csv(frame))
    assert K8.suggest(table) == {"kume": "Okul"}
    with pytest.raises(K.UploadError, match="en az 20 farklı değer"):
        _case(K8, frame.assign(**{"Haftalık çalışma saati": frame["Haftalık çalışma saati"] // 5}), KONU08)
    with pytest.raises(K.UploadError, match="küme var"):
        _case(K8, frame.assign(Okul=frame["Okul"] % 5), KONU08)
    with pytest.raises(K.UploadError, match="her gözlem ayrı bir küme"):
        _case(K8, frame.assign(Okul=np.arange(len(frame))), KONU08)
    big = pd.concat([frame] * 4, ignore_index=True)
    with pytest.raises(K.UploadError, match="en çok 2.500 gözlem"):
        _case(K8, big, KONU08)
    calls = []
    original = S._cv_compute
    monkeypatch.setattr(S, "_cv_compute", lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs))
    S._CV_CACHE.clear()
    spec, _ = _own(K8, frame.assign(**{"Sınav puanı": frame["Sınav puanı"] + 0.01}), KONU08)
    run_lab(spec)
    assert len(calls) == 1  # bant genişliği seçimi, beklenen değerler ve gösterim aynı CV tablosunu kullanır


def test_konu08_texts_follow_the_chosen_bandwidths() -> None:
    from core.labs import ornek_konu08 as K8

    spec, _ = _own(K8, K8.sample(), KONU08)
    plan = K8.own_plan(_case(K8, K8.sample(), KONU08))
    assert (plan.h_main, plan.h_other, plan.doubled) == (3.5, 2.5, False)  # küme-silmeli 3,5; birini dışarıda 2,5
    assert "daha dar bir bantla (h = 2,5)" in spec.steps[3].takeaway
    notes = _notes(spec)
    assert "eğrileri" in notes[5] and "karşılaştırılan üç noktada en çok" in notes[5]
    assert "noktasında ayrışır" in notes[3]
    assert K8._math(0.25) == "0{,}25" and K8._shift(-4.7) == "x + 4,7" and K8._shift(4.7) == "x − 4,7"


def test_konu08_equal_bandwidths_compare_with_twice_h(tmp_path: Path) -> None:
    """Zayıf küme yapısında iki CV aynı h'yi seçer; karşılaştırma bandı ana bandın iki katıdır, etiketler tektir."""

    from core.labs import ornek_konu08 as K8

    rng = np.random.default_rng(1)
    n = 300
    hours = np.round(rng.uniform(0, 50, n), 1)
    score = np.round(20 + 0.5 * hours - 0.008 * hours ** 2 + rng.normal(0, 4, n), 1)
    frame = pd.DataFrame({"Puan": score, "Saat": hours, "Okul": rng.integers(1, 31, n)})
    choices = CustomChoices(roles={"sonuc": "Puan", "aciklayici": "Saat", "kume": "Okul"})
    case = _case(K8, frame, choices)
    chosen = K8.bandwidths(case)
    assert chosen.loo == chosen.clustered
    plan = K8.own_plan(case)
    assert plan.doubled and plan.h_other == round(2 * plan.h_main, plan.h_decimals)
    spec, data = _own(K8, frame, choices)
    labels = [check.label for step in spec.steps for check in step.checks]
    assert len(labels) == len(set(labels))
    assert "aynı h'yi seçtiği için" in _notes(spec)[2] and "daha geniş" in spec.steps[3].takeaway
    _reproduces_in_both_languages(spec, data, tmp_path)


def test_konu08_isolated_values_raise_the_grid_and_r_still_draws(tmp_path: Path) -> None:
    """Uçta seyrek değerler: küçük h'lerde dışarıda bırakılan tahmin ya da grafik eğrisi sayısal olarak hesaplanamaz
    (R'de grafik çizilemezdi); ızgaranın alt ucu yükseltilir ve iki dil aynı sayıları verir."""

    from core.labs import ornek_konu08 as K8
    from core.labs import smoothing as S

    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 300)
    y = 2 * np.sin(x) + rng.normal(0, 0.5, 300)
    frame = pd.DataFrame({"Y": np.round(np.concatenate([y, [1.0, 0.5, 0.0]]), 3),
                          "X": np.round(np.concatenate([x, [4.0, 5.2, 6.8]]), 3)})
    choices = CustomChoices(roles={"sonuc": "Y", "aciklayici": "X"})
    case = _case(K8, frame, choices)
    values, outcome = case.data[case.roles["aciklayici"]], case.data[case.roles["sonuc"]]
    candidate, _ = K8._grid(values)
    plan = K8.own_plan(case)
    assert plan.raised and plan.grid[0] > candidate[0]
    worst = min(S.cv_conditioning(values, outcome, [candidate[0]]).iloc[0, 0],
                S.local_conditioning(values, K8._plot_points(values), [candidate[0]])[0])
    assert worst < S.RELIABLE
    spec, data = _own(K8, frame, choices)
    assert "Izgaranın alt ucu" in spec.steps[1].explanation
    _reproduces_in_both_languages(spec, data, tmp_path)


def test_konu08_a_skewed_regressor_is_accepted_and_reproduces(tmp_path: Path) -> None:
    """Çarpık açıklayıcının kübik terimleri yüksek koşul sayısı verir (Konu 1–2 sınırının üstünde); kontroller
    tahmin olduğu için kabul edilir ve iki dil aynı sayıları verir."""

    from core.labs import ornek_konu08 as K8
    from core.labs.ornek import CONDITION_LIMIT

    rng = np.random.default_rng(7)
    income = np.round(rng.lognormal(np.log(30), 1.0, 500), 2)
    score = np.round(50 + 10 * np.log(income / 30) + rng.normal(0, 8, 500), 1)
    frame = pd.DataFrame({"Puan": score, "Gelir (bin TL)": income})
    choices = CustomChoices(roles={"sonuc": "Puan", "aciklayici": "Gelir (bin TL)"})
    case = _case(K8, frame, choices)
    _, _, basis = K8._terms(case)
    x = case.data[case.roles["aciklayici"]].astype(float)
    design = np.column_stack([np.ones(len(x)), x] + [np.asarray(K8.E.evaluate(e, case.data), dtype=float)
                                                     for _, e, _ in basis])
    assert np.linalg.cond(design / np.linalg.norm(design, axis=0)) > CONDITION_LIMIT
    spec, data = _own(K8, frame, choices)
    _reproduces_in_both_languages(spec, data, tmp_path)


def test_konu08_exact_fit_scale_braces_and_gaps(monkeypatch) -> None:
    from core.labs import ornek_konu08 as K8
    from core.labs import smoothing as S

    frame = K8.sample()
    hours = frame["Haftalık çalışma saati"]
    with pytest.raises(K.UploadError, match="tam doğrusal bir fonksiyonu"):
        _case(K8, frame.assign(**{"Sınav puanı": 2.5 * hours + 7}), KONU08)
    with pytest.raises(K.UploadError, match="ölçeği çok küçük"):
        _case(K8, frame.assign(**{"Sınav puanı": frame["Sınav puanı"] * 1e-8}), KONU08)
    braces = frame.rename(columns={"Haftalık çalışma saati": "Saat {hafta}"})
    spec, _ = _own(K8, braces, CustomChoices(roles={**KONU08.roles, "aciklayici": "Saat {hafta}"}))
    assert any("“Saat {hafta}” = " in check.label for step in spec.steps for check in step.checks)
    table = K.read_upload(CSV, _csv(frame.rename(columns={"Okul": "İl"})))
    assert K8.suggest(table) == {"kume": "İl"}
    monkeypatch.setattr(S, "RELIABLE", 2.0)  # ρ ≤ 1: hiçbir bant genişliği güvenilir sayılmaz
    with pytest.raises(K.UploadError, match="çok büyük boşluklar var"):
        _case(K8, frame, KONU08)


def test_konu07_few_valued_outcome_and_term_scales_are_rejected(monkeypatch) -> None:
    from core.labs import ornek_konu07 as K7

    frame = K7.sample()
    rng = np.random.default_rng(3)
    with pytest.raises(K.UploadError, match="5 farklı değer var"):
        _case(K7, frame.assign(**{"Saatlik ücret (TL)": rng.integers(1, 6, len(frame))}), KONU07)
    with pytest.raises(K.UploadError, match="1 farklı değer var"):  # sabit sonuç: ölçek iletisi değil
        _case(K7, frame.assign(**{"Saatlik ücret (TL)": 10.0}), KONU07)
    days = frame.assign(**{"Deneyim (yıl)": frame["Deneyim (yıl)"] * 365 + rng.integers(0, 365, len(frame))})
    with pytest.raises(K.UploadError, match="ölçeği sonuca göre çok büyük"):
        _case(K7, days, KONU07)
    monkeypatch.setattr(K7, "H_CONDITION", 0.0)  # yoğunluk matrisinin tersi alınamıyormuş gibi
    with pytest.raises(K.UploadError, match="Hendricks–Koenker standart hataları hesaplanamıyor"):
        K7.CUSTOM.build(_case(K7, frame, KONU07))


def test_konu07_tail_standard_errors_are_described_from_the_results() -> None:
    """Düzgün dağılımlı hatada kantil standart hatası uçlarda küçülür (τ(1−τ) küçülür, yoğunluk sabit); yorum
    sonuçlardan yazılır."""

    from core.labs import ornek_konu07 as K7

    rng = np.random.default_rng(12)
    n = 800
    education, experience = rng.integers(8, 19, n), rng.integers(0, 30, n)
    group = np.where(rng.random(n) < 0.4, "Kadın", "Erkek")
    score = 50 + 2 * education + 0.5 * experience - 3 * (group == "Kadın") + rng.uniform(-10, 10, n)
    frame = pd.DataFrame({"Puan": np.round(score, 3), "Eğitim": education, "Deneyim": experience, "Cinsiyet": group})
    choices = CustomChoices(roles={"sonuc": "Puan", "aciklayici": "Eğitim", "kontrol": "Deneyim", "grup": "Cinsiyet"},
                            picks={"grup": "Kadın"}, options={"log": False})
    spec, _ = _own(K7, frame, choices)
    run = run_lab(spec)
    term = K7.own_plan(_case(K7, frame, choices)).x
    errors = [float(run.state.models[name].bse[term]) for _, name in K7.QUANTILES]
    assert errors[0] < errors[2] and errors[-1] < errors[2]
    assert "uçlarda medyandakinden büyük değil" in _notes(spec, run)[1]
    assert "uçlarda büyür" not in spec.steps[0].takeaway
    sample, _ = _own(K7, K7.sample(), KONU07)
    assert "Stata" not in sample.steps[1].code_note and "iki dilde" in sample.steps[3].explanation


def test_konu06_braces_in_a_column_name() -> None:
    frame = K6.sample().rename(columns={"Hane geliri (bin TL)": "Gelir {bin}"})
    spec, _ = _own(K6, frame, CustomChoices(roles={**KONU06.roles, "aciklayici": "Gelir {bin}"}, extra=KONU06.extra))
    assert any("“Gelir {bin}” = " in check.label for step in spec.steps for check in step.checks)
