"""Konu 9–12 ek kaynakları: alternatif örneklerin bağımsız hesabı, yeni hesap işlemleri ve kendi veri kuralları.

* RDD'nin küme-dayanıklı standart hatası (eşikteki sıçrama ve grafik eğrisi) statsmodels'ın kümeli WLS'iyle birebir
  karşılaştırılır; notların RDD yardımcıları (kümesiz) değişmez.
* AL1999 (Konu 9), DDK2011 (Konu 10) ve Card1995 (Konu 11–12) alternatiflerinin kontrol değerleri ve metinlerdeki sayılar
  uygulamanın kodundan ayrı bir hesapla doğrulanır: kümeli WLS statsmodels ile, bootstrap ayrı yazılmış bir döngüyle,
  Ridge ve Lasso scikit-learn'ün tek tek tahmin edicileriyle. Dosya yolu verilmezse atlanır
  (``IKT807_HANSEN_AL1999_PATH``, ``IKT807_HANSEN_DDK2011_PATH``, ``IKT807_HANSEN_CARD1995_PATH``).
* Dört alternatifin üretilen Python ve R kodu gerçek veri olmadan, sentetik bir ``.dta`` dosyasıyla da çalıştırılır;
  uygulama gerçek verinin Stata ``compress`` biçimiyle (byte/int sütunlar) de aynı sayıları verir.
* Kendi verin kuralları: Konu 9'da eşik ve bant genişliği alanları, eşiğin iki yanında veri ve küme sayısı, tedavi tarafı;
  Konu 10'da küme bootstrap'ı ve tanımsız tekrarlar; Konu 11'de terim sözlüğü (sabit kalan terimler, terim sayısı, tam
  bağlantı) ve kategorik değişken; Konu 12'de gözlem ve kontrol sayısı, kümeli katlar.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from core.codegen.base import render_script, script_filename
from core.codegen.python_rdd_boot import rdd_helper_names
from core.hansen_data import DATASET_MEMBERS, load_from_upload
from core.labs import kendi_veri as K
from core.labs import ornek_konu09 as K9
from core.labs import ornek_konu10 as K10
from core.labs import ornek_konu11 as K11
from core.labs import ornek_konu12 as K12
from core.labs import rdd as RD
from core.labs.ornek import CustomChoices, custom_case, with_app_values
from core.labs.registry import LABS
from core.labs.runner import run_lab
from core.labs.spec import RDD, Plot, RDDCurve, dictionary_terms

RSCRIPT = shutil.which("Rscript")
CSV = "veri.csv"
KONU09 = CustomChoices(roles={"sonuc": "Mezuniyet not ortalaması", "esik_degiskeni": "Sınav puanı", "kume": "Okul"},
                       numbers={"esik": "70", "h": ""})
KONU10 = CustomChoices(roles={"sonuc": "Matematik puanı", "hedef": "Program (1/0)", "kume": "Okul"},
                       extra=("Başlangıç puanı", "Kız (1/0)"))
KONU11 = CustomChoices(
    roles={"sonuc": "Satış fiyatı (bin TL)", "temel1": "Alan (m²)", "temel2": "Bina yaşı", "kategori": "İlçe",
           "kategori2": "Isıtma"},
    extra=("Oda sayısı", "Kat", "Merkeze uzaklık (km)", "Asansör (1/0)", "Otopark (1/0)", "Manzara (1/0)"))
KONU12 = CustomChoices(roles={"sonuc": "Log kazanç", "tedavi": "Kursa katıldı (1/0)", "kume": "Firma"},
                       extra=("Önceki log kazanç", "Yaş", "Eğitim yılı", "Kadın (1/0)"))


def _hansen(dataset: str) -> pd.DataFrame:
    """Gerçek veri, bağımsız hesap için ondalıklı sayıya çevrilmiş. Hansen'in arşivindeki .dta dosyalarında eksiksiz tam
    sayı sütunları Stata'nın sıkıştırılmış türleriyle (byte/int) saklanabilir; pandas bunları int8/int16 okur ve kare ya
    da küp uyarısız taşar (ör. deneyim² Card1995'te). Uygulama ve üretilen kod ondalıklı hesaplar; testler de öyle."""

    value = os.environ.get(f"IKT807_HANSEN_{dataset.upper()}_PATH")
    if not value or not Path(value).is_file():
        pytest.skip(f"Gerçek Hansen {DATASET_MEMBERS[dataset]} dosyası verilmedi.")
    loaded = load_from_upload(dataset, Path(value).read_bytes(), Path(value).name)
    assert loaded.matches_hansen
    frame = loaded.frame.copy()
    integers = [column for column in frame.columns if frame[column].dtype.kind in "iu"]
    frame[integers] = frame[integers].astype(float)
    return frame


def _matches(value: float, expected: float, decimals: int) -> bool:
    return abs(value - expected) <= 0.5 * 10 ** (-decimals) + 1e-12


def _expected(module, step: int, label: str) -> tuple[float, int]:
    check = next(check for item in module.alternative().steps if item.number == step for check in item.checks
                 if check.label == label)
    return check.expected, check.decimals


def _texts(module) -> str:
    spec = module.alternative()
    return " ".join(part for step in spec.steps for part in (step.explanation, step.takeaway, step.code_note or ""))


def _csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(sep=";", decimal=",", index=False).encode("cp1254")


def _case(module, frame: pd.DataFrame, choices: CustomChoices):
    return custom_case(module.CUSTOM, K.read_upload(CSV, _csv(frame)), choices)[0]


def _own(module, frame: pd.DataFrame, choices: CustomChoices):
    data = _csv(frame)
    case, _ = custom_case(module.CUSTOM, K.read_upload(CSV, data), choices)
    return module.CUSTOM.build(case), (CSV, data)


def _check_script(spec, language: str, script: str, folder: Path) -> str:
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
    return result.stdout


def _reproduces_in_both_languages(spec, data: tuple[str, bytes], folder: Path) -> None:
    (folder / data[0]).write_bytes(data[1])
    for language in ("Python", "R") if RSCRIPT else ("Python",):
        _check_script(spec, language, render_script(spec, language), folder)


def _notes(spec, run=None) -> dict[int, str]:
    run = run_lab(spec) if run is None else run
    assert all(item.passed for items in run.checks.values() for item in items)
    return {step.number: step.note_for(run.state) for step in spec.steps if step.note_for is not None}


# --- RDD: küme-dayanıklı standart hata --------------------------------------------------------------------------

def _clustered_rd(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = 120
    size = rng.integers(2, 6, groups)
    cluster = np.repeat(np.arange(groups), size)
    x = np.repeat(rng.uniform(-10, 10, groups), size) + rng.normal(0, 0.5, len(cluster))
    y = 1 + 0.2 * x + 1.5 * (x >= 0) + np.repeat(rng.normal(0, 1, groups), size) + rng.normal(0, 1, len(cluster))
    return pd.DataFrame({"x": x, "y": y, "g": cluster})


@pytest.mark.parametrize("kernel, scale", [("triangular", "hansen"), ("rectangular", "hansen"),
                                           ("triangular", "window")])
def test_rdd_cluster_errors_match_statsmodels(kernel: str, scale: str) -> None:
    data = _clustered_rd()
    fit = RD.rdd_fit(data["x"], data["y"], 0.0, 2.5, kernel, scale, cluster=data["g"])
    r = data["x"].to_numpy()
    weights = RD.kernel_weights(r, 2.5, kernel, scale)
    inside = weights > 0
    d = (r[inside] >= 0).astype(float)
    design = np.column_stack([np.ones(inside.sum()), d, r[inside], d * r[inside]])
    codes = pd.factorize(data["g"].to_numpy()[inside])[0]
    reference = sm.WLS(data["y"].to_numpy()[inside], design, weights=weights[inside]).fit(
        cov_type="cluster", cov_kwds={"groups": codes})
    assert np.allclose(fit.params.to_numpy(), reference.params, atol=1e-10)
    assert np.allclose(fit.bse.to_numpy(), reference.bse, atol=1e-10)
    assert fit.groups == len(np.unique(codes))
    hc1 = RD.rdd_fit(data["x"], data["y"], 0.0, 2.5, kernel, scale)
    assert hc1.groups is None and np.allclose(hc1.params, fit.params)


def test_rdd_curve_cluster_bands_match_pointwise_statsmodels() -> None:
    data = _clustered_rd()
    left = data[data["x"] < 0]
    points = np.linspace(-8, 0, 9)
    estimate, error = RD.local_linear_side(left["x"], left["y"], points, 2.0, groups=left["g"])
    for point, value, se in zip(points, estimate, error):
        d = left["x"].to_numpy() - point
        weights = np.maximum(1 - np.abs(d) / (2.0 * math.sqrt(6)), 0)
        inside = weights > 0
        codes = pd.factorize(left["g"].to_numpy()[inside])[0]
        reference = sm.WLS(left["y"].to_numpy()[inside], np.column_stack([np.ones(inside.sum()), d[inside]]),
                           weights=weights[inside]).fit(cov_type="cluster", cov_kwds={"groups": codes})
        assert abs(value - reference.params[0]) < 1e-10 and abs(se - reference.bse[0]) < 1e-10


def test_notes_rdd_helpers_are_unchanged_and_curves_do_not_mix() -> None:
    notes = LABS["konu09"]
    ops = [op for step in notes.steps for op in step.operations]
    layers = [layer for op in ops if isinstance(op, Plot) for layer in op.layers]
    assert rdd_helper_names(ops, layers) == ["rdd", "rdd_egri"]
    alternative = K9.alternative()
    ops = [op for step in alternative.steps for op in step.operations]
    layers = [layer for op in ops if isinstance(op, Plot) for layer in op.layers]
    assert rdd_helper_names(ops, layers) == ["rdd_kume", "rdd_egri_kume"]
    mixed = [RDDCurve("y", 0.0, 1.0, "a"), RDDCurve("y", 0.0, 1.0, "b", cluster="g")]
    with pytest.raises(ValueError, match="birlikte kullanılamaz"):
        rdd_helper_names([], mixed)
    stata = render_script(alternative, "Stata")
    assert "kume(schlcode)" in stata and "vce(`vce')" in stata
    assert 'mata: rdd_egrisi("enrollment", "avgverb", "schlcode", `rdd_n0\'' in stata
    assert "panelsetup(gs, 1)" in stata


# --- Konu 9: AL1999 ---------------------------------------------------------------------------------------------

def _wls(frame: pd.DataFrame, y: str, h: float, kernel: str = "triangular", scale: str = "hansen"):
    r = frame["enrollment"].to_numpy(dtype=float) - 41
    width = h if scale == "window" else h * math.sqrt(6 if kernel == "triangular" else 3)
    weights = np.maximum(1 - np.abs(r) / width, 0) if kernel == "triangular" else (np.abs(r) <= width).astype(float)
    inside = weights > 0
    d = (r[inside] >= 0).astype(float)
    design = np.column_stack([np.ones(inside.sum()), d, r[inside], d * r[inside]])
    codes = pd.factorize(frame["schlcode"].to_numpy()[inside])[0]
    return sm.WLS(frame[y].to_numpy(dtype=float)[inside], design, weights=weights[inside]).fit(
        cov_type="cluster", cov_kwds={"groups": codes}), int(inside.sum())


def test_konu09_alternative_numbers_follow_from_an_independent_calculation() -> None:
    frame = _hansen("al1999")
    data = frame[frame["grade"] == 5].dropna(subset=["enrollment", "avgverb", "classize", "schlcode"])
    assert len(data) == 2018 and data["schlcode"].nunique() == 1001 and int((data["enrollment"] >= 41).sum()) == 1724
    first, _ = _wls(data, "classize", 8)
    for label, value in (("İlk aşama: sınıf mevcudu sıçraması (h = 8)", first.params[1]),
                         ("İlk aşama: SH", first.bse[1])):
        assert _matches(value, *_expected(K9, 1, label)), label
    for h in K9.ALT_BANDWIDTHS:
        fit, n = _wls(data, "avgverb", h)
        tau, se = fit.params[1], fit.bse[1]
        values = {"etkin örneklem n_h": n, "τ̂": tau, "SH": se, "alt %95": tau - 1.96 * se, "üst %95": tau + 1.96 * se}
        for label, value in values.items():
            assert _matches(value, *_expected(K9, 2, f"h = {int(h)}: {label}")), (h, label, value)
        if h == 8:
            left, right = fit.params[0], fit.params[0] + fit.params[1]
    assert _matches(left, *_expected(K9, 3, "Eşiğin solunda tahmin (h = 8)"))
    assert _matches(right, *_expected(K9, 3, "Eşiğin sağında tahmin (h = 8)"))
    for model, kernel, scale in (("Dikdörtgen çekirdek", "rectangular", "hansen"), ("Pencere ±8", "triangular", "window")):
        fit, n = _wls(data, "avgverb", 8, kernel, scale)
        assert _matches(fit.params[1], *_expected(K9, 3, f"{model}: τ̂"))
        assert _matches(fit.bse[1], *_expected(K9, 3, f"{model}: SH"))
        assert n == _expected(K9, 3, f"{model}: gözlem sayısı")[0]
    assert round(float(data["avgverb"].std()), 1) == 7.7
    schools = data.groupby("schlcode")["enrollment"].agg(["min", "max"])
    assert (schools["min"] == schools["max"]).all()  # okulun bütün 5. sınıfları aynı kaydı taşır
    assert int((schools["min"] == 40).sum()) == 7 and int((schools["min"] == 41).sum()) == 16
    texts = _texts(K9)
    for number in ("2.018", "1.001", "1.724", "8,9", "1,94", "315", "960", "3,20", "1,84", "5,83", "4,26",
                   "[−0,78; 11,77]", "4,3–5,8", "7,7", "5,20/(−8,90) ≈ −0,58", "19,6", "667", "67,88", "73,08",
                   "13,86", "460", "5,90 (2,47)", "259", "4,75 (3,47)", "21–61", "81",
                   "kaydı 40 olan 7 okula karşı 41 olan 16 okul"):
        assert number in texts, number
    assert 5.20 / -8.90 == pytest.approx(-0.584, abs=0.001)


# --- Konu 10: DDK2011 -------------------------------------------------------------------------------------------

def _bootstrap(design: np.ndarray, y: np.ndarray, clusters=None, reps: int = 1000, seed: int = 807) -> np.ndarray:
    """Bağımsız bootstrap döngüsü: satırlar ya da (sıralı) kümeler yerine koyarak, tohum 807."""

    rng = np.random.default_rng(seed)
    if clusters is not None:
        labels = np.unique(clusters)
        members = [np.flatnonzero(clusters == label) for label in labels]
    values = np.empty(reps)
    for b in range(reps):
        if clusters is None:
            index = rng.integers(0, len(y), len(y))
        else:
            index = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        values[b] = np.linalg.lstsq(design[index], y[index], rcond=None)[0][1]
    return values


def test_konu10_alternative_numbers_follow_from_an_independent_calculation() -> None:
    frame = _hansen("ddk2011").dropna(subset=["totalscore", "tracking", "schoolid"])
    assert len(frame) == 5795 and frame["schoolid"].nunique() == 121
    y = frame["totalscore"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(frame)), frame["tracking"].to_numpy(dtype=float)])
    hc1 = sm.OLS(y, design).fit(cov_type="HC1")
    clustered = sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(frame["schoolid"])[0]})
    assert _matches(hc1.params[1], *_expected(K10, 1, "Tracking katsayısı"))
    assert _matches(hc1.bse[1], *_expected(K10, 1, "Tracking katsayısının HC1 SH'si"))
    assert _matches(clustered.bse[1], *_expected(K10, 3, "Okul-kümeli analitik SH"))
    for step, label, draws in ((2, "Pairs bootstrap", _bootstrap(design, y)),
                               (3, "Küme bootstrap'ı", _bootstrap(design, y, frame["schoolid"].to_numpy()))):
        assert _matches(draws.std(ddof=1), *_expected(K10, step, f"{label} SH"))
        assert _matches(np.quantile(draws, 0.025), *_expected(K10, step, f"{label} percentile alt sınır"))
        assert _matches(np.quantile(draws, 0.975), *_expected(K10, step, f"{label} percentile üst sınır"))
    assert K10.ALT_TOLERANCES == {"boot": (0.04, 0.12), "kume_boot": (0.1, 0.37)}
    texts = _texts(K10)
    for number in ("5.795", "121", "1,258", "0,239", "0,236", "[0,798; 1,745]", "0,726", "0,704",
                   "[−0,200; 2,736]", "±0,04", "±0,12", "±0,10", "±0,37"):
        assert number in texts, number


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript bulunamadı.")
def test_konu10_code_notes_show_what_r_draws(tmp_path: Path) -> None:
    """Kod notlarındaki R sonuçları (set.seed(807)) R betiğinin gerçekten verdiği sayılardır."""

    _hansen("ddk2011")
    spec = K10.alternative()
    path = os.environ["IKT807_HANSEN_DDK2011_PATH"]
    script = render_script(spec, "R").replace("yerel_dosya <- NULL", f'yerel_dosya <- "{Path(path).as_posix()}"', 1)
    output = _check_script(spec, "R", script, tmp_path)
    found = re.findall(r"hedef: bootstrap SH = ([0-9.]+), percentile %95 GA \[(-?[0-9.]+); (-?[0-9.]+)\]", output)
    assert len(found) == 2
    for (se, low, high), key in zip(found, ("boot", "kume_boot")):
        shown = tuple(f"{float(value):.3f}".replace(".", ",").replace("-", "−") for value in (se, low, high))
        assert shown == K10.ALT_R_VALUES[key], (key, shown)


# --- Konu 11: Card1995, model seçimi ----------------------------------------------------------------------------

def _card11(card: pd.DataFrame) -> pd.DataFrame:
    """Bağımsız kurulum: örneklem, merkezlenmiş değişkenler ve 132 terim (uygulamanın kodundan ayrı)."""

    data = card.dropna(subset=["lwage76", *K11.ALT_RAW]).reset_index(drop=True).copy()
    data["exp"] = data["age76"] - data["ed76"] - 6
    data["exp2_100"] = data["exp"] ** 2 / 100
    for name, source, center in K11.ALT_CENTERS:
        data[name] = data[source] - center
    base = [name for name, _, _ in K11.ALT_CENTERS] + list(K11.ALT_BINARY)
    columns = {}
    for name, _, _ in K11.ALT_CENTERS:
        columns[f"{name}_2"], columns[f"{name}_3"] = data[name] ** 2, data[name] ** 3
    for i, a in enumerate(base):
        for b in base[i + 1:]:
            columns[f"{a}_x_{b}"] = data[a] * data[b]
    return pd.concat([data, pd.DataFrame(columns)], axis=1)


def test_konu11_alternative_numbers_follow_from_scikit_learn() -> None:
    from sklearn.linear_model import Lasso, LinearRegression, Ridge

    data = _card11(_hansen("card1995"))
    terms = list(dictionary_terms(K11.ALT_DICTIONARY))
    assert len(data) == 2034 and len(terms) == 132
    u = np.mod(np.arange(1, len(data) + 1) * (math.sqrt(5) - 1) / 2, 1.0)
    train, test = u >= 0.25, u < 0.25
    folds = np.floor((u - 0.25) / 0.15) + 1
    assert train.sum() == 1524 and test.sum() == 510
    y = data["lwage76"].to_numpy()

    def scaled(columns, fit_rows, apply_rows):
        x = data[columns].to_numpy(dtype=float)
        mean, sd = x[fit_rows].mean(axis=0), x[fit_rows].std(axis=0)
        return (x[fit_rows] - mean) / sd, (x[apply_rows] - mean) / sd

    def test_mse(model, columns) -> tuple[float, object]:
        xa, xb = scaled(columns, train, test)
        model.fit(xa, y[train])
        return float(np.mean((y[test] - model.predict(xb)) ** 2)), model

    def cv_choice(make, grid) -> float:
        errors = np.zeros(len(grid))
        for k in range(1, 6):
            fit_rows, valid = train & (folds != k), train & (folds == k)
            xa, xb = scaled(terms, fit_rows, valid)
            for g, lam in enumerate(grid):
                model = make(lam).fit(xa, y[fit_rows])
                errors[g] += np.mean((y[valid] - model.predict(xb)) ** 2) / 5
        return float(grid[int(np.argmin(errors))])

    ridge_grid = np.logspace(4, -2, 40)
    lasso_grid = np.logspace(-1, -4, 40)
    ridge_lambda = cv_choice(lambda lam: Ridge(alpha=lam), ridge_grid)
    lasso_lambda = cv_choice(lambda lam: Lasso(alpha=lam, tol=1e-10, max_iter=200_000), lasso_grid)
    basic, _ = test_mse(LinearRegression(), ["ed76", "exp", "exp2_100", "black"])
    rich, ols = test_mse(LinearRegression(), terms)
    ridge, _ = test_mse(Ridge(alpha=ridge_lambda), terms)
    lasso, fitted = test_mse(Lasso(alpha=lasso_lambda, tol=1e-10, max_iter=200_000), terms)
    xa, _ = scaled(terms, train, test)
    train_mse = float(np.mean((y[train] - ols.predict(xa)) ** 2))
    for label, value in (("Basit OLS test MSE", basic), ("Zengin OLS test MSE", rich), ("Ridge CV test MSE", ridge),
                         ("Lasso CV test MSE", lasso), ("Zengin OLS eğitim MSE", train_mse),
                         ("Ridge seçilen λ (SSE ölçeği)", ridge_lambda),
                         ("Lasso seçilen λ (yazılım ölçeği)", lasso_lambda)):
        assert _matches(value, *_expected(K11, 1, label)), (label, value)
    assert int(np.sum(np.abs(fitted.coef_) > 0)) == _expected(K11, 1, "Lasso sıfırdan farklı katsayı")[0]
    kept = {name for name, coef in zip(terms, fitted.coef_) if coef != 0}
    assert {"kww_m", "iq_m", "smsa76r", "reg76r", "enroll76"} <= kept
    assert (rich - ridge) / rich == pytest.approx(0.067, abs=0.005)
    texts = _texts(K11)
    for number in ("2.034", "1.524", "510", "0,1368", "0,1357", "0,1217", "0,1276", "0,1277", "%7", "39'unu",
                   "1.193,78", "0,01", "yaklaşık 12", "yaklaşık 1.270"):
        assert number in texts, number


# --- Konu 12: Card1995, DML -------------------------------------------------------------------------------------

def test_konu12_alternative_ols_rows_counts_and_texts() -> None:
    card = _hansen("card1995")
    data = card.dropna(subset=["lwage76", "ed76", *K12.ALT_CONTROLS])
    assert len(data) == 2997
    raw = sm.OLS(data["lwage76"], sm.add_constant(data[["ed76"]])).fit(cov_type="HC1")
    linear = sm.OLS(data["lwage76"], sm.add_constant(data[["ed76", *K12.ALT_CONTROLS]])).fit(cov_type="HC1")
    for label, value in (("Kontrolsüz regresyon: katsayı", raw.params["ed76"]),
                         ("Kontrolsüz regresyon: SH (HC1)", raw.bse["ed76"]),
                         ("Doğrusal kontrollü OLS: katsayı", linear.params["ed76"]),
                         ("Doğrusal kontrollü OLS: SH (HC1)", linear.bse["ed76"])):
        assert _matches(value, *_expected(K12, 1, label)), label
    assert len(dictionary_terms(K12.ALT_DICTIONARY)) == 42
    run = run_lab(K12.alternative(), {"card1995": card})
    assert all(item.passed for items in run.checks.values() for item in items)
    dml = run.state.models["dml"]
    u, v = dml.residuals["u"].to_numpy(), dml.residuals["v"].to_numpy()
    theta = float(v @ u / (v @ v))
    score = v * (u - theta * v)
    se = math.sqrt(float(score @ score) * len(u) / (len(u) - 1)) / float(v @ v)
    assert theta == pytest.approx(float(dml.params["theta"]), abs=1e-12)
    assert se == pytest.approx(float(dml.bse["theta"]), abs=1e-12)
    splits = run.state.tables["bolme_tablosu"]["theta"]
    assert float(splits.max() - splits.min()) < float(run.state.models["bolmeler"].bse["theta"]) / 2
    assert 0.034795 - 0.033012 == pytest.approx(0.0018, abs=5e-5)
    texts = _texts(K12)
    for number in ("2.997", "0,0522", "0,0348", "0,0330", "0,0031", "0,0018", "33 terim", "29 ile 37", "0,0320",
                   "0,0335", "0,0329", "0,0032", "0,0015", "0,1315", "0,0747", "1,8 katı", "42 terimli"):
        assert number in texts, number
    notes = {check.label: check.expected for step in LABS["konu04"].steps for check in step.checks}
    ols, iv = notes["OLS eğitim katsayısı"], notes["2SLS eğitim katsayısı"]  # Konu 4'ün notlardaki sayıları
    assert (ols, iv) == (0.0747, 0.1315) and round(iv / ols, 1) == 1.8


def test_konu12_dml_frame_keeps_rows_and_uses_people_as_units() -> None:
    spec = K12.alternative()
    ops = [op for step in spec.steps for op in step.operations]
    dml = next(op for op in ops if type(op).__name__ == "CrossFitDML")
    assert dml.cluster is None and dml.grid == (1.0, -4.0, 51)
    assert not {"exp", "kww", "iq", "smsa76r", "reg76r", "enroll76"} & set(dml.features)


# --- Alternatiflerin kodu sentetik veriyle ------------------------------------------------------------------------

def _synthetic_al1999() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for school in range(1, 401):
        enrollment = int(rng.integers(15, 76))
        classes = (enrollment - 1) // 40 + 1
        effect = rng.normal(0, 3)
        for classid in range(1, classes + 1):
            size = max(5, int(round(enrollment / classes + rng.normal(0, 2))))
            score = 70 + 0.05 * enrollment + 3 * (enrollment >= 41) + effect + rng.normal(0, 4)
            rows.append({"schlcode": school, "grade": 5, "classid": classid, "classize": size,
                         "enrollment": enrollment, "avgmath": score - 5, "avgverb": score,
                         "disadvantaged": float(rng.integers(0, 40))})
    return pd.DataFrame(rows)


def _synthetic_ddk2011() -> pd.DataFrame:
    rng = np.random.default_rng(12)
    schools = 60
    sizes = rng.integers(30, 50, schools)
    school = np.repeat(np.arange(1, schools + 1), sizes)
    tracking = np.repeat((np.arange(schools) % 2 == 0).astype(int), sizes)
    score = 20 + 1.5 * tracking + np.repeat(rng.normal(0, 4, schools), sizes) + rng.normal(0, 9, len(school))
    n = len(school)
    return pd.DataFrame({"pupilid": np.arange(n), "schoolid": school, "tracking": tracking, "sbm": 0, "girl":
                         rng.integers(0, 2, n), "agetest": 9.0, "etpteacher": 0, "lowstream": 0, "std_mark": 0.0,
                         "percentile": 50.0, "totalscore": score})


def _synthetic_card1995() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    n = 1500
    education = rng.integers(8, 19, n).astype(float)
    age = rng.integers(24, 35, n).astype(float)
    frame = pd.DataFrame({
        "ed76": education, "age76": age, "black": rng.integers(0, 2, n), "nearc4": rng.integers(0, 2, n),
        "smsa76r": rng.integers(0, 2, n), "reg76r": rng.integers(0, 2, n), "smsa66r": rng.integers(0, 2, n),
        "south66": rng.integers(0, 2, n), "momed": rng.integers(4, 17, n).astype(float),
        "daded": rng.integers(4, 17, n).astype(float), "momdad14": rng.integers(0, 2, n),
        "libcrd14": rng.integers(0, 2, n), "enroll76": (rng.random(n) < 0.1).astype(int),
        "kww": rng.integers(15, 56, n).astype(float), "iq": rng.normal(100, 15, n).round(),
    })
    log_wage = (5.2 + 0.05 * education + 0.03 * (age - education - 6) - 0.15 * frame["black"] + 0.003 * frame["iq"]
                + 0.01 * frame["momed"] + rng.normal(0, 0.35, n))
    frame["lwage76"] = log_wage
    frame["wage76"] = np.exp(log_wage)
    for i in range(2, 10):
        frame[f"reg66{i}"] = (rng.integers(1, 10, n) == i).astype(int)
    frame.loc[rng.choice(n, 300, replace=False), "iq"] = np.nan
    return frame


@pytest.mark.parametrize("module, dataset, build", [(K9, "al1999", _synthetic_al1999),
                                                    (K10, "ddk2011", _synthetic_ddk2011),
                                                    (K11, "card1995", _synthetic_card1995),
                                                    (K12, "card1995", _synthetic_card1995)],
                         ids=["konu09", "konu10", "konu11", "konu12"])
def test_alternative_code_reproduces_the_app_on_synthetic_data(module, dataset: str, build, tmp_path: Path) -> None:
    frame = build()
    spec = with_app_values(module.alternative(), {dataset: frame})
    if module is K10:
        spec = K10.with_tolerances(spec, run_lab(spec, {dataset: frame}))
    path = tmp_path / DATASET_MEMBERS[dataset]
    frame.to_stata(path, write_index=False, version=118)
    for language in ("Python", "R") if RSCRIPT else ("Python",):
        before, after = {"Python": ("YEREL_DOSYA = None", f'YEREL_DOSYA = "{path.as_posix()}"'),
                         "R": ("yerel_dosya <- NULL", f'yerel_dosya <- "{path.as_posix()}"')}[language]
        script = render_script(spec, language)
        assert before in script
        _check_script(spec, language, script.replace(before, after, 1), tmp_path)


STATA_INTEGERS = ((np.int8, -127, 100), (np.int16, -32767, 32740), (np.int32, -2147483647, 2147483620))
"""Stata byte, int ve long türlerinin veri aralıkları (``compress`` bu sırayla en küçüğünü seçer)."""


def _compressed(dataset: str, folder: Path) -> Path:
    """Gerçek dosyanın Stata ``compress`` karşılığı: tam sayı değerli, eksiksiz sütunlar en küçük tamsayı türüyle.
    Hansen'in arşivindeki dosyalar böyle saklanabilir (ör. Card1995'te yaş ve eğitim byte); elimizdeki kopya ondalıklı
    saklansa da uygulama ve üretilen kod bu biçimle de sınanır."""

    value = os.environ.get(f"IKT807_HANSEN_{dataset.upper()}_PATH")
    if not value or not Path(value).is_file():
        pytest.skip(f"Gerçek Hansen {DATASET_MEMBERS[dataset]} dosyası verilmedi.")
    frame = pd.read_stata(value, convert_categoricals=False)
    for column in frame.columns:
        values = frame[column]
        if values.dtype.kind != "f" or values.isna().any() or not np.all(np.mod(values, 1) == 0):
            continue
        for dtype, low, high in STATA_INTEGERS:
            if values.min() >= low and values.max() <= high:
                frame[column] = values.astype(dtype)
                break
    path = folder / DATASET_MEMBERS[dataset]
    frame.to_stata(path, write_index=False)
    return path


@pytest.mark.parametrize("module, dataset", [(K9, "al1999"), (K10, "ddk2011"), (K11, "card1995"), (K12, "card1995")],
                         ids=["konu09", "konu10", "konu11", "konu12"])
def test_alternative_is_robust_to_compressed_stata_storage(module, dataset: str, tmp_path: Path) -> None:
    path = _compressed(dataset, tmp_path)
    loaded = load_from_upload(dataset, path.read_bytes(), path.name)
    assert any(loaded.frame[column].dtype.kind in "iu" for column in loaded.frame.columns)
    run = run_lab(module.alternative(), {dataset: loaded.frame})
    failures = [f"{item.check.label}: {item.value}" for items in run.checks.values() for item in items if not item.passed]
    assert not failures, failures
    if module is K11:  # kuvvetler ve çarpımlar en çok burada: üretilen Python kodu da aynı dosyayla
        spec = module.alternative()
        script = render_script(spec, "Python").replace("YEREL_DOSYA = None", f'YEREL_DOSYA = "{path.as_posix()}"', 1)
        _check_script(spec, "Python", script, tmp_path)


# --- Kendi verin: Konu 9 ----------------------------------------------------------------------------------------

def test_konu09_cutoff_rules_and_the_treated_side() -> None:
    frame = K9.sample()
    with pytest.raises(K.UploadError, match="bir sayı yazın"):
        _case(K9, frame, CustomChoices(roles=KONU09.roles, numbers={"esik": "", "h": ""}))
    with pytest.raises(K.UploadError, match="sayı olarak okunamadı"):
        _case(K9, frame, CustomChoices(roles=KONU09.roles, numbers={"esik": "yetmiş", "h": ""}))
    with pytest.raises(K.UploadError, match="iki türlü okunabilir"):  # 5 ya da 5000
        _case(K9, frame, CustomChoices(roles=KONU09.roles, numbers={"esik": "70", "h": "5.000"}))
    with pytest.raises(K.UploadError, match="sayı olarak okunamadı"):  # İngilizce binlik yazım
        _case(K9, frame, CustomChoices(roles=KONU09.roles, numbers={"esik": "1,070.5", "h": ""}))
    with pytest.raises(K.UploadError, match="gözlenen aralığının"):
        _case(K9, frame, CustomChoices(roles=KONU09.roles, numbers={"esik": "101", "h": ""}))
    with pytest.raises(K.UploadError, match="pozitif"):
        _case(K9, frame, CustomChoices(roles=KONU09.roles, numbers={"esik": "70", "h": "-2"}))
    with pytest.raises(K.UploadError, match="dar pencerede"):
        _case(K9, frame, CustomChoices(roles=KONU09.roles, numbers={"esik": "70", "h": "0,4"}))
    with pytest.raises(K.UploadError, match="en az 10 küme"):
        few = frame.assign(Okul=frame["Okul"] % 6)
        _case(K9, few, CustomChoices(roles=KONU09.roles, numbers={"esik": "70", "h": "4"}))
    case = _case(K9, frame, CustomChoices(roles=KONU09.roles, numbers={"esik": "70,0", "h": "2,5"}))
    plan = K9.own_plan(case)
    assert plan.bandwidths == (1.25, 1.875, 2.5, 3.125, 3.75) and plan.cutoff == 70.0 and plan.right
    default = K9.own_plan(_case(K9, frame, KONU09))
    assert default.bandwidths == (1.4, 2.1, 2.8, 3.5, 4.2)
    left = _case(K9, frame, CustomChoices(roles=KONU09.roles, numbers={"esik": "70", "h": ""}, options={"sol": True}))
    spec = K9.CUSTOM.build(left)
    assert "etkisi $-\\widehat\\theta$" in spec.steps[0].explanation
    notes = _notes(spec)
    effects = re.search(r"tahmini etkisi (−?[0-9,]+) ile (−?[0-9,]+)", notes[2])
    assert effects and effects.group(1).startswith("−")


def test_konu09_sample_reproduces_in_both_languages_without_cluster(tmp_path: Path) -> None:
    spec, data = _own(K9, K9.sample(), CustomChoices(roles={k: v for k, v in KONU09.roles.items() if k != "kume"},
                                                      numbers={"esik": "70", "h": "4"}))
    ops = [op for step in spec.steps for op in step.operations]
    assert all(op.cluster is None for op in ops if isinstance(op, RDD))
    assert "HC1" in spec.steps[1].explanation
    _reproduces_in_both_languages(spec, data, tmp_path)


# --- Kendi verin: Konu 10 ---------------------------------------------------------------------------------------

def test_konu10_cluster_bootstrap_and_its_rules(tmp_path: Path) -> None:
    frame = K10.sample()
    spec, data = _own(K10, frame, KONU10)
    notes = _notes(spec)
    assert "sıfırı içerir" in notes[3] and "katı" in notes[3]
    assert all(check.mc_tolerance > 0 for step in spec.steps for check in step.checks if check.mc_tolerance)
    _reproduces_in_both_languages(spec, data, tmp_path)
    plain, _ = _own(K10, frame, CustomChoices(roles={"sonuc": "Matematik puanı", "hedef": "Program (1/0)"},
                                              extra=("Başlangıç puanı",)))
    assert not plain.steps[2].operations and "Küme seçilmedi" in _notes(plain)[3]
    with pytest.raises(K.UploadError, match="en az 20 küme"):
        _case(K10, frame.assign(Okul=frame["Okul"] % 10), KONU10)
    rare = frame.assign(**{"Program (1/0)": (np.arange(len(frame)) < 6).astype(int)})
    with pytest.raises(K.UploadError, match="iki değerli"):
        _case(K10, rare, CustomChoices(roles={"sonuc": "Matematik puanı", "hedef": "Program (1/0)"}))


# --- Kendi verin: Konu 11 ---------------------------------------------------------------------------------------

def test_konu11_dictionary_drops_constant_terms_and_enforces_its_limits(tmp_path: Path) -> None:
    frame = K11.sample()
    spec, data = _own(K11, frame, KONU11)
    design = K11.own_design(_case(K11, frame, KONU11))
    assert not design.excluded and len(design.rich) == 41 + 5 + 2  # 8 değişkenli sözlük, İlçe (6) ve Isıtma (3) göstergeleri
    assert [column for _, column, _ in design.indicators] == ["ilce"] * 5 + ["isitma"] * 2  # kod adları
    assert "her birinde ilk kategori referans" in spec.steps[0].explanation
    _reproduces_in_both_languages(spec, data, tmp_path)
    rare = frame.assign(Yeni=(np.arange(len(frame)) % 97 == 0).astype(int))
    choices = CustomChoices(roles=KONU11.roles, extra=(*KONU11.extra, "Yeni"))
    design = K11.own_design(_case(K11, rare, choices))
    assert design.excluded and all("Yeni".lower() in term or "yeni" in term for term in design.excluded)
    assert "sözlükten çıkarıldı" in K11.CUSTOM.build(_case(K11, rare, choices)).steps[0].explanation
    plain = K11.own_design(_case(K11, frame, CustomChoices(roles=KONU11.roles, extra=KONU11.extra,
                                                           options={"sozluk": False})))
    assert plain.dictionary is None and len(plain.rich) == 8 + 5 + 2
    six = CustomChoices(roles={"sonuc": "Satış fiyatı (bin TL)", "temel1": "Alan (m²)", "temel2": "Bina yaşı",
                               "temel3": "Oda sayısı", "temel4": "Kat", "temel5": "Merkeze uzaklık (km)",
                               "temel6": "Asansör (1/0)"}, extra=("Otopark (1/0)",))
    assert len(K11.own_design(_case(K11, frame, six)).basic) == 6
    with pytest.raises(K.UploadError, match="Kategorik değişkenler için farklı"):
        _case(K11, frame, CustomChoices(roles={**KONU11.roles, "kategori3": "İlçe"}, extra=KONU11.extra))
    small = frame.iloc[:100]
    no_category = CustomChoices(roles={k: v for k, v in KONU11.roles.items() if not k.startswith("kategori")},
                                extra=KONU11.extra)
    with pytest.raises(K.UploadError, match="zengin OLS için en az"):
        _case(K11, small, no_category)
    collinear = frame.assign(Kat=frame["Oda sayısı"] * 2 + 1)
    with pytest.raises(K.UploadError, match="tam doğrusal bağlantı"):
        _case(K11, collinear, KONU11)
    sparse = frame.assign(İlçe=np.where(np.arange(len(frame)) < 5, "G", frame["İlçe"]))
    with pytest.raises(K.UploadError, match="10'dan az gözlem"):
        _case(K11, sparse, KONU11)


# --- Kendi verin: Konu 12 ---------------------------------------------------------------------------------------

def test_konu12_clustered_folds_rules_and_reproduction(tmp_path: Path) -> None:
    frame = K12.sample()
    spec, data = _own(K12, frame, KONU12)
    run = run_lab(spec)
    folds = run.state.frames["veri"].groupby("firma")["dis_kat"].nunique()
    assert (folds == 1).all()  # bütün firma aynı dış katta
    notes = _notes(spec, run)
    assert "standart hatadan büyük" in notes[1]
    linear = float(run.state.models["ayarli"].params["kursa_katildi_1_0"])
    dml = float(run.state.models["dml"].params["theta"])
    assert abs(dml - 0.08) < abs(linear - 0.08)  # kurgusal veride gerçek etki 0,08
    _reproduces_in_both_languages(spec, data, tmp_path)
    with pytest.raises(K.UploadError, match="En az 2 kontrol"):
        _case(K12, frame, CustomChoices(roles=KONU12.roles, extra=("Yaş",)))
    with pytest.raises(K.UploadError, match="en çok 5.000"):
        _case(K12, pd.concat([frame] * 4, ignore_index=True), KONU12)
    with pytest.raises(K.UploadError, match="en az 20 küme"):
        _case(K12, frame.assign(Firma=frame["Firma"] % 8), KONU12)
    rare = frame.assign(**{"Kursa katıldı (1/0)": (np.arange(len(frame)) < 10).astype(int)})
    with pytest.raises(K.UploadError, match="iki değerli"):
        _case(K12, rare, KONU12)
