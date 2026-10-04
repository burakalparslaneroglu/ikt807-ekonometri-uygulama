"""Konu 3–4 ek kaynakları: alternatif örneklerin bağımsız hesabı, yeni hesap işlemleri ve kendi veri kuralları.

* DS2004 (Konu 3) ve AK1991 (Konu 4) alternatiflerinin bütün kontrol değerleri ve metinlerdeki sayılar, uygulamanın
  kodundan ayrı bir numpy hesabıyla doğrulanır (küme-dayanıklı ve HC1 sandviç, 2SLS elle). Dosya yolu verilmezse
  atlanır: ``IKT807_HANSEN_DS2004_PATH``, ``IKT807_HANSEN_AK1991_PATH``.
* ``GroupMean``, kategorik kontrollü 2SLS ve büyük tasarımların bellekte hafifletilmesi (``_slim``) birim testleriyle
  sınanır; iki alternatifin üretilen Python ve R kodu gerçek veri olmadan, sentetik bir ``.dta`` dosyasıyla da
  çalıştırılır (CI'da gerçek veri yoksa bu yeni işlemlerin kodu yine çalışır).
* Kendi verinde Konu 3 (küme yokken HC1, az küme, birlikte seçilen roller, ortak sütun, Adım 4 alt örneklemi) ve Konu 4
  (sıfır ilk aşama, zayıf araç, logaritmasız sonuç, boş hücreli kontrol) kuralları; üretilen Python ve R kodu aynı
  sayıları verir (R yoksa R adımları atlanır).
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from core.codegen.base import render_script, render_step, script_filename
from core.hansen_data import DATASET_MEMBERS, HansenDataError, load_from_upload
from core.labs import kendi_veri as K
from core.labs import ornek_konu03 as K3
from core.labs import ornek_konu04 as K4
from core.labs.ornek import CustomChoices, custom_case, with_app_values
from core.labs.runner import LARGE_DESIGN, LabState, execute, fit_iv, fit_ols, run_lab
from core.labs.spec import IV, OLS, BreuschPagan, GroupMean, LabSpec, Predict, Scalar

RSCRIPT = shutil.which("Rscript")
CSV = "veri.csv"
KONU03 = CustomChoices(
    roles={"sonuc": "Son sınav puanı", "tedavi": "Program", "atama": "Okul", "secilmis": "Düşük başlangıç grubu",
           "secim": "Başlangıç puanı"},
    extra=("Başlangıç puanı", "Kız öğrenci (1/0)", "Yaş"), picks={"tedavi": "Var", "secilmis": "Evet"})
KONU04 = CustomChoices(
    roles={"sonuc": "Saatlik ücret (TL)", "icsel": "Eğitim yılı", "arac": "Üniversiteye yakınlık (1/0)"},
    extra=("Deneyim (yıl)", "Kadın (1/0)", "Kentte yaşıyor (1/0)"))


def _tr(value: float, decimals: int) -> str:
    """Metinlerdeki sayı yazımı: ondalık virgül, eksi işareti U+2212."""

    text = f"{abs(value):.{decimals}f}".replace(".", ",")
    return ("−" + text) if value < 0 and float(text.replace(",", ".")) != 0 else text


def _texts(table: dict) -> str:
    parts: list[str] = []
    for value in table.values():
        parts.extend(value if isinstance(value, tuple) else (value,))
    return " ".join(part for part in parts if isinstance(part, str))


def _hansen(dataset: str) -> pd.DataFrame:
    value = os.environ.get(f"IKT807_HANSEN_{dataset.upper()}_PATH")
    if not value or not Path(value).is_file():
        pytest.skip(f"Gerçek Hansen {DATASET_MEMBERS[dataset]} dosyası verilmedi.")
    loaded = load_from_upload(dataset, Path(value).read_bytes(), Path(value).name)
    assert loaded.matches_hansen
    return loaded.frame


@pytest.fixture(scope="module")
def ds() -> pd.DataFrame:
    return _hansen("ds2004")


@pytest.fixture(scope="module")
def ak() -> pd.DataFrame:
    return _hansen("ak1991")


# --- Bağımsız hesap: sandviç kovaryanslar ------------------------------------------------------------------

def _design(data: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.column_stack([np.ones(len(data)), data[columns].to_numpy(dtype=float)])


def _cluster_ols(data: pd.DataFrame, y: str, xs: list[str], cluster: str) -> tuple[np.ndarray, np.ndarray]:
    """OLS ve küme-dayanıklı SH: G/(G−1)·(N−1)/(N−K) düzeltmesiyle (Stata ``vce(cluster)``)."""

    design = _design(data, xs)
    target = data[y].to_numpy(dtype=float)
    inverse = np.linalg.inv(design.T @ design)
    beta = inverse @ design.T @ target
    residual = target - design @ beta
    scores = pd.DataFrame(design * residual[:, None]).groupby(data[cluster].to_numpy()).sum().to_numpy()
    n, k = design.shape
    groups = len(scores)
    covariance = inverse @ scores.T @ scores @ inverse * groups / (groups - 1) * (n - 1) / (n - k)
    return beta, np.sqrt(np.diag(covariance))


def _hc1(design: np.ndarray, target: np.ndarray, bread: np.ndarray | None = None,
         residual: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """OLS (ya da verilen artıkla 2SLS) ve HC1 SH: (X'X)⁻¹ Σ xᵢxᵢ'êᵢ² (X'X)⁻¹ · n/(n−k)."""

    inverse = np.linalg.inv(design.T @ design)
    beta = inverse @ design.T @ target
    if residual is None:
        residual = target - design @ beta
    scores = design * residual[:, None]
    n, k = design.shape
    covariance = inverse @ (scores.T @ scores) @ inverse * n / (n - k)
    return beta, np.sqrt(np.diag(covariance))


def _p(estimate: float, se: float) -> float:
    return float(2 * stats.norm.sf(abs(estimate / se)))


# --- Konu 3 alternatifi: Di Tella ve Schargrodsky (2004) ------------------------------------------------------

def _ds_sample(ds: pd.DataFrame) -> pd.DataFrame:
    """Saldırı öncesi (Nisan–Haziran) blok ortalaması, ardından saldırı sonrası tam aylar (uygulamanın kodundan ayrı)."""

    data = ds.copy()
    data["thefts"] = data["thefts"].astype(float)
    before = data[data["month"] <= 6].groupby("block")["thefts"].mean()
    data["onceki"] = data["block"].map(before)
    data = data[data["month"] >= 8].reset_index(drop=True)
    data["once"] = (data["barrio"] == "Once").astype(float)
    data["crespo"] = (data["barrio"] == "V. Crespo").astype(float)
    data["yuksek"] = (data["onceki"] > 0).astype(float)
    return data


def test_konu03_alternative_numbers_follow_ds2004(ds: pd.DataFrame) -> None:
    assert len(ds) == 7884 and sorted(ds["month"].unique()) == list(range(4, 13))
    data = _ds_sample(ds)
    blocks = data.groupby("block").first()
    assert len(blocks) == 876 and len(data) == 4380 == 5 * len(blocks)
    assert int(blocks["sameblock"].sum()) == 37 and int(data["sameblock"].sum()) == 185
    assert set(blocks["barrio"]) == {"Belgrano", "Once", "V. Crespo"}
    assert (ds["thefts"] * 4 == (ds["thefts"] * 4).round()).all()  # kavşaktaki hırsızlık dört bloğa 0,25
    values = K3.ALT_EXPECTED
    assert values[(1, "Analiz örneklemi (N)")] == len(data) and values[(1, "Korunan blok-ay gözlemi")] == 185

    p_values = {}
    for variable, label in K3.ALT_BALANCE:
        beta, se = _cluster_ols(data, variable, ["sameblock"], "block")
        p_values[variable] = _p(beta[1], se[1])
        assert (round(beta[1], 4), round(se[1], 4), round(p_values[variable], 3)) == (
            values[(2, f"{label}: fark")], values[(2, f"{label}: küme SH")], values[(2, f"{label}: p-değeri")]), label

    raw, raw_se = _cluster_ols(data, "thefts", ["sameblock"], "block")
    controls = [variable for variable, _ in K3.ALT_BALANCE]
    adjusted, adjusted_se = _cluster_ols(data, "thefts", ["sameblock", *controls], "block")
    assert (round(raw[1], 4), round(raw_se[1], 4)) == (values[(3, "Ham fark: τ̂")], values[(3, "Ham fark: küme SH")])
    assert (round(adjusted[1], 4), round(adjusted_se[1], 4)) == (values[(3, "Kovaryat ayarlı: τ̂")],
                                                                 values[(3, "Kovaryat ayarlı: küme SH")])
    assert values[(3, "Ham fark: n")] == values[(3, "Kovaryat ayarlı: n")] == len(data.dropna(subset=controls))
    assert _p(raw[1], raw_se[1]) < 0.001 and _p(adjusted[1], adjusted_se[1]) < 0.001

    selected, selected_se = _cluster_ols(data, "onceki", ["yuksek"], "block")
    unprotected = data[data["sameblock"] == 0]
    reversion, reversion_se = _cluster_ols(unprotected, "thefts", ["yuksek"], "block")
    assert (round(selected[1], 4), round(selected_se[1], 4)) == (values[(4, "δ̂ (seçilmiş grup)")],
                                                                 values[(4, "Küme SH")])
    assert (round(reversion[1], 4), round(reversion_se[1], 4)) == (values[(4, "Sonuç farkı (seçilmiş grup)")],
                                                                   values[(4, "Sonuç farkı: küme SH")])

    texts = _texts(K3.ALT_TEXTS)
    means = data.groupby("sameblock")["thefts"].mean()
    others = [p_values[variable] for variable in ("onceki", "public", "gasstation", "bank")]
    groups = unprotected.groupby("yuksek")[["onceki", "thefts"]].mean()
    expected = [
        "876 bloktur", "(37 blok)", "876 × 5 = 4.380 blok-ay", "185'i",
        f"p-değerleri {_tr(min(others), 2)}–{_tr(max(others), 2)}",
        f"Once'de olma olasılığı {_tr(values[(2, 'Once mahallesi: fark')], 2)} daha yüksek (p < 0,001)",
        f"Ham fark {_tr(raw[1], 4)} (küme SH {_tr(raw_se[1], 4)})",
        f"ayda ortalama {_tr(means[1], 3)}, korunmayanlarda {_tr(means[0], 3)}",
        f"Kovaryat ayarlı tahmin {_tr(adjusted[1], 4)} ({_tr(adjusted_se[1], 4)})",
        f"%{_tr(100 * blocks['yuksek'].mean(), 1)}'i",
        f"ortalaması {_tr(selected[1], 4)} daha yüksek", f"fark {_tr(reversion[1], 4)}'e iner",
        f"{_tr(groups.loc[1.0, 'onceki'], 3)}'den {_tr(groups.loc[1.0, 'thefts'], 3)}'ya düşmüş",
        f"ötekilerinki {_tr(groups.loc[0.0, 'onceki'], 0)}'dan {_tr(groups.loc[0.0, 'thefts'], 3)}'ya çıkmıştır",
        "(G = 876 blok)", "her birine 0,25 olarak yazılır",
    ]
    for number in expected:
        assert number in texts, number
    assert p_values["once"] < 0.001 and groups.loc[0.0, "onceki"] == 0.0
    assert -adjusted[1] > means[0] / 2  # "korunmayan blokların ortalamasının yarısından fazlası"
    assert 0.15 < 1 - adjusted[1] / raw[1] < 0.25  # "yaklaşık beşte bir küçülür"
    for number in ("37 bloğa", "876 blok", "4.380 blok-ay"):
        assert number in K3.STORY, number


# --- Konu 4 alternatifi: Angrist ve Krueger (1991) ------------------------------------------------------------

def _ak_design(ak: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Kontroller: sabit, doğum yılı göstergeleri (ilk yıl referans), siyahi, metropol, bölge göstergeleri."""

    data = ak.astype({column: float for column in ak.columns}).copy()
    data["q1"] = (data["qob"] == 1).astype(float)
    columns = [np.ones(len(data))]
    columns += [(data["yob"] == level).to_numpy(dtype=float) for level in sorted(data["yob"].unique())[1:]]
    columns += [data["black"].to_numpy(), data["smsa"].to_numpy()]
    columns += [(data["region"] == level).to_numpy(dtype=float) for level in sorted(data["region"].unique())[1:]]
    return data, np.column_stack(columns)


def test_konu04_alternative_numbers_follow_ak1991(ak: pd.DataFrame) -> None:
    data, controls = _ak_design(ak)
    assert len(data) == 329509 and not data.isna().any().any()
    assert (data["yob"].min(), data["yob"].max()) == (1930, 1939)
    y, x, z = (data[name].to_numpy() for name in ("logwage", "edu", "q1"))
    ols, ols_se = _hc1(np.column_stack([x, controls]), y)
    first, first_se = _hc1(np.column_stack([z, controls]), x)
    reduced, reduced_se = _hc1(np.column_stack([z, controls]), y)
    regressors, instruments = np.column_stack([x, controls]), np.column_stack([z, controls])
    fitted = instruments @ np.linalg.lstsq(instruments, regressors, rcond=None)[0]
    beta = np.linalg.solve(fitted.T @ regressors, fitted.T @ y)
    _, iv_se = _hc1(fitted, y, residual=y - regressors @ beta)
    hand, hand_se = _hc1(np.column_stack([np.column_stack([z, controls]) @ first, controls]), y)
    f_stat = (first[0] / first_se[0]) ** 2
    # İlk aşamanın eğitim eşiklerine ayrıştırılması: π̂ = Σⱼ (q1'in 1{eğitim ≥ j} üzerindeki etkisi)
    design = np.column_stack([z, controls])
    thresholds = np.column_stack([(x >= j).astype(float) for j in range(1, 21)])
    margins = np.linalg.solve(design.T @ design, design.T @ thresholds)[0]
    assert x.min() >= 0 and x.max() <= 20 and abs(margins.sum() - first[0]) < 1e-10
    low = margins[:12].sum() / first[0]  # 1–12. eşikler (lise bitirme dahil)

    values = K4.ALT_EXPECTED
    assert values[(1, "Analiz örneklemi (N)")] == len(data)
    assert (round(ols[0], 4), round(ols_se[0], 4)) == (values[(2, "OLS eğitim katsayısı")], values[(2, "OLS HC1 SH")])
    assert (round(first[0], 4), round(first_se[0], 4)) == (values[(2, "İlk aşama katsayısı")],
                                                           values[(2, "İlk aşama HC1 SH")])
    assert (round(reduced[0], 4), round(reduced_se[0], 4)) == (values[(2, "İndirgenmiş biçim katsayısı")],
                                                               values[(2, "İndirgenmiş biçim HC1 SH")])
    assert (round(beta[0], 4), round(iv_se[0], 4)) == (values[(2, "2SLS eğitim katsayısı")], values[(2, "2SLS HC1 SH")])
    assert round(f_stat, 2) == values[(2, "İlk aşama F")]
    assert round(reduced[0] / first[0], 4) == values[(2, "Oran λ̂/π̂")] == round(beta[0], 4)
    assert round(100 * (np.exp(ols[0]) - 1), 1) == values[(2, "OLS yüzde etki")]
    assert round(100 * (np.exp(beta[0]) - 1), 1) == values[(2, "2SLS yüzde etki")]
    assert abs(hand[0] - beta[0]) < 1e-8 and round(hand[0], 4) == values[(4, "Elle ikinci aşama katsayısı")]

    texts = _texts(K4.ALT_TEXTS)
    expected = [
        "1930–1939 doğumlu", "329.509 erkektir", "(1930 referans)", f"yaklaşık {_tr(-first[0], 3)} yıl daha az",
        f"$F\\simeq{_tr(f_stat, 2).replace(',', '{,}')}$",
        f"\\hat\\lambda/\\hat\\pi=({reduced[0]:.5f})/({first[0]:.5f})\\approx{beta[0]:.4f}$".replace(".", "{,}"),
        f"yaklaşık %{_tr(100 * (np.exp(ols[0]) - 1), 1)}, 2SLS'ninki %{_tr(100 * (np.exp(beta[0]) - 1), 1)}",
        f"standart hatası ({_tr(iv_se[0], 4)}) OLS'ninkinin ({_tr(ols_se[0], 5)}) yaklaşık "
        f"{int(round(iv_se[0] / ols_se[0], -1))} katıdır",
        f"yaklaşık %{round(100 * low)}'ü 12 yıl ve altındaki eşiklerden, %{round(100 * (1 - low))}'sı lise sonrası",
        f"standart hatası ({_tr(hand_se[0], 4)}) 2SLS'ninkinden ({_tr(iv_se[0], 4)}) biraz büyüktür",
    ]
    for number in expected:
        assert number in texts, number
    assert hand_se[0] > iv_se[0]
    assert "329.509 erkek" in K4.STORY


# --- Yeni hesap işlemleri -------------------------------------------------------------------------------------

def test_group_mean_uses_only_the_rows_that_meet_the_condition() -> None:
    frame = pd.DataFrame({"blok": [1, 1, 1, 2, 2, 3, 3], "ay": [4, 5, 8, 6, 9, 8, 9],
                          "y": [1.0, np.nan, 5.0, 2.0, 7.0, 3.0, 4.0]})
    state = LabState(frames={"d": frame.copy()})
    execute(GroupMean("d", "once", "y", "blok", "Saldırı öncesi ortalama", ("ay", "<=", 6)), state, {})
    execute(GroupMean("d", "hepsi", "y", "blok", "Bütün aylar"), state, {})
    result = state.frames["d"]
    assert result["once"].iloc[:5].tolist() == [1.0, 1.0, 1.0, 2.0, 2.0]  # eksik değer ortalamaya girmez
    assert result["once"].iloc[5:].isna().all()  # koşulu sağlayan satırı olmayan blok: eksik
    assert result["hepsi"].tolist() == [3.0, 3.0, 3.0, 4.5, 4.5, 3.5, 3.5]
    with pytest.raises(ValueError, match="Desteklenmeyen işleç"):
        GroupMean("d", "x", "y", "blok", "Yorum", ("ay", "~", 6))
    # Eksik grup ya da koşul değeri dillerde farklı işlenir: uygulama hesaplamaz.
    for column in ("blok", "ay"):
        missing = frame.astype({column: float})
        missing.loc[2, column] = np.nan
        with pytest.raises(ValueError, match=f"eksik değer olmamalı: {column}"):
            execute(GroupMean("d", "once", "y", "blok", "Yorum", ("ay", "<=", 6)), LabState(frames={"d": missing}), {})


def test_group_mean_and_categorical_iv_code_in_three_languages() -> None:
    step1 = {language: render_step(K3.alternative(), 1, language) for language in ("Python", "R", "Stata")}
    assert 'ds["onceki"] = ds["thefts"].where(secili).groupby(ds["block"]).transform("mean")' in step1["Python"]
    assert "ave(ifelse(ds$month <= 6, ds$thefts, NA), ds$block," in step1["R"]
    assert "mean(x, na.rm = TRUE)" in step1["R"]
    assert "egen double onceki = mean(cond(month <= 6, thefts, .)), by(block)" in step1["Stata"]
    step2 = {language: render_step(K4.alternative(), 2, language) for language in ("Python", "R", "Stata")}
    assert '"logwage ~ 1 + C(yob) + black + smsa + C(region) + [edu ~ q1]"' in step2["Python"]
    assert ("ivreg(logwage ~ edu + factor(yob) + black + smsa + factor(region) | q1 + factor(yob) + black + smsa + "
            "factor(region)") in step2["R"]
    assert "ivregress 2sls logwage i.yob black smsa i.region (edu = q1), vce(robust) small" in step2["Stata"]
    for code in step2.values():
        assert "yob, region kategorik: her düzey için bir kukla, ilk düzey referans" in code


def test_iv_with_categorical_controls_matches_dummy_coding() -> None:
    rng = np.random.default_rng(3)
    n = 400
    frame = pd.DataFrame({"g": rng.choice([3, 1, 2], n), "w": rng.normal(size=n), "z": rng.normal(size=n)})
    frame["x"] = 0.8 * frame["z"] + 0.3 * frame["w"] + 0.5 * (frame["g"] == 2) + rng.normal(size=n)
    frame["y"] = 1.5 * frame["x"] - 0.4 * frame["w"] + 0.7 * (frame["g"] == 3) + rng.normal(size=n)
    fit = fit_iv(IV("iv", "d", "y", ("x",), ("z",), ("g", "w"), categorical=("g",)), frame)
    dummies = np.column_stack([(frame["g"] == level).to_numpy(dtype=float) for level in (2, 3)])  # 1 referans
    controls = np.column_stack([np.ones(n), dummies, frame["w"].to_numpy()])
    regressors = np.column_stack([frame["x"].to_numpy(), controls])
    instruments = np.column_stack([frame["z"].to_numpy(), controls])
    fitted = instruments @ np.linalg.lstsq(instruments, regressors, rcond=None)[0]
    y = frame["y"].to_numpy()
    beta = np.linalg.solve(fitted.T @ regressors, fitted.T @ y)
    _, se = _hc1(fitted, y, residual=y - regressors @ beta)
    assert np.isclose(fit.params["x"], beta[0]) and np.isclose(fit.bse["x"], se[0])
    assert {"C(g)[T.2]", "C(g)[T.3]"} <= set(fit.params.index) and "C(g)[T.1]" not in fit.params.index
    with pytest.raises(ValueError, match="dışsal değişkenler arasında"):
        IV("iv", "d", "y", ("x",), ("z",), ("w",), categorical=("g",))


def test_large_designs_keep_their_results_but_not_the_design_matrix() -> None:
    """AK1991'deki gibi büyük tasarımda (n·k ≥ 3 milyon) model sonuçları korunur, tasarım matrisi bırakılır; tasarımı
    yeniden isteyen işlem açık bir hatayla durur. Notlardaki ve kendi verindeki tasarımlar bu eşiğin altındadır."""

    rng = np.random.default_rng(11)
    names = tuple(f"x{j}" for j in range(9))
    n = LARGE_DESIGN // (len(names) + 1) + 1
    frame = pd.DataFrame({name: rng.normal(size=n) for name in names})
    frame["y"] = frame[list(names)].sum(axis=1) + rng.normal(size=n)
    op = OLS("m", "d", "y", names, vcov="HC1")
    result = fit_ols(op, frame)
    assert result.model.slim and "exog" not in result.model.__dict__ and result.model.data.exog is None
    reference = sm.OLS(frame["y"], sm.add_constant(frame[list(names)])).fit(cov_type="HC1")
    np.testing.assert_allclose(result.params.to_numpy(), reference.params.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(result.bse.to_numpy(), reference.bse.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(result.resid.to_numpy(), reference.resid.to_numpy(), atol=1e-10)
    assert result.nobs == n and np.isclose(result.rsquared, reference.rsquared)
    state = LabState(frames={"d": frame}, models={"m": result})
    execute(Predict("m", "d", "uyum", "fitted"), state, {})
    np.testing.assert_allclose(state.frames["d"]["uyum"].to_numpy(), reference.fittedvalues.to_numpy(), atol=1e-10)
    with pytest.raises(ValueError, match="tasarım matrisi bellekte tutulmadı"):
        execute(BreuschPagan("m", "bp"), state, {})
    classic = fit_ols(OLS("k", "d", "y", names), frame)  # klasik SH'li büyük model: HC1 SH de önbellekte
    assert classic.model.slim
    np.testing.assert_allclose(classic.HC1_se.to_numpy(), reference.bse.to_numpy(), rtol=1e-10)
    small = fit_ols(op, frame.iloc[:2000])
    assert not getattr(small.model, "slim", False) and small.model.exog is not None
    assert K.MAX_ROWS * (K.MAX_EXTRA + 20) < LARGE_DESIGN  # kendi verinde tasarım eşiğe ulaşmaz


def test_new_hansen_files_are_checked_by_their_columns() -> None:
    blocks = pd.DataFrame({"block": [1, 2], "sameblock": [1, 0], "distance": [0, 3], "public": [0, 1],
                           "gasstation": [0, 0], "bank": [1, 0], "thefts": [0.0, 0.25], "month": [4, 4],
                           "oneblock": [0, 1]})

    def stata(frame: pd.DataFrame) -> bytes:
        buffer = io.BytesIO()
        frame.to_stata(buffer, write_index=False, version=118)
        return buffer.getvalue()

    with pytest.raises(HansenDataError, match="barrio") as error:
        load_from_upload("ds2004", stata(blocks), "DS2004.dta")
    assert "öğretim CSV" not in str(error.value) and "DS2004.dta" in str(error.value)
    loaded = load_from_upload("ds2004", stata(blocks.assign(barrio=["Once", "Belgrano"])), "DS2004.dta")
    assert not loaded.matches_hansen and loaded.frame["barrio"].tolist() == ["Once", "Belgrano"]
    with pytest.raises(HansenDataError, match="qob") as error:
        load_from_upload("ak1991", stata(pd.DataFrame({"edu": [12], "logwage": [5.9]})), "AK1991.dta")
    assert "öğretim CSV" not in str(error.value)


# --- Kendi verin --------------------------------------------------------------------------------------------

def _csv(frame: pd.DataFrame) -> bytes:
    """Türkçe Excel'in "CSV (noktalı virgülle ayrılmış)" kaydı: R betiği readxl istemeden okur."""

    return frame.to_csv(sep=";", decimal=",", index=False).encode("cp1254")


def _own(module, frame: pd.DataFrame, choices: CustomChoices) -> tuple[LabSpec, tuple[str, bytes]]:
    data = _csv(frame)
    case, _ = custom_case(module.CUSTOM, K.read_upload(CSV, data), choices)
    return module.CUSTOM.build(case), (CSV, data)


def _case(module, frame: pd.DataFrame, choices: CustomChoices):
    return custom_case(module.CUSTOM, K.read_upload(CSV, _csv(frame)), choices)[0]


def _check_script(spec: LabSpec, language: str, script: str, folder: Path) -> None:
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


def _reproduces_in_both_languages(spec: LabSpec, data: tuple[str, bytes], folder: Path) -> None:
    (folder / data[0]).write_bytes(data[1])
    for language in ("Python", "R") if RSCRIPT else ("Python",):
        _check_script(spec, language, render_script(spec, language), folder)


def _notes(spec: LabSpec, run=None) -> dict[int, str]:
    run = run_lab(spec) if run is None else run
    assert all(item.passed for items in run.checks.values() for item in items)
    return {step.number: step.note_for(run.state) for step in spec.steps if step.note_for is not None}


def test_konu03_sample_clusters_by_school_and_restricts_step_4_to_the_treated() -> None:
    frame = K3.sample()
    spec, _ = _own(K3, frame, KONU03)
    models = [op for step in spec.steps for op in step.operations if isinstance(op, OLS)]
    assert all(op.vcov == "cluster" and op.cluster == "okul" for op in models)
    selected = next(op for op in models if op.name == "secilmis")
    assert selected.where == (K3.T01, 1.0) and selected.outcome == "baslangic_puani"
    balance = [op.outcome for op in models if op.name.startswith("d_")]
    assert balance == ["baslangic_puani", "kiz_ogrenci_1_0", "yas"]  # ortak sütun denge tablosunda da var
    run = run_lab(spec)
    notes = _notes(spec, run)
    assert f"{frame['Okul'].nunique()} küme" in notes[1] and "yalnız tedavi grubunda" in notes[4]
    complete = frame.dropna(subset=["Yaş"])
    assert f"({K.thousands(len(frame))} → {K.thousands(len(complete))})" in notes[3]
    assert int((frame["Program"] == "Var").sum()) == int(run.state.frames["veri"][K3.T01].sum())


def test_konu03_without_an_assignment_unit_uses_hc1(tmp_path: Path) -> None:
    choices = CustomChoices(roles={**KONU03.roles, "atama": None}, extra=KONU03.extra, picks=KONU03.picks)
    spec, data = _own(K3, K3.sample(), choices)
    models = [op for step in spec.steps for op in step.operations if isinstance(op, OLS)]
    assert models and all(op.vcov == "HC1" and op.cluster is None for op in models)
    labels = [check.label for step in spec.steps for check in step.checks]
    assert "Ham fark: HC1 SH" in labels and "HC1 SH" in labels and not any("küme" in label for label in labels)
    assert spec.steps[2].code_note == K3.P_VALUE_NOTE_HC1
    assert "HC1 standart hatası kullanılır" in _notes(spec)[1]
    _reproduces_in_both_languages(spec, data, tmp_path)


def test_konu03_selected_group_outside_the_treated_uses_the_whole_sample(tmp_path: Path) -> None:
    frame = K3.sample()
    low = frame["Başlangıç puanı"] < frame["Başlangıç puanı"].median()
    frame["Düşük başlangıç grubu"] = np.where(frame["Program"] == "Yok", np.where(low, "Evet", "Hayır"), None)
    spec, data = _own(K3, frame, KONU03)
    selected = next(op for step in spec.steps for op in step.operations if isinstance(op, OLS) and op.name == "secilmis")
    assert selected.where is None
    assert "bütün örneklemde (gösterge tedavi grubunda iki kategoriyi birden almadığı için)" in _notes(spec)[4]
    _reproduces_in_both_languages(spec, data, tmp_path)


def test_konu03_role_rules_are_enforced() -> None:
    frame = K3.sample()
    with pytest.raises(K.UploadError, match="3 küme var"):
        _case(K3, frame.assign(Okul=frame["Okul"] % 3), KONU03)
    with pytest.raises(K.UploadError, match="birlikte çalışır"):
        _case(K3, frame, CustomChoices(roles={**KONU03.roles, "secim": None}, extra=KONU03.extra, picks=KONU03.picks))
    with pytest.raises(K.UploadError, match="hem bir rol için hem ek sütun"):
        _case(K3, frame, CustomChoices(roles=KONU03.roles, extra=("Son sınav puanı",), picks=KONU03.picks))
    with pytest.raises(K.UploadError, match="yalnız bir grupta gözleniyor"):
        only = frame.assign(**{"Ek puan": np.where(frame["Program"] == "Var", frame["Başlangıç puanı"], np.nan)})
        _case(K3, only, CustomChoices(roles=KONU03.roles, extra=("Ek puan",), picks=KONU03.picks))
    with pytest.raises(K.UploadError, match="tam doğrusal bağlantı"):
        double = frame.assign(**{"İki kat": 2 * frame["Başlangıç puanı"] + 1})
        _case(K3, double, CustomChoices(roles=KONU03.roles, extra=("Başlangıç puanı", "İki kat"),
                                        picks=KONU03.picks))
    case = _case(K3, frame, CustomChoices(roles={**KONU03.roles, "secilmis": None, "secim": None},
                                          extra=KONU03.extra, picks=KONU03.picks))
    spec = K3.build(case)
    assert not spec.steps[3].operations and "Seçilmiş grup göstergesi" in _notes(spec)[4]


def test_konu03_suggests_roles_from_column_names() -> None:
    table = K.read_upload(CSV, _csv(K3.sample()))
    assert K3.suggest(table) == {"tedavi": "Program", "atama": "Okul", "secilmis": "Düşük başlangıç grubu",
                                 "secim": "Başlangıç puanı"}
    renamed = K.read_upload(CSV, _csv(K3.sample().rename(columns={"Başlangıç puanı": "Puan"})))
    assert K3.suggest(renamed) == {"tedavi": "Program", "atama": "Okul"}  # seçilmiş grup ve dayandığı değişken birlikte
    # Adında "grubu" geçen tedavi sütunu seçilmiş grup sanılmaz; az kümeli birim (3 sınıf) atama birimi önerilmez.
    frame = K3.sample().rename(columns={"Program": "Deney grubu"})
    frame.insert(1, "Sınıf", np.arange(len(frame)) % 3 + 1)
    hints = K3.suggest(K.read_upload(CSV, _csv(frame)))
    assert hints == {"tedavi": "Deney grubu", "atama": "Okul", "secilmis": "Düşük başlangıç grubu",
                     "secim": "Başlangıç puanı"}
    # Tedavi öncesi değişken ve "Kontrol grubu" tedavi önerilmez (1 ile kodlanan kategori karşılaştırma grubu olurdu);
    # boş hücreli birim atama birimi önerilmez (rol boş hücre kabul etmez).
    frame = K3.sample()
    frame.insert(0, "Program öncesi kurs", np.where(np.arange(len(frame)) % 2, "Var", "Yok"))
    frame["Kontrol grubu"] = np.where(frame["Program"] == "Yok", "Evet", "Hayır")
    frame["Okul"] = frame["Okul"].astype(float).where(np.arange(len(frame)) != 5)
    hints = K3.suggest(K.read_upload(CSV, _csv(frame)))
    assert hints.get("tedavi") == "Program" and "atama" not in hints


def test_konu04_rejects_an_instrument_unrelated_to_the_endogenous_variable() -> None:
    frame = K4.sample()
    controls = list(KONU04.extra)
    rng = np.random.default_rng(1)
    design = np.column_stack([np.ones(len(frame)), frame[controls + ["Eğitim yılı"]].to_numpy(dtype=float)])
    noise = rng.normal(size=len(frame))
    frame["Rastgele araç"] = noise - design @ np.linalg.lstsq(design, noise, rcond=None)[0]
    choices = CustomChoices(roles={**KONU04.roles, "arac": "Rastgele araç"}, extra=KONU04.extra)
    with pytest.raises(K.UploadError, match="ilk aşama katsayısı sıfır"):
        _case(K4, frame, choices)


def test_konu04_weak_instrument_is_flagged_and_reproduced(tmp_path: Path) -> None:
    frame = K4.sample()
    education = frame["Eğitim yılı"].astype(float)
    rng = np.random.default_rng(5)
    frame["Zayıf araç"] = (rng.normal(size=len(frame)) + 0.05 * (education - education.mean()) / education.std()).round(3)
    spec, data = _own(K4, frame, CustomChoices(roles={**KONU04.roles, "arac": "Zayıf araç"}, extra=KONU04.extra))
    run = run_lab(spec)
    assert 1 < run.state.scalars["ilk_asama_F"] < K4.WEAK_F
    assert "F 10'un altında: araç zayıf olabilir" in spec.steps[1].note_for(run.state)
    _reproduces_in_both_languages(spec, data, tmp_path)


def test_konu04_outcome_without_logarithm_and_a_blank_control(tmp_path: Path) -> None:
    frame = K4.sample()
    frame["Saatlik ücret (TL)"] = frame["Saatlik ücret (TL)"] - 60  # negatif değerler: logaritma alınamaz
    frame["Deneyim (yıl)"] = frame["Deneyim (yıl)"].astype(float).where(np.arange(len(frame)) % 25 != 0)
    with pytest.raises(K.UploadError, match="logaritma alınamaz"):
        _case(K4, frame, KONU04)
    spec, data = _own(K4, frame, CustomChoices(roles=KONU04.roles, extra=KONU04.extra, options={"log": False}))
    scalars = {op.name for step in spec.steps for op in step.operations if isinstance(op, Scalar)}
    assert scalars == {"ilk_asama_F", "oran"} and "log_sonuc" not in render_script(spec, "Python")
    run = run_lab(spec)
    kept = int(frame["Deneyim (yıl)"].notna().sum())
    assert len(run.state.frames["veri"]) == kept  # boş kontrollü satırlar bütün modellerden çıkar
    assert {int(run.state.models[name].nobs) for name in ("ols", "ilk", "indirgenmis", "iv", "elle")} == {kept}
    _reproduces_in_both_languages(spec, data, tmp_path)


def test_konu04_role_rules_are_enforced() -> None:
    frame = K4.sample()
    with pytest.raises(K.UploadError, match="üç farklı sütun"):
        _case(K4, frame, CustomChoices(roles={**KONU04.roles, "arac": "Eğitim yılı"}, extra=KONU04.extra))
    with pytest.raises(K.UploadError, match="tam doğrusal bağlantı"):
        collinear = frame.assign(Kopya=2 * frame["Üniversiteye yakınlık (1/0)"] + 1)
        _case(K4, collinear, CustomChoices(roles=KONU04.roles, extra=("Kopya", "Deneyim (yıl)")))
    with pytest.raises(K.UploadError, match="hem bir rol için hem ek sütun"):
        _case(K4, frame, CustomChoices(roles=KONU04.roles, extra=("Eğitim yılı",)))


def test_konu03_every_clustered_sample_needs_enough_clusters() -> None:
    frame = K3.sample()
    first = frame.loc[frame["Program"] == "Var", "Okul"].iloc[0]
    single = frame[(frame["Program"] == "Yok") | (frame["Okul"] == first)]  # tek program okulu
    with pytest.raises(K.UploadError, match=r"Adım 4 \(seçilmiş grup\) örnekleminde 1 küme var.*Adım 1–3"):
        _case(K3, single, KONU03)
    without = CustomChoices(roles={**KONU03.roles, "secilmis": None, "secim": None}, extra=KONU03.extra,
                            picks=KONU03.picks)
    assert K3.build(_case(K3, single, without)).steps[0].operations  # Adım 4 rolleri kaldırılınca çalışır
    with pytest.raises(K.UploadError, match="tedavi göstergesinden farklı"):
        _case(K3, frame, CustomChoices(roles={**KONU03.roles, "secilmis": "Program"}, extra=KONU03.extra,
                                       picks=KONU03.picks))
    with pytest.raises(K.UploadError, match="sonuç değişkenini seçmeyin"):
        _case(K3, frame, CustomChoices(roles={**KONU03.roles, "secim": "Son sınav puanı"}, extra=KONU03.extra,
                                       picks=KONU03.picks))


def test_konu04_rejects_an_endogenous_variable_fully_explained_by_the_instrument() -> None:
    frame = K4.sample()
    frame["Eğitim yılı"] = 10 + 2 * frame["Üniversiteye yakınlık (1/0)"] + 0.1 * frame["Deneyim (yıl)"]
    with pytest.raises(K.UploadError, match="ilk aşama tam uyumludur"):
        _case(K4, frame, KONU04)


# --- Alternatiflerin kodu sentetik veriyle (gerçek veri gerekmez) ------------------------------------------------

def _synthetic_ds2004() -> pd.DataFrame:
    """DS2004 düzeninde kurgusal blok × ay verisi: 48 blok, Nisan–Aralık; hırsızlık 0,25'in katları; birkaç eksik kayıt
    (ortalamaya ve modellere girmez). Seçilmiş grup göstergesi iki değeri de alır."""

    rng = np.random.default_rng(3)
    protected = {1, 2, 9, 14, 21, 27, 35, 40}
    rows = []
    for block in range(1, 49):
        traits = {"public": int(rng.random() < 0.3), "gasstation": int(rng.random() < 0.2),
                  "bank": int(rng.random() < 0.25)}
        risk = 0.05 if block % 2 else 0.5  # düşük riskli blokların yaklaşık yarısında saldırı öncesi kayıt yok
        for month in range(4, 13):
            rate = risk * (0.4 if block in protected and month >= 8 else 1.0)
            rows.append({"block": block, "barrio": ("Belgrano", "Once", "V. Crespo")[block % 3], "calle": "Calle",
                         "altura": 100 * block, "sameblock": int(block in protected), "distance": block % 4,
                         **traits, "thefts": 0.25 * rng.poisson(4 * rate), "month": month,
                         "oneblock": int(block % 7 == 0)})
    frame = pd.DataFrame(rows)
    frame.loc[[3, 40, 200, 333], "thefts"] = np.nan  # Temmuz, Ağustos, Haziran ve Nisan satırları
    before = frame[frame["month"] <= 6].groupby("block")["thefts"].mean()
    assert 10 < (before > 0).sum() < 38  # her iki grupta da yeterli blok
    return frame


def _synthetic_ak1991() -> pd.DataFrame:
    """AK1991 düzeninde kurgusal veri: 4.000 erkek; araç ilk çeyrek, kategorik doğum yılı ve bölge."""

    rng = np.random.default_rng(4)
    n = 4000
    qob, yob, region = rng.integers(1, 5, n), rng.integers(1930, 1940, n), rng.integers(0, 9, n)
    black, smsa = (rng.random(n) < 0.1).astype(int), (rng.random(n) < 0.7).astype(int)
    ability = rng.normal(size=n)
    edu = np.clip(np.round(12 - 0.5 * (qob == 1) + 0.8 * ability + 0.5 * smsa + rng.normal(size=n)), 0, 20)
    logwage = 5.2 + 0.08 * edu + 0.15 * ability - 0.2 * black + 0.1 * smsa + 0.01 * (yob - 1930) \
        + rng.normal(0, 0.5, n)
    return pd.DataFrame({"ageq": 1980.0 - yob - qob / 4, "edu": edu.astype(int), "logwage": logwage,
                         "married": (rng.random(n) < 0.8).astype(int), "state": rng.integers(1, 57, n), "qob": qob,
                         "black": black, "smsa": smsa, "yob": yob, "region": region})


@pytest.mark.parametrize("module, dataset, build", [(K3, "ds2004", _synthetic_ds2004), (K4, "ak1991", _synthetic_ak1991)],
                         ids=["konu03", "konu04"])
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
