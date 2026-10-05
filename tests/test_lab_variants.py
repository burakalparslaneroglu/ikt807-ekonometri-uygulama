"""Uygulama sekmesinin ek veri kaynakları: alternatif örnek ve "Kendi verini yükle" (``core.labs.ornekler``).

Kayıttaki her konu kendiliğinden kapsanır:

* notlardaki laboratuvarların üretilen kodu değişmez (``NOTES_MD5``);
* alternatif örneğin adımları notlarla aynı numaralı ve aynı başlıklıdır, her kontrolün beklenen değeri vardır;
* alternatif örneğin sayıları gerçek Hansen verisiyle uygulamada, bağımsız bir hesapta ve üretilen Python ve R kodunda
  aynıdır (veri yolu verilmezse atlanır: ``IKT807_HANSEN_<VERİ>_PATH``, ör. ``IKT807_HANSEN_CARD1995_PATH``);
* kendi verinde örnek dosya ve elle hazırlanmış dosyalar (Türkçe CSV, işaretli sütun adları, ayrılmış adlar, tam uyum,
  logaritması alınamayan sonuç) uygulamada ve iki dilde aynı sayıları verir; kullanılamayan seçimler açık bir iletiyle
  reddedilir. Excel okuyan R betikleri ``readxl`` ister; R ya da paket yoksa yalnız R adımları atlanır.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from core.codegen.base import LANGUAGES, languages_for, render_script, render_step, script_filename
from core.hansen_data import DATASET_MEMBERS, TEACHING_CSV, load_from_upload
from core.labs import kendi_veri as K
from core.labs import ornek_konu01, ornek_konu02
from core.labs.ornek import SOURCE_LABELS, CustomChoices, custom_case, md
from core.labs.ornekler import VARIANTS
from core.labs.registry import LABS
from core.labs.runner import run_lab
from core.labs.spec import (
    CoefTarget,
    DeltaMethod,
    LabSpec,
    ReadFile,
    Scalar,
    ScalarTarget,
)

TOPICS = sorted(VARIANTS)
NOTES_MD5 = "e6eb2b280607de0fe2a5b242b5c5a38d"
"""Notlardaki 12 laboratuvarın üç dildeki bütün betik ve adım kodunun özeti. Notların kodu bilerek değiştirilirse bu
değer de güncellenir. Son bilerek değişiklik (Blok B sonrası): Konu 4 (Card1995) ve Konu 6 (CHJ2004) betiklerinden
"öğretim CSV'si de verilebilir" cümlesi kaldırıldı; bu iki öğretim CSV'sinde laboratuvarın ham değişkenleri yoktur.
Önceki değer: 5fce77e8713442f6e9db5b93886716d1."""
SAMPLE_CHOICES = {
    "konu01": CustomChoices(
        roles={"sonuc": "Saatlik ücret (TL)", "aciklayici": "Eğitim yılı", "kontrol": "Deneyim (yıl)",
               "grup": "Cinsiyet"},
        picks={"grup": "Kadın"}),
    "konu02": CustomChoices(
        roles={"sonuc": "Saatlik ücret (TL)", "aciklayici": "Eğitim yılı", "grup": "Cinsiyet"},
        extra=("Deneyim (yıl)", "Evli (1/0)", "Kıdem (yıl)"), picks={"grup": "Kadın"}),
    "konu03": CustomChoices(
        roles={"sonuc": "Son sınav puanı", "tedavi": "Program", "atama": "Okul", "secilmis": "Düşük başlangıç grubu",
               "secim": "Başlangıç puanı"},
        extra=("Başlangıç puanı", "Kız öğrenci (1/0)", "Yaş"), picks={"tedavi": "Var", "secilmis": "Evet"}),
    "konu04": CustomChoices(
        roles={"sonuc": "Saatlik ücret (TL)", "icsel": "Eğitim yılı", "arac": "Üniversiteye yakınlık (1/0)"},
        extra=("Deneyim (yıl)", "Kadın (1/0)", "Kentte yaşıyor (1/0)")),
    "konu05": CustomChoices(
        roles={"sonuc": "İşe yerleşti", "profil": "Yaş", "gosterge": "Meslek kursu"},
        extra=("Eğitim yılı", "Kadın (1/0)", "İl merkezinde yaşıyor (1/0)"), picks={"sonuc": "Evet", "gosterge": "Var"}),
    "konu06": CustomChoices(
        roles={"sonuc": "Aylık bağış (TL)", "aciklayici": "Hane geliri (bin TL)"},
        extra=("Yaş", "Hane büyüklüğü", "Kentte yaşıyor (1/0)")),
    "konu07": CustomChoices(
        roles={"sonuc": "Saatlik ücret (TL)", "aciklayici": "Eğitim yılı", "kontrol": "Deneyim (yıl)",
               "grup": "Cinsiyet"},
        extra=("Evli (1/0)", "Kıdem (yıl)"), picks={"grup": "Kadın"}),
    "konu08": CustomChoices(roles={"sonuc": "Sınav puanı", "aciklayici": "Haftalık çalışma saati", "kume": "Okul"}),
    "konu09": CustomChoices(roles={"sonuc": "Mezuniyet not ortalaması", "esik_degiskeni": "Sınav puanı", "kume": "Okul"},
                            numbers={"esik": "70", "h": ""}),
    "konu10": CustomChoices(roles={"sonuc": "Matematik puanı", "hedef": "Program (1/0)", "kume": "Okul"},
                            extra=("Başlangıç puanı", "Kız (1/0)")),
    "konu11": CustomChoices(
        roles={"sonuc": "Satış fiyatı (bin TL)", "temel1": "Alan (m²)", "temel2": "Bina yaşı", "kategori": "İlçe",
               "kategori2": "Isıtma"},
        extra=("Oda sayısı", "Kat", "Merkeze uzaklık (km)", "Asansör (1/0)", "Otopark (1/0)", "Manzara (1/0)")),
    "konu12": CustomChoices(roles={"sonuc": "Log kazanç", "tedavi": "Kursa katıldı (1/0)", "kume": "Firma"},
                            extra=("Önceki log kazanç", "Yaş", "Eğitim yılı", "Kadın (1/0)")),
}
STEP_EXCEPTIONS = {
    ("konu09", 1): "reproducibility",
    ("konu09", 3): "title",
    ("konu10", 3): "operations",
    ("konu12", 2): "title",
}
"""Alternatif adımın notlardakinden bilerek ayrıldığı yönler. Konu 9 Adım 1: ilk aşama sıçraması (sınıf mevcudu, okul
kümeli SH) tasarım satırlarına eklenir; adım ayar sabitlenince aynı sayı sınıfındadır. Konu 9 Adım 3: notların başlığı
"Hansen'in tahminini yeniden üretmek"tir; AL1999'da Hansen'in bir tahmini yoktur (aynı ölçek dersi, başka başlık).
Konu 10 Adım 3: notlarda yalnız metin olan yeniden örnekleme birimi okul bootstrap'ıyla hesaplanır (yalnız dağılımda aynı).
Konu 12 Adım 2: notların başlığı okul kümelerinden söz eder; Card verisinde birim kişidir."""
WAGE_TOPICS = ("konu01", "konu02")
"""Örnek dosyası kurgusal ücret verisi olan konular (Blok A); bu verinin sütunlarını kullanan testler bunlarla sınırlı."""
XLSX = "ornek.xlsx"
RSCRIPT = shutil.which("Rscript")


def _r_package(name: str) -> bool:
    """R ve paket kurulu mu? Excel okuyan R betikleri ``readxl`` ister; paket yoksa R adımı atlanır."""

    if RSCRIPT is None:
        return False
    result = subprocess.run([RSCRIPT, "-e", f'quit(status = if (requireNamespace("{name}", quietly = TRUE)) 0 else 1)'],
                            capture_output=True, timeout=300)
    return result.returncode == 0


R_EXCEL = _r_package("readxl")
R_EXCEL_REASON = 'Rscript ya da R readxl paketi bulunamadı (install.packages("readxl")).'


def _checks(spec: LabSpec) -> int:
    return sum(len(step.checks) for step in spec.steps)


def _data_path(dataset: str) -> Path | None:
    value = os.environ.get(f"IKT807_HANSEN_{dataset.upper()}_PATH")
    return Path(value) if value and Path(value).is_file() else None


_LOADED: dict[str, pd.DataFrame] = {}


def hansen_frame(dataset: str) -> pd.DataFrame:
    """Gerçek Hansen verisi (ortam değişkeniyle verilen dosya); verilmezse test atlanır. Dosya bir kez okunur."""

    if dataset not in _LOADED:
        path = _data_path(dataset)
        if path is None:
            pytest.skip(f"Gerçek Hansen {DATASET_MEMBERS[dataset]} dosyası verilmedi.")
        loaded = load_from_upload(dataset, path.read_bytes(), path.name)
        assert loaded.matches_hansen
        _LOADED[dataset] = loaded.frame
    return _LOADED[dataset].copy()


@pytest.fixture(scope="module")
def card() -> pd.DataFrame:
    return hansen_frame("card1995")


def _own(topic: str, frame: pd.DataFrame | None = None, choices: CustomChoices | None = None,
         file_name: str = XLSX, data: bytes | None = None) -> tuple[LabSpec, tuple[str, bytes]]:
    custom = VARIANTS[topic].custom
    if data is None:
        data = K.sample_excel(custom.sample() if frame is None else frame)
    table = K.read_upload(file_name, data)
    case, _ = custom_case(custom, table, choices or SAMPLE_CHOICES[topic])
    return custom.build(case), (file_name, data)


def _environment() -> dict[str, str]:
    return dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="cp1254", LANG="C.UTF-8", LC_ALL="C.UTF-8")


def _run(spec: LabSpec, language: str, folder: Path, data: tuple[str, bytes] | None = None,
         local: Path | None = None) -> subprocess.CompletedProcess:
    script = render_script(spec, language)
    if local is not None:
        before, after = {"Python": ("YEREL_DOSYA = None", f'YEREL_DOSYA = "{local.as_posix()}"'),
                         "R": ("yerel_dosya <- NULL", f'yerel_dosya <- "{local.as_posix()}"')}[language]
        assert before in script
        script = script.replace(before, after, 1)
    if data is not None:
        (folder / data[0]).write_bytes(data[1])
    path = folder / script_filename(spec, language)
    path.write_text(script, encoding="utf-8")
    command = [sys.executable] if language == "Python" else [RSCRIPT]
    return subprocess.run(command + [str(path)], cwd=folder, capture_output=True, encoding="utf-8", errors="replace",
                          timeout=900, env=_environment())


def _reproduces(spec: LabSpec, result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert result.stdout.count("  OK   ") == _checks(spec)
    assert "Bütün değerler uygulamadaki sonuçlarla uyuşuyor." in result.stdout


# --- Kayıt ve notlar ------------------------------------------------------------------------------------------

def test_registry_covers_the_approved_topics_and_labels() -> None:
    assert TOPICS == [f"konu{number:02d}" for number in range(1, 13)]
    assert SOURCE_LABELS == {"notlar": "Notlardaki örnek", "alternatif": "Alternatif örnek",
                             "kendi": "Kendi verini yükle"}
    for topic in TOPICS:
        variants = VARIANTS[topic]
        assert variants.story and variants.custom is not None


def test_notes_code_is_unchanged() -> None:
    digest = hashlib.md5()
    for key in sorted(LABS):
        spec = LABS[key]
        assert spec.source == "notlar"
        for language in LANGUAGES:
            digest.update(render_script(spec, language).encode("utf-8"))
            for step in spec.steps:
                digest.update(render_step(spec, step.number, language).encode("utf-8"))
    assert digest.hexdigest() == NOTES_MD5


@pytest.mark.parametrize("topic", TOPICS)
def test_alternative_mirrors_the_steps_of_the_notes(topic: str) -> None:
    notes, alternative = LABS[topic], VARIANTS[topic].alternative()
    assert alternative.source == "alternatif" and alternative.dataset in DATASET_MEMBERS
    assert [step.number for step in alternative.steps] == [step.number for step in notes.steps]
    for mine, theirs in zip(alternative.steps, notes.steps):
        exception = STEP_EXCEPTIONS.get((topic, mine.number))
        assert (mine.note.section, mine.note.step) == (theirs.note.section, theirs.note.step)
        if exception != "title":
            assert mine.title == theirs.title
        if exception != "operations":
            assert bool(mine.operations) == bool(theirs.operations)
        if exception not in ("operations", "reproducibility"):
            assert mine.reproducibility == theirs.reproducibility
        assert mine.explanation
    assert alternative.steps[0].explanation != notes.steps[0].explanation  # araştırma sorusu bu verinin sorusudur
    labels = [(step.number, check.label) for step in alternative.steps for check in step.checks]
    assert len(labels) == len(set(labels))
    assert all(np.isfinite(check.expected) for step in alternative.steps for check in step.checks)


def test_file_names_follow_the_source() -> None:
    spec, _ = _own("konu01")
    assert script_filename(LABS["konu01"], "Python") == "ikt807_konu01_uygulama.py"
    assert script_filename(VARIANTS["konu01"].alternative(), "R") == "ikt807_konu01_alternatif.R"
    assert script_filename(spec, "Python") == "ikt807_konu01_kendi_verim.py"
    assert languages_for(spec) == ("Python", "R")
    assert languages_for(VARIANTS["konu01"].alternative()) == LANGUAGES


@pytest.mark.parametrize("topic", TOPICS)
def test_every_step_renders_and_scripts_compile(topic: str) -> None:
    own, _ = _own(topic)
    for spec in (VARIANTS[topic].alternative(), own):
        for language in languages_for(spec):
            for step in spec.steps:
                assert bool(render_step(spec, step.number, language)) == bool(step.operations)
            assert "Uygulamayla karşılaştırma:" in render_script(spec, language)
        compile(render_script(spec, "Python"), f"{topic}.py", "exec")
    with pytest.raises(ValueError, match="Stata kodu üretilmez"):
        render_script(own, "Stata")


@pytest.mark.parametrize("topic", TOPICS)
def test_alternative_stata_follows_conventions(topic: str) -> None:
    spec = VARIANTS[topic].alternative()
    code = render_script(spec, "Stata")
    assert "version 14" in code and "set type double" in code
    generates = re.findall(r"^generate\b.*$", code, flags=re.MULTILINE)
    assert all(line.startswith("generate double ") for line in generates)  # Konu 10'un alternatifinde türetme yok
    assert code.count("{") == code.count("}")
    assert code.count("kontrol_et `=") == _checks(spec)
    lines = [line.strip() for line in code.splitlines()]
    assert lines.count("preserve") == lines.count("restore")
    for line in code.splitlines():
        if not line.strip().startswith("*"):
            assert line.count('"') % 2 == 0, line
    assert "uygulama: `beklenen'" in code and "Bütün değerler uygulamadaki sonuçlarla uyuşuyor." in code


# --- Alternatif örnek: Card (1995) ----------------------------------------------------------------------------

@pytest.mark.parametrize("topic", TOPICS)
def test_app_reproduces_every_alternative_number(topic: str) -> None:
    spec = VARIANTS[topic].alternative()
    run = run_lab(spec, {spec.dataset: hansen_frame(spec.dataset)})
    failures = [f"{item.check.label}: {item.value}" for items in run.checks.values() for item in items
                if not item.passed]
    assert not failures, failures


def _card_sample(card: pd.DataFrame) -> pd.DataFrame:
    """Bağımsız hesap: Card'ın 3.010 kişilik örneklemi ve türetilmiş değişkenler (uygulamanın kodundan ayrı)."""

    data = card.dropna(subset=["lwage76"]).copy()
    data["lwage"] = np.log(data["wage76"].astype(float) / 100)
    data["exper"] = data["age76"].astype(float) - data["ed76"] - 6
    data["exper2"] = data["exper"] ** 2 / 100
    return data


def _ols(data: pd.DataFrame, y: str, xs: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    design = np.column_stack([np.ones(len(data)), data[xs].to_numpy(dtype=float)])
    target = data[y].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    n, k = design.shape
    covariance = residual @ residual / (n - k) * np.linalg.inv(design.T @ design)
    r2 = 1 - residual @ residual / ((target - target.mean()) @ (target - target.mean()))
    return beta, np.sqrt(np.diag(covariance)), float(r2)


def test_konu01_alternative_numbers_follow_card1995(card: pd.DataFrame) -> None:
    data = _card_sample(card)
    assert len(card) == 3613 and card["wage76"].notna().sum() == 3017 and len(data) == 3010
    assert ((card["wage76"] < 100) & card["lwage76"].isna()).sum() == 7
    assert np.abs(data["lwage76"] - np.log(data["wage76"])).max() < 1e-5
    values = dict(ornek_konu01.ALT_EXPECTED)
    assert round(data["wage76"].mean() / 100, 4) == values[(3, "Saatlik ücret ortalaması")] == 5.7728
    assert data["wage76"].median() / 100 == values[(3, "Saatlik ücret medyanı")] == 5.375
    assert data["lwage"].min() == 0.0 and round(data["black"].mean(), 4) == 0.2336
    assert round(data["exper"].mean(), 1) == 8.9 and data["exper"].min() == 0
    means = data.groupby("black")["lwage"].mean()
    assert (round(means[0.0], 4), round(means[1.0], 4)) == (1.7309, 1.4129)
    assert (data["ed76"] == 12).sum() == 992 == data["ed76"].value_counts().max()
    b1, _, r1 = _ols(data, "lwage", ["ed76"])
    b2, _, r2 = _ols(data, "lwage", ["ed76", "exper", "exper2"])
    b3, s3, r3 = _ols(data, "lwage", ["ed76", "exper", "exper2", "black"])
    assert (round(b1[0], 4), round(b1[1], 4), round(r1, 4)) == (0.9657, 0.0521, 0.0987)
    assert (round(b2[1], 4), round(b3[1], 4), round(b3[4], 4)) == (0.0932, 0.0818, -0.2319)
    assert (round(r2, 4), round(r3, 4)) == (0.1958, 0.2411)
    assert round(100 * (np.exp(b3[1]) - 1), 2) == 8.52 and round(100 * (np.exp(b3[4]) - 1), 2) == -20.70
    assert [round(value, 4) for value in s3] == [0.0687, 0.0036, 0.0069, 0.0328, 0.0173]
    assert round(data["ed76"].corr(data["exper"]), 2) == -0.65
    texts = " ".join(step.explanation + " " + step.takeaway for step in ornek_konu01.alternative().steps)
    for number in ("3.613", "3.017", "3.010", "5,77", "5,375", "8,9 yıl", "0,2336", "1,7309", "1,4129", "992",
                   "0{,}9657+0{,}0521", "0,0521", "0,0932", "0,0818", "%8,52", "−0,2319", "%20,70", "−0,65"):
        assert number in texts, number


def test_konu02_alternative_numbers_follow_card1995(card: pd.DataFrame) -> None:
    data = _card_sample(card)
    data["ed_black"] = data["ed76"] * data["black"]
    controls = ["exper", "exper2", "black", "reg76r", "smsa76r"]
    m2 = smf.ols("lwage ~ ed76 + " + " + ".join(controls), data).fit()
    m3 = smf.ols("lwage ~ ed76 + ed_black + " + " + ".join(controls), data).fit()
    assert (round(m2.params["ed76"], 4), round(m2.bse["ed76"], 4), round(m2.HC1_se["ed76"], 4)) == \
        (0.0740, 0.0035, 0.0036)
    assert (round(m3.params["ed76"], 4), round(m3.params["ed_black"], 4)) == (0.0701, 0.0182)
    assert round(m3.params["black"], 4) == -0.4146 and data["ed76"].min() > 0
    assert round(100 * (np.exp(m2.params["ed76"]) - 1), 2) == 7.68
    residual2 = m2.resid.to_numpy() ** 2
    design = m2.model.exog
    fitted = design @ np.linalg.lstsq(design, residual2, rcond=None)[0]
    lm = len(residual2) * (1 - ((residual2 - fitted) ** 2).sum() / ((residual2 - residual2.mean()) ** 2).sum())
    assert round(lm, 2) == 6.51
    from scipy import stats

    assert round(stats.chi2.sf(lm, design.shape[1] - 1), 2) == 0.37
    robust = smf.ols("lwage ~ ed76 + ed_black + " + " + ".join(controls), data).fit(cov_type="HC1")
    weights = np.array([1.0, 1.0])
    covariance = robust.cov_params().loc[["ed76", "ed_black"], ["ed76", "ed_black"]].to_numpy()
    assert round(m3.params["ed76"] + m3.params["ed_black"], 4) == 0.0882
    assert round(float(np.sqrt(weights @ covariance @ weights)), 4) == 0.0058
    robust2 = smf.ols("lwage ~ ed76 + " + " + ".join(controls), data).fit(cov_type="HC1")
    assert round(100 * np.exp(robust2.params["ed76"]) * robust2.bse["ed76"], 2) == 0.39
    texts = " ".join(step.explanation + " " + step.takeaway for step in ornek_konu02.alternative().steps)
    for number in ("0,0740", "7{,}68", "7,7", "0,0035 ve 0,0036", "6,51", "0,37", "0,0701", "0,0182",
                   "0{,}0701+0{,}0182\\approx0{,}088", "−0,4146", "7,68", "0,39", "0,074"):
        assert number in texts, number


@pytest.mark.parametrize("topic", TOPICS)
def test_generated_python_reproduces_the_alternative(topic: str, tmp_path: Path) -> None:
    spec = VARIANTS[topic].alternative()
    hansen_frame(spec.dataset)  # dosya yoksa atlanır
    _reproduces(spec, _run(spec, "Python", tmp_path, local=_data_path(spec.dataset)))


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript bulunamadı.")
@pytest.mark.parametrize("topic", TOPICS)
def test_generated_r_reproduces_the_alternative(topic: str, tmp_path: Path) -> None:
    spec = VARIANTS[topic].alternative()
    hansen_frame(spec.dataset)
    _reproduces(spec, _run(spec, "R", tmp_path, local=_data_path(spec.dataset)))


# --- Kendi verin --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("topic", TOPICS)
def test_the_sample_file_runs_every_step_and_passes_its_checks(topic: str) -> None:
    spec, _ = _own(topic)
    assert spec.source == "kendi" and _checks(spec) >= 10
    run = run_lab(spec)
    assert all(item.passed for items in run.checks.values() for item in items)
    for step in spec.steps:
        if step.note_for is not None:
            text = step.note_for(run.state)
            assert text and "nan" not in text.lower(), (step.number, text)


@pytest.mark.parametrize("topic", TOPICS)
def test_generated_python_reads_the_uploaded_excel(topic: str, tmp_path: Path) -> None:
    spec, data = _own(topic)
    _reproduces(spec, _run(spec, "Python", tmp_path, data))


@pytest.mark.skipif(not R_EXCEL, reason=R_EXCEL_REASON)
@pytest.mark.parametrize("topic", TOPICS)
def test_generated_r_reads_the_uploaded_excel(topic: str, tmp_path: Path) -> None:
    spec, data = _own(topic)
    _reproduces(spec, _run(spec, "R", tmp_path, data))


def _turkish_csv(frame: pd.DataFrame) -> bytes:
    """Türkçe Excel'in "CSV (noktalı virgülle ayrılmış)" kaydı: ayırıcı ;, ondalık virgül, Windows-1254."""

    return frame.to_csv(sep=";", decimal=",", index=False).encode("cp1254")


@pytest.mark.parametrize("topic", TOPICS)
def test_a_turkish_csv_gives_the_same_numbers_in_both_languages(topic: str, tmp_path: Path) -> None:
    frame = VARIANTS[topic].custom.sample()
    spec, data = _own(topic, file_name="veri.csv", data=_turkish_csv(frame))
    read = next(op for step in spec.steps for op in step.operations if isinstance(op, ReadFile))
    assert (read.separator, read.decimal, read.encoding) == (";", ",", "cp1254")
    excel, _ = _own(topic)
    assert [check.expected for step in spec.steps for check in step.checks] == \
        [check.expected for step in excel.steps for check in step.checks]
    _reproduces(spec, _run(spec, "Python", tmp_path, data))
    if RSCRIPT is not None:
        _reproduces(spec, _run(spec, "R", tmp_path, data))


def test_role_rules_are_enforced() -> None:
    custom = VARIANTS["konu01"].custom
    frame = custom.sample()
    table = K.read_upload(XLSX, K.sample_excel(frame))
    roles = dict(SAMPLE_CHOICES["konu01"].roles)
    with pytest.raises(K.UploadError, match="için bir sütun seçin"):
        custom_case(custom, table, CustomChoices(roles={**roles, "grup": None}))
    with pytest.raises(K.UploadError, match="üç farklı sütun"):
        custom_case(custom, table, CustomChoices(roles={**roles, "kontrol": "Eğitim yılı"}))
    with pytest.raises(K.UploadError, match="en çok 30 farklı değer"):
        custom_case(custom, table, CustomChoices(roles={**roles, "aciklayici": "Saatlik ücret (TL)",
                                                        "sonuc": "Kıdem (yıl)"}, options={"log": False}))
    with pytest.raises(K.UploadError, match="kategori var"):
        three = frame.assign(Cinsiyet=np.where(np.arange(len(frame)) % 3 == 0, "Diğer", frame["Cinsiyet"]))
        custom_case(custom, K.read_upload(XLSX, K.sample_excel(three)), SAMPLE_CHOICES["konu01"])
    negative = frame.assign(**{"Saatlik ücret (TL)": frame["Saatlik ücret (TL)"] - 300})
    with pytest.raises(K.UploadError, match="logaritma alınamaz"):
        custom_case(custom, K.read_upload(XLSX, K.sample_excel(negative)), SAMPLE_CHOICES["konu01"])
    collinear = frame.assign(**{"Deneyim (yıl)": 2 * frame["Eğitim yılı"] + 1.0})
    with pytest.raises(K.UploadError, match="tam doğrusal bağlantı"):
        custom_case(custom, K.read_upload(XLSX, K.sample_excel(collinear)), SAMPLE_CHOICES["konu01"])
    custom2 = VARIANTS["konu02"].custom
    table2 = K.read_upload(XLSX, K.sample_excel(custom2.sample()))
    with pytest.raises(K.UploadError, match="hem bir rol için hem ek sütun"):
        custom_case(custom2, table2, CustomChoices(roles=SAMPLE_CHOICES["konu02"].roles, extra=("Eğitim yılı",)))
    with pytest.raises(K.UploadError, match="tam doğrusal bağlantı"):
        both = custom2.sample().assign(Toplam=lambda d: d["Deneyim (yıl)"] + d["Kıdem (yıl)"])
        custom_case(custom2, K.read_upload(XLSX, K.sample_excel(both)),
                    CustomChoices(roles=SAMPLE_CHOICES["konu02"].roles,
                                  extra=("Deneyim (yıl)", "Kıdem (yıl)", "Toplam"), picks={"grup": "Kadın"}))


@pytest.mark.parametrize("topic", WAGE_TOPICS)
def test_an_outcome_without_logarithm_skips_the_percent_steps(topic: str, tmp_path: Path) -> None:
    frame = VARIANTS[topic].custom.sample()
    frame["Saatlik ücret (TL)"] = frame["Saatlik ücret (TL)"] - 200  # negatif değerler: logaritma alınamaz
    base = SAMPLE_CHOICES[topic]
    choices = CustomChoices(roles=base.roles, extra=base.extra, picks=base.picks, options={"log": False})
    spec, data = _own(topic, frame, choices)
    operations = [op for step in spec.steps for op in step.operations]
    assert not any(isinstance(op, (Scalar, DeltaMethod)) for op in operations)
    assert "log_sonuc" not in render_script(spec, "Python")
    _reproduces(spec, _run(spec, "Python", tmp_path, data))
    if R_EXCEL:
        _reproduces(spec, _run(spec, "R", tmp_path, data))


SPECIAL = pd.DataFrame({
    "Ücret $": [12.5, 14.0, 9.75, 20.0, 16.5, 11.0, 18.25, 13.5, 22.0, 10.5, 15.75, 19.0, 17.5, 12.0],
    "Eğitim_1": [8, 11, 8, 15, 12, 8, 15, 11, 15, 8, 12, 15, 12, 11],
    "Deneyim*": [3, 10, 1, 7, 12, 6, 2, 9, 15, 4, 8, 11, 5, 14],
    "Grup|A": ["a|1", "b*2", "a|1", "b*2", "a|1", "b*2", "a|1", "b*2", "a|1", "b*2", "a|1", "b*2", "a|1", "b*2"],
    "Ek [1]": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
})
SPECIAL_CHOICES = {
    "konu01": CustomChoices(roles={"sonuc": "Ücret $", "aciklayici": "Eğitim_1", "kontrol": "Deneyim*",
                                   "grup": "Grup|A"}, picks={"grup": "b*2"}),
    "konu02": CustomChoices(roles={"sonuc": "Ücret $", "aciklayici": "Eğitim_1", "grup": "Grup|A"},
                            extra=("Deneyim*", "Ek [1]"), picks={"grup": "b*2"}),
}


@pytest.mark.parametrize("topic", WAGE_TOPICS)
def test_user_names_are_escaped_in_texts_and_quoted_in_code(topic: str, tmp_path: Path) -> None:
    spec, data = _own(topic, SPECIAL, SPECIAL_CHOICES[topic])
    run = run_lab(spec)
    texts = " ".join(step.explanation + " " + (step.note_for(run.state) if step.note_for else step.takeaway)
                     for step in spec.steps)
    assert "Ücret \\$" in texts and "Eğitim\\_1" in texts
    assert "Ücret $" not in texts.replace("Ücret \\$", "")
    _reproduces(spec, _run(spec, "Python", tmp_path, data))
    if R_EXCEL:
        _reproduces(spec, _run(spec, "R", tmp_path, data))


def test_columns_named_like_the_apps_own_columns_keep_their_names(tmp_path: Path) -> None:
    frame = VARIANTS["konu01"].custom.sample().rename(columns={"Deneyim (yıl)": "log_sonuc", "Cinsiyet": "grup01"})
    choices = CustomChoices(roles={"sonuc": "Saatlik ücret (TL)", "aciklayici": "Eğitim yılı", "kontrol": "log_sonuc",
                                   "grup": "grup01"}, picks={"grup": "Kadın"})
    spec, data = _own("konu01", frame, choices)
    read = spec.steps[1].operations[0]
    assert [name for name, _, _ in read.columns] == ["saatlik_ucret_tl", "egitim_yili", "log_sonuc_2", "grup01_2"]
    assert K.code_name("Log sonuç") == "log_sonuc_2" and {"log_sonuc", "grup01", "etkilesim"} <= K.RESERVED_CODES
    _reproduces(spec, _run(spec, "Python", tmp_path, data))


@pytest.mark.parametrize("topic", WAGE_TOPICS)
def test_an_exact_fit_drops_the_standard_error_checks_and_says_why(topic: str, tmp_path: Path) -> None:
    frame = VARIANTS[topic].custom.sample()
    women = (frame["Cinsiyet"] == "Kadın").astype(float)
    log_wage = 2.0 + 0.1 * frame["Eğitim yılı"] - 0.2 * women
    if topic == "konu01":
        log_wage = log_wage + 0.03 * frame["Deneyim (yıl)"] - 0.0005 * frame["Deneyim (yıl)"] ** 2
    else:
        log_wage = log_wage + 0.02 * frame["Deneyim (yıl)"] + 0.05 * frame["Evli (1/0)"] + 0.01 * frame["Kıdem (yıl)"]
    frame["Saatlik ücret (TL)"] = np.exp(log_wage)
    spec, data = _own(topic, frame)
    fragile = [check.label for step in spec.steps for check in step.checks
               if (isinstance(check.target, CoefTarget) and check.target.quantity in ("se", "se_hc1"))
               or (isinstance(check.target, ScalarTarget) and check.target.name == "bp_lm")]
    assert not fragile
    run = run_lab(spec)
    notes = " ".join(step.note_for(run.state) for step in spec.steps if step.note_for is not None)
    assert "R² ≈ 1" in notes
    _reproduces(spec, _run(spec, "Python", tmp_path, data))
    if R_EXCEL:
        _reproduces(spec, _run(spec, "R", tmp_path, data))


def test_markdown_escaping_of_user_text() -> None:
    assert md("Fiyat ($)") == "Fiyat (\\$)"
    assert md("a|b *c* _d_") == "a\\|b \\*c\\* \\_d\\_"
    assert md("1. sınıf") == "1\\. sınıf"


def test_numbers_stored_as_text_in_excel_are_read_the_same_way_in_both_languages(tmp_path: Path) -> None:
    """Excel'de metin olarak saklanmış sayı hücresi: pandas sütunu sayıya çevirir, readxl bütün sütunu metin okur; R
    kodu sütunu açıkça sayıya çevirmelidir."""

    frame = VARIANTS["konu01"].custom.sample()
    frame["Grup kodu"] = np.where(frame["Cinsiyet"] == "Kadın", 1, 2)
    frame = frame.astype({"Eğitim yılı": object, "Grup kodu": object})
    frame.loc[0, "Eğitim yılı"] = str(frame.loc[0, "Eğitim yılı"])
    frame.loc[1, "Grup kodu"] = str(frame.loc[1, "Grup kodu"])
    choices = CustomChoices(roles={"sonuc": "Saatlik ücret (TL)", "aciklayici": "Eğitim yılı",
                                   "kontrol": "Deneyim (yıl)", "grup": "Grup kodu"}, picks={"grup": "1"})
    spec, data = _own("konu01", frame, choices)
    kinds = {name: kind for name, _, kind in spec.steps[1].operations[0].columns}
    assert kinds["egitim_yili"] == "sayi" and kinds["grup_kodu"] == "kod"
    _reproduces(spec, _run(spec, "Python", tmp_path, data))
    if R_EXCEL:
        _reproduces(spec, _run(spec, "R", tmp_path, data))


@pytest.mark.parametrize("topic", WAGE_TOPICS)
def test_badly_scaled_variables_are_rejected_with_a_units_message(topic: str) -> None:
    """Milyonlarla ölçülen bir kontrolün karesi tasarım matrisini kötü koşullu yapar: statsmodels'in pinv çözümü
    katsayıları sessizce bozabilir. Uygulama hesaplamaz, birimi değiştirmeyi önerir."""

    frame = VARIANTS[topic].custom.sample()
    frame["Varlık (TL)"] = frame["Deneyim (yıl)"] * 10_000_000 + np.arange(len(frame))
    base = SAMPLE_CHOICES[topic]
    if topic == "konu01":
        choices = CustomChoices(roles={**base.roles, "kontrol": "Varlık (TL)"}, picks=base.picks)
    else:
        frame["Varlık (TL)"] = frame["Varlık (TL)"] * 100_000
        choices = CustomChoices(roles=base.roles, extra=("Varlık (TL)",), picks=base.picks)
    with pytest.raises(K.UploadError, match="birimini değiştirin"):
        _own(topic, frame, choices)


def test_logarithm_option_reads_decimal_comma_text_like_the_app() -> None:
    assert ornek_konu01._positive(pd.Series(["394,47", " 12,5 ", "NA", None]))
    assert not ornek_konu01._positive(pd.Series(["394,47", "-1,5"]))
    assert ornek_konu01._positive(pd.Series([1.5, 2.0])) and not ornek_konu01._positive(pd.Series([0.0, 2.0]))


@pytest.mark.parametrize("label, low, high", [("Yıl", 2015, 2024), ("Doğum yılı", 1985, 2001)])
def test_a_year_and_its_square_are_rejected_but_a_linear_year_is_kept(label: str, low: int, high: int,
                                                                       tmp_path: Path) -> None:
    """Dar aralıkta yıl ve karesi neredeyse doğrusal bağlantılıdır: QR (R) ve sözde ters (Python) dördüncü ondalıkta
    ayrışır. Konu 1 reddeder ve yılı ortalamayı önerir; Konu 2'de yıl doğrusal girer ve iki dil aynı sayıyı verir."""

    frame = VARIANTS["konu01"].custom.sample()
    frame[label] = np.random.default_rng(5).integers(low, high, len(frame)).astype(float)
    base = SAMPLE_CHOICES["konu01"]
    with pytest.raises(K.UploadError, match="yıl − 2015"):
        _own("konu01", frame, CustomChoices(roles={**base.roles, "kontrol": label}, picks=base.picks))
    choices = CustomChoices(roles=SAMPLE_CHOICES["konu02"].roles, extra=(label, "Evli (1/0)"), picks={"grup": "Kadın"})
    spec, data = _own("konu02", frame, choices)
    _reproduces(spec, _run(spec, "Python", tmp_path, data))
    if R_EXCEL:
        _reproduces(spec, _run(spec, "R", tmp_path, data))


def test_the_teaching_csv_is_offered_only_where_it_carries_the_lab_variables() -> None:
    """Alternatif örnek öğretim CSV'si önermez; notlarda yalnız öğretim CSV'si laboratuvarın bütün ham değişkenlerini
    taşıyan .dta veri setlerinde (DDK2011, LM2007) önerilir: Card1995 ve CHJ2004'te önerilmez."""

    for topic in TOPICS:
        for language in LANGUAGES:
            assert "öğretim CSV" not in render_script(VARIANTS[topic].alternative(), language)
    offered = {key for key, spec in LABS.items()
               if all("öğretim CSV" in render_script(spec, language) for language in LANGUAGES)}
    silent = {key for key, spec in LABS.items()
              if not any("öğretim CSV" in render_script(spec, language) for language in LANGUAGES)}
    assert offered | silent == set(LABS)
    assert offered == {key for key, spec in LABS.items()
                       if spec.dataset in TEACHING_CSV and DATASET_MEMBERS[spec.dataset].lower().endswith(".dta")}
    assert {"konu04", "konu06"} <= silent and {"konu03", "konu09"} <= offered


def test_a_symmetric_outcome_is_not_called_skewed() -> None:
    frame = VARIANTS["konu01"].custom.sample()
    frame["Saatlik ücret (TL)"] = np.random.default_rng(7).normal(200, 20, len(frame)).round(2)
    spec, _ = _own("konu01", frame)
    run = run_lab(spec)
    text = spec.steps[2].note_for(run.state)
    assert "sağa çarpıktır" not in text and "belirgin bir sağa çarpıklık yoktur" in text
    sample, _ = _own("konu01")
    assert "sağa çarpıktır" in sample.steps[2].note_for(run_lab(sample).state)


def test_python_keeps_numeric_columns_as_floats_like_the_app() -> None:
    spec, _ = _own("konu01")
    code = render_script(spec, "Python")
    assert 'veri[sayisal] = veri[sayisal].astype(float)' in code
