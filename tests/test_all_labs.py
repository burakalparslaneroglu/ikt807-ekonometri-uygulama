"""Kayıtlı bütün uygulama laboratuvarları için ortak sözleşme ve üç dil testleri.

Yeni bir konu ``core/labs/registry.py``'ye eklendiğinde bu testler onu kendiliğinden kapsar.
Gerçek veri gerektiren testler, veri setinin yolu verilmezse atlanır:
IKT807_HANSEN_CPS09MAR_PATH, IKT807_HANSEN_DDK2011_PATH, IKT807_HANSEN_CARD1995_PATH,
IKT807_HANSEN_CHJ2004_PATH.
"""

from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.codegen.base import LANGUAGES, generator, render_script, render_step
from core.hansen_data import DATASET_MEMBERS, load_from_upload
from core.labs.registry import LABS
from core.labs.runner import run_lab

SPECS = list(LABS.values())


def _experiments() -> list:
    found = []
    for number in range(1, 13):
        try:
            module = importlib.import_module(f"core.labs.sezgi_konu{number:02d}")
        except ModuleNotFoundError:
            continue
        found.extend(getattr(module, f"KONU{number:02d}_EXPERIMENTS"))
    return found


EXPERIMENTS = _experiments()
UTF8_OUTPUT = 'sys.stdout.reconfigure(encoding="utf-8")'
STATA_SPECS = [spec for spec in SPECS if DATASET_MEMBERS[spec.dataset].lower().endswith(".dta")]
STATA_INTEGER_TYPES = ((np.int8, -127, 100), (np.int16, -32767, 32740), (np.int32, -2147483647, 2147483620))
"""Stata byte, int ve long türlerinin veri aralıkları (``compress`` bu sırayla en küçüğünü seçer)."""


def _data_path(dataset: str) -> Path | None:
    names = [f"IKT807_HANSEN_{dataset.upper()}_PATH"] + (["IKT807_CPS_PATH"] if dataset == "cps09mar" else [])
    for name in names:
        value = os.environ.get(name)
        if value and Path(value).is_file():
            return Path(value)
    return None


def _data(spec) -> Path:
    path = _data_path(spec.dataset)
    if path is None:
        pytest.skip(f"Gerçek Hansen {spec.dataset} dosyası verilmedi.")
    return path


def _checks(spec) -> int:
    return sum(len(step.checks) for step in spec.steps)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_steps_are_numbered_and_tied_to_the_notes(spec) -> None:
    assert [step.number for step in spec.steps] == list(range(1, len(spec.steps) + 1))
    for step in spec.steps:
        lab_step = step.note.section == spec.note_section and step.note.step == step.number
        # Laboratuvarın son alt bölümü (ör. §3.15.5 okuma soruları) numarasız, işlemsiz bir adımdır.
        reading = (
            step is spec.steps[-1]
            and step.note.section.startswith(spec.note_section + ".")
            and step.note.step == 0
            and not step.operations
        )
        assert lab_step or reading, step.number
        assert step.title and step.explanation
    # Notlarda basılı her sayı bir Check'tir; en kısa laboratuvar (Konu 10) altı sayı basar.
    assert _checks(spec) >= 6


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_every_step_renders_and_python_compiles(spec) -> None:
    for language in LANGUAGES:
        for step in spec.steps:
            assert bool(render_step(spec, step.number, language)) == bool(step.operations)
    compile(render_script(spec, "Python"), f"{spec.topic_key}.py", "exec")


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_stata_follows_conventions(spec) -> None:
    code = render_script(spec, "Stata")
    assert "version 14" in code and "set type double" in code
    generates = [l for l in re.findall(r"^generate\b.*$", code, flags=re.MULTILINE) if l != "generate long id = _n"]
    assert all(line.startswith("generate double ") for line in generates)
    assert code.count("{") == code.count("}")
    assert code.count("kontrol_et `=") == _checks(spec)
    lines = [line.strip() for line in code.splitlines()]
    assert lines.count("preserve") == lines.count("restore")
    for line in code.splitlines():
        if not line.strip().startswith("*"):
            assert line.count('"') % 2 == 0, line


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_app_reproduces_every_number_in_the_notes(spec) -> None:
    path = _data(spec)
    loaded = load_from_upload(spec.dataset, path.read_bytes(), path.name)
    assert loaded.matches_hansen
    run = run_lab(spec, {spec.dataset: loaded.frame})
    failures = [f"{c.check.label}: {c.value}" for items in run.checks.values() for c in items if not c.passed]
    assert not failures, failures


@pytest.mark.parametrize(
    "spec", SPECS + [experiment.spec(experiment.defaults()) for experiment in EXPERIMENTS],
    ids=lambda s: f"{s.topic_key}-{s.kind}-{s.title[:20]}",
)
def test_python_scripts_write_utf8_before_any_output(spec) -> None:
    script = render_script(spec, "Python")
    assert "import sys" in script.splitlines()
    assert UTF8_OUTPUT in script
    first_print = script.find("print(")
    assert first_print == -1 or script.index(UTF8_OUTPUT) < first_print


def test_utf8_output_setup_survives_a_turkish_windows_code_page(tmp_path: Path) -> None:
    """Türkçe Windows'ta yönlendirilen çıktı cp1254'tür; τ, β̂, → bu kod sayfasında yoktur."""

    text = "τ = 0,90; β̂; θ; ε; → ; Şış Ğğ İı"
    setup = "\n".join(generator(SPECS[0], "Python").output_setup())
    environment = dict(os.environ, PYTHONIOENCODING="cp1254")
    for body, works in ((f"import sys\n{setup}\nprint({text!r})\n", True), (f"print({text!r})\n", False)):
        path = tmp_path / "cikti.py"
        path.write_text(body, encoding="utf-8")
        result = subprocess.run([sys.executable, str(path)], capture_output=True, encoding="utf-8",
                                errors="replace", timeout=60, env=environment)
        assert (result.returncode == 0) is works, result.stderr[-500:]
        if works:
            assert result.stdout.strip() == text


def _compressed_stata_copy(source: Path, folder: Path) -> Path:
    """Stata ``compress`` gibi: tam sayı değerli, eksiksiz sütunları en küçük tamsayı türüyle yazar.

    Hansen'in arşivindeki .dta dosyaları bu biçimde olabilir; pandas bu sütunları int8/int16/int32
    okur ve üretilen kod küçük tamsayı türlerinde taşmaya karşı korunmalıdır.
    """

    frame = pd.read_stata(source, convert_categoricals=False)
    for column in frame.columns:
        values = frame[column]
        if values.dtype.kind != "f" or values.isna().any() or not np.all(np.mod(values, 1) == 0):
            continue
        for dtype, low, high in STATA_INTEGER_TYPES:
            if values.min() >= low and values.max() <= high:
                frame[column] = values.astype(dtype)
                break
    target = folder / f"sikistirilmis_{source.name}"
    frame.to_stata(target, write_index=False)
    return target


def _prepared(spec, language: str, folder: Path, data: Path | None = None) -> Path:
    script = render_script(spec, language)
    local = str(data or _data(spec)).replace("\\", "/")
    before, after = {"Python": ("YEREL_DOSYA = None", f'YEREL_DOSYA = "{local}"'),
                     "R": ("yerel_dosya <- NULL", f'yerel_dosya <- "{local}"')}[language]
    path = folder / f"{spec.topic_key}.{ {'Python': 'py', 'R': 'R'}[language] }"
    path.write_text(script.replace(before, after, 1), encoding="utf-8")
    return path


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_generated_python_reproduces_the_notes(spec, tmp_path: Path) -> None:
    path = _prepared(spec, "Python", tmp_path)
    # PYTHONIOENCODING=cp1254: Türkçe Windows'ta dosyaya/boruya yönlendirilen çıktının kod sayfası. Betik
    # çıktıyı UTF-8'e çevirdiği için τ, β̂ gibi karakterler bu ortamda da yazılabilmelidir.
    result = subprocess.run([sys.executable, str(path)], cwd=tmp_path, capture_output=True, encoding="utf-8",
                            errors="replace", timeout=900,
                            env=dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="cp1254"))
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert result.stdout.count("  OK   ") == _checks(spec)


@pytest.mark.parametrize("spec", STATA_SPECS, ids=lambda s: s.topic_key)
def test_app_and_python_are_robust_to_compressed_stata_storage(spec, tmp_path: Path) -> None:
    """Tamsayıları byte/int/long olarak saklanmış .dta dosyasıyla da uygulama ve üretilen Python kodu
    notlardaki sayıları vermeli.

    Aksi hâlde örneğin int8 deneyim değişkeninin karesi uyarı vermeden taşar (12**2 = -112) ve bütün
    katsayılar değişir; Card1995'te OLS eğitim katsayısı 0,0747 yerine 0,0753 çıkar.
    """

    source = _data(spec)
    if source.suffix.lower() != ".dta":
        pytest.skip("Yerel veri .dta değil.")
    compressed = _compressed_stata_copy(source, tmp_path)
    assert any(dtype.itemsize < 8 for dtype in pd.read_stata(compressed).dtypes if dtype.kind == "i")
    loaded = load_from_upload(spec.dataset, compressed.read_bytes(), compressed.name)
    run = run_lab(spec, {spec.dataset: loaded.frame})
    failures = [f"{c.check.label}: {c.value}" for items in run.checks.values() for c in items if not c.passed]
    assert not failures, failures
    path = _prepared(spec, "Python", tmp_path, data=compressed)
    result = subprocess.run([sys.executable, str(path)], cwd=tmp_path, capture_output=True, encoding="utf-8",
                            errors="replace", timeout=900,
                            env=dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="cp1254"))
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert result.stdout.count("  OK   ") == _checks(spec)


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript bulunamadı.")
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_generated_r_reproduces_the_notes(spec, tmp_path: Path) -> None:
    path = _prepared(spec, "R", tmp_path)
    result = subprocess.run(["Rscript", str(path)], cwd=tmp_path, capture_output=True, encoding="utf-8",
                            errors="replace", timeout=900, env=dict(os.environ, LANG="C.UTF-8", LC_ALL="C.UTF-8"))
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert result.stdout.count("  OK   ") == _checks(spec)
