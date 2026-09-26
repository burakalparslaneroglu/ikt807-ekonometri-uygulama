"""Konu 1 laboratuvarı: tanım sözleşmesi, notlarla altın değerler ve üç dilde kod.

Altın değer testleri gerçek Hansen verisi gerektirir. Veri depoda tutulmadığı için
dosya yolu ortam değişkeniyle verilir; yoksa bu testler atlanır:

    IKT807_HANSEN_CPS09MAR_PATH=/yol/cps09mar.txt  (veya eski ad: IKT807_CPS_PATH)

Dosya Hansen'in arşivindeki ``cps09mar.txt`` olmalıdır (başlıksız, boşlukla ayrılmış);
üretilen öğrenci kodları bu biçimi okur.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.codegen.base import LANGUAGES, render_script, render_step
from core.hansen_data import CPS09MAR_COLUMNS, CPS09MAR_ROWS, load_from_upload
from core.labs.konu01 import KONU01_LAB
from core.labs.registry import get_lab
from core.labs.runner import run_lab
from core.labs.spec import Describe, GroupSummary, LoadHansen, OLS, RegressionTable


def _data_path() -> Path | None:
    for name in ("IKT807_HANSEN_CPS09MAR_PATH", "IKT807_CPS_PATH"):
        value = os.environ.get(name)
        if value and Path(value).is_file():
            return Path(value)
    return None


DATA = _data_path()
needs_data = pytest.mark.skipif(DATA is None, reason="Gerçek Hansen cps09mar dosyası verilmedi.")
CHECK_COUNT = sum(len(step.checks) for step in KONU01_LAB.steps)


def _synthetic(rows: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    education = rng.choice([8, 12, 13, 14, 16, 18, 20], rows)
    female = rng.integers(0, 2, rows)
    age = rng.integers(18, 70, rows)
    log_wage = 1.0 + 0.1 * education - 0.25 * female + rng.normal(0, 0.5, rows)
    hours, week = rng.integers(36, 60, rows), rng.integers(48, 53, rows)
    return pd.DataFrame(
        {
            "age": age, "female": female, "hisp": 0, "education": education,
            "earnings": np.exp(log_wage) * hours * week, "hours": hours, "week": week,
            "union": 0, "uncov": 0, "region": 1, "race": 1, "marital": 1,
        }
    )[list(CPS09MAR_COLUMNS)]


# --- Sözleşme -----------------------------------------------------------------

def test_lab_mirrors_the_nine_note_steps() -> None:
    assert get_lab("konu01") is KONU01_LAB
    assert [step.number for step in KONU01_LAB.steps] == list(range(1, 10))
    for step in KONU01_LAB.steps:
        assert step.note.section == "1.13"
        assert step.note.step == step.number
        assert step.title and step.explanation


def test_loader_uses_hansens_documented_column_order() -> None:
    loaders = [op for step in KONU01_LAB.steps for op in step.operations if isinstance(op, LoadHansen)]
    assert len(loaders) == 1
    assert loaders[0].columns == CPS09MAR_COLUMNS
    assert CPS09MAR_COLUMNS[:7] == ("age", "female", "hisp", "education", "earnings", "hours", "week")


def test_every_value_shown_to_students_has_a_turkish_label() -> None:
    for step in KONU01_LAB.steps:
        for op in step.operations:
            names: tuple[str, ...] = ()
            if isinstance(op, Describe):
                names = op.variables
            elif isinstance(op, GroupSummary):
                names = (op.by, *(name for name, _, _ in op.columns))
            elif isinstance(op, OLS):
                names = (op.outcome, *op.regressors)
            elif isinstance(op, RegressionTable):
                names = op.terms
            for name in names:
                assert KONU01_LAB.label(name) != name or name == "N", name


def test_all_steps_render_in_all_languages() -> None:
    for language in LANGUAGES:
        for step in KONU01_LAB.steps:
            snippet = render_step(KONU01_LAB, step.number, language)
            assert bool(snippet) == bool(step.operations)


def test_lab_runs_on_hansen_shaped_synthetic_data() -> None:
    run = run_lab(KONU01_LAB, {"cps09mar": _synthetic()})
    assert len(run.state.models) == 3
    assert sum(len(items) for items in run.checks.values()) == CHECK_COUNT
    assert not run.all_passed  # sentetik veri notlardaki sayıları üretmemeli


# --- Notlarla altın değerler ---------------------------------------------------

@needs_data
def test_app_reproduces_every_number_printed_in_notes() -> None:
    raw = load_from_upload("cps09mar", DATA.read_bytes(), DATA.name).frame
    assert len(raw) == CPS09MAR_ROWS
    run = run_lab(KONU01_LAB, {"cps09mar": raw})
    failures = [
        f"Adım {number}: {item.check.label} = {item.value} (notlar {item.check.expected})"
        for number, items in run.checks.items()
        for item in items
        if not item.passed
    ]
    assert not failures, "\n".join(failures)


# --- Üretilen kod ---------------------------------------------------------------

def _prepared(language: str, tmp_path: Path) -> Path:
    script = render_script(KONU01_LAB, language)
    local = str(DATA).replace("\\", "/")
    replacements = {
        "Python": ("YEREL_DOSYA = None", f'YEREL_DOSYA = "{local}"'),
        "R": ("yerel_dosya <- NULL", f'yerel_dosya <- "{local}"'),
    }
    before, after = replacements[language]
    assert before in script
    extension = {"Python": "py", "R": "R"}[language]
    path = tmp_path / f"konu01.{extension}"
    path.write_text(script.replace(before, after, 1), encoding="utf-8")
    return path


def test_generated_python_is_valid_syntax() -> None:
    compile(render_script(KONU01_LAB, "Python"), "konu01.py", "exec")


@needs_data
def test_generated_python_reproduces_notes(tmp_path: Path) -> None:
    path = _prepared("Python", tmp_path)
    environment = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="cp1254")  # Türkçe Windows kod sayfası
    result = subprocess.run(
        [sys.executable, str(path)], cwd=tmp_path, capture_output=True, encoding="utf-8", errors="replace",
        timeout=600, env=environment,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert result.stdout.count("  OK   ") == CHECK_COUNT


@needs_data
@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript bulunamadı.")
def test_generated_r_reproduces_notes(tmp_path: Path) -> None:
    path = _prepared("R", tmp_path)
    result = subprocess.run(
        ["Rscript", str(path)], cwd=tmp_path, capture_output=True, encoding="utf-8", errors="replace", timeout=600,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert result.stdout.count("  OK   ") == CHECK_COUNT


def test_generated_stata_follows_safety_conventions() -> None:
    script = render_script(KONU01_LAB, "Stata")
    assert "version 14" in script
    assert "set type double" in script
    assert "import delimited" not in script
    assert "infile " + " ".join(CPS09MAR_COLUMNS) + " ///" in script
    generates = re.findall(r"^generate\b.*$", script, flags=re.MULTILINE)
    assert generates and all(line.startswith("generate double ") for line in generates)
    assert script.count("kontrol_et `=") == CHECK_COUNT
    assert script.count("{") == script.count("}")
    lines = [line.strip() for line in script.splitlines()]
    assert lines.count("preserve") > 0 and lines.count("preserve") == lines.count("restore")
    for foreign in ("smf.", "<-", "np.", "print("):
        assert foreign not in script
    for line in script.splitlines():
        if line.strip().startswith("*"):
            continue
        assert line.count('"') % 2 == 0, line
