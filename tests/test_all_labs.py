"""Kayıtlı bütün uygulama laboratuvarları için ortak sözleşme ve üç dil testleri.

Yeni bir konu ``core/labs/registry.py``'ye eklendiğinde bu testler onu kendiliğinden kapsar.
Gerçek veri gerektiren testler IKT807_HANSEN_CPS09MAR_PATH verilmezse atlanır.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from core.codegen.base import LANGUAGES, render_script, render_step
from core.hansen_data import load_from_upload
from core.labs.registry import LABS
from core.labs.runner import run_lab

SPECS = list(LABS.values())


def _data_path() -> Path | None:
    for name in ("IKT807_HANSEN_CPS09MAR_PATH", "IKT807_CPS_PATH"):
        value = os.environ.get(name)
        if value and Path(value).is_file():
            return Path(value)
    return None


DATA = _data_path()
needs_data = pytest.mark.skipif(DATA is None, reason="Gerçek Hansen cps09mar dosyası verilmedi.")


def _checks(spec) -> int:
    return sum(len(step.checks) for step in spec.steps)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_steps_are_numbered_and_tied_to_the_notes(spec) -> None:
    assert [step.number for step in spec.steps] == list(range(1, len(spec.steps) + 1))
    for step in spec.steps:
        assert step.note.section == spec.note_section and step.note.step == step.number
        assert step.title and step.explanation
    assert _checks(spec) >= 10


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


@needs_data
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_app_reproduces_every_number_in_the_notes(spec) -> None:
    raw = load_from_upload("cps09mar", DATA.read_bytes(), DATA.name).frame
    run = run_lab(spec, {"cps09mar": raw})
    failures = [f"{c.check.label}: {c.value}" for items in run.checks.values() for c in items if not c.passed]
    assert not failures, failures


def _prepared(spec, language: str, folder: Path) -> Path:
    script = render_script(spec, language)
    local = str(DATA).replace("\\", "/")
    before, after = {"Python": ("YEREL_DOSYA = None", f'YEREL_DOSYA = "{local}"'),
                     "R": ("yerel_dosya <- NULL", f'yerel_dosya <- "{local}"')}[language]
    path = folder / f"{spec.topic_key}.{ {'Python': 'py', 'R': 'R'}[language] }"
    path.write_text(script.replace(before, after, 1), encoding="utf-8")
    return path


@needs_data
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_generated_python_reproduces_the_notes(spec, tmp_path: Path) -> None:
    path = _prepared(spec, "Python", tmp_path)
    result = subprocess.run([sys.executable, str(path)], cwd=tmp_path, capture_output=True, text=True,
                            timeout=900, env=dict(os.environ, MPLBACKEND="Agg"))
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert result.stdout.count("  OK   ") == _checks(spec)


@needs_data
@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript bulunamadı.")
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_generated_r_reproduces_the_notes(spec, tmp_path: Path) -> None:
    path = _prepared(spec, "R", tmp_path)
    result = subprocess.run(["Rscript", str(path)], cwd=tmp_path, capture_output=True, text=True, timeout=900,
                            env=dict(os.environ, LANG="C.UTF-8", LC_ALL="C.UTF-8"))
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert result.stdout.count("  OK   ") == _checks(spec)
