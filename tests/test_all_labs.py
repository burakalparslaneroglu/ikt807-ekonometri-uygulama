"""Kayıtlı bütün uygulama laboratuvarları için ortak sözleşme ve üç dil testleri.

Yeni bir konu ``core/labs/registry.py``'ye eklendiğinde bu testler onu kendiliğinden kapsar.
Gerçek veri gerektiren testler, veri setinin yolu verilmezse atlanır:
IKT807_HANSEN_CPS09MAR_PATH, IKT807_HANSEN_DDK2011_PATH, IKT807_HANSEN_CARD1995_PATH,
IKT807_HANSEN_CHJ2004_PATH.
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


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_app_reproduces_every_number_in_the_notes(spec) -> None:
    path = _data(spec)
    loaded = load_from_upload(spec.dataset, path.read_bytes(), path.name)
    assert loaded.matches_hansen
    run = run_lab(spec, {spec.dataset: loaded.frame})
    failures = [f"{c.check.label}: {c.value}" for items in run.checks.values() for c in items if not c.passed]
    assert not failures, failures


def _prepared(spec, language: str, folder: Path) -> Path:
    script = render_script(spec, language)
    local = str(_data(spec)).replace("\\", "/")
    before, after = {"Python": ("YEREL_DOSYA = None", f'YEREL_DOSYA = "{local}"'),
                     "R": ("yerel_dosya <- NULL", f'yerel_dosya <- "{local}"')}[language]
    path = folder / f"{spec.topic_key}.{ {'Python': 'py', 'R': 'R'}[language] }"
    path.write_text(script.replace(before, after, 1), encoding="utf-8")
    return path


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_generated_python_reproduces_the_notes(spec, tmp_path: Path) -> None:
    path = _prepared(spec, "Python", tmp_path)
    result = subprocess.run([sys.executable, str(path)], cwd=tmp_path, capture_output=True, text=True,
                            timeout=900, env=dict(os.environ, MPLBACKEND="Agg"))
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert result.stdout.count("  OK   ") == _checks(spec)


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript bulunamadı.")
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.topic_key)
def test_generated_r_reproduces_the_notes(spec, tmp_path: Path) -> None:
    path = _prepared(spec, "R", tmp_path)
    result = subprocess.run(["Rscript", str(path)], cwd=tmp_path, capture_output=True, text=True, timeout=900,
                            env=dict(os.environ, LANG="C.UTF-8", LC_ALL="C.UTF-8"))
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert result.stdout.count("  OK   ") == _checks(spec)
