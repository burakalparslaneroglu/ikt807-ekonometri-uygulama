from __future__ import annotations

import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import matplotlib
import matplotlib.pyplot as plt

from core.code_recipes import (
    TOPIC_SECTIONS,
    build_colab_notebook,
    build_python_recipe,
)
from core.topic_registry import list_topics


def test_every_topic_has_four_reproduction_sections() -> None:
    assert set(TOPIC_SECTIONS) == {topic.key for topic in list_topics()}
    assert all(len(sections) == 4 for sections in TOPIC_SECTIONS.values())
    assert sum(map(len, TOPIC_SECTIONS.values())) == 48


def test_every_python_recipe_compiles_and_selects_its_section() -> None:
    for topic_key, sections in TOPIC_SECTIONS.items():
        for section_key, section_title in sections:
            recipe = build_python_recipe(topic_key, section_key)
            compile(recipe.python_code, recipe.filename_stem, "exec")
            assert recipe.section_title == section_title
            assert f'ACTIVE_SECTION = "{section_key}"' in recipe.python_code
            assert "SECTIONS[ACTIVE_SECTION]()" in recipe.python_code


def test_every_colab_notebook_is_valid_and_contains_install_cell() -> None:
    for topic_key, sections in TOPIC_SECTIONS.items():
        for section_key, _ in sections:
            recipe = build_python_recipe(topic_key, section_key)
            notebook = json.loads(build_colab_notebook(recipe))
            assert notebook["nbformat"] == 4
            assert [cell["cell_type"] for cell in notebook["cells"]] == [
                "markdown",
                "code",
                "code",
            ]
            assert "%pip install" in "".join(notebook["cells"][1]["source"])
            assert recipe.python_code == "".join(notebook["cells"][2]["source"])


def test_unknown_recipe_section_is_rejected() -> None:
    try:
        build_python_recipe("konu01", "olmayan")
    except ValueError as error:
        assert "Kod tarifi olmayan bölüm" in str(error)
    else:
        raise AssertionError("Bilinmeyen kod bölümü kabul edildi.")


def test_every_python_recipe_executes_without_external_data(monkeypatch) -> None:
    matplotlib.use("Agg")
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: plt.close("all"))
    for topic_key, sections in TOPIC_SECTIONS.items():
        for section_key, _ in sections:
            recipe = build_python_recipe(topic_key, section_key)
            namespace = {"__name__": "__main__"}
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                exec(
                    compile(recipe.python_code, recipe.filename_stem, "exec"),
                    namespace,
                )
