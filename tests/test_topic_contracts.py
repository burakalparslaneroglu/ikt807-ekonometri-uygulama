from __future__ import annotations

import importlib
import inspect

from app import TOPIC_RENDERERS
from core.topic_registry import get_topic, list_topics


TOPIC_MODULES = {
    "konu01": "topics.konu01_ampirik_modelleme",
    "konu02": "topics.konu02_dogrusal_regresyon",
    "konu03": "topics.konu03_tanimlama_icsellik",
    "konu04": "topics.konu04_iv_2sls",
    "konu05": "topics.konu05_ikili_ayrik",
    "konu06": "topics.konu06_sansurleme_secim",
    "konu07": "topics.konu07_kantil_regresyon",
    "konu08": "topics.konu08_parametrik_olmayan",
    "konu09": "topics.konu09_rdd",
    "konu10": "topics.konu10_bootstrap",
    "konu11": "topics.konu11_model_secimi",
    "konu12": "topics.konu12_dml",
}


def test_registry_has_exact_course_order() -> None:
    topics = list_topics()
    assert len(topics) == 12
    assert [topic.number for topic in topics] == list(range(1, 13))
    assert [topic.key for topic in topics] == list(TOPIC_MODULES)
    assert len({topic.title for topic in topics}) == 12


def test_each_topic_has_complete_pedagogical_metadata() -> None:
    for topic in list_topics():
        assert topic.guiding_question.endswith("?")
        assert topic.estimand
        assert topic.identification_focus
        assert topic.application_focus
        assert topic.dataset_ids
        assert len(topic.methods) >= 3
        assert len(topic.questions) >= 2
        assert all(prompt.endswith("?") for prompt, _ in topic.questions)


def test_each_topic_module_exports_render() -> None:
    assert set(TOPIC_RENDERERS) == set(TOPIC_MODULES)
    for key, module_name in TOPIC_MODULES.items():
        module = importlib.import_module(module_name)
        assert callable(module.render), key
        assert TOPIC_RENDERERS[key] is module.render


def test_each_topic_exposes_code_downloads_in_all_four_sections() -> None:
    for key, module_name in TOPIC_MODULES.items():
        module = importlib.import_module(module_name)
        source = inspect.getsource(module)
        assert source.count("render_reproduction_code(TOPIC_KEY") == 4, key


def test_unknown_topic_is_rejected() -> None:
    try:
        get_topic("konu99")
    except ValueError as error:
        assert "Desteklenmeyen konu" in str(error)
    else:
        raise AssertionError("Bilinmeyen konu kabul edildi.")
