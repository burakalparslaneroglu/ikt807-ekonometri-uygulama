"""Sunum bağlantıları için doğrulanmış, Streamlit'ten bağımsız gezinme."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from importlib import import_module
import math
from typing import Any
from urllib.parse import urlencode

from core.labs.registry import get_lab
from core.topic_registry import list_topics

TABS = {"uygulama": "Uygulama", "sezgi": "Sezgi", "sinav": "Kendini sına"}
BASE_URL = "https://iktt807-ekonometri-uygulama.streamlit.app/"


def get_experiments(topic: str):
    if topic not in {item.key for item in list_topics()}:
        raise ValueError(f"Bilinmeyen konu: {topic}")
    module = import_module(f"core.labs.sezgi_{topic}")
    return getattr(module, f"{topic.upper()}_EXPERIMENTS")


def tab_key(topic: str) -> str:
    return f"{topic}_sekme"


@dataclass(frozen=True)
class LectureTarget:
    topic: str
    tab: str
    number: int | None = None
    parameters: tuple[tuple[str, float], ...] = ()


def parse_target(query: Mapping[str, str]) -> tuple[LectureTarget | None, tuple[str, ...]]:
    """Eksik/geçersiz hedefi sessizce başka bir deneye dönüştürmez."""
    if "konu" not in query:
        return None, ()
    topic = "konu" + query["konu"].removeprefix("konu").zfill(2)
    if topic not in {item.key for item in list_topics()}:
        return None, ("Bağlantıdaki konu bulunamadı; menüden bir konu seçebilirsiniz.",)
    tab = query.get("sekme", "uygulama")
    if tab not in TABS:
        return None, ("Bağlantıdaki sekme bulunamadı.",)
    number = None
    parameters = ()
    if tab != "sinav":
        items = get_experiments(topic) if tab == "sezgi" else get_lab(topic).steps
        field = "deney" if tab == "sezgi" else "adim"
        try:
            number = int(query.get(field, str(items[0].number)))
            item = next(item for item in items if item.number == number)
        except (ValueError, StopIteration):
            return None, (f"Bağlantıdaki {field} bulunamadı.",)
        if tab == "sezgi":
            values = item.defaults()
            for param in item.parameters:
                raw = query.get(f"p_{param.key}")
                if raw is None:
                    continue
                try:
                    value = float(raw)
                    position = (value - param.minimum) / param.step
                    valid = (math.isfinite(value) and param.minimum <= value <= param.maximum
                             and math.isclose(position, round(position), abs_tol=1e-7))
                except (ValueError, OverflowError):
                    valid = False
                if not valid:
                    return None, (f"Bağlantıdaki {param.label} değeri geçersiz.",)
                values[param.key] = int(value) if param.integer else value
            parameters = tuple(values.items())
    return LectureTarget(topic, tab, number, parameters), ()


def apply_query(state: MutableMapping[str, Any], query: Mapping[str, str]) -> tuple[str, ...]:
    """URL yalnız değiştiğinde uygulanır; kaydırıcı/seçici etkileşimi geri alınmaz."""
    signature = tuple(sorted(query.items()))
    if state.get("_lecture_query") == signature:
        return ()
    state["_lecture_query"] = signature
    target, errors = parse_target(query)
    if target is None:
        return errors
    state["topic_selector"] = target.topic
    state[tab_key(target.topic)] = TABS[target.tab]
    if target.tab == "uygulama":
        values = {f"{target.topic}_lab_kaynak": "notlar", f"{target.topic}_lab_step": target.number}
    elif target.tab == "sezgi":
        values = {f"{target.topic}_sezgi_deney": target.number}
        values.update({f"{target.topic}_sezgi{target.number}_{key}": value for key, value in target.parameters})
    else:
        values = {}
    for key, value in values.items():
        state[key] = value
        state[f"_kalici_{key}"] = value
    return ()


def lecture_url(topic: str, tab: str, number: int | None = None, **parameters: float) -> str:
    query = {"konu": topic.removeprefix("konu"), "sekme": tab}
    if number is not None:
        query["deney" if tab == "sezgi" else "adim"] = str(number)
    query.update({f"p_{key}": str(value) for key, value in parameters.items()})
    target, errors = parse_target(query)
    if target is None or errors:
        raise ValueError("; ".join(errors))
    return BASE_URL + "?" + urlencode(query)
