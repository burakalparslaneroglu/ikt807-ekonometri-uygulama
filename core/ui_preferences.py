"""UI'dan bağımsız metin ölçeği tercihleri."""

from __future__ import annotations


TEXT_SCALE_OPTIONS: dict[str, float] = {
    "%100": 1.0,
    "%110": 1.1,
    "%120": 1.2,
    "%130": 1.3,
}
DEFAULT_TEXT_SCALE_LABEL = "%100"


def normalize_text_scale(value: float) -> float:
    """Yalnız desteklenen metin ölçeklerini kabul eder."""

    if value not in TEXT_SCALE_OPTIONS.values():
        raise ValueError(f"Desteklenmeyen metin ölçeği: {value}")
    return float(value)


def text_scale_css(scale: float) -> str:
    """Tek bir CSS değişkeni üzerinden ölçek bildirimi üretir."""

    normalized = normalize_text_scale(scale)
    return f"<style>:root{{--ikt-text-scale:{normalized:.2f};}}</style>"


def plotly_font_size(base_size: int, scale: float) -> int:
    """Plotly fontunu CSS'ten bağımsız fakat aynı tercihle ölçekler."""

    if base_size <= 0:
        raise ValueError("base_size pozitif olmalıdır.")
    return round(base_size * normalize_text_scale(scale))
