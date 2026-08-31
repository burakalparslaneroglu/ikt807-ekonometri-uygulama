from __future__ import annotations

import pytest

from core.formatters import format_estimate, format_p_value
from core.ui_preferences import (
    TEXT_SCALE_OPTIONS,
    normalize_text_scale,
    plotly_font_size,
    text_scale_css,
)


def test_text_scale_options_cover_required_range_once() -> None:
    assert tuple(TEXT_SCALE_OPTIONS) == ("%100", "%110", "%120", "%130")
    css = text_scale_css(TEXT_SCALE_OPTIONS["%130"])
    assert css.count("--ikt-text-scale") == 1
    assert "1.30" in css
    assert plotly_font_size(10, 1.3) == 13


def test_unsupported_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="Desteklenmeyen"):
        normalize_text_scale(1.25)


def test_student_facing_number_rules() -> None:
    assert format_p_value(0.0002) == "p < 0.001"
    assert format_p_value(0.0472) == "p = 0.047"
    assert format_estimate(1.3834, digits=3) == "1,383"
